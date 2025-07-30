# factories/loader_factory.py
"""
Factory for creating different document loaders.
"""
from typing import List, Protocol
from abc import abstractmethod
from pathlib import Path
from core.types import Document
from core.config import LoaderConfig  # FIXED: Changed from experiment_config


class DocumentLoader(Protocol):
    """Protocol for document loaders"""
    @abstractmethod
    def load(self, path: str) -> List[Document]:
        pass


class TextLoader:
    """Standard text document loader"""
    def __init__(self, config: LoaderConfig):
        self.config = config
        
    def load(self, path: str) -> List[Document]:
        from langchain_community.document_loaders import PyPDFLoader, TextLoader as LangchainTextLoader, UnstructuredWordDocumentLoader
        
        documents = []
        path_obj = Path(path)
        
        if path_obj.is_file():
            if path_obj.suffix == '.pdf':
                loader = PyPDFLoader(str(path))
            elif path_obj.suffix == '.txt':
                loader = LangchainTextLoader(str(path))
            elif path_obj.suffix == '.docx':
                loader = UnstructuredWordDocumentLoader(str(path))
            else:
                raise ValueError(f"Unsupported file type: {path_obj.suffix}")
            documents = loader.load()
            
        elif path_obj.is_dir():
            for file_path in path_obj.rglob("*"):
                if file_path.suffix in ['.pdf', '.txt', '.docx']:
                    docs = self.load(str(file_path))
                    documents.extend(docs)
                    
        return documents


class TextImageLoader:
    """Loader that extracts both text and images from documents"""
    def __init__(self, config: LoaderConfig):
        self.config = config
        
    def load(self, path: str) -> List[Document]:
        # Import necessary libraries
        try:
            import fitz  # PyMuPDF for image extraction
            from PIL import Image
            import io
        except ImportError:
            print("Please install PyMuPDF and Pillow for image extraction")
            return []
        
        documents = []
        path_obj = Path(path)
        
        if path_obj.suffix == '.pdf':
            # Extract text
            text_loader = TextLoader(self.config)
            text_docs = text_loader.load(str(path))
            
            # Extract images
            pdf_document = fitz.open(str(path))
            
            for page_num, page in enumerate(pdf_document):
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_document, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes()
                        img = Image.open(io.BytesIO(img_data))
                        
                        # Create document with image metadata
                        img_doc = Document(
                            page_content=f"[Image {img_index} from page {page_num}]",
                            metadata={
                                "source": str(path),
                                "page": page_num,
                                "type": "image",
                                "image_index": img_index,
                                "image_data": img_data  # Store for later processing
                            }
                        )
                        documents.append(img_doc)
                        
            documents.extend(text_docs)
            
        return documents


class LoaderFactory:
    """Factory for creating document loaders"""
    
    @staticmethod
    def create_loader(config: LoaderConfig) -> DocumentLoader:
        if config.type == "text":
            return TextLoader(config)
        elif config.type == "text_image":
            return TextImageLoader(config)
        elif config.type == "none":
            return None
        else:
            raise ValueError(f"Unknown loader type: {config.type}")