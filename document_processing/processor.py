"""
Document processing module for loading and chunking documents.
Implements factory pattern for different loader and chunker strategies.
"""
from pathlib import Path
from typing import List, Optional, Dict, Any, Protocol
from datetime import datetime
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from abc import ABC, abstractmethod

from core.types import Document, PyPDFLoader, PyPDFDirectoryLoader, RecursiveCharacterTextSplitter
from core.config import ChunkerConfig
from utils.logger import setup_logger

logger = setup_logger(__name__)


class DocumentLoaderStrategy(ABC):
    """Abstract base class for document loading strategies"""
    
    @abstractmethod
    def load(self, path: str) -> List[Document]:
        """Load documents from the given path"""
        pass


class PDFLoaderStrategy(DocumentLoaderStrategy):
    """Strategy for loading PDF documents"""
    
    def load(self, path: str) -> List[Document]:
        path_obj = Path(path)
        
        if path_obj.is_file() and path_obj.suffix == '.pdf':
            loader = PyPDFLoader(str(path))
            logger.info(f"Loading single PDF: {path}")
        elif path_obj.is_dir():
            loader = PyPDFDirectoryLoader(str(path), glob="**/*.pdf")
            logger.info(f"Loading PDFs from directory: {path}")
        else:
            raise ValueError(f"Path {path} is neither a PDF file nor a directory")
        
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} document pages")
        return documents


class TextLoaderStrategy(DocumentLoaderStrategy):
    """Strategy for loading text documents"""
    
    def load(self, path: str) -> List[Document]:
        path_obj = Path(path)
        documents = []
        
        if path_obj.is_file():
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                doc = Document(
                    page_content=content,
                    metadata={"source": str(path), "type": "text"}
                )
                documents.append(doc)
        elif path_obj.is_dir():
            for file_path in path_obj.glob("**/*.txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    doc = Document(
                        page_content=content,
                        metadata={"source": str(file_path), "type": "text"}
                    )
                    documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} text documents")
        return documents


class ChunkerStrategy(ABC):
    """Abstract base class for chunking strategies"""
    
    @abstractmethod
    def chunk(self, documents: List[Document], config: ChunkerConfig) -> List[Document]:
        """Chunk documents according to the strategy"""
        pass


class RecursiveChunkerStrategy(ChunkerStrategy):
    """Recursive character text splitting strategy"""
    
    def chunk(self, documents: List[Document], config: ChunkerConfig) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = []
        for doc in documents:
            doc_chunks = splitter.split_documents([doc])
            for i, chunk in enumerate(doc_chunks):
                chunk.metadata['chunk_index'] = i
                chunk.metadata['total_chunks'] = len(doc_chunks)
            chunks.extend(doc_chunks)
        
        logger.info(f"Created {len(chunks)} chunks with recursive strategy")
        return chunks


class SemanticChunkerStrategy(ChunkerStrategy):
    """Semantic chunking strategy (placeholder for future implementation)"""
    
    def chunk(self, documents: List[Document], config: ChunkerConfig) -> List[Document]:
        logger.warning("Semantic chunking not yet implemented, falling back to recursive")
        return RecursiveChunkerStrategy().chunk(documents, config)


class DocumentProcessorFactory:
    """Factory for creating document loaders and chunkers"""
    
    _loader_strategies = {
        "pdf": PDFLoaderStrategy,
        "text": TextLoaderStrategy,
    }
    
    _chunker_strategies = {
        "recursive": RecursiveChunkerStrategy,
        "semantic": SemanticChunkerStrategy,
    }
    
    @classmethod
    def get_loader(cls, loader_type: str) -> DocumentLoaderStrategy:
        """Get a document loader strategy"""
        strategy_class = cls._loader_strategies.get(loader_type)
        if not strategy_class:
            raise ValueError(f"Unknown loader type: {loader_type}")
        return strategy_class()
    
    @classmethod
    def get_chunker(cls, chunker_type: str) -> ChunkerStrategy:
        """Get a chunker strategy"""
        strategy_class = cls._chunker_strategies.get(chunker_type)
        if not strategy_class:
            raise ValueError(f"Unknown chunker type: {chunker_type}")
        return strategy_class()


class DocumentProcessor:
    """Main document processor using factory pattern"""
    
    def __init__(self, config: ChunkerConfig):
        self.config = config
        self.documents = []
        self.chunks = []
        self.factory = DocumentProcessorFactory()
        
    def load_documents(self, path: str, loader_type: str = "pdf") -> List[Document]:
        """Load documents from file or directory"""
        try:
            loader = self.factory.get_loader(loader_type)
            self.documents = loader.load(path)
            
            if self.documents:
                self._display_loading_summary()
            
            return self.documents
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise
            
    def chunk_documents(self, documents: Optional[List[Document]] = None) -> List[Document]:
        """Chunk documents into smaller pieces"""
        if documents is None:
            documents = self.documents
            
        if not documents:
            raise ValueError("No documents to chunk")
        
        try:
            chunker = self.factory.get_chunker(self.config.method)
            self.chunks = chunker.chunk(documents, self.config)
            
            self._display_chunking_summary()
            return self.chunks
        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            raise
    
    def _display_loading_summary(self):
        """Display loading summary in Jupyter"""
        display(Markdown("### Loading Summary"))
        print(f"Loaded {len(self.documents)} document pages")
        
        # Show document preview
        if self.documents:
            display(Markdown("#### First Document Preview:"))
            first_doc = self.documents[0]
            print(f"Source: {first_doc.metadata.get('source', 'Unknown')}")
            print(f"Page: {first_doc.metadata.get('page', 'N/A')}")
            print(f"Content preview: {first_doc.page_content[:200]}...")
    
    def _display_chunking_summary(self):
        """Display chunking summary with visualization"""
        display(Markdown("### Chunking Summary"))
        print(f"Created {len(self.chunks)} chunks from {len(self.documents)} documents")
        print(f"Average chunk size: {sum(len(c.page_content) for c in self.chunks) / len(self.chunks):.0f} characters")
        
        # Visualize chunk distribution
        self._visualize_chunks()
    
    def _visualize_chunks(self):
        """Create visualization of chunk sizes"""
        chunk_sizes = [len(chunk.page_content) for chunk in self.chunks]
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(chunk_sizes, bins=20, alpha=0.7, color='blue')
        plt.xlabel('Chunk Size (characters)')
        plt.ylabel('Count')
        plt.title('Distribution of Chunk Sizes')
        
        plt.subplot(1, 2, 2)
        plt.plot(chunk_sizes, alpha=0.7, color='green')
        plt.xlabel('Chunk Index')
        plt.ylabel('Size (characters)')
        plt.title('Chunk Sizes by Index')
        plt.axhline(y=self.config.chunk_size, color='r', linestyle='--', label='Target size')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Summary statistics
        print(f"\nChunk size statistics:")
        print(f"  Min: {min(chunk_sizes)} chars")
        print(f"  Max: {max(chunk_sizes)} chars")
        print(f"  Mean: {sum(chunk_sizes) / len(chunk_sizes):.0f} chars")
        print(f"  Target: {self.config.chunk_size} chars")