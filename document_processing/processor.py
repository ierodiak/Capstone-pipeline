"""
Document processing module for loading and chunking documents.
"""
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from core.types import Document, PyPDFLoader, PyPDFDirectoryLoader, RecursiveCharacterTextSplitter
from core.config import RetrieverConfig


class DocumentProcessor:
    """Handles document loading and processing"""
    
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.documents = []
        self.chunks = []
        
    def load_documents(self, path: str) -> List[Document]:
        """Load documents from file or directory"""
        path_obj = Path(path)
        
        # Load single PDF file
        if path_obj.is_file() and path_obj.suffix == '.pdf':
            loader = PyPDFLoader(str(path))
            display(Markdown(f"Loading single PDF: `{path}`"))  # Markdown

        # Load multiple PDF files from a directory
        elif path_obj.is_dir():
            loader = PyPDFDirectoryLoader(str(path), glob="**/*.pdf")
            display(Markdown(f"Loading PDFs from directory: `{path}`"))
            
        else:
            raise ValueError(f"Path {path} is neither a PDF file nor a directory")
        
        self.documents = loader.load()
        
        # Display loading summary
        display(Markdown(f"### Loading Summary"))
        print(f"Loaded {len(self.documents)} document pages")
        
        # Show document preview
        if self.documents:
            display(Markdown("#### First Document Preview:"))
            first_doc = self.documents[0]
            print(f"Source: {first_doc.metadata.get('source', 'Unknown')}")
            print(f"Page: {first_doc.metadata.get('page', 'N/A')}")
            print(f"Content preview: {first_doc.page_content[:200]}...")
            
        return self.documents
    
    def chunk_documents(self, documents: Optional[List[Document]] = None) -> List[Document]:
        """Split documents into chunks"""
        if documents is None:
            documents = self.documents
            
        if not documents:
            raise ValueError("No documents to chunk")
        
        # Create text splitter (One technique)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False,
        )
        
        # Split documents
        self.chunks = text_splitter.split_documents(documents)
        
        # Add metadata
        for i, chunk in enumerate(self.chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
            chunk.metadata["timestamp"] = datetime.now().isoformat()
        
        # Display chunking summary
        display(Markdown("### Chunking Summary"))
        print(f"Created {len(self.chunks)} chunks from {len(documents)} documents")
        print(f"Average chunk size: {sum(len(c.page_content) for c in self.chunks) / len(self.chunks):.0f} characters")
        
        # Show chunk distribution
        self._display_chunk_distribution()
        
        return self.chunks
    
    def _display_chunk_distribution(self):
        """Display chunk size distribution"""
        chunk_sizes = [len(chunk.page_content) for chunk in self.chunks]
        
        plt.figure(figsize=(10, 4))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(chunk_sizes, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Chunk Size (characters)')
        plt.ylabel('Count')
        plt.title('Chunk Size Distribution')
        
        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(chunk_sizes)
        plt.ylabel('Chunk Size (characters)')
        plt.title('Chunk Size Statistics')
        
        plt.tight_layout()
        plt.show()
        
        # Statistics
        print(f"\nChunk Statistics:")
        print(f"  - Min size: {min(chunk_sizes)} chars")
        print(f"  - Max size: {max(chunk_sizes)} chars")
        print(f"  - Mean size: {sum(chunk_sizes) / len(chunk_sizes):.0f} chars")
        print(f"  - Total chunks: {len(chunk_sizes)}")