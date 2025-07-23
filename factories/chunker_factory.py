# factories/chunker_factory.py

"""
Factory for creating different chunking strategies.
"""
from typing import List, Protocol
from abc import abstractmethod
from core.types import Document
from core.experiment_config import ChunkerConfig

class ChunkingStrategy(Protocol):
    """Protocol for chunking strategies"""
    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        pass

class RecursiveChunker:
    """Recursive character text splitter"""
    def __init__(self, config: ChunkerConfig):
        self.config = config
        
    def chunk(self, documents: List[Document]) -> List[Document]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_documents(documents)

class SemanticChunker:
    """Semantic-based chunking"""
    def __init__(self, config: ChunkerConfig):
        self.config = config
        
    def chunk(self, documents: List[Document]) -> List[Document]:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_openai import OpenAIEmbeddings
        
        splitter = SemanticChunker(
            OpenAIEmbeddings(),
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=self.config.semantic_threshold
        )
        return splitter.split_documents(documents)

class SentenceChunker:
    """Sentence-based chunking"""
    def __init__(self, config: ChunkerConfig):
        self.config = config
        
    def chunk(self, documents: List[Document]) -> List[Document]:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        chunks = []
        for doc in documents:
            sentences = nltk.sent_tokenize(doc.page_content)
            
            for i in range(0, len(sentences), self.config.sentence_per_chunk):
                chunk_sentences = sentences[i:i + self.config.sentence_per_chunk]
                chunk_text = ' '.join(chunk_sentences)
                
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={**doc.metadata, "chunk_method": "sentence"}
                )
                chunks.append(chunk_doc)
                
        return chunks

class FixedSizeChunker:
    """Fixed size chunking without overlap"""
    def __init__(self, config: ChunkerConfig):
        self.config = config
        
    def chunk(self, documents: List[Document]) -> List[Document]:
        from langchain_text_splitters import CharacterTextSplitter
        
        splitter = CharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=0,
            separator=""
        )
        return splitter.split_documents(documents)

class SlidingWindowChunker:
    """Sliding window chunking with custom overlap"""
    def __init__(self, config: ChunkerConfig):
        self.config = config
        
    def chunk(self, documents: List[Document]) -> List[Document]:
        chunks = []
        
        for doc in documents:
            text = doc.page_content
            stride = self.config.chunk_size - self.config.chunk_overlap
            
            for i in range(0, len(text), stride):
                chunk_text = text[i:i + self.config.chunk_size]
                if len(chunk_text) < self.config.chunk_size * 0.5:  # Skip very small chunks
                    continue
                    
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_method": "sliding_window",
                        "chunk_start": i,
                        "chunk_end": i + len(chunk_text)
                    }
                )
                chunks.append(chunk_doc)
                
        return chunks

class ChunkerFactory:
    """Factory for creating chunking strategies"""
    
    @staticmethod
    def create_chunker(config: ChunkerConfig) -> ChunkingStrategy:
        chunkers = {
            "recursive": RecursiveChunker,
            "semantic": SemanticChunker,
            "sentence": SentenceChunker,
            "fixed": FixedSizeChunker,
            "sliding_window": SlidingWindowChunker
        }
        
        chunker_class = chunkers.get(config.method)
        if not chunker_class:
            raise ValueError(f"Unknown chunking method: {config.method}")
            
        return chunker_class(config)