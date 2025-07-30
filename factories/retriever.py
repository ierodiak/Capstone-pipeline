"""
Retriever factory implementing strategy pattern for different retrieval methods.
"""
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

from core.types import BaseRetriever, Document, EnsembleRetriever
from core.config import RetrievalConfig
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from utils.logger import setup_logger

logger = setup_logger(__name__)


class RetrieverStrategy(ABC):
    """Abstract base class for retriever strategies"""
    
    @abstractmethod
    def create(self, config: RetrievalConfig, **kwargs) -> BaseRetriever:
        """Create a retriever instance"""
        pass


class VectorRetrieverStrategy(RetrieverStrategy):
    """Strategy for vector-based retrieval"""
    
    def create(self, config: RetrievalConfig, **kwargs) -> BaseRetriever:
        vector_store = kwargs.get('vector_store')
        if not vector_store:
            raise ValueError("Vector store is required for vector retrieval")
        
        search_kwargs = {"k": config.top_k}
        
        if config.search_type == "similarity":
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            )
        elif config.search_type == "mmr":
            search_kwargs["fetch_k"] = config.top_k * 2
            search_kwargs["lambda_mult"] = config.mmr_lambda
            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs=search_kwargs
            )
        else:
            raise ValueError(f"Unknown search type: {config.search_type}")
        
        logger.info(f"Created vector retriever with {config.search_type} search")
        return retriever


class BM25RetrieverStrategy(RetrieverStrategy):
    """Strategy for BM25 keyword-based retrieval"""
    
    def create(self, config: RetrievalConfig, **kwargs) -> BaseRetriever:
        documents = kwargs.get('documents')
        if not documents:
            raise ValueError("Documents are required for BM25 retrieval")
        
        retriever = BM25Retriever.from_documents(
            documents,
            k=config.top_k
        )
        
        logger.info(f"Created BM25 retriever with k={config.top_k}")
        return retriever


class HybridRetrieverStrategy(RetrieverStrategy):
    """Strategy for hybrid retrieval combining vector and BM25"""
    
    def create(self, config: RetrievalConfig, **kwargs) -> BaseRetriever:
        vector_store = kwargs.get('vector_store')
        documents = kwargs.get('documents')
        
        if not vector_store or not documents:
            raise ValueError("Both vector store and documents are required for hybrid retrieval")
        
        # Create base retrievers
        vector_retriever = VectorRetrieverStrategy().create(config, **kwargs)
        bm25_retriever = BM25RetrieverStrategy().create(config, **kwargs)
        
        # Create ensemble with weights
        weights = config.hybrid_weights if config.hybrid_weights else [0.7, 0.3]
        retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=weights
        )
        
        logger.info(f"Created hybrid retriever with weights {weights}")
        return retriever


class MultiQueryRetrieverStrategy(RetrieverStrategy):
    """Strategy for multi-query retrieval"""
    
    def create(self, config: RetrievalConfig, **kwargs) -> BaseRetriever:
        vector_store = kwargs.get('vector_store')
        if not vector_store:
            raise ValueError("Vector store is required for multi-query retrieval")
        
        base_retriever = VectorRetrieverStrategy().create(config, **kwargs)
        llm = ChatOpenAI(temperature=0)
        
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm
        )
        
        logger.info("Created multi-query retriever")
        return retriever


class RetrieverFactory:
    """Factory for creating retrievers using strategy pattern"""
    
    def __init__(self):
        self.vector_store: Optional[FAISS] = None
        self.documents: Optional[List[Document]] = None
        self._strategies = {
            "vector": VectorRetrieverStrategy(),
            "bm25": BM25RetrieverStrategy(),
            "hybrid": HybridRetrieverStrategy(),
            "multi_query": MultiQueryRetrieverStrategy(),
        }
    
    def register_strategy(self, name: str, strategy: RetrieverStrategy):
        """Register a new retriever strategy"""
        self._strategies[name] = strategy
        logger.info(f"Registered retriever strategy: {name}")
    
    def create_retriever(self, config: RetrievalConfig) -> BaseRetriever:
        """Create a retriever based on configuration"""
        strategy = self._strategies.get(config.strategy)
        if not strategy:
            raise ValueError(f"Unknown retriever strategy: {config.strategy}")
        
        # Prepare kwargs
        kwargs = {
            'vector_store': self.vector_store,
            'documents': self.documents
        }
        
        try:
            retriever = strategy.create(config, **kwargs)
            logger.info(f"Successfully created {config.strategy} retriever")
            return retriever
        except Exception as e:
            logger.error(f"Error creating {config.strategy} retriever: {e}")
            raise
    
    def set_vector_store(self, vector_store: FAISS):
        """Set the vector store for retrievers"""
        self.vector_store = vector_store
        logger.info("Vector store set in retriever factory")
    
    def set_documents(self, documents: List[Document]):
        """Set the documents for BM25 retrieval"""
        self.documents = documents
        logger.info(f"Set {len(documents)} documents in retriever factory")