"""
Retriever factory module for creating different retriever strategies.
"""
from typing import List, Dict, Any
from core.types import (
    Document, BaseRetriever, BM25Retriever, EnsembleRetriever,
    ConfigurableField
)
from core.config import RetrieverConfig
from core.protocols import RetrieverStrategy


class RetrieverFactory:
    """Factory for creating different retriever strategies"""
    
    def __init__(self, vector_store=None, documents=None):
        self.vector_store = vector_store
        self.documents = documents
        self._retrievers = {}
        
    def create_retriever(self, config: RetrieverConfig) -> BaseRetriever:
        """Create retriever based on configuration"""
        
        if config.type == "vector":
            if not self.vector_store:
                raise ValueError("Vector store required for vector retriever")
            
            retriever = self.vector_store.as_retriever(
                search_type=config.search_type,
                search_kwargs={"k": config.top_k}
            )
            
        elif config.type == "bm25":
            if not self.documents:
                raise ValueError("Documents required for BM25 retriever")
            
            retriever = BM25Retriever.from_documents(
                self.documents,
                k=config.top_k
            )
            
        elif config.type == "hybrid":
            if not self.vector_store or not self.documents:
                raise ValueError("Both vector store and documents required for hybrid retriever")
            
            vector_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": config.top_k}
            )
            bm25_retriever = BM25Retriever.from_documents(
                self.documents,
                k=config.top_k
            )
            
            retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.7, 0.3]  # Can be configured
            )
            
        else:
            raise ValueError(f"Unknown retriever type: {config.type}")
        
        self._retrievers[config.type] = retriever
        return retriever
    
    def get_configurable_retriever(self) -> BaseRetriever:
        """Create a configurable retriever that can switch between strategies"""
        if not self._retrievers:
            raise ValueError("No retrievers created yet")
        
        # Get the first retriever as default
        default_retriever = list(self._retrievers.values())[0]
        
        # Create configurable alternatives
        return default_retriever.configurable_alternatives(
            ConfigurableField(id="retriever_type"),
            default_key=list(self._retrievers.keys())[0],
            **self._retrievers
        )