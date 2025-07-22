"""
Abstract base classes and protocols for the RAG pipeline components.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseChatModel


class RetrieverStrategy(ABC):
    """Abstract base class for retriever strategies"""
    
    @abstractmethod
    def retrieve(self, query: str) -> List[Document]:
        pass
    
    @abstractmethod
    def get_retriever(self) -> BaseRetriever:
        pass


class GeneratorStrategy(ABC):
    """Abstract base class for generator strategies"""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
    
    @abstractmethod
    def get_model(self) -> BaseChatModel:
        pass


class EvaluatorStrategy(ABC):
    """Abstract base class for evaluation strategies"""
    
    @abstractmethod
    async def evaluate(self, data: Dict[str, Any]) -> Dict[str, float]:
        pass