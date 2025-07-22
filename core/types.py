"""
Core type definitions and imports for the RAG pipeline.
"""
from typing import List, Dict, Optional, Tuple, Any, Protocol, TypedDict, Literal
from pathlib import Path
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Core LangChain imports (v0.3+ pattern)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough, 
    RunnableParallel,
    RunnableLambda,
    ConfigurableField,
    ConfigurableFieldSpec
)
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseChatModel

# Document processing
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Models and Embeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama

# Vector stores
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Configuration management
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

# LangGraph for advanced pipelines
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Type definitions for LangGraph
class RAGState(TypedDict):
    """State for iterative RAG pipeline"""
    question: str
    context: List[Document]
    answer: str
    iteration: int
    quality_score: float