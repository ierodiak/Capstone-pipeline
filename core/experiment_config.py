# core/experiment_config.py

"""
Enhanced configuration system with experiment tracking and variant management.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime
import hashlib
import json

@dataclass
class LoaderConfig:
    """Configuration for document loaders"""
    type: Literal["text", "text_image", "none"] = "text"
    pdf_extract_images: bool = False
    image_description_model: str = "gpt-4-vision-preview"
    supported_formats: List[str] = field(default_factory=lambda: ["pdf", "txt", "docx"])

@dataclass
class ChunkerConfig:
    """Configuration for chunking strategies"""
    method: Literal["recursive", "semantic", "sentence", "fixed", "sliding_window"] = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 50
    sentence_per_chunk: int = 5  # for sentence chunker
    semantic_threshold: float = 0.8  # for semantic chunker
    
@dataclass
class EmbeddingConfig:
    """Configuration for embedding models"""
    provider: Literal["openai", "cohere", "huggingface", "sentence_transformers"] = "openai"
    model: str = "text-embedding-3-small"
    dimension: Optional[int] = None
    batch_size: int = 100

@dataclass
class StorageConfig:
    """Configuration for vector stores"""
    type: Literal["faiss", "chroma", "pinecone", "weaviate", "qdrant"] = "faiss"
    persist: bool = True
    index_type: str = "IVF"  # for FAISS
    metric: str = "cosine"

@dataclass
class RetrievalConfig:
    """Configuration for retrieval strategies"""
    strategy: Literal["vector", "bm25", "hybrid", "mmr", "rerank"] = "vector"
    top_k: int = 5
    rerank_model: Optional[str] = None
    hybrid_weights: List[float] = field(default_factory=lambda: [0.7, 0.3])
    mmr_lambda: float = 0.5

@dataclass
class GenerationConfig:
    """Configuration for generation models"""
    provider: Literal["openai", "anthropic", "cohere", "huggingface", "ollama"] = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1000
    prompt_template: str = "default"

@dataclass
class ExperimentConfig:
    """Master configuration for experiments with tracking"""
    experiment_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    experiment_name: str = "default_experiment"
    tags: List[str] = field(default_factory=list)
    
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    # Evaluation settings
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        "faithfulness", "answer_relevancy", "context_precision", 
        "response_time", "token_usage", "retrieval_precision"
    ])
    
    def get_variant_id(self) -> str:
        """Generate unique ID for this configuration variant"""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def get_variant_description(self) -> Dict[str, str]:
        """Get human-readable variant description"""
        return {
            "loader": f"{self.loader.type}",
            "chunker": f"{self.chunker.method}_{self.chunker.chunk_size}",
            "embedding": f"{self.embedding.provider}_{self.embedding.model}",
            "storage": f"{self.storage.type}",
            "retrieval": f"{self.retrieval.strategy}_k{self.retrieval.top_k}",
            "generation": f"{self.generation.provider}_{self.generation.model}"
        }