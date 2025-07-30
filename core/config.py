# core/config.py - MERGE with experiment_config.py
"""
Unified configuration management for the RAG pipeline.
Combines old config.py and experiment_config.py
"""
import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Any, Literal, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib
from IPython.display import display, Markdown

# Component configurations
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
    search_type: str = "similarity"  # Added from old RetrieverConfig
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
class MetricsConfig:
    """Configuration for evaluation metrics."""
    metric_names: List[str] = field(default_factory=lambda: [
        # RAGAS metrics
        "ragas_faithfulness",
        "ragas_answer_relevancy",
        # CoFE-RAG metrics
        "cofe_retrieval_recall",
        "cofe_retrieval_accuracy", 
        "cofe_generation_faithfulness",
        "cofe_generation_relevance",
        "cofe_pipeline_score",
        # OmniEval metrics
        "omni_accuracy",
        "omni_completeness",
        "omni_hallucination",
        "omni_utilization",
        "omni_weighted_score",
        # General metrics
        "response_time",
        "aggregate_score"
    ])
    
    # Framework-specific settings
    ragas_use_reference_metrics: bool = False
    cofe_evaluate_chunks: bool = True
    omni_task_aware: bool = True

@dataclass
class RAGConfig:
    """Master configuration for experiments with tracking"""
    experiment_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    experiment_name: str = "default_experiment"
    tags: List[str] = field(default_factory=list)
    
    # Component configurations
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    # Flexible metrics configuration
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    
    # Pipeline configuration
    pipeline_type: str = "linear"  # linear, parallel, iterative
    
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

# Backwards compatibility aliases
ExperimentConfig = RAGConfig
RetrieverConfig = RetrievalConfig  # Alias for old code
GeneratorConfig = GenerationConfig  # Already matches
EvaluationConfig = MetricsConfig  # Update old references

class ConfigManager:
    """Manages configuration with YAML/JSON support and runtime updates"""
    
    def __init__(self, config: Optional[Union[str, RAGConfig]] = None):
        """
        Initialize ConfigManager with either a config path or RAGConfig object.
        
        Args:
            config: Either a path to config file (str) or a RAGConfig object
        """
        if isinstance(config, str):
            self.config_path = config
            self.config = self._load_config()
        elif isinstance(config, RAGConfig):
            self.config_path = None
            self.config = config
        else:
            self.config_path = None
            self.config = RAGConfig()
        
    def _load_config(self) -> RAGConfig:
        """Load configuration from file or use defaults"""
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml'):
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)
            return self._dict_to_config(config_dict)
        return RAGConfig()
    
    def _dict_to_config(self, config_dict: dict) -> RAGConfig:
        """Convert dictionary to configuration objects"""
        # Parse each component configuration
        loader_config = LoaderConfig(**config_dict.get('loader', {}))
        chunker_config = ChunkerConfig(**config_dict.get('chunker', {}))
        embedding_config = EmbeddingConfig(**config_dict.get('embedding', {}))
        storage_config = StorageConfig(**config_dict.get('storage', {}))
        retrieval_config = RetrievalConfig(**config_dict.get('retrieval', {}))
        generation_config = GenerationConfig(**config_dict.get('generation', {}))
        metrics_config = MetricsConfig(**config_dict.get('metrics', {}))
        
        return RAGConfig(
            experiment_id=config_dict.get('experiment_id', datetime.now().strftime("%Y%m%d_%H%M%S")),
            experiment_name=config_dict.get('experiment_name', 'default_experiment'),
            tags=config_dict.get('tags', []),
            loader=loader_config,
            chunker=chunker_config,
            embedding=embedding_config,
            storage=storage_config,
            retrieval=retrieval_config,
            generation=generation_config,
            metrics=metrics_config,
            pipeline_type=config_dict.get('pipeline_type', 'linear')
        )
    
    def save_config(self, path: str):
        """Save current configuration to file"""
        config_dict = asdict(self.config)
        
        with open(path, 'w') as f:
            if path.endswith('.yaml'):
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                json.dump(config_dict, f, indent=2)
                
    def update_config(self, updates: dict):
        """Update configuration at runtime"""
        for key, value in updates.items():
            if hasattr(self.config, key):
                if isinstance(value, dict):
                    # Update nested configuration
                    config_obj = getattr(self.config, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(config_obj, sub_key):
                            setattr(config_obj, sub_key, sub_value)
                else:
                    setattr(self.config, key, value)
    
    def display_config(self):
        """Display current configuration in a formatted way"""
        config_dict = {
            'Experiment': {
                'id': self.config.experiment_id,
                'name': self.config.experiment_name,
                'tags': self.config.tags
            },
            'Loader': asdict(self.config.loader),
            'Chunker': asdict(self.config.chunker),
            'Embedding': asdict(self.config.embedding),
            'Storage': asdict(self.config.storage),
            'Retrieval': asdict(self.config.retrieval),
            'Generation': asdict(self.config.generation),
            'Metrics': asdict(self.config.metrics),
            'Pipeline Type': self.config.pipeline_type
        }
        
        display(Markdown("### Current Configuration"))
        for section, settings in config_dict.items():
            display(Markdown(f"**{section}:**"))
            if isinstance(settings, dict):
                for key, value in settings.items():
                    print(f"  - {key}: {value}")
            else:
                print(f"  - {settings}")
            print()

def create_default_config(save_path: str = "./configs/default_rag_config.yaml"):
    """Create and save a default configuration file"""
    default_config = RAGConfig()
    
    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_dict = asdict(default_config)
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    return default_config