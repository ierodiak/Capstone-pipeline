"""
Configuration management for the RAG pipeline.
"""
import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from IPython.display import display, Markdown


@dataclass
class RetrieverConfig:
    """Configuration for retriever components"""
    type: str = "vector"  # vector, bm25, hybrid, ...
    model: str = "text-embedding-3-small"  # text-embedding-3-small, text-embedding-3-large, ...
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    search_type: str = "similarity"  # similarity, mmr, ..


@dataclass
class GeneratorConfig:
    """Configuration for generator components"""
    provider: str = "openai"  # openai, anthropic, ollama, ..
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1000


@dataclass
class EvaluationConfig:
    """Configuration for evaluation frameworks"""
    frameworks: List[str] = field(default_factory=lambda: ["ragas"])
    metrics: List[str] = field(default_factory=lambda: ["faithfulness", "answer_relevancy"])
    sample_size: Optional[int] = None


@dataclass
class RAGConfig:
    """Master configuration for the RAG system"""
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    vector_store_type: str = "faiss"  # faiss, chroma,
    pipeline_type: str = "linear"  # linear, conditional, iterative


class ConfigManager:
    """Manages configuration with YAML/JSON support and runtime updates"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> RAGConfig:
        """Load configuration from file or use defaults"""
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml'):  # yaml
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)  # json
            return self._dict_to_config(config_dict)
        return RAGConfig()
    
    def _dict_to_config(self, config_dict: dict) -> RAGConfig:
        """Convert dictionary to configuration objects"""
        retriever_config = RetrieverConfig(**config_dict.get('retriever', {}))
        generator_config = GeneratorConfig(**config_dict.get('generator', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        
        return RAGConfig(
            retriever=retriever_config,
            generator=generator_config,
            evaluation=evaluation_config,
            vector_store_type=config_dict.get('vector_store_type', 'faiss'),
            pipeline_type=config_dict.get('pipeline_type', 'linear')
        )
    
    def save_config(self, path: str):
        """Save current configuration to file"""
        config_dict = {
            'retriever': self.config.retriever.__dict__,
            'generator': self.config.generator.__dict__,
            'evaluation': self.config.evaluation.__dict__,
            'vector_store_type': self.config.vector_store_type,
            'pipeline_type': self.config.pipeline_type
        }
        
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
            'Retriever': self.config.retriever.__dict__,
            'Generator': self.config.generator.__dict__,
            'Evaluation': self.config.evaluation.__dict__,
            'Vector Store': self.config.vector_store_type,
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
    default_config = {
        'retriever': {
            'type': 'hybrid',
            'model': 'text-embedding-3-small',
            'chunk_size': 500,
            'chunk_overlap': 50,
            'top_k': 5,
            'search_type': 'similarity'
        },
        'generator': {
            'provider': 'openai',
            'model': 'gpt-4o-mini',
            'temperature': 0.0,
            'max_tokens': 1000
        },
        'evaluation': {
            'frameworks': ['ragas', 'custom'],
            'metrics': ['faithfulness', 'answer_relevancy', 'context_precision'],
            'sample_size': 100
        },
        'vector_store_type': 'faiss',
        'pipeline_type': 'linear'
    }
    
    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(save_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    return default_config