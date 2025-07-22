"""
Helper functions and utilities for the RAG pipeline.
"""
import os
from pathlib import Path
from typing import Dict, List, Any


def create_project_directories():
    """Create necessary directories for the project"""
    directories = {
        "documents": "./documents",
        "data": "./data",
        "configs": "./configs",
        "results": "./results",
        "logs": "./logs"
    }
    
    for name, path in directories.items():
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"Created {name} directory: {path}")
    
    return directories


def verify_api_keys():
    """Verify that required API keys are set"""
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY")  # Optional
    }
    
    verified = {}
    for key_name, key_value in api_keys.items():
        if key_value:
            print(f"{key_name} is configured")
            verified[key_name] = True
        else:
            print(f"{key_name} is not set")
            verified[key_name] = False
    
    return verified


def format_document_metadata(doc_metadata: Dict[str, Any]) -> str:
    """Format document metadata for display"""
    formatted = []
    for key, value in doc_metadata.items():
        formatted.append(f"{key}: {value}")
    return " | ".join(formatted)


def chunk_text_by_sentences(text: str, sentences_per_chunk: int = 5) -> List[str]:
    """Alternative chunking method by sentences"""
    sentences = text.split('. ')
    chunks = []
    
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = '. '.join(sentences[i:i + sentences_per_chunk])
        if chunk:
            chunks.append(chunk + '.')
    
    return chunks


def calculate_token_count(text: str, model: str = "gpt-4") -> int:
    """Estimate token count for a given text"""
    # Simple approximation: ~4 characters per token
    # For more accurate counting, use tiktoken library
    return len(text) // 4


def format_time_delta(seconds: float) -> str:
    """Format time delta in human-readable format"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"