# =====================================
# Root __init__.py (at the project root)
"""
Modular RAG Pipeline Package

A comprehensive, modular RAG (Retrieval-Augmented Generation) pipeline
with multi-framework evaluation support.
"""

__version__ = "0.1.0"

# Import main components for easier access
from system import ModularRAGSystem
from core import ConfigManager, ExperimentConfig
from utils import create_project_directories, verify_api_keys

__all__ = [
    "ModularRAGSystem",
    "ConfigManager",
    "ExperimentConfig",
    "create_project_directories",
    "verify_api_keys",
]