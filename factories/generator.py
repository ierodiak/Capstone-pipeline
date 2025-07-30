"""
Generator factory implementing strategy pattern for different LLM providers.
"""
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

from core.types import BaseChatModel, ChatOpenAI, ChatAnthropic, Ollama
from core.config import GenerationConfig
from utils.logger import setup_logger

logger = setup_logger(__name__)


class GeneratorStrategy(ABC):
    """Abstract base class for generator strategies"""
    
    @abstractmethod
    def create(self, config: GenerationConfig) -> BaseChatModel:
        """Create a generator instance"""
        pass


class OpenAIGeneratorStrategy(GeneratorStrategy):
    """Strategy for OpenAI models"""
    
    def create(self, config: GenerationConfig) -> BaseChatModel:
        try:
            model = ChatOpenAI(
                model_name=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                streaming=True
            )
            logger.info(f"Created OpenAI generator with model {config.model}")
            return model
        except Exception as e:
            logger.error(f"Error creating OpenAI generator: {e}")
            raise


class AnthropicGeneratorStrategy(GeneratorStrategy):
    """Strategy for Anthropic models"""
    
    def create(self, config: GenerationConfig) -> BaseChatModel:
        try:
            model = ChatAnthropic(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            logger.info(f"Created Anthropic generator with model {config.model}")
            return model
        except Exception as e:
            logger.error(f"Error creating Anthropic generator: {e}")
            raise


class OllamaGeneratorStrategy(GeneratorStrategy):
    """Strategy for Ollama local models"""
    
    def create(self, config: GenerationConfig) -> BaseChatModel:
        try:
            model = Ollama(
                model=config.model,
                temperature=config.temperature,
                num_predict=config.max_tokens
            )
            logger.info(f"Created Ollama generator with model {config.model}")
            return model
        except Exception as e:
            logger.error(f"Error creating Ollama generator: {e}")
            raise


class GeneratorFactory:
    """Factory for creating generators using strategy pattern"""
    
    def __init__(self):
        self._strategies = {
            "openai": OpenAIGeneratorStrategy(),
            "anthropic": AnthropicGeneratorStrategy(),
            "ollama": OllamaGeneratorStrategy(),
        }
        self._cache: Dict[str, BaseChatModel] = {}
    
    def register_strategy(self, name: str, strategy: GeneratorStrategy):
        """Register a new generator strategy"""
        self._strategies[name] = strategy
        logger.info(f"Registered generator strategy: {name}")
    
    def create_generator(self, config: GenerationConfig, use_cache: bool = True) -> BaseChatModel:
        """
        Create a generator based on configuration.
        
        Args:
            config: Generation configuration
            use_cache: Whether to use cached instances
            
        Returns:
            Language model instance
        """
        # Create cache key
        cache_key = f"{config.provider}_{config.model}_{config.temperature}"
        
        # Check cache if enabled
        if use_cache and cache_key in self._cache:
            logger.info(f"Using cached generator for {cache_key}")
            return self._cache[cache_key]
        
        # Get strategy
        strategy = self._strategies.get(config.provider)
        if not strategy:
            raise ValueError(f"Unknown generator provider: {config.provider}")
        
        try:
            generator = strategy.create(config)
            
            # Cache the generator
            if use_cache:
                self._cache[cache_key] = generator
            
            logger.info(f"Successfully created {config.provider} generator")
            return generator
        except Exception as e:
            logger.error(f"Error creating {config.provider} generator: {e}")
            raise
    
    def clear_cache(self):
        """Clear the generator cache"""
        self._cache.clear()
        logger.info("Cleared generator cache")
    
    def get_available_providers(self) -> list:
        """Get list of available generator providers"""
        return list(self._strategies.keys())