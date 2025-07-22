"""
Generator factory module for creating different generator strategies.
"""
from core.types import BaseChatModel, ChatOpenAI, ChatAnthropic, Ollama, ConfigurableField
from core.config import GeneratorConfig
from core.protocols import GeneratorStrategy


class GeneratorFactory:
    """Factory for creating different generator strategies"""
    
    def __init__(self):
        self._generators = {}
        
    def create_generator(self, config: GeneratorConfig) -> BaseChatModel:
        """Create generator based on configuration"""
        
        if config.provider == "openai":
            generator = ChatOpenAI(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            
        elif config.provider == "anthropic":
            generator = ChatAnthropic(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            
        elif config.provider == "ollama":
            generator = Ollama(
                model=config.model,
                temperature=config.temperature
            )
            
        else:
            raise ValueError(f"Unknown generator provider: {config.provider}")
        
        self._generators[f"{config.provider}_{config.model}"] = generator
        return generator
    
    def get_configurable_generator(self) -> BaseChatModel:
        """Create a configurable generator that can switch between models"""
        if not self._generators:
            raise ValueError("No generators created yet")
        
        # Get the first generator as default
        default_generator = list(self._generators.values())[0]
        
        # Create configurable alternatives
        return default_generator.configurable_alternatives(
            ConfigurableField(id="generator_model"),
            default_key=list(self._generators.keys())[0],
            **self._generators
        )