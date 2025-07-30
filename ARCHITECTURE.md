# RAG Evaluation Pipeline - System Architecture

## Overview

This repository implements a modular, extensible RAG (Retrieval-Augmented Generation) evaluation system using factory and strategy design patterns throughout. The system is designed for easy experimentation, A/B testing, and comprehensive evaluation of RAG pipelines.

## Core Design Principles

1. **Modularity**: Each component is independent and can be swapped or extended
2. **Factory Pattern**: Consistent creation of components through factories
3. **Strategy Pattern**: Different implementations for each component type
4. **Async-First**: Built for high performance with async/await
5. **Type Safety**: Full type hints and protocols for better IDE support

## Architecture Components

### 1. Configuration System (`core/config.py`)

The configuration system uses dataclasses for type-safe configuration:

```python
# Master configuration
RAGConfig
├── LoaderConfig      # Document loading settings
├── ChunkerConfig     # Text chunking strategy
├── EmbeddingConfig   # Embedding model configuration
├── StorageConfig     # Vector store settings
├── RetrievalConfig   # Retrieval strategy configuration
├── GenerationConfig  # LLM generation settings
└── MetricsConfig     # Evaluation metrics configuration
```

**ConfigManager**: Handles configuration loading from files or objects
- Supports YAML/JSON configuration files
- Runtime configuration updates
- Configuration validation

### 2. Document Processing (`document_processing/`)

**Factory Pattern Implementation**:
```python
DocumentProcessorFactory
├── DocumentLoaderStrategy (Abstract)
│   ├── PDFLoaderStrategy
│   └── TextLoaderStrategy
└── ChunkerStrategy (Abstract)
    ├── RecursiveChunkerStrategy
    └── SemanticChunkerStrategy
```

**Features**:
- Multiple document format support
- Configurable chunking strategies
- Automatic metadata enrichment
- Visual chunk analysis

### 3. Vector Store Management (`vector_stores/`)

**VectorStoreManager**: Centralized vector store operations
- Multiple backend support (FAISS, Chroma, Pinecone, etc.)
- Automatic persistence and loading
- Index optimization
- Retrieval testing utilities

### 4. Retrieval Factory (`factories/retriever.py`)

**Strategy Pattern Implementation**:
```python
RetrieverFactory
└── RetrieverStrategy (Abstract)
    ├── VectorRetrieverStrategy
    ├── BM25RetrieverStrategy
    ├── HybridRetrieverStrategy
    └── MultiQueryRetrieverStrategy
```

**Features**:
- Multiple retrieval strategies
- Configurable search parameters
- Ensemble retrieval support
- Custom strategy registration

### 5. Generation Factory (`factories/generator.py`)

**Strategy Pattern Implementation**:
```python
GeneratorFactory
└── GeneratorStrategy (Abstract)
    ├── OpenAIGeneratorStrategy
    ├── AnthropicGeneratorStrategy
    └── OllamaGeneratorStrategy
```

**Features**:
- Multiple LLM provider support
- Response caching
- Streaming support
- Temperature and parameter control

### 6. Pipeline Builder (`pipelines/builder.py`)

**RAGPipelineBuilder**: Constructs different pipeline patterns
- Linear Pipeline: Standard retrieve → generate flow
- Parallel Pipeline: Multiple retrievers with fusion
- Iterative Pipeline: Multi-step refinement (planned)

### 7. Evaluation System

#### 7.1 Metrics Registry (`core/metrics_registry.py`)

**Pluggable Metric System**:
```python
MetricsRegistry
└── BaseMetric (Abstract)
    ├── Requirements (what the metric needs)
    ├── Computation logic
    └── Result format
```

#### 7.2 Metric Factory (`metrics/metric_factory.py`)

**Pre-registered Metrics**:
- RAGAS metrics (faithfulness, relevancy, precision/recall)
- Custom metrics (completeness, efficiency, utilization)
- Performance metrics (response time, token usage)

#### 7.3 Modular Evaluator (`evaluation/modular_evaluator.py`)

**Features**:
- Automatic metric selection based on available data
- Parallel metric computation
- Error handling per metric
- Extensible for new metrics

### 8. Main System (`system/rag_system.py`)

**ModularRAGSystem**: Orchestrates all components
- Component initialization
- Document processing pipeline
- Query execution
- Evaluation coordination
- Batch processing
- Result export

## Usage Patterns

### Basic Usage

```python
# 1. Create configuration
config = RAGConfig(
    chunker=ChunkerConfig(method="recursive", chunk_size=500),
    retrieval=RetrievalConfig(strategy="hybrid", top_k=5),
    generation=GenerationConfig(provider="openai", model="gpt-4")
)

# 2. Initialize system
rag_system = ModularRAGSystem(ConfigManager(config))
rag_system.initialize_components()

# 3. Process documents
chunks = rag_system.load_and_process_documents("./documents")
vector_store = rag_system.create_or_load_vector_store(chunks)

# 4. Build pipeline
pipeline = rag_system.build_pipeline("linear")

# 5. Query with evaluation
result = await rag_system.query(
    question="What is RAG?",
    evaluate=True,
    metrics=["ragas_faithfulness", "response_time"]
)
```

### Adding Custom Components

#### Custom Metric

```python
class CustomMetric(BaseMetric):
    def __init__(self):
        super().__init__(
            name="custom_metric",
            requirements=MetricRequirements(requires_contexts=True)
        )
    
    async def compute(self, **kwargs) -> MetricResult:
        # Implementation
        return MetricResult(metric_name=self.name, value=score)

# Register
rag_system.metric_factory.registry.register_metric(
    CustomMetric(), 
    groups=["custom"]
)
```

#### Custom Retriever

```python
class CustomRetrieverStrategy(RetrieverStrategy):
    def create(self, config: RetrievalConfig, **kwargs) -> BaseRetriever:
        # Implementation
        return custom_retriever

# Register
rag_system.retriever_factory.register_strategy(
    "custom", 
    CustomRetrieverStrategy()
)
```

## Configuration Examples

### Experiment Configuration

```yaml
experiment_name: "high_quality_rag"
tags: ["production", "quality_focus"]

chunker:
  method: "semantic"
  chunk_size: 300
  semantic_threshold: 0.8

retrieval:
  strategy: "hybrid"
  top_k: 10
  hybrid_weights: [0.6, 0.4]

generation:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.1

metrics:
  metric_names:
    - "ragas_faithfulness"
    - "ragas_answer_relevancy"
    - "semantic_similarity"
  metric_groups:
    - "quality"
```

### A/B Testing Setup

```python
configs = {
    "baseline": RAGConfig(...),
    "experimental": RAGConfig(...),
}

results = {}
for name, config in configs.items():
    system = ModularRAGSystem(ConfigManager(config))
    # ... run experiments
    results[name] = await system.batch_query(test_questions)
```

## Best Practices

1. **Configuration Management**
   - Save successful configurations for reuse
   - Use meaningful experiment names and tags
   - Version control your configurations

2. **Component Development**
   - Follow the established factory/strategy patterns
   - Add comprehensive logging
   - Include error handling
   - Write unit tests for new components

3. **Evaluation**
   - Choose metrics based on your use case
   - Use reference answers when available
   - Run multiple iterations for stability
   - Export results for further analysis

4. **Performance**
   - Reuse vector stores when possible
   - Use async operations for parallelism
   - Enable caching for generators
   - Batch operations when feasible

## Future Extensions

The architecture supports easy addition of:
- New document loaders (e.g., web scraping, databases)
- Advanced chunking strategies (e.g., semantic, hierarchical)
- Additional vector stores (e.g., Weaviate, Qdrant)
- Novel retrieval methods (e.g., GraphRAG, tree-based)
- More LLM providers (e.g., Gemini, Claude)
- Custom evaluation frameworks
- Pipeline patterns (e.g., iterative, multi-agent)

## Troubleshooting

See `docs/RAGAS_Troubleshooting.md` for common issues and solutions. 