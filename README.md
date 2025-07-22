# Modular RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) system with pluggable evaluation frameworks, following the latest LangChain v0.3+ patterns.

## Features

- **Modular Architecture**: Clean separation of concerns with Factory and Strategy patterns
- **Multiple Retrieval Strategies**: Vector, BM25, and Hybrid retrieval
- **Configurable Components**: YAML/JSON-based configuration management
- **Multi-Framework Evaluation**: RAGAS, custom metrics, and advanced evaluators
- **Pipeline Patterns**: Linear, parallel, and iterative (LangGraph) pipelines
- **Comprehensive Analysis**: Performance dashboards and detailed reports

## Project Structure

```
rag_pipeline/
├── core/               # Core abstractions and configuration
├── document_processing/# Document loading and chunking
├── vector_stores/      # Vector store management
├── factories/          # Component factories (Strategy pattern)
├── pipelines/          # Different pipeline implementations
├── evaluation/         # Evaluation frameworks
├── system/            # Main system orchestrator
├── analysis/          # Results analysis and visualization
├── utils/             # Helper functions
└── examples/          # Usage examples
```

## Installation

### Using pip

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

2. **Add your PDF documents**:
   ```bash
   mkdir documents
   # Copy your PDF files to the documents directory
   ```

3. **Run basic example**:
   ```bash
   python examples/basic_usage.py
   ```

## Usage

### Basic Pipeline

```python
from core.config import ConfigManager
from system.rag_system import ModularRAGSystem

# Initialize configuration
config_manager = ConfigManager()

# Initialize RAG system
rag_system = ModularRAGSystem(config_manager)
rag_system.initialize_components()

# Load and process documents
chunks = rag_system.load_and_process_documents("./documents")

# Create vector store
vector_store = rag_system.create_or_load_vector_store(chunks)

# Build pipeline
pipeline = rag_system.build_pipeline("linear")

# Query
result = await rag_system.query("What is the main topic?", evaluate=True)
```

### Configuration

Create a custom configuration file:

```yaml
retriever:
  type: hybrid
  model: text-embedding-3-small
  chunk_size: 500
  chunk_overlap: 50
  top_k: 5
  
generator:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.0
  max_tokens: 1000
  
evaluation:
  frameworks: [ragas, custom]
  metrics: [faithfulness, answer_relevancy]
```

### Advanced Features

#### Iterative Pipeline with LangGraph

```python
from pipelines.iterative import IterativeRAGPipeline

iterative_pipeline = IterativeRAGPipeline(
    retriever=retriever,
    generator=generator,
    max_iterations=3
)

result = await iterative_pipeline.run("Complex question?")
```

#### Custom Evaluation Framework

```python
from evaluation.advanced_evaluators import register_evaluation_framework

# Register custom evaluator
register_evaluation_framework(
    eval_orchestrator,
    'custom_framework',
    CustomEvaluatorClass,
    config
)
```

## Evaluation Frameworks

### Built-in Evaluators

1. **RAGAS**: Faithfulness, answer relevancy, context precision
2. **Custom**: Answer length, context coverage, response time
3. **BERGEN**: End-to-end benchmarking
4. **FlashRAG**: Comprehensive toolkit evaluation
5. **CoFE-RAG**: Full-chain stage-wise evaluation
6. **VERA**: Statistical validation and enhancement
7. **TRACe**: Explainable metrics
8. **OmniEval**: Omnidirectional task-based evaluation

### Adding New Evaluators

Implement the `EvaluatorStrategy` protocol:

```python
from core.protocols import EvaluatorStrategy

class MyEvaluator(EvaluatorStrategy):
    async def evaluate(self, data: Dict[str, Any]) -> Dict[str, float]:
        # Implementation
        pass
```

## Design Patterns

### Factory Pattern
- `RetrieverFactory`: Creates different retriever strategies
- `GeneratorFactory`: Creates different LLM providers

### Strategy Pattern
- Abstract base classes for pluggable components
- Easy extension with new implementations

### Builder Pattern
- `RAGPipelineBuilder`: Constructs different pipeline configurations

## Performance Analysis

The system includes comprehensive performance analysis:

- Response time distribution
- Metric correlations
- Time-series analysis
- Performance dashboards
- Excel/CSV export

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## License

MIT License

## Acknowledgments

- Built with LangChain v0.3+
- Evaluation frameworks: RAGAS, BERGEN, FlashRAG, etc.
- Inspired by modern RAG best practices