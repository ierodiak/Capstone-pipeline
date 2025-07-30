# RAG Architecture Showcase Notebook Guide

## Overview

The `04. RAG Architecture Showcase.ipynb` notebook is a comprehensive demonstration of our modular RAG (Retrieval-Augmented Generation) evaluation system. It follows the same structure as notebook 03 but focuses on showcasing the architecture's capabilities without requiring users to write additional code or classes.

## Notebook Structure

The notebook follows the same structure as `03. RAG New Modular Pipeline.ipynb`:

1. **Setup and Environment Configuration**
   - Import handling with nest_asyncio
   - API key verification
   - Project directory setup

2. **Basic Configuration**
   - Comprehensive RAGConfig setup
   - All component configurations

3. **Initialize RAG System**
   - System initialization
   - Available metrics display

4. **Document Processing**
   - Document loading and chunking
   - Vector store creation

5. **Building RAG Pipelines**
   - Linear pipeline construction

6. **Basic Querying**
   - Simple queries without evaluation

7. **Query with Evaluation**
   - Automatic evaluation without references

8. **Reference-based Evaluation**
   - Evaluation with ground truth

9. **Metric Groups**
   - Recommended metric sets

10. **A/B Testing**
    - Configuration variants comparison

11. **Results Analysis**
    - Visualization and statistics

12. **Custom Metrics**
    - Creating and registering new metrics

13. **Batch Evaluation**
    - Multiple queries and export

14. **Configuration Save**
    - Final summary and best practices

## Getting Started

1. **Setup Environment**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API Keys**
   Create a `.env` file with:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

3. **Run the Notebook**
   - Open `04. RAG Architecture Showcase.ipynb`
   - Run cells sequentially
   - Each section is self-contained

## Quick Start Templates

### Basic Configuration
```python
config = RAGConfig(
    experiment_name="my_experiment",
    chunker=ChunkerConfig(chunk_size=400),
    retrieval=RetrievalConfig(top_k=5)
)
```

### Initialize System
```python
system = ModularRAGSystem(ConfigManager(config))
system.initialize_components()
```

### Query with Evaluation
```python
result = await system.query(
    question="Your question here",
    evaluate=True,
    metrics=["ragas_faithfulness", "response_time"]
)
```

## Best Practices

1. **Start Simple**: Use default configurations first
2. **Iterate**: Test different configurations with A/B testing
3. **Monitor**: Check logs in `./logs/` directory
4. **Export**: Save successful configurations for reuse
5. **Extend**: Add custom metrics for your domain

## Troubleshooting

### Import Errors
- Ensure you're running from the project root
- Check that all `__init__.py` files exist
- Verify Python path includes project root

### API Key Issues
- Check `.env` file exists and is properly formatted
- Ensure API key has proper permissions

### Performance Issues
- Reuse vector stores when possible
- Batch queries for efficiency
- Use appropriate chunk sizes

## Next Steps

1. **Customize Components**: Add new strategies to factories
2. **Create Metrics**: Implement domain-specific metrics
3. **Run Experiments**: Use the experiment runner for grid search
4. **Deploy**: Create API endpoints for production use

## Support

For questions or issues:
1. Check the logs in `./logs/`
2. Review the architecture documentation
3. Examine the source code with type hints
4. Create an issue with reproducible examples 