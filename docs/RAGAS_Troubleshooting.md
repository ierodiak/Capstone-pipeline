# RAGAS Troubleshooting Guide

## Common Issues and Solutions

### 1. RAGAS Installation Issues

```bash
# If RAGAS is not installed
pip install ragas

# For specific version compatibility
pip install ragas==0.2.0  # or latest stable version
```

### 2. Reference Answer Requirements

**Problem**: Getting "N/A (no reference provided)" for some metrics

**Solution**: The following metrics require reference answers:
- `context_precision`: Needs reference to measure precision of retrieved contexts
- `context_recall`: Needs reference to measure recall of retrieved contexts

Metrics that work WITHOUT reference:
- `faithfulness`: Only needs answer and contexts
- `answer_relevancy`: Only needs question and answer

### 3. Type Errors in Evaluation

**Problem**: Type mismatch when returning evaluation results

**Solution**: The `ragas_evaluator.py` has been updated to return `Dict[str, Any]` instead of `Dict[str, float]` to handle mixed return types.

### 4. Using RAGAS with the RAG System

#### Option 1: Direct Query with Reference (after system update)
```python
result = await rag_system.query(
    "Your question here",
    evaluate=True,
    reference="Your reference answer here"
)
```

#### Option 2: Manual Evaluation with Orchestrator
```python
# Query without evaluation
result = await rag_system.query("Your question", evaluate=False)

# Manually evaluate with reference
eval_results = await rag_system.eval_orchestrator.evaluate_comprehensive(
    question="Your question",
    answer=result["answer"],
    contexts=result["contexts"],
    response_time=result["response_time"],
    reference="Your reference answer"
)
```

#### Option 3: Direct RAGAS Evaluator
```python
from evaluation.ragas_evaluator import RAGASEvaluator

ragas_evaluator = RAGASEvaluator(config)
scores = await ragas_evaluator.evaluate_single(
    question="Your question",
    answer="Generated answer",
    contexts=["context1", "context2"],
    reference="Reference answer"  # Optional
)
```

### 5. Handling Mixed Evaluation Scenarios

When evaluating multiple questions with mixed reference availability:

```python
for sample in evaluation_samples:
    if "reference" in sample:
        # Full evaluation with all metrics
        result = await rag_system.query(
            sample["question"],
            evaluate=True,
            reference=sample["reference"]
        )
    else:
        # Limited evaluation without reference-dependent metrics
        result = await rag_system.query(
            sample["question"],
            evaluate=True,
            reference=None
        )
```

### 6. Async/Await Issues in Jupyter Notebooks

**Problem**: "RuntimeError: This event loop is already running"

**Solution**: Use nest_asyncio
```python
import nest_asyncio
nest_asyncio.apply()
```

### 7. Memory Issues with Large Evaluations

**Solution**: Process in batches
```python
batch_size = 10
for i in range(0, len(questions), batch_size):
    batch = questions[i:i+batch_size]
    # Process batch
```

### 8. Debugging RAGAS Metrics

To understand what's happening inside RAGAS evaluation:

```python
# Enable verbose mode in RAGAS evaluator
ragas_evaluator = RAGASEvaluator(config)

# Check available metrics
print("With reference:", ragas_evaluator.get_available_metrics(with_reference=True))
print("Without reference:", ragas_evaluator.get_available_metrics(with_reference=False))
```

### 9. Best Practices

1. **Reference Answer Quality**: Keep reference answers concise and factual
2. **Context Length**: Ensure contexts are not too long (can affect metrics)
3. **Metric Selection**: Choose metrics based on your evaluation goals
4. **Error Handling**: Always check for RAGAS_AVAILABLE before using
5. **Performance**: Run evaluation on samples, not every query in production

### 10. Metric Interpretation

- **Faithfulness** (0-1): Higher is better. Measures if answer is grounded in context
- **Answer Relevancy** (0-1): Higher is better. Measures answer relevance to question
- **Context Precision** (0-1): Higher is better. Measures precision of retrieval
- **Context Recall** (0-1): Higher is better. Measures recall of retrieval

### Need More Help?

Check the [main tutorial](../01.%20RAG%20Pipeline%20Tutorial.ipynb) for complete examples. 