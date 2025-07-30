"""
RAGAS evaluation framework implementation.
"""
from typing import List, Dict, Optional, Any
import pandas as pd
from core.types import ChatOpenAI, OpenAIEmbeddings
from core.config import MetricsConfig

try:
    from ragas import SingleTurnSample
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("RAGAS not installed. Install with: pip install ragas")

class RAGASEvaluator:
    """RAGAS evaluation framework adapter"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.metrics = {}
        self.metrics_requiring_reference = ['context_precision', 'context_recall']
        
        if RAGAS_AVAILABLE:
            # Initialize LLM and embeddings wrappers
            self.llm_wrapper = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
            self.embeddings_wrapper = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
            
            # Initialize metrics
            self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize RAGAS metrics based on configuration"""
        available_metrics = {
            "faithfulness": Faithfulness(llm=self.llm_wrapper),
            "answer_relevancy": AnswerRelevancy(llm=self.llm_wrapper, embeddings=self.embeddings_wrapper),
            "context_precision": ContextPrecision(llm=self.llm_wrapper),
            "context_recall": ContextRecall(llm=self.llm_wrapper)
        }
        
        # Only initialize metrics that are in config
        for metric_name in self.config.metric_names:
            if metric_name.startswith('ragas_'):
                actual_name = metric_name.replace('ragas_', '')
                if actual_name in available_metrics:
                    self.metrics[actual_name] = available_metrics[actual_name]
    
    async def evaluate_single(self, question: str, answer: str, contexts: List[str],
                            reference: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate a single RAG response"""
        if not RAGAS_AVAILABLE:
            return {"error": "RAGAS not available"}
        
        results = {}
        
        for metric_name, metric in self.metrics.items():
            try:
                # Check if metric requires reference
                if metric_name in self.metrics_requiring_reference and reference is None:
                    results[metric_name] = "N/A (no reference)"
                    continue
                
                # Create sample
                sample = SingleTurnSample(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=contexts,
                    reference=reference
                )
                
                # Evaluate
                score = await metric.single_turn_ascore(sample)
                results[f"ragas_{metric_name}"] = float(score) if score is not None else 0.0
                
            except Exception as e:
                results[f"ragas_{metric_name}"] = 0.0
        
        return results