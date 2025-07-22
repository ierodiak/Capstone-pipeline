"""
Updated RAGAS evaluation framework adapter that properly handles all metrics with references.
"""
from typing import List, Dict, Optional, Any
import pandas as pd

from core.types import ChatOpenAI, OpenAIEmbeddings
from core.config import EvaluationConfig

try:
    from ragas import SingleTurnSample
    from ragas.metrics import (
        Faithfulness, 
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        AspectCritic
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("RAGAS not installed. Install with: pip install ragas")


class RAGASEvaluator:
    """RAGAS evaluation framework adapter"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.metrics = {}
        # Updated: Both context_precision and context_recall need references
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
        for metric_name in self.config.metrics:
            if metric_name in available_metrics:
                self.metrics[metric_name] = available_metrics[metric_name]
                print(f"Initialized metric: {metric_name}")
    
    async def evaluate_single(self, question: str, answer: str, contexts: List[str], 
                             reference: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate a single RAG response"""
        if not RAGAS_AVAILABLE:
            return {"error": "RAGAS not available"}
        
        print(f"Evaluating with reference: {reference is not None}")
        print(f"Metrics to evaluate: {list(self.metrics.keys())}")
        
        # Evaluate with each metric
        results = {}
        
        for metric_name, metric in self.metrics.items():
            try:
                # Check if metric requires reference
                if metric_name in self.metrics_requiring_reference and reference is None:
                    # Skip metrics that require reference when none is provided
                    results[metric_name] = "N/A (no reference provided)"
                    print(f"Skipping {metric_name} - no reference")
                    continue
                
                # Create sample with all required fields
                sample = SingleTurnSample(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=contexts,
                    reference=reference  # Always include reference if available
                )
                
                # Evaluate
                print(f"Evaluating {metric_name}...")
                score = await metric.single_turn_ascore(sample)
                results[metric_name] = float(score) if score is not None else 0.0
                print(f"{metric_name} score: {results[metric_name]}")
                
            except Exception as e:
                # More informative error handling
                error_msg = str(e)
                print(f"Error in {metric_name}: {error_msg}")
                
                if 'reference' in error_msg.lower():
                    results[metric_name] = "N/A (requires reference)"
                else:
                    results[metric_name] = f"Error: {error_msg[:50]}..."
        
        return results
    
    async def evaluate_batch(self, samples: List[Dict[str, Any]]) -> pd.DataFrame:
        """Evaluate a batch of samples"""
        all_results = []
        
        for sample in samples:
            result = await self.evaluate_single(
                question=sample["question"],
                answer=sample["answer"],
                contexts=sample["contexts"],
                reference=sample.get("reference")
            )
            result["question"] = sample["question"]
            all_results.append(result)
        
        return pd.DataFrame(all_results)
    
    def get_available_metrics(self, with_reference: bool = False) -> List[str]:
        """Get list of available metrics based on whether reference is available"""
        if with_reference:
            return list(self.metrics.keys())
        else:
            # Return only metrics that don't require reference
            return [m for m in self.metrics.keys() if m not in self.metrics_requiring_reference]