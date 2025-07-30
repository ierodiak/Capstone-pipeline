"""
Orchestrates evaluation using RAGAS, CoFE-RAG, and OmniEval.
"""
from typing import Dict, List, Optional, Any
import pandas as pd
from IPython.display import display, Markdown

from .ragas_evaluator import RAGASEvaluator
from .cofe_evaluator import CoFERAGEvaluator
from .omnieval_evaluator import OmniEvalEvaluator
from core.types import Document
from utils.logger import setup_logger

logger = setup_logger(__name__)

class EvaluationOrchestrator:
    """Orchestrates multiple evaluation frameworks."""
    
    def __init__(self, ragas_evaluator: Optional[RAGASEvaluator] = None,
                 cofe_evaluator: Optional[CoFERAGEvaluator] = None,
                 omni_evaluator: Optional[OmniEvalEvaluator] = None):
        """Initialize with evaluator instances."""
        self.ragas_evaluator = ragas_evaluator
        self.cofe_evaluator = cofe_evaluator or CoFERAGEvaluator()
        self.omni_evaluator = omni_evaluator or OmniEvalEvaluator()
        self.results_history = []
        
    async def evaluate_comprehensive(self,
                                   question: str,
                                   answer: str,
                                   contexts: List[str],
                                   response_time: float,
                                   reference: Optional[str] = None,
                                   chunks: Optional[List[Document]] = None) -> Dict[str, Any]:
        """Run comprehensive evaluation using all evaluators."""
        results = {
            "response_time": response_time,
            "num_contexts": len(contexts)
        }
        
        # RAGAS evaluation
        if self.ragas_evaluator:
            try:
                ragas_results = await self.ragas_evaluator.evaluate_single(
                    question, answer, contexts, reference
                )
                results.update(ragas_results)
            except Exception as e:
                logger.error(f"RAGAS evaluation failed: {e}")
                results.update({"ragas_faithfulness": 0.0, "ragas_answer_relevancy": 0.0})
        
        # CoFE-RAG evaluation
        try:
            cofe_results = await self.cofe_evaluator.evaluate_single(
                question, answer, contexts, reference, chunks
            )
            results.update(cofe_results)
        except Exception as e:
            logger.error(f"CoFE-RAG evaluation failed: {e}")
            results["cofe_pipeline_score"] = 0.0
        
        # OmniEval evaluation
        try:
            omni_results = await self.omni_evaluator.evaluate_single(
                question, answer, contexts, reference
            )
            results.update(omni_results)
        except Exception as e:
            logger.error(f"OmniEval evaluation failed: {e}")
            results["omni_weighted_score"] = 0.0
        
        # Calculate aggregate score
        results["aggregate_score"] = self._calculate_aggregate_score(results)
        
        # Store in history
        self.results_history.append({
            "question": question,
            "results": results,
            "reference": reference is not None
        })
        
        return results
    
    def _calculate_aggregate_score(self, results: Dict[str, float]) -> float:
        """Calculate weighted aggregate score."""
        scores = []
        weights = []
        
        # RAGAS
        if "ragas_faithfulness" in results and "ragas_answer_relevancy" in results:
            scores.append((results["ragas_faithfulness"] + results["ragas_answer_relevancy"]) / 2)
            weights.append(0.3)
        
        # CoFE-RAG
        if "cofe_pipeline_score" in results:
            scores.append(results["cofe_pipeline_score"])
            weights.append(0.3)
        
        # OmniEval
        if "omni_weighted_score" in results:
            scores.append(results["omni_weighted_score"])
            weights.append(0.4)
        
        return sum(s * w for s, w in zip(scores, weights)) / sum(weights) if weights else 0.0
    
    def get_results_summary(self) -> pd.DataFrame:
        """Get summary of all evaluation results."""
        if not self.results_history:
            return pd.DataFrame()
        
        rows = []
        for item in self.results_history:
            row = {"question": item["question"][:50] + "..."}
            row.update(item["results"])
            rows.append(row)
        
        return pd.DataFrame(rows)