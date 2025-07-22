"""
Custom evaluation metrics for RAG pipeline.
"""
from typing import List, Dict, Optional


class CustomEvaluator:
    """Custom evaluation metrics for RAG pipeline"""
    
    def __init__(self):
        self.metrics = {
            "answer_length": self._evaluate_answer_length,
            "context_coverage": self._evaluate_context_coverage,
            "response_time": self._evaluate_response_time,
            "keyword_presence": self._evaluate_keyword_presence
        }
    
    def _evaluate_answer_length(self, answer: str, contexts: List[str]) -> float:
        """Evaluate if answer length is appropriate"""
        context_length = sum(len(c) for c in contexts)
        answer_length = len(answer)
        
        # Ideal ratio: answer should be 10-20% of context length
        ratio = answer_length / max(context_length, 1)
        
        if 0.1 <= ratio <= 0.2:
            return 1.0
        elif ratio < 0.1:
            return ratio / 0.1
        else:
            return max(0, 1 - (ratio - 0.2) / 0.8)
    
    def _evaluate_context_coverage(self, answer: str, contexts: List[str]) -> float:
        """Evaluate how much of the context is covered in the answer"""
        # Simple word overlap metric
        answer_words = set(answer.lower().split())
        context_words = set()
        for context in contexts:
            context_words.update(context.lower().split())
        
        if not context_words:
            return 0.0
        
        overlap = answer_words.intersection(context_words)
        return len(overlap) / len(context_words)
    
    def _evaluate_response_time(self, response_time: float) -> float:
        """Evaluate response time (in seconds)"""
        # Ideal: < 2 seconds, Acceptable: < 5 seconds
        if response_time <= 2:
            return 1.0
        elif response_time <= 5:
            return 1 - (response_time - 2) / 3
        else:
            return max(0, 1 - (response_time - 5) / 10)
    
    def _evaluate_keyword_presence(self, answer: str, keywords: List[str]) -> float:
        """Evaluate presence of important keywords in answer"""
        if not keywords:
            return 1.0
        
        answer_lower = answer.lower()
        present = sum(1 for keyword in keywords if keyword.lower() in answer_lower)
        
        return present / len(keywords)
    
    def evaluate(self, answer: str, contexts: List[str], 
                 response_time: float, keywords: Optional[List[str]] = None) -> Dict[str, float]:
        """Run all custom evaluations"""
        results = {
            "answer_length": self._evaluate_answer_length(answer, contexts),
            "context_coverage": self._evaluate_context_coverage(answer, contexts),
            "response_time": self._evaluate_response_time(response_time)
        }
        
        if keywords:
            results["keyword_presence"] = self._evaluate_keyword_presence(answer, keywords)
        
        return results