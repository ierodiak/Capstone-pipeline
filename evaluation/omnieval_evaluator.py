"""
OmniEval omnidirectional evaluation implementation.
"""
from typing import Dict, Optional, List
import re
from enum import Enum

class TaskType(Enum):
    FACTUAL_QA = "factual_qa"
    REASONING = "reasoning"
    COMPARATIVE = "comparative"
    LONG_FORM = "long_form"

class OmniEvalEvaluator:
    """OmniEval omnidirectional evaluation framework."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.task_weights = {
            TaskType.FACTUAL_QA: {"accuracy": 0.5, "completeness": 0.2, "hallucination": 0.2, "utilization": 0.1},
            TaskType.REASONING: {"accuracy": 0.3, "completeness": 0.4, "hallucination": 0.2, "utilization": 0.1},
            TaskType.COMPARATIVE: {"accuracy": 0.4, "completeness": 0.3, "hallucination": 0.2, "utilization": 0.1},
            TaskType.LONG_FORM: {"accuracy": 0.2, "completeness": 0.5, "hallucination": 0.2, "utilization": 0.1}
        }
    
    async def evaluate_single(self,
                            question: str,
                            answer: str,
                            contexts: List[str],
                            reference: Optional[str] = None) -> Dict[str, float]:
        """Evaluate using OmniEval methodology."""
        # Classify task
        task_type = self._classify_task(question)
        
        # Calculate metrics
        scores = {
            "omni_accuracy": self._evaluate_accuracy(answer, reference, task_type, question),
            "omni_completeness": self._evaluate_completeness(answer, task_type),
            "omni_hallucination": self._evaluate_hallucination(answer, contexts),
            "omni_utilization": self._evaluate_utilization(answer, contexts)
        }
        
        # Weighted score
        weights = self.task_weights[task_type]
        scores["omni_weighted_score"] = sum(
            scores[f"omni_{m}"] * w for m, w in weights.items()
        )
        
        # Task-specific
        if task_type == TaskType.REASONING:
            scores["omni_reasoning_quality"] = self._evaluate_reasoning_quality(answer)
        elif task_type == TaskType.COMPARATIVE:
            scores["omni_comparison_quality"] = self._evaluate_comparison_quality(answer)
        
        scores["omni_task_type"] = task_type.value
        
        return scores
    
    def _classify_task(self, question: str) -> TaskType:
        """Classify question into task type."""
        q_lower = question.lower()
        
        # Pattern matching
        if re.search(r'compare|versus|vs\.?|difference between', q_lower):
            return TaskType.COMPARATIVE
        elif re.search(r'why|how does|explain|what causes', q_lower):
            return TaskType.REASONING
        elif re.search(r'discuss|elaborate|comprehensive|detailed', q_lower):
            return TaskType.LONG_FORM
        else:
            return TaskType.FACTUAL_QA
    
    def _evaluate_accuracy(self, answer: str, reference: Optional[str], 
                          task_type: TaskType, question: str) -> float:
        """Evaluate accuracy based on task type."""
        if not reference:
            return 0.5
            
        if task_type == TaskType.FACTUAL_QA:
            # Fact matching
            answer_facts = set(re.findall(r'\b\d+\b|\b[A-Z][a-z]+\b', answer))
            ref_facts = set(re.findall(r'\b\d+\b|\b[A-Z][a-z]+\b', reference))
            return len(answer_facts.intersection(ref_facts)) / len(ref_facts) if ref_facts else 0.0
        else:
            # Term overlap
            answer_terms = set(re.findall(r'\b\w{4,}\b', answer.lower()))
            ref_terms = set(re.findall(r'\b\w{4,}\b', reference.lower()))
            return len(answer_terms.intersection(ref_terms)) / len(ref_terms) if ref_terms else 0.0
    
    def _evaluate_completeness(self, answer: str, task_type: TaskType) -> float:
        """Evaluate completeness based on expected length."""
        word_count = len(answer.split())
        expected_ranges = {
            TaskType.FACTUAL_QA: (10, 50),
            TaskType.REASONING: (30, 100),
            TaskType.COMPARATIVE: (40, 120),
            TaskType.LONG_FORM: (100, 300)
        }
        
        min_len, ideal_len = expected_ranges[task_type]
        
        if word_count < min_len:
            return word_count / min_len * 0.5
        elif word_count < ideal_len:
            return 0.5 + (word_count - min_len) / (ideal_len - min_len) * 0.5
        else:
            return 1.0
    
    def _evaluate_hallucination(self, answer: str, contexts: List[str]) -> float:
        """Evaluate hallucination (1.0 = no hallucination)."""
        if not contexts:
            return 0.5
            
        context_text = " ".join(contexts).lower()
        answer_terms = set(re.findall(r'\b\w{4,}\b', answer.lower()))
        
        if not answer_terms:
            return 1.0
            
        grounded = sum(1 for term in answer_terms if term in context_text)
        return grounded / len(answer_terms)
    
    def _evaluate_utilization(self, answer: str, contexts: List[str]) -> float:
        """Evaluate context utilization."""
        if not contexts:
            return 0.0
            
        answer_terms = set(re.findall(r'\b\w{4,}\b', answer.lower()))
        utilized = 0
        
        for context in contexts:
            context_terms = set(re.findall(r'\b\w{4,}\b', context.lower()))
            if len(answer_terms.intersection(context_terms)) >= 3:
                utilized += 1
                
        return utilized / len(contexts)
    
    def _evaluate_reasoning_quality(self, answer: str) -> float:
        """Evaluate reasoning quality."""
        patterns = [r'first|initially', r'then|next', r'finally|therefore|thus']
        matches = sum(1 for p in patterns if re.search(p, answer.lower()))
        return min(1.0, matches / len(patterns))
    
    def _evaluate_comparison_quality(self, answer: str) -> float:
        """Evaluate comparison quality."""
        comparison_terms = ["similar", "different", "whereas", "while", "both", "neither"]
        found = sum(1 for term in comparison_terms if term in answer.lower())
        return min(1.0, found / 4)