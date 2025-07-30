"""
CoFE-RAG full-chain evaluation implementation.
"""
from typing import Dict, List, Optional, Set
import re
from collections import Counter
import nltk
from core.types import Document

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger') 
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

class CoFERAGEvaluator:
    """CoFE-RAG full-chain evaluation framework."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['what', 'when', 'where', 'which', 'who', 'whom', 'why', 'how'])
        
    async def evaluate_single(self,
                            question: str,
                            answer: str,
                            contexts: List[str],
                            reference: Optional[str] = None,
                            chunks: Optional[List[Document]] = None) -> Dict[str, float]:
        """Evaluate using CoFE-RAG methodology."""
        scores = {}
        
        # Extract keywords
        keywords = self._extract_keywords(question, reference)
        
        # Stage evaluations
        if chunks:
            scores["cofe_chunking_quality"] = self._evaluate_chunking(chunks, keywords)
        
        scores["cofe_retrieval_recall"] = self._evaluate_retrieval_recall(contexts, keywords["fine"])
        scores["cofe_retrieval_accuracy"] = self._evaluate_retrieval_accuracy(contexts, keywords["coarse"])
        scores["cofe_generation_faithfulness"] = self._evaluate_generation_faithfulness(answer, contexts)
        scores["cofe_generation_relevance"] = self._evaluate_generation_relevance(answer, question, keywords)
        
        if reference:
            scores["cofe_generation_correctness"] = self._evaluate_generation_correctness(answer, reference)
        
        scores["cofe_pipeline_score"] = sum(scores.values()) / len(scores)
        
        return scores
    
    def _extract_keywords(self, question: str, reference: Optional[str] = None) -> Dict[str, List[str]]:
        """Extract multi-granularity keywords."""
        text = question + (" " + reference if reference else "")
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        
        # Coarse keywords (proper nouns, noun phrases)
        coarse = []
        i = 0
        while i < len(pos_tags):
            if pos_tags[i][1] in ['NNP', 'NNPS'] and pos_tags[i][0].lower() not in self.stop_words:
                phrase = [pos_tags[i][0]]
                j = i + 1
                while j < len(pos_tags) and pos_tags[j][1] in ['NNP', 'NNPS']:
                    phrase.append(pos_tags[j][0])
                    j += 1
                coarse.append(' '.join(phrase))
                i = j
            else:
                i += 1
        
        # Fine keywords
        fine = [t.lower() for t in tokens 
                if len(t) > 3 and t.lower() not in self.stop_words 
                and re.match(r'^[a-zA-Z0-9]+$', t) and not t.isdigit()]
        
        return {"coarse": coarse[:5], "fine": list(Counter(fine).most_common(7))}
    
    def _evaluate_chunking(self, chunks: List[Document], keywords: Dict[str, List[str]]) -> float:
        """Evaluate chunking quality."""
        if not chunks:
            return 0.0
            
        all_keywords = keywords["coarse"] + [kw[0] for kw in keywords["fine"]]
        if not all_keywords:
            return 1.0
        
        chunks_with_keywords = 0
        for chunk in chunks:
            if any(kw.lower() in chunk.page_content.lower() for kw in all_keywords):
                chunks_with_keywords += 1
        
        return chunks_with_keywords / len(chunks)
    
    def _evaluate_retrieval_recall(self, contexts: List[str], fine_keywords: List[tuple]) -> float:
        """Evaluate retrieval recall."""
        if not fine_keywords:
            return 1.0
        
        retrieved_text = " ".join(contexts).lower()
        found = sum(1 for kw, _ in fine_keywords if kw in retrieved_text)
        return found / len(fine_keywords)
    
    def _evaluate_retrieval_accuracy(self, contexts: List[str], coarse_keywords: List[str]) -> float:
        """Evaluate retrieval accuracy."""
        if not coarse_keywords:
            return 1.0
            
        retrieved_text = " ".join(contexts).lower()
        found = sum(1 for kw in coarse_keywords if kw.lower() in retrieved_text)
        return found / len(coarse_keywords)
    
    def _evaluate_generation_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Evaluate generation faithfulness."""
        if not contexts or not answer:
            return 0.0
            
        context_text = " ".join(contexts).lower()
        answer_sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if s.strip()]
        
        if not answer_sentences:
            return 0.0
        
        faithful = 0
        for sentence in answer_sentences:
            terms = set(re.findall(r'\b\w{4,}\b', sentence.lower()))
            if not terms or sum(1 for t in terms if t in context_text) >= len(terms) * 0.5:
                faithful += 1
                
        return faithful / len(answer_sentences)
    
    def _evaluate_generation_relevance(self, answer: str, question: str, keywords: Dict) -> float:
        """Evaluate generation relevance."""
        all_keywords = keywords["coarse"] + [kw[0] for kw in keywords["fine"]]
        if not all_keywords:
            return 1.0
            
        found = sum(1 for kw in all_keywords if kw.lower() in answer.lower())
        return found / len(all_keywords)
    
    def _evaluate_generation_correctness(self, answer: str, reference: str) -> float:
        """Evaluate generation correctness."""
        answer_terms = set(re.findall(r'\b\w{4,}\b', answer.lower()))
        ref_terms = set(re.findall(r'\b\w{4,}\b', reference.lower()))
        
        if not ref_terms:
            return 1.0
            
        intersection = len(answer_terms.intersection(ref_terms))
        union = len(answer_terms.union(ref_terms))
        return intersection / union if union > 0 else 0.0