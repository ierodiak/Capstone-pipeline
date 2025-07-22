"""
Iterative RAG pipeline using LangGraph for advanced patterns.
"""
from typing import Dict, Any, Literal
from core.types import (
    Document, BaseRetriever, BaseChatModel, RAGState,
    StateGraph, END
)


class IterativeRAGPipeline:
    """Iterative RAG pipeline using LangGraph"""
    
    def __init__(self, retriever: BaseRetriever, generator: BaseChatModel, max_iterations: int = 3):
        self.retriever = retriever
        self.generator = generator
        self.max_iterations = max_iterations
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> StateGraph:
        """Build the iterative workflow"""
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("evaluate_context", self._evaluate_context)
        workflow.add_node("generate", self._generate)
        workflow.add_node("evaluate_answer", self._evaluate_answer)
        
        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "evaluate_context")
        
        # Conditional edge based on context quality
        workflow.add_conditional_edges(
            "evaluate_context",
            self._should_continue_retrieval,
            {
                "continue": "retrieve",
                "generate": "generate"
            }
        )
        
        workflow.add_edge("generate", "evaluate_answer")
        
        # Conditional edge based on answer quality
        workflow.add_conditional_edges(
            "evaluate_answer",
            self._should_regenerate,
            {
                "regenerate": "retrieve",
                "finish": END
            }
        )
        
        return workflow.compile()
    
    def _retrieve(self, state: RAGState) -> RAGState:
        """Retrieve documents"""
        # Modify query based on iteration
        query = state["question"]
        if state.get("iteration", 0) > 0:
            query = f"{query} (iteration {state['iteration']})"
        
        docs = self.retriever.invoke(query)
        
        # Append to existing context
        existing_context = state.get("context", [])
        state["context"] = existing_context + docs
        state["iteration"] = state.get("iteration", 0) + 1
        
        return state
    
    def _evaluate_context(self, state: RAGState) -> RAGState:
        """Evaluate context quality"""
        # Simple heuristic: check if we have enough diverse content
        context_length = sum(len(doc.page_content) for doc in state["context"])
        unique_sources = len(set(doc.metadata.get("source", "") for doc in state["context"]))
        
        # Quality score based on length and diversity
        quality_score = min(1.0, (context_length / 2000) * (unique_sources / 3))
        state["quality_score"] = quality_score
        
        return state
    
    def _should_continue_retrieval(self, state: RAGState) -> Literal["continue", "generate"]:
        """Decide whether to continue retrieval"""
        if state["iteration"] >= self.max_iterations:
            return "generate"
        
        if state["quality_score"] < 0.7:
            return "continue"
        
        return "generate"
    
    def _generate(self, state: RAGState) -> RAGState:
        """Generate answer"""
        # Format context
        context = "\n\n".join([doc.page_content for doc in state["context"]])
        
        # Generate using the prompt
        prompt = f"Context: {context}\n\nQuestion: {state['question']}\n\nAnswer:"
        answer = self.generator.invoke(prompt).content
        
        state["answer"] = answer
        return state
    
    def _evaluate_answer(self, state: RAGState) -> RAGState:
        """Evaluate answer quality"""
        # Simple heuristic: check answer length and keywords
        answer_quality = min(1.0, len(state["answer"]) / 500)
        state["quality_score"] = answer_quality
        
        return state
    
    def _should_regenerate(self, state: RAGState) -> Literal["regenerate", "finish"]:
        """Decide whether to regenerate answer"""
        if state["iteration"] >= self.max_iterations:
            return "finish"
        
        if state["quality_score"] < 0.5 and state["iteration"] < 2:
            return "regenerate"
        
        return "finish"
    
    async def run(self, question: str) -> Dict[str, Any]:
        """Run the iterative pipeline"""
        initial_state = RAGState(
            question=question,
            context=[],
            answer="",
            iteration=0,
            quality_score=0.0
        )
        
        result = await self.workflow.ainvoke(initial_state)
        return result