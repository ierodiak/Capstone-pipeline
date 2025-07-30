# =====================================
# system/rag_system.py - UPDATED with modular metrics
"""
Main RAG system orchestrator with integrated modular metrics.
"""
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from IPython.display import display, Markdown
import pandas as pd

from core.config import ConfigManager, RAGConfig
from core.types import Document, BaseRetriever, BaseChatModel
from document_processing.processor import DocumentProcessor
from vector_stores.manager import VectorStoreManager
from factories.retriever import RetrieverFactory
from factories.generator import GeneratorFactory
from pipelines.builder import RAGPipelineBuilder
from evaluation.ragas_evaluator import RAGASEvaluator, RAGAS_AVAILABLE
from evaluation.orchestrator import EvaluationOrchestrator
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ModularRAGSystem:
    """Complete modular RAG system with flexible evaluation"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager.config
        self.config_manager = config_manager
        self.doc_processor = None
        self.vs_manager = None
        self.retriever_factory = None
        self.generator_factory = None
        self.pipeline_builder = None
        self.pipeline = None
        
        # Initialize both evaluation systems
        self.eval_orchestrator = None  # Legacy
        self.modular_evaluator = None  # New modular system
        self.metric_factory = None
        
    def initialize_components(self):
        """Initialize all system components"""
        logger.info("Initializing RAG system components")
        display(Markdown("## Initializing RAG System Components"))
        
        # Document processor
        self.doc_processor = DocumentProcessor(self.config.chunker)
        logger.info("Document processor initialized")
        print("✓ Document processor initialized")
        
        # Vector store manager
        self.vs_manager = VectorStoreManager(self.config)
        logger.info("Vector store manager initialized")
        print("✓ Vector store manager initialized")
        
        # Factories
        self.retriever_factory = RetrieverFactory()
        self.generator_factory = GeneratorFactory()
        logger.info("Component factories initialized")
        print("✓ Component factories initialized")
        
        # Pipeline builder
        self.pipeline_builder = RAGPipelineBuilder(
            self.retriever_factory,
            self.generator_factory
        )
        logger.info("Pipeline builder initialized")
        print("✓ Pipeline builder initialized")
        
        # Initialize metrics and evaluation
        self._initialize_evaluation()
        logger.info("Evaluation systems initialized")
        print("✓ Evaluation systems initialized")
        
    def _initialize_evaluation(self):
        """Initialize evaluation systems"""
        # Check if RAGAS should be used
        ragas_eval = None
        if any(m.startswith('ragas_') for m in self.config.metrics.metric_names):
            from evaluation.ragas_evaluator import RAGASEvaluator, RAGAS_AVAILABLE
            if RAGAS_AVAILABLE:
                ragas_eval = RAGASEvaluator(self.config.metrics)
        
        # Initialize orchestrator
        from evaluation.orchestrator import EvaluationOrchestrator
        self.eval_orchestrator = EvaluationOrchestrator(ragas_evaluator=ragas_eval)
        
    def load_documents(self, path: str, loader_type: Optional[str] = None) -> List[Document]:
        """Load documents from specified path"""
        if loader_type is None:
            # Infer loader type from path
            path_obj = Path(path)
            if path_obj.suffix == '.pdf' or (path_obj.is_dir() and any(path_obj.glob("**/*.pdf"))):
                loader_type = "pdf"
            else:
                loader_type = "text"
        
        return self.doc_processor.load_documents(path, loader_type)
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents into chunks"""
        return self.doc_processor.chunk_documents(documents)
    
    def load_and_process_documents(self, path: str, loader_type: Optional[str] = None) -> List[Document]:
        """Convenience method to load and process documents in one step"""
        documents = self.load_documents(path, loader_type)
        chunks = self.process_documents(documents)
        return chunks
    
    def create_or_load_vector_store(self, chunks: Optional[List[Document]] = None, force_rebuild: bool = False):
        """Create or load vector store"""
        display(Markdown("## Vector Store Management"))
        
        if not force_rebuild:
            # Try to load existing vector store
            vector_store = self.vs_manager.load_vector_store()
            if vector_store:
                self.retriever_factory.vector_store = vector_store
                return vector_store
        
        # Create new vector store
        if chunks is None:
            raise ValueError("Chunks required to create new vector store")
        
        vector_store = self.vs_manager.create_vector_store(chunks)
        self.retriever_factory.vector_store = vector_store
        self.retriever_factory.documents = chunks
        
        return vector_store
    
    def build_pipeline(self, pipeline_type: Optional[str] = None) -> Any:
        """Build RAG pipeline"""
        display(Markdown("## Building RAG Pipeline"))
        
        if pipeline_type is None:
            pipeline_type = self.config.pipeline_type
        
        # Create retriever
        retriever = self.retriever_factory.create_retriever(self.config.retrieval)
        print(f"Created {self.config.retrieval.strategy} retriever")
        
        # Create generator
        generator = self.generator_factory.create_generator(self.config.generation)
        print(f"Created {self.config.generation.provider} generator")
        
        # Build pipeline based on type
        if pipeline_type == "linear":
            self.pipeline = self.pipeline_builder.build_linear_pipeline(retriever, generator)
            print("Built linear pipeline")
            
        elif pipeline_type == "parallel":
            # Create multiple retrievers for parallel pipeline
            retrievers = {}
            
            # Vector retriever
            if self.retriever_factory.vector_store:
                from core.config import RetrievalConfig
                vector_config = RetrievalConfig(strategy="vector", top_k=self.config.retrieval.top_k)
                retrievers["vector"] = self.retriever_factory.create_retriever(vector_config)
            
            # BM25 retriever
            if self.retriever_factory.documents:
                from core.config import RetrievalConfig
                bm25_config = RetrievalConfig(strategy="bm25", top_k=self.config.retrieval.top_k)
                retrievers["bm25"] = self.retriever_factory.create_retriever(bm25_config)
            
            self.pipeline = self.pipeline_builder.build_parallel_pipeline(retrievers, generator)
            print("Built parallel pipeline")
            
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        
        return self.pipeline
    
    async def query(self, 
                   question: str,
                   evaluate: bool = True,
                   reference: Optional[str] = None,
                   use_modular: bool = True,
                   metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Query the RAG system with flexible evaluation options.
        
        Args:
            question: The question to ask
            evaluate: Whether to run evaluation
            reference: Optional reference answer for evaluation
            use_modular: Use new modular evaluation system (True) or legacy (False)
            metrics: Optional list of specific metrics to use
        """
        if not self.pipeline:
            raise ValueError("Pipeline not built. Call build_pipeline() first.")
        
        display(Markdown(f"### Query: {question}"))
        
        # Track timing
        start_time = time.time()
        
        # Get retriever for context tracking
        retriever = self.retriever_factory.create_retriever(self.config.retrieval)
        contexts = retriever.invoke(question)
        context_texts = [doc.page_content for doc in contexts]
        
        # Generate answer
        answer = await self.pipeline.ainvoke(question)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Prepare result
        result = {
            "question": question,
            "answer": answer,
            "contexts": context_texts,
            "response_time": response_time,
            "num_contexts": len(contexts),
            "config_id": self.config.get_variant_id()
        }
        
        # Display answer
        display(Markdown(f"**Answer:** {answer}"))
        print(f"\n⏱️ Response time: {response_time:.2f}s")
        print(f"📄 Retrieved {len(contexts)} contexts")
        
        # Run evaluation if requested
        if evaluate:
            if use_modular and self.modular_evaluator:
                result["evaluation"] = await self._evaluate_with_modular(
                    question, answer, context_texts, reference, response_time, metrics
                )
            elif self.eval_orchestrator:
                result["evaluation"] = await self._evaluate_with_legacy(
                    question, answer, context_texts, reference, response_time
                )
            else:
                logger.warning("No evaluation system available")
        
        return result
    
    async def _evaluate_with_modular(self, 
                                   question: str, 
                                   answer: str, 
                                   contexts: List[str], 
                                   reference: Optional[str],
                                   response_time: float,
                                   metrics: Optional[List[str]]) -> Dict[str, Any]:
        """Evaluate using the modular system"""
        # Use configured metrics or provided metrics
        if metrics is None:
            metrics = self.config.metrics.get_all_metrics(self.metric_factory.registry)
        
        eval_results = await self.modular_evaluator.evaluate(
            metric_names=metrics,
            question=question,
            answer=answer,
            contexts=contexts,
            reference=reference,
            response_time=response_time
        )
        
        # Convert to simple dict for compatibility
        return {
            name: {
                "value": res.value,
                "error": res.error,
                "computation_time": res.computation_time,
                "metadata": res.metadata
            }
            for name, res in eval_results.items()
        }
    
    async def _evaluate_with_legacy(self, 
                                  question: str, 
                                  answer: str, 
                                  contexts: List[str], 
                                  reference: Optional[str],
                                  response_time: float) -> Dict[str, Any]:
        """Evaluate using the legacy system"""
        return await self.eval_orchestrator.evaluate_comprehensive(
            question=question,
            answer=answer,
            contexts=contexts,
            response_time=response_time,
            reference=reference
        )
    
    async def batch_query(self, 
                         questions: List[Union[str, Dict[str, str]]], 
                         evaluate: bool = True,
                         use_modular: bool = True,
                         metrics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Process multiple queries in batch"""
        results = []
        
        for i, item in enumerate(questions):
            # Handle both string questions and dict with question/reference
            if isinstance(item, str):
                question = item
                reference = None
            else:
                question = item.get("question")
                reference = item.get("reference")
            
            logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
            
            result = await self.query(
                question=question,
                evaluate=evaluate,
                reference=reference,
                use_modular=use_modular,
                metrics=metrics
            )
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save evaluation results to file"""
        df = pd.DataFrame(results)
        
        if output_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
        elif output_path.endswith('.json'):
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError("Output path must be .csv or .json")
        
        logger.info(f"Results saved to {output_path}")
        print(f"📁 Results saved to: {output_path}")