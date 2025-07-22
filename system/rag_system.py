"""
Main RAG system orchestrator that integrates all components.
"""
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from IPython.display import display, Markdown
import pandas as pd

from core.config import ConfigManager, RetrieverConfig
from core.types import Document, BaseRetriever, BaseChatModel
from document_processing.processor import DocumentProcessor
from vector_stores.manager import VectorStoreManager
from factories.retriever import RetrieverFactory
from factories.generator import GeneratorFactory
from pipelines.builder import RAGPipelineBuilder
from evaluation.ragas_evaluator import RAGASEvaluator, RAGAS_AVAILABLE
from evaluation.custom_evaluator import CustomEvaluator
from evaluation.orchestrator import EvaluationOrchestrator


class ModularRAGSystem:
    """Complete modular RAG system with evaluation"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager.config
        self.doc_processor = None
        self.vs_manager = None
        self.retriever_factory = None
        self.generator_factory = None
        self.pipeline_builder = None
        self.pipeline = None
        self.eval_orchestrator = None
        
    def initialize_components(self):
        """Initialize all system components"""
        display(Markdown("## Initializing RAG System Components"))
        
        # Document processor
        self.doc_processor = DocumentProcessor(self.config.retriever)
        print("Document processor initialized")
        
        # Vector store manager
        self.vs_manager = VectorStoreManager(self.config)
        print("Vector store manager initialized")
        
        # Factories
        self.retriever_factory = RetrieverFactory()
        self.generator_factory = GeneratorFactory()
        print("Component factories initialized (Retriever and Generator)")
        
        # Pipeline builder
        self.pipeline_builder = RAGPipelineBuilder(
            self.retriever_factory, 
            self.generator_factory
        )
        print("Pipeline builder initialized")
        
        # Evaluation orchestrator
        evaluators = {
            "custom": CustomEvaluator(),
            "ragas": RAGASEvaluator(self.config.evaluation) if RAGAS_AVAILABLE else None
        }
        self.eval_orchestrator = EvaluationOrchestrator(evaluators)
        print("Evaluation orchestrator initialized")
        
    def load_and_process_documents(self, pdf_path: str) -> List[Document]:
        """Load and process documents"""
        display(Markdown("## Document Processing"))
        
        # Load documents
        documents = self.doc_processor.load_documents(pdf_path)
        
        # Chunk documents
        chunks = self.doc_processor.chunk_documents(documents)
        
        return chunks
    
    def create_or_load_vector_store(self, chunks: Optional[List[Document]] = None, 
                                   force_rebuild: bool = False) -> Any:
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
        retriever = self.retriever_factory.create_retriever(self.config.retriever)
        print(f"Created {self.config.retriever.type} retriever")
        
        # Create generator
        generator = self.generator_factory.create_generator(self.config.generator)
        print(f"Created {self.config.generator.provider} generator")
        
        # Build pipeline based on type
        if pipeline_type == "linear":
            self.pipeline = self.pipeline_builder.build_linear_pipeline(retriever, generator)
            print("Built linear pipeline")
            
        elif pipeline_type == "parallel":
            # Create multiple retrievers for parallel pipeline
            retrievers = {}
            
            # Vector retriever
            if self.retriever_factory.vector_store:
                retrievers["vector"] = self.retriever_factory.create_retriever(
                    RetrieverConfig(type="vector")
                )
            
            # BM25 retriever
            if self.retriever_factory.documents:
                retrievers["bm25"] = self.retriever_factory.create_retriever(
                    RetrieverConfig(type="bm25")
                )
            
            self.pipeline = self.pipeline_builder.build_parallel_pipeline(retrievers, generator)
            print("Built parallel pipeline")
            
        elif pipeline_type == "configurable":
            self.pipeline = self.pipeline_builder.build_configurable_pipeline(retriever, generator)
            print("Built configurable pipeline")
            
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        
        return self.pipeline
    
    async def query(self, question: str, evaluate: bool = True, reference: Optional[str] = None) -> Dict[str, Any]:
        """Query the RAG system
        
        Args:
            question: The question to ask
            evaluate: Whether to run evaluation
            reference: Optional reference answer for evaluation metrics that require it
        """
        if not self.pipeline:
            raise ValueError("Pipeline not built. Call build_pipeline() first.")
        
        display(Markdown(f"### Query: {question}"))
        
        # Track timing
        start_time = time.time()
        
        # Get retriever for context tracking
        retriever = self.retriever_factory.create_retriever(self.config.retriever)
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
            "num_contexts": len(contexts)
        }
        
        # Display answer
        display(Markdown(f"### Answer:\n{answer}"))
        print(f"\nResponse time: {response_time:.2f} seconds")
        print(f"Used {len(contexts)} context chunks")
        
        # Evaluation
        if evaluate and self.eval_orchestrator:
            display(Markdown("### Evaluation Results"))
            
            eval_results = await self.eval_orchestrator.evaluate_comprehensive(
                question=question,
                answer=answer,
                contexts=context_texts,
                response_time=response_time,
                reference=reference
            )
            
            # Display evaluation results
            self._display_evaluation_results(eval_results)
            
            result["evaluation"] = eval_results
        
        return result
    
    def _display_evaluation_results(self, eval_results: Dict[str, Any]):
        """Display evaluation results in a formatted way"""
        
        # RAGAS results
        if "ragas" in eval_results:
            display(Markdown("#### RAGAS Metrics"))
            ragas_df = pd.DataFrame([eval_results["ragas"]])
            display(ragas_df)
        
        # Custom results
        if "custom" in eval_results:
            display(Markdown("#### Custom Metrics"))
            custom_df = pd.DataFrame([eval_results["custom"]])
            display(custom_df)
    
    def run_benchmark(self, questions: List[str]) -> pd.DataFrame:
        """Run benchmark on multiple questions"""
        display(Markdown("## Running Benchmark"))
        
        results = []
        
        for i, question in enumerate(questions):
            display(Markdown(f"### Question {i+1}/{len(questions)}"))
            
            result = asyncio.run(self.query(question, evaluate=True))
            results.append(result)
            
            print("\n" + "="*80 + "\n")
        
        # Display overall results
        display(Markdown("## Benchmark Summary"))
        self.eval_orchestrator.plot_evaluation_results()
        
        return self.eval_orchestrator.get_results_summary()