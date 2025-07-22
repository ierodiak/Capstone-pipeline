"""
Advanced pipeline example demonstrating configuration updates, 
multiple retrievers, and custom evaluation frameworks.
"""
import asyncio
from pathlib import Path

# Add parent directory to path to import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.config import ConfigManager, RetrieverConfig
from system.rag_system import ModularRAGSystem
from pipelines.iterative import IterativeRAGPipeline
from evaluation.advanced_evaluators import (
    CoFERAGEvaluator, VERAEvaluator, TRACeEvaluator,
    register_evaluation_framework
)
from analysis.results import ResultsAnalyzer
from utils.helpers import create_project_directories, verify_api_keys


async def test_iterative_pipeline(rag_system):
    """Test the iterative RAG pipeline with LangGraph"""
    print("\n=== Testing Iterative Pipeline ===")
    
    # Create iterative pipeline
    retriever = rag_system.retriever_factory.create_retriever(rag_system.config.retriever)
    generator = rag_system.generator_factory.create_generator(rag_system.config.generator)
    
    iterative_pipeline = IterativeRAGPipeline(
        retriever=retriever,
        generator=generator,
        max_iterations=3
    )
    
    # Test question
    question = "What are the most important technical details in the document?"
    
    print(f"\nRunning iterative pipeline for: '{question}'")
    result = await iterative_pipeline.run(question)
    
    print(f"\nIterations completed: {result['iteration']}")
    print(f"Final quality score: {result['quality_score']:.2f}")
    print(f"\nAnswer: {result['answer'][:200]}...")
    
    return result


async def test_advanced_evaluators(rag_system):
    """Test advanced evaluation frameworks"""
    print("\n=== Testing Advanced Evaluators ===")
    
    # Register CoFE-RAG evaluator
    cofe_config = {
        'use_reranker': False,
        'stages': ['chunking', 'retrieval', 'generation']
    }
    cofe_evaluator = register_evaluation_framework(
        rag_system.eval_orchestrator,
        'cofe_rag',
        CoFERAGEvaluator,
        cofe_config
    )
    
    # Register VERA evaluator
    vera_config = {
        'bootstrap_samples': 100,
        'confidence_level': 0.95,
        'enhancement_threshold': 0.7
    }
    vera_evaluator = register_evaluation_framework(
        rag_system.eval_orchestrator,
        'vera',
        VERAEvaluator,
        vera_config
    )
    
    # Register TRACe evaluator
    trace_config = {
        'metrics': ['relevance', 'utilization', 'completeness', 'adherence']
    }
    trace_evaluator = register_evaluation_framework(
        rag_system.eval_orchestrator,
        'trace',
        TRACeEvaluator,
        trace_config
    )
    
    # Run pipeline-level evaluation with CoFE-RAG
    print("\n1. Running CoFE-RAG Pipeline Evaluation...")
    test_data = [
        {
            'question': 'What is the main methodology?',
            'contexts': ['Sample context 1', 'Sample context 2'],
            'answer': 'The methodology involves...'
        }
    ]
    
    cofe_results = await cofe_evaluator.evaluate_pipeline(rag_system, test_data)
    print(f"CoFE-RAG Results: {cofe_results}")
    
    # Run explainable evaluation with TRACe
    print("\n2. Running TRACe Explainable Evaluation...")
    trace_results = await trace_evaluator.evaluate_explainable(test_data)
    for result in trace_results:
        print(f"\nTRACe Metrics:")
        print(f"  - Relevance: {result['relevance']:.2f}")
        print(f"  - Utilization: {result['utilization']:.2f}")
        print(f"  - Completeness: {result['completeness']:.2f}")
        print(f"  - Adherence: {result['adherence']}")
        print(f"  - Explanation: {result['explanation']}")


def test_configuration_updates(config_manager, rag_system):
    """Test dynamic configuration updates"""
    print("\n=== Testing Configuration Updates ===")
    
    # Update to hybrid retriever with different parameters
    print("\n1. Updating to Hybrid Retriever Configuration...")
    new_config = {
        'retriever': {
            'type': 'hybrid',
            'top_k': 7,
            'chunk_size': 300,
            'chunk_overlap': 75
        },
        'generator': {
            'temperature': 0.3,
            'max_tokens': 1500
        }
    }
    
    config_manager.update_config(new_config)
    config_manager.display_config()
    
    # Rebuild pipeline with new configuration
    print("\n2. Rebuilding Pipeline with New Configuration...")
    pipeline = rag_system.build_pipeline("parallel")
    
    return pipeline


async def run_comprehensive_benchmark(rag_system):
    """Run a comprehensive benchmark with multiple configurations"""
    print("\n=== Running Comprehensive Benchmark ===")
    
    benchmark_questions = [
        "What is the primary objective of this research?",
        "Explain the key methodology in detail.",
        "What are the main findings and their implications?",
        "What limitations does the study acknowledge?",
        "What future directions are suggested?"
    ]
    
    configurations = [
        {'name': 'Vector Only', 'retriever_type': 'vector', 'top_k': 5},
        {'name': 'BM25 Only', 'retriever_type': 'bm25', 'top_k': 5},
        {'name': 'Hybrid', 'retriever_type': 'hybrid', 'top_k': 7}
    ]
    
    all_results = []
    
    for config in configurations:
        print(f"\n--- Testing {config['name']} Configuration ---")
        
        # Update retriever configuration
        rag_system.config.retriever.type = config['retriever_type']
        rag_system.config.retriever.top_k = config['top_k']
        
        # Rebuild pipeline
        rag_system.build_pipeline()
        
        # Run queries
        for question in benchmark_questions[:3]:  # Test with first 3 questions
            result = await rag_system.query(question, evaluate=True)
            result['configuration'] = config['name']
            all_results.append(result)
    
    return all_results


def main():
    """Main function demonstrating advanced RAG pipeline features"""
    
    # Setup
    print("=== RAG Pipeline Advanced Example ===\n")
    
    # Create project directories
    create_project_directories()
    
    # Verify API keys
    api_keys = verify_api_keys()
    if not api_keys.get("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY not set. Please set it in your .env file.")
        return
    
    # Initialize configuration with custom settings
    print("\n1. Initializing Custom Configuration...")
    config_manager = ConfigManager()
    
    # Initialize RAG system
    print("\n2. Initializing RAG System...")
    rag_system = ModularRAGSystem(config_manager)
    rag_system.initialize_components()
    
    # Set the path to your PDFs
    PDF_PATH = "./documents"
    
    # Check if documents exist
    if not Path(PDF_PATH).exists():
        print(f"\nWarning: No documents found at {PDF_PATH}")
        print("Please add PDF files to the documents directory.")
        return
    
    # Load and process documents
    print("\n3. Loading and Processing Documents...")
    chunks = rag_system.load_and_process_documents(PDF_PATH)
    
    # Create vector store
    print("\n4. Creating Vector Store...")
    vector_store = rag_system.create_or_load_vector_store(chunks, force_rebuild=False)
    
    # Test configuration updates
    print("\n5. Testing Configuration Updates...")
    test_configuration_updates(config_manager, rag_system)
    
    # Test iterative pipeline
    print("\n6. Testing Iterative Pipeline...")
    asyncio.run(test_iterative_pipeline(rag_system))
    
    # Test advanced evaluators
    print("\n7. Testing Advanced Evaluators...")
    asyncio.run(test_advanced_evaluators(rag_system))
    
    # Run comprehensive benchmark
    print("\n8. Running Comprehensive Benchmark...")
    benchmark_results = asyncio.run(run_comprehensive_benchmark(rag_system))
    
    # Analyze results
    print("\n9. Analyzing Results...")
    results_analyzer = ResultsAnalyzer(rag_system.eval_orchestrator)
    results_analyzer.create_performance_dashboard()
    results_analyzer.generate_report()
    
    # Export results
    print("\n10. Exporting Results...")
    results_analyzer.export_results("./results/advanced_pipeline_results.csv")
    
    print("\n=== Advanced Example Complete ===")


if __name__ == "__main__":
    main()