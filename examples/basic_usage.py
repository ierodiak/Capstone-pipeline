"""
Basic usage example for the modular RAG pipeline.
"""
import asyncio
from pathlib import Path

# Add parent directory to path to import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.config import ConfigManager
from system.rag_system import ModularRAGSystem
from analysis.results import ResultsAnalyzer
from utils.helpers import create_project_directories, verify_api_keys


def main():
    """Main function demonstrating basic RAG pipeline usage"""
    
    # Setup
    print("=== RAG Pipeline Basic Usage Example ===\n")
    
    # Create project directories
    create_project_directories()
    
    # Verify API keys
    api_keys = verify_api_keys()
    if not api_keys.get("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY not set. Please set it in your .env file.")
        return
    
    # Initialize configuration
    print("\n1. Initializing Configuration...")
    config_manager = ConfigManager()
    config_manager.display_config()
    
    # Initialize RAG system
    print("\n2. Initializing RAG System...")
    rag_system = ModularRAGSystem(config_manager)
    rag_system.initialize_components()
    
    # Set the path to your PDFs
    PDF_PATH = "./documents"  # Change this to your PDF directory or file
    
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
    vector_store = rag_system.create_or_load_vector_store(chunks, force_rebuild=True)
    
    # Test retrieval
    print("\n5. Testing Retrieval...")
    rag_system.vs_manager.test_retrieval("What is the main topic of the document?")
    
    # Build pipeline
    print("\n6. Building RAG Pipeline...")
    pipeline = rag_system.build_pipeline("linear")
    
    # Test queries
    print("\n7. Running Test Queries...")
    test_questions = [
        "What are the main findings discussed in the document?",
        "Can you summarize the methodology used?",
        "What recommendations are provided?"
    ]
    
    # Run queries
    for i, question in enumerate(test_questions[:2]):  # Test with first 2 questions
        print(f"\n--- Query {i+1} ---")
        result = asyncio.run(rag_system.query(question, evaluate=True))
        print("\n" + "="*50)
    
    # Analyze results
    print("\n8. Analyzing Results...")
    results_analyzer = ResultsAnalyzer(rag_system.eval_orchestrator)
    results_analyzer.create_performance_dashboard()
    results_analyzer.generate_report()
    
    # Export results
    print("\n9. Exporting Results...")
    results_analyzer.export_results()
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()