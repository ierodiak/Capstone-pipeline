# system/experiment_runner.py

"""
System for running experiments with different configurations and tracking results.
"""
import asyncio
import time
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime
from itertools import product

from core.experiment_config import ExperimentConfig
from system.rag_system import ModularRAGSystem
from core.config import ConfigManager

class ExperimentRunner:
    """Runs experiments with different configurations and tracks results"""
    
    def __init__(self, base_path: str = "./experiments"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.results = []
        
    def create_config_variants(self, base_config: ExperimentConfig, 
                             variants: Dict[str, List[Any]]) -> List[ExperimentConfig]:
        """Create all combinations of configuration variants"""
        configs = []
        
        # Get all combinations
        variant_keys = list(variants.keys())
        variant_values = [variants[k] for k in variant_keys]
        
        for combination in product(*variant_values):
            # Create new config
            config = ExperimentConfig(
                experiment_id=f"{base_config.experiment_id}_variant_{len(configs)}",
                experiment_name=base_config.experiment_name,
                tags=base_config.tags.copy()
            )
            
            # Apply variant values
            for key, value in zip(variant_keys, combination):
                # Parse nested keys (e.g., "chunker.method")
                parts = key.split('.')
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
                
            configs.append(config)
            
        return configs
    
    async def run_experiment(self, config: ExperimentConfig, 
                           test_questions: List[Dict[str, str]],
                           document_path: str) -> Dict[str, Any]:
        """Run a single experiment with given configuration"""
        
        # Create config manager from experiment config
        config_manager = self._create_config_manager(config)
        
        # Initialize RAG system
        rag_system = ModularRAGSystem(config_manager)
        rag_system.initialize_components()
        
        # Load and process documents
        chunks = rag_system.load_and_process_documents(document_path)
        
        # Create vector store
        vector_store = rag_system.create_or_load_vector_store(chunks, force_rebuild=True)
        
        # Build pipeline
        pipeline = rag_system.build_pipeline()
        
        # Run test questions
        experiment_results = {
            "config": config,
            "variant_id": config.get_variant_id(),
            "variant_description": config.get_variant_description(),
            "timestamp": datetime.now().isoformat(),
            "results": []
        }
        
        for test_item in test_questions:
            question = test_item["question"]
            reference = test_item.get("reference")
            
            # Time the query
            start_time = time.time()
            
            # Run query
            result = await rag_system.query(
                question=question,
                evaluate=True,
                reference=reference
            )
            
            # Add timing
            result["total_time"] = time.time() - start_time
            
            # Store result
            experiment_results["results"].append(result)
            
        return experiment_results
    
    async def run_grid_search(self, base_config: ExperimentConfig,
                            variants: Dict[str, List[Any]],
                            test_questions: List[Dict[str, str]],
                            document_path: str) -> pd.DataFrame:
        """Run grid search over configuration variants"""
        
        print(f"Starting grid search with {len(variants)} variant dimensions")
        
        # Create all config variants
        configs = self.create_config_variants(base_config, variants)
        print(f"Created {len(configs)} configuration variants")
        
        # Run experiments
        all_results = []
        
        for i, config in enumerate(configs):
            print(f"\nRunning variant {i+1}/{len(configs)}")
            print(f"Variant: {config.get_variant_description()}")
            
            try:
                experiment_results = await self.run_experiment(
                    config, test_questions, document_path
                )
                all_results.append(experiment_results)
                
                # Save intermediate results
                self._save_experiment_results(experiment_results)
                
            except Exception as e:
                print(f"Error in variant {i+1}: {str(e)}")
                continue
        
        # Convert to DataFrame
        df = self._results_to_dataframe(all_results)
        
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(self.base_path / f"grid_search_results_{timestamp}.csv", index=False)
        
        return df
    
    def _create_config_manager(self, experiment_config: ExperimentConfig):
        """Convert ExperimentConfig to ConfigManager format"""
        from core.config import ConfigManager, RAGConfig, RetrieverConfig, GeneratorConfig, EvaluationConfig
        
        config_manager = ConfigManager()
        
        # Update retriever config
        config_manager.config.retriever = RetrieverConfig(
            type=experiment_config.retrieval.strategy,
            model=experiment_config.embedding.model,
            chunk_size=experiment_config.chunker.chunk_size,
            chunk_overlap=experiment_config.chunker.chunk_overlap,
            top_k=experiment_config.retrieval.top_k
        )
        
        # Update generator config
        config_manager.config.generator = GeneratorConfig(
            provider=experiment_config.generation.provider,
            model=experiment_config.generation.model,
            temperature=experiment_config.generation.temperature,
            max_tokens=experiment_config.generation.max_tokens
        )
        
        # Update evaluation config
        config_manager.config.evaluation = EvaluationConfig(
            metrics=experiment_config.evaluation_metrics
        )
        
        # Update storage
        config_manager.config.vector_store_type = experiment_config.storage.type
        
        return config_manager
    
    def _save_experiment_results(self, results: Dict[str, Any]):
        """Save experiment results to file"""
        variant_id = results["variant_id"]
        file_path = self.base_path / f"experiment_{variant_id}.json"
        
        # Convert config to dict for JSON serialization
        results_copy = results.copy()
        results_copy["config"] = asdict(results["config"])
        
        with open(file_path, 'w') as f:
            json.dump(results_copy, f, indent=2)
    
    def _results_to_dataframe(self, all_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert experiment results to DataFrame for analysis"""
        rows = []
        
        for experiment in all_results:
            variant_desc = experiment["variant_description"]
            
            for result in experiment["results"]:
                row = {
                    "variant_id": experiment["variant_id"],
                    "timestamp": experiment["timestamp"],
                    **variant_desc,  # Unpack variant description
                    "question": result["question"],
                    "answer_preview": result["answer"][:100] + "...",
                    "response_time": result["response_time"],
                    "num_contexts": result["num_contexts"]
                }
                
                # Add evaluation metrics
                if "evaluation" in result:
                    if "ragas" in result["evaluation"]:
                        for metric, value in result["evaluation"]["ragas"].items():
                            if isinstance(value, (int, float)):
                                row[f"ragas_{metric}"] = value
                                
                    if "custom" in result["evaluation"]:
                        for metric, value in result["evaluation"]["custom"].items():
                            if isinstance(value, (int, float)):
                                row[f"custom_{metric}"] = value
                
                rows.append(row)
        
        return pd.DataFrame(rows)