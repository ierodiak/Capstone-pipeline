"""
Experiment runner with A/B testing capabilities.
"""
import asyncio
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from itertools import product
from tqdm import tqdm

from core.config import RAGConfig, ConfigManager
from system.rag_system import ModularRAGSystem

class ExperimentRunner:
    """Run A/B testing experiments with different configurations."""
    
    def __init__(self, base_path: str = "./experiments"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
    def create_config_variants(self, base_config: RAGConfig,
                             variants: Dict[str, List[Any]]) -> List[RAGConfig]:
        """Create all combinations of configuration variants."""
        configs = []
        
        # Get all combinations
        variant_keys = list(variants.keys())
        variant_values = [variants[k] for k in variant_keys]
        
        for i, combination in enumerate(product(*variant_values)):
            # Create new config with unique ID
            config = RAGConfig(
                experiment_id=f"{base_config.experiment_id}_v{i}",
                experiment_name=base_config.experiment_name,
                tags=base_config.tags.copy(),
                loader=base_config.loader,
                chunker=base_config.chunker,
                embedding=base_config.embedding,
                storage=base_config.storage,
                retrieval=base_config.retrieval,
                generation=base_config.generation,
                metrics=base_config.metrics,
                pipeline_type=base_config.pipeline_type
            )
            
            # Apply variant values
            for key, value in zip(variant_keys, combination):
                parts = key.split('.')
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            
            configs.append(config)
        
        return configs
    
    async def run_ab_test(self, base_config: RAGConfig,
                        variants: Dict[str, List[Any]],
                        test_questions: List[Dict[str, str]],
                        document_path: str) -> pd.DataFrame:
        """Run A/B test with multiple configurations."""
        # Create variants
        configs = self.create_config_variants(base_config, variants)
        print(f"Running A/B test with {len(configs)} configurations")
        
        all_results = []
        
        for i, config in enumerate(configs):
            print(f"\n{'='*60}")
            print(f"Configuration {i+1}/{len(configs)}")
            print(f"Variant: {config.get_variant_description()}")
            print('='*60)
            
            try:
                # Run experiment
                results = await self._run_single_config(config, test_questions, document_path)
                all_results.extend(results)
            except Exception as e:
                print(f"Error in configuration {i+1}: {str(e)}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.base_path / f"ab_test_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Save Excel analysis
        excel_path = self.base_path / f"ab_test_analysis_{timestamp}.xlsx"
        self._save_excel_analysis(df, excel_path)
        print(f"Analysis saved to: {excel_path}")
        
        return df
    
    async def _run_single_config(self, config: RAGConfig, 
                               test_questions: List[Dict[str, str]],
                               document_path: str) -> List[Dict]:
        """Run experiment with single configuration."""
        # Initialize system
        config_manager = ConfigManager(config)
        rag_system = ModularRAGSystem(config_manager)
        rag_system.initialize_components()
        
        # Load documents
        chunks = rag_system.load_and_process_documents(document_path)
        
        # Create vector store
        rag_system.create_or_load_vector_store(chunks, force_rebuild=True)
        
        # Build pipeline
        rag_system.build_pipeline()
        
        # Run queries
        results = []
        for q_data in tqdm(test_questions, desc="Evaluating"):
            try:
                result = await rag_system.query(
                    question=q_data["question"],
                    evaluate=True,
                    reference=q_data.get("reference")
                )
                
                # Create record
                record = {
                    'config_id': config.get_variant_id(),
                    'config_description': config.get_variant_description(),
                    'question_id': q_data.get('id', ''),
                    'question': q_data['question'],
                    'answer': result['answer'],
                    'reference': q_data.get('reference', ''),
                    'category': q_data.get('category', 'general'),
                    'timestamp': datetime.now().isoformat(),
                    # Add metrics
                    **result.get('evaluation', {}),
                    # Add config details
                    'chunker_method': config.chunker.method,
                    'chunk_size': config.chunker.chunk_size,
                    'embedding_model': config.embedding.model,
                    'retriever_strategy': config.retrieval.strategy,
                    'retriever_top_k': config.retrieval.top_k,
                    'generator_model': config.generation.model
                }
                results.append(record)
                
            except Exception as e:
                print(f"Error processing question: {e}")
                
        return results
    
    def _save_excel_analysis(self, df: pd.DataFrame, path: Path):
        """Save Excel file with analysis sheets."""
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            # Raw results
            df.to_excel(writer, sheet_name='Raw Results', index=False)
            
            # Config summary
            config_cols = ['aggregate_score', 'ragas_faithfulness', 'ragas_answer_relevancy',
                          'cofe_pipeline_score', 'omni_weighted_score', 'response_time']
            available_cols = [c for c in config_cols if c in df.columns]
            
            config_summary = df.groupby('config_description')[available_cols].agg(['mean', 'std', 'count'])
            config_summary.to_excel(writer, sheet_name='Config Summary')
            
            # Best configs
            best_configs = df.groupby('config_description')['aggregate_score'].mean().sort_values(ascending=False).head(10)
            best_configs.to_excel(writer, sheet_name='Top 10 Configs')