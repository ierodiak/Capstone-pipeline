"""
Results analysis and visualization module for RAG pipeline evaluation.
"""
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from IPython.display import display, Markdown

class ResultsAnalyzer:
    """Analyze and visualize RAG system results"""
    
    def __init__(self, results_df: pd.DataFrame):
        self.df = results_df
        
    def get_best_configuration(self, metric: str = 'aggregate_score') -> dict:
        """Get the best performing configuration."""
        config_scores = self.df.groupby('config_description')[metric].agg(['mean', 'std', 'count'])
        best_config = config_scores['mean'].idxmax()
        best_stats = config_scores.loc[best_config]
        
        # Get configuration details
        best_row = self.df[self.df['config_description'] == best_config].iloc[0]
        
        return {
            'configuration': best_config,
            'mean_score': best_stats['mean'],
            'std': best_stats['std'],
            'count': best_stats['count'],
            'details': {
                'chunker_method': best_row.get('chunker_method', 'N/A'),
                'chunk_size': best_row.get('chunk_size', 'N/A'),
                'retriever_strategy': best_row.get('retriever_strategy', 'N/A'),
                'top_k': best_row.get('retriever_top_k', 'N/A'),
                'generator_model': best_row.get('generator_model', 'N/A')
            }
        }
    
    def compare_configurations(self) -> pd.DataFrame:
        """Compare all configurations."""
        metrics = ['aggregate_score', 'ragas_faithfulness', 'ragas_answer_relevancy', 
                  'cofe_pipeline_score', 'omni_weighted_score', 'response_time']
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in self.df.columns]
        
        return self.df.groupby('config_description')[available_metrics].agg(['mean', 'std', 'count']).round(3)
    
    def analyze_by_component(self) -> dict:
        """Analyze impact of each component."""
        components = ['chunker_method', 'chunk_size', 'retriever_strategy', 
                     'retriever_top_k', 'generator_model']
        
        analyses = {}
        for component in components:
            if component in self.df.columns:
                analyses[component] = self.df.groupby(component)['aggregate_score'].agg(['mean', 'std', 'count']).round(3)
                
        return analyses
    
    def create_performance_dashboard(self):
        """Create a performance comparison dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Configuration comparison
        ax1 = axes[0, 0]
        config_scores = self.df.groupby('config_description')['aggregate_score'].mean().sort_values(ascending=False).head(10)
        config_scores.plot(kind='barh', ax=ax1)
        ax1.set_title('Top 10 Configurations by Score')
        ax1.set_xlabel('Aggregate Score')
        
        # 2. Framework scores comparison
        ax2 = axes[0, 1]
        framework_cols = {
            'RAGAS': ['ragas_faithfulness', 'ragas_answer_relevancy'],
            'CoFE-RAG': ['cofe_pipeline_score'],
            'OmniEval': ['omni_weighted_score']
        }
        
        framework_means = {}
        for name, cols in framework_cols.items():
            available = [c for c in cols if c in self.df.columns]
            if available:
                framework_means[name] = self.df[available].mean().mean()
        
        if framework_means:
            pd.Series(framework_means).plot(kind='bar', ax=ax2)
            ax2.set_title('Average Scores by Framework')
            ax2.set_ylabel('Score')
            ax2.set_ylim(0, 1)
        
        # 3. Response time vs quality
        ax3 = axes[1, 0]
        ax3.scatter(self.df['response_time'], self.df['aggregate_score'], alpha=0.5)
        ax3.set_xlabel('Response Time (s)')
        ax3.set_ylabel('Aggregate Score')
        ax3.set_title('Response Time vs Quality Trade-off')
        
        # 4. Component impact
        ax4 = axes[1, 1]
        component_impact = self.analyze_by_component()
        if component_impact:
            first_component = list(component_impact.keys())[0]
            component_impact[first_component]['mean'].plot(kind='bar', ax=ax4)
            ax4.set_title(f'Impact of {first_component}')
            ax4.set_ylabel('Mean Score')
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, output_path: str):
        """Export results to Excel with analysis."""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Raw data
            self.df.to_excel(writer, sheet_name='Raw Results', index=False)
            
            # Configuration comparison
            self.compare_configurations().to_excel(writer, sheet_name='Config Comparison')
            
            # Component analysis
            for component, analysis in self.analyze_by_component().items():
                analysis.to_excel(writer, sheet_name=f'{component}_Analysis')