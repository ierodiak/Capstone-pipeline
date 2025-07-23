# utils/experiment_analysis.py

"""
Utilities for analyzing experiment results.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional

class ExperimentAnalyzer:
    """Analyze and visualize experiment results"""

    def __init__(self, results_path: str = "./experiments"):
        self.results_path = Path(results_path)
        
    def load_results(self, file_pattern: str = "grid_search_results_*.csv") -> pd.DataFrame:
        """Load experiment results from CSV files"""
        files = list(self.results_path.glob(file_pattern))
        
        if not files:
            print(f"No files found matching {file_pattern}")
            return pd.DataFrame()
        
        # Load most recent file 
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        print(f"Loading results from: {latest_file}")
        
        return pd.read_csv(latest_file)
    
    def compare_variants(self, df: pd.DataFrame, metric: str, 
                        groupby: str, top_n: int = 10):
        """Compare variants by a specific metric"""
        
        # Group and calculate mean
        grouped = df.groupby(groupby)[metric].agg(['mean', 'std', 'count'])
        grouped = grouped.sort_values('mean', ascending=False).head(top_n)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bar plot with error bars
        x = range(len(grouped))
        ax.bar(x, grouped['mean'], yerr=grouped['std'], capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index, rotation=45, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(f'Top {top_n} {groupby} by {metric}')
        
        # Add sample counts
        for i, (idx, row) in enumerate(grouped.iterrows()):
            ax.text(i, row['mean'] + row['std'], f"n={row['count']}", 
                   ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        return grouped
    
    def create_heatmap(self, df: pd.DataFrame, metrics: List[str]):
        """Create heatmap of variant performance across metrics"""
        
        # Create pivot table
        pivot_data = {}
        
        for metric in metrics:
            if metric in df.columns:
                # Group by variant_id and calculate mean
                grouped = df.groupby('variant_id')[metric].mean()
                pivot_data[metric] = grouped
        
        pivot_df = pd.DataFrame(pivot_data)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df.T, annot=True, fmt='.3f', cmap='RdYlGn')
        plt.title('Variant Performance Heatmap')
        plt.xlabel('Variant ID')
        plt.ylabel('Metric')
        plt.tight_layout()
        plt.show()
        
        return pivot_df
    
    def find_best_configuration(self, df: pd.DataFrame, 
                               metrics: List[str], 
                               weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """Find best configuration based on weighted metrics"""
        
        if weights is None:
            weights = {metric: 1.0 for metric in metrics}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate weighted score for each variant
        scores = {}
        
        for variant_id in df['variant_id'].unique():
            variant_data = df[df['variant_id'] == variant_id]
            score = 0
            
            for metric in metrics:
                if metric in variant_data.columns:
                    # Normalize metric to 0-1 range
                    metric_values = df[metric].dropna()
                    min_val = metric_values.min()
                    max_val = metric_values.max()
                    
                    if max_val > min_val:
                        normalized_value = (variant_data[metric].mean() - min_val) / (max_val - min_val)
                    else:
                        normalized_value = 1.0
                    
                    score += weights.get(metric, 0) * normalized_value
            
            scores[variant_id] = score
        
        # Find best variant
        best_variant = max(scores, key=scores.get)
        best_score = scores[best_variant]
        
        # Get configuration details
        best_config = df[df['variant_id'] == best_variant].iloc[0]
        
        print(f"Best configuration: {best_variant}")
        print(f"Score: {best_score:.3f}")
        print("\nConfiguration details:")
        for col in ['loader', 'chunker', 'embedding', 'storage', 'retrieval', 'generation']:
            if col in best_config:
                print(f"  {col}: {best_config[col]}")
        
        return best_config