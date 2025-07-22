"""
Evaluation orchestrator for managing multiple evaluation frameworks.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown


class EvaluationOrchestrator:
    """Orchestrates evaluation across multiple frameworks"""
    
    def __init__(self, evaluators: Dict[str, Any]):
        self.evaluators = evaluators
        self.results_history = []
        
    async def evaluate_comprehensive(self, 
                                   question: str, 
                                   answer: str, 
                                   contexts: List[str],
                                   response_time: float,
                                   reference: Optional[str] = None,
                                   keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run comprehensive evaluation across all frameworks"""
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer[:100] + "..." if len(answer) > 100 else answer,
            "num_contexts": len(contexts),
            "response_time": response_time
        }
        
        # RAGAS evaluation
        if "ragas" in self.evaluators and self.evaluators["ragas"]:
            ragas_results = await self.evaluators["ragas"].evaluate_single(
                question=question,
                answer=answer,
                contexts=contexts,
                reference=reference
            )
            results["ragas"] = ragas_results
        
        # Custom evaluation
        if "custom" in self.evaluators:
            custom_results = self.evaluators["custom"].evaluate(
                answer=answer,
                contexts=contexts,
                response_time=response_time,
                keywords=keywords
            )
            results["custom"] = custom_results
        
        # Store results
        self.results_history.append(results)
        
        return results
    
    def get_results_summary(self) -> pd.DataFrame:
        """Get summary of all evaluation results"""
        if not self.results_history:
            return pd.DataFrame()
        
        # Flatten results for DataFrame
        flattened_results = []
        for result in self.results_history:
            flat_result = {
                "timestamp": result["timestamp"],
                "question": result["question"],
                "response_time": result["response_time"]
            }
            
            # Add RAGAS metrics
            if "ragas" in result:
                for metric, value in result["ragas"].items():
                    flat_result[f"ragas_{metric}"] = value
            
            # Add custom metrics
            if "custom" in result:
                for metric, value in result["custom"].items():
                    flat_result[f"custom_{metric}"] = value
            
            flattened_results.append(flat_result)
        
        return pd.DataFrame(flattened_results)
    
    def plot_evaluation_results(self):
        """Visualize evaluation results"""
        df = self.get_results_summary()
        
        if df.empty:
            print("No evaluation results to plot")
            return
        
        # Get metric columns
        metric_cols = [col for col in df.columns if col.startswith(('ragas_', 'custom_'))]
        
        if not metric_cols:
            print("No metrics found in results")
            return
        
        # Create subplots
        n_metrics = len(metric_cols)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for i, metric in enumerate(metric_cols):
            if i < len(axes):
                ax = axes[i]
                
                # Filter out non-numeric values
                numeric_values = pd.to_numeric(df[metric], errors='coerce')
                valid_values = numeric_values.dropna()
                
                if len(valid_values) > 0:
                    valid_values.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
                    ax.set_title(metric.replace('_', ' ').title())
                    ax.set_ylabel('Score')
                    ax.set_xlabel('Sample')
                    ax.set_ylim(0, 1.1)
                    
                    # Add average line
                    avg = valid_values.mean()
                    ax.axhline(y=avg, color='red', linestyle='--', alpha=0.7, 
                              label=f'Avg: {avg:.3f}')
                    ax.legend()
        
        # Remove empty subplots
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()
        
        # Display summary statistics
        display(Markdown("### Evaluation Summary Statistics"))
        numeric_cols = [col for col in metric_cols if pd.to_numeric(df[col], errors='coerce').notna().any()]
        if numeric_cols:
            summary = df[numeric_cols].apply(pd.to_numeric, errors='coerce').describe()
            display(summary)