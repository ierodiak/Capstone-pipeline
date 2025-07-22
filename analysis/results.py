"""
Results analysis and visualization module for RAG pipeline evaluation.
"""
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from IPython.display import display, Markdown

from evaluation.orchestrator import EvaluationOrchestrator


class ResultsAnalyzer:
    """Analyze and visualize RAG system results"""
    
    def __init__(self, eval_orchestrator: EvaluationOrchestrator):
        self.eval_orchestrator = eval_orchestrator
        
    def create_performance_dashboard(self):
        """Create a comprehensive performance dashboard"""
        df = self.eval_orchestrator.get_results_summary()
        
        if df.empty:
            print("No results to analyze")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Response Time Distribution
        ax1 = plt.subplot(2, 3, 1)
        df['response_time'].hist(bins=10, ax=ax1, edgecolor='black')
        ax1.set_title('Response Time Distribution')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Frequency')
        
        # 2. Average Metrics by Framework
        ax2 = plt.subplot(2, 3, 2)
        metric_cols = [col for col in df.columns if col.startswith(('ragas_', 'custom_'))]
        
        if metric_cols:
            avg_metrics = df[metric_cols].apply(pd.to_numeric, errors='coerce').mean()
            
            # Separate by framework
            ragas_metrics = {k.replace('ragas_', ''): v for k, v in avg_metrics.items() if k.startswith('ragas_')}
            custom_metrics = {k.replace('custom_', ''): v for k, v in avg_metrics.items() if k.startswith('custom_')}
            
            # Plot grouped bar chart
            x = list(range(len(ragas_metrics) + len(custom_metrics)))
            labels = list(ragas_metrics.keys()) + list(custom_metrics.keys())
            values = list(ragas_metrics.values()) + list(custom_metrics.values())
            colors = ['skyblue'] * len(ragas_metrics) + ['lightcoral'] * len(custom_metrics)
            
            bars = ax2.bar(x, values, color=colors, edgecolor='black')
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, rotation=45, ha='right')
            ax2.set_title('Average Metrics by Framework')
            ax2.set_ylabel('Score')
            ax2.set_ylim(0, 1.1)
            
            # Add legend
            legend_elements = [
                Patch(facecolor='skyblue', label='RAGAS'),
                Patch(facecolor='lightcoral', label='Custom')
            ]
            ax2.legend(handles=legend_elements)
        
        # 3. Metrics Over Time
        ax3 = plt.subplot(2, 3, 3)
        if 'timestamp' in df.columns and len(df) > 1:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Plot key metrics over time
            for col in ['ragas_faithfulness', 'custom_context_coverage']:
                if col in df.columns:
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    ax3.plot(range(len(df)), numeric_col, marker='o', label=col.replace('_', ' ').title())
            
            ax3.set_xlabel('Query Number')
            ax3.set_ylabel('Score')
            ax3.set_title('Metrics Over Time')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Correlation Heatmap
        ax4 = plt.subplot(2, 3, 4)
        numeric_df = df[metric_cols].apply(pd.to_numeric, errors='coerce')
        
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            
            im = ax4.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax4.set_xticks(range(len(corr_matrix.columns)))
            ax4.set_yticks(range(len(corr_matrix.columns)))
            ax4.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            ax4.set_yticklabels(corr_matrix.columns)
            ax4.set_title('Metric Correlations')
            
            # Add colorbar
            plt.colorbar(im, ax=ax4)
            
            # Add correlation values
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    text = ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=8)
        
        # 5. Performance Summary Table
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('tight')
        ax5.axis('off')
        
        # Create summary statistics
        summary_data = []
        summary_data.append(['Total Queries', len(df)])
        summary_data.append(['Avg Response Time', f'{df["response_time"].mean():.2f}s'])
        
        for col in metric_cols[:5]:  # Show top 5 metrics
            if col in numeric_df.columns:
                mean_val = numeric_df[col].mean()
                if not pd.isna(mean_val):
                    summary_data.append([col.replace('_', ' ').title(), f'{mean_val:.3f}'])
        
        table = ax5.table(cellText=summary_data, cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax5.set_title('Performance Summary', pad=20)
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, output_path: str = "./results/rag_evaluation_results.csv"):
        """Export results to CSV"""
        df = self.eval_orchestrator.get_results_summary()
        
        if not df.empty:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            print(f"Results exported to: {output_path}")
            
            # Also save as Excel with formatting
            excel_path = output_path.replace('.csv', '.xlsx')
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Results', index=False)
                
                # Add summary sheet
                summary_df = df.describe()
                summary_df.to_excel(writer, sheet_name='Summary')
            
            print(f"Excel report saved to: {excel_path}")
        else:
            print("No results to export")
    
    def generate_report(self):
        """Generate a comprehensive evaluation report"""
        df = self.eval_orchestrator.get_results_summary()
        
        if df.empty:
            print("No results to report")
            return
        
        display(Markdown("# RAG System Evaluation Report"))
        display(Markdown(f"**Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
        
        # Executive Summary
        display(Markdown("## Executive Summary"))
        
        total_queries = len(df)
        avg_response_time = df['response_time'].mean()
        
        display(Markdown(f"""
- **Total Queries Evaluated**: {total_queries}
- **Average Response Time**: {avg_response_time:.2f} seconds
- **Evaluation Frameworks Used**: {', '.join(self.eval_orchestrator.evaluators.keys())}
        """))
        
        # Performance Metrics
        display(Markdown("## Performance Metrics"))
        
        metric_cols = [col for col in df.columns if col.startswith(('ragas_', 'custom_'))]
        if metric_cols:
            metrics_df = df[metric_cols].apply(pd.to_numeric, errors='coerce').describe()
            display(metrics_df)
        
        # Recommendations
        display(Markdown("## Recommendations"))
        
        recommendations = []
        
        # Response time analysis
        if avg_response_time > 3:
            recommendations.append("- Consider optimizing retrieval or using a faster model to reduce response time")
        
        # Metric-specific recommendations
        for col in metric_cols:
            if col in df.columns:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                mean_val = numeric_col.mean()
                
                if not pd.isna(mean_val):
                    if 'faithfulness' in col and mean_val < 0.8:
                        recommendations.append("- Improve context quality or prompt engineering to increase faithfulness")
                    elif 'relevancy' in col and mean_val < 0.7:
                        recommendations.append("- Enhance retrieval strategy to improve answer relevancy")
                    elif 'coverage' in col and mean_val < 0.6:
                        recommendations.append("- Consider increasing context size or improving chunking strategy")
        
        if recommendations:
            for rec in recommendations:
                display(Markdown(rec))
        else:
            display(Markdown("- System is performing well across all metrics"))