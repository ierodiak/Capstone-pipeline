# utils/statistical_analysis.py

"""
Statistical analysis module for rigorous RAG pipeline evaluation.
Suitable for MSc-level research with proper hypothesis testing.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    """
    Performs rigorous statistical analysis on RAG pipeline experiments.
    Includes hypothesis testing, effect sizes, and publication-ready outputs.
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.results = {}
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for statistical analysis by handling missing values
        and creating appropriate groupings.
        """
        # Create composite keys for full configuration
        df['full_config'] = df.apply(
            lambda row: f"{row['chunker']}_{row['retrieval']}_{row['embedding']}", 
            axis=1
        )
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        return df
    
    def check_assumptions(self, groups: List[pd.Series], test_name: str = ""):
        """
        Check statistical test assumptions: normality and homogeneity of variance.
        """
        results = {
            'test': test_name,
            'normality': {},
            'homogeneity': None
        }
        
        # Normality test (Shapiro-Wilk)
        for i, group in enumerate(groups):
            if len(group) >= 3:
                stat, p_value = stats.shapiro(group)
                results['normality'][f'group_{i}'] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'normal': p_value > self.alpha
                }
        
        # Homogeneity of variance (Levene's test)
        if len(groups) >= 2:
            stat, p_value = stats.levene(*groups)
            results['homogeneity'] = {
                'statistic': stat,
                'p_value': p_value,
                'equal_variance': p_value > self.alpha
            }
        
        return results
    
    def paired_comparison(self, df: pd.DataFrame, 
                         config1: str, config2: str, 
                         metric: str) -> Dict:
        """
        Perform paired comparison between two configurations.
        Uses paired t-test if same questions, independent t-test otherwise.
        """
        # Get data for each configuration
        data1 = df[df['full_config'] == config1][metric]
        data2 = df[df['full_config'] == config2][metric]
        
        # Check if we have paired data (same questions)
        questions1 = df[df['full_config'] == config1]['question'].values
        questions2 = df[df['full_config'] == config2]['question'].values
        
        is_paired = len(questions1) == len(questions2) and all(questions1 == questions2)
        
        # Check assumptions
        assumptions = self.check_assumptions([data1, data2], 
                                           "paired_t" if is_paired else "independent_t")
        
        if is_paired:
            # Paired t-test
            statistic, p_value = stats.ttest_rel(data1, data2)
            test_type = "paired_t_test"
        else:
            # Independent t-test (Welch's if unequal variance)
            equal_var = assumptions['homogeneity']['equal_variance'] if assumptions['homogeneity'] else False
            statistic, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
            test_type = "welch_t_test" if not equal_var else "independent_t_test"
        
        # Calculate effect size (Cohen's d)
        mean_diff = np.mean(data1) - np.mean(data2)
        pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
        
        # Confidence interval for mean difference
        se = pooled_std * np.sqrt(1/len(data1) + 1/len(data2))
        ci_low = mean_diff - 1.96 * se
        ci_high = mean_diff + 1.96 * se
        
        return {
            'test_type': test_type,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'mean_diff': mean_diff,
            'cohens_d': cohens_d,
            'effect_size': self._interpret_cohens_d(cohens_d),
            'confidence_interval': (ci_low, ci_high),
            'assumptions': assumptions,
            'sample_sizes': (len(data1), len(data2)),
            'means': (np.mean(data1), np.mean(data2)),
            'stds': (np.std(data1), np.std(data2))
        }
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def anova_analysis(self, df: pd.DataFrame, 
                      factor: str, 
                      metric: str,
                      include_tukey: bool = True) -> Dict:
        """
        Perform one-way ANOVA to compare multiple groups.
        Includes post-hoc Tukey HSD if significant.
        """
        # Prepare data
        groups = df.groupby(factor)[metric].apply(list)
        group_data = [group for group in groups if len(group) >= 3]
        
        if len(group_data) < 2:
            return {"error": "Not enough groups with sufficient data"}
        
        # Check assumptions
        assumptions = self.check_assumptions(group_data, "ANOVA")
        
        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*group_data)
        
        # Calculate eta squared (effect size)
        ss_between = sum(len(g) * (np.mean(g) - df[metric].mean())**2 for g in group_data)
        ss_total = sum((df[metric] - df[metric].mean())**2)
        eta_squared = ss_between / ss_total if ss_total != 0 else 0
        
        results = {
            'test': 'one_way_anova',
            'factor': factor,
            'metric': metric,
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'eta_squared': eta_squared,
            'effect_size': self._interpret_eta_squared(eta_squared),
            'assumptions': assumptions,
            'group_stats': {}
        }
        
        # Group statistics
        for name, group in groups.items():
            if len(group) >= 3:
                results['group_stats'][name] = {
                    'mean': np.mean(group),
                    'std': np.std(group),
                    'n': len(group)
                }
        
        # Post-hoc analysis if significant
        if p_value < self.alpha and include_tukey:
            # Prepare data for Tukey HSD
            data_for_tukey = []
            for name, group in groups.items():
                if len(group) >= 3:
                    for value in group:
                        data_for_tukey.append({'value': value, 'group': name})
            
            tukey_df = pd.DataFrame(data_for_tukey)
            tukey_result = pairwise_tukeyhsd(tukey_df['value'], tukey_df['group'], alpha=self.alpha)
            
            results['post_hoc'] = {
                'test': 'tukey_hsd',
                'summary': str(tukey_result),
                'significant_pairs': []
            }
            
            # Extract significant pairs
            for i in range(len(tukey_result.groupsunique)):
                for j in range(i+1, len(tukey_result.groupsunique)):
                    if tukey_result.reject[i * len(tukey_result.groupsunique) + j]:
                        results['post_hoc']['significant_pairs'].append(
                            (tukey_result.groupsunique[i], tukey_result.groupsunique[j])
                        )
        
        return results
    
    def _interpret_eta_squared(self, eta: float) -> str:
        """Interpret eta squared effect size."""
        if eta < 0.01:
            return "negligible"
        elif eta < 0.06:
            return "small"
        elif eta < 0.14:
            return "medium"
        else:
            return "large"
    
    def factorial_anova(self, df: pd.DataFrame, 
                       factors: List[str], 
                       metric: str) -> Dict:
        """
        Perform factorial ANOVA to analyze interactions between factors.
        """
        # Create formula for statsmodels
        formula = f"{metric} ~ " + " * ".join(factors)
        
        # Fit the model
        model = ols(formula, data=df).fit()
        anova_table = anova_lm(model, typ=2)
        
        # Calculate effect sizes
        ss_total = anova_table['sum_sq'].sum()
        anova_table['eta_squared'] = anova_table['sum_sq'] / ss_total
        anova_table['effect_size'] = anova_table['eta_squared'].apply(self._interpret_eta_squared)
        
        return {
            'test': 'factorial_anova',
            'factors': factors,
            'metric': metric,
            'anova_table': anova_table,
            'model_summary': model.summary(),
            'r_squared': model.rsquared,
            'significant_effects': anova_table[anova_table['PR(>F)'] < self.alpha].index.tolist()
        }
    
    def correlation_analysis(self, df: pd.DataFrame, 
                           metrics: List[str]) -> Dict:
        """
        Analyze correlations between different metrics.
        """
        # Calculate correlation matrix
        corr_matrix = df[metrics].corr(method='pearson')
        
        # Calculate p-values for correlations
        p_values = pd.DataFrame(np.zeros_like(corr_matrix), 
                              columns=corr_matrix.columns, 
                              index=corr_matrix.index)
        
        for i in range(len(metrics)):
            for j in range(len(metrics)):
                if i != j:
                    r, p = stats.pearsonr(df[metrics[i]].dropna(), 
                                        df[metrics[j]].dropna())
                    p_values.iloc[i, j] = p
        
        # Find significant correlations
        significant_corr = []
        for i in range(len(metrics)):
            for j in range(i+1, len(metrics)):
                if p_values.iloc[i, j] < self.alpha:
                    significant_corr.append({
                        'metric1': metrics[i],
                        'metric2': metrics[j],
                        'correlation': corr_matrix.iloc[i, j],
                        'p_value': p_values.iloc[i, j]
                    })
        
        return {
            'correlation_matrix': corr_matrix,
            'p_values': p_values,
            'significant_correlations': significant_corr
        }
    
    def sample_size_analysis(self, df: pd.DataFrame, 
                           metric: str, 
                           effect_size: float = 0.5,
                           power: float = 0.8) -> Dict:
        """
        Perform power analysis to determine required sample sizes.
        """
        from statsmodels.stats.power import TTestPower
        
        power_analysis = TTestPower()
        
        # Calculate required sample size
        required_n = power_analysis.solve_power(
            effect_size=effect_size, 
            power=power, 
            alpha=self.alpha
        )
        
        # Calculate current power
        current_n = len(df[metric].dropna())
        current_power = power_analysis.solve_power(
            effect_size=effect_size,
            nobs=current_n,
            alpha=self.alpha
        )
        
        return {
            'metric': metric,
            'current_sample_size': current_n,
            'current_power': current_power,
            'required_sample_size': int(np.ceil(required_n)),
            'effect_size': effect_size,
            'alpha': self.alpha,
            'target_power': power,
            'sufficient_power': current_power >= power
        }
    
    def generate_statistical_report(self, df: pd.DataFrame, 
                                  output_path: str = "statistical_report.html"):
        """
        Generate comprehensive statistical report suitable for MSc thesis.
        """
        from datetime import datetime
        
        html_content = f"""
        <html>
        <head>
            <title>RAG Pipeline Statistical Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .significant {{ color: red; font-weight: bold; }}
                .assumption-failed {{ background-color: #ffe6e6; }}
                .effect-large {{ color: green; font-weight: bold; }}
                .effect-medium {{ color: orange; }}
                .effect-small {{ color: #999; }}
            </style>
        </head>
        <body>
            <h1>RAG Pipeline Statistical Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Significance level (α): {self.alpha}</p>
            
            <h2>Dataset Overview</h2>
            <ul>
                <li>Total experiments: {len(df)}</li>
                <li>Unique configurations: {df['full_config'].nunique()}</li>
                <li>Questions tested: {df['question'].nunique()}</li>
            </ul>
        """
        
        # Add ANOVA results for each metric
        metrics = ['ragas_faithfulness', 'ragas_answer_relevancy', 'response_time']
        
        html_content += "<h2>ANOVA Results by Factor</h2>"
        
        for metric in metrics:
            if metric in df.columns:
                html_content += f"<h3>Metric: {metric}</h3>"
                
                # Test each factor
                for factor in ['chunker', 'retrieval', 'embedding']:
                    if factor in df.columns:
                        anova_result = self.anova_analysis(df, factor, metric)
                        
                        if 'error' not in anova_result:
                            sig_class = "significant" if anova_result['significant'] else ""
                            effect_class = f"effect-{anova_result['effect_size']}"
                            
                            html_content += f"""
                            <h4>Factor: {factor}</h4>
                            <table>
                                <tr><th>Statistic</th><th>Value</th></tr>
                                <tr><td>F-statistic</td><td>{anova_result['f_statistic']:.4f}</td></tr>
                                <tr><td>p-value</td><td class="{sig_class}">{anova_result['p_value']:.4f}</td></tr>
                                <tr><td>Effect size (η²)</td><td class="{effect_class}">{anova_result['eta_squared']:.4f} ({anova_result['effect_size']})</td></tr>
                            </table>
                            
                            <table>
                                <tr><th>Group</th><th>Mean</th><th>Std</th><th>N</th></tr>
                            """
                            
                            for group, stats in anova_result['group_stats'].items():
                                html_content += f"""
                                <tr>
                                    <td>{group}</td>
                                    <td>{stats['mean']:.4f}</td>
                                    <td>{stats['std']:.4f}</td>
                                    <td>{stats['n']}</td>
                                </tr>
                                """
                            
                            html_content += "</table>"
                            
                            # Add post-hoc results if available
                            if 'post_hoc' in anova_result and anova_result['post_hoc']['significant_pairs']:
                                html_content += "<p><strong>Significant pairwise differences (Tukey HSD):</strong></p><ul>"
                                for pair in anova_result['post_hoc']['significant_pairs']:
                                    html_content += f"<li>{pair[0]} vs {pair[1]}</li>"
                                html_content += "</ul>"
        
        # Add correlation analysis
        html_content += "<h2>Metric Correlations</h2>"
        corr_result = self.correlation_analysis(df, metrics)
        
        if corr_result['significant_correlations']:
            html_content += "<table><tr><th>Metric 1</th><th>Metric 2</th><th>Correlation</th><th>p-value</th></tr>"
            
            for corr in corr_result['significant_correlations']:
                html_content += f"""
                <tr>
                    <td>{corr['metric1']}</td>
                    <td>{corr['metric2']}</td>
                    <td>{corr['correlation']:.4f}</td>
                    <td>{corr['p_value']:.4f}</td>
                </tr>
                """
            html_content += "</table>"
        
        # Add sample size analysis
        html_content += "<h2>Statistical Power Analysis</h2>"
        html_content += "<table><tr><th>Metric</th><th>Current N</th><th>Current Power</th><th>Required N (80% power)</th></tr>"
        
        for metric in metrics:
            if metric in df.columns:
                power_result = self.sample_size_analysis(df, metric)
                power_class = "significant" if power_result['sufficient_power'] else "assumption-failed"
                
                html_content += f"""
                <tr>
                    <td>{metric}</td>
                    <td>{power_result['current_sample_size']}</td>
                    <td class="{power_class}">{power_result['current_power']:.3f}</td>
                    <td>{power_result['required_sample_size']}</td>
                </tr>
                """
        
        html_content += """
            </table>
            
            <h2>Recommendations</h2>
            <ul>
                <li>Ensure sufficient sample sizes for reliable conclusions</li>
                <li>Check assumption violations and consider non-parametric tests if needed</li>
                <li>Report effect sizes alongside p-values for practical significance</li>
                <li>Consider multiple comparison corrections for many tests</li>
            </ul>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Statistical report saved to: {output_path}")
        
        return output_path
    
    def export_for_spss(self, df: pd.DataFrame, output_path: str = "rag_data_for_spss.csv"):
        """
        Export data in a format suitable for SPSS analysis.
        """
        # Create wide format for repeated measures
        spss_df = df.pivot_table(
            index='question',
            columns='full_config',
            values=['ragas_faithfulness', 'ragas_answer_relevancy', 'response_time']
        )
        
        # Flatten column names
        spss_df.columns = ['_'.join(col).strip() for col in spss_df.columns.values]
        
        # Save with proper encoding
        spss_df.to_csv(output_path, encoding='utf-8', index=True)
        
        print(f"SPSS-ready data exported to: {output_path}")
        
        return spss_df

# Enhanced ExperimentRunner with statistical tracking
def enhance_experiment_runner():
    """
    Enhance the ExperimentRunner to include statistical considerations.
    """
    
    class StatisticalExperimentRunner(ExperimentRunner):
        """Enhanced runner with statistical experiment design."""
        
        def __init__(self, base_path: str = "./experiments", 
                    min_runs_per_config: int = 30):
            super().__init__(base_path)
            self.min_runs_per_config = min_runs_per_config
            self.statistical_analyzer = StatisticalAnalyzer()
            
        async def run_statistical_experiment(self, 
                                           configs: List[ExperimentConfig],
                                           test_questions: List[Dict[str, str]],
                                           document_path: str,
                                           runs_per_config: int = None) -> pd.DataFrame:
            """
            Run experiments with proper statistical design.
            """
            if runs_per_config is None:
                runs_per_config = self.min_runs_per_config
            
            print(f"Running statistical experiment design:")
            print(f"- Configurations: {len(configs)}")
            print(f"- Questions per run: {len(test_questions)}")
            print(f"- Runs per configuration: {runs_per_config}")
            print(f"- Total experiments: {len(configs) * len(test_questions) * runs_per_config}")
            
            all_results = []
            
            for config_idx, config in enumerate(configs):
                print(f"\nConfiguration {config_idx + 1}/{len(configs)}")
                
                for run in range(runs_per_config):
                    print(f"  Run {run + 1}/{runs_per_config}")
                    
                    # Add run number to experiment ID
                    config.experiment_id = f"{config.experiment_id}_run{run}"
                    
                    # Randomize question order to avoid order effects
                    import random
                    shuffled_questions = test_questions.copy()
                    random.shuffle(shuffled_questions)
                    
                    try:
                        results = await self.run_experiment(
                            config, shuffled_questions, document_path
                        )
                        results['run_number'] = run
                        all_results.append(results)
                    except Exception as e:
                        print(f"    Error: {e}")
                        continue
            
            # Convert to DataFrame with statistical considerations
            df = self._results_to_statistical_dataframe(all_results)
            
            # Generate statistical report
            self.statistical_analyzer.generate_statistical_report(
                df, 
                self.base_path / "statistical_analysis_report.html"
            )
            
            return df
        
        def _results_to_statistical_dataframe(self, all_results: List[Dict[str, Any]]) -> pd.DataFrame:
            """Enhanced DataFrame creation with statistical metadata."""
            df = super()._results_to_dataframe(all_results)
            
            # Add statistical metadata
            df['experiment_date'] = pd.to_datetime(df['timestamp']).dt.date
            df['experiment_time'] = pd.to_datetime(df['timestamp']).dt.time
            
            # Check for batch effects
            df['batch'] = pd.cut(pd.to_datetime(df['timestamp']), bins=5, labels=False)
            
            return df
    
    return StatisticalExperimentRunner