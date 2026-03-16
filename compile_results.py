"""
Results compilation and comparison script.
Generates tables and plots comparing different architectures.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from datetime import datetime


class ResultsCompiler:
    """Compile results from multiple model training runs."""

    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        self.results = {}

    def add_result(self, model_name, metrics):
        """
        Add model results.

        Args:
            model_name: Name of the model (e.g., 'CNN', 'ViT', 'ViT+Temporal', 'ViT+Temporal+Distill', 'Mamba')
            metrics: Dict with keys: auc, params, inference_time, final_loss, etc.
        """
        self.results[model_name] = metrics

    def load_from_file(self, filepath):
        """Load results from CSV file."""
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            metrics = row.to_dict()
            model_name = metrics.pop('Model')
            self.results[model_name] = metrics

    def create_comparison_table(self, output_path=None):
        """
        Create comparison table of all models.

        Returns:
            DataFrame with comparison results
        """
        if not self.results:
            print("No results to compare")
            return None

        rows = []
        for model_name, metrics in self.results.items():
            row = {'Model': model_name}
            row.update(metrics)
            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by AUC descending
        if 'AUC-ROC' in df.columns:
            df = df.sort_values('AUC-ROC', ascending=False)

        # Save to CSV
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Saved comparison table to {output_path}")

        return df

    def plot_comparison(self, metrics_to_plot=['AUC-ROC', 'Params', 'Inference (ms/frame)'],
                       output_path=None):
        """
        Create comparison plots.

        Args:
            metrics_to_plot: List of metrics to visualize
            output_path: Path to save combined plot
        """
        if not self.results:
            print("No results to plot")
            return

        # Filter available metrics
        available_metrics = []
        for metric in metrics_to_plot:
            if any(metric in str(v) for v in self.results.values()):
                available_metrics.append(metric)

        if not available_metrics:
            print("No matching metrics found")
            return

        num_plots = len(available_metrics)
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))

        if num_plots == 1:
            axes = [axes]

        model_names = list(self.results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]

            values = []
            for model_name in model_names:
                metrics = self.results[model_name]
                # Find the value for this metric
                val = None
                for key, v in metrics.items():
                    if metric.lower() in key.lower():
                        val = v
                        break

                if val is None:
                    val = 0

                values.append(val)

            bars = ax.bar(range(len(model_names)), values, color=colors)
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.grid(alpha=0.3, axis='y')

            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            print(f"Saved comparison plot to {output_path}")

        plt.close()

    def generate_latex_table(self):
        """Generate LaTeX table for paper."""
        df = self.create_comparison_table()

        if df is None:
            return ""

        latex = df.to_latex(index=False, float_format=lambda x: f'{x:.2f}')

        return latex


def create_example_results():
    """Create example results for testing."""
    compiler = ResultsCompiler()

    # Add example results (from blueprint expectations)
    compiler.add_result('CNN Autoencoder', {
        'AUC-ROC': 0.76,
        'Params (M)': 2.0,
        'Inference (ms/frame)': 5.0,
        'Training Time (min)': 15,
    })

    compiler.add_result('ViT-S + Temporal', {
        'AUC-ROC': 0.88,
        'Params (M)': 22.0,
        'Inference (ms/frame)': 30.0,
        'Training Time (min)': 120,
    })

    compiler.add_result('ViT-S + Temporal + Distill', {
        'AUC-ROC': 0.86,
        'Params (M)': 16.0,
        'Inference (ms/frame)': 23.0,
        'Training Time (min)': 140,
    })

    compiler.add_result('VideoMamba (MambaVision-T)', {
        'AUC-ROC': 0.87,
        'Params (M)': 8.0,
        'Inference (ms/frame)': 15.0,
        'Training Time (min)': 100,
    })

    return compiler


def main():
    parser = argparse.ArgumentParser(description='Compile and visualize results')
    parser.add_argument('--input', type=str, default=None,
                      help='Input CSV file with results')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Output directory')
    parser.add_argument('--example', action='store_true',
                      help='Generate example results')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.example:
        print("Generating example results...")
        compiler = create_example_results()
    else:
        compiler = ResultsCompiler(args.output_dir)

        if args.input:
            print(f"Loading results from {args.input}...")
            compiler.load_from_file(args.input)
        else:
            print("No input file provided. Use --example to generate example results")
            return

    # Create table
    print("\n=== Comparison Table ===")
    df = compiler.create_comparison_table(
        output_path=os.path.join(args.output_dir, 'comparison_table.csv')
    )

    if df is not None:
        print(df.to_string(index=False))

    # Create plots
    print("\nGenerating comparison plots...")
    compiler.plot_comparison(
        output_path=os.path.join(args.output_dir, 'comparison_plots.png')
    )

    # Generate LaTeX
    latex_table = compiler.generate_latex_table()
    if latex_table:
        latex_path = os.path.join(args.output_dir, 'comparison_table.tex')
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"Saved LaTeX table to {latex_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
