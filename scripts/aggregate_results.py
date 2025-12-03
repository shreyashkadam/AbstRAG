"""Aggregate and summarize evaluation results"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from abstrag.utils.path_resolver import find_file


def aggregate_results(results_file: str) -> pd.DataFrame:
    """Aggregate evaluation results by method
    
    Args:
        results_file: Path to evaluation results CSV
        
    Returns:
        DataFrame with aggregated metrics per method
    """
    # Resolve file path
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resolved_path = find_file(results_file, script_dir=script_dir)
    df = pd.read_csv(resolved_path, sep=";")
    
    # Group by method
    grouped = df.groupby("method")
    
    # Metrics to aggregate
    metrics = [
        "precision@1", "precision@3", "precision@5", "precision@10",
        "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10",
        "hit_rate", "mrr",
        "relevance_score", "factual_accuracy_score",
        "retrieval_latency", "generation_latency", "total_latency",
    ]
    
    aggregated = {}
    
    for method_name, group in grouped:
        method_metrics = {}
        
        for metric in metrics:
            if metric in group.columns:
                values = group[metric].dropna()
                if len(values) > 0:
                    method_metrics[f"{metric}_mean"] = values.mean()
                    method_metrics[f"{metric}_std"] = values.std()
                    method_metrics[f"{metric}_median"] = values.median()
        
        # Count queries
        method_metrics["num_queries"] = len(group)
        
        aggregated[method_name] = method_metrics
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(aggregated).T
    summary_df.index.name = "method"
    
    return summary_df


def create_comparison_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Create a comparison table with key metrics
    
    Args:
        summary_df: Aggregated summary DataFrame
        
    Returns:
        Comparison table with selected metrics
    """
    # Select key metrics for comparison
    comparison_metrics = [
        "precision@5_mean",
        "ndcg@5_mean",
        "hit_rate_mean",
        "mrr_mean",
        "relevance_score_mean",
        "factual_accuracy_score_mean",
        "total_latency_mean",
    ]
    
    # Filter to available metrics
    available_metrics = [m for m in comparison_metrics if m in summary_df.columns]
    
    comparison_df = summary_df[available_metrics].copy()
    
    # Rename columns for readability
    column_mapping = {
        "precision@5_mean": "Precision@5",
        "ndcg@5_mean": "nDCG@5",
        "hit_rate_mean": "Hit Rate",
        "mrr_mean": "MRR",
        "relevance_score_mean": "Relevance",
        "factual_accuracy_score_mean": "Factual Accuracy",
        "total_latency_mean": "Latency (s)",
    }
    
    comparison_df = comparison_df.rename(columns=column_mapping)
    
    return comparison_df


def main():
    """Main aggregation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to evaluation results CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: results_file with _summary suffix)",
    )
    
    args = parser.parse_args()
    
    # Aggregate results
    print(f"Loading results from {args.results_file}...")
    summary_df = aggregate_results(args.results_file)
    
    # Create comparison table
    comparison_df = create_comparison_table(summary_df)
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        base_name = args.results_file.replace(".csv", "")
        output_file = f"{base_name}_summary.csv"
    
    # Save summary
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    summary_df.to_csv(output_file, sep=";")
    print(f"Summary saved to {output_file}")
    
    # Save comparison table
    comparison_file = output_file.replace("_summary.csv", "_comparison.csv")
    comparison_df.to_csv(comparison_file, sep=";")
    print(f"Comparison table saved to {comparison_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(summary_df.to_string())
    
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(comparison_df.to_string())
    
    return summary_df, comparison_df


if __name__ == "__main__":
    summary, comparison = main()

