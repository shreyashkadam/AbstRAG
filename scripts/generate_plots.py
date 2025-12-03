"""Generate visualization plots for evaluation results"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from abstrag.utils.path_resolver import find_file

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def plot_precision_curves(results_file: str, output_dir: str):
    """Plot Precision@k curves for all methods
    
    Args:
        results_file: Path to evaluation results CSV
        output_dir: Directory to save plots
    """
    # Resolve file path
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resolved_path = find_file(results_file, script_dir=script_dir)
    df = pd.read_csv(resolved_path, sep=";")
    
    k_values = [1, 3, 5, 10]
    methods = df["method"].unique()
    
    precision_data = []
    for method in methods:
        method_df = df[df["method"] == method]
        for k in k_values:
            precision_col = f"precision@{k}"
            if precision_col in method_df.columns:
                mean_precision = method_df[precision_col].mean()
                precision_data.append({
                    "Method": method,
                    "k": k,
                    "Precision": mean_precision,
                })
    
    precision_df = pd.DataFrame(precision_data)
    
    plt.figure(figsize=(10, 6))
    for method in methods:
        method_data = precision_df[precision_df["Method"] == method]
        plt.plot(method_data["k"], method_data["Precision"], marker="o", label=method)
    
    plt.xlabel("k")
    plt.ylabel("Precision@k")
    plt.title("Precision@k Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "precision_at_k.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved precision curve to {output_path}")


def plot_ndcg_comparison(results_file: str, output_dir: str):
    """Plot nDCG@k comparison bar chart
    
    Args:
        results_file: Path to evaluation results CSV
        output_dir: Directory to save plots
    """
    # Resolve file path
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resolved_path = find_file(results_file, script_dir=script_dir)
    df = pd.read_csv(resolved_path, sep=";")
    
    k_values = [1, 3, 5, 10]
    methods = df["method"].unique()
    
    ndcg_data = []
    for method in methods:
        method_df = df[df["method"] == method]
        for k in k_values:
            ndcg_col = f"ndcg@{k}"
            if ndcg_col in method_df.columns:
                mean_ndcg = method_df[ndcg_col].mean()
                ndcg_data.append({
                    "Method": method,
                    "k": k,
                    "nDCG": mean_ndcg,
                })
    
    ndcg_df = pd.DataFrame(ndcg_data)
    
    plt.figure(figsize=(10, 6))
    ndcg_pivot = ndcg_df.pivot(index="Method", columns="k", values="nDCG")
    ndcg_pivot.plot(kind="bar", ax=plt.gca())
    
    plt.xlabel("Method")
    plt.ylabel("nDCG@k")
    plt.title("nDCG@k Comparison")
    plt.legend(title="k", labels=[f"k={k}" for k in k_values])
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "ndcg_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved nDCG comparison to {output_path}")


def plot_latency_comparison(results_file: str, output_dir: str):
    """Plot latency comparison
    
    Args:
        results_file: Path to evaluation results CSV
        output_dir: Directory to save plots
    """
    # Resolve file path
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resolved_path = find_file(results_file, script_dir=script_dir)
    df = pd.read_csv(resolved_path, sep=";")
    
    methods = df["method"].unique()
    
    latency_data = []
    for method in methods:
        method_df = df[df["method"] == method]
        latency_data.append({
            "Method": method,
            "Retrieval": method_df["retrieval_latency"].mean(),
            "Generation": method_df["generation_latency"].mean(),
            "Total": method_df["total_latency"].mean(),
        })
    
    latency_df = pd.DataFrame(latency_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(methods))
    width = 0.25
    
    ax.bar([i - width for i in x], latency_df["Retrieval"], width, label="Retrieval", alpha=0.8)
    ax.bar(x, latency_df["Generation"], width, label="Generation", alpha=0.8)
    ax.bar([i + width for i in x], latency_df["Total"], width, label="Total", alpha=0.8)
    
    ax.set_xlabel("Method")
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Latency Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "latency_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved latency comparison to {output_path}")


def plot_answer_quality(results_file: str, output_dir: str):
    """Plot answer quality metrics
    
    Args:
        results_file: Path to evaluation results CSV
        output_dir: Directory to save plots
    """
    # Resolve file path
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resolved_path = find_file(results_file, script_dir=script_dir)
    df = pd.read_csv(resolved_path, sep=";")
    
    methods = df["method"].unique()
    
    quality_data = []
    for method in methods:
        method_df = df[df["method"] == method]
        quality_data.append({
            "Method": method,
            "Relevance": method_df["relevance_score"].mean(),
            "Factual Accuracy": method_df["factual_accuracy_score"].mean(),
        })
    
    quality_df = pd.DataFrame(quality_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(methods))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], quality_df["Relevance"], width, label="Relevance", alpha=0.8)
    ax.bar([i + width/2 for i in x], quality_df["Factual Accuracy"], width, label="Factual Accuracy", alpha=0.8)
    
    ax.set_xlabel("Method")
    ax.set_ylabel("Score")
    ax.set_title("Answer Quality Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "answer_quality.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved answer quality plot to {output_path}")


def main():
    """Main plotting function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to evaluation results CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/evaluation/figures",
        help="Output directory for plots",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating plots from {args.results_file}...")
    
    # Generate all plots
    plot_precision_curves(args.results_file, args.output_dir)
    plot_ndcg_comparison(args.results_file, args.output_dir)
    plot_latency_comparison(args.results_file, args.output_dir)
    plot_answer_quality(args.results_file, args.output_dir)
    
    print(f"\nAll plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()

