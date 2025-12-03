"""Generate paper-ready results (LaTeX tables, formatted output)"""

import os
import sys
import pandas as pd
from typing import Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from abstrag.utils.path_resolver import find_file


def create_latex_table(summary_df: pd.DataFrame, caption: str = "Evaluation Results") -> str:
    """Create LaTeX table from summary DataFrame
    
    Args:
        summary_df: Aggregated summary DataFrame
        caption: Table caption
        
    Returns:
        LaTeX table string
    """
    # Select key metrics
    key_metrics = [
        "precision@5_mean",
        "ndcg@5_mean",
        "hit_rate_mean",
        "mrr_mean",
        "relevance_score_mean",
        "factual_accuracy_score_mean",
        "total_latency_mean",
    ]
    
    # Filter to available metrics
    available_metrics = [m for m in key_metrics if m in summary_df.columns]
    table_df = summary_df[available_metrics].copy()
    
    # Rename columns
    column_mapping = {
        "precision@5_mean": "Precision@5",
        "ndcg@5_mean": "nDCG@5",
        "hit_rate_mean": "Hit Rate",
        "mrr_mean": "MRR",
        "relevance_score_mean": "Relevance",
        "factual_accuracy_score_mean": "Factual Acc.",
        "total_latency_mean": "Latency (s)",
    }
    
    table_df = table_df.rename(columns=column_mapping)
    
    # Format numbers
    for col in table_df.columns:
        if "Latency" in col:
            table_df[col] = table_df[col].apply(lambda x: f"{x:.3f}")
        else:
            table_df[col] = table_df[col].apply(lambda x: f"{x:.3f}")
    
    # Generate LaTeX
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += "\\label{tab:evaluation_results}\n"
    
    # Column specification
    num_cols = len(table_df.columns) + 1  # +1 for method column
    col_spec = "l" + "c" * (num_cols - 1)
    latex += f"\\begin{{tabular}}{{{col_spec}}}\n"
    latex += "\\toprule\n"
    
    # Header
    latex += "Method & " + " & ".join(table_df.columns) + " \\\\\n"
    latex += "\\midrule\n"
    
    # Rows
    for method, row in table_df.iterrows():
        latex += f"{method} & " + " & ".join(row.values) + " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex


def create_markdown_table(summary_df: pd.DataFrame) -> str:
    """Create markdown table from summary DataFrame
    
    Args:
        summary_df: Aggregated summary DataFrame
        
    Returns:
        Markdown table string
    """
    # Select key metrics
    key_metrics = [
        "precision@5_mean",
        "ndcg@5_mean",
        "hit_rate_mean",
        "mrr_mean",
        "relevance_score_mean",
        "factual_accuracy_score_mean",
        "total_latency_mean",
    ]
    
    # Filter to available metrics
    available_metrics = [m for m in key_metrics if m in summary_df.columns]
    table_df = summary_df[available_metrics].copy()
    
    # Rename columns
    column_mapping = {
        "precision@5_mean": "Precision@5",
        "ndcg@5_mean": "nDCG@5",
        "hit_rate_mean": "Hit Rate",
        "mrr_mean": "MRR",
        "relevance_score_mean": "Relevance",
        "factual_accuracy_score_mean": "Factual Accuracy",
        "total_latency_mean": "Latency (s)",
    }
    
    table_df = table_df.rename(columns=column_mapping)
    
    # Format numbers
    for col in table_df.columns:
        if "Latency" in col:
            table_df[col] = table_df[col].apply(lambda x: f"{x:.3f}")
        else:
            table_df[col] = table_df[col].apply(lambda x: f"{x:.3f}")
    
    # Generate markdown
    markdown = "## Evaluation Results\n\n"
    markdown += "| Method | " + " | ".join(table_df.columns) + " |\n"
    markdown += "|" + "|".join(["---"] * (len(table_df.columns) + 1)) + "|\n"
    
    for method, row in table_df.iterrows():
        markdown += f"| {method} | " + " | ".join(row.values) + " |\n"
    
    return markdown


def generate_paper_results(summary_file: str, output_file: str):
    """Generate paper-ready results document
    
    Args:
        summary_file: Path to summary CSV file
        output_file: Path to output markdown file
    """
    # Resolve file path
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resolved_path = find_file(summary_file, script_dir=script_dir)
    summary_df = pd.read_csv(resolved_path, sep=";", index_col=0)
    
    # Create markdown document
    content = "# Evaluation Results\n\n"
    content += "This document contains the evaluation results comparing our 2-step RAG pipeline against baselines.\n\n"
    
    # Methods section
    content += "## Methods\n\n"
    content += "We compare three methods:\n\n"
    content += "- **B1_BM25**: BM25 keyword retrieval baseline\n"
    content += "- **B2_SingleStep**: Single-step semantic RAG baseline\n"
    content += "- **OurMethod_2Step**: Our 2-step pipeline (Abstract → Passage retrieval)\n\n"
    
    # Metrics section
    content += "## Metrics\n\n"
    content += "### Information Retrieval Metrics\n\n"
    content += "- **Precision@k**: Proportion of retrieved documents that are relevant\n"
    content += "- **nDCG@k**: Normalized Discounted Cumulative Gain\n"
    content += "- **Hit Rate**: Proportion of queries where relevant document is retrieved\n"
    content += "- **MRR**: Mean Reciprocal Rank\n\n"
    
    content += "### Answer Quality Metrics\n\n"
    content += "- **Relevance**: LLM-as-a-judge evaluation of answer relevance (0-1 scale)\n"
    content += "- **Factual Accuracy**: LLM-as-a-judge evaluation of factual correctness (0-1 scale)\n\n"
    
    content += "### Efficiency Metrics\n\n"
    content += "- **Latency**: Average query processing time (seconds)\n\n"
    
    # Results section
    content += "## Results\n\n"
    content += create_markdown_table(summary_df)
    content += "\n\n"
    
    # Analysis section
    content += "## Analysis\n\n"
    content += "### Key Findings\n\n"
    
    # Find best method for each metric
    best_precision = summary_df["precision@5_mean"].idxmax()
    best_ndcg = summary_df["ndcg@5_mean"].idxmax()
    best_relevance = summary_df["relevance_score_mean"].idxmax()
    best_factual = summary_df["factual_accuracy_score_mean"].idxmax()
    fastest = summary_df["total_latency_mean"].idxmin()
    
    content += f"- **Best Precision@5**: {best_precision} ({summary_df.loc[best_precision, 'precision@5_mean']:.3f})\n"
    content += f"- **Best nDCG@5**: {best_ndcg} ({summary_df.loc[best_ndcg, 'ndcg@5_mean']:.3f})\n"
    content += f"- **Best Relevance**: {best_relevance} ({summary_df.loc[best_relevance, 'relevance_score_mean']:.3f})\n"
    content += f"- **Best Factual Accuracy**: {best_factual} ({summary_df.loc[best_factual, 'factual_accuracy_score_mean']:.3f})\n"
    content += f"- **Fastest**: {fastest} ({summary_df.loc[fastest, 'total_latency_mean']:.3f}s)\n\n"
    
    # LaTeX table
    content += "## LaTeX Table\n\n"
    content += "```latex\n"
    content += create_latex_table(summary_df)
    content += "```\n"
    
    # Save
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "w") as f:
        f.write(content)
    
    print(f"Paper results saved to {output_file}")
    
    # Also save LaTeX table separately
    latex_file = output_file.replace(".md", "_latex.tex")
    with open(latex_file, "w") as f:
        f.write(create_latex_table(summary_df))
    
    print(f"LaTeX table saved to {latex_file}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate paper-ready results")
    parser.add_argument(
        "summary_file",
        type=str,
        help="Path to summary CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/evaluation/paper_results.md",
        help="Output markdown file path",
    )
    
    args = parser.parse_args()
    
    generate_paper_results(args.summary_file, args.output)


if __name__ == "__main__":
    main()

