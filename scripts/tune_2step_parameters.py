"""Parameter tuning script for 2-step RAG method

Tests different combinations of abstract_k, passage_k, and final_k
to optimize precision@5 while maintaining good performance.
"""

import os
import sys
import ast
import time
from datetime import datetime
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import pandas as pd
from tqdm.auto import tqdm
from groq import Groq
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from abstrag.core.database import (
    open_db_connection,
    PostgresParams,
    SemanticSearch,
)
from abstrag.core.retrieval import retrieve_similar_documents
from abstrag.utils.evaluation_metrics import compute_all_metrics
from abstrag.utils.path_resolver import find_env_file, find_evaluation_questions_file

# Find .env file
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    env_path = find_env_file(".env", script_dir=script_dir)
    load_dotenv(env_path)
except FileNotFoundError:
    load_dotenv()

EMBEDDING_MODEL_NAME = "multi-qa-mpnet-base-dot-v1"
TABLE_EMBEDDING_ARTICLE = f"embedding_article_{EMBEDDING_MODEL_NAME}".replace("-", "_")
TABLE_EMBEDDING_ABSTRACT = f"embedding_abstract_{EMBEDDING_MODEL_NAME}".replace("-", "_")

POSTGRES_USER = os.environ["POSTGRES_USER"]
POSTGRES_PWD = os.environ["POSTGRES_PWD"]
POSTGRES_DB = os.environ["POSTGRES_DB"]
POSTGRES_HOST = os.environ["POSTGRES_HOST"]
POSTGRES_PORT = os.environ["POSTGRES_PORT"]

K_VALUES = [1, 3, 5, 10]
RANDOM_SEED = 42
MAX_QUERIES = 3  # Limited for API call conservation

# Parameter search space
# Testing wider ranges to find optimal values
PARAMETER_COMBINATIONS = [
    # Current baseline
    {"abstract_k": 10, "passage_k": 15, "final_k": 5},
    
    # Increase abstract_k to catch more candidate papers
    {"abstract_k": 15, "passage_k": 20, "final_k": 5},
    {"abstract_k": 20, "passage_k": 25, "final_k": 5},
    {"abstract_k": 25, "passage_k": 30, "final_k": 5},
    
    # Increase passage_k to get more passages from candidate papers
    {"abstract_k": 10, "passage_k": 25, "final_k": 5},
    {"abstract_k": 10, "passage_k": 30, "final_k": 5},
    {"abstract_k": 15, "passage_k": 30, "final_k": 5},
    
    # Increase final_k to return more passages (but still evaluate precision@5)
    {"abstract_k": 15, "passage_k": 25, "final_k": 10},
    {"abstract_k": 20, "passage_k": 30, "final_k": 10},
    
    # Balanced approach - moderate increases
    {"abstract_k": 15, "passage_k": 20, "final_k": 5},
    {"abstract_k": 20, "passage_k": 25, "final_k": 5},
]


def load_evaluation_questions(path: str) -> pd.DataFrame:
    """Load evaluation questions from CSV"""
    from abstrag.utils.path_resolver import find_evaluation_questions_file
    
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        resolved_path = find_evaluation_questions_file(
            os.path.basename(path) if os.path.dirname(path) else path,
            script_dir=script_dir
        )
    except FileNotFoundError:
        if os.path.exists(path):
            resolved_path = path
        else:
            raise FileNotFoundError(
                f"Evaluation questions file not found: {path}\n"
                f"Please generate it first using: python scripts/generate_evaluation_questions.py"
            )
    
    path = resolved_path
    evaluation_questions = pd.read_csv(path, index_col=[0], sep=";")
    
    list_article_id = []
    list_questions = []
    for idx, row in evaluation_questions.iterrows():
        article_id = row["document_id"]
        raw_questions = row["questions"]
        try:
            questions = ast.literal_eval(
                raw_questions.replace('"["', '["').replace('"]"', '"]')
            )
        except Exception as e:
            questions = ast.literal_eval(
                raw_questions.replace('"["', '["').replace("[", '["').replace('"]"', '"]')
            )
        
        list_article_id.append(article_id)
        list_questions.append(questions)
    
    dict_evaluation = {}
    for i in range(len(list_article_id)):
        dict_evaluation[list_article_id[i]] = list_questions[i]
    
    frame_evaluation = pd.DataFrame.from_dict(dict_evaluation, orient="index")
    return frame_evaluation


def evaluate_parameter_config(
    params: Dict,
    query: str,
    ground_truth_doc_id: str,
    conn,
) -> Dict:
    """Evaluate a single parameter configuration on a single query
    
    Returns:
        Dictionary with metrics
    """
    results = {
        "abstract_k": params["abstract_k"],
        "passage_k": params["passage_k"],
        "final_k": params["final_k"],
        "query": query,
        "ground_truth_doc_id": ground_truth_doc_id,
    }
    
    # Prepare retrieval parameters for 2-step method
    retrieval_params = [
        SemanticSearch(
            query=query,
            table=TABLE_EMBEDDING_ABSTRACT,
            similarity_metric="<#>",
            embedding_model=EMBEDDING_MODEL_NAME,
            max_documents=params["abstract_k"],
        ),
        SemanticSearch(
            query=query,
            table=TABLE_EMBEDDING_ARTICLE,
            similarity_metric="<#>",
            embedding_model=EMBEDDING_MODEL_NAME,
            max_documents=params["passage_k"],
        ),
    ]
    
    # Measure retrieval latency
    start_time = time.time()
    retrieved_docs = retrieve_similar_documents(
        conn=conn,
        retrieval_method="pg_semantic_abstract+article",
        retrieval_parameters=retrieval_params,
        abstract_k=params["abstract_k"],
        passage_k=params["passage_k"],
        final_k=params["final_k"],
    )
    retrieval_latency = time.time() - start_time
    results["retrieval_latency"] = retrieval_latency
    
    relevant_docs = {ground_truth_doc_id}
    retrieved_doc_ids = retrieved_docs["references"]
    
    ir_metrics = compute_all_metrics(
        relevant_docs=relevant_docs,
        retrieved_docs=retrieved_doc_ids,
        k_values=K_VALUES,
    )
    results.update(ir_metrics)
    
    return results


def main():
    """Main tuning function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tune 2-step RAG parameters")
    parser.add_argument(
        "--questions-file",
        type=str,
        default=None,
        help="Path to evaluation questions CSV file",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=MAX_QUERIES,
        help="Maximum number of queries to evaluate",
    )
    
    args = parser.parse_args()
    
    # Resolve questions file path
    if args.questions_file is None:
        # Try multiple possible default files
        default_files = [
            "metadata_evaluation_questions_725_fixed.csv",
            "metadata_evaluation_questions_84_fixed.csv",
            "metadata_evaluation_questions_84.csv",
        ]
        args.questions_file = None
        for default_file in default_files:
            try:
                default_path = find_evaluation_questions_file(
                    default_file,
                    script_dir=script_dir
                )
                args.questions_file = default_path
                break
            except FileNotFoundError:
                continue
        
        if args.questions_file is None:
            raise FileNotFoundError(
                "No evaluation questions file specified and default file not found.\n"
                "Please specify --questions-file or generate evaluation questions using:\n"
                "   python scripts/generate_evaluation_questions.py"
            )
    
    print("Loading evaluation questions...")
    frame_evaluation = load_evaluation_questions(args.questions_file)
    
    # Filter by documents in database
    postgres_params = PostgresParams(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        pwd=POSTGRES_PWD,
        database=POSTGRES_DB,
    )
    conn = open_db_connection(
        connection_params=postgres_params,
        autocommit=True,
    )
    
    if conn is not None:
        cur = conn.cursor()
        cur.execute(f"SELECT article_id FROM {TABLE_EMBEDDING_ABSTRACT}")
        document_ids = cur.fetchall()
        document_ids = [doc_id[0] for doc_id in document_ids]
        cur.close()
    
    frame_evaluation_filt = frame_evaluation.loc[document_ids, :]
    
    # Sample queries
    num_queries = min(args.max_queries, len(frame_evaluation_filt))
    frame_evaluation_filt = frame_evaluation_filt.sample(
        n=num_queries,
        random_state=RANDOM_SEED,
    )
    print(f"Testing with {num_queries} queries")
    print(f"Testing {len(PARAMETER_COMBINATIONS)} parameter combinations")
    print(f"Total evaluations: {num_queries * len(PARAMETER_COMBINATIONS)}")
    
    # Run evaluation
    all_results = []
    
    for param_idx, params in enumerate(PARAMETER_COMBINATIONS):
        print(f"\n{'='*80}")
        print(f"Testing configuration {param_idx + 1}/{len(PARAMETER_COMBINATIONS)}")
        print(f"abstract_k={params['abstract_k']}, passage_k={params['passage_k']}, final_k={params['final_k']}")
        print(f"{'='*80}")
        
        for original_id, row in tqdm(
            frame_evaluation_filt.iterrows(),
            total=len(frame_evaluation_filt),
            desc=f"Config {param_idx + 1}",
        ):
            # Use first question from each document
            question = row.iloc[0] if len(row) > 0 else None
            if question:
                try:
                    result = evaluate_parameter_config(
                        params=params,
                        query=question,
                        ground_truth_doc_id=original_id,
                        conn=conn,
                    )
                    all_results.append(result)
                    time.sleep(0.1)  # Small delay to avoid overwhelming DB
                except Exception as e:
                    print(f"Error evaluating config {param_idx + 1} on query '{question}': {e}")
                    continue
    
    results_df = pd.DataFrame(all_results)
    
    # Aggregate results by parameter configuration
    grouped = results_df.groupby(["abstract_k", "passage_k", "final_k"])
    
    aggregated_results = []
    for (abstract_k, passage_k, final_k), group in grouped:
        agg_result = {
            "abstract_k": abstract_k,
            "passage_k": passage_k,
            "final_k": final_k,
            "num_queries": len(group),
            "precision@1_mean": group["precision@1"].mean(),
            "precision@3_mean": group["precision@3"].mean(),
            "precision@5_mean": group["precision@5"].mean(),
            "precision@10_mean": group["precision@10"].mean(),
            "ndcg@1_mean": group["ndcg@1"].mean(),
            "ndcg@3_mean": group["ndcg@3"].mean(),
            "ndcg@5_mean": group["ndcg@5"].mean(),
            "ndcg@10_mean": group["ndcg@10"].mean(),
            "hit_rate_mean": group["hit_rate"].mean(),
            "mrr_mean": group["mrr"].mean(),
            "retrieval_latency_mean": group["retrieval_latency"].mean(),
        }
        aggregated_results.append(agg_result)
    
    summary_df = pd.DataFrame(aggregated_results)
    
    # Sort by precision@5 (primary) and ndcg@5 (secondary)
    summary_df = summary_df.sort_values(
        by=["precision@5_mean", "ndcg@5_mean"],
        ascending=False
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"reports/evaluation/results/parameter_tuning_{timestamp}.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False, sep=";")
    
    summary_file = f"reports/evaluation/results/parameter_tuning_{timestamp}_summary.csv"
    summary_df.to_csv(summary_file, index=False, sep=";")
    
    print(f"\n{'='*80}")
    print("PARAMETER TUNING RESULTS")
    print(f"{'='*80}")
    print("\nTop 5 configurations by Precision@5:")
    print(summary_df.head(5).to_string(index=False))
    
    print(f"\n{'='*80}")
    print("BEST CONFIGURATION")
    print(f"{'='*80}")
    best_config = summary_df.iloc[0]
    print(f"abstract_k: {best_config['abstract_k']}")
    print(f"passage_k: {best_config['passage_k']}")
    print(f"final_k: {best_config['final_k']}")
    print(f"\nMetrics:")
    print(f"  Precision@5: {best_config['precision@5_mean']:.4f}")
    print(f"  nDCG@5: {best_config['ndcg@5_mean']:.4f}")
    print(f"  Hit Rate: {best_config['hit_rate_mean']:.4f}")
    print(f"  MRR: {best_config['mrr_mean']:.4f}")
    print(f"  Retrieval Latency: {best_config['retrieval_latency_mean']:.4f}s")
    
    print(f"\nResults saved to:")
    print(f"  Detailed: {output_file}")
    print(f"  Summary: {summary_file}")
    
    return summary_df


if __name__ == "__main__":
    try:
        results = main()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo generate evaluation questions, run:")
        print("   python scripts/generate_evaluation_questions.py")
        sys.exit(1)

