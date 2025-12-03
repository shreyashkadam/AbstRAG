"""Comprehensive baseline evaluation script

Evaluates three methods:
- B1: BM25 keyword retrieval
- B2: Single-step semantic RAG
- Our Method: 2-step RAG pipeline (Abstract → Passage retrieval)

Computes:
- IR Metrics: Precision@k, nDCG@k, Hit Rate, MRR
- Answer Quality: Relevance, Factual Accuracy
- Efficiency: Latency metrics
"""

import os
import sys
import ast
import time
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv
import pandas as pd
from tqdm.auto import tqdm
from groq import Groq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from abstrag.core.database import (
    open_db_connection,
    PostgresParams,
    SemanticSearch,
    TextSearch,
)
from abstrag.core.retrieval import retrieve_similar_documents
from abstrag.core.llm import build_rag_prompt, GroqParams
from abstrag.utils.evaluation_metrics import compute_all_metrics
from abstrag.utils.answer_quality import (
    evaluate_relevance,
    evaluate_factual_accuracy,
    score_to_numeric,
)
from abstrag.utils.latency import measure_latency
from abstrag.utils.path_resolver import find_env_file, find_evaluation_questions_file

# Find .env file
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    env_path = find_env_file(".env", script_dir=script_dir)
    load_dotenv(env_path)
except FileNotFoundError:
    # Fallback to default behavior
    load_dotenv()

# Default evaluation questions path (lazy resolution - only used if no CLI arg provided)
def get_default_evaluation_questions_path():
    """Get default evaluation questions path, handling FileNotFoundError gracefully"""
    try:
        return find_evaluation_questions_file(
            "metadata_evaluation_questions_725_fixed.csv",
            script_dir=script_dir
        )
    except FileNotFoundError:
        # Return None if default file doesn't exist - will be resolved in main() if needed
        return None

PATH_EVALUATION_QUESTIONS = None  # Will be resolved lazily if needed

EMBEDDING_MODEL_NAME = "multi-qa-mpnet-base-dot-v1"
TABLE_EMBEDDING_ARTICLE = f"embedding_article_{EMBEDDING_MODEL_NAME}".replace("-", "_")
TABLE_EMBEDDING_ABSTRACT = f"embedding_abstract_{EMBEDDING_MODEL_NAME}".replace("-", "_")

POSTGRES_USER = os.environ["POSTGRES_USER"]
POSTGRES_PWD = os.environ["POSTGRES_PWD"]
POSTGRES_DB = os.environ["POSTGRES_DB"]
POSTGRES_HOST = os.environ["POSTGRES_HOST"]
POSTGRES_PORT = os.environ["POSTGRES_PORT"]

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
LLM_MODEL = "llama-3.1-8b-instant"
JUDGE_MODEL = "llama-3.1-8b-instant"

K_VALUES = [1, 3, 5, 10]
MAX_QUERIES = None
RANDOM_SEED = 42
METHODS = {
    "B1_BM25": {
        "retrieval_method": "pg_bm25_article",
        "retrieval_k": 5,
    },
    "B2_SingleStep": {
        "retrieval_method": "pg_semantic_article",
        "retrieval_k": 5,
    },
    "OurMethod_2Step": {
        "retrieval_method": "pg_semantic_abstract+article",
        "abstract_k": 10,  # 2-step: abstract search then body search
        "passage_k": 25,   # Optimized: increased from 15 to 25 for better passage diversity
        "final_k": 5,
    },
}


def load_evaluation_questions(path: str) -> pd.DataFrame:
    """Load evaluation questions from CSV"""
    from abstrag.utils.path_resolver import find_evaluation_questions_file
    
    # Resolve file path using utility function
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        resolved_path = find_evaluation_questions_file(
            os.path.basename(path) if os.path.dirname(path) else path,
            script_dir=script_dir
        )
    except FileNotFoundError:
        # If utility can't find it, try the provided path as-is
        if os.path.exists(path):
            resolved_path = path
        else:
            raise FileNotFoundError(
                f"Evaluation questions file not found: {path}\n"
                f"Please generate it first using: python scripts/generate_evaluation_questions.py\n"
                f"Or place the file at: {path}"
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


def evaluate_method(
    method_name: str,
    method_config: Dict,
    query: str,
    ground_truth_doc_id: str,
    conn,
    groq_client: Groq,
) -> Dict:
    """Evaluate a single method on a single query
    
    Returns:
        Dictionary with all metrics
    """
    results = {
        "method": method_name,
        "query": query,
        "ground_truth_doc_id": ground_truth_doc_id,
    }
    
    # Prepare retrieval parameters
    if method_config["retrieval_method"] == "pg_bm25_article":
        retrieval_params = [
            TextSearch(
                query=query,
                table=TABLE_EMBEDDING_ARTICLE,
                max_documents=method_config["retrieval_k"],
            )
        ]
    elif method_config["retrieval_method"] == "pg_semantic_article":
        retrieval_params = [
            SemanticSearch(
                query=query,
                table=TABLE_EMBEDDING_ARTICLE,
                similarity_metric="<#>",
                embedding_model=EMBEDDING_MODEL_NAME,
                max_documents=method_config["retrieval_k"],
            )
        ]
    elif method_config["retrieval_method"] == "pg_semantic_abstract+article":
        retrieval_params = [
            SemanticSearch(
                query=query,
                table=TABLE_EMBEDDING_ABSTRACT,
                similarity_metric="<#>",
                embedding_model=EMBEDDING_MODEL_NAME,
                max_documents=method_config.get("abstract_k", 25),
            ),
            SemanticSearch(
                query=query,
                table=TABLE_EMBEDDING_ARTICLE,
                similarity_metric="<#>",
                embedding_model=EMBEDDING_MODEL_NAME,
                max_documents=method_config.get("passage_k", 50),
            ),
        ]
    else:
        raise ValueError(f"Unknown retrieval method: {method_config['retrieval_method']}")
    
    # Measure retrieval latency
    with measure_latency() as retrieval_timer:
        retrieved_docs = retrieve_similar_documents(
            conn=conn,
            retrieval_method=method_config["retrieval_method"],
            retrieval_parameters=retrieval_params,
            abstract_k=method_config.get("abstract_k"),
            passage_k=method_config.get("passage_k"),
            final_k=method_config.get("final_k"),
        )
    
    retrieval_latency = retrieval_timer.elapsed()
    results["retrieval_latency"] = retrieval_latency
    
    relevant_docs = {ground_truth_doc_id}
    retrieved_doc_ids = retrieved_docs["references"]
    
    ir_metrics = compute_all_metrics(
        relevant_docs=relevant_docs,
        retrieved_docs=retrieved_doc_ids,
        k_values=K_VALUES,
    )
    results.update(ir_metrics)
    
    prompt = build_rag_prompt(
        user_question=query,
        context=retrieved_docs["documents"],
    )
    
    # Measure generation latency
    with measure_latency() as generation_timer:
        try:
            chat_completion = groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            answer = chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating answer: {e}")
            answer = ""
    
    generation_latency = generation_timer.elapsed()
    results["generation_latency"] = generation_latency
    results["total_latency"] = retrieval_latency + generation_latency
    results["answer"] = answer
    
    if answer:
        time.sleep(1)
        relevance_result = evaluate_relevance(
            question=query,
            answer=answer,
            groq_client=groq_client,
            judge_model=JUDGE_MODEL,
        )
        results["relevance"] = relevance_result["relevance"]
        results["relevance_score"] = score_to_numeric(relevance_result["relevance"], "relevance")
        results["relevance_explanation"] = relevance_result["explanation"]
        
        time.sleep(1)
        factual_result = evaluate_factual_accuracy(
            question=query,
            answer=answer,
            retrieved_documents=retrieved_docs["documents"],
            groq_client=groq_client,
            judge_model=JUDGE_MODEL,
        )
        results["factual_accuracy"] = factual_result["factual_accuracy"]
        results["factual_accuracy_score"] = score_to_numeric(
            factual_result["factual_accuracy"], "factual_accuracy"
        )
        results["factual_accuracy_explanation"] = factual_result["explanation"]
    else:
        results["relevance"] = "PARTLY_RELEVANT"
        results["relevance_score"] = 0.5
        results["relevance_explanation"] = "No answer generated"
        results["factual_accuracy"] = "PARTIALLY_CORRECT"
        results["factual_accuracy_score"] = 0.5
        results["factual_accuracy_explanation"] = "No answer generated"
    
    return results


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG baselines")
    parser.add_argument(
        "--questions-file",
        type=str,
        default=None,
        help="Path to evaluation questions CSV file",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum number of queries to evaluate (for testing)",
    )
    
    args = parser.parse_args()
    
    # Resolve questions file path
    if args.questions_file is None:
        # Try to use default path
        default_path = get_default_evaluation_questions_path()
        if default_path is None:
            raise FileNotFoundError(
                "No evaluation questions file specified and default file not found.\n"
                "Please specify --questions-file or generate evaluation questions using:\n"
                "   python scripts/generate_evaluation_questions.py"
            )
        args.questions_file = default_path
    
    # Override MAX_QUERIES if provided
    global MAX_QUERIES
    if args.max_queries:
        MAX_QUERIES = args.max_queries
    
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
    
    # Sample if needed
    if MAX_QUERIES:
        num_queries = min(MAX_QUERIES, len(frame_evaluation_filt))
        frame_evaluation_filt = frame_evaluation_filt.sample(
            n=num_queries,
            random_state=RANDOM_SEED,
        )
        print(f"Sampling {num_queries} queries for evaluation")
    
    print(f"Evaluating {len(frame_evaluation_filt)} queries across {len(METHODS)} methods")
    
    # Initialize Groq client
    groq_client = Groq(api_key=GROQ_API_KEY)
    
    # Run evaluation
    all_results = []
    
    for method_name, method_config in METHODS.items():
        print(f"\nEvaluating {method_name}...")
        
        for original_id, row in tqdm(
            frame_evaluation_filt.iterrows(),
            total=len(frame_evaluation_filt),
            desc=method_name,
        ):
            for question in row:
                try:
                    result = evaluate_method(
                        method_name=method_name,
                        method_config=method_config,
                        query=question,
                        ground_truth_doc_id=original_id,
                        conn=conn,
                        groq_client=groq_client,
                    )
                    all_results.append(result)
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Error evaluating {method_name} on query '{question}': {e}")
                    continue
    
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"reports/evaluation/results/baseline_evaluation_{timestamp}.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False, sep=";")
    
    print(f"\nResults saved to {output_file}")
    print(f"Total evaluations: {len(results_df)}")
    
    return results_df


if __name__ == "__main__":
    try:
        results = main()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 To generate evaluation questions, run:")
        print("   python scripts/generate_evaluation_questions.py")
        print("\n   Note: This requires metadata_all.csv and article_markdown.csv files")
        sys.exit(1)

