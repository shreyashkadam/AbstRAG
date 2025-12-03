"""Generate evaluation questions from papers in the database

This script extracts paper IDs and abstracts from the database
and generates evaluation questions using LLM.
"""

import os
import sys
import time
import json
import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm
from typing import TypedDict, List, Final

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from abstrag.core.database import (
    open_db_connection,
    PostgresParams,
    get_article_id_data,
)
from abstrag.core.llm import (
    llm_chat_completion,
    GroqParams,
    build_retrieval_evaluation_prompt,
)
from abstrag.config import get_config
from abstrag.utils.path_resolver import find_env_file

# Find .env file
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    env_path = find_env_file(".env", script_dir=script_dir)
    load_dotenv(env_path)
except FileNotFoundError:
    # Fallback to current directory
    load_dotenv(".env")

# Configuration
config_dict = get_config()
if config_dict:
    from abstrag.config import IngestionConfig
    config_ingestion = IngestionConfig(**config_dict["ingestion"])
else:
    raise ValueError("Failed to load configuration")

EMBEDDING_MODEL_NAME: Final = config_ingestion.embedding_model_name
TABLE_EMBEDDING_ABSTRACT = f"embedding_abstract_{EMBEDDING_MODEL_NAME}".replace("-", "_")

# Database connection
POSTGRES_USER = os.environ["POSTGRES_USER"]
POSTGRES_PWD = os.environ["POSTGRES_PWD"]
POSTGRES_DB = os.environ["POSTGRES_DB"]
POSTGRES_HOST = os.environ["POSTGRES_HOST"]
POSTGRES_PORT = os.environ["POSTGRES_PORT"]

# LLM configuration
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
LLM_MODEL: Final = "groq"
LLM_MODEL_PARAMS = GroqParams(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

# Evaluation configuration
MAX_PAPERS = None  # Set to None to process all papers, or a number to limit
RATE_LIMIT_DELAY = 2  # Seconds to wait between API calls


class EvaluationQuestions(TypedDict):
    document_id: str
    questions: List[str]


def get_abstracts_from_database(conn, table_name: str, paper_ids: List[str]) -> dict:
    """Get abstracts for given paper IDs from database
    
    Args:
        conn: Database connection
        table_name: Name of abstract table
        paper_ids: List of paper IDs
        
    Returns:
        Dictionary mapping paper_id to abstract text
    """
    from psycopg import sql
    
    abstracts = {}
    table_identifier = sql.Identifier(table_name)
    
    # Query abstracts for each paper ID
    # We'll get the first abstract entry for each paper (abstracts are stored as single chunks)
    # Use a subquery to get one abstract per paper_id
    query = sql.SQL(
        "SELECT DISTINCT ON (article_id) article_id, content FROM {} "
        "WHERE article_id = ANY(%s::text[]) ORDER BY article_id, id"
    ).format(table_identifier)
    
    try:
        with conn.cursor() as cur:
            cur.execute(query, (paper_ids,))
            results = cur.fetchall()
            for paper_id, content in results:
                abstracts[paper_id] = content
    except Exception as e:
        print(f"Error querying abstracts with batch query: {e}")
        print("Falling back to individual queries...")
        # Fallback: query one by one
        for paper_id in tqdm(paper_ids, desc="Fetching abstracts"):
            query_single = sql.SQL(
                "SELECT content FROM {} WHERE article_id = %s LIMIT 1"
            ).format(table_identifier)
            try:
                with conn.cursor() as cur:
                    cur.execute(query_single, (paper_id,))
                    result = cur.fetchone()
                    if result:
                        abstracts[paper_id] = result[0]
            except Exception as e:
                print(f"Error fetching abstract for {paper_id}: {e}")
    
    return abstracts


def main():
    """Main function to generate evaluation questions"""
    print("Connecting to database...")
    postgres_params = PostgresParams(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        pwd=POSTGRES_PWD,
        database=POSTGRES_DB,
    )
    conn = open_db_connection(connection_params=postgres_params, autocommit=True)
    
    if conn is None:
        raise ConnectionError("Failed to connect to database")
    
    print(f"Fetching paper IDs from table {TABLE_EMBEDDING_ABSTRACT}...")
    paper_ids = get_article_id_data(conn=conn, table_name=TABLE_EMBEDDING_ABSTRACT)
    print(f"Found {len(paper_ids)} papers in database")
    
    # Limit if specified
    if MAX_PAPERS and len(paper_ids) > MAX_PAPERS:
        print(f"Limiting to {MAX_PAPERS} papers")
        paper_ids = paper_ids[:MAX_PAPERS]
    
    # Get abstracts
    print("Fetching abstracts from database...")
    abstracts = get_abstracts_from_database(conn, TABLE_EMBEDDING_ABSTRACT, paper_ids)
    print(f"Retrieved {len(abstracts)} abstracts")
    
    if len(abstracts) == 0:
        raise ValueError("No abstracts found in database. Make sure papers are ingested.")
    
    # Generate questions
    print("Generating evaluation questions using LLM...")
    eval_questions = []
    failed_to_parse = []
    
    for paper_id in tqdm(paper_ids, total=len(paper_ids), desc="Generating questions"):
        if paper_id not in abstracts:
            print(f"Warning: No abstract found for {paper_id}, skipping")
            continue
        
        abstract = abstracts[paper_id]
        
        # Generate questions using LLM
        query = build_retrieval_evaluation_prompt(document=abstract, number_questions=3)
        
        try:
            response = llm_chat_completion(
                query=query, llm_model=LLM_MODEL, llm_parameters=LLM_MODEL_PARAMS
            )
            
            # Parse response into list
            try:
                questions = json.loads(response["response"])
                if not isinstance(questions, list):
                    # Try to extract list from string
                    questions = json.loads(questions)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                response_text = response["response"]
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                    questions = json.loads(response_text)
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                    questions = json.loads(response_text)
                else:
                    print(f"Failed to parse questions for {paper_id}: {response['response']}")
                    failed_to_parse.append(paper_id)
                    continue
            
            # Ensure we have exactly 3 questions
            if isinstance(questions, list) and len(questions) >= 3:
                questions = questions[:3]  # Take first 3
            elif isinstance(questions, list) and len(questions) > 0:
                # Pad with empty strings if needed
                questions = questions + [""] * (3 - len(questions))
            else:
                print(f"Invalid questions format for {paper_id}: {questions}")
                failed_to_parse.append(paper_id)
                continue
            
            eval_question = EvaluationQuestions(document_id=paper_id, questions=questions)
            eval_questions.append(eval_question)
            
            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)
            
        except Exception as e:
            print(f"Error generating questions for {paper_id}: {e}")
            failed_to_parse.append(paper_id)
            continue
    
    # Store as CSV file
    if failed_to_parse:
        print(f"\nFailed to parse questions for {len(failed_to_parse)} papers:")
        print(failed_to_parse[:10])  # Show first 10
    
    if eval_questions:
        # Save to reports/evaluation/data/ directory
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "reports", "evaluation", "data"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"metadata_evaluation_questions_{len(eval_questions)}.csv")
        df = pd.DataFrame(eval_questions)
        df.to_csv(output_file, sep=";", index=True)
        print(f"\n✅ Successfully generated {len(eval_questions)} evaluation questions")
        print(f"📁 Saved to: {output_file}")
        
        # Also create the "fixed" version
        fixed_file = output_file.replace(".csv", "_fixed.csv")
        df.to_csv(fixed_file, sep=";", index=True)
        print(f"📁 Also saved as: {fixed_file}")
    else:
        print("\n❌ No evaluation questions generated. Check errors above.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate evaluation questions from database")
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Maximum number of papers to process (for testing)",
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=2.0,
        help="Seconds to wait between API calls",
    )
    
    args = parser.parse_args()
    
    if args.max_papers:
        MAX_PAPERS = args.max_papers
    if args.rate_limit_delay:
        RATE_LIMIT_DELAY = args.rate_limit_delay
    
    main()

