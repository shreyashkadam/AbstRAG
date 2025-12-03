"""
Illustrate document retrieval
"""

import os
import sys
from dotenv import load_dotenv
import pandas as pd
from typing import Final

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from abstrag.database import (
    open_db_connection,
    PostgresParams,
    SemanticSearch,
)
from abstrag.retrieval import retrieve_similar_documents
from abstrag.utils.path_resolver import find_env_file

# Find .env file
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    env_path = find_env_file(".env", script_dir=script_dir)
    load_dotenv(env_path)
except FileNotFoundError:
    # Fallback to current directory
    load_dotenv("./.env")

# Default embedding parameters
EMBEDDING_MODEL_NAME: Final = "multi-qa-mpnet-base-dot-v1"

TABLE_EMBEDDING_ARTICLE = f"embedding_article_{EMBEDDING_MODEL_NAME}".replace("-", "_")
TABLE_EMBEDDING_ABSTRACT = f"embedding_abstract_{EMBEDDING_MODEL_NAME}".replace(
    "-", "_"
)

POSTGRES_USER = os.environ["POSTGRES_USER"]
POSTGRES_PWD = os.environ["POSTGRES_PWD"]
POSTGRES_DB = os.environ["POSTGRES_DB"]
POSTGRES_HOST = os.environ["POSTGRES_HOST"]
POSTGRES_PORT = os.environ["POSTGRES_PORT"]

RETRIEVAL_METHOD: Final = "pg_semantic_abstract+article"

# Open connection to database
postgres_connection_params = PostgresParams(
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
    user=POSTGRES_USER,
    pwd=POSTGRES_PWD,
    database=POSTGRES_DB,
)

conn = open_db_connection(connection_params=postgres_connection_params, autocommit=True)

# Execute retrieval
user_question = "What are stocks?"
user_question = "What is the momentum of a stock?"
user_question = "How is risk measured in finance?"
user_question = "How are financial derivatives priced?"
user_question = "What are risk parity portfolios?"
semantic_search_abstract = SemanticSearch(
    query=user_question,
    table=TABLE_EMBEDDING_ABSTRACT,
    similarity_metric="<#>",
    embedding_model=EMBEDDING_MODEL_NAME,
    max_documents=5,
)

semantic_search_article = SemanticSearch(
    query=user_question,
    table=TABLE_EMBEDDING_ARTICLE,
    similarity_metric="<#>",
    embedding_model=EMBEDDING_MODEL_NAME,
    max_documents=5,
)

semantic_search_hierarchy = [semantic_search_abstract, semantic_search_article]

relevant_documents = retrieve_similar_documents(
    conn=conn,
    retrieval_method=RETRIEVAL_METHOD,
    retrieval_parameters=semantic_search_hierarchy,
)
