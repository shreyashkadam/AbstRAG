"""Interact with PostgreSQL database"""

import re
import datetime
import logging
import psycopg
from psycopg import sql
from pgvector.psycopg import register_vector
from typing import List, Literal, Optional, TypedDict, TYPE_CHECKING, Union, Any

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from abstrag.core.embedding import PaperEmbedding

# Initialize logger
logger = logging.getLogger(__name__)


class PostgresParams(TypedDict):
    host: str
    port: str
    user: str
    pwd: str
    database: str


class SemanticSearch(TypedDict):
    query: str
    table: str
    similarity_metric: Literal["<#>", "<=>", "<->", "<+>"]
    embedding_model: Union[str, Any]  # str or SentenceTransformer
    max_documents: int


class TextSearch(TypedDict):
    query: str
    table: str
    max_documents: int


class UserFeedback(TypedDict):
    user_id: str
    question: str
    answer: str
    thumbs: Optional[int]
    documents_retrieved: Optional[str]
    similarity: Optional[float]
    relevance: Optional[str]
    llm_model: Optional[str]
    embedding_model: Optional[str]
    elapsed_time: Optional[datetime.timedelta]
    feedback_timestamp: Optional[datetime.datetime]


def open_db_connection(
    connection_params: PostgresParams, autocommit: bool = True
) -> psycopg.Connection | None:
    """Open connection to PostgreSQL database

    Args:
        connection_params (PostgresParams): Connection parameters for
            opening connection to PostgreSQL database
        autocommit (bool, optional): Whether to create connection using
            autocommit model. Commands have immediate effect.
            Defaults to True.

    Returns:
        psycopg.Connection | None: Either a connection to the database
            or None if unable to create connection
    """
    try:
        conn = psycopg.connect(
            host=connection_params["host"],
            port=connection_params["port"],
            user=connection_params["user"],
            password=connection_params["pwd"],
            dbname=connection_params["database"],
            autocommit=autocommit,
        )
        curs = conn.cursor()

        # Execute an SQL query to test connection
        curs.execute("SELECT version();")
        db_version = curs.fetchone()
        logger.info(f"Connected to PostgreSQL - {db_version}")
        curs.close()
        return conn
    except psycopg.DatabaseError as error:
        logger.error(f"Database error while connecting to PostgreSQL: {error}")
        return None
    except Exception as error:
        logger.error(f"Unexpected error while connecting to PostgreSQL: {error}")
        return None


def check_connection_health(conn: psycopg.Connection) -> bool:
    """Check if database connection is healthy
    
    Args:
        conn (psycopg.Connection): Database connection to check
        
    Returns:
        bool: True if connection is healthy, False otherwise
    """
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            return True
    except Exception as e:
        logger.warning(f"Connection health check failed: {e}")
        return False


def validate_table_name(table_name: str) -> bool:
    """Validate table name to prevent SQL injection
    
    Args:
        table_name (str): Table name to validate
        
    Returns:
        bool: True if table name is valid, False otherwise
    """
    # Only allow alphanumeric characters, underscores, and hyphens
    # Table names should follow pattern: embedding_<type>_<model_name>
    pattern = r'^embedding_(abstract|article)_[a-zA-Z0-9_-]+$'
    return bool(re.match(pattern, table_name))


def create_embedding_table(
    conn: psycopg.Connection, table_name: str, embedding_dimension: int
):
    """
    Create a table for storing both text and vector embeddings

    The table contains the following fields:
    - article_id: string that specified the identified of the paper
    - content: string that contains the raw text
    - embedding: vector that contains the embedding of the raw text

    Args:
        conn (psycopg.Connection): Connection to the database
        table_name (str): Name of the table to be created. It should follow
            the following structure: 'embedding_<type>_<model_name>', where
            <type> is either 'abstract' or 'article' and <model_name> is
            the name of the embedding model used.
        embedding_dimension (int): Integer specifying the dimension of
            the embedding vectors
            
    Raises:
        ValueError: If table name is invalid
    """
    # Validate table name
    if not validate_table_name(table_name):
        raise ValueError(f"Invalid table name: {table_name}. Must follow pattern: embedding_<type>_<model_name>")
    
    # Use SQL identifier for safe table name quoting
    table_identifier = sql.Identifier(table_name)
    
    # Execute create table statement using safe SQL composition
    create_sql = sql.SQL("""
    CREATE TABLE IF NOT EXISTS {} (
        id bigserial PRIMARY KEY,
        article_id text,
        content text,
        embedding vector({})
    )
    """).format(table_identifier, sql.Literal(embedding_dimension))
    
    try:
        conn.execute(create_sql)
        logger.info(f"Created table: {table_name}")
    except psycopg.DatabaseError as e:
        logger.error(f"Error creating table {table_name}: {e}")
        raise

    # Execute create index statement for full-text search
    index_name = sql.Identifier(f"{table_name}_text_idx")
    index_sql = sql.SQL("""
    CREATE INDEX IF NOT EXISTS {} 
    ON {} USING GIN (to_tsvector('english', content))
    """).format(
        index_name,
        table_identifier
    )
    
    try:
        conn.execute(index_sql)
        logger.info(f"Created text index for table: {table_name}")
    except psycopg.DatabaseError as e:
        logger.warning(f"Error creating text index for {table_name}: {e}")

    # Create HNSW index for vector similarity search (optimization)
    vector_index_name = sql.Identifier(f"{table_name}_embedding_idx")
    vector_index_sql = sql.SQL("""
    CREATE INDEX IF NOT EXISTS {} 
    ON {} 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    """).format(
        vector_index_name,
        table_identifier
    )
    
    try:
        conn.execute(vector_index_sql)
        logger.info(f"Created vector index for table: {table_name}")
    except psycopg.DatabaseError as e:
        logger.warning(f"Error creating vector index for {table_name}: {e}. HNSW may not be available.")

    # Register pg_vector vector
    register_vector(conn)


def create_user_feedback_table(
    conn: psycopg.Connection, table_name: str = "user_feedback"
):

    # Execute create table statement
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        feedback_id SERIAL PRIMARY KEY,            -- Unique identifier for each feedback entry
        unique_user_id VARCHAR(255) NOT NULL,      -- Unique identifier for the user
        user_question TEXT NOT NULL,               -- The question asked by the user
        answer TEXT NOT NULL,                      -- The answer generated by the system
        thumbs SMALLINT,                           -- User rating: -1 for thumbs down, 1 for thumbs up, or NULL if not provided
        documents_retrieved TEXT,                  -- A list of document identifiers or titles retrieved for the query
        similarity FLOAT,                          -- Similarity score between the query and retrieved documents
        relevance TEXT,                            -- User's feedback on the relevance of the answer
        llm_model VARCHAR(255),                    -- Name of the LLM model used to generate the answer
        embedding_model VARCHAR(255),              -- Name of the embedding model used for document retrieval
        elapsed_time INTERVAL,                     -- Time elapsed between user query and LLM response
        feedback_timestamp TIMESTAMP DEFAULT NOW() -- Timestamp when the feedback was submitted
    )"""
    conn.execute(create_sql)


def insert_embedding_data(
    conn: psycopg.Connection, table_name: str, paper_embedding: List["PaperEmbedding"]
):
    """Insert paper embeddings into a PostgreSQL table

    Args:
        conn (psycopg.Connection): Connection to the database
        table_name (str): Table name where data will be inserted.
        paper_embedding (List[PaperEmbedding]): List of paper embeddings (PaperEmbedding imported from abstrag.core.embedding)
            that will be stored in the database
            
    Raises:
        ValueError: If table name is invalid
    """
    # Validate table name
    if not validate_table_name(table_name):
        raise ValueError(f"Invalid table name: {table_name}")
    
    register_vector(conn)

    # Use batch insert for better performance
    table_identifier = sql.Identifier(table_name)
    insert_sql = sql.SQL("INSERT INTO {} (article_id, content, embedding) VALUES (%s, %s, %s)").format(table_identifier)
    
    try:
        with conn.cursor() as curs:
            # Batch insert all rows
            data = [(row["id"], row["content"], row["embeddings"]) for row in paper_embedding]
            curs.executemany(insert_sql, data)
            logger.info(f"Inserted {len(paper_embedding)} embeddings into {table_name}")
    except psycopg.DatabaseError as e:
        logger.error(f"Error inserting embeddings into {table_name}: {e}")
        raise


def insert_user_feedback(
    conn: psycopg.Connection, feedback: UserFeedback, table_name: str = "user_feedback"
):
    """
    Insert a new record into the user_feedback table.

    Parameters:
        conn (psycopg.Connection): Connection object to the PostgreSQL database.
        feedback (UserFeedback): Instance of UserFeedback containing feedback data.
        table_name (str): The name of the table where the feedback will be inserted (default is 'user_feedback').
    """

    insert_sql = f"""
    INSERT INTO {table_name} (
        unique_user_id, user_question, answer, thumbs, documents_retrieved, similarity, relevance,
        llm_model, embedding_model, elapsed_time, feedback_timestamp
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    # Use feedback data to populate SQL parameters
    with conn.cursor() as cursor:
        cursor.execute(
            insert_sql,
            (
                feedback["user_id"],
                feedback["question"],
                feedback["answer"],
                feedback["thumbs"],
                feedback["documents_retrieved"],
                feedback["similarity"],
                feedback["relevance"],
                feedback["llm_model"],
                feedback["embedding_model"],
                feedback["elapsed_time"],
                feedback["feedback_timestamp"]
                or datetime.datetime.now(),  # Use current time if not provided
            ),
        )
        conn.commit()  # Commit the transaction to save the changes


def get_article_id_data(conn: psycopg.Connection, table_name: str) -> List[str]:
    """Get article ids already present in the database

    Args:
        conn (psycopg.Connection): Connection to the database
        table_name (str): Table name where data will be queried.

    Returns:
        List[str]: List of strings containing the different
            document ids
            
    Raises:
        ValueError: If table name is invalid
    """
    # Validate table name
    if not validate_table_name(table_name):
        raise ValueError(f"Invalid table name: {table_name}")
    
    table_identifier = sql.Identifier(table_name)
    query = sql.SQL("SELECT DISTINCT article_id FROM {}").format(table_identifier)
    
    try:
        with conn.cursor() as curs:
            curs.execute(query)
            data = curs.fetchall()
            # Format as list of strings
            document_ids = [docs[0] for docs in data]
            logger.debug(f"Retrieved {len(document_ids)} article IDs from {table_name}")
            return document_ids
    except psycopg.DatabaseError as e:
        logger.error(f"Error retrieving article IDs from {table_name}: {e}")
        raise


def semantic_search_postgres(
    conn: psycopg.Connection,
    semantic_search_params: SemanticSearch,
    filter_id: Optional[List[str]] = None,
):
    """Perform semantic search using vector similarity
    
    Args:
        conn (psycopg.Connection): Database connection
        semantic_search_params (SemanticSearch): Search parameters
        filter_id (Optional[List[str]]): Optional list of article IDs to filter by
        
    Returns:
        tuple: (search_results, query_embedding)
        
    Raises:
        ValueError: If table name is invalid or model cannot be loaded
    """
    # Cosine distance: <#>
    # negative inner product: <=>
    # L2 distance: <->
    # L1 distance: <+>
    
    # Get or load embedding model (cached models should be passed as SentenceTransformer)
    embedding_model = semantic_search_params["embedding_model"]
    if isinstance(embedding_model, str):
        # Import here to avoid circular dependency
        from abstrag.core.embedding import get_embedding_model
        try:
            embedding_model = get_embedding_model(embedding_model)
        except Exception as e:
            logger.error(f"Error loading embedding model {embedding_model}: {e}")
            raise ValueError(f"Unable to load embedding model {embedding_model}") from e
    
    query = semantic_search_params["query"]
    table_name = semantic_search_params["table"]
    max_documents = semantic_search_params["max_documents"]
    similarity_metric = semantic_search_params["similarity_metric"]

    # Validate table name
    if not validate_table_name(table_name):
        raise ValueError(f"Invalid table name: {table_name}")

    # Generate query embedding
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)

    register_vector(conn)

    # Build safe SQL query with parameterized filter
    table_identifier = sql.Identifier(table_name)
    
    if filter_id and len(filter_id) > 0:
        # Use parameterized query for filter_id to prevent SQL injection
        placeholders = sql.SQL(", ").join([sql.Placeholder()] * len(filter_id))
        filter_clause = sql.SQL(" WHERE article_id IN ({})").format(placeholders)
        query_sql = sql.SQL(
            "SELECT article_id, content, embedding FROM {} {} "
            "ORDER BY embedding {} %s LIMIT %s"
        ).format(
            table_identifier,
            filter_clause,
            sql.SQL(similarity_metric)
        )
        params = tuple(filter_id) + (query_embedding, max_documents)
    else:
        query_sql = sql.SQL(
            "SELECT article_id, content, embedding FROM {} "
            "ORDER BY embedding {} %s LIMIT %s"
        ).format(
            table_identifier,
            sql.SQL(similarity_metric)
        )
        params = (query_embedding, max_documents)

    try:
        with conn.cursor() as cur:
            cur.execute(query_sql, params)
            results = cur.fetchall()
            logger.debug(f"Semantic search returned {len(results)} results from {table_name}")
            return results, query_embedding
    except psycopg.DatabaseError as e:
        logger.error(f"Error performing semantic search on {table_name}: {e}")
        raise


def keyword_search_postgres(conn: psycopg.Connection, text_search_params: TextSearch):
    """Keyword search using PostgreSQL full-text search
    
    Note: This is a basic keyword search implementation with equal weight to each word.

    Args:
        conn (psycopg.Connection): Database connection
        text_search_params (TextSearch): Search parameters
        
    Returns:
        List: Search results
        
    Raises:
        ValueError: If table name is invalid
    """
    query = text_search_params["query"]
    table_name = text_search_params["table"]
    max_documents = text_search_params["max_documents"]

    # Validate table name
    if not validate_table_name(table_name):
        raise ValueError(f"Invalid table name: {table_name}")

    # Sanitize query: remove invalid characters and sanitize the input for to_tsquery
    query = re.sub(r"[^a-zA-Z0-9\s]", "", query)
    # query = re.sub(r'\s+', ' & ', query)  # Replace spaces with `&` for AND

    query_use = query.replace(" ", " | ")

    # Remove the trailing operator (and other redundant ones)
    query_use = re.sub(r"\|\s*$", "", query_use)  # Remove trailing `|`
    query_use = query_use.replace(" |  | ", " | ")
    # query_use = re.sub(r'\s*\|\s*', ' | ', query_use)  # Ensure correct spacing around `|`

    table_identifier = sql.Identifier(table_name)
    query_sql = sql.SQL(
        "SELECT article_id, content, embedding FROM {}, to_tsquery('english', %s) query "
        "WHERE to_tsvector('english', content) @@ query "
        "ORDER BY ts_rank_cd(to_tsvector('english', content), query) DESC LIMIT %s"
    ).format(table_identifier)

    try:
        with conn.cursor() as cur:
            cur.execute(query_sql, (query_use, max_documents))
            results = cur.fetchall()
            logger.debug(f"Keyword search returned {len(results)} results from {table_name}")
            return results
    except psycopg.DatabaseError as e:
        logger.error(f"Error performing keyword search on {table_name}: {e}")
        raise

