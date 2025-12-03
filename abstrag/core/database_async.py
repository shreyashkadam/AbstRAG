"""Async database operations"""

import logging
from typing import List, Optional
import asyncpg
from pgvector.asyncpg import register_vector
from abstrag.core.database import PostgresParams, SemanticSearch, TextSearch

logger = logging.getLogger(__name__)


async def open_db_connection_async(
    connection_params: PostgresParams,
) -> Optional[asyncpg.Connection]:
    """Open async connection to PostgreSQL database
    
    Args:
        connection_params: Connection parameters
        
    Returns:
        Async connection or None if failed
    """
    try:
        conn = await asyncpg.connect(
            host=connection_params["host"],
            port=connection_params["port"],
            user=connection_params["user"],
            password=connection_params["pwd"],
            database=connection_params["database"],
        )
        await register_vector(conn)
        logger.info("Opened async database connection")
        return conn
    except Exception as e:
        logger.error(f"Error opening async database connection: {e}")
        return None


async def semantic_search_postgres_async(
    conn: asyncpg.Connection,
    semantic_search_params: SemanticSearch,
    filter_id: Optional[List[str]] = None,
):
    """Async semantic search
    
    Args:
        conn: Async database connection
        semantic_search_params: Search parameters
        filter_id: Optional article ID filter
        
    Returns:
        Search results and query embedding
    """
    # Import here to avoid circular dependency
    from abstrag.core.embedding import get_embedding_model
    
    embedding_model = semantic_search_params["embedding_model"]
    if isinstance(embedding_model, str):
        embedding_model = get_embedding_model(embedding_model)
    
    query = semantic_search_params["query"]
    table_name = semantic_search_params["table"]
    max_documents = semantic_search_params["max_documents"]
    similarity_metric = semantic_search_params["similarity_metric"]
    
    # Generate query embedding
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)
    
    # Build query
    if filter_id and len(filter_id) > 0:
        placeholders = ", ".join([f"${i+1}" for i in range(len(filter_id))])
        query_sql = f"""
        SELECT article_id, content, embedding FROM {table_name}
        WHERE article_id IN ({placeholders})
        ORDER BY embedding {similarity_metric} ${len(filter_id) + 1}
        LIMIT ${len(filter_id) + 2}
        """
        params = list(filter_id) + [query_embedding, max_documents]
    else:
        query_sql = f"""
        SELECT article_id, content, embedding FROM {table_name}
        ORDER BY embedding {similarity_metric} $1
        LIMIT $2
        """
        params = [query_embedding, max_documents]
    
    try:
        results = await conn.fetch(query_sql, *params)
        logger.debug(f"Async semantic search returned {len(results)} results")
        return results, query_embedding
    except Exception as e:
        logger.error(f"Error in async semantic search: {e}")
        raise


