"""BM25 retrieval implementation for baseline evaluation"""

import logging
from typing import List, Tuple, Optional, Dict
import psycopg
from abstrag.core.database import TextSearch

logger = logging.getLogger(__name__)

# Try to import rank_bm25, but make it optional
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 not available. Install with: pip install rank-bm25")


class BM25Index:
    """BM25 index for document retrieval"""
    
    def __init__(self, documents: List[str], paper_ids: List[str]):
        """Initialize BM25 index
        
        Args:
            documents: List of document texts
            paper_ids: List of corresponding paper IDs
        """
        if not BM25_AVAILABLE:
            raise ImportError("rank_bm25 is required for BM25 retrieval. Install with: pip install rank-bm25")
        
        if len(documents) != len(paper_ids):
            raise ValueError("documents and paper_ids must have the same length")
        
        # Tokenize documents
        tokenized_docs = [doc.lower().split() for doc in documents]
        
        # Initialize BM25
        self.bm25 = BM25Okapi(tokenized_docs)
        self.documents = documents
        self.paper_ids = paper_ids
        logger.info(f"Initialized BM25 index with {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Search using BM25
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of (paper_id, document, score) tuples sorted by score (descending)
        """
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort by score (descending) and get top k
        scored_docs = list(zip(self.paper_ids, self.documents, scores))
        scored_docs.sort(key=lambda x: x[2], reverse=True)
        
        return scored_docs[:top_k]


def bm25_search_postgres(
    conn: psycopg.Connection,
    text_search_params: TextSearch,
    bm25_index: Optional[BM25Index] = None,
) -> List[Tuple[str, str, float]]:
    """Perform BM25 search on PostgreSQL database
    
    Args:
        conn: Database connection
        text_search_params: Search parameters
        bm25_index: Pre-built BM25 index (if None, will build from database)
        
    Returns:
        List of (article_id, content, score) tuples
    """
    if not BM25_AVAILABLE:
        raise ImportError("rank_bm25 is required for BM25 retrieval. Install with: pip install rank-bm25")
    
    query = text_search_params["query"]
    table_name = text_search_params["table"]
    max_documents = text_search_params["max_documents"]
    
    # If index not provided, build it from database
    if bm25_index is None:
        logger.info(f"Building BM25 index from table {table_name}")
        # Fetch all documents from table
        from psycopg import sql
        
        table_identifier = sql.Identifier(table_name)
        query_sql = sql.SQL("SELECT article_id, content FROM {}").format(table_identifier)
        
        with conn.cursor() as cur:
            cur.execute(query_sql)
            results = cur.fetchall()
        
        if not results:
            logger.warning(f"No documents found in table {table_name}")
            return []
        
        paper_ids = [row[0] for row in results]
        documents = [row[1] for row in results]
        
        bm25_index = BM25Index(documents=documents, paper_ids=paper_ids)
    
    # Perform search
    results = bm25_index.search(query=query, top_k=max_documents)
    
    # Convert to expected format: (article_id, content, score)
    return [(paper_id, doc, score) for paper_id, doc, score in results]


