"""Ingestion service for processing and storing papers"""

import logging
from typing import List, Optional
from datetime import datetime
import psycopg
from tqdm.auto import tqdm

from abstrag.core.database import (
    open_db_connection,
    PostgresParams,
    get_article_id_data,
    insert_embedding_data,
    create_embedding_table,
)
from abstrag.core.embedding import (
    PaperEmbedding,
    ChunkParams,
    chunk_document,
    document_embedding,
    get_embedding_model,
)
from abstrag.models.paper import PaperID
from abstrag.config import ARXIV_FIELDS

# Import ingestion functions (will be moved to core later)
from abstrag.ingest import retrieve_arxiv_metadata, paper_html_to_markdown

logger = logging.getLogger(__name__)


class IngestionService:
    """Service for ingesting papers into the database"""
    
    def __init__(
        self,
        db_connection_params: PostgresParams,
        embedding_model_name: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """Initialize ingestion service
        
        Args:
            db_connection_params: Database connection parameters
            embedding_model_name: Name of embedding model
            chunk_size: Size of chunks
            chunk_overlap: Overlap between chunks
        """
        self.db_params = db_connection_params
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Table names
        self.table_article = f"embedding_article_{embedding_model_name}".replace("-", "_")
        self.table_abstract = f"embedding_abstract_{embedding_model_name}".replace("-", "_")
    
    def ingest_papers(
        self,
        max_papers: int,
        conn: Optional[psycopg.Connection] = None,
    ) -> int:
        """Ingest papers from arXiv
        
        Args:
            max_papers: Maximum number of papers to ingest
            conn: Optional database connection
            
        Returns:
            Number of papers successfully ingested
        """
        if conn is None:
            conn = open_db_connection(self.db_params, autocommit=True)
            if conn is None:
                raise ConnectionError("Failed to connect to database")
        
        # Get existing article IDs
        article_ids_stored = get_article_id_data(
            conn=conn, table_name=self.table_article
        )
        
        # Fetch new papers
        metadata = retrieve_arxiv_metadata(
            max_results=max_papers,
            exclude_ids=article_ids_stored,
            verbose=True,
        )
        
        # Process documents
        markdown_text = []
        for paper_id in tqdm(metadata, total=len(metadata), desc="Fetching papers"):
            article_markdown = paper_html_to_markdown(paper_id=paper_id, verbose=True)
            if article_markdown:
                dict_markdown = dict(id=paper_id["id"], article=article_markdown)
                dict_markdown["abstract"] = [
                    meta["summary"] for meta in metadata if meta["id"] == paper_id["id"]
                ]
                markdown_text.append(dict_markdown)
            else:
                logger.warning(f"Unable to parse document {paper_id['id']}")
        
        # Chunk and embed
        chunk_parameters = ChunkParams(
            method="MarkdownTextSplitter",
            size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        
        list_article_embeddings = []
        list_abstract_embeddings = []
        embedding_dimension = 0
        
        for article in tqdm(markdown_text, total=len(markdown_text), desc="Processing papers"):
            article_id = article["id"]
            document = article["article"]
            abstract = article["abstract"][0] if article["abstract"] else ""
            
            # Chunk document
            document_chunks = chunk_document(
                document=document, chunk_params=chunk_parameters
            )
            
            # Document embedding
            article_embedding = document_embedding(
                chunks=document_chunks, embedding_model=self.embedding_model_name
            )
            
            # Abstract embedding
            abstract_embedding = document_embedding(
                chunks=[abstract], embedding_model=self.embedding_model_name
            )
            
            if embedding_dimension == 0:
                embedding_dimension = article_embedding["dimension"]
            
            # Format output
            for i in range(len(article_embedding["content"])):
                row_store = PaperEmbedding(
                    id=article_id,
                    content=article_embedding["content"][i],
                    embeddings=article_embedding["embedding"][i, :],
                )
                list_article_embeddings.append(row_store)
            
            for i in range(len(abstract_embedding["content"])):
                row_store = PaperEmbedding(
                    id=article_id,
                    content=abstract_embedding["content"][i],
                    embeddings=abstract_embedding["embedding"][i, :],
                )
                list_abstract_embeddings.append(row_store)
        
        # Store in database
        if list_article_embeddings:
            insert_embedding_data(
                conn=conn,
                table_name=self.table_article,
                paper_embedding=list_article_embeddings,
            )
        
        if list_abstract_embeddings:
            insert_embedding_data(
                conn=conn,
                table_name=self.table_abstract,
                paper_embedding=list_abstract_embeddings,
            )
        
        logger.info(f"Ingested {len(markdown_text)} papers")
        return len(markdown_text)


