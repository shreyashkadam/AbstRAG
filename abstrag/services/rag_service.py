"""RAG service - Main orchestration service for RAG pipeline"""

import logging
from datetime import datetime, timedelta
from typing import Optional
import psycopg

from abstrag.core.database import (
    open_db_connection,
    PostgresParams,
    SemanticSearch,
    check_connection_health,
)
from abstrag.core.retrieval import retrieve_similar_documents, RetrievalMethod
from abstrag.core.llm import build_rag_prompt, GroqParams
from abstrag.models.query import Query, QueryResult
from abstrag.models.response import RAGResponse
from abstrag.config import get_config
from groq import Groq

logger = logging.getLogger(__name__)


class RAGService:
    """Main service for RAG operations"""
    
    def __init__(
        self,
        db_connection_params: PostgresParams,
        groq_api_key: str,
        embedding_model_name: str,
        llm_model: str,
        retrieval_method: RetrievalMethod,
        abstract_retrieval_k: Optional[int] = None,
        passage_retrieval_k: Optional[int] = None,
        final_k: Optional[int] = None,
    ):
        """Initialize RAG service
        
        Args:
            db_connection_params: Database connection parameters
            groq_api_key: Groq API key
            embedding_model_name: Name of embedding model
            llm_model: Name of LLM model
            retrieval_method: Retrieval method to use
            abstract_retrieval_k: Number of candidate papers to retrieve from abstracts (default: 10)
            passage_retrieval_k: Number of passages to retrieve (default: 15)
            final_k: Number of final passages to return (default: 5)
        """
        self.db_params = db_connection_params
        self.groq_api_key = groq_api_key
        self.embedding_model_name = embedding_model_name
        self.llm_model = llm_model
        self.retrieval_method = retrieval_method
        
        # 2-step retrieval configuration
        self.abstract_retrieval_k = abstract_retrieval_k or 10
        self.passage_retrieval_k = passage_retrieval_k or 15
        self.final_k = final_k or 5
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=groq_api_key)
        
        # Table names
        self.table_article = f"embedding_article_{embedding_model_name}".replace("-", "_")
        self.table_abstract = f"embedding_abstract_{embedding_model_name}".replace("-", "_")
        
        # Database connection (lazy loaded)
        self._conn: Optional[psycopg.Connection] = None
    
    def _get_connection(self) -> psycopg.Connection:
        """Get or create database connection"""
        if self._conn is None or not check_connection_health(self._conn):
            self._conn = open_db_connection(self.db_params, autocommit=True)
            if self._conn is None:
                raise ConnectionError("Failed to connect to database")
        return self._conn
    
    def query(self, user_query: str, user_id: Optional[str] = None) -> RAGResponse:
        """Process a user query through the RAG pipeline
        
        Args:
            user_query: User's question
            user_id: Optional user identifier
            
        Returns:
            RAGResponse with answer and metadata
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Retrieve relevant documents
            query_result = self.retrieve(user_query)
            
            # Step 2: Build prompt
            prompt = build_rag_prompt(
                user_question=query_result["query"],
                context=query_result["documents"],
            )
            
            # Step 3: Generate answer
            chat_completion = self.groq_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            
            answer = chat_completion.choices[0].message.content
            end_time = datetime.now()
            
            response = RAGResponse(
                answer=answer,
                query=user_query,
                documents_used=query_result["documents"],
                references=query_result["references"],
                model=f"groq - {self.llm_model}",
                response_time=end_time - start_time,
                timestamp=end_time,
                confidence=None,
            )
            
            logger.info(f"Generated response for query in {(end_time - start_time).total_seconds():.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            raise
    
    def retrieve(self, query: str) -> QueryResult:
        """Retrieve relevant documents for a query
        
        Args:
            query: User query
            
        Returns:
            QueryResult with documents and references
        """
        start_time = datetime.now()
        conn = self._get_connection()
        
        # Build semantic search parameters
        # Note: max_documents here is used as fallback if abstract_k/passage_k not provided
        # The actual k values are passed separately to retrieve_similar_documents
        semantic_search_abstract = SemanticSearch(
            query=query,
            table=self.table_abstract,
            similarity_metric="<#>",
            embedding_model=self.embedding_model_name,
            max_documents=self.abstract_retrieval_k,
        )
        
        semantic_search_article = SemanticSearch(
            query=query,
            table=self.table_article,
            similarity_metric="<#>",
            embedding_model=self.embedding_model_name,
            max_documents=self.passage_retrieval_k,
        )
        
        semantic_search_hierarchy = [
            semantic_search_abstract,
            semantic_search_article,
        ]
        
        # Retrieve documents with 2-step pipeline
        relevant_documents = retrieve_similar_documents(
            conn=conn,
            retrieval_method=self.retrieval_method,
            retrieval_parameters=semantic_search_hierarchy,
            abstract_k=self.abstract_retrieval_k,
            passage_k=self.passage_retrieval_k,
            final_k=self.final_k,
        )
        
        end_time = datetime.now()
        
        result = QueryResult(
            query=query,
            documents=relevant_documents["documents"],
            references=relevant_documents["references"],
            similarity_scores=None,
            retrieval_time=end_time - start_time,
        )
        
        logger.info(f"Retrieved {len(result['documents'])} documents in {(end_time - start_time).total_seconds():.2f}s")
        return result

