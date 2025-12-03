"""Feedback service for managing user feedback"""

import logging
from typing import Optional
import psycopg
from datetime import datetime, timedelta

from abstrag.core.database import (
    open_db_connection,
    PostgresParams,
    insert_user_feedback,
    UserFeedback,
)

logger = logging.getLogger(__name__)


class FeedbackService:
    """Service for managing user feedback"""
    
    def __init__(self, db_connection_params: PostgresParams):
        """Initialize feedback service
        
        Args:
            db_connection_params: Database connection parameters
        """
        self.db_params = db_connection_params
    
    def submit_feedback(
        self,
        user_id: str,
        question: str,
        answer: str,
        thumbs: Optional[int] = None,
        documents_retrieved: Optional[str] = None,
        elapsed_time: Optional[timedelta] = None,
        conn: Optional[psycopg.Connection] = None,
    ) -> None:
        """Submit user feedback
        
        Args:
            user_id: User identifier
            question: User's question
            answer: System's answer
            thumbs: Thumbs up (1) or down (-1)
            documents_retrieved: Retrieved document IDs
            elapsed_time: Response time
            conn: Optional database connection
        """
        if conn is None:
            conn = open_db_connection(self.db_params, autocommit=True)
            if conn is None:
                raise ConnectionError("Failed to connect to database")
        
        feedback = UserFeedback(
            user_id=user_id,
            question=question,
            answer=answer,
            thumbs=thumbs,
            documents_retrieved=documents_retrieved,
            similarity=None,
            relevance=None,
            llm_model=None,
            embedding_model=None,
            elapsed_time=elapsed_time,
            feedback_timestamp=datetime.now(),
        )
        
        insert_user_feedback(conn=conn, feedback=feedback)
        logger.info(f"Feedback submitted by user {user_id}")

