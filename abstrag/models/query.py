"""Query data models"""

from typing import List, Optional, TypedDict
from datetime import datetime, timedelta


class Query(TypedDict):
    """User query model"""
    text: str
    timestamp: datetime
    user_id: Optional[str]


class QueryResult(TypedDict):
    """Query result with metadata"""
    query: str
    documents: List[str]
    references: List[str]
    similarity_scores: Optional[List[float]]
    retrieval_time: Optional[timedelta]


