"""Response data models"""

from typing import List, Optional, TypedDict
from datetime import datetime, timedelta


class RAGResponse(TypedDict):
    """RAG system response"""
    answer: str
    query: str
    documents_used: List[str]
    references: List[str]
    model: str
    response_time: timedelta
    timestamp: datetime
    confidence: Optional[float]


