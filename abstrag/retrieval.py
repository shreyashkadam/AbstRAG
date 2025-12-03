"""Backward compatibility - imports from core.retrieval"""

from abstrag.core.retrieval import (
    retrieve_similar_documents,
    RelevantDocuments,
    RetrievalMethod,
    pg_semantic_retrieval_hierarchical,
    pg_semantic_retrieval,
    pg_text_retrieval,
)

__all__ = [
    "retrieve_similar_documents",
    "RelevantDocuments",
    "RetrievalMethod",
    "pg_semantic_retrieval_hierarchical",
    "pg_semantic_retrieval",
    "pg_text_retrieval",
]
