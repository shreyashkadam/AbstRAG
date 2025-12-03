"""Retrieve similar documents from database"""

from typing import List, Literal, TypedDict, Union, Optional, Any, get_args, TYPE_CHECKING
import psycopg
from abstrag.core.database import (
    SemanticSearch,
    semantic_search_postgres,
    keyword_search_postgres,
    TextSearch,
)

if TYPE_CHECKING:
    from abstrag.core.bm25_retrieval import BM25Index

RetrievalMethod = Literal[
    "pg_semantic_abstract+article",
    "pg_semantic_article",
    "pg_text_article",
    "pg_bm25_article",
]
RetrievalParameters = Union[SemanticSearch, TextSearch]


class RelevantDocuments(TypedDict):
    question: str
    documents: List[str]
    references: List[str]


def retrieve_similar_documents(
    retrieval_method: RetrievalMethod | str,
    retrieval_parameters: List[Any],
    conn: Optional[psycopg.Connection],
    abstract_k: Optional[int] = None,
    passage_k: Optional[int] = None,
    final_k: Optional[int] = None,
) -> RelevantDocuments:
    if retrieval_method == "pg_semantic_abstract+article":
        if isinstance(conn, psycopg.Connection):
            relevant_documents = pg_semantic_retrieval_hierarchical(
                conn=conn,
                retrieval_parameters=retrieval_parameters,
                abstract_k=abstract_k,
                passage_k=passage_k,
                final_k=final_k,
            )
        else:
            raise ValueError("Database connection not opened")
    elif retrieval_method == "pg_semantic_article":
        if isinstance(conn, psycopg.Connection):
            relevant_documents = pg_semantic_retrieval(
                conn=conn, retrieval_parameters=retrieval_parameters
            )
        else:
            raise ValueError("Database connection not opened")
    elif retrieval_method == "pg_text_article":
        if isinstance(conn, psycopg.Connection):
            relevant_documents = pg_text_retrieval(
                conn=conn, retrieval_parameters=retrieval_parameters
            )
        else:
            raise ValueError("Database connection not opened")
    elif retrieval_method == "pg_bm25_article":
        if isinstance(conn, psycopg.Connection):
            relevant_documents = pg_bm25_retrieval(
                conn=conn, retrieval_parameters=retrieval_parameters
            )
        else:
            raise ValueError("Database connection not opened")
    else:
        raise ValueError(f"Retrieval method {retrieval_method} not implemented")
    return relevant_documents


def pg_semantic_retrieval_hierarchical(
    conn: psycopg.Connection,
    retrieval_parameters: List[SemanticSearch],
    abstract_k: Optional[int] = None,
    passage_k: Optional[int] = None,
    final_k: Optional[int] = None,
) -> RelevantDocuments:
    """
    2-step retrieval pipeline:
    1. Retrieve candidate papers from abstracts (abstract_k papers)
    2. Retrieve passages from those papers (passage_k passages) and return top final_k by similarity
    
    Args:
        conn: Database connection
        retrieval_parameters: List containing [abstract_search, article_search] parameters
        abstract_k: Number of candidate papers to retrieve (default: from SemanticSearch max_documents)
        passage_k: Number of passages to retrieve (default: from SemanticSearch max_documents)
        final_k: Number of final passages to return (default: 5)
    """
    if conn:
        # Step 1: Semantic search on abstracts to get candidate papers
        semantic_search_abstract = retrieval_parameters[0]
        
        # Use abstract_k if provided, otherwise use max_documents from SemanticSearch
        abstract_max_docs = abstract_k if abstract_k is not None else semantic_search_abstract["max_documents"]
        
        # Temporarily override max_documents for abstract search
        original_abstract_max = semantic_search_abstract["max_documents"]
        semantic_search_abstract["max_documents"] = abstract_max_docs
        
        semantic_search_results_abstract, question_embedding = semantic_search_postgres(
            conn=conn,
            semantic_search_params=semantic_search_abstract,
        )
        
        # Restore original max_documents
        semantic_search_abstract["max_documents"] = original_abstract_max

        # Get IDs of candidate papers
        id_relevant_documents = [
            result[0] for result in semantic_search_results_abstract
        ]

        # Step 2: Semantic search on article passages filtered by candidate paper IDs
        semantic_search_article = retrieval_parameters[1]
        
        # Use passage_k if provided, otherwise use max_documents from SemanticSearch
        passage_max_docs = passage_k if passage_k is not None else semantic_search_article["max_documents"]
        
        # Temporarily override max_documents for article search
        original_article_max = semantic_search_article["max_documents"]
        semantic_search_article["max_documents"] = passage_max_docs
        
        semantic_search_results_articles, _ = semantic_search_postgres(
            conn=conn,
            semantic_search_params=semantic_search_article,
            filter_id=id_relevant_documents,
        )
        
        # Restore original max_documents
        semantic_search_article["max_documents"] = original_article_max

        # Extract passages and their paper IDs
        passage_documents = [document[1] for document in semantic_search_results_articles]
        passage_paper_ids = [document[0] for document in semantic_search_results_articles]

        # Return top final_k by similarity (already sorted by semantic_search_postgres)
        final_k_value = final_k if final_k is not None else 5
        final_documents = passage_documents[:final_k_value]
        final_paper_ids = passage_paper_ids[:final_k_value]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_paper_ids = []
        for pid in final_paper_ids:
            if pid not in seen:
                seen.add(pid)
                unique_paper_ids.append(pid)

        relevant_documents = RelevantDocuments(
            question=semantic_search_abstract["query"],
            documents=final_documents,
            references=unique_paper_ids,
        )
    else:
        raise ValueError("Database connection not opened")
    return relevant_documents


def pg_semantic_retrieval(
    conn: psycopg.Connection, retrieval_parameters: List[SemanticSearch]
) -> RelevantDocuments:
    if conn:
        # Semantic search on articles
        semantic_search_article = retrieval_parameters[0]
        semantic_search_results_article, question_embedding = semantic_search_postgres(
            conn=conn,
            semantic_search_params=semantic_search_article,
        )

        # Get ID of relevant documents
        id_relevant_documents = [
            result[0] for result in semantic_search_results_article
        ]

        # Prepare output
        final_documents = [document[1] for document in semantic_search_results_article]

        relevant_documents = RelevantDocuments(
            question=semantic_search_article["query"],
            documents=final_documents,
            references=id_relevant_documents,
        )
    else:
        raise ValueError("Database connection not opened")
    return relevant_documents


def pg_text_retrieval(
    conn: psycopg.Connection, retrieval_parameters: List[TextSearch]
) -> RelevantDocuments:
    if conn:
        # Text search on articles
        text_search_article = retrieval_parameters[0]
        text_search_results_article = keyword_search_postgres(
            conn=conn,
            text_search_params=text_search_article,
        )

        # Get ID of relevant documents
        id_relevant_documents = [result[0] for result in text_search_results_article]

        # Prepare output
        final_documents = [document[1] for document in text_search_results_article]

        relevant_documents = RelevantDocuments(
            question=text_search_article["query"],
            documents=final_documents,
            references=id_relevant_documents,
        )
    else:
        raise ValueError("Database connection not opened")
    return relevant_documents


def pg_bm25_retrieval(
    conn: psycopg.Connection, retrieval_parameters: List[TextSearch]
) -> RelevantDocuments:
    """BM25 retrieval baseline
    
    Args:
        conn: Database connection
        retrieval_parameters: List containing TextSearch parameters
        
    Returns:
        RelevantDocuments with retrieved documents
    """
    if conn:
        from abstrag.core.bm25_retrieval import bm25_search_postgres
        
        # BM25 search on articles
        text_search_article = retrieval_parameters[0]
        bm25_search_results = bm25_search_postgres(
            conn=conn,
            text_search_params=text_search_article,
            bm25_index=None,  # Will build index from database
        )

        # Get ID of relevant documents
        id_relevant_documents = [result[0] for result in bm25_search_results]

        # Prepare output
        final_documents = [document[1] for document in bm25_search_results]

        relevant_documents = RelevantDocuments(
            question=text_search_article["query"],
            documents=final_documents,
            references=id_relevant_documents,
        )
    else:
        raise ValueError("Database connection not opened")
    return relevant_documents

