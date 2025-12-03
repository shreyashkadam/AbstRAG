"""Information Retrieval evaluation metrics"""

from typing import List, Set, Optional
import numpy as np


def precision_at_k(relevant_docs: Set[str], retrieved_docs: List[str], k: int) -> float:
    """Calculate Precision@k
    
    Args:
        relevant_docs: Set of relevant document IDs
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        k: Number of top documents to consider
        
    Returns:
        Precision@k score (0.0 to 1.0)
    """
    if k == 0:
        return 0.0
    
    # Get top k retrieved documents
    top_k = retrieved_docs[:k]
    
    # Count how many are relevant
    relevant_count = sum(1 for doc_id in top_k if doc_id in relevant_docs)
    
    return relevant_count / k


def recall_at_k(relevant_docs: Set[str], retrieved_docs: List[str], k: int) -> float:
    """Calculate Recall@k
    
    Args:
        relevant_docs: Set of relevant document IDs
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        k: Number of top documents to consider
        
    Returns:
        Recall@k score (0.0 to 1.0)
    """
    if len(relevant_docs) == 0:
        return 0.0
    
    # Get top k retrieved documents
    top_k = retrieved_docs[:k]
    
    # Count how many relevant docs were retrieved
    retrieved_relevant = sum(1 for doc_id in top_k if doc_id in relevant_docs)
    
    return retrieved_relevant / len(relevant_docs)


def dcg_at_k(relevant_docs: Set[str], retrieved_docs: List[str], k: int, 
              scores: Optional[List[float]] = None) -> float:
    """Calculate Discounted Cumulative Gain at k
    
    Args:
        relevant_docs: Set of relevant document IDs
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        k: Number of top documents to consider
        scores: Optional list of relevance scores (if None, binary relevance is used)
        
    Returns:
        DCG@k score
    """
    if k == 0:
        return 0.0
    
    dcg = 0.0
    top_k = retrieved_docs[:k]
    
    for i, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_docs:
            if scores is not None and i <= len(scores):
                rel = scores[i - 1]
            else:
                rel = 1.0  # Binary relevance
            
            dcg += rel / np.log2(i + 1)
    
    return dcg


def idcg_at_k(relevant_docs: Set[str], k: int, 
               scores: Optional[List[float]] = None) -> float:
    """Calculate Ideal DCG@k (DCG for perfect ranking)
    
    Args:
        relevant_docs: Set of relevant document IDs
        k: Number of top documents to consider
        scores: Optional list of relevance scores (if None, binary relevance is used)
        
    Returns:
        IDCG@k score
    """
    if k == 0 or len(relevant_docs) == 0:
        return 0.0
    
    # For ideal ranking, all relevant docs come first
    # Use binary relevance or provided scores
    if scores is not None:
        # Sort scores descending and take top k
        ideal_scores = sorted(scores, reverse=True)[:k]
        ideal_scores = ideal_scores[:min(k, len(relevant_docs))]
    else:
        # Binary relevance: all relevant docs have score 1.0
        num_relevant = min(k, len(relevant_docs))
        ideal_scores = [1.0] * num_relevant
    
    idcg = 0.0
    for i, rel in enumerate(ideal_scores, start=1):
        idcg += rel / np.log2(i + 1)
    
    return idcg


def ndcg_at_k(relevant_docs: Set[str], retrieved_docs: List[str], k: int,
               scores: Optional[List[float]] = None) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k
    
    Args:
        relevant_docs: Set of relevant document IDs
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        k: Number of top documents to consider
        scores: Optional list of relevance scores (if None, binary relevance is used)
        
    Returns:
        nDCG@k score (0.0 to 1.0)
    """
    dcg = dcg_at_k(relevant_docs, retrieved_docs, k, scores)
    idcg = idcg_at_k(relevant_docs, k, scores)
    
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg


def hit_rate(relevant_docs: Set[str], retrieved_docs: List[str]) -> bool:
    """Calculate Hit Rate (whether at least one relevant doc is retrieved)
    
    Args:
        relevant_docs: Set of relevant document IDs
        retrieved_docs: List of retrieved document IDs
        
    Returns:
        True if at least one relevant doc is retrieved, False otherwise
    """
    return any(doc_id in relevant_docs for doc_id in retrieved_docs)


def mean_reciprocal_rank(relevant_docs: Set[str], retrieved_docs: List[str]) -> float:
    """Calculate Mean Reciprocal Rank (MRR)
    
    Args:
        relevant_docs: Set of relevant document IDs
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        
    Returns:
        Reciprocal rank of first relevant document (0.0 if none found)
    """
    for i, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant_docs:
            return 1.0 / i
    
    return 0.0


def compute_all_metrics(relevant_docs: Set[str], retrieved_docs: List[str],
                        k_values: List[int] = [1, 3, 5, 10],
                        scores: Optional[List[float]] = None) -> dict:
    """Compute all IR metrics for a query
    
    Args:
        relevant_docs: Set of relevant document IDs
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        k_values: List of k values to compute metrics for
        scores: Optional list of relevance scores
        
    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}
    
    # Precision@k for each k
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(relevant_docs, retrieved_docs, k)
    
    # Recall@k for each k
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(relevant_docs, retrieved_docs, k)
    
    # nDCG@k for each k
    for k in k_values:
        metrics[f"ndcg@{k}"] = ndcg_at_k(relevant_docs, retrieved_docs, k, scores)
    
    # Hit rate and MRR
    metrics["hit_rate"] = hit_rate(relevant_docs, retrieved_docs)
    metrics["mrr"] = mean_reciprocal_rank(relevant_docs, retrieved_docs)
    
    return metrics


