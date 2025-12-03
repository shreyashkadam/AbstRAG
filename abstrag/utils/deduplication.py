"""Chunk deduplication utilities"""

import logging
from typing import List, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from abstrag.core.embedding import PaperEmbedding

logger = logging.getLogger(__name__)


def deduplicate_chunks(
    chunks: List[str],
    similarity_threshold: float = 0.95,
    embedding_model_name: str = "multi-qa-mpnet-base-dot-v1",
) -> List[str]:
    """Remove duplicate or highly similar chunks
    
    Args:
        chunks: List of text chunks
        similarity_threshold: Cosine similarity threshold (0-1) above which chunks are considered duplicates
        embedding_model_name: Embedding model to use for similarity calculation
        
    Returns:
        Deduplicated list of chunks
    """
    if len(chunks) <= 1:
        return chunks
    
    logger.info(f"Deduplicating {len(chunks)} chunks with threshold {similarity_threshold}")
    
    # Import here to avoid circular import
    from abstrag.core.embedding import get_embedding_model
    
    # Get embedding model
    embedding_model = get_embedding_model(embedding_model_name)
    
    # Generate embeddings for all chunks
    embeddings = embedding_model.encode(chunks, normalize_embeddings=True)
    
    # Track which chunks to keep
    keep_indices = []
    
    for i, embedding in enumerate(embeddings):
        is_duplicate = False
        
        # Check similarity with already kept chunks
        for kept_idx in keep_indices:
            similarity = np.dot(embedding, embeddings[kept_idx])
            if similarity >= similarity_threshold:
                is_duplicate = True
                logger.debug(f"Chunk {i} is duplicate of chunk {kept_idx} (similarity: {similarity:.3f})")
                break
        
        if not is_duplicate:
            keep_indices.append(i)
    
    deduplicated_chunks = [chunks[i] for i in keep_indices]
    
    logger.info(
        f"Deduplication complete: {len(chunks)} -> {len(deduplicated_chunks)} chunks "
        f"({len(chunks) - len(deduplicated_chunks)} removed)"
    )
    
    return deduplicated_chunks


def deduplicate_paper_embeddings(
    paper_embeddings: List["PaperEmbedding"],
    similarity_threshold: float = 0.95,
) -> List["PaperEmbedding"]:
    """Deduplicate paper embeddings based on content similarity
    
    Args:
        paper_embeddings: List of paper embeddings
        similarity_threshold: Similarity threshold for deduplication
        
    Returns:
        Deduplicated list of embeddings
    """
    if len(paper_embeddings) <= 1:
        return paper_embeddings
    
    logger.info(f"Deduplicating {len(paper_embeddings)} paper embeddings")
    
    # Extract embeddings as numpy array
    embeddings_array = np.array([pe["embeddings"] for pe in paper_embeddings])
    
    # Track which to keep
    keep_indices = []
    
    for i, embedding in enumerate(embeddings_array):
        is_duplicate = False
        
        for kept_idx in keep_indices:
            similarity = np.dot(embedding, embeddings_array[kept_idx])
            if similarity >= similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            keep_indices.append(i)
    
    deduplicated = [paper_embeddings[i] for i in keep_indices]
    
    logger.info(
        f"Deduplication complete: {len(paper_embeddings)} -> {len(deduplicated)} embeddings"
    )
    
    return deduplicated

