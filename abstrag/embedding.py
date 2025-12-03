"""Backward compatibility - imports from core.embedding"""

from abstrag.core.embedding import (
    chunk_document,
    document_embedding,
    get_embedding_model,
    ChunkParams,
    Embedding,
    PaperEmbedding,
    ChunkMethod,
    EmbeddingModel,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_METHOD,
    EMBEDDING_MODEL_NAME,
)

__all__ = [
    "chunk_document",
    "document_embedding",
    "get_embedding_model",
    "ChunkParams",
    "Embedding",
    "PaperEmbedding",
    "ChunkMethod",
    "EmbeddingModel",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "CHUNK_METHOD",
    "EMBEDDING_MODEL_NAME",
]
