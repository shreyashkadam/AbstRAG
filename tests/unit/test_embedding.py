"""Unit tests for embedding module"""

import pytest
import numpy as np
from abstrag.core.embedding import (
    chunk_document,
    ChunkParams,
    get_embedding_model,
)


class TestChunking:
    """Test document chunking"""
    
    def test_chunk_document_markdown(self, sample_chunks):
        """Test chunking a markdown document"""
        document = "\n\n".join(sample_chunks)
        chunk_params = ChunkParams(
            method="MarkdownTextSplitter",
            size=100,
            overlap=20,
        )
        chunks = chunk_document(document, chunk_params)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_chunk_document_invalid_method(self):
        """Test chunking with invalid method"""
        chunk_params = ChunkParams(
            method="InvalidMethod",  # type: ignore
            size=100,
            overlap=20,
        )
        with pytest.raises(ValueError):
            chunk_document("test document", chunk_params)


class TestEmbeddingModel:
    """Test embedding model loading"""
    
    def test_get_embedding_model_caching(self):
        """Test that embedding models are cached"""
        model1 = get_embedding_model("multi-qa-mpnet-base-dot-v1")
        model2 = get_embedding_model("multi-qa-mpnet-base-dot-v1")
        # Should be the same instance due to caching
        assert model1 is model2


