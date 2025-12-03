"""Unit tests for configuration module"""

import pytest
import tempfile
import yaml
from abstrag.config import load_config, IngestionConfig, RAGConfig, Config


class TestConfigLoading:
    """Test configuration loading"""
    
    def test_load_valid_config(self):
        """Test loading valid config file"""
        config = load_config("config.yaml")
        assert config is not None
        assert "ingestion" in config
        assert "rag" in config
    
    def test_load_nonexistent_config(self):
        """Test loading non-existent config"""
        config = load_config("nonexistent.yaml")
        assert config is None


class TestConfigValidation:
    """Test configuration validation"""
    
    def test_valid_ingestion_config(self):
        """Test valid ingestion config"""
        config = IngestionConfig(
            max_documents_arxiv=20,
            chunk_size=500,
            chunk_overlap=50,
            chunk_method="MarkdownTextSplitter",
            embedding_model_name="multi-qa-mpnet-base-dot-v1",
        )
        assert config.chunk_size == 500
    
    def test_invalid_chunk_overlap(self):
        """Test invalid chunk overlap (greater than chunk size)"""
        with pytest.raises(ValueError):
            IngestionConfig(
                max_documents_arxiv=20,
                chunk_size=500,
                chunk_overlap=600,  # Invalid: greater than chunk_size
                chunk_method="MarkdownTextSplitter",
                embedding_model_name="multi-qa-mpnet-base-dot-v1",
            )
    
    def test_invalid_chunk_size_range(self):
        """Test invalid chunk size (out of range)"""
        with pytest.raises(Exception):  # Pydantic validation error
            IngestionConfig(
                max_documents_arxiv=20,
                chunk_size=50,  # Invalid: less than 100
                chunk_overlap=25,
                chunk_method="MarkdownTextSplitter",
                embedding_model_name="multi-qa-mpnet-base-dot-v1",
            )


