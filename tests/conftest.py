"""Pytest configuration and fixtures"""

import pytest
import os
from unittest.mock import Mock, MagicMock
from dotenv import load_dotenv

# Load test environment variables
load_dotenv(".env")


@pytest.fixture
def mock_db_connection():
    """Mock database connection"""
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.execute.return_value = None
    conn.cursor.return_value.__enter__.return_value.fetchall.return_value = []
    conn.cursor.return_value.__enter__.return_value.fetchone.return_value = ("PostgreSQL 15.0",)
    return conn


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model"""
    model = MagicMock()
    model.encode.return_value = [[0.1] * 768]
    model.get_sentence_embedding_dimension.return_value = 768
    return model


@pytest.fixture
def sample_paper_metadata():
    """Sample paper metadata for testing"""
    return {
        "id": "http://arxiv.org/abs/2024.12345v1",
        "summary": "This is a test paper about quantitative finance.",
        "authors": ["John Doe", "Jane Smith"],
        "entry_url": "http://arxiv.org/abs/2024.12345v1",
        "published": "2024-01-01T00:00:00Z",
        "primary_category": "q-fin.CP",
        "categories": ["q-fin.CP", "q-fin.GN"],
    }


@pytest.fixture
def sample_chunks():
    """Sample document chunks for testing"""
    return [
        "This is the first chunk of a document about finance.",
        "This is the second chunk discussing quantitative methods.",
        "This is the third chunk with mathematical models.",
    ]


@pytest.fixture
def db_connection_params():
    """Database connection parameters for testing"""
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "pwd": os.getenv("POSTGRES_PWD", "password"),
        "database": os.getenv("POSTGRES_DB", "abstrag_test_db"),
    }


