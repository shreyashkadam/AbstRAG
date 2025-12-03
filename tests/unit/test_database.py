"""Unit tests for database module"""

import pytest
from abstrag.core.database import (
    validate_table_name,
    PostgresParams,
    SemanticSearch,
    TextSearch,
)


class TestTableValidation:
    """Test table name validation"""
    
    def test_valid_table_name_abstract(self):
        """Test valid abstract table name"""
        assert validate_table_name("embedding_abstract_multi_qa_mpnet_base_dot_v1") is True
    
    def test_valid_table_name_article(self):
        """Test valid article table name"""
        assert validate_table_name("embedding_article_multi_qa_mpnet_base_dot_v1") is True
    
    def test_invalid_table_name_wrong_prefix(self):
        """Test invalid table name with wrong prefix"""
        assert validate_table_name("invalid_table_name") is False
    
    def test_invalid_table_name_sql_injection(self):
        """Test SQL injection attempt"""
        assert validate_table_name("embedding_abstract_test'; DROP TABLE users; --") is False
    
    def test_invalid_table_name_wrong_type(self):
        """Test invalid table name with wrong type"""
        assert validate_table_name("embedding_invalid_multi_qa_mpnet_base_dot_v1") is False


class TestSemanticSearch:
    """Test SemanticSearch TypedDict"""
    
    def test_semantic_search_creation(self):
        """Test creating SemanticSearch dict"""
        search = SemanticSearch(
            query="test query",
            table="embedding_abstract_test",
            similarity_metric="<#>",
            embedding_model="test-model",
            max_documents=5,
        )
        assert search["query"] == "test query"
        assert search["max_documents"] == 5


