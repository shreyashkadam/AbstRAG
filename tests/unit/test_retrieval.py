"""Unit tests for retrieval module"""

import pytest
from abstrag.core.retrieval import (
    retrieve_similar_documents,
    RelevantDocuments,
    RetrievalMethod,
)


class TestRetrievalMethods:
    """Test retrieval methods"""
    
    def test_invalid_retrieval_method(self, mock_db_connection):
        """Test invalid retrieval method"""
        with pytest.raises(ValueError):
            retrieve_similar_documents(
                retrieval_method="invalid_method",  # type: ignore
                retrieval_parameters=[],
                conn=mock_db_connection,
            )
    
    def test_missing_connection(self):
        """Test retrieval without connection"""
        with pytest.raises(ValueError):
            retrieve_similar_documents(
                retrieval_method="pg_semantic_abstract+article",
                retrieval_parameters=[],
                conn=None,
            )


