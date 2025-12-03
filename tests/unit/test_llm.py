"""Unit tests for LLM module"""

import pytest
from abstrag.core.llm import (
    build_rag_prompt,
    build_retrieval_evaluation_prompt,
    GroqParams,
)


class TestPromptBuilding:
    """Test prompt building functions"""
    
    def test_build_rag_prompt(self):
        """Test building RAG prompt"""
        question = "What is quantitative finance?"
        context = ["Document 1 about finance", "Document 2 about trading"]
        prompt = build_rag_prompt(question, context)
        
        assert question in prompt
        assert "Document 1" in prompt
        assert "Document 2" in prompt
        assert "CONTEXT" in prompt
        assert "QUESTION" in prompt
    
    def test_build_evaluation_prompt(self):
        """Test building evaluation prompt"""
        document = "Test document content"
        num_questions = 3
        prompt = build_retrieval_evaluation_prompt(document, num_questions)
        
        assert document in prompt
        assert str(num_questions) in prompt
        assert "quantitative finance" in prompt.lower()


