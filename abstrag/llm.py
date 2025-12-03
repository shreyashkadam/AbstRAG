"""Backward compatibility - imports from core.llm"""

from abstrag.core.llm import (
    llm_chat_completion,
    groq_chat_completion,
    build_rag_prompt,
    build_retrieval_evaluation_prompt,
    GroqParams,
    LLMResponse,
    GroqModels,
    LLM,
    LLMParameters,
)

__all__ = [
    "llm_chat_completion",
    "groq_chat_completion",
    "build_rag_prompt",
    "build_retrieval_evaluation_prompt",
    "GroqParams",
    "LLMResponse",
    "GroqModels",
    "LLM",
    "LLMParameters",
]
