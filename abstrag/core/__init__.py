"""Core business logic modules"""

from abstrag.core.database import (
    open_db_connection,
    check_connection_health,
    validate_table_name,
    create_embedding_table,
    create_user_feedback_table,
    insert_embedding_data,
    insert_user_feedback,
    get_article_id_data,
    semantic_search_postgres,
    keyword_search_postgres,
    PostgresParams,
    SemanticSearch,
    TextSearch,
    UserFeedback,
)

from abstrag.core.embedding import (
    chunk_document,
    document_embedding,
    get_embedding_model,
    ChunkParams,
    Embedding,
    PaperEmbedding,
)

from abstrag.core.retrieval import (
    retrieve_similar_documents,
    RelevantDocuments,
    RetrievalMethod,
)

from abstrag.core.llm import (
    llm_chat_completion,
    groq_chat_completion,
    build_rag_prompt,
    build_retrieval_evaluation_prompt,
    GroqParams,
    LLMResponse,
)

__all__ = [
    # Database
    "open_db_connection",
    "check_connection_health",
    "validate_table_name",
    "create_embedding_table",
    "create_user_feedback_table",
    "insert_embedding_data",
    "insert_user_feedback",
    "get_article_id_data",
    "semantic_search_postgres",
    "keyword_search_postgres",
    "PostgresParams",
    "SemanticSearch",
    "TextSearch",
    "UserFeedback",
    # Embedding
    "chunk_document",
    "document_embedding",
    "get_embedding_model",
    "ChunkParams",
    "Embedding",
    "PaperEmbedding",
    # Retrieval
    "retrieve_similar_documents",
    "RelevantDocuments",
    "RetrievalMethod",
    # LLM
    "llm_chat_completion",
    "groq_chat_completion",
    "build_rag_prompt",
    "build_retrieval_evaluation_prompt",
    "GroqParams",
    "LLMResponse",
]


