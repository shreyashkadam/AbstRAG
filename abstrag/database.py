"""Backward compatibility - imports from core.database"""

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

__all__ = [
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
]
