"""Utility modules"""

from abstrag.utils.helpers import get_user_agent, normalize_vector
from abstrag.utils.cache import QueryCache
from abstrag.utils.rate_limiter import RateLimiter
from abstrag.utils.prompts import PromptTemplate, get_template_manager
from abstrag.utils.deduplication import deduplicate_chunks, deduplicate_paper_embeddings
from abstrag.utils.metrics import MetricsCollector, get_metrics_collector
from abstrag.utils.logging_config import setup_logging, JSONFormatter
from abstrag.utils.token_counter import count_tokens, get_model_token_limit, truncate_context
from abstrag.utils.path_resolver import (
    find_file,
    find_evaluation_questions_file,
    find_env_file,
    find_config_file,
)

__all__ = [
    # Legacy
    "get_user_agent",
    "normalize_vector",
    # New utilities
    "QueryCache",
    "RateLimiter",
    "PromptTemplate",
    "get_template_manager",
    "deduplicate_chunks",
    "deduplicate_paper_embeddings",
    "MetricsCollector",
    "get_metrics_collector",
    "setup_logging",
    "JSONFormatter",
    "count_tokens",
    "get_model_token_limit",
    "truncate_context",
    # Path resolution
    "find_file",
    "find_evaluation_questions_file",
    "find_env_file",
    "find_config_file",
]
