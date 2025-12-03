"""Token counting and context window management"""

import logging
from typing import List, Optional
import tiktoken

logger = logging.getLogger(__name__)

# Model token limits (approximate)
MODEL_TOKEN_LIMITS = {
    "llama3-70b-8192": 8192,
    "llama-3.1-70b-versatile": 8192,
    "llama-3.3-70b-versatile": 8192,
    "gemma2-9b-it": 8192,
    "llama3-groq-70b-8192-tool-use-preview": 8192,
}

# Default encoding (cl100k_base used by GPT-4, works reasonably well for other models)
DEFAULT_ENCODING = "cl100k_base"


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """Count tokens in text
    
    Args:
        text: Text to count tokens for
        model: Model name (optional, for model-specific encoding)
        
    Returns:
        Number of tokens
    """
    try:
        # Try to get model-specific encoding if available
        encoding_name = DEFAULT_ENCODING
        if model:
            # Map models to encodings if needed
            # For now, use default encoding
            pass
        
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens: {e}. Using approximate count.")
        # Fallback: approximate 4 characters per token
        return len(text) // 4


def get_model_token_limit(model: str) -> int:
    """Get token limit for a model
    
    Args:
        model: Model name
        
    Returns:
        Token limit
    """
    return MODEL_TOKEN_LIMITS.get(model, 4096)  # Default to 4096 if unknown


def truncate_context(
    context_chunks: List[str],
    max_tokens: int,
    query: str = "",
    reserve_tokens: int = 500,
) -> List[str]:
    """Truncate context to fit within token limit
    
    Args:
        context_chunks: List of context chunks
        max_tokens: Maximum tokens allowed
        query: User query (to reserve space for)
        reserve_tokens: Tokens to reserve for prompt template and response
        
    Returns:
        Truncated list of chunks
    """
    available_tokens = max_tokens - reserve_tokens - count_tokens(query)
    
    if available_tokens <= 0:
        logger.warning("No tokens available for context after reserving for query and response")
        return []
    
    truncated_chunks = []
    current_tokens = 0
    
    # Add chunks until we exceed the limit
    for chunk in context_chunks:
        chunk_tokens = count_tokens(chunk)
        if current_tokens + chunk_tokens <= available_tokens:
            truncated_chunks.append(chunk)
            current_tokens += chunk_tokens
        else:
            # Try to fit partial chunk if there's space
            remaining_tokens = available_tokens - current_tokens
            if remaining_tokens > 100:  # Only if meaningful space remains
                # Truncate chunk to fit
                encoding = tiktoken.get_encoding(DEFAULT_ENCODING)
                tokens = encoding.encode(chunk)
                truncated_tokens = tokens[:remaining_tokens]
                truncated_text = encoding.decode(truncated_tokens)
                truncated_chunks.append(truncated_text + "...")
            break
    
    logger.info(
        f"Truncated context: {len(context_chunks)} -> {len(truncated_chunks)} chunks, "
        f"{current_tokens}/{available_tokens} tokens used"
    )
    
    return truncated_chunks


