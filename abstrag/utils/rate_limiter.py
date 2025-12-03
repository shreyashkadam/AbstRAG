"""Rate limiting utilities"""

import logging
import time
from typing import Dict, Optional
from collections import defaultdict
from threading import Lock

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, requests_per_minute: int = 60):
        """Initialize rate limiter
        
        Args:
            requests_per_minute: Maximum requests per minute
        """
        self.requests_per_minute = requests_per_minute
        self.tokens_per_second = requests_per_minute / 60.0
        self.bucket_size = requests_per_minute
        
        # Track tokens per key (e.g., per user or API)
        self._buckets: Dict[str, tuple[float, float]] = defaultdict(
            lambda: (self.bucket_size, time.time())
        )
        self._lock = Lock()
    
    def _refill_bucket(self, key: str) -> tuple[float, float]:
        """Refill token bucket for a key
        
        Args:
            key: Identifier for the bucket (e.g., user_id or API name)
            
        Returns:
            Tuple of (tokens, last_refill_time)
        """
        tokens, last_refill = self._buckets[key]
        now = time.time()
        elapsed = now - last_refill
        
        # Add tokens based on elapsed time
        tokens = min(
            self.bucket_size,
            tokens + elapsed * self.tokens_per_second
        )
        
        self._buckets[key] = (tokens, now)
        return tokens, now
    
    def acquire(self, key: str = "default", tokens: int = 1) -> bool:
        """Try to acquire tokens from the bucket
        
        Args:
            key: Identifier for the bucket
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens acquired, False if rate limited
        """
        with self._lock:
            current_tokens, _ = self._refill_bucket(key)
            
            if current_tokens >= tokens:
                self._buckets[key] = (current_tokens - tokens, time.time())
                logger.debug(f"Rate limit: Acquired {tokens} tokens for {key}, {current_tokens - tokens:.2f} remaining")
                return True
            else:
                logger.warning(f"Rate limit exceeded for {key}. Available: {current_tokens:.2f}, Required: {tokens}")
                return False
    
    def wait_if_needed(self, key: str = "default", tokens: int = 1) -> None:
        """Wait until tokens are available
        
        Args:
            key: Identifier for the bucket
            tokens: Number of tokens needed
        """
        while not self.acquire(key, tokens):
            # Calculate wait time
            current_tokens, last_refill = self._buckets[key]
            needed_tokens = tokens - current_tokens
            wait_time = needed_tokens / self.tokens_per_second
            
            logger.info(f"Rate limited. Waiting {wait_time:.2f} seconds for {key}")
            time.sleep(min(wait_time, 60.0))  # Cap wait at 60 seconds
    
    def reset(self, key: Optional[str] = None) -> None:
        """Reset rate limiter for a key or all keys
        
        Args:
            key: Key to reset, or None to reset all
        """
        with self._lock:
            if key is None:
                self._buckets.clear()
                logger.info("Reset all rate limiters")
            else:
                self._buckets[key] = (self.bucket_size, time.time())
                logger.info(f"Reset rate limiter for {key}")

