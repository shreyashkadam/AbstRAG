"""Query result caching"""

import logging
import hashlib
import json
from functools import lru_cache
from typing import Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class QueryCache:
    """Cache for query results"""
    
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 128):
        """Initialize query cache
        
        Args:
            ttl_seconds: Time to live for cache entries in seconds
            max_size: Maximum number of cache entries
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: dict[str, tuple[Any, datetime]] = {}
    
    def _generate_key(self, query: str, **kwargs) -> str:
        """Generate cache key from query and parameters
        
        Args:
            query: User query
            **kwargs: Additional parameters
            
        Returns:
            Cache key string
        """
        key_data = {"query": query, **kwargs}
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, query: str, **kwargs) -> Optional[Any]:
        """Get cached result
        
        Args:
            query: User query
            **kwargs: Additional parameters
            
        Returns:
            Cached result or None if not found/expired
        """
        key = self._generate_key(query, **kwargs)
        
        if key not in self._cache:
            return None
        
        result, timestamp = self._cache[key]
        
        # Check if expired
        if datetime.now() - timestamp > timedelta(seconds=self.ttl_seconds):
            del self._cache[key]
            logger.debug(f"Cache entry expired for query: {query[:50]}...")
            return None
        
        logger.debug(f"Cache hit for query: {query[:50]}...")
        return result
    
    def set(self, query: str, result: Any, **kwargs) -> None:
        """Set cache entry
        
        Args:
            query: User query
            result: Result to cache
            **kwargs: Additional parameters
        """
        key = self._generate_key(query, **kwargs)
        
        # Evict oldest if at max size
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
            logger.debug("Evicted oldest cache entry")
        
        self._cache[key] = (result, datetime.now())
        logger.debug(f"Cached result for query: {query[:50]}...")
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self._cache)


