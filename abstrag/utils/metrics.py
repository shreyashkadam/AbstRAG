"""Metrics collection and monitoring"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect and track system metrics"""
    
    def __init__(self, max_history: int = 1000):
        """Initialize metrics collector
        
        Args:
            max_history: Maximum number of data points to keep in history
        """
        self.max_history = max_history
        self._lock = threading.Lock()
        
        # Metrics storage
        self.query_latencies: deque = deque(maxlen=max_history)
        self.retrieval_qualities: deque = deque(maxlen=max_history)
        self.token_usage: deque = deque(maxlen=max_history)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.api_calls: Dict[str, int] = defaultdict(int)
        
        logger.info("Initialized metrics collector")
    
    def record_query_latency(self, latency_seconds: float) -> None:
        """Record query processing latency
        
        Args:
            latency_seconds: Query latency in seconds
        """
        with self._lock:
            self.query_latencies.append({
                "latency": latency_seconds,
                "timestamp": datetime.now(),
            })
    
    def record_retrieval_quality(
        self,
        num_documents: int,
        num_papers: int,
        avg_similarity: Optional[float] = None,
    ) -> None:
        """Record retrieval quality metrics
        
        Args:
            num_documents: Number of documents retrieved
            num_papers: Number of unique papers
            avg_similarity: Average similarity score
        """
        with self._lock:
            self.retrieval_qualities.append({
                "num_documents": num_documents,
                "num_papers": num_papers,
                "avg_similarity": avg_similarity,
                "timestamp": datetime.now(),
            })
    
    def record_token_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
    ) -> None:
        """Record token usage
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            model: Model name
        """
        with self._lock:
            self.token_usage.append({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "model": model,
                "timestamp": datetime.now(),
            })
    
    def record_error(self, error_type: str) -> None:
        """Record an error occurrence
        
        Args:
            error_type: Type of error
        """
        with self._lock:
            self.error_counts[error_type] += 1
    
    def record_api_call(self, api_name: str) -> None:
        """Record an API call
        
        Args:
            api_name: Name of API (e.g., 'groq', 'arxiv')
        """
        with self._lock:
            self.api_calls[api_name] += 1
    
    def get_average_latency(self, window_minutes: int = 60) -> Optional[float]:
        """Get average query latency over time window
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Average latency or None if no data
        """
        with self._lock:
            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            recent_latencies = [
                m["latency"]
                for m in self.query_latencies
                if m["timestamp"] >= cutoff
            ]
            
            if not recent_latencies:
                return None
            
            return sum(recent_latencies) / len(recent_latencies)
    
    def get_error_rate(self, window_minutes: int = 60) -> float:
        """Get error rate over time window
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Error rate (errors per minute)
        """
        with self._lock:
            total_errors = sum(self.error_counts.values())
            return total_errors / window_minutes if window_minutes > 0 else 0.0
    
    def get_stats(self) -> Dict:
        """Get summary statistics
        
        Returns:
            Dictionary with summary stats
        """
        with self._lock:
            return {
                "total_queries": len(self.query_latencies),
                "average_latency": (
                    sum(m["latency"] for m in self.query_latencies) / len(self.query_latencies)
                    if self.query_latencies else 0
                ),
                "total_errors": sum(self.error_counts.values()),
                "error_breakdown": dict(self.error_counts),
                "api_calls": dict(self.api_calls),
                "total_token_usage": (
                    sum(m["total_tokens"] for m in self.token_usage)
                    if self.token_usage else 0
                ),
            }
    
    def reset(self) -> None:
        """Reset all metrics"""
        with self._lock:
            self.query_latencies.clear()
            self.retrieval_qualities.clear()
            self.token_usage.clear()
            self.error_counts.clear()
            self.api_calls.clear()
            logger.info("Metrics reset")


# Global metrics instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


