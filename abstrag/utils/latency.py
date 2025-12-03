"""Latency measurement utilities"""

import time
from typing import Dict, List, Optional
from contextlib import contextmanager
from datetime import datetime, timedelta
import statistics


class LatencyTimer:
    """Context manager for measuring latency"""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.duration is not None:
            return self.duration
        elif self.start_time is not None:
            return time.time() - self.start_time
        else:
            return 0.0


@contextmanager
def measure_latency():
    """Context manager for measuring latency
    
    Usage:
        with measure_latency() as timer:
            # code to measure
            pass
        latency = timer.elapsed()
    """
    timer = LatencyTimer()
    with timer:
        yield timer


def compute_latency_stats(latencies: List[float]) -> Dict[str, float]:
    """Compute statistical summary of latencies
    
    Args:
        latencies: List of latency measurements in seconds
        
    Returns:
        Dictionary with mean, median, std, p95, p99
    """
    if not latencies:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    
    return {
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "std": statistics.stdev(latencies) if n > 1 else 0.0,
        "min": min(latencies),
        "max": max(latencies),
        "p95": sorted_latencies[int(0.95 * n)] if n > 0 else 0.0,
        "p99": sorted_latencies[int(0.99 * n)] if n > 0 else 0.0,
    }


class LatencyTracker:
    """Track latencies for different components"""
    
    def __init__(self):
        self.retrieval_latencies: List[float] = []
        self.generation_latencies: List[float] = []
        self.total_latencies: List[float] = []
    
    def record_retrieval(self, latency: float):
        """Record retrieval latency"""
        self.retrieval_latencies.append(latency)
    
    def record_generation(self, latency: float):
        """Record generation latency"""
        self.generation_latencies.append(latency)
    
    def record_total(self, latency: float):
        """Record total latency"""
        self.total_latencies.append(latency)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistical summary for all components"""
        return {
            "retrieval": compute_latency_stats(self.retrieval_latencies),
            "generation": compute_latency_stats(self.generation_latencies),
            "total": compute_latency_stats(self.total_latencies),
        }
    
    def reset(self):
        """Reset all latency tracking"""
        self.retrieval_latencies.clear()
        self.generation_latencies.clear()
        self.total_latencies.clear()

