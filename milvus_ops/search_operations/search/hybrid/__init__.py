"""
Hybrid Search Module

This module provides production-grade hybrid search implementations combining
dense and sparse vector search with comprehensive fault tolerance, metrics,
and observability features.
"""

from .core.bm25 import BM25SparseVectorGenerator
from .core.engine import HybridSearch
from .core.fusion import fuse_results_rrf, fuse_results_weighted
from .resilience.circuit_breaker import CircuitBreaker
from .utils.config import BM25Config, HybridSearchMode, RetryConfig
from .utils.metrics import HybridSearchMetrics, SearchStatus

__all__ = [
    "HybridSearch",
    "BM25SparseVectorGenerator",
    "fuse_results_rrf",
    "fuse_results_weighted",
    "HybridSearchMetrics",
    "SearchStatus",
    "HybridSearchMode",
    "BM25Config",
    "RetryConfig",
    "CircuitBreaker",
]
