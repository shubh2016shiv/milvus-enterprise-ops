"""
Search Implementations Module

This module contains production-grade search implementations including
semantic and hybrid search strategies with comprehensive fault tolerance,
metrics, and observability features.
"""

# Import semantic search components
# Import hybrid search components
from .hybrid import (
    BM25Config,
    BM25SparseVectorGenerator,
    CircuitBreaker,
    HybridSearch,
    HybridSearchMetrics,
    HybridSearchMode,
    RetryConfig,
    SearchStatus,
    fuse_results_rrf,
    fuse_results_weighted,
)
from .semantic import SemanticSearch

__all__ = [
    # Semantic search
    "SemanticSearch",
    # Hybrid search
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
