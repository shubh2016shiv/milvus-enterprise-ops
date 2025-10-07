"""
Search Implementations Module

This module contains production-grade search implementations including
semantic and hybrid search strategies with comprehensive fault tolerance,
metrics, and observability features.
"""

# Import semantic search components
from .semantic import SemanticSearch, SemanticSearchWithReRanking

# Import hybrid search components
from .hybrid import (
    HybridSearch,
    BM25SparseVectorGenerator,
    fuse_results_rrf,
    fuse_results_weighted,
    HybridSearchMetrics,
    SearchStatus,
    HybridSearchMode,
    BM25Config,
    RetryConfig,
    CircuitBreaker,
)

__all__ = [
    # Semantic search
    "SemanticSearch",
    "SemanticSearchWithReRanking",
    
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


