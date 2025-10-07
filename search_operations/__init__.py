"""
Search Operations Module

This module provides advanced techniques for optimizing Milvus search operations:
- Search parameter tuning (nprobe, ef, etc.)
- Metric type selection and configuration
- Result reranking strategies
- Hybrid search weight optimization
- Performance benchmarking tools
- Search accuracy evaluation
- Query plan optimization

Implements production-tested strategies for balancing search speed,
accuracy, and resource utilization in enterprise environments.
"""

# Core exports
from .core import (
    SearchManager,
    SearchResult,
    BaseSearch,
    SearchError,
    InvalidSearchParametersError,
    EmbeddingGenerationError,
    SearchTimeoutError,
    ReRankingError,
    HybridSearchError,
    FusionError,
    EmptyResultError,
)

# Configuration exports
from .config import (
    SearchType,
    MetricType,
    ReRankingMethod,
    FusionMethod,
    BaseSearchConfig,
    SemanticSearchConfig,
    HybridSearchConfig,
    FusionSearchConfig,
    ReRankingConfig,
    SearchParams,
)

# Provider exports
from .providers import (
    EmbeddingProvider,
    EmbeddingResult,
    GeminiEmbeddingProvider,
    DimensionMismatchError,
    TaskType,
)

# Search implementations exports
from .search import (
    SemanticSearch,
    SemanticSearchWithReRanking,
    HybridSearch,
    HybridSearchWithReRanking,
)

# Reranking exports
from .reranking import (
    MilvusReRanker,
    MilvusReRankingMethod,
)

__all__ = [
    # Core
    "SearchManager",
    "SearchResult",
    "BaseSearch",
    
    # Exceptions
    "SearchError",
    "InvalidSearchParametersError",
    "EmbeddingGenerationError",
    "SearchTimeoutError",
    "ReRankingError",
    "HybridSearchError",
    "FusionError",
    "EmptyResultError",
    
    # Configuration
    "SearchType",
    "MetricType",
    "ReRankingMethod",
    "FusionMethod",
    "BaseSearchConfig",
    "SemanticSearchConfig",
    "HybridSearchConfig",
    "FusionSearchConfig",
    "ReRankingConfig",
    "SearchParams",
    
    # Providers
    "EmbeddingProvider",
    "EmbeddingResult",
    "GeminiEmbeddingProvider",
    "DimensionMismatchError",
    "TaskType",
    
    # Search implementations
    "SemanticSearch",
    "SemanticSearchWithReRanking",
    "HybridSearch",
    "HybridSearchWithReRanking",
    
    # Reranking
    "MilvusReRanker",
    "MilvusReRankingMethod",
]
