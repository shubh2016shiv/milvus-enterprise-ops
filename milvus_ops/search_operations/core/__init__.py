"""
Core Search Operations Module

This module provides the core infrastructure for search operations,
including base classes, exceptions, and the main search manager.
"""

from .base import BaseSearch, SearchResult
from .manager import SearchManager
from .search_ops_exceptions import (
    EmbeddingGenerationError,
    EmptyResultError,
    FusionError,
    HybridSearchError,
    InvalidSearchParametersError,
    ReRankingError,
    SearchError,
    SearchTimeoutError,
)

__all__ = [
    # Base classes
    "BaseSearch",
    "SearchResult",
    # Manager
    "SearchManager",
    # Exceptions
    "SearchError",
    "InvalidSearchParametersError",
    "EmbeddingGenerationError",
    "SearchTimeoutError",
    "ReRankingError",
    "HybridSearchError",
    "FusionError",
    "EmptyResultError",
]
