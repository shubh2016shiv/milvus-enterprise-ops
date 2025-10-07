"""
Core Search Operations Module

This module provides the core infrastructure for search operations,
including base classes, exceptions, and the main search manager.
"""

from .base import BaseSearch, SearchResult
from .exceptions import (
    SearchError,
    InvalidSearchParametersError,
    EmbeddingGenerationError,
    SearchTimeoutError,
    ReRankingError,
    HybridSearchError,
    FusionError,
    EmptyResultError
)
from .manager import SearchManager

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


