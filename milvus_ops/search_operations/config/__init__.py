"""
Search Configuration Module

This module provides configuration classes for all search types,
including enums, base configs, and validation models.
"""

from .base import (
    BaseSearchConfig,
    FusionMethod,
    MetricType,
    ReRankingMethod,
    SearchType,
)
from .fusion import FusionSearchConfig
from .hybrid import HybridSearchConfig
from .reranking import ReRankingConfig
from .semantic import SemanticSearchConfig
from .validation import SearchParams

__all__ = [
    # Enums
    "SearchType",
    "MetricType",
    "ReRankingMethod",
    "FusionMethod",
    # Base config
    "BaseSearchConfig",
    # Search configs
    "SemanticSearchConfig",
    "HybridSearchConfig",
    "FusionSearchConfig",
    "ReRankingConfig",
    # Validation
    "SearchParams",
]
