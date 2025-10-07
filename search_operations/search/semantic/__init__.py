"""
Semantic Search Module

This module provides production-grade semantic (dense vector) search implementations
with comprehensive fault tolerance, observability, and optimization capabilities.

Main Components:
- SemanticSearch: Core semantic search engine
- Metrics: Comprehensive metrics collection and monitoring
- Resilience: Circuit breaker and retry logic
- Validation: Parameter validation and query sanitization
- Optimization: Query optimization and parameter tuning
"""

from .engine import SemanticSearch
from .metrics import SearchMetrics, SearchStatus, MetricsCollector
from .resilience import (
    RetryConfig,
    CircuitBreaker,
    RetryHandler,
    ResilienceManager,
    CircuitState
)
from .validation import SearchValidator, QuerySanitizer
from .optimization import QueryOptimizer, SearchParamsBuilder

__all__ = [
    # Core search implementations
    "SemanticSearch",
    "SemanticSearchWithReRanking",
    
    # Metrics
    "SearchMetrics",
    "SearchStatus",
    "MetricsCollector",
    
    # Resilience
    "RetryConfig",
    "CircuitBreaker",
    "RetryHandler",
    "ResilienceManager",
    "CircuitState",
    
    # Validation
    "SearchValidator",
    "QuerySanitizer",
    
    # Optimization
    "QueryOptimizer",
    "SearchParamsBuilder",
]
