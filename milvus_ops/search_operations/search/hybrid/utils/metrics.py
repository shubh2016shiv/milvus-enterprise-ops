"""
Metrics Module

This module provides metrics tracking for hybrid search operations,
including status enumerations and comprehensive metrics dataclasses.
"""

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class SearchStatus(Enum):
    """
    Enumeration of search operation states.
    
    Used to track the final status of search operations for monitoring
    and observability purposes.
    """
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    PARTIAL = "partial"
    RETRYING = "retrying"
    DEGRADED = "degraded"


@dataclass
class HybridSearchMetrics:
    """
    Comprehensive metrics for hybrid search operations.
    
    This dataclass tracks detailed timing information, result counts,
    retry attempts, and status for each search operation, enabling
    thorough monitoring and analysis.
    
    Attributes:
        query_hash: Hash of the query for identification
        embedding_time_ms: Time taken to generate dense embedding
        sparse_generation_time_ms: Time taken to generate sparse vector
        search_time_ms: Time taken for search execution
        fusion_time_ms: Time taken for result fusion
        total_time_ms: Total end-to-end time
        results_count: Number of results returned
        retry_count: Number of retry attempts made
        status: Final status of the search operation
        error_message: Error message if search failed
        cache_hit: Whether embedding was retrieved from cache
        collection_name: Name of the collection searched
        search_mode: Mode of hybrid search used
        timestamp: Unix timestamp when search was initiated
        dense_results: Number of results from dense vector search
        sparse_results: Number of results from sparse vector search
        keyword_results: Number of results from keyword search
    """
    query_hash: str
    embedding_time_ms: float = 0.0
    sparse_generation_time_ms: float = 0.0
    search_time_ms: float = 0.0
    fusion_time_ms: float = 0.0
    total_time_ms: float = 0.0
    results_count: int = 0
    retry_count: int = 0
    status: SearchStatus = SearchStatus.SUCCESS
    error_message: Optional[str] = None
    cache_hit: bool = False
    collection_name: str = ""
    search_mode: str = "vector_only"
    timestamp: float = field(default_factory=time.time)
    dense_results: int = 0
    sparse_results: int = 0
    keyword_results: int = 0
    
    def to_dict(self):
        """
        Convert metrics to dictionary format.
        
        Returns:
            Dictionary representation of metrics
        """
        return {
            "query_hash": self.query_hash,
            "embedding_time_ms": round(self.embedding_time_ms, 2),
            "sparse_generation_time_ms": round(self.sparse_generation_time_ms, 2),
            "search_time_ms": round(self.search_time_ms, 2),
            "fusion_time_ms": round(self.fusion_time_ms, 2),
            "total_time_ms": round(self.total_time_ms, 2),
            "results_count": self.results_count,
            "retry_count": self.retry_count,
            "status": self.status.value,
            "error_message": self.error_message,
            "cache_hit": self.cache_hit,
            "collection_name": self.collection_name,
            "search_mode": self.search_mode,
            "timestamp": self.timestamp,
            "dense_results": self.dense_results,
            "sparse_results": self.sparse_results,
            "keyword_results": self.keyword_results,
        }

