"""
Search Metrics Module

This module provides comprehensive metrics collection and monitoring
for semantic search operations, enabling observability and performance tracking.
"""

import time
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class SearchStatus(Enum):
    """Enumeration of search operation states"""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    PARTIAL = "partial"
    RETRYING = "retrying"


@dataclass
class SearchMetrics:
    """Comprehensive metrics for search operations"""
    query_hash: str
    collection_name: str
    embedding_time_ms: float = 0.0
    search_time_ms: float = 0.0
    total_time_ms: float = 0.0
    results_count: int = 0
    retry_count: int = 0
    status: SearchStatus = SearchStatus.SUCCESS
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    request_id: Optional[str] = None


class MetricsCollector:
    """
    Collects and manages search metrics for monitoring and analysis.
    
    Features:
    - Thread-safe metrics storage
    - Automatic metric aggregation
    - Configurable history retention
    - Metrics callback support
    """
    
    def __init__(
        self,
        max_history: int = 1000,
        metrics_callback: Optional[Callable[[SearchMetrics], None]] = None
    ):
        """
        Initialize the metrics collector.
        
        Args:
            max_history: Maximum number of metrics to retain
            metrics_callback: Optional callback for real-time metrics reporting
        """
        self._metrics_history: List[SearchMetrics] = []
        self._max_history = max_history
        self._metrics_callback = metrics_callback
        self._lock = asyncio.Lock()
    
    async def record_metric(self, metric: SearchMetrics) -> None:
        """
        Record a search metric.
        
        Args:
            metric: SearchMetrics object to record
        """
        async with self._lock:
            self._metrics_history.append(metric)
            
            # Maintain history size
            if len(self._metrics_history) > self._max_history:
                self._metrics_history = self._metrics_history[-self._max_history:]
        
        # Call metrics callback if provided
        if self._metrics_callback:
            try:
                self._metrics_callback(metric)
            except Exception:
                # Silently ignore callback failures to not impact search operations
                pass
    
    async def get_summary(self) -> Dict[str, Any]:
        """
        Get aggregated metrics summary.
        
        Returns:
            Dictionary containing metrics summary
        """
        async with self._lock:
            if not self._metrics_history:
                return {"message": "No metrics available"}
            
            total_searches = len(self._metrics_history)
            successful = sum(1 for m in self._metrics_history if m.status == SearchStatus.SUCCESS)
            failed = sum(1 for m in self._metrics_history if m.status == SearchStatus.FAILURE)
            timeouts = sum(1 for m in self._metrics_history if m.status == SearchStatus.TIMEOUT)
            
            # Calculate averages for successful searches
            successful_metrics = [m for m in self._metrics_history if m.status == SearchStatus.SUCCESS]
            
            if successful_metrics:
                avg_embedding_time = sum(m.embedding_time_ms for m in successful_metrics) / len(successful_metrics)
                avg_search_time = sum(m.search_time_ms for m in successful_metrics) / len(successful_metrics)
                avg_total_time = sum(m.total_time_ms for m in successful_metrics) / len(successful_metrics)
                avg_results = sum(m.results_count for m in successful_metrics) / len(successful_metrics)
            else:
                avg_embedding_time = 0.0
                avg_search_time = 0.0
                avg_total_time = 0.0
                avg_results = 0.0
            
            return {
                "total_searches": total_searches,
                "successful": successful,
                "failed": failed,
                "timeouts": timeouts,
                "success_rate": round(successful / total_searches, 4) if total_searches > 0 else 0,
                "avg_embedding_time_ms": round(avg_embedding_time, 2),
                "avg_search_time_ms": round(avg_search_time, 2),
                "avg_total_time_ms": round(avg_total_time, 2),
                "avg_results_count": round(avg_results, 2),
                "metrics_retained": len(self._metrics_history)
            }
    
    async def get_recent_metrics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent metrics.
        
        Args:
            limit: Number of recent metrics to return
            
        Returns:
            List of recent metrics as dictionaries
        """
        async with self._lock:
            recent = self._metrics_history[-limit:]
            return [
                {
                    "query_hash": m.query_hash,
                    "collection": m.collection_name,
                    "status": m.status.value,
                    "total_time_ms": round(m.total_time_ms, 2),
                    "results_count": m.results_count,
                    "timestamp": m.timestamp,
                    "request_id": m.request_id
                }
                for m in recent
            ]
    
    async def clear(self) -> None:
        """Clear all metrics history."""
        async with self._lock:
            self._metrics_history.clear()

