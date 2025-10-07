"""
Semantic Search Operations

This module provides production-grade implementation for semantic (dense vector)
search operations, with comprehensive fault tolerance, observability, and optimization.
"""

import time
from typing import Optional, Callable, Dict, Any, List

from connection_management import ConnectionManager
from ...core.exceptions import SearchError, InvalidSearchParametersError
from ...config.semantic import SemanticSearchConfig
from ...providers.embedding import EmbeddingProvider
from ...core.base import BaseSearch, SearchResult

from .metrics import SearchMetrics, SearchStatus, MetricsCollector
from .resilience import ResilienceManager, RetryConfig
from .validation import SearchValidator, QuerySanitizer
from .optimization import QueryOptimizer, SearchParamsBuilder


class SemanticSearch(BaseSearch[SemanticSearchConfig]):
    """
    Production-grade implementation of semantic (dense vector) search.
    
    Features:
    - Automatic retry with exponential backoff
    - Circuit breaker for fault tolerance
    - Comprehensive metrics and observability
    - Request validation and sanitization
    - Query optimization
    - Graceful error handling
    
    This class implements enterprise-level semantic search with clear
    separation of concerns through dedicated modules for metrics,
    resilience, validation, and optimization.
    """
    
    def __init__(
        self,
        connection_manager: ConnectionManager,
        embedding_provider: EmbeddingProvider,
        enable_circuit_breaker: bool = True,
        enable_retry: bool = True,
        retry_config: Optional[RetryConfig] = None,
        enable_query_optimization: bool = True,
        metrics_callback: Optional[Callable[[SearchMetrics], None]] = None,
        max_metrics_history: int = 1000
    ):
        """
        Initialize semantic search with production features.
        
        Args:
            connection_manager: ConnectionManager for Milvus operations
            embedding_provider: Provider for generating embeddings
            enable_circuit_breaker: Enable circuit breaker pattern
            enable_retry: Enable automatic retry logic
            retry_config: Configuration for retry behavior
            enable_query_optimization: Enable query optimization
            metrics_callback: Optional callback for real-time metrics reporting
            max_metrics_history: Maximum metrics history to retain
        """
        super().__init__(connection_manager, embedding_provider)
        
        # Initialize resilience manager
        self.resilience_manager = ResilienceManager(
            enable_circuit_breaker=enable_circuit_breaker,
            enable_retry=enable_retry,
            retry_config=retry_config
        )
        
        # Initialize metrics collector
        self.metrics_collector = MetricsCollector(
            max_history=max_metrics_history,
            metrics_callback=metrics_callback
        )
        
        # Initialize optimizer
        self.query_optimizer = QueryOptimizer(
            enable_auto_tuning=enable_query_optimization
        )
        
        # Initialize validator and sanitizer
        self.validator = SearchValidator()
        self.sanitizer = QuerySanitizer()
    
    async def search(
        self,
        collection_name: str,
        query: str,
        config: SemanticSearchConfig,
        request_id: Optional[str] = None
    ) -> SearchResult:
        """
        Perform semantic search operation with full production features.
        
        This method handles the complete semantic search workflow:
        1. Parameter validation and sanitization
        2. Embedding generation
        3. Query optimization
        4. Search execution with resilience
        5. Metrics collection
        
        Args:
            collection_name: Name of the collection to search
            query: Query text
            config: Semantic search configuration
            request_id: Optional request ID for tracing
            
        Returns:
            SearchResult with hits and metadata
            
        Raises:
            InvalidSearchParametersError: If parameters are invalid
            SearchError: If search operation fails after retries
        """
        start_time = time.time()
        
        # Initialize metrics
        metrics = SearchMetrics(
            query_hash=str(hash(query)),
            collection_name=collection_name,
            request_id=request_id
        )
        
        try:
            # Step 1: Validate parameters
            self.validator.validate_search_params(collection_name, query, config)
            
            # Step 2: Sanitize inputs
            sanitized_query = self.sanitizer.sanitize_query(query)
            sanitized_collection = self.sanitizer.sanitize_collection_name(collection_name)
            
            # Step 3: Generate embedding with resilience
            embedding_start = time.time()
            query_vector = await self.resilience_manager.execute(
                self._generate_embedding,
                sanitized_query
            )
            metrics.embedding_time_ms = (time.time() - embedding_start) * 1000
            
            # Step 4: Build optimized search parameters
            params_builder = SearchParamsBuilder(config)
            search_params = params_builder.build(query_vector)
            
            # Step 5: Execute search with resilience
            search_start = time.time()
            results, execution_time_ms = await self.resilience_manager.execute(
                self._execute_search,
                collection_name=sanitized_collection,
                search_params=search_params,
                timeout=config.timeout
            )
            metrics.search_time_ms = (time.time() - search_start) * 1000
            metrics.results_count = len(results)
            metrics.status = SearchStatus.SUCCESS
            
            # Step 6: Create search result
            search_result = SearchResult(
                hits=results,
                total_hits=len(results),
                took_ms=execution_time_ms,
                search_params={
                    "type": "semantic",
                    "field": config.search_field,
                    "top_k": config.top_k,
                    "metric_type": config.metric_type,
                    "params": search_params.get("param", {}),
                    "optimized": True,
                    "request_id": request_id
                }
            )
            
            return search_result
            
        except InvalidSearchParametersError as e:
            metrics.status = SearchStatus.FAILURE
            metrics.error_message = str(e)
            raise
            
        except Exception as e:
            metrics.status = SearchStatus.FAILURE
            metrics.error_message = str(e)
            raise SearchError(f"Semantic search failed: {str(e)}") from e
            
        finally:
            # Record metrics
            metrics.total_time_ms = (time.time() - start_time) * 1000
            await self.metrics_collector.record_metric(metrics)
    
    async def batch_search(
        self,
        collection_name: str,
        queries: List[str],
        config: SemanticSearchConfig,
        request_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Perform batch semantic search operations.
        
        Args:
            collection_name: Name of the collection to search
            queries: List of query texts
            config: Semantic search configuration
            request_id: Optional request ID for tracing
            
        Returns:
            List of SearchResult objects
            
        Raises:
            InvalidSearchParametersError: If parameters are invalid
            SearchError: If any search operation fails
        """
        if not queries:
            raise InvalidSearchParametersError("Queries list cannot be empty")
        
        results = []
        for i, query in enumerate(queries):
            batch_request_id = f"{request_id}_batch_{i}" if request_id else None
            result = await self.search(
                collection_name=collection_name,
                query=query,
                config=config,
                request_id=batch_request_id
            )
            results.append(result)
        
        return results
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of search metrics.
        
        Returns:
            Dictionary with aggregated metrics
        """
        return await self.metrics_collector.get_summary()
    
    async def get_recent_metrics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent search metrics.
        
        Args:
            limit: Number of recent metrics to return
            
        Returns:
            List of recent metrics
        """
        return await self.metrics_collector.get_recent_metrics(limit)
    
    def get_resilience_status(self) -> Dict[str, Any]:
        """
        Get resilience system status.
        
        Returns:
            Dictionary with circuit breaker and retry status
        """
        return self.resilience_manager.get_status()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Returns:
            Health status dictionary
        """
        health = {
            "status": "healthy",
            "timestamp": time.time(),
            "components": {}
        }
        
        try:
            # Check resilience manager
            resilience_status = self.resilience_manager.get_status()
            health["components"]["resilience"] = resilience_status
            
            # Check metrics
            metrics_summary = await self.metrics_collector.get_summary()
            health["components"]["metrics"] = metrics_summary
            
            # Check if circuit breaker is open
            if resilience_status.get("circuit_breaker_enabled"):
                circuit_state = resilience_status.get("circuit_breaker", {}).get("state")
                if circuit_state == "open":
                    health["status"] = "degraded"
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        return health
    
    async def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        await self.metrics_collector.clear()
