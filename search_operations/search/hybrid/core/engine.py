"""
Hybrid Search Engine

This module provides the production-grade implementation of hybrid search,
combining dense vector search with sparse vectors (BM25) or keyword search
with comprehensive reliability, fault tolerance, and observability features.
"""

import time
import logging
import asyncio
from typing import List, Dict, Any, Optional, Callable
from contextlib import asynccontextmanager

from connection_management import ConnectionManager
from ....core.search_ops_exceptions import (
    SearchError,
    HybridSearchError,
    InvalidSearchParametersError,
    EmbeddingGenerationError,
    SparseVectorGenerationError,
)
from ....config.hybrid import HybridSearchConfig
from ....providers.embedding import EmbeddingProvider
from ....core.base import BaseSearch, SearchResult

from .bm25 import BM25SparseVectorGenerator
from .fusion import fuse_results_rrf, fuse_results_weighted
from ..resilience.circuit_breaker import CircuitBreaker
from ..resilience.retry import execute_with_retry
from ..utils.metrics import HybridSearchMetrics, SearchStatus
from ..utils.config import HybridSearchMode, BM25Config, RetryConfig
from ..utils.validation import validate_search_params, sanitize_query

logger = logging.getLogger(__name__)


class HybridSearch(BaseSearch[HybridSearchConfig]):
    """
    Production-grade implementation of hybrid search.
    
    Features:
    - BM25 sparse vector generation
    - Multiple search mode support (vector+sparse, vector+keyword, vector-only, all methods)
    - Automatic retry with exponential backoff
    - Circuit breaker for fault tolerance
    - Result fusion strategies (RRF, weighted)
    - Comprehensive metrics and observability
    - Parameter validation and sanitization
    - Graceful degradation and fallback
    - Batch processing support
    - Health checking and diagnostics
    """
    
    def __init__(
        self,
        connection_manager: ConnectionManager,
        embedding_provider: EmbeddingProvider,
        enable_caching: bool = True,
        bm25_config: Optional[BM25Config] = None,
        retry_config: Optional[RetryConfig] = None,
        enable_circuit_breaker: bool = True,
        metrics_callback: Optional[Callable[[HybridSearchMetrics], None]] = None,
        max_batch_size: int = 50,
        enable_query_optimization: bool = True,
        fallback_to_vector: bool = True
    ):
        """
        Initialize hybrid search with production features.
        
        Args:
            connection_manager: ConnectionManager for Milvus operations
            embedding_provider: Provider for generating embeddings
            enable_caching: Whether to enable caching
            bm25_config: BM25 configuration (uses defaults if not provided)
            retry_config: Configuration for retry behavior
            enable_circuit_breaker: Enable circuit breaker pattern
            metrics_callback: Optional callback for metrics reporting
            max_batch_size: Maximum batch size for bulk operations
            enable_query_optimization: Enable query optimization features
            fallback_to_vector: Fallback to vector-only search on sparse errors
        """
        super().__init__(connection_manager, embedding_provider)
        
        self.enable_caching = enable_caching
        self.max_batch_size = max_batch_size
        self.enable_query_optimization = enable_query_optimization
        self.fallback_to_vector = fallback_to_vector
        self.metrics_callback = metrics_callback
        
        # Initialize BM25 generator
        self.bm25_generator = BM25SparseVectorGenerator(
            config=bm25_config,
            enable_caching=enable_caching
        )
        
        # Initialize retry configuration
        self.retry_config = retry_config or RetryConfig()
        
        # Initialize circuit breaker
        self.circuit_breaker = None
        if enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60.0,
                expected_exception=HybridSearchError
            )
        
        # Metrics storage
        self._metrics_history: List[HybridSearchMetrics] = []
        self._lock = asyncio.Lock()
        
        logger.info(
            f"HybridSearch initialized - "
            f"caching: {enable_caching}, "
            f"circuit_breaker: {enable_circuit_breaker}, "
            f"fallback: {fallback_to_vector}"
        )
    
    def _determine_search_mode(self, config: HybridSearchConfig) -> HybridSearchMode:
        """
        Determine the hybrid search mode based on configuration.
        
        Args:
            config: Hybrid search configuration
            
        Returns:
            HybridSearchMode enum value
        """
        has_sparse = bool(config.sparse_field)
        has_keyword = bool(config.keyword_field)
        
        if has_sparse and has_keyword:
            return HybridSearchMode.ALL_METHODS
        elif has_sparse:
            return HybridSearchMode.VECTOR_SPARSE
        elif has_keyword:
            return HybridSearchMode.VECTOR_KEYWORD
        else:
            return HybridSearchMode.VECTOR_ONLY
    
    async def search(
        self,
        collection_name: str,
        query: str,
        config: HybridSearchConfig,
        request_id: Optional[str] = None,
        fusion_strategy: str = "rrf"
    ) -> SearchResult:
        """
        Perform hybrid search operation with full production features.
        
        This method handles the complete hybrid search workflow with validation,
        retry logic, fallback handling, result fusion, and comprehensive metrics.
        
        Args:
            collection_name: Name of the collection to search
            query: Query text
            config: Hybrid search configuration
            request_id: Optional request ID for tracing
            fusion_strategy: Strategy for fusing results ("rrf" or "weighted")
            
        Returns:
            SearchResult with fused hits and metadata
            
        Raises:
            InvalidSearchParametersError: If parameters are invalid
            HybridSearchError: If search operation fails after retries
        """
        start_time = time.time()
        search_mode = self._determine_search_mode(config)
        
        metrics = HybridSearchMetrics(
            query_hash=str(hash(query)),
            collection_name=collection_name,
            search_mode=search_mode.value
        )
        
        try:
            # Validate and sanitize
            validate_search_params(collection_name, query, config)
            sanitized_query = sanitize_query(query)
            
            # Generate dense embedding with retry
            embedding_start = time.time()
            try:
                query_vector = await execute_with_retry(
                    self._generate_embedding,
                    self.retry_config,
                    sanitized_query
                )
                metrics.embedding_time_ms = (time.time() - embedding_start) * 1000
            except EmbeddingGenerationError as e:
                metrics.status = SearchStatus.FAILURE
                metrics.error_message = f"Embedding generation failed: {str(e)}"
                logger.error(f"Embedding generation failed: {str(e)}")
                raise HybridSearchError(f"Embedding generation failed: {str(e)}") from e
            
            # Prepare search results storage
            all_results = []
            
            # Execute vector search with retry
            search_start = time.time()
            vector_results = None
            
            try:
                vector_search_params = {
                    "data": [query_vector],
                    "anns_field": config.vector_field,
                    "param": config.params or {},
                    "limit": config.top_k,
                    "expr": config.expr,
                    "output_fields": config.output_fields or []
                }
                
                try:
                    vector_results, _ = await execute_with_retry(
                        self._execute_search,
                        self.retry_config,
                        collection_name=collection_name,
                        search_params=vector_search_params,
                        timeout=config.timeout
                    )
                    
                    metrics.dense_results = len(vector_results)
                    all_results.append(vector_results)
                    logger.debug(f"Vector search returned {len(vector_results)} results")
                except SearchError as e:
                    logger.error(f"Vector search failed with SearchError: {str(e)}")
                    metrics.status = SearchStatus.FAILURE
                    metrics.error_message = f"Vector search failed: {str(e)}"
                    raise HybridSearchError(f"Vector search failed: {str(e)}") from e
                
            except Exception as e:
                logger.error(f"Vector search failed with unexpected error: {str(e)}")
                if not self.fallback_to_vector:
                    raise
            
            # Execute sparse search if configured
            sparse_results = None
            if config.sparse_field and search_mode in [
                HybridSearchMode.VECTOR_SPARSE,
                HybridSearchMode.ALL_METHODS
            ]:
                try:
                    sparse_start = time.time()
                    try:
                        sparse_vector = await self.bm25_generator.generate(sanitized_query)
                        metrics.sparse_generation_time_ms = (time.time() - sparse_start) * 1000
                    except SparseVectorGenerationError as e:
                        logger.error(f"Sparse vector generation failed: {str(e)}")
                        metrics.status = SearchStatus.DEGRADED
                        if self.fallback_to_vector and vector_results:
                            logger.info("Falling back to vector-only results due to sparse vector generation failure")
                            # Skip sparse search and use only vector results
                            sparse_vector = None
                            raise  # This will be caught by the outer try block
                        else:
                            raise HybridSearchError(f"Sparse vector generation failed: {str(e)}") from e
                    
                    sparse_search_params = {
                        "data": [sparse_vector],
                        "anns_field": config.sparse_field,
                        "param": {},
                        "limit": config.top_k,
                        "expr": config.expr,
                        "output_fields": config.output_fields or []
                    }
                    
                    try:
                        sparse_results, _ = await execute_with_retry(
                            self._execute_search,
                            self.retry_config,
                            collection_name=collection_name,
                            search_params=sparse_search_params,
                            timeout=config.timeout
                        )
                        
                        metrics.sparse_results = len(sparse_results)
                        all_results.append(sparse_results)
                        logger.debug(f"Sparse search returned {len(sparse_results)} results")
                    except SearchError as e:
                        logger.error(f"Sparse search failed with SearchError: {str(e)}")
                        metrics.status = SearchStatus.DEGRADED
                        if self.fallback_to_vector and vector_results:
                            logger.info("Falling back to vector-only results due to sparse search failure")
                            # Don't add sparse results to all_results
                        else:
                            raise HybridSearchError(f"Sparse search failed: {str(e)}") from e
                    
                except Exception as e:
                    logger.warning(f"Sparse search failed: {str(e)}")
                    if self.fallback_to_vector and vector_results:
                        logger.info("Falling back to vector-only results")
                        metrics.status = SearchStatus.DEGRADED
                    else:
                        raise
            
            metrics.search_time_ms = (time.time() - search_start) * 1000
            
            # Fuse results
            fusion_start = time.time()
            if len(all_results) > 1:
                if fusion_strategy == "rrf":
                    fused_results = await fuse_results_rrf(all_results)
                else:
                    fused_results = await fuse_results_weighted(
                        vector_results or [],
                        sparse_results or [],
                        config.vector_weight,
                        config.sparse_weight
                    )
                fused_results = fused_results[:config.top_k]
            elif len(all_results) == 1:
                fused_results = all_results[0][:config.top_k]
            else:
                fused_results = []
            
            metrics.fusion_time_ms = (time.time() - fusion_start) * 1000
            metrics.results_count = len(fused_results)
            
            # Create search result
            search_result = SearchResult(
                hits=fused_results,
                total_hits=len(fused_results),
                took_ms=metrics.search_time_ms + metrics.fusion_time_ms,
                search_params={
                    "type": "hybrid",
                    "mode": search_mode.value,
                    "vector_field": config.vector_field,
                    "sparse_field": config.sparse_field,
                    "keyword_field": config.keyword_field,
                    "top_k": config.top_k,
                    "fusion_strategy": fusion_strategy,
                    "vector_weight": config.vector_weight,
                    "sparse_weight": config.sparse_weight,
                    "request_id": request_id,
                    "dense_results": metrics.dense_results,
                    "sparse_results": metrics.sparse_results
                }
            )
            
            logger.info(
                f"Hybrid search completed - collection: {collection_name}, "
                f"mode: {search_mode.value}, results: {len(fused_results)}, "
                f"embedding: {metrics.embedding_time_ms:.2f}ms, "
                f"sparse_gen: {metrics.sparse_generation_time_ms:.2f}ms, "
                f"search: {metrics.search_time_ms:.2f}ms, "
                f"fusion: {metrics.fusion_time_ms:.2f}ms"
            )
            
            return search_result
            
        except InvalidSearchParametersError as e:
            metrics.status = SearchStatus.FAILURE
            metrics.error_message = str(e)
            logger.error(f"Invalid search parameters: {str(e)}")
            raise
            
        except Exception as e:
            metrics.status = SearchStatus.FAILURE
            metrics.error_message = str(e)
            error_msg = f"Hybrid search failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise HybridSearchError(error_msg) from e
            
        finally:
            # Record total time and store metrics
            metrics.total_time_ms = (time.time() - start_time) * 1000
            
            async with self._lock:
                self._metrics_history.append(metrics)
                if len(self._metrics_history) > 1000:
                    self._metrics_history = self._metrics_history[-1000:]
            
            if self.metrics_callback:
                try:
                    self.metrics_callback(metrics)
                except Exception as e:
                    logger.error(f"Metrics callback failed: {str(e)}")
    
    async def batch_search(
        self,
        collection_name: str,
        queries: List[str],
        config: HybridSearchConfig,
        request_id: Optional[str] = None,
        fusion_strategy: str = "rrf"
    ) -> List[SearchResult]:
        """
        Perform batch hybrid search with automatic batching.
        
        Args:
            collection_name: Name of the collection to search
            queries: List of query texts
            config: Hybrid search configuration
            request_id: Optional request ID for tracing
            fusion_strategy: Strategy for fusing results
            
        Returns:
            List of SearchResult objects
            
        Raises:
            InvalidSearchParametersError: If parameters are invalid
            HybridSearchError: If search operation fails
        """
        if not queries:
            raise InvalidSearchParametersError("Queries list cannot be empty")
        
        results = []
        for i in range(0, len(queries), self.max_batch_size):
            batch = queries[i:i + self.max_batch_size]
            batch_results = await asyncio.gather(
                *[
                    self.search(
                        collection_name,
                        query,
                        config,
                        f"{request_id}-{j}" if request_id else None,
                        fusion_strategy
                    )
                    for j, query in enumerate(batch, start=i)
                ],
                return_exceptions=True
            )
            
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Query {i + j} failed in batch: {str(result)}")
                    raise HybridSearchError(
                        f"Batch search failed at index {i + j}"
                    ) from result
                results.append(result)
        
        return results
    
    async def search_with_reranking(
        self,
        collection_name: str,
        query: str,
        config: HybridSearchConfig,
        rerank_top_k: int = 100,
        final_top_k: Optional[int] = None,
        request_id: Optional[str] = None
    ) -> SearchResult:
        """
        Perform hybrid search with two-stage retrieval.
        
        First retrieves more results, then returns top results.
        Note: Actual reranking is integrated via SearchManager using
        Milvus native reranking capabilities.
        
        Args:
            collection_name: Name of the collection
            query: Query text
            config: Search configuration
            rerank_top_k: Number of results to retrieve for reranking
            final_top_k: Final number of results to return
            request_id: Optional request ID
            
        Returns:
            SearchResult with top results
        """
        final_top_k = final_top_k or config.top_k
        
        # First stage: retrieve more results
        original_top_k = config.top_k
        config.top_k = rerank_top_k
        
        try:
            stage1_result = await self.search(
                collection_name,
                query,
                config,
                request_id
            )
            
            # Return top final_top_k results
            reranked_hits = stage1_result.hits[:final_top_k]
            
            return SearchResult(
                hits=reranked_hits,
                total_hits=len(reranked_hits),
                took_ms=stage1_result.took_ms,
                search_params={
                    **stage1_result.search_params,
                    "reranked": True,
                    "rerank_top_k": rerank_top_k,
                    "final_top_k": final_top_k
                }
            )
        finally:
            # Restore original top_k
            config.top_k = original_top_k
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of search metrics.
        
        Returns:
            Dictionary with metrics summary including success rates,
            average timings, and component statistics
        """
        async with self._lock:
            if not self._metrics_history:
                return {"message": "No metrics available"}
            
            total_searches = len(self._metrics_history)
            successful = sum(
                1 for m in self._metrics_history
                if m.status == SearchStatus.SUCCESS
            )
            failed = sum(
                1 for m in self._metrics_history
                if m.status == SearchStatus.FAILURE
            )
            degraded = sum(
                1 for m in self._metrics_history
                if m.status == SearchStatus.DEGRADED
            )
            
            # Calculate averages
            avg_embedding = sum(m.embedding_time_ms for m in self._metrics_history) / total_searches
            avg_sparse = sum(m.sparse_generation_time_ms for m in self._metrics_history) / total_searches
            avg_search = sum(m.search_time_ms for m in self._metrics_history) / total_searches
            avg_fusion = sum(m.fusion_time_ms for m in self._metrics_history) / total_searches
            avg_total = sum(m.total_time_ms for m in self._metrics_history) / total_searches
            
            # Count by search mode
            from collections import defaultdict
            mode_counts = defaultdict(int)
            for m in self._metrics_history:
                mode_counts[m.search_mode] += 1
            
            return {
                "total_searches": total_searches,
                "successful": successful,
                "failed": failed,
                "degraded": degraded,
                "success_rate": successful / total_searches if total_searches > 0 else 0,
                "avg_embedding_time_ms": round(avg_embedding, 2),
                "avg_sparse_generation_ms": round(avg_sparse, 2),
                "avg_search_time_ms": round(avg_search, 2),
                "avg_fusion_time_ms": round(avg_fusion, 2),
                "avg_total_time_ms": round(avg_total, 2),
                "search_modes": dict(mode_counts),
                "bm25_stats": self.bm25_generator.get_stats(),
                "circuit_breaker_state": self.circuit_breaker.state if self.circuit_breaker else "disabled"
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the hybrid search system.
        
        Returns:
            Health status dictionary with component status
        """
        health = {
            "status": "healthy",
            "timestamp": time.time(),
            "components": {}
        }
        
        try:
            # Check BM25 generator
            bm25_stats = self.bm25_generator.get_stats()
            health["components"]["bm25"] = {
                "status": "healthy",
                "cache_hit_rate": bm25_stats["cache_hit_rate"],
                "doc_count": bm25_stats["doc_count"]
            }
            
            # Check circuit breaker
            if self.circuit_breaker:
                health["components"]["circuit_breaker"] = {
                    "state": self.circuit_breaker.state,
                    "failures": self.circuit_breaker.failure_count
                }
                
                if self.circuit_breaker.state == "open":
                    health["status"] = "unhealthy"
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
            logger.error(f"Health check failed: {str(e)}")
        
        return health
    
    async def explain_search(
        self,
        collection_name: str,
        query: str,
        config: HybridSearchConfig,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Explain hybrid search results with detailed breakdown.
        
        Args:
            collection_name: Collection name
            query: Query text
            config: Search configuration
            request_id: Optional request ID
            
        Returns:
            Detailed explanation of search process and results
        """
        start_time = time.time()
        explanation = {
            "query": query,
            "search_mode": self._determine_search_mode(config).value,
            "stages": {},
            "timing": {}
        }
        
        try:
            # Tokenization
            tokens = self.bm25_generator._tokenize(query)
            explanation["stages"]["tokenization"] = {
                "original_query": query,
                "tokens": tokens,
                "token_count": len(tokens)
            }
            
            # Embedding generation
            embed_start = time.time()
            query_vector = await self._generate_embedding(query)
            explanation["timing"]["embedding_ms"] = (time.time() - embed_start) * 1000
            explanation["stages"]["embedding"] = {
                "dimension": len(query_vector),
                "norm": sum(x**2 for x in query_vector)**0.5
            }
            
            # Sparse vector generation
            if config.sparse_field:
                sparse_start = time.time()
                sparse_vector = await self.bm25_generator.generate(query)
                explanation["timing"]["sparse_generation_ms"] = (time.time() - sparse_start) * 1000
                explanation["stages"]["sparse_vector"] = {
                    "dimension": len(sparse_vector["indices"]),
                    "sparsity": len(sparse_vector["indices"]) / self.bm25_generator.config.max_dimensions,
                    "top_indices": sparse_vector["indices"][:10],
                    "top_values": sparse_vector["values"][:10]
                }
            
            # Execute search
            result = await self.search(collection_name, query, config, request_id)
            
            explanation["timing"]["total_ms"] = (time.time() - start_time) * 1000
            explanation["results"] = {
                "count": len(result.hits),
                "top_scores": [hit.get("fusion_score", 0.0) for hit in result.hits[:5]]
            }
            explanation["search_params"] = result.search_params
            
        except Exception as e:
            explanation["error"] = str(e)
            logger.error(f"Search explanation failed: {str(e)}")
        
        return explanation
    
    async def benchmark(
        self,
        collection_name: str,
        test_queries: List[str],
        config: HybridSearchConfig,
        iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmark hybrid search performance.
        
        Args:
            collection_name: Collection to benchmark
            test_queries: List of test queries
            config: Search configuration
            iterations: Number of iterations per query
            
        Returns:
            Benchmark results with timing statistics
        """
        results = {
            "query_count": len(test_queries),
            "iterations": iterations,
            "timings": [],
            "errors": []
        }
        
        for iteration in range(iterations):
            for i, query in enumerate(test_queries):
                try:
                    start = time.time()
                    await self.search(collection_name, query, config)
                    elapsed = (time.time() - start) * 1000
                    results["timings"].append(elapsed)
                except Exception as e:
                    results["errors"].append(f"Query {i} iteration {iteration}: {str(e)}")
        
        if results["timings"]:
            sorted_timings = sorted(results["timings"])
            results["avg_ms"] = sum(results["timings"]) / len(results["timings"])
            results["min_ms"] = min(results["timings"])
            results["max_ms"] = max(results["timings"])
            results["median_ms"] = sorted_timings[len(sorted_timings) // 2]
            results["p50_ms"] = sorted_timings[int(len(sorted_timings) * 0.50)]
            results["p90_ms"] = sorted_timings[int(len(sorted_timings) * 0.90)]
            results["p95_ms"] = sorted_timings[int(len(sorted_timings) * 0.95)]
            results["p99_ms"] = sorted_timings[int(len(sorted_timings) * 0.99)]
        
        return results
    
    async def optimize_bm25_params(
        self,
        sample_documents: List[str],
        param_ranges: Optional[Dict[str, List[float]]] = None
    ) -> BM25Config:
        """
        Optimize BM25 parameters using sample documents.
        
        Args:
            sample_documents: Sample documents for parameter tuning
            param_ranges: Optional parameter ranges to search
            
        Returns:
            Optimized BM25Config
        """
        if not param_ranges:
            param_ranges = {
                "k1": [1.0, 1.2, 1.5, 1.8, 2.0],
                "b": [0.5, 0.65, 0.75, 0.85, 1.0]
            }
        
        best_config = None
        best_score = float('-inf')
        
        # Simple grid search
        for k1 in param_ranges.get("k1", [1.5]):
            for b in param_ranges.get("b", [0.75]):
                test_config = BM25Config(k1=k1, b=b)
                test_generator = BM25SparseVectorGenerator(config=test_config)
                
                # Generate vectors and compute quality metrics
                vectors = []
                for doc in sample_documents[:100]:  # Limit sample size
                    try:
                        vec = await test_generator.generate(doc)
                        vectors.append(vec)
                    except Exception:
                        continue
                
                # Simple quality metric: average sparsity
                if vectors:
                    avg_sparsity = sum(len(v["indices"]) for v in vectors) / len(vectors)
                    score = avg_sparsity
                    
                    if score > best_score:
                        best_score = score
                        best_config = test_config
        
        logger.info(f"Optimized BM25 params - k1: {best_config.k1}, b: {best_config.b}")
        return best_config or BM25Config()
    
    async def close(self) -> None:
        """Gracefully shutdown the hybrid search instance."""
        logger.info("Shutting down HybridSearch...")
        
        # Log final metrics
        summary = await self.get_metrics_summary()
        logger.info(f"Final metrics: {summary}")
        
        # Clear caches
        await self.bm25_generator.clear_cache()
        
        # Clear metrics history
        async with self._lock:
            self._metrics_history.clear()
        
        logger.info("HybridSearch shutdown complete")
    
    @asynccontextmanager
    async def search_context(self):
        """
        Context manager for search operations with automatic cleanup.
        
        Usage:
            async with hybrid_search.search_context():
                result = await hybrid_search.search(...)
        """
        try:
            yield self
        finally:
            pass

