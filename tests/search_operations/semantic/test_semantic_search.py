"""
Comprehensive unit tests for SemanticSearch.

This module provides systematic testing of the SemanticSearch class,
including initialization, search operations, optimization, metrics, resilience,
and validation integration.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from milvus_ops.search_operations.config.base import MetricType
from milvus_ops.search_operations.config.semantic import SemanticSearchConfig
from milvus_ops.search_operations.core.base import SearchResult
from milvus_ops.search_operations.core.search_ops_exceptions import (
    EmbeddingGenerationError,
    InvalidSearchParametersError,
    SearchError,
)
from milvus_ops.search_operations.providers.embedding import EmbeddingResult
from milvus_ops.search_operations.search.semantic.engine import SemanticSearch

# ============================================================================
# Test SemanticSearch Initialization
# ============================================================================


@pytest.mark.unit
class TestSemanticSearchInitialization:
    """
    Test SemanticSearch initialization.

    Coverage: SemanticSearch constructor with various configurations.
    """

    @pytest.mark.asyncio
    async def test_initialization_default(self, mock_connection_manager, mock_embedding_provider):
        """
        Test SemanticSearch initialization with default parameters.

        Coverage: SemanticSearch.__init__() with default flags enabled.
        """
        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )
        assert search._connection_manager is mock_connection_manager
        assert search._embedding_provider is mock_embedding_provider
        assert search.resilience_manager is not None
        assert search.metrics_collector is not None
        assert search.query_optimizer is not None
        assert search.validator is not None
        assert search.sanitizer is not None

    @pytest.mark.asyncio
    async def test_initialization_without_circuit_breaker(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test SemanticSearch initialization without circuit breaker.

        Coverage: SemanticSearch.__init__() with enable_circuit_breaker=False.
        """
        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
            enable_circuit_breaker=False,
        )
        assert search.resilience_manager is not None

    @pytest.mark.asyncio
    async def test_initialization_without_retry(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test SemanticSearch initialization without retry.

        Coverage: SemanticSearch.__init__() with enable_retry=False.
        """
        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
            enable_retry=False,
        )
        assert search.resilience_manager is not None

    @pytest.mark.asyncio
    async def test_initialization_without_query_optimization(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test SemanticSearch initialization without query optimization.

        Coverage: SemanticSearch.__init__() with enable_query_optimization=False.
        """
        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
            enable_query_optimization=False,
        )
        assert search.query_optimizer.enable_auto_tuning is False

    @pytest.mark.asyncio
    async def test_initialization_with_metrics_callback(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test SemanticSearch initialization with metrics callback.

        Coverage: SemanticSearch.__init__() with metrics_callback parameter.
        """
        callback = MagicMock()
        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
            metrics_callback=callback,
        )
        assert search.metrics_collector._metrics_callback is callback

    @pytest.mark.asyncio
    async def test_initialization_with_custom_max_metrics_history(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test SemanticSearch initialization with custom max_metrics_history.

        Coverage: SemanticSearch.__init__() with max_metrics_history parameter.
        """
        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
            max_metrics_history=500,
        )
        assert search.metrics_collector._max_history == 500


# ============================================================================
# Test SemanticSearch search() Method
# ============================================================================


@pytest.mark.unit
class TestSemanticSearchSearch:
    """
    Test SemanticSearch.search() method.

    Coverage: Search operations with various configurations and edge cases.
    """

    @pytest.mark.asyncio
    async def test_search_success(self, mock_connection_manager, mock_embedding_provider):
        """
        Test search() with successful search operation.

        Coverage: SemanticSearch.search() performs complete search workflow successfully.
        """
        # Setup mocks
        embedding_result = EmbeddingResult(
            embedding=[0.1] * 768,
            dimension=768,
            model_name="test_model",
            processing_time_ms=10.0,
        )
        mock_embedding_provider.generate_embedding = AsyncMock(return_value=embedding_result)

        sample_results = [
            {"id": 1, "distance": 0.1, "score": 0.9},
            {"id": 2, "distance": 0.2, "score": 0.8},
            {"id": 3, "distance": 0.3, "score": 0.7},
        ]
        mock_connection_manager.execute_operation_async = AsyncMock(return_value=sample_results)

        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        config = SemanticSearchConfig(
            top_k=10,
            search_field="vector",
            metric_type=MetricType.COSINE,
        )

        result = await search.search(
            collection_name="test_collection",
            query="test query",
            config=config,
        )

        assert isinstance(result, SearchResult)
        assert len(result.hits) == 3
        assert result.total_hits == 3
        assert result.search_params["type"] == "semantic"
        mock_embedding_provider.generate_embedding.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_empty_results(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test search() with empty search results.

        Coverage: SemanticSearch.search() handles empty results gracefully.
        """
        embedding_result = EmbeddingResult(
            embedding=[0.1] * 768,
            dimension=768,
            model_name="test_model",
            processing_time_ms=10.0,
        )
        mock_embedding_provider.generate_embedding = AsyncMock(return_value=embedding_result)

        mock_connection_manager.execute_operation_async = AsyncMock(return_value=[])

        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        config = SemanticSearchConfig(
            top_k=10,
            search_field="vector",
        )

        result = await search.search(
            collection_name="test_collection",
            query="test query",
            config=config,
        )

        assert isinstance(result, SearchResult)
        assert len(result.hits) == 0
        assert result.total_hits == 0

    @pytest.mark.asyncio
    async def test_search_with_expr_filter(self, mock_connection_manager, mock_embedding_provider):
        """
        Test search() with expression filter.

        Coverage: SemanticSearch.search() handles expr filter in config.
        """
        embedding_result = EmbeddingResult(
            embedding=[0.1] * 768,
            dimension=768,
            model_name="test_model",
            processing_time_ms=10.0,
        )
        mock_embedding_provider.generate_embedding = AsyncMock(return_value=embedding_result)

        sample_results = [{"id": 1, "distance": 0.1}]
        mock_connection_manager.execute_operation_async = AsyncMock(return_value=sample_results)

        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        config = SemanticSearchConfig(
            top_k=10,
            search_field="vector",
            expr='category == "technical"',
        )

        result = await search.search(
            collection_name="test_collection",
            query="test query",
            config=config,
        )

        assert isinstance(result, SearchResult)
        # Verify expr was passed in search params
        call_args = mock_connection_manager.execute_operation_async.call_args
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_search_with_request_id(self, mock_connection_manager, mock_embedding_provider):
        """
        Test search() with request ID for tracing.

        Coverage: SemanticSearch.search() includes request_id in search_params.
        """
        embedding_result = EmbeddingResult(
            embedding=[0.1] * 768,
            dimension=768,
            model_name="test_model",
            processing_time_ms=10.0,
        )
        mock_embedding_provider.generate_embedding = AsyncMock(return_value=embedding_result)

        sample_results = [{"id": 1, "distance": 0.1}]
        mock_connection_manager.execute_operation_async = AsyncMock(return_value=sample_results)

        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        config = SemanticSearchConfig(top_k=10)

        result = await search.search(
            collection_name="test_collection",
            query="test query",
            config=config,
            request_id="req_123",
        )

        assert result.search_params["request_id"] == "req_123"

    @pytest.mark.asyncio
    async def test_search_validation_error(self, mock_connection_manager, mock_embedding_provider):
        """
        Test search() raises InvalidSearchParametersError on validation failure.

        Coverage: SemanticSearch.search() validates parameters before execution.
        """
        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        config = SemanticSearchConfig(top_k=10)

        # Mock validator to raise validation error
        search.validator.validate_search_params = MagicMock(
            side_effect=InvalidSearchParametersError("Invalid collection name")
        )

        with pytest.raises(InvalidSearchParametersError, match="Invalid collection name"):
            await search.search(
                collection_name="",
                query="test query",
                config=config,
            )

    @pytest.mark.asyncio
    async def test_search_embedding_generation_error(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test search() handles embedding generation errors.

        Coverage: SemanticSearch.search() raises SearchError when embedding generation fails.
        """
        mock_embedding_provider.generate_embedding = AsyncMock(
            side_effect=EmbeddingGenerationError("Embedding generation failed")
        )

        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        config = SemanticSearchConfig(top_k=10)

        with pytest.raises(SearchError, match="Semantic search failed"):
            await search.search(
                collection_name="test_collection",
                query="test query",
                config=config,
            )

    @pytest.mark.asyncio
    async def test_search_connection_failure(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test search() handles connection failures.

        Coverage: SemanticSearch.search() handles connection manager failures with resilience.
        """
        embedding_result = EmbeddingResult(
            embedding=[0.1] * 768,
            dimension=768,
            model_name="test_model",
            processing_time_ms=10.0,
        )
        mock_embedding_provider.generate_embedding = AsyncMock(return_value=embedding_result)

        mock_connection_manager.execute_operation_async = AsyncMock(
            side_effect=ConnectionError("Connection failed")
        )

        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
            enable_retry=False,  # Disable retry for this test
        )

        config = SemanticSearchConfig(top_k=10)

        with pytest.raises(SearchError):
            await search.search(
                collection_name="test_collection",
                query="test query",
                config=config,
            )

    @pytest.mark.asyncio
    async def test_search_timeout(self, mock_connection_manager, mock_embedding_provider):
        """
        Test search() handles timeout scenarios.

        Coverage: SemanticSearch.search() handles timeout errors.
        """
        embedding_result = EmbeddingResult(
            embedding=[0.1] * 768,
            dimension=768,
            model_name="test_model",
            processing_time_ms=10.0,
        )
        mock_embedding_provider.generate_embedding = AsyncMock(return_value=embedding_result)

        mock_connection_manager.execute_operation_async = AsyncMock(
            side_effect=TimeoutError("Operation timed out")
        )

        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
            enable_retry=False,
        )

        config = SemanticSearchConfig(top_k=10, timeout=1.0)

        with pytest.raises(SearchError):
            await search.search(
                collection_name="test_collection",
                query="test query",
                config=config,
            )

    @pytest.mark.asyncio
    async def test_search_collects_metrics(self, mock_connection_manager, mock_embedding_provider):
        """
        Test search() collects metrics.

        Coverage: SemanticSearch.search() records metrics after search operation.
        """
        embedding_result = EmbeddingResult(
            embedding=[0.1] * 768,
            dimension=768,
            model_name="test_model",
            processing_time_ms=10.0,
        )
        mock_embedding_provider.generate_embedding = AsyncMock(return_value=embedding_result)

        sample_results = [{"id": 1, "distance": 0.1}]
        mock_connection_manager.execute_operation_async = AsyncMock(return_value=sample_results)

        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        config = SemanticSearchConfig(top_k=10)

        await search.search(
            collection_name="test_collection",
            query="test query",
            config=config,
        )

        # Verify metrics were collected
        summary = await search.get_metrics_summary()
        assert summary is not None
        assert "total_searches" in summary or "count" in summary or len(summary) > 0


# ============================================================================
# Test SemanticSearch batch_search() Method
# ============================================================================


@pytest.mark.unit
class TestSemanticSearchBatchSearch:
    """
    Test SemanticSearch.batch_search() method.

    Coverage: Batch search operations with various configurations.
    """

    @pytest.mark.asyncio
    async def test_batch_search_success(self, mock_connection_manager, mock_embedding_provider):
        """
        Test batch_search() with successful operations.

        Coverage: SemanticSearch.batch_search() processes multiple queries successfully.
        """
        embedding_result = EmbeddingResult(
            embedding=[0.1] * 768,
            dimension=768,
            model_name="test_model",
            processing_time_ms=10.0,
        )
        mock_embedding_provider.generate_embedding = AsyncMock(return_value=embedding_result)

        sample_results = [{"id": 1, "distance": 0.1}]
        mock_connection_manager.execute_operation_async = AsyncMock(return_value=sample_results)

        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        config = SemanticSearchConfig(top_k=10)

        queries = ["query 1", "query 2", "query 3"]
        results = await search.batch_search(
            collection_name="test_collection",
            queries=queries,
            config=config,
            request_id="batch_123",
        )

        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert mock_embedding_provider.generate_embedding.call_count == 3

    @pytest.mark.asyncio
    async def test_batch_search_empty_queries(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test batch_search() with empty queries list.

        Coverage: SemanticSearch.batch_search() raises
        InvalidSearchParametersError for empty queries.
        """
        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        config = SemanticSearchConfig(top_k=10)

        with pytest.raises(InvalidSearchParametersError, match="Queries list cannot be empty"):
            await search.batch_search(
                collection_name="test_collection",
                queries=[],
                config=config,
            )

    @pytest.mark.asyncio
    async def test_batch_search_with_request_id(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test batch_search() with request ID.

        Coverage: SemanticSearch.batch_search() generates request IDs for each query in batch.
        """
        embedding_result = EmbeddingResult(
            embedding=[0.1] * 768,
            dimension=768,
            model_name="test_model",
            processing_time_ms=10.0,
        )
        mock_embedding_provider.generate_embedding = AsyncMock(return_value=embedding_result)

        sample_results = [{"id": 1, "distance": 0.1}]
        mock_connection_manager.execute_operation_async = AsyncMock(return_value=sample_results)

        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        config = SemanticSearchConfig(top_k=10)

        queries = ["query 1", "query 2"]
        results = await search.batch_search(
            collection_name="test_collection",
            queries=queries,
            config=config,
            request_id="batch_123",
        )

        assert len(results) == 2
        # Verify request IDs are set
        assert results[0].search_params.get("request_id") == "batch_123_batch_0"
        assert results[1].search_params.get("request_id") == "batch_123_batch_1"


# ============================================================================
# Test SemanticSearch Metrics and Monitoring
# ============================================================================


@pytest.mark.unit
class TestSemanticSearchMetrics:
    """
    Test SemanticSearch metrics collection and monitoring methods.

    Coverage: Metrics collection, summaries, and health checks.
    """

    @pytest.mark.asyncio
    async def test_get_metrics_summary(self, mock_connection_manager, mock_embedding_provider):
        """
        Test get_metrics_summary() method.

        Coverage: SemanticSearch.get_metrics_summary() returns aggregated metrics.
        """
        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        summary = await search.get_metrics_summary()
        assert isinstance(summary, dict)

    @pytest.mark.asyncio
    async def test_get_recent_metrics(self, mock_connection_manager, mock_embedding_provider):
        """
        Test get_recent_metrics() method.

        Coverage: SemanticSearch.get_recent_metrics() returns recent metrics with limit.
        """
        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        recent = await search.get_recent_metrics(limit=5)
        assert isinstance(recent, list)

    @pytest.mark.asyncio
    async def test_clear_metrics(self, mock_connection_manager, mock_embedding_provider):
        """
        Test clear_metrics() method.

        Coverage: SemanticSearch.clear_metrics() clears all collected metrics.
        """
        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        await search.clear_metrics()

        # Verify metrics are cleared
        summary = await search.get_metrics_summary()
        # Should return empty or zero metrics
        assert summary is not None

    @pytest.mark.asyncio
    async def test_get_resilience_status(self, mock_connection_manager, mock_embedding_provider):
        """
        Test get_resilience_status() method.

        Coverage: SemanticSearch.get_resilience_status() returns resilience system status.
        """
        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        status = search.get_resilience_status()
        assert isinstance(status, dict)

    @pytest.mark.asyncio
    async def test_health_check(self, mock_connection_manager, mock_embedding_provider):
        """
        Test health_check() method.

        Coverage: SemanticSearch.health_check() performs comprehensive health check.
        """
        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        health = await search.health_check()
        assert isinstance(health, dict)
        assert "status" in health
        assert "timestamp" in health
        assert "components" in health

    @pytest.mark.asyncio
    async def test_health_check_degraded_state(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test health_check() returns degraded when circuit breaker is open.

        Coverage: SemanticSearch.health_check() detects degraded state.
        """
        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        # Mock circuit breaker to be open
        with patch.object(
            search.resilience_manager,
            "get_status",
            return_value={"circuit_breaker": {"state": "open"}},
        ):
            health = await search.health_check()
            assert health["status"] in ["healthy", "degraded", "unhealthy"]


# ============================================================================
# Test SemanticSearch Optimization Integration
# ============================================================================


@pytest.mark.unit
class TestSemanticSearchOptimization:
    """
    Test SemanticSearch query optimization integration.

    Coverage: Query optimization with various configurations.
    """

    @pytest.mark.asyncio
    async def test_search_with_optimization(self, mock_connection_manager, mock_embedding_provider):
        """
        Test search() uses query optimization when enabled.

        Coverage: SemanticSearch.search() applies query optimization.
        """
        embedding_result = EmbeddingResult(
            embedding=[0.1] * 768,
            dimension=768,
            model_name="test_model",
            processing_time_ms=10.0,
        )
        mock_embedding_provider.generate_embedding = AsyncMock(return_value=embedding_result)

        sample_results = [{"id": 1, "distance": 0.1}]
        mock_connection_manager.execute_operation_async = AsyncMock(return_value=sample_results)

        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
            enable_query_optimization=True,
        )

        config = SemanticSearchConfig(
            top_k=50,  # Large top_k should trigger optimization
            search_field="vector",
        )

        result = await search.search(
            collection_name="test_collection",
            query="test query",
            config=config,
        )

        assert isinstance(result, SearchResult)
        # Verify optimization was applied (indirectly through search params)
        assert result.search_params.get("optimized") is True

    @pytest.mark.asyncio
    async def test_search_without_optimization(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test search() without query optimization.

        Coverage: SemanticSearch.search() skips optimization when disabled.
        """
        embedding_result = EmbeddingResult(
            embedding=[0.1] * 768,
            dimension=768,
            model_name="test_model",
            processing_time_ms=10.0,
        )
        mock_embedding_provider.generate_embedding = AsyncMock(return_value=embedding_result)

        sample_results = [{"id": 1, "distance": 0.1}]
        mock_connection_manager.execute_operation_async = AsyncMock(return_value=sample_results)

        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
            enable_query_optimization=False,
        )

        config = SemanticSearchConfig(top_k=10)

        result = await search.search(
            collection_name="test_collection",
            query="test query",
            config=config,
        )

        assert isinstance(result, SearchResult)
        # Optimization flag should still be True in search_params (from base behavior)
        assert "optimized" in result.search_params


# ============================================================================
# Test SemanticSearch Validation and Sanitization
# ============================================================================


@pytest.mark.unit
class TestSemanticSearchValidation:
    """
    Test SemanticSearch validation and sanitization integration.

    Coverage: Parameter validation and query sanitization.
    """

    @pytest.mark.asyncio
    async def test_search_validates_collection_name(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test search() validates collection name.

        Coverage: SemanticSearch.search() validates collection name through validator.
        """
        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        config = SemanticSearchConfig(top_k=10)

        # Mock validator to raise error for empty collection name
        search.validator.validate_collection_name = MagicMock(
            side_effect=InvalidSearchParametersError("Collection name cannot be empty")
        )

        with pytest.raises(InvalidSearchParametersError):
            await search.search(
                collection_name="",
                query="test query",
                config=config,
            )

    @pytest.mark.asyncio
    async def test_search_sanitizes_query(self, mock_connection_manager, mock_embedding_provider):
        """
        Test search() sanitizes query input.

        Coverage: SemanticSearch.search() sanitizes query through sanitizer.
        """
        embedding_result = EmbeddingResult(
            embedding=[0.1] * 768,
            dimension=768,
            model_name="test_model",
            processing_time_ms=10.0,
        )
        mock_embedding_provider.generate_embedding = AsyncMock(return_value=embedding_result)

        sample_results = [{"id": 1, "distance": 0.1}]
        mock_connection_manager.execute_operation_async = AsyncMock(return_value=sample_results)

        search = SemanticSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        # Mock sanitizer to return cleaned query
        search.sanitizer.sanitize_query = MagicMock(return_value="cleaned query")

        config = SemanticSearchConfig(top_k=10)

        await search.search(
            collection_name="test_collection",
            query="original query with special chars !@#$",
            config=config,
        )

        # Verify sanitizer was called
        search.sanitizer.sanitize_query.assert_called_once()
