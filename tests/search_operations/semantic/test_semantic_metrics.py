"""
Comprehensive unit tests for semantic search metrics collection.

This module provides systematic testing of MetricsCollector, metrics aggregation,
history limits, callbacks, and SearchMetrics dataclass.
"""

import pytest

from milvus_ops.search_operations.search.semantic.metrics import (
    MetricsCollector,
    SearchMetrics,
    SearchStatus,
)

# ============================================================================
# Test SearchMetrics Dataclass
# ============================================================================


@pytest.mark.unit
class TestSearchMetrics:
    """
    Test SearchMetrics dataclass.

    Coverage: SearchMetrics initialization, properties, default values.
    """

    def test_search_metrics_initialization(self):
        """
        Test SearchMetrics initialization with required fields.

        Coverage: SearchMetrics.__init__() with query_hash and collection_name.
        """
        metrics = SearchMetrics(
            query_hash="abc123",
            collection_name="test_collection",
        )
        assert metrics.query_hash == "abc123"
        assert metrics.collection_name == "test_collection"
        assert metrics.status == SearchStatus.SUCCESS
        assert metrics.embedding_time_ms == 0.0
        assert metrics.search_time_ms == 0.0
        assert metrics.total_time_ms == 0.0
        assert metrics.results_count == 0
        assert metrics.retry_count == 0
        assert metrics.error_message is None
        assert metrics.timestamp > 0
        assert metrics.request_id is None

    def test_search_metrics_full(self):
        """
        Test SearchMetrics initialization with all fields.

        Coverage: SearchMetrics.__init__() with all optional fields.
        """
        metrics = SearchMetrics(
            query_hash="abc123",
            collection_name="test_collection",
            embedding_time_ms=10.5,
            search_time_ms=25.3,
            total_time_ms=35.8,
            results_count=5,
            retry_count=2,
            status=SearchStatus.SUCCESS,
            error_message=None,
            request_id="req_123",
        )
        assert metrics.embedding_time_ms == 10.5
        assert metrics.search_time_ms == 25.3
        assert metrics.total_time_ms == 35.8
        assert metrics.results_count == 5
        assert metrics.retry_count == 2
        assert metrics.request_id == "req_123"


# ============================================================================
# Test SearchStatus Enum
# ============================================================================


@pytest.mark.unit
class TestSearchStatus:
    """
    Test SearchStatus enum.

    Coverage: SearchStatus enum values.
    """

    def test_search_status_enum_values(self):
        """
        Test SearchStatus enum has correct values.

        Coverage: SearchStatus enum contains expected values.
        """
        assert SearchStatus.SUCCESS.value == "success"
        assert SearchStatus.FAILURE.value == "failure"
        assert SearchStatus.TIMEOUT.value == "timeout"
        assert SearchStatus.PARTIAL.value == "partial"
        assert SearchStatus.RETRYING.value == "retrying"


# ============================================================================
# Test MetricsCollector
# ============================================================================


@pytest.mark.unit
class TestMetricsCollector:
    """
    Test MetricsCollector class.

    Coverage: Initialization, record_metric, get_summary, get_recent_metrics, clear, history limits.
    """

    def test_metrics_collector_initialization_default(self):
        """
        Test MetricsCollector initialization with default parameters.

        Coverage: MetricsCollector.__init__() with default max_history.
        """
        collector = MetricsCollector()
        assert collector._max_history == 1000
        assert collector._metrics_callback is None
        assert len(collector._metrics_history) == 0

    def test_metrics_collector_initialization_custom(self):
        """
        Test MetricsCollector initialization with custom parameters.

        Coverage: MetricsCollector.__init__() with custom max_history and callback.
        """
        callback = lambda m: None  # noqa: E731
        collector = MetricsCollector(max_history=500, metrics_callback=callback)
        assert collector._max_history == 500
        assert collector._metrics_callback is callback

    @pytest.mark.asyncio
    async def test_record_metric_success(self):
        """
        Test record_metric() records successful search metrics.

        Coverage: MetricsCollector.record_metric() records SearchMetrics successfully.
        """
        collector = MetricsCollector()
        metrics = SearchMetrics(
            query_hash="abc123",
            collection_name="test_collection",
            status=SearchStatus.SUCCESS,
            results_count=5,
        )
        await collector.record_metric(metrics)
        assert len(collector._metrics_history) == 1
        assert collector._metrics_history[0] == metrics

    @pytest.mark.asyncio
    async def test_record_metric_failure(self):
        """
        Test record_metric() records failed search metrics.

        Coverage: MetricsCollector.record_metric() records failed searches.
        """
        collector = MetricsCollector()
        metrics = SearchMetrics(
            query_hash="abc123",
            collection_name="test_collection",
            status=SearchStatus.FAILURE,
            error_message="Search failed",
        )
        await collector.record_metric(metrics)
        assert len(collector._metrics_history) == 1
        assert collector._metrics_history[0].status == SearchStatus.FAILURE

    @pytest.mark.asyncio
    async def test_record_metric_timeout(self):
        """
        Test record_metric() records timeout metrics.

        Coverage: MetricsCollector.record_metric() records timeout searches.
        """
        collector = MetricsCollector()
        metrics = SearchMetrics(
            query_hash="abc123",
            collection_name="test_collection",
            status=SearchStatus.TIMEOUT,
        )
        await collector.record_metric(metrics)
        assert collector._metrics_history[0].status == SearchStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_record_metric_history_limit(self):
        """
        Test record_metric() maintains history limit.

        Coverage: MetricsCollector.record_metric() maintains max_history size.
        """
        collector = MetricsCollector(max_history=5)
        # Add more metrics than max_history
        for i in range(10):
            metrics = SearchMetrics(
                query_hash=f"hash{i}",
                collection_name="test_collection",
            )
            await collector.record_metric(metrics)
        # Should only retain last 5
        assert len(collector._metrics_history) == 5

    @pytest.mark.asyncio
    async def test_record_metric_callback(self):
        """
        Test record_metric() invokes callback.

        Coverage: MetricsCollector.record_metric() calls metrics_callback if provided.
        """
        callback_called = False

        def callback(metrics: SearchMetrics) -> None:
            nonlocal callback_called
            callback_called = True

        collector = MetricsCollector(metrics_callback=callback)
        metrics = SearchMetrics(
            query_hash="abc123",
            collection_name="test_collection",
        )
        await collector.record_metric(metrics)
        assert callback_called is True

    @pytest.mark.asyncio
    async def test_get_summary_empty(self):
        """
        Test get_summary() with empty metrics.

        Coverage: MetricsCollector.get_summary() returns empty message when no metrics.
        """
        collector = MetricsCollector()
        summary = await collector.get_summary()
        assert isinstance(summary, dict)
        assert "message" in summary or summary == {}

    @pytest.mark.asyncio
    async def test_get_summary_with_metrics(self):
        """
        Test get_summary() aggregates metrics.

        Coverage: MetricsCollector.get_summary() aggregates metrics correctly.
        """
        collector = MetricsCollector()
        # Add successful search
        await collector.record_metric(
            SearchMetrics(
                query_hash="hash1",
                collection_name="test_collection",
                status=SearchStatus.SUCCESS,
                embedding_time_ms=10.0,
                search_time_ms=20.0,
                total_time_ms=30.0,
                results_count=5,
            )
        )
        # Add failed search
        await collector.record_metric(
            SearchMetrics(
                query_hash="hash2",
                collection_name="test_collection",
                status=SearchStatus.FAILURE,
            )
        )
        summary = await collector.get_summary()
        assert summary["total_searches"] == 2
        assert summary["successful"] == 1
        assert summary["failed"] == 1
        assert "success_rate" in summary
        assert "avg_embedding_time_ms" in summary
        assert "avg_search_time_ms" in summary
        assert "avg_total_time_ms" in summary

    @pytest.mark.asyncio
    async def test_get_recent_metrics(self):
        """
        Test get_recent_metrics() returns recent metrics.

        Coverage: MetricsCollector.get_recent_metrics() returns recent metrics with limit.
        """
        collector = MetricsCollector()
        # Add multiple metrics
        for i in range(10):
            await collector.record_metric(
                SearchMetrics(
                    query_hash=f"hash{i}",
                    collection_name="test_collection",
                )
            )
        recent = await collector.get_recent_metrics(limit=5)
        assert isinstance(recent, list)
        assert len(recent) == 5
        assert all("query_hash" in m for m in recent)
        assert all("collection" in m for m in recent)
        assert all("status" in m for m in recent)

    @pytest.mark.asyncio
    async def test_clear_metrics(self):
        """
        Test clear() clears all metrics.

        Coverage: MetricsCollector.clear() removes all metrics.
        """
        collector = MetricsCollector()
        # Add metrics
        for i in range(5):
            await collector.record_metric(
                SearchMetrics(
                    query_hash=f"hash{i}",
                    collection_name="test_collection",
                )
            )
        assert len(collector._metrics_history) == 5
        await collector.clear()
        assert len(collector._metrics_history) == 0

    @pytest.mark.asyncio
    async def test_record_metric_with_various_statuses(self):
        """
        Test record_metric() with various search statuses.

        Coverage: MetricsCollector.record_metric() handles all SearchStatus values.
        """
        collector = MetricsCollector()
        statuses = [
            SearchStatus.SUCCESS,
            SearchStatus.FAILURE,
            SearchStatus.TIMEOUT,
            SearchStatus.PARTIAL,
            SearchStatus.RETRYING,
        ]
        for status in statuses:
            metrics = SearchMetrics(
                query_hash=f"hash_{status.value}",
                collection_name="test_collection",
                status=status,
            )
            await collector.record_metric(metrics)
        summary = await collector.get_summary()
        assert summary["total_searches"] == len(statuses)
