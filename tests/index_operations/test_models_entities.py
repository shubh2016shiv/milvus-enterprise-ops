"""
Comprehensive unit tests for index operation entities.

This module provides systematic testing of all entity models in the
index_operations.models.entities module, including IndexState enum,
IndexDescription, IndexBuildProgress, IndexStats, and IndexResult.
"""

import pytest

from milvus_ops.index_operations.models.entities import (
    IndexBuildProgress,
    IndexDescription,
    IndexResult,
    IndexState,
    IndexStats,
)

# ============================================================================
# Test IndexState Enum
# ============================================================================


@pytest.mark.unit
class TestIndexState:
    """
    Test IndexState enum.

    Coverage: IndexState enum values and comparisons.
    """

    def test_index_state_values(self):
        """
        Test IndexState enum values.

        Coverage: IndexState has correct enum values.
        """
        assert IndexState.NONE == "none"
        assert IndexState.CREATING == "creating"
        assert IndexState.CREATED == "created"
        assert IndexState.FAILED == "failed"

    def test_index_state_comparison(self):
        """
        Test IndexState enum comparison.

        Coverage: IndexState values can be compared.
        """
        assert IndexState.NONE != IndexState.CREATING
        assert IndexState.CREATING != IndexState.CREATED
        assert IndexState.CREATED != IndexState.FAILED


# ============================================================================
# Test IndexDescription
# ============================================================================


@pytest.mark.unit
class TestIndexDescription:
    """
    Test IndexDescription class.

    Coverage: IndexDescription initialization, properties, and methods.
    """

    def test_index_description_initialization(self):
        """
        Test IndexDescription initialization.

        Coverage: IndexDescription can be created with required fields.
        """
        description = IndexDescription(
            collection_name="test_collection",
            field_name="embedding",
            index_name="embedding_index",
            index_type="HNSW",
            metric_type="COSINE",
        )
        assert description.collection_name == "test_collection"
        assert description.field_name == "embedding"
        assert description.index_name == "embedding_index"
        assert description.index_type == "HNSW"
        assert description.metric_type == "COSINE"
        assert description.state == IndexState.NONE
        assert description.params == {}

    def test_index_description_with_params(self):
        """
        Test IndexDescription with parameters.

        Coverage: IndexDescription can include index parameters.
        """
        params = {"M": 16, "efConstruction": 200}
        description = IndexDescription(
            collection_name="test_collection",
            field_name="embedding",
            index_name="embedding_index",
            index_type="HNSW",
            metric_type="COSINE",
            params=params,
            state=IndexState.CREATED,
        )
        assert description.params == params
        assert description.state == IndexState.CREATED

    def test_index_description_index_size_mb(self):
        """
        Test IndexDescription index_size_mb property.

        Coverage: IndexDescription.index_size_mb converts bytes to MB.
        """
        description = IndexDescription(
            collection_name="test_collection",
            field_name="embedding",
            index_name="embedding_index",
            index_type="HNSW",
            metric_type="COSINE",
            index_size_bytes=1024 * 1024 * 50,  # 50 MB
        )
        assert description.index_size_mb == 50.0

    def test_index_description_index_size_mb_none(self):
        """
        Test IndexDescription index_size_mb when size is None.

        Coverage: IndexDescription.index_size_mb returns None when size_bytes is None.
        """
        description = IndexDescription(
            collection_name="test_collection",
            field_name="embedding",
            index_name="embedding_index",
            index_type="HNSW",
            metric_type="COSINE",
        )
        assert description.index_size_mb is None

    def test_index_description_is_available(self):
        """
        Test IndexDescription is_available property.

        Coverage: IndexDescription.is_available returns True for CREATED state.
        """
        description = IndexDescription(
            collection_name="test_collection",
            field_name="embedding",
            index_name="embedding_index",
            index_type="HNSW",
            metric_type="COSINE",
            state=IndexState.CREATED,
        )
        assert description.is_available is True

    def test_index_description_is_available_false(self):
        """
        Test IndexDescription is_available property for non-CREATED states.

        Coverage: IndexDescription.is_available returns False for non-CREATED states.
        """
        for state in [IndexState.NONE, IndexState.CREATING, IndexState.FAILED]:
            description = IndexDescription(
                collection_name="test_collection",
                field_name="embedding",
                index_name="embedding_index",
                index_type="HNSW",
                metric_type="COSINE",
                state=state,
            )
            assert description.is_available is False


# ============================================================================
# Test IndexBuildProgress
# ============================================================================


@pytest.mark.unit
class TestIndexBuildProgress:
    """
    Test IndexBuildProgress class.

    Coverage: IndexBuildProgress initialization, validation, and properties.
    """

    def test_index_build_progress_initialization(self):
        """
        Test IndexBuildProgress initialization.

        Coverage: IndexBuildProgress can be created with required fields.
        """
        progress = IndexBuildProgress(
            collection_name="test_collection",
            field_name="embedding",
            state=IndexState.CREATING,
        )
        assert progress.collection_name == "test_collection"
        assert progress.field_name == "embedding"
        assert progress.state == IndexState.CREATING
        assert progress.percentage == 0.0

    def test_index_build_progress_percentage_validation_negative(self):
        """
        Test IndexBuildProgress percentage validation for negative values.

        Coverage: IndexBuildProgress percentage is clamped to 0 for negative values.
        """
        progress = IndexBuildProgress(
            collection_name="test_collection",
            field_name="embedding",
            state=IndexState.CREATING,
            percentage=-10.0,
        )
        assert progress.percentage == 0.0

    def test_index_build_progress_percentage_validation_above_100(self):
        """
        Test IndexBuildProgress percentage validation for values above 100.

        Coverage: IndexBuildProgress percentage is clamped to 100 for values > 100.
        """
        progress = IndexBuildProgress(
            collection_name="test_collection",
            field_name="embedding",
            state=IndexState.CREATING,
            percentage=150.0,
        )
        assert progress.percentage == 100.0

    def test_index_build_progress_percentage_valid_range(self):
        """
        Test IndexBuildProgress percentage in valid range.

        Coverage: IndexBuildProgress percentage accepts values 0-100.
        """
        for pct in [0.0, 25.0, 50.0, 75.0, 100.0]:
            progress = IndexBuildProgress(
                collection_name="test_collection",
                field_name="embedding",
                state=IndexState.CREATING,
                percentage=pct,
            )
            assert progress.percentage == pct

    def test_index_build_progress_is_complete(self):
        """
        Test IndexBuildProgress is_complete property.

        Coverage: IndexBuildProgress.is_complete returns True for CREATED state.
        """
        progress = IndexBuildProgress(
            collection_name="test_collection",
            field_name="embedding",
            state=IndexState.CREATED,
            percentage=100.0,
        )
        assert progress.is_complete is True

    def test_index_build_progress_is_complete_false(self):
        """
        Test IndexBuildProgress is_complete property for non-CREATED states.

        Coverage: IndexBuildProgress.is_complete returns False for non-CREATED states.
        """
        for state in [IndexState.NONE, IndexState.CREATING, IndexState.FAILED]:
            progress = IndexBuildProgress(
                collection_name="test_collection",
                field_name="embedding",
                state=state,
            )
            assert progress.is_complete is False

    def test_index_build_progress_has_failed(self):
        """
        Test IndexBuildProgress has_failed property.

        Coverage: IndexBuildProgress.has_failed returns True for FAILED state.
        """
        progress = IndexBuildProgress(
            collection_name="test_collection",
            field_name="embedding",
            state=IndexState.FAILED,
            failed_reason="Memory error",
        )
        assert progress.has_failed is True

    def test_index_build_progress_has_failed_false(self):
        """
        Test IndexBuildProgress has_failed property for non-FAILED states.

        Coverage: IndexBuildProgress.has_failed returns False for non-FAILED states.
        """
        for state in [IndexState.NONE, IndexState.CREATING, IndexState.CREATED]:
            progress = IndexBuildProgress(
                collection_name="test_collection",
                field_name="embedding",
                state=state,
            )
            assert progress.has_failed is False

    def test_index_build_progress_formatted_eta_seconds(self):
        """
        Test IndexBuildProgress formatted_eta for seconds.

        Coverage: IndexBuildProgress.formatted_eta formats seconds correctly.
        """
        progress = IndexBuildProgress(
            collection_name="test_collection",
            field_name="embedding",
            state=IndexState.CREATING,
            estimated_remaining_time_seconds=45.5,
        )
        eta = progress.formatted_eta
        assert eta == "45.5 seconds"

    def test_index_build_progress_formatted_eta_minutes(self):
        """
        Test IndexBuildProgress formatted_eta for minutes.

        Coverage: IndexBuildProgress.formatted_eta formats minutes correctly.
        """
        progress = IndexBuildProgress(
            collection_name="test_collection",
            field_name="embedding",
            state=IndexState.CREATING,
            estimated_remaining_time_seconds=120.0,
        )
        eta = progress.formatted_eta
        assert eta == "2.0 minutes"

    def test_index_build_progress_formatted_eta_hours(self):
        """
        Test IndexBuildProgress formatted_eta for hours.

        Coverage: IndexBuildProgress.formatted_eta formats hours correctly.
        """
        progress = IndexBuildProgress(
            collection_name="test_collection",
            field_name="embedding",
            state=IndexState.CREATING,
            estimated_remaining_time_seconds=3660.0,  # 1 hour, 1 minute
        )
        eta = progress.formatted_eta
        assert "1 hours" in eta
        assert "1 minutes" in eta

    def test_index_build_progress_formatted_eta_none(self):
        """
        Test IndexBuildProgress formatted_eta when eta is None.

        Coverage: IndexBuildProgress.formatted_eta returns None when eta_seconds is None.
        """
        progress = IndexBuildProgress(
            collection_name="test_collection",
            field_name="embedding",
            state=IndexState.CREATING,
        )
        assert progress.formatted_eta is None


# ============================================================================
# Test IndexStats
# ============================================================================


@pytest.mark.unit
class TestIndexStats:
    """
    Test IndexStats class.

    Coverage: IndexStats initialization and properties.
    """

    def test_index_stats_initialization(self):
        """
        Test IndexStats initialization.

        Coverage: IndexStats can be created with required fields.
        """
        stats = IndexStats(
            collection_name="test_collection",
            field_name="embedding",
            index_type="HNSW",
        )
        assert stats.collection_name == "test_collection"
        assert stats.field_name == "embedding"
        assert stats.index_type == "HNSW"
        assert stats.query_latency_ms is None
        assert stats.memory_usage_bytes is None
        assert stats.disk_usage_bytes is None

    def test_index_stats_memory_usage_mb(self):
        """
        Test IndexStats memory_usage_mb property.

        Coverage: IndexStats.memory_usage_mb converts bytes to MB.
        """
        stats = IndexStats(
            collection_name="test_collection",
            field_name="embedding",
            index_type="HNSW",
            memory_usage_bytes=1024 * 1024 * 100,  # 100 MB
        )
        assert stats.memory_usage_mb == 100.0

    def test_index_stats_memory_usage_mb_none(self):
        """
        Test IndexStats memory_usage_mb when size is None.

        Coverage: IndexStats.memory_usage_mb returns None when bytes is None.
        """
        stats = IndexStats(
            collection_name="test_collection",
            field_name="embedding",
            index_type="HNSW",
        )
        assert stats.memory_usage_mb is None

    def test_index_stats_disk_usage_mb(self):
        """
        Test IndexStats disk_usage_mb property.

        Coverage: IndexStats.disk_usage_mb converts bytes to MB.
        """
        stats = IndexStats(
            collection_name="test_collection",
            field_name="embedding",
            index_type="HNSW",
            disk_usage_bytes=1024 * 1024 * 80,  # 80 MB
        )
        assert stats.disk_usage_mb == 80.0

    def test_index_stats_disk_usage_mb_none(self):
        """
        Test IndexStats disk_usage_mb when size is None.

        Coverage: IndexStats.disk_usage_mb returns None when bytes is None.
        """
        stats = IndexStats(
            collection_name="test_collection",
            field_name="embedding",
            index_type="HNSW",
        )
        assert stats.disk_usage_mb is None


# ============================================================================
# Test IndexResult
# ============================================================================


@pytest.mark.unit
class TestIndexResult:
    """
    Test IndexResult class.

    Coverage: IndexResult initialization and properties.
    """

    def test_index_result_initialization_success(self):
        """
        Test IndexResult initialization for success.

        Coverage: IndexResult can be created for successful operations.
        """
        result = IndexResult(
            success=True,
            collection_name="test_collection",
            field_name="embedding",
            operation="create",
            state=IndexState.CREATED,
        )
        assert result.success is True
        assert result.collection_name == "test_collection"
        assert result.field_name == "embedding"
        assert result.operation == "create"
        assert result.state == IndexState.CREATED
        assert result.error_message is None

    def test_index_result_initialization_failure(self):
        """
        Test IndexResult initialization for failure.

        Coverage: IndexResult can be created for failed operations.
        """
        result = IndexResult(
            success=False,
            collection_name="test_collection",
            field_name="embedding",
            operation="create",
            state=IndexState.FAILED,
            error_message="Build failed",
        )
        assert result.success is False
        assert result.state == IndexState.FAILED
        assert result.error_message == "Build failed"

    def test_index_result_is_complete_created(self):
        """
        Test IndexResult is_complete property for CREATED state.

        Coverage: IndexResult.is_complete returns True for CREATED state.
        """
        result = IndexResult(
            success=True,
            collection_name="test_collection",
            field_name="embedding",
            operation="create",
            state=IndexState.CREATED,
        )
        assert result.is_complete is True

    def test_index_result_is_complete_failed(self):
        """
        Test IndexResult is_complete property for FAILED state.

        Coverage: IndexResult.is_complete returns True for FAILED state.
        """
        result = IndexResult(
            success=False,
            collection_name="test_collection",
            field_name="embedding",
            operation="create",
            state=IndexState.FAILED,
        )
        assert result.is_complete is True

    def test_index_result_is_complete_false(self):
        """
        Test IndexResult is_complete property for non-terminal states.

        Coverage: IndexResult.is_complete returns False for non-terminal states.
        """
        for state in [IndexState.NONE, IndexState.CREATING]:
            result = IndexResult(
                success=True,
                collection_name="test_collection",
                field_name="embedding",
                operation="create",
                state=state,
            )
            assert result.is_complete is False

    def test_index_result_is_in_progress(self):
        """
        Test IndexResult is_in_progress property.

        Coverage: IndexResult.is_in_progress returns True for CREATING state.
        """
        result = IndexResult(
            success=True,
            collection_name="test_collection",
            field_name="embedding",
            operation="create",
            state=IndexState.CREATING,
        )
        assert result.is_in_progress is True

    def test_index_result_is_in_progress_false(self):
        """
        Test IndexResult is_in_progress property for non-CREATING states.

        Coverage: IndexResult.is_in_progress returns False for non-CREATING states.
        """
        for state in [IndexState.NONE, IndexState.CREATED, IndexState.FAILED]:
            result = IndexResult(
                success=True,
                collection_name="test_collection",
                field_name="embedding",
                operation="create",
                state=state,
            )
            assert result.is_in_progress is False

    def test_index_result_execution_time(self):
        """
        Test IndexResult execution_time_ms.

        Coverage: IndexResult can include execution time.
        """
        result = IndexResult(
            success=True,
            collection_name="test_collection",
            field_name="embedding",
            operation="create",
            state=IndexState.CREATED,
            execution_time_ms=1234.5,
        )
        assert result.execution_time_ms == 1234.5
