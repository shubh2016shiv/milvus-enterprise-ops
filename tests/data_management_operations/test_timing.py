"""
Comprehensive unit tests for timing utilities.

This module provides systematic testing of timing utilities including
PerformanceTimer, TimingResult, BatchTimingResult, and the time_operation decorator.
"""

import asyncio
from datetime import datetime
import time

import pytest

from milvus_ops.data_management_operations.utils.timing import (
    BatchTimingResult,
    PerformanceTimer,
    TimingResult,
    time_operation,
)

# ============================================================================
# Test TimingResult
# ============================================================================


@pytest.mark.unit
class TestTimingResult:
    """
    Test TimingResult model.

    Coverage: TimingResult model creation and attribute access.
    """

    def test_timing_result_creation(self):
        """
        Test TimingResult creation with all fields.

        Coverage: TimingResult can be created with all attributes.
        """
        metadata = {"collection_name": "test_collection", "batch_size": 100}
        result = TimingResult(
            operation_name="insert",
            execution_time=0.5,
            timestamp=datetime.now(),
            success=True,
            metadata=metadata,
        )
        assert result.operation_name == "insert"
        assert result.execution_time == 0.5
        assert isinstance(result.timestamp, datetime)
        assert result.success is True
        assert result.metadata == metadata

    def test_timing_result_default_timestamp(self):
        """
        Test TimingResult default timestamp.

        Coverage: TimingResult uses current timestamp if not provided.
        """
        before = datetime.now()
        result = TimingResult(operation_name="insert", execution_time=0.5)
        after = datetime.now()
        assert before <= result.timestamp <= after

    def test_timing_result_default_success(self):
        """
        Test TimingResult default success flag.

        Coverage: TimingResult defaults success to True.
        """
        result = TimingResult(operation_name="insert", execution_time=0.5)
        assert result.success is True

    def test_timing_result_default_metadata(self):
        """
        Test TimingResult default metadata.

        Coverage: TimingResult defaults metadata to empty dict.
        """
        result = TimingResult(operation_name="insert", execution_time=0.5)
        assert isinstance(result.metadata, dict)
        assert len(result.metadata) == 0

    def test_timing_result_with_metadata(self):
        """
        Test TimingResult with metadata.

        Coverage: TimingResult stores metadata correctly.
        """
        metadata = {"key1": "value1", "key2": 123, "key3": [1, 2, 3]}
        result = TimingResult(operation_name="insert", execution_time=0.5, metadata=metadata)
        assert result.metadata == metadata
        assert result.metadata["key1"] == "value1"
        assert result.metadata["key2"] == 123


# ============================================================================
# Test BatchTimingResult
# ============================================================================


@pytest.mark.unit
class TestBatchTimingResult:
    """
    Test BatchTimingResult model.

    Coverage: BatchTimingResult statistics calculation.
    """

    def test_batch_timing_result_creation(self):
        """
        Test BatchTimingResult creation.

        Coverage: BatchTimingResult can be created with all attributes.
        """
        result = BatchTimingResult(
            operation_name="insert",
            total_operations=10,
            successful_operations=8,
            failed_operations=2,
            average_execution_time=0.5,
            median_execution_time=0.48,
            min_execution_time=0.3,
            max_execution_time=0.7,
            p95_execution_time=0.65,
        )
        assert result.operation_name == "insert"
        assert result.total_operations == 10
        assert result.successful_operations == 8
        assert result.failed_operations == 2
        assert result.average_execution_time == 0.5
        assert result.median_execution_time == 0.48
        assert result.min_execution_time == 0.3
        assert result.max_execution_time == 0.7
        assert result.p95_execution_time == 0.65

    def test_batch_timing_result_success_rate(self):
        """
        Test BatchTimingResult success_rate property.

        Coverage: success_rate property calculation.
        """
        # 100% success rate
        result = BatchTimingResult(
            operation_name="insert",
            total_operations=10,
            successful_operations=10,
            failed_operations=0,
            average_execution_time=0.5,
            median_execution_time=0.5,
            min_execution_time=0.3,
            max_execution_time=0.7,
            p95_execution_time=0.65,
        )
        assert result.success_rate == 100.0

        # 80% success rate
        result = BatchTimingResult(
            operation_name="insert",
            total_operations=10,
            successful_operations=8,
            failed_operations=2,
            average_execution_time=0.5,
            median_execution_time=0.48,
            min_execution_time=0.3,
            max_execution_time=0.7,
            p95_execution_time=0.65,
        )
        assert result.success_rate == 80.0

        # 0% success rate
        result = BatchTimingResult(
            operation_name="insert",
            total_operations=10,
            successful_operations=0,
            failed_operations=10,
            average_execution_time=0.5,
            median_execution_time=0.5,
            min_execution_time=0.3,
            max_execution_time=0.7,
            p95_execution_time=0.65,
        )
        assert result.success_rate == 0.0

    def test_batch_timing_result_success_rate_zero_total(self):
        """
        Test BatchTimingResult success_rate with zero total operations.

        Coverage: success_rate returns 0.0 when total_operations is 0.
        """
        result = BatchTimingResult(
            operation_name="insert",
            total_operations=0,
            successful_operations=0,
            failed_operations=0,
            average_execution_time=0.0,
            median_execution_time=0.0,
            min_execution_time=0.0,
            max_execution_time=0.0,
            p95_execution_time=0.0,
        )
        assert result.success_rate == 0.0


# ============================================================================
# Test PerformanceTimer
# ============================================================================


@pytest.mark.unit
class TestPerformanceTimer:
    """
    Test PerformanceTimer class.

    Coverage: PerformanceTimer initialization and operations.
    """

    def test_performance_timer_init_with_logging(self):
        """
        Test PerformanceTimer initialization with logging enabled.

        Coverage: PerformanceTimer.__init__() with enable_logging=True.
        """
        timer = PerformanceTimer(enable_logging=True)
        assert timer._enable_logging is True
        assert isinstance(timer._timing_history, list)
        assert len(timer._timing_history) == 0

    def test_performance_timer_init_without_logging(self):
        """
        Test PerformanceTimer initialization with logging disabled.

        Coverage: PerformanceTimer.__init__() with enable_logging=False.
        """
        timer = PerformanceTimer(enable_logging=False)
        assert timer._enable_logging is False
        assert isinstance(timer._timing_history, list)
        assert len(timer._timing_history) == 0

    def test_performance_timer_init_default_logging(self):
        """
        Test PerformanceTimer initialization with default logging.

        Coverage: PerformanceTimer.__init__() defaults enable_logging=True.
        """
        timer = PerformanceTimer()
        assert timer._enable_logging is True


@pytest.mark.unit
@pytest.mark.asyncio
class TestPerformanceTimerTimeOperation:
    """
    Test PerformanceTimer.time_operation context manager.

    Coverage: PerformanceTimer timing operations (success and failure cases).
    """

    async def test_time_operation_success(self):
        """
        Test time_operation with successful operation.

        Coverage: time_operation() records successful operation timing.
        """
        timer = PerformanceTimer(enable_logging=False)
        metadata = {"test_key": "test_value"}

        async with timer.time_operation(operation_name="test_op", metadata=metadata) as result:
            await asyncio.sleep(0.01)  # Small delay to ensure timing is recorded
            assert result.operation_name == "test_op"
            assert result.metadata == metadata
            assert result.success is True  # Will be set at end of context

        assert len(timer._timing_history) == 1
        timing = timer._timing_history[0]
        assert timing.operation_name == "test_op"
        assert timing.success is True
        assert timing.execution_time > 0
        assert timing.metadata == metadata

    async def test_time_operation_failure(self):
        """
        Test time_operation with failed operation.

        Coverage: time_operation() records failed operation timing.
        """
        import asyncio

        timer = PerformanceTimer(enable_logging=False)

        with pytest.raises(ValueError):
            async with timer.time_operation(operation_name="test_op"):
                await asyncio.sleep(0.001)  # Small delay to ensure measurable execution time
                raise ValueError("Test error")

        assert len(timer._timing_history) == 1
        timing = timer._timing_history[0]
        assert timing.operation_name == "test_op"
        assert timing.success is False
        assert timing.execution_time > 0

    async def test_time_operation_multiple_operations(self):
        """
        Test time_operation with multiple operations.

        Coverage: time_operation() records multiple operations in history.
        """
        timer = PerformanceTimer(enable_logging=False)

        async with timer.time_operation(operation_name="op1"):
            await asyncio.sleep(0.01)

        async with timer.time_operation(operation_name="op2"):
            await asyncio.sleep(0.01)

        async with timer.time_operation(operation_name="op3"):
            await asyncio.sleep(0.01)

        assert len(timer._timing_history) == 3
        assert timer._timing_history[0].operation_name == "op1"
        assert timer._timing_history[1].operation_name == "op2"
        assert timer._timing_history[2].operation_name == "op3"


@pytest.mark.unit
class TestPerformanceTimerHistory:
    """
    Test PerformanceTimer history methods.

    Coverage: PerformanceTimer timing history access.
    """

    @pytest.mark.asyncio
    async def test_get_timing_history_empty(self):
        """
        Test get_timing_history with empty history.

        Coverage: get_timing_history() returns empty list when no operations.
        """
        timer = PerformanceTimer(enable_logging=False)
        history = timer.get_timing_history()
        assert isinstance(history, list)
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_get_timing_history_single(self):
        """
        Test get_timing_history with single operation.

        Coverage: get_timing_history() returns list with single timing result.
        """
        timer = PerformanceTimer(enable_logging=False)

        async with timer.time_operation(operation_name="test_op"):
            await asyncio.sleep(0.01)

        history = timer.get_timing_history()
        assert len(history) == 1
        assert history[0].operation_name == "test_op"

    @pytest.mark.asyncio
    async def test_get_timing_history_multiple(self):
        """
        Test get_timing_history with multiple operations.

        Coverage: get_timing_history() returns all timing results.
        """
        timer = PerformanceTimer(enable_logging=False)

        for i in range(5):
            async with timer.time_operation(operation_name=f"op_{i}"):
                await asyncio.sleep(0.01)

        history = timer.get_timing_history()
        assert len(history) == 5
        for i, timing in enumerate(history):
            assert timing.operation_name == f"op_{i}"

    @pytest.mark.asyncio
    async def test_get_timing_history_returns_copy(self):
        """
        Test get_timing_history returns a copy.

        Coverage: get_timing_history() returns a copy, not the original list.
        """
        timer = PerformanceTimer(enable_logging=False)

        async with timer.time_operation(operation_name="test_op"):
            await asyncio.sleep(0.01)

        history1 = timer.get_timing_history()
        history2 = timer.get_timing_history()
        assert history1 is not history2  # Different objects
        assert history1 == history2  # Same content


@pytest.mark.unit
class TestPerformanceTimerStats:
    """
    Test PerformanceTimer statistics methods.

    Coverage: PerformanceTimer operation statistics calculation.
    """

    @pytest.mark.asyncio
    async def test_get_operation_stats_no_operations(self):
        """
        Test get_operation_stats with no operations.

        Coverage: get_operation_stats() returns None when no operations found.
        """
        timer = PerformanceTimer(enable_logging=False)
        stats = timer.get_operation_stats("nonexistent_op")
        assert stats is None

    @pytest.mark.asyncio
    async def test_get_operation_stats_single_operation(self):
        """
        Test get_operation_stats with single operation.

        Coverage: get_operation_stats() calculates stats for single operation.
        """
        timer = PerformanceTimer(enable_logging=False)

        async with timer.time_operation(operation_name="insert"):
            await asyncio.sleep(0.05)

        stats = timer.get_operation_stats("insert")
        assert stats is not None
        assert stats.operation_name == "insert"
        assert stats.total_operations == 1
        assert stats.successful_operations == 1
        assert stats.failed_operations == 0
        assert stats.average_execution_time > 0
        assert stats.min_execution_time > 0
        assert stats.max_execution_time > 0
        assert stats.median_execution_time > 0

    @pytest.mark.asyncio
    async def test_get_operation_stats_multiple_operations(self):
        """
        Test get_operation_stats with multiple operations.

        Coverage: get_operation_stats() calculates aggregated stats for multiple operations.
        """
        timer = PerformanceTimer(enable_logging=False)

        execution_times = [0.01, 0.02, 0.03, 0.04, 0.05]
        for delay in execution_times:
            async with timer.time_operation(operation_name="insert"):
                await asyncio.sleep(delay)

        stats = timer.get_operation_stats("insert")
        assert stats is not None
        assert stats.operation_name == "insert"
        assert stats.total_operations == 5
        assert stats.successful_operations == 5
        assert stats.failed_operations == 0
        assert stats.average_execution_time > 0
        assert stats.min_execution_time > 0
        assert stats.max_execution_time > 0
        assert stats.p95_execution_time > 0

    @pytest.mark.asyncio
    async def test_get_operation_stats_with_failures(self):
        """
        Test get_operation_stats with failed operations.

        Coverage: get_operation_stats() tracks successful and failed operations.
        """
        timer = PerformanceTimer(enable_logging=False)

        # Successful operations
        for _ in range(3):
            async with timer.time_operation(operation_name="insert"):
                await asyncio.sleep(0.01)

        # Failed operations
        for _ in range(2):
            try:
                async with timer.time_operation(operation_name="insert"):
                    raise ValueError("Test error")
            except ValueError:
                pass

        stats = timer.get_operation_stats("insert")
        assert stats is not None
        assert stats.total_operations == 5
        assert stats.successful_operations == 3
        assert stats.failed_operations == 2

    @pytest.mark.asyncio
    async def test_get_summary_empty(self):
        """
        Test get_summary with empty history.

        Coverage: get_summary() returns empty dict when no operations.
        """
        timer = PerformanceTimer(enable_logging=False)
        summary = timer.get_summary()
        assert isinstance(summary, dict)
        assert len(summary) == 0

    @pytest.mark.asyncio
    async def test_get_summary_single_operation_type(self):
        """
        Test get_summary with single operation type.

        Coverage: get_summary() returns stats for single operation type.
        """
        timer = PerformanceTimer(enable_logging=False)

        for _ in range(3):
            async with timer.time_operation(operation_name="insert"):
                await asyncio.sleep(0.01)

        summary = timer.get_summary()
        assert len(summary) == 1
        assert "insert" in summary
        assert summary["insert"].total_operations == 3

    @pytest.mark.asyncio
    async def test_get_summary_multiple_operation_types(self):
        """
        Test get_summary with multiple operation types.

        Coverage: get_summary() returns stats for all operation types.
        """
        timer = PerformanceTimer(enable_logging=False)

        # Insert operations
        for _ in range(2):
            async with timer.time_operation(operation_name="insert"):
                await asyncio.sleep(0.01)

        # Upsert operations
        for _ in range(3):
            async with timer.time_operation(operation_name="upsert"):
                await asyncio.sleep(0.01)

        # Delete operations
        for _ in range(1):
            async with timer.time_operation(operation_name="delete"):
                await asyncio.sleep(0.01)

        summary = timer.get_summary()
        assert len(summary) == 3
        assert "insert" in summary
        assert "upsert" in summary
        assert "delete" in summary
        assert summary["insert"].total_operations == 2
        assert summary["upsert"].total_operations == 3
        assert summary["delete"].total_operations == 1


@pytest.mark.unit
class TestPerformanceTimerClearHistory:
    """
    Test PerformanceTimer clear_history method.

    Coverage: PerformanceTimer history clearing.
    """

    @pytest.mark.asyncio
    async def test_clear_history(self):
        """
        Test clear_history removes all timing history.

        Coverage: clear_history() removes all timing results.
        """
        timer = PerformanceTimer(enable_logging=False)

        # Add some operations
        for _ in range(5):
            async with timer.time_operation(operation_name="test_op"):
                await asyncio.sleep(0.01)

        assert len(timer._timing_history) == 5

        # Clear history
        timer.clear_history()

        assert len(timer._timing_history) == 0
        assert timer.get_timing_history() == []

    @pytest.mark.asyncio
    async def test_clear_history_empty(self):
        """
        Test clear_history with empty history.

        Coverage: clear_history() works correctly with empty history.
        """
        timer = PerformanceTimer(enable_logging=False)
        timer.clear_history()
        assert len(timer._timing_history) == 0


# ============================================================================
# Test time_operation Decorator
# ============================================================================


@pytest.mark.unit
class TestTimeOperationDecorator:
    """
    Test time_operation decorator.

    Coverage: time_operation decorator for synchronous functions.
    """

    def test_time_operation_decorator_success(self):
        """
        Test time_operation decorator with successful function.

        Coverage: time_operation() decorator records successful function execution.
        """

        @time_operation(operation_name="test_function", metadata={"test": True})
        def test_func(x, y):
            time.sleep(0.01)
            return x + y

        result = test_func(2, 3)
        assert result == 5

    def test_time_operation_decorator_failure(self):
        """
        Test time_operation decorator with failing function.

        Coverage: time_operation() decorator records failed function execution.
        """

        @time_operation(operation_name="test_function")
        def test_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            test_func()

    def test_time_operation_decorator_with_args(self):
        """
        Test time_operation decorator with function arguments.

        Coverage: time_operation() decorator preserves function arguments.
        """

        @time_operation(operation_name="test_function")
        def test_func(a, b, c=10):
            return a + b + c

        result = test_func(1, 2, c=3)
        assert result == 6

    def test_time_operation_decorator_with_kwargs(self):
        """
        Test time_operation decorator with keyword arguments.

        Coverage: time_operation() decorator preserves keyword arguments.
        """

        @time_operation(operation_name="test_function")
        def test_func(**kwargs):
            return sum(kwargs.values())

        result = test_func(a=1, b=2, c=3)
        assert result == 6
