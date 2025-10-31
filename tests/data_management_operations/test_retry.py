"""
Comprehensive unit tests for retry utilities.

This module provides systematic testing of retry utilities including
retry_on_transient_error, with_transient_retry decorator, and is_transient_milvus_error.
"""

import asyncio

import pytest

from milvus_ops.data_management_operations import DataOperationConfig
from milvus_ops.data_management_operations.data_ops_exceptions import TransientOperationError
from milvus_ops.data_management_operations.utils.retry import (
    is_transient_milvus_error,
    retry_on_transient_error,
    with_transient_retry,
)

# ============================================================================
# Test retry_on_transient_error
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestRetryOnTransientError:
    """
    Test retry_on_transient_error function.

    Coverage: Retry logic for transient errors.
    """

    async def test_retry_on_transient_error_success(self):
        """
        Test retry_on_transient_error with successful operation.

        Coverage: retry_on_transient_error() executes operation without retries on success.
        """
        config = DataOperationConfig(retry_transient_errors=True, max_transient_retries=3)

        call_count = 0

        async def successful_operation():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await retry_on_transient_error(successful_operation, config, "test_operation")
        assert result == "success"
        assert call_count == 1

    async def test_retry_on_transient_error_transient_succeeds_after_retries(self):
        """
        Test retry_on_transient_error with transient errors that succeed after retries.

        Coverage: retry_on_transient_error() retries on TransientOperationError and succeeds.
        """
        config = DataOperationConfig(
            retry_transient_errors=True,
            max_transient_retries=3,
            transient_retry_delay=0.01,
        )

        call_count = 0

        async def operation_with_transient_error():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TransientOperationError(f"Transient error {call_count}")
            return "success"

        result = await retry_on_transient_error(
            operation_with_transient_error, config, "test_operation"
        )
        assert result == "success"
        assert call_count == 3

    async def test_retry_on_transient_error_exhausts_retries(self):
        """
        Test retry_on_transient_error exhausts all retries.

        Coverage: retry_on_transient_error() raises TransientOperationError after
        all retries exhausted.
        """
        config = DataOperationConfig(
            retry_transient_errors=True,
            max_transient_retries=3,
            transient_retry_delay=0.01,
        )

        call_count = 0

        async def always_failing_operation():
            nonlocal call_count
            call_count += 1
            raise TransientOperationError(f"Transient error {call_count}")

        with pytest.raises(TransientOperationError, match="Transient error 3"):
            await retry_on_transient_error(always_failing_operation, config, "test_operation")

        assert call_count == 3

    async def test_retry_on_transient_error_non_transient_propagates(self):
        """
        Test retry_on_transient_error propagates non-transient errors immediately.

        Coverage: retry_on_transient_error() propagates non-transient errors without retry.
        """
        config = DataOperationConfig(retry_transient_errors=True, max_transient_retries=3)

        call_count = 0

        async def operation_with_non_transient_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Non-transient error")

        with pytest.raises(ValueError, match="Non-transient error"):
            await retry_on_transient_error(
                operation_with_non_transient_error, config, "test_operation"
            )

        assert call_count == 1  # Should not retry

    async def test_retry_on_transient_error_retry_disabled(self):
        """
        Test retry_on_transient_error with retry disabled.

        Coverage: retry_on_transient_error() executes once when retry is disabled.
        """
        config = DataOperationConfig(retry_transient_errors=False, max_transient_retries=3)

        call_count = 0

        async def operation_with_transient_error():
            nonlocal call_count
            call_count += 1
            raise TransientOperationError("Transient error")

        with pytest.raises(TransientOperationError, match="Transient error"):
            await retry_on_transient_error(operation_with_transient_error, config, "test_operation")

        assert call_count == 1  # Should not retry when disabled

    async def test_retry_on_transient_error_linear_backoff(self):
        """
        Test retry_on_transient_error uses linear backoff.

        Coverage: retry_on_transient_error() uses linear backoff delay calculation.
        """
        config = DataOperationConfig(
            retry_transient_errors=True,
            max_transient_retries=3,
            transient_retry_delay=0.1,
        )

        delays = []

        async def operation_with_transient_error():
            if len(delays) < 2:
                delays.append(asyncio.get_event_loop().time())
                raise TransientOperationError("Transient error")
            return "success"

        await retry_on_transient_error(operation_with_transient_error, config, "test_operation")

        # Check that delays increase (linear backoff)
        if len(delays) >= 2:
            actual_delay = delays[1] - delays[0]
            # Should be approximately transient_retry_delay * 1 (first retry)
            assert actual_delay >= 0.05  # Allow some tolerance

    async def test_retry_on_transient_error_with_args_and_kwargs(self):
        """
        Test retry_on_transient_error with operation arguments.

        Coverage: retry_on_transient_error() passes arguments correctly to operation.
        """
        config = DataOperationConfig(retry_transient_errors=True, max_transient_retries=3)

        async def operation_with_args(arg1, arg2, kwarg1=None):
            return f"{arg1}_{arg2}_{kwarg1}"

        result = await retry_on_transient_error(
            operation_with_args, config, "test_operation", "value1", "value2", kwarg1="kwarg_value"
        )
        assert result == "value1_value2_kwarg_value"


# ============================================================================
# Test with_transient_retry Decorator
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestWithTransientRetryDecorator:
    """
    Test with_transient_retry decorator.

    Coverage: Decorator wrapping behavior for async methods.
    """

    async def test_with_transient_retry_success(self):
        """
        Test with_transient_retry with successful operation.

        Coverage: with_transient_retry() decorator executes successfully.
        """

        class TestClass:
            def __init__(self):
                self._config = DataOperationConfig(
                    retry_transient_errors=True, max_transient_retries=3
                )

            @with_transient_retry("test_operation")
            async def test_method(self):
                return "success"

        obj = TestClass()
        result = await obj.test_method()
        assert result == "success"

    async def test_with_transient_retry_transient_error(self):
        """
        Test with_transient_retry with transient error.

        Coverage: with_transient_retry() decorator retries on TransientOperationError.
        """

        class TestClass:
            def __init__(self):
                self._config = DataOperationConfig(
                    retry_transient_errors=True,
                    max_transient_retries=3,
                    transient_retry_delay=0.01,
                )
                self.call_count = 0

            @with_transient_retry("test_operation")
            async def test_method(self):
                self.call_count += 1
                if self.call_count < 2:
                    raise TransientOperationError("Transient error")
                return "success"

        obj = TestClass()
        result = await obj.test_method()
        assert result == "success"
        assert obj.call_count == 2

    async def test_with_transient_retry_no_config(self):
        """
        Test with_transient_retry without _config attribute.

        Coverage: with_transient_retry() decorator executes without retry when no config.
        """

        class TestClass:
            @with_transient_retry("test_operation")
            async def test_method(self):
                return "success"

        obj = TestClass()
        result = await obj.test_method()
        assert result == "success"

    async def test_with_transient_retry_with_args(self):
        """
        Test with_transient_retry with method arguments.

        Coverage: with_transient_retry() decorator preserves method arguments.
        """

        class TestClass:
            def __init__(self):
                self._config = DataOperationConfig(
                    retry_transient_errors=True, max_transient_retries=3
                )

            @with_transient_retry("test_operation")
            async def test_method(self, arg1, arg2, kwarg1=None):
                return f"{arg1}_{arg2}_{kwarg1}"

        obj = TestClass()
        result = await obj.test_method("value1", "value2", kwarg1="kwarg_value")
        assert result == "value1_value2_kwarg_value"


# ============================================================================
# Test is_transient_milvus_error
# ============================================================================


@pytest.mark.unit
class TestIsTransientMilvusError:
    """
    Test is_transient_milvus_error function.

    Coverage: Pattern matching for transient error messages.
    """

    @pytest.mark.parametrize(
        "error_message,expected",
        [
            ("schema not ready", True),
            ("collection is being loaded", True),
            ("collection is loading", True),
            ("temporary unavailable", True),
            ("temporarily unavailable", True),
            ("rate limit exceeded", True),
            ("too many requests", True),
            ("Schema not ready", True),  # Case insensitive
            ("SCHEMA NOT READY", True),  # Case insensitive
            ("collection is being loaded now", True),  # Substring match
            ("collection is loading data", True),  # Substring match
        ],
    )
    def test_is_transient_milvus_error_patterns(self, error_message, expected):
        """
        Test is_transient_milvus_error with various error patterns.

        Coverage: is_transient_milvus_error() matches transient error patterns.
        """
        error = Exception(error_message)
        result = is_transient_milvus_error(error)
        assert result == expected

    def test_is_transient_milvus_error_non_transient(self):
        """
        Test is_transient_milvus_error with non-transient errors.

        Coverage: is_transient_milvus_error() returns False for non-transient errors.
        """
        error = Exception("Collection not found")
        result = is_transient_milvus_error(error)
        assert result is False

    def test_is_transient_milvus_error_invalid_type(self):
        """
        Test is_transient_milvus_error with invalid type.

        Coverage: is_transient_milvus_error() returns False for non-Exception types.
        """
        result = is_transient_milvus_error("not an exception")
        assert result is False

    def test_is_transient_milvus_error_empty_message(self):
        """
        Test is_transient_milvus_error with empty message.

        Coverage: is_transient_milvus_error() returns False for empty error messages.
        """
        error = Exception("")
        result = is_transient_milvus_error(error)
        assert result is False

    def test_is_transient_milvus_error_none(self):
        """
        Test is_transient_milvus_error with None.

        Coverage: is_transient_milvus_error() returns False for None.
        """
        result = is_transient_milvus_error(None)
        assert result is False

    def test_is_transient_milvus_error_milvus_exception(self):
        """
        Test is_transient_milvus_error with PyMilvus exception.

        Coverage: is_transient_milvus_error() works with PyMilvus exceptions.
        """
        from pymilvus.exceptions import MilvusException

        error = MilvusException("schema not ready")
        result = is_transient_milvus_error(error)
        assert result is True

        error = MilvusException("Collection not found")
        result = is_transient_milvus_error(error)
        assert result is False

    def test_is_transient_milvus_error_multiple_patterns(self):
        """
        Test is_transient_milvus_error with multiple patterns.

        Coverage: is_transient_milvus_error() matches any transient pattern.
        """
        # Error message containing multiple patterns
        error = Exception("schema not ready and collection is loading")
        result = is_transient_milvus_error(error)
        assert result is True  # Matches first pattern
