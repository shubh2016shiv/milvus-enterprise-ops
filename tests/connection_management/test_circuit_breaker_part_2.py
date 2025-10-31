"""
Continued unit tests for circuit breaker implementation.

This file contains additional tests for metrics, reset functionality,
edge cases, and string representations.
"""

import asyncio
from contextlib import suppress
from unittest.mock import patch

import pytest

from milvus_ops.connection_management.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitState,
    MilvusCircuitBreaker,
)
from milvus_ops.connection_management.connection_exceptions import (
    ConnectionError,
    ServerUnavailableError,
)

# ============================================================================
# Metrics Tests (continued from main file)
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
class TestCircuitBreakerMetrics:
    """
    Test metrics collection and reporting.

    Coverage: Verifies get_metrics() returns comprehensive metrics
    including state, counters, timing, and success rate.
    """

    async def test_metrics_in_closed_state(self):
        """
        Test metrics in CLOSED state.

        Coverage: get_metrics() with CLOSED state.
        """
        breaker = MilvusCircuitBreaker(name="test_breaker")
        metrics = breaker.get_metrics()

        assert metrics["name"] == "test_breaker"
        assert metrics["state"] == "closed"
        assert metrics["configuration"]["failure_threshold"] == 5
        assert metrics["counters"]["total_requests"] == 0
        assert metrics["counters"]["failure_count"] == 0
        assert metrics["current_state"]["half_open_requests"] == 0

    async def test_metrics_in_open_state(self):
        """
        Test metrics in OPEN state.

        Coverage: get_metrics() with OPEN state.
        """
        breaker = MilvusCircuitBreaker()
        await breaker._transition_to_open()

        metrics = breaker.get_metrics()
        assert metrics["state"] == "open"
        assert metrics["timing"]["time_since_last_failure"] is not None

    async def test_metrics_success_rate_calculation(self):
        """
        Test success rate calculation in metrics.

        Coverage: get_metrics() success_rate_percent calculation.
        """
        breaker = MilvusCircuitBreaker()

        # Successful operations
        async def success_op() -> str:
            return "success"

        for _ in range(8):
            await breaker.execute_milvus_operation(success_op)

        # Failed operations
        async def fail_op() -> None:
            raise ConnectionError("failure")

        for _ in range(2):
            with suppress(ConnectionError):
                await breaker.execute_milvus_operation(fail_op)

        metrics = breaker.get_metrics()
        # 8 successes out of 10 requests = 80%
        assert metrics["current_state"]["success_rate_percent"] == 80.0

    async def test_metrics_timing_information(self):
        """
        Test timing information in metrics.

        Coverage: get_metrics() timing calculations.
        """
        # Mock time throughout the entire test to ensure consistent timing
        with patch("time.monotonic") as mock_monotonic:
            mock_monotonic.return_value = 1000.0

            breaker = MilvusCircuitBreaker()

            # Trigger a failure to set last_failure_time
            async def fail_op() -> None:
                raise ConnectionError("failure")

            with suppress(ConnectionError):
                await breaker.execute_milvus_operation(fail_op)

            # Advance time
            mock_monotonic.return_value = 1005.0

            metrics = breaker.get_metrics()
            assert metrics["timing"]["last_failure_time"] == 1000.0
            assert metrics["timing"]["time_since_last_failure"] == 5.0
            assert metrics["timing"]["time_since_state_change"] > 0


# ============================================================================
# Manual Reset Tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
class TestCircuitBreakerReset:
    """
    Test manual reset functionality.

    Coverage: Verifies reset() method resets circuit breaker to CLOSED state.
    """

    async def test_reset_from_open_state(self):
        """
        Test reset from OPEN state.

        Coverage: reset() transitioning from OPEN to CLOSED.
        """
        breaker = MilvusCircuitBreaker()
        await breaker._transition_to_open()
        assert breaker.is_open()

        await breaker.reset()

        assert breaker.is_closed()
        assert breaker._failure_count == 0
        assert breaker._success_count == 0
        assert breaker._half_open_requests == 0

    async def test_reset_from_half_open_state(self):
        """
        Test reset from HALF_OPEN state.

        Coverage: reset() transitioning from HALF_OPEN to CLOSED.
        """
        breaker = MilvusCircuitBreaker()
        await breaker._transition_to_half_open()
        assert breaker.is_half_open()

        await breaker.reset()

        assert breaker.is_closed()
        assert breaker._half_open_requests == 0

    async def test_reset_increments_state_change_count(self):
        """
        Test that reset increments state change count.

        Coverage: reset() incrementing _state_change_count.
        """
        breaker = MilvusCircuitBreaker()
        initial_count = breaker._state_change_count

        await breaker._transition_to_open()
        await breaker.reset()

        metrics = breaker.get_metrics()
        assert metrics["counters"]["state_change_count"] > initial_count


# ============================================================================
# Edge Cases Tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
class TestCircuitBreakerEdgeCases:
    """
    Test edge cases and boundary conditions.

    Coverage: Verifies behavior with edge case scenarios
    including rapid failures, boundary values, and error conditions.
    """

    async def test_rapid_failures_open_circuit_quickly(self):
        """
        Test rapid failures trigger circuit opening quickly.

        Coverage: Multiple _on_failure() calls in quick succession.
        """
        breaker = MilvusCircuitBreaker(CircuitBreakerConfig(failure_threshold=3))

        # Rapid failures
        for _ in range(3):
            await breaker._on_failure()

        assert breaker.is_open()

    async def test_recovery_timeout_edge_case(self):
        """
        Test recovery timeout at exact boundary.

        Coverage: _check_circuit_state() with recovery_timeout exactly elapsed.
        """
        breaker = MilvusCircuitBreaker(
            CircuitBreakerConfig(failure_threshold=1, recovery_timeout=5.0)
        )
        await breaker._transition_to_open()
        breaker._last_failure_time = 0.0

        # Exactly at recovery timeout
        with patch("time.monotonic", return_value=5.0):
            await breaker._check_circuit_state()
            assert breaker.is_half_open()

    async def test_recovery_timeout_one_millisecond_before(self):
        """
        Test recovery timeout one millisecond before boundary.

        Coverage: _check_circuit_state() just before recovery_timeout.
        """
        breaker = MilvusCircuitBreaker(
            CircuitBreakerConfig(failure_threshold=1, recovery_timeout=5.0)
        )
        await breaker._transition_to_open()
        current_time = 1000.0
        breaker._last_failure_time = current_time - 4.999  # Just before timeout

        with patch("time.monotonic", return_value=current_time):
            with pytest.raises(ServerUnavailableError):
                await breaker._check_circuit_state()

            assert breaker.is_open()

    async def test_half_open_request_counter_edge_case(self):
        """
        Test half-open request counter edge case.

        Coverage: _execute_context() with counter at boundary values.
        """
        breaker = MilvusCircuitBreaker(CircuitBreakerConfig(max_half_open_requests=1))
        await breaker._transition_to_half_open()

        # Set counter to max
        breaker._half_open_requests = 1

        # Should block
        with pytest.raises(ServerUnavailableError):
            await breaker._check_circuit_state()

    async def test_operation_with_exception_in_finally(self):
        """
        Test operation with exception handling in finally block.

        Coverage: _execute_context() cleanup even with exceptions.
        """
        breaker = MilvusCircuitBreaker()
        await breaker._transition_to_half_open()

        try:
            async with breaker._execute_context():
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Counter should still be decremented in finally block
        assert breaker._half_open_requests == 0


# ============================================================================
# String Representation Tests
# ============================================================================


@pytest.mark.unit
class TestCircuitBreakerStringRepresentation:
    """
    Test string representation methods.

    Coverage: Verifies __str__ and __repr__ methods.
    """

    def test_str_representation(self):
        """
        Test string representation.

        Coverage: MilvusCircuitBreaker.__str__().
        """
        breaker = MilvusCircuitBreaker(name="test_breaker")
        str_repr = str(breaker)
        assert "MilvusCircuitBreaker" in str_repr
        assert "test_breaker" in str_repr
        assert "closed" in str_repr

    def test_repr_representation(self):
        """
        Test detailed representation.

        Coverage: MilvusCircuitBreaker.__repr__().
        """
        breaker = MilvusCircuitBreaker(name="test_breaker")
        repr_str = repr(breaker)
        assert "MilvusCircuitBreaker" in repr_str
        assert "test_breaker" in repr_str
        assert "closed" in repr_str
        assert "failures=" in repr_str
        assert "requests=" in repr_str

    def test_str_representation_with_open_state(self):
        """
        Test string representation in OPEN state.

        Coverage: __str__() with different states.
        """
        breaker = MilvusCircuitBreaker()
        breaker._state = CircuitState.OPEN
        str_repr = str(breaker)
        assert "open" in str_repr


# ============================================================================
# Integration Style Tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
class TestCircuitBreakerIntegration:
    """
    Test circuit breaker behavior in realistic scenarios.

    Coverage: End-to-end behavior with multiple operations and state transitions.
    """

    async def test_complete_cycle_closed_open_half_open_closed(self):
        """
        Test complete state cycle: CLOSED -> OPEN -> HALF_OPEN -> CLOSED.

        Coverage: Full state machine cycle with transitions.
        """
        breaker = MilvusCircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=2,
                recovery_timeout=0.1,
                half_open_success_threshold=2,
            )
        )

        # Start CLOSED
        assert breaker.is_closed()

        # Failures open circuit
        async def fail_op() -> None:
            raise ConnectionError("Failure")

        for _ in range(2):
            with suppress(ConnectionError):
                await breaker.execute_milvus_operation(fail_op)

        assert breaker.is_open()

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Success operations should transition to HALF_OPEN then CLOSED
        async def success_op() -> str:
            return "success"

        # First success transitions to HALF_OPEN
        result1 = await breaker.execute_milvus_operation(success_op)
        assert result1 == "success"

        # Second success closes circuit
        result2 = await breaker.execute_milvus_operation(success_op)
        assert result2 == "success"
        assert breaker.is_closed()

    async def test_operation_with_args_and_kwargs(self):
        """
        Test operation execution with positional and keyword arguments.

        Coverage: execute_milvus_operation() with *args and **kwargs.
        """
        breaker = MilvusCircuitBreaker()

        def operation(arg1: str, arg2: int, kwarg1: str = "default") -> str:
            return f"{arg1}_{arg2}_{kwarg1}"

        result = await breaker.execute_milvus_operation(operation, "test", 42, kwarg1="custom")
        assert result == "test_42_custom"
