"""
Unit tests for circuit breaker implementation.

This module provides comprehensive tests for the MilvusCircuitBreaker class,
testing state machine transitions, configuration validation, exception handling,
and edge cases to achieve high code coverage.
"""

import asyncio
from unittest.mock import patch

import pytest

from milvus_ops.connection_management.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitState,
    MilvusCircuitBreaker,
)
from milvus_ops.connection_management.connection_exceptions import (
    ConnectionError,
    ConnectionTimeoutError,
    MaxRetriesExceededError,
    ServerUnavailableError,
)

# ============================================================================
# CircuitBreakerConfig Tests
# ============================================================================


@pytest.mark.unit
class TestCircuitBreakerConfig:
    """
    Test circuit breaker configuration validation and defaults.

    Coverage: Verifies configuration validation and default values.
    """

    def test_default_config(self):
        """
        Test default configuration values.

        Coverage: CircuitBreakerConfig.__init__ with no arguments.
        """
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 30.0
        assert config.half_open_success_threshold == 3
        assert config.max_half_open_requests == 2
        assert ValueError in config.milvus_exclude_exceptions
        assert TypeError in config.milvus_exclude_exceptions
        assert AttributeError in config.milvus_exclude_exceptions

    def test_custom_config(self):
        """
        Test custom configuration values.

        Coverage: CircuitBreakerConfig.__init__ with custom values.
        """
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=10.0,
            half_open_success_threshold=2,
            max_half_open_requests=1,
        )
        assert config.failure_threshold == 3
        assert config.recovery_timeout == 10.0
        assert config.half_open_success_threshold == 2
        assert config.max_half_open_requests == 1

    def test_config_with_custom_exclude_exceptions(self):
        """
        Test configuration with custom excluded exceptions.

        Coverage: CircuitBreakerConfig.milvus_exclude_exceptions customization.
        """
        config = CircuitBreakerConfig()
        # Note: milvus_exclude_exceptions is a tuple, can't be set directly
        # This test verifies the default is a tuple
        assert isinstance(config.milvus_exclude_exceptions, tuple)


# ============================================================================
# CircuitState Enum Tests
# ============================================================================


@pytest.mark.unit
class TestCircuitState:
    """
    Test CircuitState enum values.

    Coverage: Verifies all enum values are correctly defined.
    """

    def test_circuit_state_values(self):
        """
        Test all circuit state enum values.

        Coverage: CircuitState enum value access.
        """
        assert CircuitState.CLOSED == "closed"
        assert CircuitState.OPEN == "open"
        assert CircuitState.HALF_OPEN == "half-open"

    def test_circuit_state_enum_inheritance(self):
        """
        Test that CircuitState is a string enum.

        Coverage: CircuitState enum inheritance and type.
        """
        assert isinstance(CircuitState.CLOSED, str)
        assert isinstance(CircuitState.OPEN, str)
        assert isinstance(CircuitState.HALF_OPEN, str)


# ============================================================================
# MilvusCircuitBreaker Initialization Tests
# ============================================================================


@pytest.mark.unit
class TestCircuitBreakerInitialization:
    """
    Test circuit breaker initialization and validation.

    Coverage: Verifies initialization with default/custom configs,
    configuration validation, and initial state setup.
    """

    def test_initialization_with_default_config(self):
        """
        Test initialization with default configuration.

        Coverage: MilvusCircuitBreaker.__init__ with None config.
        """
        breaker = MilvusCircuitBreaker()
        assert breaker.config.failure_threshold == 5
        assert breaker.config.recovery_timeout == 30.0
        assert breaker.name == "milvus"
        assert breaker.is_closed()
        assert breaker._failure_count == 0
        assert breaker._success_count == 0
        assert breaker._total_requests == 0

    def test_initialization_with_custom_config(self):
        """
        Test initialization with custom configuration.

        Coverage: MilvusCircuitBreaker.__init__ with custom CircuitBreakerConfig.
        """
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=5.0,
            half_open_success_threshold=2,
            max_half_open_requests=1,
        )
        breaker = MilvusCircuitBreaker(config, name="test_breaker")
        assert breaker.config.failure_threshold == 3
        assert breaker.config.recovery_timeout == 5.0
        assert breaker.name == "test_breaker"
        assert breaker.is_closed()

    def test_config_validation_failure_threshold_too_low(self):
        """
        Test configuration validation rejects invalid failure_threshold.

        Coverage: MilvusCircuitBreaker._validate_config with failure_threshold < 1.
        """
        config = CircuitBreakerConfig(failure_threshold=0)
        with pytest.raises(ValueError, match="failure_threshold must be at least 1"):
            MilvusCircuitBreaker(config)

    def test_config_validation_recovery_timeout_negative(self):
        """
        Test configuration validation rejects negative recovery_timeout.

        Coverage: MilvusCircuitBreaker._validate_config with recovery_timeout < 0.
        """
        config = CircuitBreakerConfig(recovery_timeout=-1.0)
        with pytest.raises(ValueError, match="recovery_timeout cannot be negative"):
            MilvusCircuitBreaker(config)

    def test_config_validation_half_open_success_threshold_too_low(self):
        """
        Test configuration validation rejects invalid half_open_success_threshold.

        Coverage: MilvusCircuitBreaker._validate_config with half_open_success_threshold < 1.
        """
        config = CircuitBreakerConfig(half_open_success_threshold=0)
        with pytest.raises(ValueError, match="half_open_success_threshold must be at least 1"):
            MilvusCircuitBreaker(config)

    def test_config_validation_max_half_open_requests_too_low(self):
        """
        Test configuration validation rejects invalid max_half_open_requests.

        Coverage: MilvusCircuitBreaker._validate_config with max_half_open_requests < 1.
        """
        config = CircuitBreakerConfig(max_half_open_requests=0)
        with pytest.raises(ValueError, match="max_half_open_requests must be at least 1"):
            MilvusCircuitBreaker(config)

    def test_initial_metrics(self):
        """
        Test initial metrics state.

        Coverage: MilvusCircuitBreaker.get_metrics() on initialization.
        """
        breaker = MilvusCircuitBreaker()
        metrics = breaker.get_metrics()
        assert metrics["state"] == "closed"
        assert metrics["counters"]["total_requests"] == 0
        assert metrics["counters"]["total_failures"] == 0
        assert metrics["counters"]["failure_count"] == 0
        assert metrics["counters"]["success_count"] == 0


# ============================================================================
# Circuit Breaker State Queries Tests
# ============================================================================


@pytest.mark.unit
class TestCircuitBreakerStateQueries:
    """
    Test state query methods.

    Coverage: Verifies is_open, is_half_open, is_closed, and get_state methods.
    """

    @pytest.mark.asyncio
    async def test_initial_state_closed(self):
        """
        Test that circuit breaker starts in CLOSED state.

        Coverage: MilvusCircuitBreaker.is_closed() and get_state().
        """
        breaker = MilvusCircuitBreaker()
        assert breaker.is_closed()
        assert not breaker.is_open()
        assert not breaker.is_half_open()
        assert breaker.get_state() == "closed"

    @pytest.mark.asyncio
    async def test_state_after_transition_to_open(self):
        """
        Test state queries after transitioning to OPEN.

        Coverage: State queries after _transition_to_open().
        """
        breaker = MilvusCircuitBreaker(
            CircuitBreakerConfig(failure_threshold=1, recovery_timeout=1.0)
        )
        # Trigger transition to OPEN
        await breaker._on_failure()
        # Manually set to OPEN for testing
        await breaker._transition_to_open()
        assert breaker.is_open()
        assert not breaker.is_closed()
        assert not breaker.is_half_open()
        assert breaker.get_state() == "open"

    @pytest.mark.asyncio
    async def test_state_after_transition_to_half_open(self):
        """
        Test state queries after transitioning to HALF_OPEN.

        Coverage: State queries after _transition_to_half_open().
        """
        breaker = MilvusCircuitBreaker()
        await breaker._transition_to_half_open()
        assert breaker.is_half_open()
        assert not breaker.is_closed()
        assert not breaker.is_open()
        assert breaker.get_state() == "half-open"


# ============================================================================
# Exception Classification Tests
# ============================================================================


@pytest.mark.unit
class TestExceptionClassification:
    """
    Test exception classification logic.

    Coverage: Verifies _should_count_as_failure correctly classifies
    exceptions as server-side (count as failure) vs client-side (don't count).
    """

    def test_connection_error_counts_as_failure(self):
        """
        Test that ConnectionError counts as failure.

        Coverage: _should_count_as_failure with ConnectionError.
        """
        breaker = MilvusCircuitBreaker()
        error = ConnectionError("Connection failed")
        assert breaker._should_count_as_failure(error) is True

    def test_connection_timeout_error_counts_as_failure(self):
        """
        Test that ConnectionTimeoutError counts as failure.

        Coverage: _should_count_as_failure with ConnectionTimeoutError.
        """
        breaker = MilvusCircuitBreaker()
        error = ConnectionTimeoutError("Timeout")
        assert breaker._should_count_as_failure(error) is True

    def test_server_unavailable_error_counts_as_failure(self):
        """
        Test that ServerUnavailableError counts as failure.

        Coverage: _should_count_as_failure with ServerUnavailableError.
        """
        breaker = MilvusCircuitBreaker()
        error = ServerUnavailableError("Server unavailable")
        assert breaker._should_count_as_failure(error) is True

    def test_max_retries_exceeded_error_counts_as_failure(self):
        """
        Test that MaxRetriesExceededError counts as failure.

        Coverage: _should_count_as_failure with MaxRetriesExceededError.
        """
        breaker = MilvusCircuitBreaker()
        error = MaxRetriesExceededError("Max retries exceeded")
        assert breaker._should_count_as_failure(error) is True

    def test_value_error_does_not_count_as_failure(self):
        """
        Test that ValueError does not count as failure.

        Coverage: _should_count_as_failure with excluded exceptions.
        """
        breaker = MilvusCircuitBreaker()
        error = ValueError("Invalid parameter")
        assert breaker._should_count_as_failure(error) is False

    def test_type_error_does_not_count_as_failure(self):
        """
        Test that TypeError does not count as failure.

        Coverage: _should_count_as_failure with excluded exceptions.
        """
        breaker = MilvusCircuitBreaker()
        error = TypeError("Type mismatch")
        assert breaker._should_count_as_failure(error) is False

    def test_attribute_error_does_not_count_as_failure(self):
        """
        Test that AttributeError does not count as failure.

        Coverage: _should_count_as_failure with excluded exceptions.
        """
        breaker = MilvusCircuitBreaker()
        error = AttributeError("Attribute not found")
        assert breaker._should_count_as_failure(error) is False

    def test_unknown_exception_counts_as_failure(self):
        """
        Test that unknown exceptions count as failure (conservative approach).

        Coverage: _should_count_as_failure with unknown exception types.
        """
        breaker = MilvusCircuitBreaker()
        error = RuntimeError("Unknown error")
        # Should count as failure for safety
        assert breaker._should_count_as_failure(error) is True


# ============================================================================
# State Transition Tests: CLOSED -> OPEN
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
class TestClosedToOpenTransition:
    """
    Test transitions from CLOSED to OPEN state.

    Coverage: Verifies failure counting, threshold detection,
    and proper transition to OPEN state.
    """

    async def test_single_failure_does_not_open(self):
        """
        Test that single failure does not open circuit.

        Coverage: _on_failure() in CLOSED state with below-threshold failures.
        """
        breaker = MilvusCircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
        await breaker._on_failure()
        assert breaker.is_closed()
        assert breaker._failure_count == 1

    async def test_multiple_failures_below_threshold(self):
        """
        Test multiple failures below threshold.

        Coverage: _on_failure() accumulating failures below threshold.
        """
        breaker = MilvusCircuitBreaker(CircuitBreakerConfig(failure_threshold=5))
        for _ in range(4):
            await breaker._on_failure()
        assert breaker.is_closed()
        assert breaker._failure_count == 4

    async def test_failure_threshold_opens_circuit(self):
        """
        Test that reaching failure threshold opens circuit.

        Coverage: _on_failure() triggering _transition_to_open().
        """
        breaker = MilvusCircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
        for _ in range(3):
            await breaker._on_failure()
        assert breaker.is_open()
        assert breaker._failure_count == 3

    async def test_failure_resets_on_success(self):
        """
        Test that success in CLOSED state resets failure count.

        Coverage: _on_success() resetting _failure_count in CLOSED state.
        """
        breaker = MilvusCircuitBreaker(CircuitBreakerConfig(failure_threshold=5))
        # Accumulate some failures
        await breaker._on_failure()
        await breaker._on_failure()
        assert breaker._failure_count == 2

        # Success should reset failure count
        await breaker._on_success()
        assert breaker._failure_count == 0
        assert breaker.is_closed()


# ============================================================================
# State Transition Tests: OPEN -> HALF_OPEN
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
class TestOpenToHalfOpenTransition:
    """
    Test transitions from OPEN to HALF_OPEN state.

    Coverage: Verifies recovery timeout handling and transition logic.
    """

    async def test_open_circuit_fails_fast(self):
        """
        Test that OPEN circuit fails fast without executing operation.

        Coverage: _check_circuit_state() raising ServerUnavailableError when OPEN.
        """
        breaker = MilvusCircuitBreaker(
            CircuitBreakerConfig(failure_threshold=1, recovery_timeout=10.0)
        )
        await breaker._transition_to_open()

        with pytest.raises(ServerUnavailableError, match="circuit open"):
            await breaker._check_circuit_state()

    async def test_recovery_timeout_transitions_to_half_open(self):
        """
        Test that recovery timeout elapsed transitions to HALF_OPEN.

        Coverage: _check_circuit_state() with elapsed recovery timeout.
        """
        breaker = MilvusCircuitBreaker(
            CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        )
        await breaker._transition_to_open()
        breaker._last_failure_time = 0.0  # Set to past

        with patch("time.monotonic", return_value=1.0):
            await breaker._check_circuit_state()
            assert breaker.is_half_open()

    async def test_recovery_timeout_not_elapsed_stays_open(self):
        """
        Test that circuit stays OPEN if recovery timeout not elapsed.

        Coverage: _check_circuit_state() with unelapsed recovery timeout.
        """
        breaker = MilvusCircuitBreaker(
            CircuitBreakerConfig(failure_threshold=1, recovery_timeout=10.0)
        )
        await breaker._transition_to_open()

        # Set failure time to recent past (within timeout)
        current_time = 1000.0
        with patch("time.monotonic", return_value=current_time):
            breaker._last_failure_time = current_time - 5.0  # 5 seconds ago

            with pytest.raises(ServerUnavailableError):
                await breaker._check_circuit_state()

            assert breaker.is_open()


# ============================================================================
# State Transition Tests: HALF_OPEN -> CLOSED (Success)
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
class TestHalfOpenToClosedSuccess:
    """
    Test transitions from HALF_OPEN to CLOSED on successful recovery.

    Coverage: Verifies success counting and threshold detection in half-open state.
    """

    async def test_single_success_in_half_open(self):
        """
        Test single success in HALF_OPEN state.

        Coverage: _on_success() incrementing success_count in HALF_OPEN state.
        """
        breaker = MilvusCircuitBreaker(CircuitBreakerConfig(half_open_success_threshold=3))
        await breaker._transition_to_half_open()

        await breaker._on_success()
        assert breaker.is_half_open()
        assert breaker._success_count == 1

    async def test_success_threshold_closes_circuit(self):
        """
        Test that reaching success threshold closes circuit.

        Coverage: _on_success() triggering _transition_to_closed().
        """
        breaker = MilvusCircuitBreaker(CircuitBreakerConfig(half_open_success_threshold=2))
        await breaker._transition_to_half_open()

        await breaker._on_success()
        await breaker._on_success()

        assert breaker.is_closed()
        assert breaker._success_count == 0  # Reset after transition
        assert breaker._failure_count == 0

    async def test_partial_successes_below_threshold(self):
        """
        Test partial successes below threshold.

        Coverage: _on_success() with below-threshold successes.
        """
        breaker = MilvusCircuitBreaker(CircuitBreakerConfig(half_open_success_threshold=3))
        await breaker._transition_to_half_open()

        await breaker._on_success()
        await breaker._on_success()

        assert breaker.is_half_open()
        assert breaker._success_count == 2


# ============================================================================
# State Transition Tests: HALF_OPEN -> OPEN (Failure)
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
class TestHalfOpenToOpenFailure:
    """
    Test transitions from HALF_OPEN to OPEN on failure.

    Coverage: Verifies that any failure in half-open immediately opens circuit.
    """

    async def test_failure_in_half_open_immediately_opens(self):
        """
        Test that any failure in HALF_OPEN immediately opens circuit.

        Coverage: _on_failure() in HALF_OPEN state triggering _transition_to_open().
        """
        breaker = MilvusCircuitBreaker()
        await breaker._transition_to_half_open()
        assert breaker.is_half_open()

        await breaker._on_failure()

        assert breaker.is_open()
        assert breaker._success_count == 0  # Reset after transition


# ============================================================================
# Half-Open Concurrent Request Limiting Tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
class TestHalfOpenConcurrentLimiting:
    """
    Test concurrent request limiting in HALF_OPEN state.

    Coverage: Verifies max_half_open_requests limit and proper tracking.
    """

    async def test_half_open_allows_requests_below_limit(self):
        """
        Test that half-open allows requests below max limit.

        Coverage: _execute_context() in HALF_OPEN with below-limit requests.
        """
        breaker = MilvusCircuitBreaker(CircuitBreakerConfig(max_half_open_requests=2))
        await breaker._transition_to_half_open()

        async with breaker._execute_context():
            assert breaker._half_open_requests == 1

    async def test_half_open_blocks_requests_at_limit(self):
        """
        Test that half-open blocks requests at max limit.

        Coverage: _check_circuit_state() raising ServerUnavailableError at limit.
        """
        breaker = MilvusCircuitBreaker(CircuitBreakerConfig(max_half_open_requests=1))
        await breaker._transition_to_half_open()

        # First request should be allowed
        async with breaker._execute_context():
            assert breaker._half_open_requests == 1

            # Second request should be blocked
            with pytest.raises(ServerUnavailableError, match="recovery testing"):
                await breaker._check_circuit_state()

    async def test_half_open_request_counter_decrements(self):
        """
        Test that half-open request counter decrements properly.

        Coverage: _execute_context() cleanup in finally block.
        """
        breaker = MilvusCircuitBreaker(CircuitBreakerConfig(max_half_open_requests=1))
        await breaker._transition_to_half_open()

        async with breaker._execute_context():
            assert breaker._half_open_requests == 1

        # After context exit, counter should decrement
        assert breaker._half_open_requests == 0

    async def test_half_open_multiple_sequential_requests(self):
        """
        Test multiple sequential requests in half-open state.

        Coverage: _execute_context() with sequential half-open requests.
        """
        breaker = MilvusCircuitBreaker(CircuitBreakerConfig(max_half_open_requests=2))
        await breaker._transition_to_half_open()

        async with breaker._execute_context():
            assert breaker._half_open_requests == 1

        async with breaker._execute_context():
            assert breaker._half_open_requests == 1

        assert breaker._half_open_requests == 0


# ============================================================================
# Operation Execution Tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
class TestOperationExecution:
    """
    Test operation execution through circuit breaker.

    Coverage: Verifies execute_milvus_operation handles sync/async operations,
    tracks metrics, and properly handles exceptions.
    """

    async def test_successful_sync_operation(self):
        """
        Test successful synchronous operation execution.

        Coverage: execute_milvus_operation with sync callable returning result.
        """
        breaker = MilvusCircuitBreaker()

        def sync_operation() -> str:
            return "success"

        result = await breaker.execute_milvus_operation(sync_operation)
        assert result == "success"
        assert breaker._total_requests == 1
        assert breaker._total_failures == 0

    async def test_successful_async_operation(self):
        """
        Test successful asynchronous operation execution.

        Coverage: execute_milvus_operation with async callable.
        """
        breaker = MilvusCircuitBreaker()

        async def async_operation() -> str:
            await asyncio.sleep(0.01)
            return "async success"

        result = await breaker.execute_milvus_operation(async_operation)
        assert result == "async success"
        assert breaker._total_requests == 1

    async def test_failed_operation_counts_failure(self):
        """
        Test that failed operation counts as failure.

        Coverage: execute_milvus_operation exception handling and failure counting.
        """
        breaker = MilvusCircuitBreaker(CircuitBreakerConfig(failure_threshold=5))

        def failing_operation() -> None:
            raise ConnectionError("Operation failed")

        with pytest.raises(ConnectionError):
            await breaker.execute_milvus_operation(failing_operation)

        assert breaker._total_requests == 1
        assert breaker._total_failures == 1
        assert breaker._failure_count == 1

    async def test_client_error_does_not_count_as_failure(self):
        """
        Test that client errors don't count as circuit breaker failure.

        Coverage: execute_milvus_operation with excluded exception types.
        """
        breaker = MilvusCircuitBreaker()

        def client_error_operation() -> None:
            raise ValueError("Client error")

        with pytest.raises(ValueError):
            await breaker.execute_milvus_operation(client_error_operation)

        # Should not count as failure
        assert breaker._total_requests == 1
        assert breaker._total_failures == 0
        assert breaker._failure_count == 0

    async def test_operation_with_open_circuit_fails_fast(self):
        """
        Test that operation with OPEN circuit fails fast.

        Coverage: execute_milvus_operation with OPEN circuit state.
        """
        breaker = MilvusCircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
        await breaker._transition_to_open()

        def operation() -> str:
            return "should not execute"

        with pytest.raises(ServerUnavailableError, match="circuit open"):
            await breaker.execute_milvus_operation(operation)

        assert breaker._total_fast_failures > 0
