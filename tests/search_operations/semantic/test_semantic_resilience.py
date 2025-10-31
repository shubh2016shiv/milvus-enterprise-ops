"""
Comprehensive unit tests for semantic search resilience manager.

This module provides systematic testing of ResilienceManager, retry logic,
circuit breaker states, error recovery, fallback mechanisms, and concurrent handling.
"""

import asyncio
import contextlib
from unittest.mock import AsyncMock

import pytest

from milvus_ops.search_operations.core.search_ops_exceptions import SearchError
from milvus_ops.search_operations.search.semantic.resilience import (
    CircuitBreaker,
    CircuitState,
    ResilienceManager,
    RetryConfig,
    RetryHandler,
)

# ============================================================================
# Test RetryConfig
# ============================================================================


@pytest.mark.unit
class TestRetryConfig:
    """
    Test RetryConfig dataclass.

    Coverage: RetryConfig initialization, default values, parameter validation.
    """

    def test_retry_config_default(self):
        """
        Test RetryConfig initialization with defaults.

        Coverage: RetryConfig.__init__() with default values.
        """
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert len(config.retriable_exceptions) > 0

    def test_retry_config_custom(self):
        """
        Test RetryConfig initialization with custom values.

        Coverage: RetryConfig.__init__() accepts custom parameters.
        """
        config = RetryConfig(
            max_retries=5,
            initial_delay=1.0,
            max_delay=60.0,
            exponential_base=3.0,
            jitter=False,
        )
        assert config.max_retries == 5
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 3.0
        assert config.jitter is False


# ============================================================================
# Test CircuitBreaker
# ============================================================================


@pytest.mark.unit
class TestCircuitBreaker:
    """
    Test CircuitBreaker class.

    Coverage: Circuit breaker states (CLOSED, OPEN, HALF_OPEN), failure handling, recovery.
    """

    def test_circuit_breaker_initialization(self):
        """
        Test CircuitBreaker initialization.

        Coverage: CircuitBreaker.__init__() initializes with default or custom parameters.
        """
        breaker = CircuitBreaker()
        assert breaker.failure_threshold == 5
        assert breaker.recovery_timeout == 60.0
        assert breaker.failure_count == 0
        assert breaker.state == CircuitState.CLOSED.value

    def test_circuit_breaker_state_property(self):
        """
        Test circuit_breaker.state property.

        Coverage: CircuitBreaker.state returns current state.
        """
        breaker = CircuitBreaker()
        assert breaker.state in ["closed", "open", "half_open"]

    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self):
        """
        Test circuit breaker in CLOSED state allows requests.

        Coverage: CircuitBreaker.call() allows requests when state is CLOSED.
        """
        breaker = CircuitBreaker()
        mock_func = AsyncMock(return_value="success")

        result = await breaker.call(mock_func)
        assert result == "success"
        mock_func.assert_called_once()

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """
        Test circuit breaker opens after threshold failures.

        Coverage: CircuitBreaker.call() opens circuit after failure_threshold failures.
        """
        breaker = CircuitBreaker(failure_threshold=3)
        mock_func = AsyncMock(side_effect=SearchError("Test error"))

        # Trigger failures up to threshold
        for _ in range(3):
            with contextlib.suppress(SearchError):
                await breaker.call(mock_func)

        # Circuit should be open now
        assert breaker.state == CircuitState.OPEN.value

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_blocks_requests(self):
        """
        Test circuit breaker in OPEN state blocks requests.

        Coverage: CircuitBreaker.call() raises SearchError when circuit is OPEN.
        """
        breaker = CircuitBreaker(failure_threshold=1)
        mock_func = AsyncMock(side_effect=SearchError("Test error"))

        # Trigger failure to open circuit
        with contextlib.suppress(SearchError):
            await breaker.call(mock_func)

        # Circuit is now open, should raise error
        with pytest.raises(SearchError, match="Circuit breaker is OPEN"):
            await breaker.call(mock_func)

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_after_timeout(self):
        """
        Test circuit breaker transitions to HALF_OPEN after recovery timeout.

        Coverage: CircuitBreaker.call() transitions to HALF_OPEN after recovery_timeout.
        """
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        mock_func = AsyncMock(side_effect=SearchError("Test error"))

        # Trigger failure to open circuit
        with contextlib.suppress(SearchError):
            await breaker.call(mock_func)

        assert breaker.state == CircuitState.OPEN.value

        # Wait for recovery timeout
        await asyncio.sleep(0.2)

        # Next call should transition to HALF_OPEN
        mock_func.reset_mock()
        mock_func.side_effect = None
        mock_func.return_value = "success"

        result = await breaker.call(mock_func)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED.value  # Should close on success

    @pytest.mark.asyncio
    async def test_circuit_breaker_reset(self):
        """
        Test circuit breaker reset() method.

        Coverage: CircuitBreaker.reset() resets circuit to CLOSED state.
        """
        breaker = CircuitBreaker(failure_threshold=1)
        mock_func = AsyncMock(side_effect=SearchError("Test error"))

        # Trigger failure to open circuit
        with contextlib.suppress(SearchError):
            await breaker.call(mock_func)

        assert breaker.state == CircuitState.OPEN.value

        # Reset circuit
        await breaker.reset()
        assert breaker.state == CircuitState.CLOSED.value
        assert breaker.failure_count == 0

    def test_circuit_breaker_get_status(self):
        """
        Test circuit breaker get_status() method.

        Coverage: CircuitBreaker.get_status() returns status dictionary.
        """
        breaker = CircuitBreaker()
        status = breaker.get_status()
        assert isinstance(status, dict)
        assert "state" in status
        assert "failure_count" in status
        assert "failure_threshold" in status


# ============================================================================
# Test RetryHandler
# ============================================================================


@pytest.mark.unit
class TestRetryHandler:
    """
    Test RetryHandler class.

    Coverage: Retry logic with exponential backoff, jitter, retriable exceptions.
    """

    def test_retry_handler_initialization(self):
        """
        Test RetryHandler initialization.

        Coverage: RetryHandler.__init__() initializes with RetryConfig.
        """
        config = RetryConfig(max_retries=3)
        handler = RetryHandler(config)
        assert handler.config == config

    def test_retry_handler_default_config(self):
        """
        Test RetryHandler initialization with default config.

        Coverage: RetryHandler.__init__() uses default RetryConfig if not provided.
        """
        handler = RetryHandler()
        assert handler.config is not None
        assert isinstance(handler.config, RetryConfig)

    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self):
        """
        Test execute_with_retry() succeeds on first attempt.

        Coverage: RetryHandler.execute_with_retry() returns result on success.
        """
        handler = RetryHandler(RetryConfig(max_retries=3))
        mock_func = AsyncMock(return_value="success")

        result = await handler.execute_with_retry(mock_func)
        assert result == "success"
        mock_func.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_retry_retries_on_failure(self):
        """
        Test execute_with_retry() retries on retriable exceptions.

        Coverage: RetryHandler.execute_with_retry() retries on retriable exceptions.
        """
        config = RetryConfig(
            max_retries=2,
            initial_delay=0.01,
            retriable_exceptions=(ConnectionError,),
        )
        handler = RetryHandler(config)
        mock_func = AsyncMock(
            side_effect=[ConnectionError("Retry"), ConnectionError("Retry"), "success"]
        )

        result = await handler.execute_with_retry(mock_func)
        assert result == "success"
        assert mock_func.call_count == 3  # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_execute_with_retry_exhausts_retries(self):
        """
        Test execute_with_retry() exhausts retries and raises exception.

        Coverage: RetryHandler.execute_with_retry() raises exception when retries exhausted.
        """
        config = RetryConfig(
            max_retries=2,
            initial_delay=0.01,
            retriable_exceptions=(ConnectionError,),
        )
        handler = RetryHandler(config)
        mock_func = AsyncMock(side_effect=ConnectionError("Retry"))

        with pytest.raises(ConnectionError):
            await handler.execute_with_retry(mock_func)
        assert mock_func.call_count == 3  # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_execute_with_retry_non_retriable_exception(self):
        """
        Test execute_with_retry() fails immediately on non-retriable exception.

        Coverage: RetryHandler.execute_with_retry() doesn't retry non-retriable exceptions.
        """
        config = RetryConfig(
            max_retries=3,
            initial_delay=0.01,
            retriable_exceptions=(ConnectionError,),
        )
        handler = RetryHandler(config)
        mock_func = AsyncMock(side_effect=ValueError("Non-retriable"))

        with pytest.raises(ValueError):
            await handler.execute_with_retry(mock_func)
        # Should only be called once (no retries)
        assert mock_func.call_count == 1


# ============================================================================
# Test ResilienceManager
# ============================================================================


@pytest.mark.unit
class TestResilienceManager:
    """
    Test ResilienceManager class.

    Coverage: Initialization, execute with circuit breaker and retry, status, error recovery.
    """

    def test_resilience_manager_initialization_default(self):
        """
        Test ResilienceManager initialization with default parameters.

        Coverage: ResilienceManager.__init__() enables circuit breaker and retry by default.
        """
        manager = ResilienceManager()
        assert manager.circuit_breaker is not None
        assert manager.retry_handler is not None

    def test_resilience_manager_initialization_without_circuit_breaker(self):
        """
        Test ResilienceManager initialization without circuit breaker.

        Coverage: ResilienceManager.__init__() with enable_circuit_breaker=False.
        """
        manager = ResilienceManager(enable_circuit_breaker=False)
        assert manager.circuit_breaker is None
        assert manager.retry_handler is not None

    def test_resilience_manager_initialization_without_retry(self):
        """
        Test ResilienceManager initialization without retry.

        Coverage: ResilienceManager.__init__() with enable_retry=False.
        """
        manager = ResilienceManager(enable_retry=False)
        assert manager.circuit_breaker is not None
        assert manager.retry_handler is None

    def test_resilience_manager_initialization_without_both(self):
        """
        Test ResilienceManager initialization without circuit breaker or retry.

        Coverage: ResilienceManager.__init__() with both disabled.
        """
        manager = ResilienceManager(enable_circuit_breaker=False, enable_retry=False)
        assert manager.circuit_breaker is None
        assert manager.retry_handler is None

    @pytest.mark.asyncio
    async def test_execute_with_circuit_breaker_and_retry(self):
        """
        Test execute() with both circuit breaker and retry enabled.

        Coverage: ResilienceManager.execute() uses both circuit breaker and retry.
        """
        manager = ResilienceManager(
            enable_circuit_breaker=True,
            enable_retry=True,
            retry_config=RetryConfig(
                max_retries=2,
                initial_delay=0.01,
            ),
        )
        mock_func = AsyncMock(return_value="success")

        result = await manager.execute(mock_func)
        assert result == "success"
        mock_func.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_circuit_breaker_only(self):
        """
        Test execute() with only circuit breaker enabled.

        Coverage: ResilienceManager.execute() uses circuit breaker when retry disabled.
        """
        manager = ResilienceManager(enable_circuit_breaker=True, enable_retry=False)
        mock_func = AsyncMock(return_value="success")

        result = await manager.execute(mock_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_execute_with_retry_only(self):
        """
        Test execute() with only retry enabled.

        Coverage: ResilienceManager.execute() uses retry when circuit breaker disabled.
        """
        manager = ResilienceManager(enable_circuit_breaker=False, enable_retry=True)
        mock_func = AsyncMock(return_value="success")

        result = await manager.execute(mock_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_execute_without_resilience(self):
        """
        Test execute() without resilience features.

        Coverage: ResilienceManager.execute() executes directly when both disabled.
        """
        manager = ResilienceManager(enable_circuit_breaker=False, enable_retry=False)
        mock_func = AsyncMock(return_value="success")

        result = await manager.execute(mock_func)
        assert result == "success"
        mock_func.assert_called_once()

    def test_get_status(self):
        """
        Test get_status() method.

        Coverage: ResilienceManager.get_status() returns status dictionary.
        """
        manager = ResilienceManager()
        status = manager.get_status()
        assert isinstance(status, dict)
        assert "circuit_breaker_enabled" in status
        assert "retry_enabled" in status
        assert status["circuit_breaker_enabled"] is True
        assert status["retry_enabled"] is True

    @pytest.mark.asyncio
    async def test_execute_error_recovery(self):
        """
        Test execute() error recovery with retry.

        Coverage: ResilienceManager.execute() recovers from errors with retry logic.
        """
        config = RetryConfig(
            max_retries=1,
            initial_delay=0.01,
            retriable_exceptions=(ConnectionError,),
        )
        manager = ResilienceManager(
            enable_circuit_breaker=False,
            enable_retry=True,
            retry_config=config,
        )
        mock_func = AsyncMock(side_effect=[ConnectionError("Retry"), "success"])

        result = await manager.execute(mock_func)
        assert result == "success"
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_concurrent_requests(self):
        """
        Test execute() handles concurrent requests.

        Coverage: ResilienceManager.execute() handles concurrent requests correctly.
        """
        manager = ResilienceManager()
        mock_func = AsyncMock(return_value="success")

        # Execute multiple concurrent requests
        results = await asyncio.gather(*[manager.execute(mock_func) for _ in range(5)])

        assert len(results) == 5
        assert all(r == "success" for r in results)
        assert mock_func.call_count == 5
