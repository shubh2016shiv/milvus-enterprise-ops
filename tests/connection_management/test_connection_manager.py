"""
Unit tests for connection manager implementation.

This module provides comprehensive tests for ConnectionManager,
testing sync/async operations, retry logic, rate limiting, timeout handling,
and all edge cases to achieve high code coverage.
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from milvus_ops.connection_management.circuit_breaker import (
    MilvusCircuitBreaker,
)
from milvus_ops.connection_management.connection_exceptions import (
    ConnectionError,
    MaxRetriesExceededError,
    OperationTimeoutError,
)
from milvus_ops.connection_management.connection_manager import ConnectionManager
from milvus_ops.utils.rate_limiter import TokenBucketRateLimiter
from milvus_ops.utils.retry_budget import RetryBudget

# ============================================================================
# Test Utilities
# ============================================================================


def reset_connection_pool_singleton():
    """Reset MilvusConnectionPool singleton before each test for isolation."""
    from milvus_ops.connection_management.connection_pool import MilvusConnectionPool

    with MilvusConnectionPool._lock:
        MilvusConnectionPool._instance = None
        MilvusConnectionPool._initialized = False


def create_mock_pool_context_manager(conn_alias="conn_0"):
    """Create a properly mocked context manager for pool.get_connection()."""
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = Mock(return_value=conn_alias)
    mock_ctx.__exit__ = Mock(return_value=None)
    return mock_ctx


# ============================================================================
# Initialization Tests
# ============================================================================


@pytest.mark.unit
class TestConnectionManagerInitialization:
    """
    Test connection manager initialization.

    Coverage: Verifies initialization with various configurations including
    circuit breaker, rate limiter, and retry budget options.
    """

    def test_initialization_with_default_config(self, mock_load_settings, mock_milvus_settings):
        """
        Test initialization with default configuration.

        Coverage: ConnectionManager.__init__ with None config.
        """
        reset_connection_pool_singleton()
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.acquire_reference = Mock()
            mock_ctx = create_mock_pool_context_manager()
            mock_pool.get_connection.return_value = mock_ctx

            manager = ConnectionManager()
            assert manager.config is not None
            assert manager._circuit_breaker is not None
            assert manager._executor is not None

    def test_initialization_with_custom_config(
        self, mock_milvus_settings, mock_pymilvus_connections
    ):
        """
        Test initialization with custom configuration.

        Coverage: ConnectionManager.__init__ with MilvusSettings.
        """
        reset_connection_pool_singleton()
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.acquire_reference = Mock()
            mock_ctx = create_mock_pool_context_manager()
            mock_pool.get_connection.return_value = mock_ctx

            manager = ConnectionManager(mock_milvus_settings, enable_circuit_breaker=True)
            assert manager.config is mock_milvus_settings
            assert manager._circuit_breaker is not None
            assert isinstance(manager._circuit_breaker, MilvusCircuitBreaker)

    def test_initialization_without_circuit_breaker(
        self, mock_milvus_settings, mock_pymilvus_connections
    ):
        """
        Test initialization without circuit breaker.

        Coverage: ConnectionManager.__init__ with enable_circuit_breaker=False.
        """
        reset_connection_pool_singleton()
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.acquire_reference = Mock()
            mock_ctx = create_mock_pool_context_manager()
            mock_pool.get_connection.return_value = mock_ctx

            manager = ConnectionManager(mock_milvus_settings, enable_circuit_breaker=False)
            assert manager._circuit_breaker is None

    def test_initialization_with_rate_limiter(
        self, mock_milvus_settings, mock_pymilvus_connections
    ):
        """
        Test initialization with rate limiter enabled.

        Coverage: ConnectionManager.__init__ with max_requests_per_second > 0.
        """
        reset_connection_pool_singleton()
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.acquire_reference = Mock()
            mock_ctx = create_mock_pool_context_manager()
            mock_pool.get_connection.return_value = mock_ctx

            manager = ConnectionManager(mock_milvus_settings)
            # Rate limiter should be initialized if max_requests_per_second > 0
            if mock_milvus_settings.connection.max_requests_per_second > 0:
                assert manager._rate_limiter is not None
                assert isinstance(manager._rate_limiter, TokenBucketRateLimiter)

    def test_initialization_without_rate_limiter(
        self, mock_milvus_settings, mock_pymilvus_connections
    ):
        """
        Test initialization without rate limiter (when max_rps is 0).

        Coverage: ConnectionManager.__init__ with max_requests_per_second=0.
        """
        reset_connection_pool_singleton()
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.acquire_reference = Mock()
            mock_ctx = create_mock_pool_context_manager()
            mock_pool.get_connection.return_value = mock_ctx

            mock_milvus_settings.connection.max_requests_per_second = 0
            manager = ConnectionManager(mock_milvus_settings)
            assert manager._rate_limiter is None

    def test_initialization_with_retry_budget(
        self, mock_milvus_settings, mock_pymilvus_connections
    ):
        """
        Test initialization with retry budget enabled.

        Coverage: ConnectionManager.__init__ with enable_retry_budget=True.
        """
        reset_connection_pool_singleton()
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.acquire_reference = Mock()
            mock_ctx = create_mock_pool_context_manager()
            mock_pool.get_connection.return_value = mock_ctx

            manager = ConnectionManager(mock_milvus_settings)
            # Retry budget should be initialized if enabled
            if mock_milvus_settings.connection.enable_retry_budget:
                assert manager._retry_budget is not None
                assert isinstance(manager._retry_budget, RetryBudget)

    def test_initialization_thread_pool_executor(
        self, mock_milvus_settings, mock_pymilvus_connections
    ):
        """
        Test initialization creates dedicated ThreadPoolExecutor.

        Coverage: ConnectionManager.__init__ creating _executor.
        """
        reset_connection_pool_singleton()
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.acquire_reference = Mock()
            mock_ctx = create_mock_pool_context_manager()
            mock_pool.get_connection.return_value = mock_ctx

            manager = ConnectionManager(mock_milvus_settings)
            assert manager._executor is not None
            assert (
                manager._executor._max_workers
                == mock_milvus_settings.connection.connection_pool_size
            )


# ============================================================================
# Sync Operation Tests
# ============================================================================


@pytest.mark.unit
class TestSyncOperations:
    """
    Test synchronous operation execution.

    Coverage: Verifies execute_operation handles sync operations correctly,
    integrates with circuit breaker, pool, and retry logic.
    """

    def test_execute_operation_success(
        self, mock_milvus_settings, mock_pymilvus_connections, test_operation
    ):
        """
        Test successful operation execution.

        Coverage: execute_operation() with successful operation.
        """
        reset_connection_pool_singleton()
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.acquire_reference = Mock()
            mock_ctx = create_mock_pool_context_manager("conn_0")
            mock_pool.get_connection.return_value = mock_ctx
            mock_pymilvus_connections.has_connection.return_value = True

            manager = ConnectionManager(mock_milvus_settings, enable_circuit_breaker=False)

            result = manager.execute_operation(test_operation)
            assert "Operation result for conn_0" in result

    def test_execute_operation_with_circuit_breaker(
        self, mock_milvus_settings, mock_pymilvus_connections, test_operation
    ):
        """
        Test operation execution with circuit breaker.

        Coverage: execute_operation() with circuit breaker enabled.
        """
        reset_connection_pool_singleton()
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.acquire_reference = Mock()
            mock_ctx = create_mock_pool_context_manager("conn_0")
            mock_pool.get_connection.return_value = mock_ctx
            mock_pymilvus_connections.has_connection.return_value = True

            manager = ConnectionManager(mock_milvus_settings, enable_circuit_breaker=True)

            # Mock async execution
            with patch.object(
                manager, "_execute_with_circuit_breaker", new_callable=AsyncMock
            ) as mock_execute:
                mock_execute.return_value = "success"

                with patch("asyncio.run") as mock_asyncio_run:
                    mock_asyncio_run.return_value = "success"
                    result = manager.execute_operation(test_operation)
                    assert result == "success"

    def test_execute_operation_with_retry(
        self, mock_milvus_settings, mock_pymilvus_connections, failing_operation
    ):
        """
        Test operation execution with retry logic.

        Coverage: execute_operation() retry behavior.
        """
        reset_connection_pool_singleton()
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.acquire_reference = Mock()
            mock_ctx = create_mock_pool_context_manager("conn_0")
            mock_pool.get_connection.return_value = mock_ctx
            mock_pymilvus_connections.has_connection.return_value = True

            manager = ConnectionManager(mock_milvus_settings, enable_circuit_breaker=False)
            manager.config.connection.retry_count = 2
            manager.config.connection.retry_interval = 0.01  # Set retry interval for test

            with pytest.raises(MaxRetriesExceededError):
                manager.execute_operation(failing_operation)


# ============================================================================
# Async Operation Tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
class TestAsyncOperations:
    """
    Test asynchronous operation execution.

    Coverage: Verifies execute_operation_async handles async operations correctly,
    integrates with rate limiting, circuit breaker, and timeout handling.
    """

    async def test_execute_operation_async_success(
        self, mock_milvus_settings, mock_pymilvus_connections, test_operation
    ):
        """
        Test successful async operation execution.

        Coverage: execute_operation_async() with successful operation.
        """
        reset_connection_pool_singleton()
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.acquire_reference = Mock()
            mock_ctx = create_mock_pool_context_manager("conn_0")
            mock_pool.get_connection.return_value = mock_ctx
            mock_pymilvus_connections.has_connection.return_value = True

            manager = ConnectionManager(mock_milvus_settings, enable_circuit_breaker=False)

            result = await manager.execute_operation_async(test_operation)
            assert "Operation result for conn_0" in result

    async def test_execute_operation_async_with_rate_limiting(
        self, mock_milvus_settings, mock_pymilvus_connections, test_operation
    ):
        """
        Test async operation execution with rate limiting.

        Coverage: execute_operation_async() applying rate limiting.
        """
        reset_connection_pool_singleton()
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.acquire_reference = Mock()
            mock_ctx = create_mock_pool_context_manager("conn_0")
            mock_pool.get_connection.return_value = mock_ctx
            mock_pymilvus_connections.has_connection.return_value = True

            manager = ConnectionManager(mock_milvus_settings)
            if manager._rate_limiter:
                # Rate limiter should be called
                with patch.object(
                    manager._rate_limiter, "acquire", new_callable=AsyncMock
                ) as mock_acquire:
                    mock_acquire.return_value = 0.0  # No wait time

                    result = await manager.execute_operation_async(test_operation)
                    assert "Operation result for conn_0" in result
                    mock_acquire.assert_called_once()

    async def test_execute_operation_async_with_timeout(
        self, mock_milvus_settings, mock_pymilvus_connections, slow_operation
    ):
        """
        Test async operation execution with timeout.

        Coverage: execute_operation_async() timeout enforcement.
        """
        reset_connection_pool_singleton()
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.acquire_reference = Mock()
            mock_ctx = create_mock_pool_context_manager("conn_0")
            mock_pool.get_connection.return_value = mock_ctx
            mock_pymilvus_connections.has_connection.return_value = True

            manager = ConnectionManager(mock_milvus_settings, enable_circuit_breaker=False)

            # Operation with 2 second delay, but timeout of 0.5 seconds
            with pytest.raises(OperationTimeoutError, match="exceeded timeout"):
                await manager.execute_operation_async(slow_operation, timeout=0.5)


# ============================================================================
# Retry Logic Tests
# ============================================================================


@pytest.mark.unit
class TestRetryLogic:
    """
    Test retry logic implementation.

    Coverage: Verifies with_retry decorator handles retries correctly,
    implements exponential backoff, and respects retry budget.
    """

    def test_retry_on_connection_error(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test retry on connection error.

        Coverage: with_retry() retrying on ConnectionError.
        """
        reset_connection_pool_singleton()
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.acquire_reference = Mock()
            mock_ctx = create_mock_pool_context_manager("conn_0")
            mock_pool.get_connection.return_value = mock_ctx
            mock_pymilvus_connections.has_connection.return_value = True

            manager = ConnectionManager(mock_milvus_settings, enable_circuit_breaker=False)
            manager.config.connection.retry_count = 2
            manager.config.connection.retry_interval = 0.01

            call_count = 0

            def failing_operation(conn_alias: str) -> str:
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionError("Transient failure")
                return "success"

            @manager.with_retry
            def test_op(conn_alias: str) -> str:
                return failing_operation(conn_alias)

            result = test_op("conn_0")
            assert result == "success"
            assert call_count == 3  # Initial attempt + 2 retries

    def test_retry_with_retry_budget(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test retry with retry budget enforcement.

        Coverage: with_retry() respecting retry budget.
        """
        reset_connection_pool_singleton()
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.acquire_reference = Mock()
            mock_ctx = create_mock_pool_context_manager("conn_0")
            mock_pool.get_connection.return_value = mock_ctx
            mock_pymilvus_connections.has_connection.return_value = True

            manager = ConnectionManager(mock_milvus_settings, enable_circuit_breaker=False)
            manager.config.connection.retry_count = 5
            manager.config.connection.retry_interval = 0.01

            if manager._retry_budget:
                # Record many failures to exhaust retry budget
                for _ in range(10):
                    manager._retry_budget.record_attempt(success=False)

                def failing_operation(conn_alias: str) -> None:
                    raise ConnectionError("Failure")

                @manager.with_retry
                def test_op(conn_alias: str) -> None:
                    return failing_operation(conn_alias)

                # Should fail without retry due to exhausted budget
                with pytest.raises(MaxRetriesExceededError):
                    test_op("conn_0")


# ============================================================================
# Server Status Tests
# ============================================================================


@pytest.mark.unit
class TestServerStatus:
    """
    Test server status checking.

    Coverage: Verifies check_server_status correctly determines server availability.
    """

    def test_check_server_status_available(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test server status check when server is available.

        Coverage: check_server_status() with available server.
        """
        reset_connection_pool_singleton()
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.acquire_reference = Mock()
            mock_ctx = create_mock_pool_context_manager("conn_0")
            mock_pool.get_connection.return_value = mock_ctx
            mock_pymilvus_connections.has_connection.return_value = True

            manager = ConnectionManager(mock_milvus_settings)
            assert manager.check_server_status() is True

    def test_check_server_status_unavailable(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test server status check when server is unavailable.

        Coverage: check_server_status() with unavailable server.
        """
        reset_connection_pool_singleton()
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.acquire_reference = Mock()
            mock_pool.get_connection.side_effect = ConnectionError("Pool exhausted")
            mock_pymilvus_connections.has_connection.return_value = False

            manager = ConnectionManager(mock_milvus_settings)
            assert manager.check_server_status() is False


# ============================================================================
# Metrics Tests
# ============================================================================


@pytest.mark.unit
class TestMetrics:
    """
    Test metrics collection and retrieval.

    Coverage: Verifies all metrics methods return correct data.
    """

    def test_get_circuit_breaker_metrics(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test circuit breaker metrics retrieval.

        Coverage: get_circuit_breaker_metrics() returning metrics.
        """
        reset_connection_pool_singleton()
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.acquire_reference = Mock()
            mock_ctx = create_mock_pool_context_manager()
            mock_pool.get_connection.return_value = mock_ctx

            manager = ConnectionManager(mock_milvus_settings, enable_circuit_breaker=True)
            metrics = manager.get_circuit_breaker_metrics()
            assert metrics is not None
            assert "state" in metrics
            assert "counters" in metrics

    def test_get_rate_limiter_metrics(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test rate limiter metrics retrieval.

        Coverage: get_rate_limiter_metrics() returning metrics.
        """
        reset_connection_pool_singleton()
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.acquire_reference = Mock()
            mock_ctx = create_mock_pool_context_manager()
            mock_pool.get_connection.return_value = mock_ctx

            manager = ConnectionManager(mock_milvus_settings)
            metrics = manager.get_rate_limiter_metrics()
            if manager._rate_limiter:
                assert metrics is not None

    def test_get_retry_budget_metrics(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test retry budget metrics retrieval.

        Coverage: get_retry_budget_metrics() returning metrics.
        """
        reset_connection_pool_singleton()
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.acquire_reference = Mock()
            mock_ctx = create_mock_pool_context_manager()
            mock_pool.get_connection.return_value = mock_ctx

            manager = ConnectionManager(mock_milvus_settings)
            metrics = manager.get_retry_budget_metrics()
            if manager._retry_budget:
                assert metrics is not None

    def test_get_all_metrics(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test comprehensive metrics retrieval.

        Coverage: get_all_metrics() returning all metrics.
        """
        reset_connection_pool_singleton()
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.acquire_reference = Mock()
            mock_ctx = create_mock_pool_context_manager()
            mock_pool.get_connection.return_value = mock_ctx

            manager = ConnectionManager(mock_milvus_settings)
            all_metrics = manager.get_all_metrics()
            assert "circuit_breaker" in all_metrics
            assert "rate_limiter" in all_metrics
            assert "retry_budget" in all_metrics


# ============================================================================
# Resource Cleanup Tests
# ============================================================================


@pytest.mark.unit
class TestResourceCleanup:
    """
    Test resource cleanup and management.

    Coverage: Verifies close() properly releases all resources.
    """

    def test_close_releases_resources(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test that close releases all resources.

        Coverage: close() cleaning up executor and pool references.
        """
        reset_connection_pool_singleton()
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.acquire_reference = Mock()
            mock_pool.release_reference = Mock(return_value=False)
            mock_ctx = create_mock_pool_context_manager()
            mock_pool.get_connection.return_value = mock_ctx

            manager = ConnectionManager(mock_milvus_settings)
            manager.close()

            # Executor should be shut down
            assert manager._executor._shutdown is True
            # Pool reference should be released
            mock_pool.release_reference.assert_called()
