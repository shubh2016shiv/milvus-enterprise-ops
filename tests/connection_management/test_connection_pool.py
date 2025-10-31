"""
Unit tests for connection pool implementation.

This module provides comprehensive tests for MilvusConnectionPool,
testing singleton pattern, thread safety, health checks, resource cleanup,
and all edge cases to achieve high code coverage.
"""

import threading
import time
from unittest.mock import patch

import pytest

from milvus_ops.connection_management.connection_exceptions import (
    ConnectionClosedError,
    ConnectionError,
    ConnectionInitializationError,
    ConnectionPoolExhaustedError,
)
from milvus_ops.connection_management.connection_pool import MilvusConnectionPool
from milvus_ops.milvus_ops_exceptions import ConfigurationError

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before each test to ensure clean state."""
    with MilvusConnectionPool._lock:
        MilvusConnectionPool._instance = None
        MilvusConnectionPool._initialized = False
    yield


# ============================================================================
# Singleton Pattern Tests
# ============================================================================


@pytest.mark.unit
class TestSingletonPattern:
    """
    Test singleton pattern implementation.

    Coverage: Verifies that only one pool instance exists across multiple
    instantiations, ensuring consistent connection management.
    """

    def test_singleton_instance(self, mock_milvus_settings):
        """
        Test that multiple instantiations return the same instance.

        Coverage: MilvusConnectionPool.__new__ singleton implementation.
        """
        pool1 = MilvusConnectionPool(mock_milvus_settings)
        pool2 = MilvusConnectionPool(mock_milvus_settings)

        assert pool1 is pool2
        assert id(pool1) == id(pool2)

    def test_singleton_with_none_config(self):
        """
        Test singleton behavior with None config.

        Coverage: MilvusConnectionPool.__new__ with None config.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool1 = MilvusConnectionPool(None)
        pool2 = MilvusConnectionPool(None)

        assert pool1 is pool2

    def test_singleton_reset_after_close(self, mock_milvus_settings):
        """
        Test that singleton resets after pool is closed.

        Coverage: _close_pool_internal() resetting _instance to None.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool1 = MilvusConnectionPool(mock_milvus_settings)
        pool1._close_pool_internal()

        # After close, instance should be None, allowing new pool
        with MilvusConnectionPool._lock:
            assert MilvusConnectionPool._instance is None

        # New instantiation should create new pool
        pool2 = MilvusConnectionPool(mock_milvus_settings)
        assert pool2 is not pool1


# ============================================================================
# Initialization Tests
# ============================================================================


@pytest.mark.unit
class TestPoolInitialization:
    """
    Test pool initialization with various configurations.

    Coverage: Verifies initialization with default/custom configs,
    configuration validation, and connection pool creation.
    """

    def test_initialization_with_config(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test initialization with provided configuration.

        Coverage: MilvusConnectionPool.__init__ with MilvusSettings.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)

        assert pool.config is mock_milvus_settings
        assert pool._initialized is True
        assert pool._closed is False
        assert pool._connection_count == 3  # From mock_connection_settings
        assert pool._reference_count == 0

    def test_reinitialization_with_same_config(
        self, mock_milvus_settings, mock_pymilvus_connections
    ):
        """
        Test reinitialization with the same configuration.

        Coverage: MilvusConnectionPool.__init__ idempotent behavior.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool1 = MilvusConnectionPool(mock_milvus_settings)
        connection_count_before = pool1._connection_count

        # Reinitialize with same config should not create new connections
        pool2 = MilvusConnectionPool(mock_milvus_settings)
        assert pool2 is pool1
        assert pool2._connection_count == connection_count_before

    def test_reinitialization_with_different_config(
        self, mock_milvus_settings, mock_pymilvus_connections
    ):
        """
        Test reinitialization with different configuration raises error.

        Coverage: MilvusConnectionPool.__init__ configuration mismatch detection.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        _ = MilvusConnectionPool(mock_milvus_settings)

        # Create different config
        from config import MilvusSettings
        from config.settings import ConnectionSettings

        different_settings = MilvusSettings(
            connection=ConnectionSettings(
                host="different_host",
                connection_pool_size=5,
            )
        )

        with pytest.raises(ConfigurationError, match="already initialized"):
            MilvusConnectionPool(different_settings)

    def test_initialization_failure_raises_error(self, mock_milvus_settings):
        """
        Test that initialization failure raises ConnectionInitializationError.

        Coverage: _initialize_pool() exception handling.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        # Mock connection.connect to fail
        with (
            patch("pymilvus.connections.connect", side_effect=Exception("Connection failed")),
            pytest.raises(ConnectionInitializationError, match="Failed to initialize"),
        ):
            MilvusConnectionPool(mock_milvus_settings)


# ============================================================================
# Connection Creation Tests
# ============================================================================


@pytest.mark.unit
class TestConnectionCreation:
    """
    Test connection creation and management.

    Coverage: Verifies _create_connection creates connections correctly,
    and _initialize_pool sets up the pool properly.
    """

    def test_create_connection(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test connection creation with proper parameters.

        Coverage: _create_connection() calling connections.connect correctly.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        _ = MilvusConnectionPool(mock_milvus_settings)

        # Verify connections.connect was called with correct parameters
        mock_pymilvus_connections.connect.assert_called()
        calls = mock_pymilvus_connections.connect.call_args_list
        assert len(calls) == 3  # connection_pool_size = 3

        # Check first call has correct parameters
        first_call = calls[0]
        assert first_call.kwargs["alias"].startswith("conn_")
        assert first_call.kwargs["host"] == mock_milvus_settings.connection.host
        assert first_call.kwargs["port"] == mock_milvus_settings.connection.port

    def test_initialize_pool_creates_connections(
        self, mock_milvus_settings, mock_pymilvus_connections
    ):
        """
        Test that pool initialization creates all connections.

        Coverage: _initialize_pool() creating connections for pool size.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)

        # Verify correct number of connections created
        assert pool._connection_count == 3
        assert pool._available_connections.qsize() == 3


# ============================================================================
# Health Check Tests
# ============================================================================


@pytest.mark.unit
class TestHealthChecks:
    """
    Test connection health check functionality.

    Coverage: Verifies _is_connection_healthy correctly identifies
    healthy and unhealthy connections.
    """

    def test_healthy_connection_check(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test health check for healthy connection.

        Coverage: _is_connection_healthy() with healthy connection.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)
        mock_pymilvus_connections.has_connection.return_value = True

        is_healthy = pool._is_connection_healthy("conn_0")
        assert is_healthy is True

    def test_unhealthy_connection_check(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test health check for unhealthy connection.

        Coverage: _is_connection_healthy() with unhealthy connection.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)
        mock_pymilvus_connections.has_connection.return_value = False

        is_healthy = pool._is_connection_healthy("conn_0")
        assert is_healthy is False

    def test_health_check_with_exception(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test health check handles exceptions gracefully.

        Coverage: _is_connection_healthy() exception handling.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)
        mock_pymilvus_connections.has_connection.side_effect = Exception("Check failed")

        is_healthy = pool._is_connection_healthy("conn_0")
        assert is_healthy is False


# ============================================================================
# Connection Acquisition Tests
# ============================================================================


@pytest.mark.unit
class TestConnectionAcquisition:
    """
    Test connection acquisition from pool.

    Coverage: Verifies get_connection() acquires connections correctly,
    handles timeouts, and tracks in-use connections.
    """

    def test_get_connection_success(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test successful connection acquisition.

        Coverage: get_connection() context manager with successful acquisition.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)
        mock_pymilvus_connections.has_connection.return_value = True

        with pool.get_connection() as conn_alias:
            assert conn_alias is not None
            assert conn_alias.startswith("conn_")
            assert conn_alias in pool._in_use_connections

        # Connection should be returned to pool after context exit
        assert conn_alias not in pool._in_use_connections

    def test_get_connection_with_timeout(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test connection acquisition with custom timeout.

        Coverage: get_connection() timeout parameter handling.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)
        mock_pymilvus_connections.has_connection.return_value = True

        with pool.get_connection(timeout=5.0) as conn_alias:
            assert conn_alias is not None

    def test_get_connection_pool_exhausted(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test connection acquisition when pool is exhausted.

        Coverage: get_connection() raising ConnectionPoolExhaustedError.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)
        mock_pymilvus_connections.has_connection.return_value = True

        # Acquire all connections
        connections_held = []
        for _ in range(3):  # pool size is 3
            conn = pool.get_connection()
            conn.__enter__()
            connections_held.append(conn)

        # Next acquisition should fail with timeout
        with (
            pytest.raises(ConnectionPoolExhaustedError, match="No connections available"),
            pool.get_connection(timeout=0.1),
        ):
            pass

        # Release connections
        for conn in connections_held:
            conn.__exit__(None, None, None)

    def test_get_connection_closed_pool(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test connection acquisition from closed pool.

        Coverage: get_connection() raising ConnectionClosedError.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)
        pool._closed = True

        with pytest.raises(ConnectionClosedError, match="pool is closed"), pool.get_connection():
            pass

    def test_get_connection_stale_connection_recreated(
        self, mock_milvus_settings, mock_pymilvus_connections
    ):
        """
        Test that stale connections are recreated.

        Coverage: get_connection() detecting and recreating stale connections.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)

        # First call returns False (stale), second returns True (recreated)
        mock_pymilvus_connections.has_connection.side_effect = [False, True]

        with pool.get_connection() as conn_alias:
            assert conn_alias is not None
            # Verify disconnect and reconnect were called
            mock_pymilvus_connections.disconnect.assert_called()
            mock_pymilvus_connections.connect.assert_called()

    def test_get_connection_stale_recreation_fails(
        self, mock_milvus_settings, mock_pymilvus_connections
    ):
        """
        Test handling when stale connection recreation fails.

        Coverage: get_connection() exception handling when recreation fails.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)
        mock_pymilvus_connections.has_connection.return_value = False
        mock_pymilvus_connections.disconnect.side_effect = Exception("Disconnect failed")

        with (
            pytest.raises(ConnectionError, match="Failed to restore connection"),
            pool.get_connection(),
        ):
            pass


# ============================================================================
# Connection Return Tests
# ============================================================================


@pytest.mark.unit
class TestConnectionReturn:
    """
    Test connection return to pool.

    Coverage: Verifies connections are properly returned, health checked,
    and recreated if unhealthy.
    """

    def test_connection_return_healthy(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test returning a healthy connection to pool.

        Coverage: get_connection() returning healthy connection.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)
        mock_pymilvus_connections.has_connection.return_value = True

        initial_available = pool._available_connections.qsize()

        with pool.get_connection() as conn_alias:
            pass

        # Connection should be returned
        assert pool._available_connections.qsize() == initial_available
        assert conn_alias not in pool._in_use_connections

    def test_connection_return_unhealthy_recreated(
        self, mock_milvus_settings, mock_pymilvus_connections
    ):
        """
        Test returning unhealthy connection triggers recreation.

        Coverage: get_connection() detecting unhealthy connection and recreating.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)
        # Healthy on acquisition, unhealthy on return
        mock_pymilvus_connections.has_connection.side_effect = [True, False, True]

        with pool.get_connection():
            pass

        # Should recreate connection
        mock_pymilvus_connections.disconnect.assert_called()
        assert pool._available_connections.qsize() > 0

    def test_connection_return_recreation_fails(
        self, mock_milvus_settings, mock_pymilvus_connections
    ):
        """
        Test handling when connection recreation fails on return.

        Coverage: get_connection() handling recreation failure gracefully.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)
        # Healthy on acquisition, unhealthy on return
        mock_pymilvus_connections.has_connection.side_effect = [True, False]
        mock_pymilvus_connections.disconnect.side_effect = Exception("Recreation failed")

        # Should not raise exception, but log error
        with pool.get_connection():
            pass

        # Pool size should be reduced (connection not returned)
        assert pool._available_connections.qsize() < 3

    def test_connection_return_when_pool_closed(
        self, mock_milvus_settings, mock_pymilvus_connections
    ):
        """
        Test connection return behavior when pool is closed.

        Coverage: get_connection() closing connection when pool is closed.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)
        mock_pymilvus_connections.has_connection.return_value = True

        # Acquire connection
        with pool.get_connection():
            # Close pool while connection is in use
            pool._closed = True

        # Connection should be disconnected
        mock_pymilvus_connections.disconnect.assert_called()


# ============================================================================
# Reference Counting Tests
# ============================================================================


@pytest.mark.unit
class TestReferenceCounting:
    """
    Test reference counting functionality.

    Coverage: Verifies acquire_reference and release_reference work correctly,
    and pool closes only when reference count reaches zero.
    """

    def test_acquire_reference(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test acquiring a reference increments count.

        Coverage: acquire_reference() incrementing _reference_count.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)
        assert pool._reference_count == 0

        pool.acquire_reference()
        assert pool._reference_count == 1

        pool.acquire_reference()
        assert pool._reference_count == 2

    def test_release_reference_decrements_count(
        self, mock_milvus_settings, mock_pymilvus_connections
    ):
        """
        Test releasing a reference decrements count.

        Coverage: release_reference() decrementing _reference_count.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)
        pool.acquire_reference()
        pool.acquire_reference()
        assert pool._reference_count == 2

        pool.release_reference()
        assert pool._reference_count == 1

    def test_release_reference_closes_pool_at_zero(
        self, mock_milvus_settings, mock_pymilvus_connections
    ):
        """
        Test that pool closes when reference count reaches zero.

        Coverage: release_reference() calling _close_pool_internal() at zero.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)
        pool.acquire_reference()

        result = pool.release_reference()
        assert result is True
        assert pool._closed is True

    def test_release_reference_does_not_close_when_still_in_use(
        self, mock_milvus_settings, mock_pymilvus_connections
    ):
        """
        Test that pool does not close when references still exist.

        Coverage: release_reference() not closing pool with remaining references.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)
        pool.acquire_reference()
        pool.acquire_reference()

        result = pool.release_reference()
        assert result is False
        assert pool._closed is False
        assert pool._reference_count == 1

    def test_release_reference_at_zero_does_not_raise(
        self, mock_milvus_settings, mock_pymilvus_connections
    ):
        """
        Test releasing reference when already at zero.

        Coverage: release_reference() handling zero reference count.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)
        assert pool._reference_count == 0

        result = pool.release_reference()
        assert result is False


# ============================================================================
# Pool Cleanup Tests
# ============================================================================


@pytest.mark.unit
class TestPoolCleanup:
    """
    Test pool cleanup and resource release.

    Coverage: Verifies _close_pool_internal closes all connections,
    handles cleanup gracefully, and resets singleton.
    """

    def test_close_pool_internal_closes_connections(
        self, mock_milvus_settings, mock_pymilvus_connections
    ):
        """
        Test that closing pool disconnects all available connections.

        Coverage: _close_pool_internal() closing available connections.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)
        initial_disconnect_calls = mock_pymilvus_connections.disconnect.call_count

        pool._close_pool_internal()

        # Should disconnect all available connections
        assert pool._closed is True
        assert mock_pymilvus_connections.disconnect.call_count > initial_disconnect_calls

    def test_close_pool_internal_idempotent(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test that closing pool multiple times is idempotent.

        Coverage: _close_pool_internal() idempotent behavior.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)

        pool._close_pool_internal()
        assert pool._closed is True

        # Second close should not raise exception
        pool._close_pool_internal()
        assert pool._closed is True

    def test_close_pool_internal_warns_about_in_use(
        self, mock_milvus_settings, mock_pymilvus_connections
    ):
        """
        Test that closing pool warns about in-use connections.

        Coverage: _close_pool_internal() warning about in-use connections.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)
        mock_pymilvus_connections.has_connection.return_value = True

        # Acquire a connection
        conn = pool.get_connection()
        conn_alias = conn.__enter__()
        assert conn_alias in pool._in_use_connections

        # Close pool while connection is in use
        pool._close_pool_internal()
        assert pool._closed is True

    def test_close_deprecated_method(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test deprecated close() method.

        Coverage: close() calling _close_pool_internal().
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)

        pool.close()
        assert pool._closed is True


# ============================================================================
# Thread Safety Tests (Basic)
# ============================================================================


@pytest.mark.threading
@pytest.mark.unit
class TestThreadSafety:
    """
    Test thread safety of connection pool operations.

    Coverage: Verifies concurrent access patterns are safe and don't cause
    race conditions or data corruption.
    """

    def test_concurrent_connection_acquisition(
        self, mock_milvus_settings, mock_pymilvus_connections, thread_count
    ):
        """
        Test concurrent connection acquisition from multiple threads.

        Coverage: get_connection() thread safety.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)
        mock_pymilvus_connections.has_connection.return_value = True

        results = []
        errors = []
        lock = threading.Lock()

        def acquire_connection(thread_id: int) -> None:
            try:
                with pool.get_connection() as conn_alias:
                    with lock:
                        results.append((thread_id, conn_alias))
                    time.sleep(0.01)  # Simulate work
            except Exception as e:
                with lock:
                    errors.append((thread_id, e))

        threads = []
        for i in range(thread_count):
            thread = threading.Thread(target=acquire_connection, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Some should succeed (within pool size), others may timeout
        assert len(results) > 0
        # Errors should be ConnectionPoolExhaustedError only
        for _, error in errors:
            assert isinstance(error, ConnectionPoolExhaustedError)

    def test_concurrent_reference_operations(
        self, mock_milvus_settings, mock_pymilvus_connections, thread_count
    ):
        """
        Test concurrent reference acquire/release operations.

        Coverage: acquire_reference() and release_reference() thread safety.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)

        def acquire_release(thread_id: int) -> None:
            for _ in range(10):
                pool.acquire_reference()
                time.sleep(0.001)
                pool.release_reference()

        threads = []
        for i in range(thread_count):
            thread = threading.Thread(target=acquire_release, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Reference count should be zero after all releases
        assert pool._reference_count == 0


# ============================================================================
# Edge Cases Tests
# ============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """
    Test edge cases and boundary conditions.

    Coverage: Verifies behavior with edge case scenarios including
    empty pool, boundary values, and error conditions.
    """

    def test_get_connection_with_none_timeout(
        self, mock_milvus_settings, mock_pymilvus_connections
    ):
        """
        Test connection acquisition with None timeout uses config timeout.

        Coverage: get_connection() using config timeout when None provided.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)
        mock_pymilvus_connections.has_connection.return_value = True

        with pool.get_connection(timeout=None) as conn_alias:
            assert conn_alias is not None

    def test_connection_not_in_use_on_return(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test that connection removed from in_use even if not present.

        Coverage: get_connection() finally block safety.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)
        mock_pymilvus_connections.has_connection.return_value = True

        with pool.get_connection() as conn_alias:
            # Manually remove from in_use to simulate edge case
            pool._in_use_connections.discard(conn_alias)

        # Should not raise exception
        assert conn_alias not in pool._in_use_connections

    def test_pool_cleanup_with_exception(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test pool cleanup handles exceptions gracefully.

        Coverage: _close_pool_internal() exception handling in cleanup.
        """
        # Reset singleton for test isolation
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None

        pool = MilvusConnectionPool(mock_milvus_settings)
        # Make disconnect raise exception
        mock_pymilvus_connections.disconnect.side_effect = Exception("Disconnect error")

        # Should not raise, but handle gracefully
        pool._close_pool_internal()
        assert pool._closed is True
