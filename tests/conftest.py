"""
Pytest configuration and shared fixtures for connection management tests.

This module provides comprehensive fixtures for mocking Milvus connections,
configurations, and test utilities to support thorough testing of the
connection management module.
"""

import asyncio
from collections.abc import Callable, Generator
from dataclasses import dataclass
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from config import MilvusSettings
from config.settings import ConnectionSettings
import pytest

# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def mock_connection_settings() -> MagicMock:
    """
    Create a mock connection settings object for testing.

    Coverage: Tests connection settings initialization and attribute access.
    """
    settings = MagicMock(spec=ConnectionSettings)
    # Set actual ConnectionSettings attributes
    settings.host = "localhost"
    settings.port = "19530"
    settings.user = "test_user"
    settings.password = "test_password"
    settings.secure = False
    settings.timeout = 10
    settings.connection_pool_size = 3  # Small pool for testing
    settings.retry_count = 2
    settings.retry_interval = 0.1  # Fast retries for testing
    settings.max_requests_per_second = 100
    settings.rate_limiter_burst_multiplier = 2.0
    settings.enable_retry_budget = True
    settings.retry_budget_min_success_rate = 0.8
    settings.retry_budget_window_seconds = 10
    # Add circuit breaker attributes (accessed via getattr)
    settings.circuit_breaker_failure_threshold = 3
    settings.circuit_breaker_recovery_timeout = 5.0
    settings.circuit_breaker_success_threshold = 2
    settings.circuit_breaker_max_half_open = 1
    return settings


@pytest.fixture
def mock_milvus_settings(mock_connection_settings: ConnectionSettings) -> MilvusSettings:
    """
    Create a mock MilvusSettings object for testing.

    Coverage: Tests MilvusSettings initialization with nested connection settings.
    """
    return MilvusSettings(connection=mock_connection_settings)


@pytest.fixture
def minimal_milvus_settings() -> MilvusSettings:
    """
    Create minimal MilvusSettings with defaults for testing edge cases.

    Coverage: Tests behavior with minimal/default configuration.
    """
    return MilvusSettings()


@pytest.fixture
def invalid_milvus_settings() -> dict[str, Any]:
    """
    Create invalid settings dictionary for testing validation.

    Coverage: Tests configuration validation and error handling.
    """
    return {
        "connection": {
            "host": "localhost",
            "port": "invalid_port",  # Invalid port
            "connection_pool_size": -1,  # Invalid pool size
        }
    }


# ============================================================================
# Mock Milvus Connection Fixtures
# ============================================================================


@pytest.fixture
def mock_pymilvus_connections() -> Generator[MagicMock, None, None]:
    """
    Mock the pymilvus.connections module.

    Coverage: Tests interaction with pymilvus connection API.
    Provides comprehensive mocking of connection operations.
    """
    with patch("milvus_ops.connection_management.connection_pool.connections") as mock_conn:
        # Mock connect method
        mock_conn.connect = MagicMock(return_value=None)

        # Mock disconnect method
        mock_conn.disconnect = MagicMock(return_value=None)

        # Mock has_connection to return True by default
        mock_conn.has_connection = MagicMock(return_value=True)

        # Mock get_connection method
        mock_connection_obj = MagicMock()
        mock_conn.get_connection = MagicMock(return_value=mock_connection_obj)

        yield mock_conn


@pytest.fixture
def mock_healthy_connection() -> MagicMock:
    """
    Create a mock healthy Milvus connection object.

    Coverage: Tests successful connection operations.
    """
    mock_conn = MagicMock()
    mock_conn.list_collections = MagicMock(return_value=["collection1", "collection2"])
    mock_conn.has_connection = MagicMock(return_value=True)
    return mock_conn


@pytest.fixture
def mock_unhealthy_connection() -> MagicMock:
    """
    Create a mock unhealthy Milvus connection object.

    Coverage: Tests handling of stale/broken connections.
    """
    mock_conn = MagicMock()
    mock_conn.list_collections = MagicMock(side_effect=Exception("Connection lost"))
    mock_conn.has_connection = MagicMock(return_value=False)
    return mock_conn


@pytest.fixture
def mock_connection_factory() -> Callable[[str, bool], MagicMock]:
    """
    Factory fixture to create mock connections with configurable behavior.

    Args:
        alias: Connection alias
        healthy: Whether connection should be healthy

    Returns:
        Mock connection object

    Coverage: Tests dynamic connection creation with various states.
    """

    def _create_mock_connection(alias: str = "conn_0", healthy: bool = True) -> MagicMock:
        mock_conn = MagicMock()
        mock_conn.alias = alias

        if healthy:
            mock_conn.has_connection = MagicMock(return_value=True)
            mock_conn.list_collections = MagicMock(return_value=[])
            mock_conn.search = MagicMock(return_value=[])
        else:
            mock_conn.has_connection = MagicMock(return_value=False)
            mock_conn.list_collections = MagicMock(side_effect=Exception("Connection failed"))

        return mock_conn

    return _create_mock_connection


# ============================================================================
# Mock load_settings Fixture
# ============================================================================


@pytest.fixture
def mock_load_settings(mock_milvus_settings: MilvusSettings) -> Generator[MagicMock, None, None]:
    """
    Mock the load_settings function to return test configuration.

    Coverage: Tests configuration loading in components that use load_settings.
    """
    with patch("config.load_settings") as mock_load:
        mock_load.return_value = mock_milvus_settings
        yield mock_load


# ============================================================================
# Async Test Fixtures
# ============================================================================


@pytest.fixture
def event_loop():
    """
    Create an event loop for async tests.

    Coverage: Ensures proper async test execution and cleanup.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def async_mock_operation() -> Callable[[Any], AsyncMock]:
    """
    Factory for creating async mock operations.

    Coverage: Tests async operation execution paths.
    """

    def _create_async_mock(
        result: Any = None, delay: float = 0.0, error: Exception | None = None
    ) -> AsyncMock:
        async def _async_operation(*args, **kwargs):
            if delay > 0:
                await asyncio.sleep(delay)
            if error:
                raise error
            return result

        return AsyncMock(side_effect=_async_operation)

    return _create_async_mock


# ============================================================================
# Circuit Breaker Test Fixtures
# ============================================================================


@pytest.fixture
def mock_circuit_breaker_config() -> dict[str, Any]:
    """
    Create circuit breaker configuration for testing.

    Coverage: Tests circuit breaker with various configurations.
    """
    return {
        "failure_threshold": 3,
        "recovery_timeout": 2.0,  # Short timeout for testing
        "half_open_success_threshold": 2,
        "max_half_open_requests": 1,
    }


# ============================================================================
# Time and Performance Test Fixtures
# ============================================================================


@pytest.fixture
def mock_time() -> Generator[MagicMock, None, None]:
    """
    Mock time module for testing time-dependent behavior.

    Coverage: Tests timeout handling, recovery delays, backoff calculations.
    """
    with (
        patch("time.time") as mock_time_func,
        patch("time.monotonic") as mock_monotonic,
        patch("time.sleep") as mock_sleep,
    ):
        current_time = 1000.0

        def _time_side_effect() -> float:
            return current_time

        def _monotonic_side_effect() -> float:
            return current_time

        def _sleep_side_effect(duration: float) -> None:
            nonlocal current_time
            current_time += duration

        mock_time_func.side_effect = _time_side_effect
        mock_monotonic.side_effect = _monotonic_side_effect
        mock_sleep.side_effect = _sleep_side_effect

        yield mock_time_func


@pytest.fixture
def performance_timer() -> Callable[[], float]:
    """
    Timer utility for performance benchmarks.

    Coverage: Performance test measurement accuracy.
    """

    def _timer() -> float:
        return time.perf_counter()

    return _timer


# ============================================================================
# Error Simulation Fixtures
# ============================================================================


@pytest.fixture
def error_scenarios() -> dict[str, Exception]:
    """
    Common error scenarios for testing error handling paths.

    Coverage: Tests all exception handling branches.
    """
    from milvus_ops.connection_management.connection_exceptions import (
        ConnectionError,
        ConnectionTimeoutError,
        ServerUnavailableError,
    )

    return {
        "connection_error": ConnectionError("Connection failed"),
        "timeout_error": ConnectionTimeoutError("Operation timed out"),
        "server_unavailable": ServerUnavailableError("Server unavailable"),
        "generic_error": Exception("Generic error"),
        "network_error": OSError("Network unreachable"),
    }


# ============================================================================
# Threading Test Fixtures
# ============================================================================


@pytest.fixture
def thread_count() -> int:
    """
    Default thread count for threading safety tests.

    Coverage: Multi-threading test configuration.
    """
    return 10


@pytest.fixture
def concurrent_operations() -> Callable[[int, Callable], list[Any]]:
    """
    Utility for running operations concurrently.

    Coverage: Tests concurrent access patterns.
    """
    import threading

    def _run_concurrent(num_threads: int, operation: Callable) -> list[Any]:
        results = []
        errors = []
        lock = threading.Lock()

        def _worker(thread_id: int) -> None:
            try:
                result = operation(thread_id)
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=_worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        if errors:
            raise Exception(f"Errors in concurrent operations: {errors}")

        return results

    return _run_concurrent


# ============================================================================
# Test Data Fixtures
# ============================================================================


@dataclass
class TestConnection:
    """Test connection data structure."""

    alias: str
    healthy: bool
    last_used: float


@pytest.fixture
def connection_aliases() -> list[str]:
    """
    Generate connection aliases for testing.

    Coverage: Tests with multiple connection identifiers.
    """
    return [f"conn_{i}" for i in range(5)]


@pytest.fixture
def test_operation() -> Callable[[str], str]:
    """
    Simple test operation function.

    Coverage: Tests operation execution through connection manager.
    """

    def _operation(conn_alias: str) -> str:
        return f"Operation result for {conn_alias}"

    return _operation


@pytest.fixture
def failing_operation() -> Callable[[str], None]:
    """
    Test operation that always fails.

    Coverage: Tests error handling and retry logic.
    """
    from milvus_ops.connection_management.connection_exceptions import (
        ConnectionError,
    )

    def _operation(conn_alias: str) -> None:
        raise ConnectionError(f"Operation failed for {conn_alias}")

    return _operation


@pytest.fixture
def slow_operation() -> Callable[[str, float], str]:
    """
    Test operation that simulates slow execution.

    Coverage: Tests timeout handling.
    """

    def _operation(conn_alias: str, delay: float = 1.0) -> str:
        time.sleep(delay)
        return f"Slow operation result for {conn_alias}"

    return _operation


# ============================================================================
# Cleanup Utilities
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_singletons():
    """
    Cleanup singleton instances between tests.

    Coverage: Ensures test isolation by resetting singletons.
    """
    yield

    # Reset connection pool singleton
    from milvus_ops.connection_management.connection_pool import MilvusConnectionPool

    with MilvusConnectionPool._lock:
        MilvusConnectionPool._instance = None
        MilvusConnectionPool._lock = type(MilvusConnectionPool._lock)()


# ============================================================================
# Marker Registration
# ============================================================================


def pytest_configure(config):
    """
    Register custom pytest markers.

    Coverage: Test organization and selective test execution.
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "performance: marks tests as performance benchmarks")
    config.addinivalue_line("markers", "threading: marks tests that require threading")
