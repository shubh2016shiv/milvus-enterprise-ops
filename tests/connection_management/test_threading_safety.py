"""
Threading safety tests for connection management module.

This module provides comprehensive tests for thread safety,
race conditions, and concurrent access patterns.
"""

import threading
import time
from unittest.mock import patch

import pytest

from milvus_ops.connection_management.circuit_breaker import (
    CircuitBreakerConfig,
    MilvusCircuitBreaker,
)
from milvus_ops.connection_management.connection_pool import MilvusConnectionPool

# ============================================================================
# Thread Safety Tests
# ============================================================================


@pytest.mark.threading
@pytest.mark.unit
class TestThreadingSafety:
    """
    Test thread safety of connection management components.

    Coverage: Verifies concurrent access patterns are safe and don't
    cause race conditions or data corruption.
    """

    def test_concurrent_connection_pool_access(
        self, mock_milvus_settings, mock_pymilvus_connections, thread_count
    ):
        """
        Test concurrent access to connection pool.

        Coverage: MilvusConnectionPool thread safety with concurrent access.
        """
        with patch("milvus_ops.connection_management.connection_pool.MilvusConnectionPool"):
            # Reset singleton for test isolation
            with MilvusConnectionPool._lock:
                MilvusConnectionPool._instance = None

            pool = MilvusConnectionPool(mock_milvus_settings)
            mock_pymilvus_connections.has_connection.return_value = True

            results = []
            errors = []
            lock = threading.Lock()

            def worker(thread_id: int) -> None:
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
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            # Some should succeed (within pool size), others may timeout
            assert len(results) > 0

    def test_concurrent_circuit_breaker_operations(self, mock_circuit_breaker_config):
        """
        Test concurrent circuit breaker operations.

        Coverage: MilvusCircuitBreaker thread safety with concurrent operations.
        """
        import asyncio

        async def concurrent_operations():
            # Convert dict to CircuitBreakerConfig
            config = CircuitBreakerConfig(**mock_circuit_breaker_config)
            breaker = MilvusCircuitBreaker(config=config, name="test_breaker")

            async def operation(value: int) -> int:
                return value

            # Execute multiple operations concurrently
            tasks = [breaker.execute_milvus_operation(operation, i) for i in range(10)]
            results = await asyncio.gather(*tasks)

            # All should succeed
            assert len(results) == 10
            assert results == list(range(10))

        asyncio.run(concurrent_operations())

    def test_concurrent_reference_counting(
        self, mock_milvus_settings, mock_pymilvus_connections, thread_count
    ):
        """
        Test concurrent reference counting operations.

        Coverage: MilvusConnectionPool reference counting thread safety.
        """
        with patch("milvus_ops.connection_management.connection_pool.MilvusConnectionPool"):
            # Reset singleton for test isolation
            with MilvusConnectionPool._lock:
                MilvusConnectionPool._instance = None

            pool = MilvusConnectionPool(mock_milvus_settings)

            def worker(thread_id: int) -> None:
                for _ in range(10):
                    pool.acquire_reference()
                    time.sleep(0.001)
                    pool.release_reference()

            threads = []
            for i in range(thread_count):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            # Reference count should be zero after all releases
            assert pool._reference_count == 0

    def test_race_condition_prevention(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test race condition prevention in connection pool.

        Coverage: MilvusConnectionPool preventing race conditions.
        """
        # Reset singleton for test isolation (autouse fixture should handle this)
        pool = MilvusConnectionPool(mock_milvus_settings)

        # Track concurrent connections in use
        concurrent_connections = set()
        max_concurrent_seen = [0]  # Use list to make it mutable across threads
        lock = threading.Lock()
        event = threading.Event()

        def worker(thread_id: int) -> None:
            try:
                with pool.get_connection() as conn_alias:
                    with lock:
                        concurrent_connections.add(conn_alias)
                        max_concurrent_seen[0] = max(
                            max_concurrent_seen[0], len(concurrent_connections)
                        )

                    # Signal that we've started
                    if thread_id == 0:
                        event.set()

                    time.sleep(0.05)  # Hold connection briefly

                    with lock:
                        concurrent_connections.discard(conn_alias)
            except Exception:
                pass

        # Start several threads simultaneously
        threads = []
        for i in range(10):  # Use fewer threads to avoid overwhelming the small test pool
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # With a pool size of 3, we should never see more than 3 connections in use simultaneously
        assert max_concurrent_seen[0] <= 3


# ============================================================================
# Deadlock Prevention Tests
# ============================================================================


@pytest.mark.threading
@pytest.mark.unit
class TestDeadlockPrevention:
    """
    Test deadlock prevention in concurrent scenarios.

    Coverage: Verifies system doesn't deadlock under concurrent load.
    """

    def test_no_deadlock_with_multiple_pools(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test no deadlock with multiple pool operations.

        Coverage: Deadlock prevention with multiple concurrent pool operations.
        """
        with patch("milvus_ops.connection_management.connection_pool.MilvusConnectionPool"):
            # Reset singleton for test isolation
            with MilvusConnectionPool._lock:
                MilvusConnectionPool._instance = None

            pool = MilvusConnectionPool(mock_milvus_settings)
            mock_pymilvus_connections.has_connection.return_value = True

            def worker() -> None:
                for _ in range(10):
                    try:
                        with pool.get_connection():
                            time.sleep(0.001)
                    except Exception:
                        pass

            threads = []
            for _ in range(5):
                thread = threading.Thread(target=worker)
                threads.append(thread)
                thread.start()

            # Wait with timeout to detect deadlocks
            for thread in threads:
                thread.join(timeout=5.0)
                # If thread didn't complete, it might be deadlocked
                assert not thread.is_alive()
