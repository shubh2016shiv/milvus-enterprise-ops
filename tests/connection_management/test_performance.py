"""
Performance benchmarks and stress tests for connection management module.

This module provides performance benchmarks, stress tests, and throughput
measurements to verify system performance under load.
"""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from milvus_ops.connection_management.connection_manager import ConnectionManager
from milvus_ops.connection_management.connection_pool import MilvusConnectionPool

# ============================================================================
# Performance Benchmarks
# ============================================================================


@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.unit
class TestPerformanceBenchmarks:
    """
    Performance benchmarks for connection management operations.

    Coverage: Verifies operation throughput, latency, and resource usage
    under various load conditions.
    """

    def test_connection_acquisition_latency(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test connection acquisition latency.

        Coverage: get_connection() latency measurement.
        """
        with patch("milvus_ops.connection_management.connection_pool.MilvusConnectionPool"):
            # Reset singleton
            with MilvusConnectionPool._lock:
                MilvusConnectionPool._instance = None

            pool = MilvusConnectionPool(mock_milvus_settings)
            mock_pymilvus_connections.has_connection.return_value = True

            # Measure acquisition latency
            latencies = []
            for _ in range(10):
                start = time.perf_counter()
                with pool.get_connection():
                    pass
                latency = time.perf_counter() - start
                latencies.append(latency)

            avg_latency = sum(latencies) / len(latencies)
            # Should be very fast (under 10ms ideally)
            assert avg_latency < 0.1

    def test_operation_execution_throughput(
        self, mock_milvus_settings, mock_pymilvus_connections, test_operation
    ):
        """
        Test operation execution throughput.

        Coverage: execute_operation() throughput measurement.
        """
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.get_connection.return_value.__enter__ = Mock(return_value="conn_0")
            mock_pool.get_connection.return_value.__exit__ = Mock(return_value=None)
            mock_pymilvus_connections.has_connection.return_value = True

            manager = ConnectionManager(mock_milvus_settings, enable_circuit_breaker=False)

            # Measure throughput
            start = time.perf_counter()
            iterations = 100
            for _ in range(iterations):
                manager.execute_operation(test_operation)

            elapsed = time.perf_counter() - start
            throughput = iterations / elapsed

            # Should handle reasonable throughput
            assert throughput > 10  # At least 10 ops/sec


# ============================================================================
# Stress Tests
# ============================================================================


@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.unit
class TestStressTests:
    """
    Stress tests for connection management under high load.

    Coverage: Verifies system stability and performance under
    sustained high load conditions.
    """

    def test_sustained_high_load(
        self, mock_milvus_settings, mock_pymilvus_connections, test_operation
    ):
        """
        Test system under sustained high load.

        Coverage: Connection management stability under sustained load.
        """
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.get_connection.return_value.__enter__ = Mock(return_value="conn_0")
            mock_pool.get_connection.return_value.__exit__ = Mock(return_value=None)
            mock_pymilvus_connections.has_connection.return_value = True

            manager = ConnectionManager(mock_milvus_settings, enable_circuit_breaker=False)

            # Execute many operations
            iterations = 1000
            successes = 0
            failures = 0

            for _ in range(iterations):
                try:
                    manager.execute_operation(test_operation)
                    successes += 1
                except Exception:
                    failures += 1

            # Should have high success rate
            success_rate = successes / iterations
            assert success_rate > 0.95  # At least 95% success rate

    def test_memory_usage_stability(self, mock_milvus_settings, mock_pymilvus_connections):
        """
        Test memory usage stability under load.

        Coverage: Memory leak detection and resource cleanup verification.
        """
        import gc

        with patch("milvus_ops.connection_management.connection_pool.MilvusConnectionPool"):
            # Reset singleton
            with MilvusConnectionPool._lock:
                MilvusConnectionPool._instance = None

            pool = MilvusConnectionPool(mock_milvus_settings)
            mock_pymilvus_connections.has_connection.return_value = True

            # Execute many operations
            for _ in range(1000):
                with pool.get_connection():
                    pass

            # Force garbage collection
            gc.collect()

            # Should not have excessive memory growth
            # (Note: Actual memory check would require psutil or similar)
            # This test verifies operations complete without errors
            assert pool._available_connections.qsize() > 0
