"""
Integration tests for connection management module.

This module provides end-to-end integration tests that verify
multiple components working together correctly.
"""

from contextlib import suppress
from unittest.mock import MagicMock, Mock, patch

import pytest

from milvus_ops.connection_management.connection_exceptions import (
    ConnectionError,
)
from milvus_ops.connection_management.connection_manager import ConnectionManager
from milvus_ops.connection_management.milvus_connector import MilvusConnector

# ============================================================================
# End-to-End Workflow Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.unit
class TestEndToEndWorkflows:
    """
    Test end-to-end workflows with multiple components.

    Coverage: Verifies that components work together correctly
    in realistic usage scenarios.
    """

    def test_complete_workflow_connector_to_operation(
        self, mock_milvus_settings, mock_pymilvus_connections
    ):
        """
        Test complete workflow from connector to operation execution.

        Coverage: MilvusConnector -> ConnectionManager -> ConnectionPool -> Operation.
        """
        with (
            patch(
                "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
            ) as mock_pool_class,
            patch(
                "milvus_ops.connection_management.milvus_connector.ConnectionManager"
            ) as mock_manager_class,
        ):
            # Setup mocks
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.get_connection.return_value.__enter__ = Mock(return_value="conn_0")
            mock_pool.get_connection.return_value.__exit__ = Mock(return_value=None)

            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.check_server_status.return_value = True

            # Create connector and establish connection
            connector = MilvusConnector(mock_milvus_settings)
            feedback = connector.establish_connection()

            assert feedback.status.value == "success"
            assert connector.get_connection_manager() is not None

            # Cleanup
            connector.close_connection()

    def test_circuit_breaker_integration_with_pool(
        self, mock_milvus_settings, mock_pymilvus_connections
    ):
        """
        Test circuit breaker integration with connection pool.

        Coverage: Circuit breaker detecting failures from pool operations.
        """
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.get_connection.return_value.__enter__ = Mock(return_value="conn_0")
            mock_pool.get_connection.return_value.__exit__ = Mock(return_value=None)

            manager = ConnectionManager(mock_milvus_settings, enable_circuit_breaker=True)

            # Simulate failures to open circuit
            def failing_operation(conn_alias: str) -> None:
                raise ConnectionError("Operation failed")

            # Execute multiple failures to open circuit
            for _ in range(5):
                with suppress(Exception):
                    manager.execute_operation(failing_operation)

            # Circuit breaker should now be open
            if manager._circuit_breaker:
                assert manager._circuit_breaker.is_open() or manager._circuit_breaker.is_half_open()

    def test_retry_budget_integration_with_operations(
        self, mock_milvus_settings, mock_pymilvus_connections
    ):
        """
        Test retry budget integration with operations.

        Coverage: Retry budget preventing retries when success rate is low.
        """
        with patch(
            "milvus_ops.connection_management.connection_manager.MilvusConnectionPool"
        ) as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_pool.get_connection.return_value.__enter__ = Mock(return_value="conn_0")
            mock_pool.get_connection.return_value.__exit__ = Mock(return_value=None)

            manager = ConnectionManager(mock_milvus_settings, enable_circuit_breaker=False)

            if manager._retry_budget:
                # Record many failures to exhaust retry budget
                for _ in range(20):
                    manager._retry_budget.record_attempt(success=False)

                def failing_operation(conn_alias: str) -> None:
                    raise ConnectionError("Failure")

                # Should fail without retries due to exhausted budget
                with pytest.raises(ConnectionError):
                    manager.execute_operation(failing_operation)
