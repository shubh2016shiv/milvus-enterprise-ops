"""
Unit tests for Milvus connector implementation.

This module provides comprehensive tests for MilvusConnector,
testing connection establishment, context manager behavior,
and all edge cases to achieve high code coverage.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from milvus_ops.connection_management.connection_exceptions import (
    ConnectionError,
    ServerUnavailableError,
)
from milvus_ops.connection_management.milvus_connector import (
    ConnectionFeedback,
    ConnectionStatus,
    MilvusConnector,
)

# ============================================================================
# Initialization Tests
# ============================================================================


@pytest.mark.unit
class TestMilvusConnectorInitialization:
    """
    Test MilvusConnector initialization.

    Coverage: Verifies initialization with default/custom configs and
    circuit breaker options.
    """

    def test_initialization_with_default_config(self, mock_load_settings):
        """
        Test initialization with default configuration.

        Coverage: MilvusConnector.__init__ with None config.
        """
        connector = MilvusConnector()
        assert connector.config is not None
        assert connector.enable_circuit_breaker is True
        assert connector._connection_manager is None

    def test_initialization_with_custom_config(self, mock_milvus_settings):
        """
        Test initialization with custom configuration.

        Coverage: MilvusConnector.__init__ with MilvusSettings.
        """
        connector = MilvusConnector(mock_milvus_settings, enable_circuit_breaker=True)
        assert connector.config is mock_milvus_settings
        assert connector.enable_circuit_breaker is True

    def test_initialization_without_circuit_breaker(self, mock_milvus_settings):
        """
        Test initialization without circuit breaker.

        Coverage: MilvusConnector.__init__ with enable_circuit_breaker=False.
        """
        connector = MilvusConnector(mock_milvus_settings, enable_circuit_breaker=False)
        assert connector.enable_circuit_breaker is False


# ============================================================================
# Connection Establishment Tests
# ============================================================================


@pytest.mark.unit
class TestConnectionEstablishment:
    """
    Test connection establishment functionality.

    Coverage: Verifies establish_connection handles various scenarios
    including success, failure, and unavailable states.
    """

    def test_establish_connection_success(self, mock_milvus_settings):
        """
        Test successful connection establishment.

        Coverage: establish_connection() with successful connection.
        """
        with patch(
            "milvus_ops.connection_management.milvus_connector.ConnectionManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.check_server_status.return_value = True

            connector = MilvusConnector(mock_milvus_settings)
            feedback = connector.establish_connection()

            assert feedback.status == ConnectionStatus.SUCCESS
            assert "milvus-conn-" in feedback.milvus_connection_id
            assert feedback.connection_manager is mock_manager
            assert "established successfully" in feedback.message

    def test_establish_connection_unavailable(self, mock_milvus_settings):
        """
        Test connection establishment when server is unavailable.

        Coverage: establish_connection() with unavailable server.
        """
        with patch(
            "milvus_ops.connection_management.milvus_connector.ConnectionManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.check_server_status.return_value = False

            connector = MilvusConnector(mock_milvus_settings)
            feedback = connector.establish_connection()

            assert feedback.status == ConnectionStatus.UNAVAILABLE
            assert "milvus-conn-" in feedback.milvus_connection_id
            assert feedback.connection_manager is mock_manager
            assert "unavailable" in feedback.message.lower()

    def test_establish_connection_with_connection_error(self, mock_milvus_settings):
        """
        Test connection establishment with ConnectionError.

        Coverage: establish_connection() exception handling for ConnectionError.
        """
        with patch(
            "milvus_ops.connection_management.milvus_connector.ConnectionManager"
        ) as mock_manager_class:
            mock_manager_class.side_effect = ConnectionError("Connection failed")

            connector = MilvusConnector(mock_milvus_settings)
            feedback = connector.establish_connection()

            assert feedback.status == ConnectionStatus.FAILURE
            assert "milvus-conn-" in feedback.milvus_connection_id
            assert "Failed to establish" in feedback.message

    def test_establish_connection_with_server_unavailable_error(self, mock_milvus_settings):
        """
        Test connection establishment with ServerUnavailableError.

        Coverage: establish_connection() exception handling for ServerUnavailableError.
        """
        with patch(
            "milvus_ops.connection_management.milvus_connector.ConnectionManager"
        ) as mock_manager_class:
            mock_manager_class.side_effect = ServerUnavailableError("Server unavailable")

            connector = MilvusConnector(mock_milvus_settings)
            feedback = connector.establish_connection()

            assert feedback.status == ConnectionStatus.FAILURE
            assert "milvus-conn-" in feedback.milvus_connection_id
            assert "Failed to establish" in feedback.message

    def test_establish_connection_with_unexpected_error(self, mock_milvus_settings):
        """
        Test connection establishment with unexpected error.

        Coverage: establish_connection() exception handling for generic exceptions.
        """
        with patch(
            "milvus_ops.connection_management.milvus_connector.ConnectionManager"
        ) as mock_manager_class:
            mock_manager_class.side_effect = ValueError("Unexpected error")

            connector = MilvusConnector(mock_milvus_settings)
            feedback = connector.establish_connection()

            assert feedback.status == ConnectionStatus.FAILURE
            assert "milvus-conn-" in feedback.milvus_connection_id
            assert "unexpected error" in feedback.message.lower()

    def test_establish_connection_generates_unique_id(self, mock_milvus_settings):
        """
        Test that connection establishment generates unique IDs.

        Coverage: establish_connection() UUID generation.
        """
        with patch(
            "milvus_ops.connection_management.milvus_connector.ConnectionManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.check_server_status.return_value = True

            connector = MilvusConnector(mock_milvus_settings)
            feedback1 = connector.establish_connection()
            feedback2 = connector.establish_connection()

            assert feedback1.milvus_connection_id != feedback2.milvus_connection_id


# ============================================================================
# Connection Manager Access Tests
# ============================================================================


@pytest.mark.unit
class TestConnectionManagerAccess:
    """
    Test connection manager access methods.

    Coverage: Verifies get_connection_manager returns the correct manager.
    """

    def test_get_connection_manager_before_establishment(self, mock_milvus_settings):
        """
        Test getting connection manager before establishment.

        Coverage: get_connection_manager() before establish_connection().
        """
        connector = MilvusConnector(mock_milvus_settings)
        assert connector.get_connection_manager() is None

    def test_get_connection_manager_after_establishment(self, mock_milvus_settings):
        """
        Test getting connection manager after establishment.

        Coverage: get_connection_manager() after establish_connection().
        """
        with patch(
            "milvus_ops.connection_management.milvus_connector.ConnectionManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.check_server_status.return_value = True

            connector = MilvusConnector(mock_milvus_settings)
            connector.establish_connection()

            assert connector.get_connection_manager() is mock_manager


# ============================================================================
# Connection Cleanup Tests
# ============================================================================


@pytest.mark.unit
class TestConnectionCleanup:
    """
    Test connection cleanup functionality.

    Coverage: Verifies close_connection properly releases resources.
    """

    def test_close_connection_with_manager(self, mock_milvus_settings):
        """
        Test closing connection when manager exists.

        Coverage: close_connection() with active connection manager.
        """
        with patch(
            "milvus_ops.connection_management.milvus_connector.ConnectionManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.check_server_status.return_value = True

            connector = MilvusConnector(mock_milvus_settings)
            connector.establish_connection()
            connector.close_connection()

            mock_manager.close.assert_called_once()

    def test_close_connection_without_manager(self, mock_milvus_settings):
        """
        Test closing connection when no manager exists.

        Coverage: close_connection() without active connection manager.
        """
        connector = MilvusConnector(mock_milvus_settings)
        # Should not raise exception
        connector.close_connection()
        assert connector._connection_manager is None


# ============================================================================
# Context Manager Tests
# ============================================================================


@pytest.mark.unit
class TestContextManager:
    """
    Test context manager functionality.

    Coverage: Verifies __enter__ and __exit__ methods work correctly.
    """

    def test_context_manager_success(self, mock_milvus_settings):
        """
        Test context manager with successful connection.

        Coverage: __enter__() and __exit__() with successful connection.
        """
        with patch(
            "milvus_ops.connection_management.milvus_connector.ConnectionManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.check_server_status.return_value = True

            connector = MilvusConnector(mock_milvus_settings)

            with connector:
                assert connector._connection_manager is not None

            # Connection should be closed on exit
            mock_manager.close.assert_called_once()

    def test_context_manager_connection_failure(self, mock_milvus_settings):
        """
        Test context manager with connection failure.

        Coverage: __enter__() raising ConnectionError on failure.
        """
        with patch(
            "milvus_ops.connection_management.milvus_connector.ConnectionManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.check_server_status.return_value = False

            connector = MilvusConnector(mock_milvus_settings)

            with pytest.raises(ConnectionError, match="Failed to establish connection"), connector:
                pass

    def test_context_manager_exit_always_closes(self, mock_milvus_settings):
        """
        Test that context manager exit always closes connection.

        Coverage: __exit__() always calling close_connection().
        """
        with patch(
            "milvus_ops.connection_management.milvus_connector.ConnectionManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.check_server_status.return_value = True

            connector = MilvusConnector(mock_milvus_settings)

            try:
                with connector:
                    raise ValueError("Test exception")
            except ValueError:
                pass

            # Connection should still be closed even with exception
            mock_manager.close.assert_called_once()


# ============================================================================
# ConnectionStatus Enum Tests
# ============================================================================


@pytest.mark.unit
class TestConnectionStatus:
    """
    Test ConnectionStatus enum.

    Coverage: Verifies all enum values are correctly defined.
    """

    def test_connection_status_values(self):
        """
        Test all connection status enum values.

        Coverage: ConnectionStatus enum value access.
        """
        assert ConnectionStatus.SUCCESS == "success"
        assert ConnectionStatus.FAILURE == "failure"
        assert ConnectionStatus.UNAVAILABLE == "unavailable"
        assert ConnectionStatus.PENDING == "pending"

    def test_connection_status_enum_inheritance(self):
        """
        Test that ConnectionStatus is a string enum.

        Coverage: ConnectionStatus enum inheritance and type.
        """
        assert isinstance(ConnectionStatus.SUCCESS, str)
        assert isinstance(ConnectionStatus.FAILURE, str)


# ============================================================================
# ConnectionFeedback Tests
# ============================================================================


@pytest.mark.unit
class TestConnectionFeedback:
    """
    Test ConnectionFeedback dataclass.

    Coverage: Verifies ConnectionFeedback dataclass initialization and attributes.
    """

    def test_connection_feedback_initialization(self):
        """
        Test ConnectionFeedback initialization.

        Coverage: ConnectionFeedback.__init__() with all parameters.
        """
        feedback = ConnectionFeedback(
            milvus_connection_id="test-id",
            status=ConnectionStatus.SUCCESS,
            message="Success",
            connection_manager=Mock(),
        )
        assert feedback.milvus_connection_id == "test-id"
        assert feedback.status == ConnectionStatus.SUCCESS
        assert feedback.message == "Success"
        assert feedback.connection_manager is not None

    def test_connection_feedback_with_none_manager(self):
        """
        Test ConnectionFeedback with None connection manager.

        Coverage: ConnectionFeedback with default None connection_manager.
        """
        feedback = ConnectionFeedback(
            milvus_connection_id="test-id",
            status=ConnectionStatus.FAILURE,
            message="Failure",
        )
        assert feedback.connection_manager is None
