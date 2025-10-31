"""
Unit tests for connection management exceptions.

This module tests all exception classes in the connection_management module,
verifying inheritance, exception messages, and exception chaining behavior.
"""

import pytest

from milvus_ops.connection_management.connection_exceptions import (
    ConnectionAuthenticationError,
    ConnectionClosedError,
    ConnectionError,
    ConnectionInitializationError,
    ConnectionPoolExhaustedError,
    ConnectionTimeoutError,
    MaxRetriesExceededError,
    OperationTimeoutError,
    ServerUnavailableError,
)
from milvus_ops.milvus_ops_exceptions import ConnectionError as BaseConnectionError

# ============================================================================
# Exception Inheritance Tests
# ============================================================================


@pytest.mark.unit
class TestExceptionInheritance:
    """
    Test exception inheritance hierarchy.

    Coverage: Verifies all exception classes properly inherit from their
    base classes, ensuring correct exception handling behavior.
    """

    def test_connection_error_inheritance(self):
        """
        Test that ConnectionError inherits from BaseConnectionError.

        Coverage: ConnectionError.__class__.__mro__ and isinstance checks.
        """
        error = ConnectionError("test error")
        assert isinstance(error, ConnectionError)
        assert isinstance(error, BaseConnectionError)
        assert isinstance(error, Exception)

    def test_connection_pool_exhausted_error_inheritance(self):
        """
        Test that ConnectionPoolExhaustedError inherits from ConnectionError.

        Coverage: ConnectionPoolExhaustedError.__class__.__mro__ and isinstance checks.
        """
        error = ConnectionPoolExhaustedError("pool exhausted")
        assert isinstance(error, ConnectionPoolExhaustedError)
        assert isinstance(error, ConnectionError)
        assert isinstance(error, BaseConnectionError)
        assert isinstance(error, Exception)

    def test_connection_timeout_error_inheritance(self):
        """
        Test that ConnectionTimeoutError inherits from ConnectionError.

        Coverage: ConnectionTimeoutError.__class__.__mro__ and isinstance checks.
        """
        error = ConnectionTimeoutError("timeout occurred")
        assert isinstance(error, ConnectionTimeoutError)
        assert isinstance(error, ConnectionError)
        assert isinstance(error, BaseConnectionError)
        assert isinstance(error, Exception)

    def test_connection_authentication_error_inheritance(self):
        """
        Test that ConnectionAuthenticationError inherits from ConnectionError.

        Coverage: ConnectionAuthenticationError.__class__.__mro__ and isinstance checks.
        """
        error = ConnectionAuthenticationError("auth failed")
        assert isinstance(error, ConnectionAuthenticationError)
        assert isinstance(error, ConnectionError)
        assert isinstance(error, BaseConnectionError)
        assert isinstance(error, Exception)

    def test_connection_closed_error_inheritance(self):
        """
        Test that ConnectionClosedError inherits from ConnectionError.

        Coverage: ConnectionClosedError.__class__.__mro__ and isinstance checks.
        """
        error = ConnectionClosedError("connection closed")
        assert isinstance(error, ConnectionClosedError)
        assert isinstance(error, ConnectionError)
        assert isinstance(error, BaseConnectionError)
        assert isinstance(error, Exception)

    def test_connection_initialization_error_inheritance(self):
        """
        Test that ConnectionInitializationError inherits from ConnectionError.

        Coverage: ConnectionInitializationError.__class__.__mro__ and isinstance checks.
        """
        error = ConnectionInitializationError("init failed")
        assert isinstance(error, ConnectionInitializationError)
        assert isinstance(error, ConnectionError)
        assert isinstance(error, BaseConnectionError)
        assert isinstance(error, Exception)

    def test_max_retries_exceeded_error_inheritance(self):
        """
        Test that MaxRetriesExceededError inherits from ConnectionError.

        Coverage: MaxRetriesExceededError.__class__.__mro__ and isinstance checks.
        """
        error = MaxRetriesExceededError("max retries exceeded")
        assert isinstance(error, MaxRetriesExceededError)
        assert isinstance(error, ConnectionError)
        assert isinstance(error, BaseConnectionError)
        assert isinstance(error, Exception)

    def test_server_unavailable_error_inheritance(self):
        """
        Test that ServerUnavailableError inherits from ConnectionError.

        Coverage: ServerUnavailableError.__class__.__mro__ and isinstance checks.
        """
        error = ServerUnavailableError("server unavailable")
        assert isinstance(error, ServerUnavailableError)
        assert isinstance(error, ConnectionError)
        assert isinstance(error, BaseConnectionError)
        assert isinstance(error, Exception)

    def test_operation_timeout_error_imported(self):
        """
        Test that OperationTimeoutError is properly imported and accessible.

        Coverage: OperationTimeoutError import from connection_exceptions module.
        """
        assert OperationTimeoutError is not None
        error = OperationTimeoutError("operation timeout")
        assert isinstance(error, OperationTimeoutError)
        assert isinstance(error, Exception)


# ============================================================================
# Exception Message Tests
# ============================================================================


@pytest.mark.unit
class TestExceptionMessages:
    """
    Test exception message handling and preservation.

    Coverage: Verifies that exception messages are correctly stored and
    can be retrieved, ensuring proper error reporting.
    """

    def test_connection_error_message(self):
        """
        Test ConnectionError message storage and retrieval.

        Coverage: Exception.__str__ and exception message attribute access.
        """
        message = "Connection failed to Milvus server"
        error = ConnectionError(message)
        assert str(error) == message
        assert error.args[0] == message

    def test_connection_pool_exhausted_error_message(self):
        """
        Test ConnectionPoolExhaustedError message handling.

        Coverage: ConnectionPoolExhaustedError message storage and retrieval.
        """
        message = "No connections available in pool"
        error = ConnectionPoolExhaustedError(message)
        assert str(error) == message
        assert error.args[0] == message

    def test_connection_timeout_error_message(self):
        """
        Test ConnectionTimeoutError message handling.

        Coverage: ConnectionTimeoutError message storage and retrieval.
        """
        message = "Connection attempt timed out after 30 seconds"
        error = ConnectionTimeoutError(message)
        assert str(error) == message
        assert error.args[0] == message

    def test_connection_authentication_error_message(self):
        """
        Test ConnectionAuthenticationError message handling.

        Coverage: ConnectionAuthenticationError message storage and retrieval.
        """
        message = "Authentication failed: invalid credentials"
        error = ConnectionAuthenticationError(message)
        assert str(error) == message
        assert error.args[0] == message

    def test_connection_closed_error_message(self):
        """
        Test ConnectionClosedError message handling.

        Coverage: ConnectionClosedError message storage and retrieval.
        """
        message = "Attempted to use closed connection"
        error = ConnectionClosedError(message)
        assert str(error) == message
        assert error.args[0] == message

    def test_connection_initialization_error_message(self):
        """
        Test ConnectionInitializationError message handling.

        Coverage: ConnectionInitializationError message storage and retrieval.
        """
        message = "Failed to initialize connection pool"
        error = ConnectionInitializationError(message)
        assert str(error) == message
        assert error.args[0] == message

    def test_max_retries_exceeded_error_message(self):
        """
        Test MaxRetriesExceededError message handling.

        Coverage: MaxRetriesExceededError message storage and retrieval.
        """
        message = "Operation failed after 5 retry attempts"
        error = MaxRetriesExceededError(message)
        assert str(error) == message
        assert error.args[0] == message

    def test_server_unavailable_error_message(self):
        """
        Test ServerUnavailableError message handling.

        Coverage: ServerUnavailableError message storage and retrieval.
        """
        message = "Milvus server is not responding"
        error = ServerUnavailableError(message)
        assert str(error) == message
        assert error.args[0] == message

    def test_empty_message_handling(self):
        """
        Test exception creation with empty messages.

        Coverage: Exception handling with empty or None messages.
        """
        error = ConnectionError("")
        assert str(error) == ""
        assert error.args[0] == ""

    def test_multiple_args_handling(self):
        """
        Test exception creation with multiple arguments.

        Coverage: Exception.__init__ with multiple arguments handling.
        """
        error = ConnectionError("Error", "Additional context", 404)
        assert len(error.args) == 3
        assert error.args[0] == "Error"
        assert error.args[1] == "Additional context"
        assert error.args[2] == 404


# ============================================================================
# Exception Chaining Tests
# ============================================================================


@pytest.mark.unit
class TestExceptionChaining:
    """
    Test exception chaining behavior.

    Coverage: Verifies that exceptions can be chained using 'from' and
    '__cause__' attribute is properly set, enabling proper error tracing.
    """

    def test_exception_chaining_with_from(self):
        """
        Test exception chaining using 'raise ... from ...'.

        Coverage: Exception.__cause__ attribute setting and exception chaining.
        """
        original_error = ValueError("Original error")
        try:
            raise ConnectionError("Chained error") from original_error
        except ConnectionError as chained_error:
            assert chained_error.__cause__ is original_error
            assert isinstance(chained_error.__cause__, ValueError)

    def test_exception_chaining_with_from_none(self):
        """
        Test exception chaining with 'from None' to suppress context.

        Coverage: Exception.__cause__ = None behavior.
        """
        try:
            raise ValueError("Original error")
        except ValueError:
            try:
                raise ConnectionError("New error") from None
            except ConnectionError as chained_error:
                assert chained_error.__cause__ is None

    def test_exception_chaining_context(self):
        """
        Test exception chaining context preservation.

        Coverage: Exception.__context__ attribute and automatic chaining.
        """
        try:
            raise ValueError("Inner error")
        except ValueError:
            try:
                raise ConnectionError("Outer error")
            except ConnectionError as outer_error:
                # __context__ is set automatically when raising in except block
                assert outer_error.__context__ is not None
                assert isinstance(outer_error.__context__, ValueError)

    def test_nested_exception_chaining(self):
        """
        Test nested exception chaining across multiple levels.

        Coverage: Multi-level exception chaining with __cause__.
        """
        level1 = ValueError("Level 1")
        try:
            try:
                raise ConnectionError("Level 2") from level1
            except ConnectionError as level2:
                raise ConnectionTimeoutError("Level 3") from level2
        except ConnectionTimeoutError as level3:
            assert level3.__cause__ is not None
            assert isinstance(level3.__cause__, ConnectionError)
            assert level3.__cause__.__cause__ is level1


# ============================================================================
# Exception Type Checking Tests
# ============================================================================


@pytest.mark.unit
class TestExceptionTypeChecking:
    """
    Test exception type checking and catching behavior.

    Coverage: Verifies that exceptions can be caught by their base classes
    and that isinstance/issubclass checks work correctly.
    """

    def test_catching_base_exception(self):
        """
        Test catching specific exceptions with base exception handler.

        Coverage: Exception catching hierarchy and multiple except clauses.
        """
        try:
            raise ConnectionPoolExhaustedError("Pool exhausted")
        except ConnectionError as e:
            assert isinstance(e, ConnectionPoolExhaustedError)
            assert str(e) == "Pool exhausted"

    def test_catching_specific_exception(self):
        """
        Test catching specific exception type.

        Coverage: Specific exception type catching vs base class.
        """
        try:
            raise ConnectionTimeoutError("Timeout")
        except ConnectionTimeoutError as e:
            assert isinstance(e, ConnectionTimeoutError)
            assert str(e) == "Timeout"

    def test_issubclass_checks(self):
        """
        Test issubclass checks for exception classes.

        Coverage: issubclass() with exception class hierarchy.
        """
        assert issubclass(ConnectionError, BaseConnectionError)
        assert issubclass(ConnectionPoolExhaustedError, ConnectionError)
        assert issubclass(ConnectionPoolExhaustedError, BaseConnectionError)
        assert issubclass(ConnectionPoolExhaustedError, Exception)

    def test_multiple_exception_handling(self):
        """
        Test handling multiple exception types in one handler.

        Coverage: Catching multiple exception types in tuple.
        """
        exceptions = [
            ConnectionPoolExhaustedError("Pool exhausted"),
            ConnectionTimeoutError("Timeout"),
            ConnectionAuthenticationError("Auth failed"),
        ]

        for exc in exceptions:
            try:
                raise exc
            except (ConnectionPoolExhaustedError, ConnectionTimeoutError) as e:
                assert isinstance(e, ConnectionPoolExhaustedError | ConnectionTimeoutError)
            except ConnectionError as e:
                # AuthenticationError should be caught here
                assert isinstance(e, ConnectionAuthenticationError)


# ============================================================================
# Exception Repr and Str Tests
# ============================================================================


@pytest.mark.unit
class TestExceptionRepresentation:
    """
    Test exception string representation.

    Coverage: Verifies that __repr__ and __str__ methods work correctly
    for all exception types, enabling proper debugging.
    """

    def test_exception_str_representation(self):
        """
        Test exception string representation.

        Coverage: Exception.__str__ method and string conversion.
        """
        error = ConnectionError("Test error")
        str_repr = str(error)
        assert isinstance(str_repr, str)
        assert "Test error" in str_repr

    def test_exception_repr_representation(self):
        """
        Test exception repr representation.

        Coverage: Exception.__repr__ method and repr() conversion.
        """
        error = ConnectionError("Test error")
        repr_str = repr(error)
        assert isinstance(repr_str, str)
        assert "ConnectionError" in repr_str
        assert "Test error" in repr_str


# ============================================================================
# Exception Equality and Hashing Tests
# ============================================================================


@pytest.mark.unit
class TestExceptionEquality:
    """
    Test exception equality and hashing.

    Coverage: Verifies exception comparison behavior (exceptions are
    compared by identity, not value by default).
    """

    def test_exception_identity(self):
        """
        Test that exceptions are compared by identity, not value.

        Coverage: Exception identity comparison (is vs ==).
        """
        error1 = ConnectionError("Same message")
        error2 = ConnectionError("Same message")

        # Exceptions are compared by identity, not value
        assert error1 is not error2
        # Note: Exception.__eq__ is not implemented by default,
        # so == comparison uses identity
        assert error1 != error2

    def test_exception_hash(self):
        """
        Test exception hashing behavior.

        Coverage: Exception.__hash__ and hashability for use in sets/dicts.
        """
        error1 = ConnectionError("Error 1")
        error2 = ConnectionError("Error 2")

        # Exceptions should be hashable (they inherit from Exception)
        assert isinstance(hash(error1), int)
        assert isinstance(hash(error2), int)

        # Different exceptions should have different hashes (likely, but not guaranteed)
        # The important thing is they're hashable and can be used in sets
        error_set = {error1, error2}
        assert len(error_set) == 2


# ============================================================================
# Module __all__ Test
# ============================================================================


@pytest.mark.unit
class TestModuleExports:
    """
    Test module __all__ exports.

    Coverage: Verifies that all exceptions are properly exported and
    accessible via __all__ attribute.
    """

    def test_all_exceptions_exported(self):
        """
        Test that all exceptions are in __all__.

        Coverage: connection_exceptions.__all__ completeness.
        """
        from milvus_ops.connection_management import connection_exceptions

        assert hasattr(connection_exceptions, "__all__")
        exported = connection_exceptions.__all__

        expected_exceptions = [
            "ConnectionError",
            "ConnectionPoolExhaustedError",
            "ConnectionTimeoutError",
            "ConnectionAuthenticationError",
            "ConnectionClosedError",
            "ConnectionInitializationError",
            "MaxRetriesExceededError",
            "ServerUnavailableError",
            "OperationTimeoutError",
        ]

        for exc_name in expected_exceptions:
            assert exc_name in exported, f"{exc_name} not in __all__"

    def test_all_exceptions_importable(self):
        """
        Test that all exceptions can be imported from module.

        Coverage: Exception importability from connection_exceptions module.
        """
        # This test verifies that all exceptions in __all__ are actually importable
        from milvus_ops.connection_management.connection_exceptions import (
            ConnectionAuthenticationError,
            ConnectionClosedError,
            ConnectionError,
            ConnectionInitializationError,
            ConnectionPoolExhaustedError,
            ConnectionTimeoutError,
            MaxRetriesExceededError,
            OperationTimeoutError,
            ServerUnavailableError,
        )

        # If we get here, all imports succeeded
        assert ConnectionError is not None
        assert ConnectionPoolExhaustedError is not None
        assert ConnectionTimeoutError is not None
        assert ConnectionAuthenticationError is not None
        assert ConnectionClosedError is not None
        assert ConnectionInitializationError is not None
        assert MaxRetriesExceededError is not None
        assert ServerUnavailableError is not None
        assert OperationTimeoutError is not None
