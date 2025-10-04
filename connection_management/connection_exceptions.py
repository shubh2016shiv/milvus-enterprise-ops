"""
Connection Management Exceptions

This module defines specialized exceptions for Milvus connection management,
providing detailed error reporting and handling for connection-related issues.

These exceptions contribute to system robustness by:
- Enabling precise error handling and recovery strategies
- Providing clear diagnostic information for debugging
- Allowing applications to implement graceful degradation
- Supporting monitoring and alerting systems with specific error types
- Facilitating proper resource cleanup in error scenarios
"""

from exceptions import ConnectionError as BaseConnectionError


class ConnectionError(BaseConnectionError):
    """
    Base exception for all connection-related errors.

    This base class ensures consistent error handling across the connection
    management system and allows applications to catch all connection errors
    uniformly while still providing access to specific error details.
    """
    pass


class ConnectionPoolExhaustedError(ConnectionError):
    """
    Raised when the connection pool has no available connections.

    This exception helps applications detect when the system is under high
    load and no connections are available in the pool. Applications can use
    this to implement queuing, scaling, or graceful degradation strategies.
    """
    pass


class ConnectionTimeoutError(ConnectionError):
    """
    Raised when a connection attempt times out.

    This exception indicates network issues, server overload, or configuration
    problems. It enables applications to distinguish between different types
    of connection failures and implement appropriate retry or fallback strategies.
    """
    pass


class ConnectionAuthenticationError(ConnectionError):
    """
    Raised when authentication to Milvus server fails.

    This exception signals credential or authorization issues, allowing
    applications to handle authentication failures separately from other
    connection problems and implement proper security measures.
    """
    pass


class ConnectionClosedError(ConnectionError):
    """
    Raised when attempting to use a closed connection.

    This exception prevents the use of stale or closed connections, which
    could lead to unpredictable behavior or errors. It ensures resource
    safety and helps detect connection lifecycle issues.
    """
    pass


class ConnectionInitializationError(ConnectionError):
    """
    Raised when the connection pool fails to initialize.

    This exception indicates fundamental setup problems during application
    startup, allowing for early failure detection and proper initialization
    error handling rather than encountering issues during runtime.
    """
    pass


class MaxRetriesExceededError(ConnectionError):
    """
    Raised when maximum connection retry attempts have been exceeded.

    This exception signals that transient failures have persisted beyond
    the configured retry limits, indicating potential systemic issues.
    Applications can use this to trigger circuit breakers or escalate to
    human intervention.
    """
    pass


class ServerUnavailableError(ConnectionError):
    """
    Raised when the Milvus server is unavailable.

    This exception helps distinguish between client-side connection issues
    and server-side availability problems, enabling applications to implement
    different recovery strategies for each scenario.
    """
    pass
