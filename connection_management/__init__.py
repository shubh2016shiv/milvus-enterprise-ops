"""
Connection Management Module

This module provides enterprise-grade connection management for Milvus vector database,
designed to handle high-volume concurrent access with built-in resilience and fault tolerance.

Key capabilities:
- Thread-safe connection pooling for efficient resource management
- Singleton pattern ensuring consistent connection management across the application
- Exponential backoff with jitter for robust retry mechanisms
- Connection health monitoring to prevent stale connection usage
- Automatic retry policies for handling transient failures
- Comprehensive error handling with specific exception types
- Both synchronous and asynchronous operation support
- Proper resource cleanup to prevent connection leaks

The module is specifically designed for production environments handling millions of users,
providing the scalability, robustness, and fault tolerance required for enterprise applications.
"""

from .connection_manager import ConnectionManager
from .connection_pool import MilvusConnectionPool
from .circuit_breaker import MilvusCircuitBreaker, CircuitBreakerConfig
from .milvus_connector import MilvusConnector, ConnectionStatus, ConnectionFeedback
from .connection_exceptions import (
    ConnectionError,
    ConnectionPoolExhaustedError,
    ConnectionTimeoutError,
    ConnectionAuthenticationError,
    ConnectionClosedError,
    ConnectionInitializationError,
    MaxRetriesExceededError,
    ServerUnavailableError
)

__all__ = [
    'ConnectionManager',
    'MilvusConnectionPool',
    'MilvusCircuitBreaker',
    'CircuitBreakerConfig',
    'MilvusConnector',
    'ConnectionStatus',
    'ConnectionFeedback',
    'ConnectionError',
    'ConnectionPoolExhaustedError',
    'ConnectionTimeoutError',
    'ConnectionAuthenticationError',
    'ConnectionClosedError',
    'ConnectionInitializationError',
    'MaxRetriesExceededError',
    'ServerUnavailableError',
]