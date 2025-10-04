"""
Milvus Connection Pool

This module provides a thread-safe connection pool for Milvus,
designed to handle high-volume concurrent access efficiently.
"""

import time
import logging
import threading
import queue
from typing import Dict, Any, Optional, List, Callable
from contextlib import contextmanager

from pymilvus import connections

from config import MilvusSettings
from .connection_exceptions import (
    ConnectionPoolExhaustedError,
    ConnectionInitializationError,
    ConnectionClosedError
)

# Logger setup
logger = logging.getLogger(__name__)


class MilvusConnectionPool:
    """
    Thread-safe connection pool for Milvus.

    This class implements a singleton connection pool that manages a collection
    of Milvus connections, providing thread-safe access for high-concurrency
    applications. The singleton pattern ensures only one pool exists per
    application, preventing resource conflicts and ensuring consistent
    connection management.

    Key contributions to system robustness:
    - Efficient connection reuse reduces connection overhead
    - Thread safety prevents race conditions in multi-threaded environments
    - Singleton pattern ensures consistent connection management across the app
    - Automatic connection health checks prevent use of stale connections
    - Proper resource cleanup prevents connection leaks
    - Reference counting ensures pool stays alive while clients are using it
    """
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MilvusConnectionPool, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, config: MilvusSettings = None):
        """
        Initialize the connection pool (if not already initialized).

        This method ensures the connection pool is properly initialized with the
        specified configuration. It uses thread-safe singleton initialization to
        prevent multiple pool instances and ensures all connections are created
        during startup for immediate availability.

        Args:
            config: MilvusSettings object containing connection configuration.
                   If None, default settings will be used.

        Note: This method is thread-safe and idempotent - multiple calls will
        not create duplicate pools or connections.
        """
        with self._lock:
            if self._initialized:
                if config is not None and config != self.config:
                    from exceptions import ConfigurationError
                    raise ConfigurationError(
                        "MilvusConnectionPool already initialized with a different configuration. "
                        "This could lead to inconsistent connection behavior."
                    )
                return
                
            self.config = config
            self._available_connections = queue.Queue()
            self._in_use_connections = set()
            self._initialized = True
            self._closed = False
            self._connection_count = 0
            self._reference_count = 0  # Track how many ConnectionManagers are using this pool
            
            # Initialize the pool
            self._initialize_pool()
            
            logger.info(f"Milvus connection pool initialized with size {self.config.connection.connection_pool_size}")
    
    def _initialize_pool(self):
        """
        Initialize the connection pool with the configured number of connections.

        This method creates the initial set of Milvus connections based on the
        configured pool size. By pre-creating connections during initialization,
        it ensures connections are immediately available when needed, reducing
        latency for the first operations and preventing connection creation
        bottlenecks under load.

        Raises:
            ConnectionInitializationError: If connection creation fails, ensuring
                the application fails fast rather than encountering issues during
                runtime operations.
        """
        try:
            pool_size = self.config.connection.connection_pool_size
            for i in range(pool_size):
                conn_alias = f"conn_{i}"
                self._create_connection(conn_alias)
                self._available_connections.put(conn_alias)
                self._connection_count += 1
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise ConnectionInitializationError(f"Failed to initialize connection pool: {e}")

    def _create_connection(self, alias: str):
        """
        Create a new connection with the given alias.

        This method establishes a new Milvus connection using the configured
        parameters. Each connection is given a unique alias for identification
        and management purposes. Proper connection creation is essential for
        the pool's functionality and contributes to system reliability.

        Args:
            alias: Unique identifier for this connection, used for tracking
                   and management within the pool.

        Note: This method handles all connection parameters (host, port, auth,
        timeout, etc.) from the configuration, ensuring consistent connection
        behavior across the application.
        """
        connections.connect(
            alias=alias,
            host=self.config.connection.host,
            port=self.config.connection.port,
            user=self.config.connection.user,
            password=self.config.connection.password,
            secure=self.config.connection.secure,
            timeout=self.config.connection.timeout
        )
        logger.debug(f"Created new Milvus connection: {alias}")

    def _is_connection_healthy(self, alias: str) -> bool:
        """
        Check if a connection is healthy and available for use.

        This method validates that a connection is still active and responsive.
        Connection health checks are crucial for preventing the use of stale or
        broken connections, which could lead to operation failures, timeouts,
        or inconsistent behavior in high-traffic scenarios.

        Args:
            alias: The connection alias to check

        Returns:
            bool: True if the connection is healthy, False otherwise

        Note: This method is lightweight and designed for frequent checking
        without significantly impacting performance.
        """
        try:
            return connections.has_connection(alias)
        except Exception:
            return False

    @contextmanager
    def get_connection(self, timeout: Optional[float] = None):
        """
        Get a connection from the pool as a context manager.

        This method provides thread-safe access to Milvus connections, implementing
        the context manager protocol for automatic resource management. It ensures
        connections are properly acquired and returned to the pool, preventing
        resource leaks and connection exhaustion in long-running applications.

        The method includes connection health validation to prevent the use of
        stale connections, which is essential for maintaining system reliability
        in production environments where connections may become invalid due to
        network issues or server restarts.

        Args:
            timeout: Maximum time to wait for an available connection (seconds).
                    If None, uses the timeout from configuration.

        Yields:
            str: Connection alias that can be used with pymilvus.connections

        Raises:
            ConnectionPoolExhaustedError: If no connections are available within
                the timeout period, indicating the system may be under excessive load
            ConnectionClosedError: If the pool has been closed, ensuring proper
                resource cleanup and preventing use of disposed resources

        Example:
            >>> with pool.get_connection() as conn_alias:
            ...     # Use conn_alias for Milvus operations
            ...     pass  # Connection automatically returned to pool
        """
        if self._closed:
            raise ConnectionClosedError("Connection pool is closed")
            
        if timeout is None:
            timeout = self.config.connection.timeout
            
        try:
            # Try to get a connection from the pool
            conn_alias = self._available_connections.get(timeout=timeout)
            
            # Check if connection is healthy
            if not self._is_connection_healthy(conn_alias):
                logger.warning(f"Stale connection {conn_alias} detected, attempting to reconnect.")
                try:
                    connections.disconnect(alias=conn_alias)
                    self._create_connection(conn_alias)
                except Exception as e:
                    logger.error(f"Failed to recreate connection {conn_alias}: {e}")
                    # Put it back and let another thread try
                    self._available_connections.put(conn_alias)
                    raise ConnectionError(f"Failed to restore connection {conn_alias}")

            with self._lock:
                self._in_use_connections.add(conn_alias)
                
            logger.info(f"Connection {conn_alias} acquired from pool. In-use connections: {len(self._in_use_connections)}, Available: {self._available_connections.qsize()}")
                
            try:
                # Yield the connection to the caller
                yield conn_alias
            finally:
                # Return the connection to the pool
                with self._lock:
                    if conn_alias in self._in_use_connections:
                        self._in_use_connections.remove(conn_alias)
                        
                if not self._closed:
                    # PRODUCTION FIX: Validate connection health before returning to pool
                    # This prevents unhealthy connections from accumulating in the pool,
                    # which would cause failures for the next operation that borrows them.
                    # Without this check, a connection that became stale or broken during
                    # an operation would be reused, leading to cascading failures.
                    if self._is_connection_healthy(conn_alias):
                        # Connection is healthy, return it to the pool
                        self._available_connections.put(conn_alias)
                        logger.info(f"Connection {conn_alias} returned to pool. In-use connections: {len(self._in_use_connections)}, Available: {self._available_connections.qsize()}")
                    else:
                        # Connection is unhealthy, attempt to recreate it
                        logger.warning(f"Connection {conn_alias} is unhealthy after use, recreating...")
                        try:
                            # Disconnect the stale connection
                            connections.disconnect(alias=conn_alias)
                            
                            # Create a new connection with the same alias
                            self._create_connection(conn_alias)
                            
                            # Return the new healthy connection to the pool
                            self._available_connections.put(conn_alias)
                            logger.info(f"Connection {conn_alias} recreated and returned to pool. In-use connections: {len(self._in_use_connections)}, Available: {self._available_connections.qsize()}")
                        except Exception as e:
                            # Failed to recreate connection - this is serious
                            logger.error(f"Failed to recreate connection {conn_alias}: {e}")
                            # Don't return the connection to the pool - pool size is now reduced by 1
                            # This should trigger monitoring alerts for degraded pool capacity
                            logger.error(
                                f"Connection pool capacity reduced to {self._available_connections.qsize()} "
                                f"available connections (target: {self.config.connection.connection_pool_size})"
                            )
                else:
                    # If pool is closed, actually close this connection
                    try:
                        connections.disconnect(alias=conn_alias)
                        logger.info(f"Connection {conn_alias} closed (pool is shutting down)")
                    except Exception:
                        pass
                        
        except queue.Empty:
            raise ConnectionPoolExhaustedError(
                f"No connections available in the pool within {timeout} seconds. "
                f"Consider increasing connection_pool_size (current: {self.config.connection.connection_pool_size})."
            )
    
    def acquire_reference(self):
        """
        Increment the reference count when a ConnectionManager starts using this pool.
        
        This method implements reference counting to ensure the pool stays alive
        as long as at least one ConnectionManager is using it. This is the
        enterprise-grade solution for managing shared resources across multiple clients.
        """
        with self._lock:
            self._reference_count += 1
            logger.debug(f"Pool reference acquired. Current reference count: {self._reference_count}")
    
    def release_reference(self):
        """
        Decrement the reference count when a ConnectionManager is done with this pool.
        
        This method decrements the reference count and only closes the pool when
        the count reaches zero (i.e., no more clients are using it). This ensures
        the pool stays alive for other clients while properly cleaning up when
        no longer needed.
        
        Returns:
            bool: True if the pool was actually closed, False if still in use
        """
        with self._lock:
            if self._reference_count > 0:
                self._reference_count -= 1
                logger.debug(f"Pool reference released. Current reference count: {self._reference_count}")
                
                # Only actually close the pool when no more references exist
                if self._reference_count == 0:
                    logger.info("No more references to connection pool, closing it")
                    self._close_pool_internal()
                    return True
            return False
    
    def _close_pool_internal(self):
        """
        Internal method to actually close the pool and its connections.
        
        This method contains the actual pool closing logic that was previously
        in the close() method. It's now only called when the reference count
        reaches zero, ensuring proper resource cleanup.
        """
        if self._closed:
            return
            
        self._closed = True
        
        # Close all available connections
        while not self._available_connections.empty():
            try:
                conn_alias = self._available_connections.get_nowait()
                connections.disconnect(alias=conn_alias)
                logger.debug(f"Closed connection: {conn_alias}")
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
        
        # Log warning about in-use connections
        if self._in_use_connections:
            logger.warning(
                f"{len(self._in_use_connections)} connections still in use during pool shutdown"
            )
            
        logger.info("Milvus connection pool closed")
        
        # Reset the singleton instance so a new pool can be created if needed
        with MilvusConnectionPool._lock:
            MilvusConnectionPool._instance = None
    
    def close(self):
        """
        Close all connections in the pool and release resources.

        This method ensures proper cleanup of all connections in the pool,
        preventing resource leaks and ensuring clean shutdown. It handles
        the case where connections may still be in use by logging warnings
        but proceeding with cleanup of available connections.

        The method is thread-safe and idempotent - it can be called multiple
        times safely without causing errors or resource issues. This is
        essential for graceful application shutdown and prevents connection
        leaks in long-running applications.

        Note: In-use connections will be logged as warnings but the pool
        will still be marked as closed, preventing new connection acquisitions.
        
        DEPRECATED: This method is kept for backward compatibility but should
        not be called directly. Use release_reference() instead for proper
        reference counting behavior.
        """
        logger.warning("close() called directly on connection pool - this bypasses reference counting")
        self._close_pool_internal()
            
    def __del__(self):
        """Ensure connections are closed when the pool is garbage collected"""
        try:
            self.close()
        except Exception:
            pass
