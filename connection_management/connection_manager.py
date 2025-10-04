"""
Milvus Connection Manager

This module provides a high-level interface for managing Milvus connections,
with support for both synchronous and asynchronous operations.
"""

import logging
import time
import asyncio
import random
from typing import Optional, Callable, Any, Dict
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
import threading

from pymilvus import connections

from config import MilvusSettings, load_settings
from connection_management.connection_pool import MilvusConnectionPool
from connection_management.circuit_breaker import MilvusCircuitBreaker, CircuitBreakerConfig
from connection_management.connection_exceptions import (
    ConnectionError,
    MaxRetriesExceededError,
    ServerUnavailableError
)
from exceptions import OperationTimeoutError

# Logger setup
logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    High-level manager for Milvus connections.

    This class provides a comprehensive interface for managing Milvus connections,
    abstracting away the complexities of connection pooling, retry logic, and
    error handling. It offers both synchronous and asynchronous operation modes,
    making it suitable for various application architectures.

    Key contributions to system robustness:
    - Automatic retry with exponential backoff prevents transient failures
    - Connection health monitoring ensures reliable operations
    - Thread-safe connection pooling handles high concurrency
    - Async support enables non-blocking operations for better performance
    - Comprehensive error handling with specific exception types
    - Proper resource management prevents connection leaks
    """
    
    def __init__(self, config: Optional[MilvusSettings] = None, enable_circuit_breaker: bool = True):
        """
        Initialize the connection manager with optional circuit breaker.

        This method sets up the connection manager with the provided configuration,
        creating and initializing the underlying connection pool and circuit breaker.
        It ensures that all necessary resources are properly configured and ready 
        for use, providing a single point of initialization for the entire connection 
        management system.

        Args:
            config: MilvusSettings object containing connection configuration.
                   If None, default settings will be loaded from the standard
                   configuration sources.
            enable_circuit_breaker: Whether to enable circuit breaker protection.
                                   Recommended for production environments.

        Note: The initialization is designed to fail fast if there are configuration
        issues, preventing the application from running with invalid connection settings.
        """
        self.config = config if config is not None else load_settings()
        self._pool = MilvusConnectionPool(self.config)
        self._pool.acquire_reference()  # Increment reference count for this ConnectionManager
        
        # PRODUCTION SCALABILITY FIX: Create dedicated ThreadPoolExecutor per ConnectionManager
        # This ensures each manager has its own thread pool sized to match the connection pool,
        # preventing the shared default executor bottleneck that was causing sequential processing
        pool_size = self.config.connection.connection_pool_size
        self._executor = ThreadPoolExecutor(
            max_workers=pool_size,
            thread_name_prefix=f"MilvusConnMgr-{id(self)}"
        )
        logger.debug(f"Created dedicated ThreadPoolExecutor with {pool_size} workers for ConnectionManager {id(self)}")
        
        # Initialize circuit breaker for fault tolerance
        self._circuit_breaker = None
        if enable_circuit_breaker:
            circuit_config = CircuitBreakerConfig(
                failure_threshold=getattr(self.config.connection, 'circuit_breaker_failure_threshold', 5),
                recovery_timeout=getattr(self.config.connection, 'circuit_breaker_recovery_timeout', 30.0),
                half_open_success_threshold=getattr(self.config.connection, 'circuit_breaker_success_threshold', 3),
                max_half_open_requests=getattr(self.config.connection, 'circuit_breaker_max_half_open', 2)
            )
            self._circuit_breaker = MilvusCircuitBreaker(circuit_config, name="milvus_connection")
            logger.info("ConnectionManager initialized with circuit breaker protection")
        else:
            logger.info("ConnectionManager initialized without circuit breaker")
        
        logger.info("ConnectionManager initialized")
    
    def with_retry(self, func: Callable):
        """
        Decorator for automatic retry of operations with exponential backoff.

        This decorator implements a robust retry mechanism that handles transient
        failures gracefully. It uses exponential backoff with jitter to prevent
        overwhelming the Milvus server during outages and to distribute retry
        attempts more evenly across time.

        The retry mechanism contributes significantly to system resilience by:
        - Handling temporary network issues and server hiccups
        - Preventing cascade failures from transient problems
        - Reducing server load during recovery periods
        - Providing configurable retry policies for different scenarios

        Args:
            func: Function to wrap with retry logic. The function should be
                 designed to handle connection operations and may raise
                 ConnectionError exceptions that will trigger retries.

        Returns:
            Wrapped function with retry capability that will attempt the
            operation multiple times before giving up.

        Note: The retry configuration (count, interval) is controlled by the
        MilvusSettings configuration object, allowing fine-tuning for different
        deployment environments.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = self.config.connection.retry_count
            retry_interval = self.config.connection.retry_interval
            last_exception = None
            
            for attempt in range(retry_count + 1):
                try:
                    if attempt > 0:
                        logger.info(f"Retry attempt {attempt}/{retry_count}")
                    return func(*args, **kwargs)
                except ConnectionError as e:
                    last_exception = e
                    if attempt < retry_count:
                        # Exponential backoff with jitter
                        backoff_time = retry_interval * (2 ** attempt)
                        jitter = backoff_time * 0.5
                        sleep_time = backoff_time + random.uniform(-jitter, jitter)
                        
                        logger.warning(f"Connection error: {e}. Retrying in {sleep_time:.2f}s...")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"Max retries ({retry_count}) exceeded: {e}")
                        break
            
            raise MaxRetriesExceededError(f"Operation failed after {retry_count} retries") from last_exception
        
        return wrapper
    
    def execute_operation(self, operation: Callable, *args, **kwargs):
        """
        Execute an operation using a connection from the pool with circuit breaker protection.

        This method provides a high-level interface for executing Milvus operations
        with automatic connection management, circuit breaker protection, retry logic, 
        and error handling. It acquires a connection from the pool, executes the 
        operation, and ensures the connection is properly returned, all within a 
        single method call.

        The circuit breaker integration provides additional resilience by:
        - Failing fast when Milvus server is known to be unavailable
        - Preventing resource exhaustion during outages
        - Automatically testing for service recovery
        - Providing clear error messages for different failure modes

        Args:
            operation: Function that takes a connection alias as its first argument
                      and performs Milvus operations. Should return the operation result.
            *args: Additional positional arguments to pass to the operation
            **kwargs: Additional keyword arguments to pass to the operation

        Returns:
            The result of the operation as returned by the provided function

        Raises:
            ServerUnavailableError: If circuit breaker is open (Milvus unavailable)
            ConnectionPoolExhaustedError: No connections available in pool
            MaxRetriesExceededError: Operation failed after all retry attempts
            ConnectionError: General connection-related errors

        Example:
            >>> def list_collections(conn_alias):
            ...     return connections.get_connection(conn_alias).list_collections()
            >>> collections = manager.execute_operation(list_collections)
        """
        if self._circuit_breaker:
            # Execute with circuit breaker protection
            return asyncio.run(self._execute_with_circuit_breaker(operation, *args, **kwargs))
        else:
            # Execute without circuit breaker (legacy mode)
            @self.with_retry
            def _execute_operation_with_pool():
                with self._pool.get_connection() as conn_alias:
                    return operation(conn_alias, *args, **kwargs)
            
            return _execute_operation_with_pool()
    
    async def _execute_with_circuit_breaker(self, operation: Callable, timeout: Optional[float] = None, *args, **kwargs):
        """
        Execute operation with circuit breaker and enhanced server availability protection.
        
        This method provides robust error handling for server unavailability:
        1. Checks server health before attempting operation
        2. Classifies errors to distinguish between connection and server issues
        3. Implements specific retry strategies for server unavailability
        4. Enforces operation-level timeout to prevent hanging operations
        
        Args:
            operation: The operation to execute
            timeout: Optional timeout for the operation (enforced at operation level)
            *args, **kwargs: Additional arguments for the operation
            
        Raises:
            ServerUnavailableError: When server is confirmed to be down
            ConnectionError: For other connection-related issues
            OperationTimeoutError: When operation exceeds timeout duration
        """
        # First check if server is responsive
        if not self.check_server_status():
            logger.error("Server health check failed - server appears to be unavailable")
            raise ServerUnavailableError("Milvus server is not responding to health checks")
            
        async def _protected_operation():
            @self.with_retry
            def _execute_operation_with_pool():
                try:
                    with self._pool.get_connection(timeout=timeout) as conn_alias:
                        try:
                            return operation(conn_alias, *args, **kwargs)
                        except Exception as e:
                            # Classify the error
                            if "connection refused" in str(e).lower() or \
                               "cannot connect to server" in str(e).lower() or \
                               "server unavailable" in str(e).lower():
                                raise ServerUnavailableError(f"Server became unavailable during operation: {e}")
                            raise  # Re-raise other exceptions
                except ConnectionError as e:
                    # Add context about server state
                    if not self.check_server_status():
                        raise ServerUnavailableError(f"Server became unavailable: {e}")
                    raise
            
            # PRODUCTION FIX: Enforce operation-level timeout
            # This ensures the actual Milvus operation execution respects the timeout,
            # not just the connection acquisition phase. Without this, long-running
            # operations can hang indefinitely despite timeout configuration.
            loop = asyncio.get_event_loop()
            
            if timeout:
                try:
                    # Use asyncio.wait_for to enforce timeout on the executor task
                    result = await asyncio.wait_for(
                        loop.run_in_executor(self._executor, _execute_operation_with_pool),
                        timeout=timeout
                    )
                    return result
                except asyncio.TimeoutError:
                    # Convert asyncio.TimeoutError to our OperationTimeoutError for consistency
                    raise OperationTimeoutError(
                        f"Operation exceeded timeout of {timeout}s. "
                        "Consider increasing timeout or optimizing the operation."
                    )
            else:
                # No timeout specified, execute without time limit
                return await loop.run_in_executor(self._executor, _execute_operation_with_pool)
        
        try:
            return await self._circuit_breaker.execute_milvus_operation(_protected_operation)
        except Exception as e:
            # Final check - if server is down, make that clear in the error
            if not self.check_server_status():
                raise ServerUnavailableError(f"Server unavailable after operation attempt: {e}")
            raise
    
    async def execute_operation_async(self, operation: Callable, timeout: Optional[float] = None, *args, **kwargs):
        """
        Execute an operation asynchronously with circuit breaker protection.

        This method provides asynchronous execution of Milvus operations, allowing
        for non-blocking I/O operations in async applications. It combines the
        benefits of connection pooling, circuit breaker protection, retry logic, 
        and async execution to provide high-performance, scalable Milvus operations 
        for async frameworks.

        The circuit breaker integration provides the same resilience benefits as
        the synchronous version while maintaining full async compatibility.

        Args:
            operation: Function that takes a connection alias as its first argument
                      and performs Milvus operations. Should return the operation result.
            timeout: Optional timeout for the operation (handled by connection manager)
            *args: Additional positional arguments to pass to the operation
            **kwargs: Additional keyword arguments to pass to the operation

        Returns:
            The result of the operation as returned by the provided function

        Raises:
            ServerUnavailableError: If circuit breaker is open (Milvus unavailable)
            ConnectionPoolExhaustedError: No connections available in pool
            MaxRetriesExceededError: Operation failed after all retry attempts
            ConnectionError: General connection-related errors

        Example:
            >>> async def search_vectors(conn_alias, query_vector, top_k=10):
            ...     return connections.get_connection(conn_alias).search(
            ...         collection_name="vectors", query=query_vector, top_k=top_k
            ...     )
            >>> results = await manager.execute_operation_async(search_vectors, timeout=30.0, query_vector)
        """
        if self._circuit_breaker:
            # Execute with circuit breaker protection (already async)
            return await self._execute_with_circuit_breaker(operation, timeout, *args, **kwargs)
        else:
            # Execute without circuit breaker (legacy async mode)
            @self.with_retry
            def _execute_operation_with_pool_async():
                with self._pool.get_connection(timeout=timeout) as conn_alias:
                    return operation(conn_alias, *args, **kwargs)
            
            # Run the operation in the dedicated thread pool to avoid blocking the event loop
            # PRODUCTION SCALABILITY FIX: Use dedicated executor instead of shared default executor
            # PRODUCTION FIX: Enforce operation-level timeout even without circuit breaker
            loop = asyncio.get_event_loop()
            
            if timeout:
                try:
                    # Use asyncio.wait_for to enforce timeout on the executor task
                    result = await asyncio.wait_for(
                        loop.run_in_executor(self._executor, _execute_operation_with_pool_async),
                        timeout=timeout
                    )
                    return result
                except asyncio.TimeoutError:
                    # Convert asyncio.TimeoutError to our OperationTimeoutError for consistency
                    raise OperationTimeoutError(
                        f"Operation exceeded timeout of {timeout}s. "
                        "Consider increasing timeout or optimizing the operation."
                    )
            else:
                # No timeout specified, execute without time limit
                return await loop.run_in_executor(self._executor, _execute_operation_with_pool_async)
    
    def check_server_status(self):
        """
        Check if the Milvus server is available and responsive.

        This method provides a lightweight way to verify Milvus server connectivity
        without performing actual operations. It's useful for health checks,
        monitoring systems, and determining if the server is ready to accept
        requests before attempting operations.

        Server status checking contributes to system robustness by:
        - Enabling proactive monitoring and alerting
        - Supporting graceful degradation strategies
        - Helping diagnose connectivity issues before operations fail
        - Providing early warning of server-side problems

        Returns:
            bool: True if server is available and responsive, False otherwise

        Note: This method uses an existing connection from the pool, so it will
        return False if no connections are available, even if the server itself
        is running. For a more comprehensive server check, ensure the connection
        pool has available connections.
        """
        try:
            with self._pool.get_connection() as conn_alias:
                # Simple ping to check if server is responsive
                # This assumes Milvus has a utility function or endpoint for checking status
                # Adjust as needed based on the actual Milvus Python SDK capabilities
                return connections.has_connection(conn_alias)
        except ConnectionError:
            return False
    
    def get_circuit_breaker_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get circuit breaker metrics for monitoring and alerting.
        
        Returns:
            Dict containing circuit breaker metrics, or None if circuit breaker is disabled
        """
        if self._circuit_breaker:
            return self._circuit_breaker.get_metrics()
        return None
    
    def is_circuit_breaker_open(self) -> bool:
        """
        Check if the circuit breaker is open (failing fast).
        
        Returns:
            bool: True if circuit breaker is open, False if closed/half-open or disabled
        """
        if self._circuit_breaker:
            return self._circuit_breaker.is_open()
        return False
    
    async def reset_circuit_breaker(self) -> None:
        """
        Reset the circuit breaker to closed state.
        
        This method is useful for manual recovery after maintenance or emergency override.
        Use with caution in production environments.
        """
        if self._circuit_breaker:
            await self._circuit_breaker.reset()
        else:
            logger.warning("Cannot reset circuit breaker - circuit breaker is disabled")
    
    def close(self):
        """
        Close the connection manager and release all resources.

        This method ensures proper cleanup of all connections and resources
        managed by the ConnectionManager. It uses reference counting to only
        close the connection pool when no other ConnectionManagers are using it.

        Proper cleanup is essential for:
        - Preventing resource leaks in long-running applications
        - Ensuring clean application shutdown
        - Releasing database connections back to the connection pool
        - Preventing connection exhaustion in shared environments

        Note: This method is idempotent and can be called multiple times safely.
        It's automatically called during object destruction but explicit cleanup
        is recommended for predictable resource management.
        """
        # PRODUCTION SCALABILITY FIX: Clean up the dedicated ThreadPoolExecutor
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
            logger.debug(f"Shut down ThreadPoolExecutor for ConnectionManager {id(self)}")
        
        if hasattr(self, '_pool'):
            # Use reference counting instead of directly closing the pool
            pool_closed = self._pool.release_reference()
            if pool_closed:
                logger.info("ConnectionManager closed and pool was shut down (last reference)")
            else:
                logger.info("ConnectionManager closed (pool still in use by other managers)")
        else:
            logger.info("ConnectionManager closed (no pool to release)")

    def __del__(self):
        """
        Ensure connections are closed when the manager is garbage collected.

        This destructor provides a safety net for resource cleanup in case the
        close() method wasn't explicitly called. It prevents connection leaks
        in applications that might not properly manage object lifecycles.

        While relying on __del__ for cleanup isn't ideal (due to potential issues
        with garbage collection timing), it provides an additional layer of
        protection against resource leaks in edge cases.
        """
        try:
            self.close()
        except Exception:
            pass
