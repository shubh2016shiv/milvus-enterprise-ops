"""
Milvus Connection Circuit Breaker

This module provides a production-ready circuit breaker specifically designed for
Milvus connection management, capable of handling millions of concurrent users
with proper async support and integration with the connection pool.

The circuit breaker prevents cascading failures when Milvus server is unavailable
or experiencing issues, providing fast failure and automatic recovery mechanisms.
"""

import time
import asyncio
import logging
from enum import Enum
from typing import Callable, Any, TypeVar, Awaitable, Optional, Dict
from contextlib import asynccontextmanager
from dataclasses import dataclass

from .connection_exceptions import (
    ConnectionError,
    ServerUnavailableError,
    ConnectionTimeoutError,
    MaxRetriesExceededError
)

logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar("T")


class CircuitState(str, Enum):
    """
    Circuit breaker states following industry standards.
    
    These states provide clear separation of concerns for handling different
    failure scenarios in high-traffic Milvus environments:
    - CLOSED: Normal operation, all requests pass through to Milvus
    - OPEN: Circuit is open, requests fail fast without hitting Milvus
    - HALF_OPEN: Testing if Milvus service has recovered
    """
    CLOSED = "closed"        # Normal operation, requests pass through
    OPEN = "open"           # Circuit is open, requests fail fast
    HALF_OPEN = "half-open" # Testing if Milvus service is healthy again


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for Milvus circuit breaker optimized for high-scale environments.
    
    These defaults are tuned for Milvus workloads handling millions of users:
    - Quick failure detection to prevent resource exhaustion
    - Reasonable recovery times to balance availability and stability
    - Conservative half-open testing to avoid overwhelming recovering servers
    """
    failure_threshold: int = 5              # Failures before opening circuit
    recovery_timeout: float = 30.0          # Seconds before attempting recovery
    half_open_success_threshold: int = 3    # Successes needed to close circuit
    max_half_open_requests: int = 2         # Concurrent test requests in half-open
    
    # Milvus-specific exception handling
    milvus_exclude_exceptions: tuple = (
        # These exceptions should not trigger circuit breaker
        # as they represent client-side issues, not server failures
        ValueError,           # Invalid parameters
        TypeError,           # Type mismatches
        AttributeError,      # Programming errors
    )


class MilvusCircuitBreaker:
    """
    Production-ready circuit breaker for Milvus connections.
    
    This implementation is specifically designed for high-scale Milvus deployments
    with millions of concurrent users. It provides:
    
    - Thread-safe async operations with proper concurrency control
    - Milvus-specific failure detection and recovery patterns
    - Integration with connection pool metrics and monitoring
    - Fast failure paths to prevent resource exhaustion
    - Automatic recovery with controlled testing
    
    The circuit breaker works in conjunction with the connection pool to provide
    a comprehensive fault tolerance strategy:
    1. Connection pool handles individual connection failures
    2. Circuit breaker handles systemic Milvus server failures
    3. Together they provide multi-layer resilience
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None, name: str = "milvus"):
        """
        Initialize the Milvus circuit breaker.
        
        Args:
            config: Circuit breaker configuration. If None, uses production defaults
            name: Circuit breaker name for logging and metrics identification
        """
        self.config = config or CircuitBreakerConfig()
        self.name = name
        
        # Validate configuration for production use
        self._validate_config()
        
        # State management with proper async synchronization
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._lock = asyncio.Lock()
        
        # Half-open state management for controlled recovery testing
        self._half_open_requests = 0
        self._half_open_lock = asyncio.Lock()
        
        # Metrics for monitoring and alerting
        self._total_requests = 0
        self._total_failures = 0
        self._total_fast_failures = 0  # Requests failed due to open circuit
        self._state_change_count = 0
        self._last_state_change = time.monotonic()
        
        logger.info(
            f"MilvusCircuitBreaker '{self.name}' initialized: "
            f"failure_threshold={self.config.failure_threshold}, "
            f"recovery_timeout={self.config.recovery_timeout}s, "
            f"success_threshold={self.config.half_open_success_threshold}"
        )
    
    def _validate_config(self):
        """Validate configuration parameters for production safety."""
        if self.config.failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if self.config.recovery_timeout < 0:
            raise ValueError("recovery_timeout cannot be negative")
        if self.config.half_open_success_threshold < 1:
            raise ValueError("half_open_success_threshold must be at least 1")
        if self.config.max_half_open_requests < 1:
            raise ValueError("max_half_open_requests must be at least 1")
    
    async def execute_milvus_operation(
        self, 
        operation: Callable[..., T], 
        *args: Any, 
        **kwargs: Any
    ) -> T:
        """
        Execute a Milvus operation with circuit breaker protection.
        
        This method provides comprehensive protection for Milvus operations:
        - Fast failure when Milvus is known to be unavailable
        - Automatic retry and recovery testing
        - Proper exception handling and classification
        - Metrics collection for monitoring
        
        Args:
            operation: Milvus operation function (sync or async)
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            T: Operation result
            
        Raises:
            ServerUnavailableError: If circuit is open (Milvus unavailable)
            ConnectionError: If operation fails for connection reasons
            Exception: Other operation-specific exceptions
        """
        # Track total requests for metrics
        self._total_requests += 1
        
        # Use context manager for proper half-open request tracking
        async with self._execute_context():
            try:
                # Execute the Milvus operation
                result = operation(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
                
                # Handle successful operation
                await self._on_success()
                return result
                
            except Exception as e:
                # Classify the exception and handle appropriately
                if self._should_count_as_failure(e):
                    await self._on_failure()
                    self._total_failures += 1
                raise
    
    @asynccontextmanager
    async def _execute_context(self):
        """
        Context manager for proper request lifecycle management.
        
        This ensures that:
        - Circuit state is checked before operation execution
        - Half-open requests are properly tracked and limited
        - Resources are cleaned up even if exceptions occur
        - Metrics are accurately maintained
        """
        # Check if we can proceed with the operation
        await self._check_circuit_state()
        
        # Track half-open requests for controlled recovery testing
        was_half_open = self._state == CircuitState.HALF_OPEN
        if was_half_open:
            async with self._half_open_lock:
                self._half_open_requests += 1
                logger.debug(
                    f"Circuit '{self.name}': Half-open request started. "
                    f"Active requests: {self._half_open_requests}/{self.config.max_half_open_requests}"
                )
        
        try:
            yield
        finally:
            # Always decrement half-open request counter
            if was_half_open:
                async with self._half_open_lock:
                    self._half_open_requests = max(0, self._half_open_requests - 1)
                    logger.debug(
                        f"Circuit '{self.name}': Half-open request completed. "
                        f"Active requests: {self._half_open_requests}"
                    )
    
    async def _check_circuit_state(self) -> None:
        """
        Check circuit state and determine if operation can proceed.
        
        This method implements the core circuit breaker logic:
        - CLOSED: Allow all operations
        - OPEN: Check if recovery timeout has elapsed, transition to HALF_OPEN if so
        - HALF_OPEN: Limit concurrent operations for controlled testing
        
        Raises:
            ServerUnavailableError: If circuit is open or half-open with max requests
        """
        async with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                current_time = time.monotonic()
                time_since_failure = current_time - self._last_failure_time
                
                if time_since_failure >= self.config.recovery_timeout:
                    logger.info(
                        f"Circuit '{self.name}': Recovery timeout elapsed "
                        f"({time_since_failure:.1f}s), transitioning to half-open"
                    )
                    await self._transition_to_half_open()
                else:
                    self._total_fast_failures += 1
                    remaining_time = self.config.recovery_timeout - time_since_failure
                    raise ServerUnavailableError(
                        f"Milvus service unavailable (circuit open). "
                        f"Retry in {remaining_time:.1f} seconds."
                    )
            
            elif self._state == CircuitState.HALF_OPEN:
                # Limit concurrent requests in half-open state
                if self._half_open_requests >= self.config.max_half_open_requests:
                    self._total_fast_failures += 1
                    raise ServerUnavailableError(
                        "Milvus service recovery testing in progress. "
                        "Please retry in a moment."
                    )
    
    async def _transition_to_half_open(self):
        """Transition circuit to half-open state for recovery testing."""
        self._state = CircuitState.HALF_OPEN
        self._success_count = 0
        self._half_open_requests = 0
        self._state_change_count += 1
        self._last_state_change = time.monotonic()
    
    def _should_count_as_failure(self, exception: Exception) -> bool:
        """
        Determine if an exception should count as a circuit breaker failure.
        
        This method implements Milvus-specific failure classification:
        - Connection-related errors count as failures
        - Server-side errors count as failures  
        - Client-side errors (bad parameters, etc.) do not count as failures
        
        Args:
            exception: Exception to classify
            
        Returns:
            bool: True if exception should trigger circuit breaker
        """
        # Don't count excluded exceptions (client-side errors)
        if isinstance(exception, self.config.milvus_exclude_exceptions):
            return False
        
        # Count Milvus connection and server errors as failures
        if isinstance(exception, (
            ConnectionError,
            ConnectionTimeoutError,
            ServerUnavailableError,
            MaxRetriesExceededError,
            # Add other Milvus server-side exceptions as needed
        )):
            return True
        
        # For unknown exceptions, be conservative and count as failure
        # This prevents the circuit from staying closed during unexpected issues
        logger.warning(
            f"Circuit '{self.name}': Unknown exception type {type(exception).__name__}, "
            f"counting as failure for safety: {exception}"
        )
        return True
    
    async def _on_success(self) -> None:
        """
        Handle successful Milvus operation.
        
        Success handling varies by circuit state:
        - CLOSED: Reset failure count to maintain healthy state
        - HALF_OPEN: Count towards recovery threshold, close if threshold met
        """
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                logger.debug(
                    f"Circuit '{self.name}': Half-open success {self._success_count}/"
                    f"{self.config.half_open_success_threshold}"
                )
                
                # Check if we've reached the success threshold for recovery
                if self._success_count >= self.config.half_open_success_threshold:
                    logger.info(
                        f"Circuit '{self.name}': Recovery successful, closing circuit "
                        f"after {self._success_count} successful operations"
                    )
                    await self._transition_to_closed()
                    
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success in closed state
                if self._failure_count > 0:
                    logger.debug(f"Circuit '{self.name}': Resetting failure count after success")
                    self._failure_count = 0
    
    async def _transition_to_closed(self):
        """Transition circuit to closed state after successful recovery."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._state_change_count += 1
        self._last_state_change = time.monotonic()
    
    async def _on_failure(self) -> None:
        """
        Handle failed Milvus operation.
        
        Failure handling varies by circuit state:
        - CLOSED: Count failures, open circuit if threshold exceeded
        - HALF_OPEN: Immediately open circuit (recovery failed)
        """
        async with self._lock:
            # Record failure time for recovery timeout calculation
            self._last_failure_time = time.monotonic()
            
            if self._state == CircuitState.CLOSED:
                self._failure_count += 1
                logger.debug(
                    f"Circuit '{self.name}': Failure {self._failure_count}/"
                    f"{self.config.failure_threshold}"
                )
                
                # Check if failure threshold exceeded
                if self._failure_count >= self.config.failure_threshold:
                    logger.warning(
                        f"Circuit '{self.name}': Failure threshold exceeded "
                        f"({self._failure_count} failures), opening circuit"
                    )
                    await self._transition_to_open()
                    
            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately opens the circuit
                logger.warning(
                    f"Circuit '{self.name}': Recovery attempt failed, reopening circuit"
                )
                await self._transition_to_open()
    
    async def _transition_to_open(self):
        """Transition circuit to open state due to failures."""
        self._state = CircuitState.OPEN
        self._success_count = 0
        self._half_open_requests = 0
        self._state_change_count += 1
        self._last_state_change = time.monotonic()
    
    def is_open(self) -> bool:
        """Check if the circuit is open (failing fast)."""
        return self._state == CircuitState.OPEN
    
    def is_half_open(self) -> bool:
        """Check if the circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN
    
    def is_closed(self) -> bool:
        """Check if the circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED
    
    def get_state(self) -> str:
        """Get the current circuit state as string."""
        return self._state.value
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive circuit breaker metrics for monitoring.
        
        These metrics are essential for:
        - Production monitoring and alerting
        - Capacity planning and optimization
        - Troubleshooting and root cause analysis
        - SLA tracking and reporting
        
        Returns:
            Dict containing all circuit breaker metrics
        """
        current_time = time.monotonic()
        
        # Calculate time since last failure
        time_since_last_failure = None
        if self._last_failure_time > 0:
            time_since_last_failure = current_time - self._last_failure_time
        
        # Calculate time since last state change
        time_since_state_change = current_time - self._last_state_change
        
        # Calculate success rate
        success_rate = 0.0
        if self._total_requests > 0:
            successful_requests = self._total_requests - self._total_failures
            success_rate = (successful_requests / self._total_requests) * 100
        
        return {
            "name": self.name,
            "state": self._state.value,
            "configuration": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "half_open_success_threshold": self.config.half_open_success_threshold,
                "max_half_open_requests": self.config.max_half_open_requests,
            },
            "counters": {
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "total_requests": self._total_requests,
                "total_failures": self._total_failures,
                "total_fast_failures": self._total_fast_failures,
                "state_change_count": self._state_change_count,
            },
            "timing": {
                "last_failure_time": self._last_failure_time,
                "time_since_last_failure": time_since_last_failure,
                "time_since_state_change": time_since_state_change,
            },
            "current_state": {
                "half_open_requests": self._half_open_requests,
                "success_rate_percent": round(success_rate, 2),
            }
        }
    
    async def reset(self) -> None:
        """
        Reset circuit breaker to closed state.
        
        This method is useful for:
        - Manual recovery after maintenance
        - Testing and development scenarios
        - Emergency override situations
        
        Note: Use with caution in production environments.
        """
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_requests = 0
            self._last_failure_time = 0.0
            self._state_change_count += 1
            self._last_state_change = time.monotonic()
            
            logger.warning(f"Circuit '{self.name}': Manually reset to closed state")
    
    def __str__(self) -> str:
        """String representation for logging and debugging."""
        return f"MilvusCircuitBreaker(name='{self.name}', state={self._state.value})"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"MilvusCircuitBreaker(name='{self.name}', state={self._state.value}, "
            f"failures={self._failure_count}/{self.config.failure_threshold}, "
            f"requests={self._total_requests})"
        )
