"""
Resilience Module

This module provides fault tolerance and resilience patterns including
retry logic with exponential backoff and circuit breaker pattern for
preventing cascading failures.
"""

import time
import asyncio
import random
from typing import Callable, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 3
    initial_delay: float = 0.5
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    retriable_exceptions: Tuple = (ConnectionError, TimeoutError)


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for fault tolerance.
    
    Prevents cascading failures by temporarily blocking requests when
    error threshold is exceeded. Implements three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests are blocked
    - HALF_OPEN: Testing if service has recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        """
        Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before attempting recovery
            expected_exception: Exception type that triggers the circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self._state = CircuitState.CLOSED
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> str:
        """Get current circuit state."""
        return self._state.value
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                else:
                    from ...core.exceptions import SearchError
                    raise SearchError(
                        f"Circuit breaker is OPEN - service unavailable "
                        f"(failures: {self.failure_count})"
                    )
        
        try:
            result = await func(*args, **kwargs)
            
            async with self._lock:
                if self._state == CircuitState.HALF_OPEN:
                    self._state = CircuitState.CLOSED
                    self.failure_count = 0
            
            return result
            
        except self.expected_exception as e:
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
            
            raise
    
    async def reset(self) -> None:
        """Manually reset the circuit breaker."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None
    
    def get_status(self) -> dict:
        """
        Get circuit breaker status.
        
        Returns:
            Dictionary with circuit breaker state and metrics
        """
        return {
            "state": self._state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time
        }


class RetryHandler:
    """
    Handles retry logic with exponential backoff and jitter.
    
    Features:
    - Configurable retry attempts
    - Exponential backoff
    - Optional jitter to prevent thundering herd
    - Selective exception retry
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry handler.
        
        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay with optional jitter.
        
        Args:
            attempt: Current retry attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        delay = min(
            self.config.initial_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )
        
        if self.config.jitter:
            delay *= (0.5 + random.random())
        
        return delay
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with automatic retry logic.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries exhausted or non-retriable exception
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return await func(*args, **kwargs)
                
            except self.config.retriable_exceptions as e:
                last_exception = e
                
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    await asyncio.sleep(delay)
                    
            except Exception as e:
                # Non-retriable exception - fail immediately
                raise
        
        # All retries exhausted
        raise last_exception


class ResilienceManager:
    """
    Manages both circuit breaker and retry logic for comprehensive resilience.
    
    Combines circuit breaker pattern with retry logic to provide
    robust fault tolerance for search operations.
    """
    
    def __init__(
        self,
        enable_circuit_breaker: bool = True,
        enable_retry: bool = True,
        retry_config: Optional[RetryConfig] = None,
        circuit_failure_threshold: int = 5,
        circuit_recovery_timeout: float = 60.0
    ):
        """
        Initialize resilience manager.
        
        Args:
            enable_circuit_breaker: Enable circuit breaker pattern
            enable_retry: Enable retry logic
            retry_config: Retry configuration
            circuit_failure_threshold: Circuit breaker failure threshold
            circuit_recovery_timeout: Circuit breaker recovery timeout
        """
        self.circuit_breaker = None
        if enable_circuit_breaker:
            from ...core.exceptions import SearchError
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=circuit_failure_threshold,
                recovery_timeout=circuit_recovery_timeout,
                expected_exception=SearchError
            )
        
        self.retry_handler = None
        if enable_retry:
            self.retry_handler = RetryHandler(retry_config)
    
    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with full resilience protection.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        if self.circuit_breaker and self.retry_handler:
            # Both circuit breaker and retry enabled
            return await self.circuit_breaker.call(
                self.retry_handler.execute_with_retry,
                func,
                *args,
                **kwargs
            )
        elif self.circuit_breaker:
            # Only circuit breaker
            return await self.circuit_breaker.call(func, *args, **kwargs)
        elif self.retry_handler:
            # Only retry
            return await self.retry_handler.execute_with_retry(func, *args, **kwargs)
        else:
            # No resilience - direct execution
            return await func(*args, **kwargs)
    
    def get_status(self) -> dict:
        """
        Get resilience manager status.
        
        Returns:
            Dictionary with circuit breaker and retry status
        """
        status = {
            "circuit_breaker_enabled": self.circuit_breaker is not None,
            "retry_enabled": self.retry_handler is not None
        }
        
        if self.circuit_breaker:
            status["circuit_breaker"] = self.circuit_breaker.get_status()
        
        if self.retry_handler:
            status["retry_config"] = {
                "max_retries": self.retry_handler.config.max_retries,
                "initial_delay": self.retry_handler.config.initial_delay,
                "max_delay": self.retry_handler.config.max_delay
            }
        
        return status

