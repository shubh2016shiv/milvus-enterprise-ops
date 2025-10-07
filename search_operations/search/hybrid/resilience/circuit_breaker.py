"""
Circuit Breaker Module

This module implements the circuit breaker pattern for fault tolerance,
preventing cascading failures by temporarily blocking operations to failing services.
"""

import time
import asyncio
import logging
from typing import Callable, Any, Optional

from ....core.exceptions import HybridSearchError

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for fault tolerance.
    
    The circuit breaker acts as a proxy for operations that might fail.
    It monitors failures and prevents further attempts when a threshold is reached,
    allowing the system to recover before trying again.
    
    States:
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
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type that triggers circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half_open
        self._lock = asyncio.Lock()
        
        logger.info(
            f"CircuitBreaker initialized - "
            f"threshold: {failure_threshold}, "
            f"recovery_timeout: {recovery_timeout}s"
        )
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Result of function execution
            
        Raises:
            HybridSearchError: If circuit is open
            Exception: Original exception from function if circuit allows
        """
        async with self._lock:
            if self.state == "open":
                # Check if recovery timeout has elapsed
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = "half_open"
                    logger.info("Circuit breaker entering half-open state")
                else:
                    raise HybridSearchError(
                        f"Circuit breaker is OPEN - too many failures "
                        f"({self.failure_count}/{self.failure_threshold}). "
                        f"Service unavailable."
                    )
        
        try:
            # Execute the function
            result = await func(*args, **kwargs)
            
            # Success - update state
            async with self._lock:
                if self.state == "half_open":
                    self.state = "closed"
                    self.failure_count = 0
                    logger.info("Circuit breaker closed - service recovered")
            
            return result
            
        except self.expected_exception as e:
            # Failure - update failure count and state
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    logger.error(
                        f"Circuit breaker OPENED after {self.failure_count} failures"
                    )
                else:
                    logger.warning(
                        f"Circuit breaker failure {self.failure_count}/"
                        f"{self.failure_threshold}"
                    )
            
            raise
    
    def get_state(self) -> str:
        """
        Get current circuit breaker state.
        
        Returns:
            Current state ('closed', 'open', or 'half_open')
        """
        return self.state
    
    def get_failure_count(self) -> int:
        """
        Get current failure count.
        
        Returns:
            Number of consecutive failures
        """
        return self.failure_count
    
    async def reset(self) -> None:
        """
        Manually reset the circuit breaker to closed state.
        
        This method is useful for testing or manual recovery operations.
        """
        async with self._lock:
            self.state = "closed"
            self.failure_count = 0
            self.last_failure_time = None
        
        logger.info("Circuit breaker manually reset to closed state")
    
    def get_stats(self) -> dict:
        """
        Get circuit breaker statistics.
        
        Returns:
            Dictionary with current state and statistics
        """
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "recovery_timeout": self.recovery_timeout
        }

