"""
Retry Module

This module provides retry logic with exponential backoff for handling
transient failures in search operations.
"""

import asyncio
import random
import logging
from typing import Callable, Any, TypeVar

from ..utils.config import RetryConfig

logger = logging.getLogger(__name__)

T = TypeVar('T')


def calculate_backoff_delay(
    attempt: int,
    config: RetryConfig
) -> float:
    """
    Calculate exponential backoff delay with optional jitter.
    
    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration
        
    Returns:
        Delay in seconds before next retry
    """
    # Calculate exponential backoff
    delay = min(
        config.initial_delay * (config.exponential_base ** attempt),
        config.max_delay
    )
    
    # Add jitter if enabled (randomize between 50% and 100% of calculated delay)
    if config.jitter:
        delay *= (0.5 + random.random() * 0.5)
    
    return delay


async def execute_with_retry(
    func: Callable[..., T],
    config: RetryConfig,
    *args,
    **kwargs
) -> T:
    """
    Execute function with retry logic and exponential backoff.
    
    This function wraps an async callable and retries it on failure according
    to the provided retry configuration. Only exceptions listed in
    retriable_exceptions will trigger retries.
    
    Args:
        func: Async function to execute
        config: Retry configuration
        *args: Positional arguments for function
        **kwargs: Keyword arguments for function
        
    Returns:
        Result of successful function execution
        
    Raises:
        Exception: The last exception if all retries are exhausted
    """
    last_exception = None
    
    for attempt in range(config.max_retries + 1):
        try:
            result = await func(*args, **kwargs)
            
            # Log successful retry if this wasn't the first attempt
            if attempt > 0:
                logger.info(f"Operation succeeded after {attempt} retries")
            
            return result
            
        except config.retriable_exceptions as e:
            last_exception = e
            
            # Check if we have retries left
            if attempt < config.max_retries:
                delay = calculate_backoff_delay(attempt, config)
                logger.warning(
                    f"Attempt {attempt + 1}/{config.max_retries + 1} failed: {str(e)}. "
                    f"Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"All {config.max_retries + 1} attempts failed. "
                    f"Last error: {str(e)}"
                )
                
        except Exception as e:
            # Non-retriable exception - fail immediately
            logger.error(f"Non-retriable exception encountered: {str(e)}")
            raise
    
    # All retries exhausted
    raise last_exception


class RetryContext:
    """
    Context manager for retry operations with statistics tracking.
    
    Usage:
        retry_ctx = RetryContext(config)
        async with retry_ctx:
            result = await retry_ctx.execute(some_async_function, arg1, arg2)
    """
    
    def __init__(self, config: RetryConfig):
        """
        Initialize retry context.
        
        Args:
            config: Retry configuration
        """
        self.config = config
        self.total_attempts = 0
        self.successful_attempts = 0
        self.failed_attempts = 0
        self.total_retry_time = 0.0
    
    async def __aenter__(self):
        """Enter context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and log statistics."""
        if self.total_attempts > 0:
            logger.info(
                f"Retry statistics - "
                f"total: {self.total_attempts}, "
                f"successful: {self.successful_attempts}, "
                f"failed: {self.failed_attempts}, "
                f"total_retry_time: {self.total_retry_time:.2f}s"
            )
        return False
    
    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with retry tracking.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result of function execution
        """
        import time
        start_time = time.time()
        
        try:
            self.total_attempts += 1
            result = await execute_with_retry(func, self.config, *args, **kwargs)
            self.successful_attempts += 1
            return result
        except Exception:
            self.failed_attempts += 1
            raise
        finally:
            self.total_retry_time += (time.time() - start_time)
    
    def get_stats(self) -> dict:
        """
        Get retry statistics.
        
        Returns:
            Dictionary with retry statistics
        """
        return {
            "total_attempts": self.total_attempts,
            "successful_attempts": self.successful_attempts,
            "failed_attempts": self.failed_attempts,
            "total_retry_time": round(self.total_retry_time, 2),
            "success_rate": (
                self.successful_attempts / self.total_attempts
                if self.total_attempts > 0 else 0.0
            )
        }

