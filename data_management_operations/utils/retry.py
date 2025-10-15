"""
Retry Utilities for Data Management Operations

Provides retry logic for transient failures that are not handled
by the ConnectionManager. This includes application-level transient
conditions like temporary schema unavailability or collection locks.
"""

import asyncio
import logging
from typing import Callable, Any, TypeVar, Awaitable
from functools import wraps

from ..data_ops_exceptions import TransientOperationError
from ..data_ops_config import DataOperationConfig

logger = logging.getLogger(__name__)

T = TypeVar('T')


async def retry_on_transient_error(
    operation: Callable[..., Awaitable[T]],
    config: DataOperationConfig,
    operation_name: str,
    *args,
    **kwargs
) -> T:
    """
    Retry an async operation on transient errors.
    
    This function implements retry logic specifically for transient errors
    that are not handled by ConnectionManager (e.g., temporary schema
    unavailability, collection temporarily locked). It does NOT retry
    permanent failures or errors that ConnectionManager already handles.
    
    The retry mechanism uses linear backoff, increasing the delay between
    retries with each attempt to avoid overwhelming the system during
    transient issues.
    
    Args:
        operation: Async function to execute with retry logic
        config: Data operation configuration containing retry settings
        operation_name: Human-readable name for logging
        *args: Positional arguments to pass to operation
        **kwargs: Keyword arguments to pass to operation
    
    Returns:
        Result of the operation if successful
    
    Raises:
        TransientOperationError: If all retry attempts are exhausted
        Any other exception: Propagated immediately without retry
    
    Example:
        ```python
        async def risky_operation(data):
            # Operation that might fail transiently
            ...
        
        result = await retry_on_transient_error(
            risky_operation,
            config,
            "insert_batch",
            data=my_data
        )
        ```
    """
    if not config.retry_transient_errors:
        # Retry disabled, execute once
        return await operation(*args, **kwargs)
    
    last_error: Optional[TransientOperationError] = None
    
    for attempt in range(config.max_transient_retries):
        try:
            return await operation(*args, **kwargs)
            
        except TransientOperationError as e:
            last_error = e
            
            if attempt < config.max_transient_retries - 1:
                # Calculate delay with linear backoff
                delay = config.transient_retry_delay * (attempt + 1)
                
                logger.debug(
                    f"Transient error in {operation_name} (attempt {attempt + 1}/"
                    f"{config.max_transient_retries}): {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                await asyncio.sleep(delay)
            else:
                logger.warning(
                    f"Transient error in {operation_name} persisted after "
                    f"{config.max_transient_retries} attempts: {e}"
                )
            
            continue
            
        except Exception:
            # Non-transient error, propagate immediately
            raise
    
    # All retries exhausted
    raise last_error


def with_transient_retry(operation_name: str):
    """
    Decorator for adding transient error retry logic to async methods.
    
    This decorator wraps an async method to automatically retry on
    TransientOperationError. The method must have access to self._config
    which should be a DataOperationConfig instance.
    
    Args:
        operation_name: Human-readable operation name for logging
    
    Example:
        ```python
        class DataManager:
            def __init__(self, config):
                self._config = config
            
            @with_transient_retry("insert_batch")
            async def _insert_batch(self, documents):
                # Implementation that might raise TransientOperationError
                ...
        ```
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(self, *args, **kwargs) -> T:
            # Extract config from self
            config = getattr(self, '_config', None)
            
            if config is None:
                # No config available, execute without retry
                logger.warning(
                    f"{func.__name__} has no _config attribute, "
                    "executing without transient retry"
                )
                return await func(self, *args, **kwargs)
            
            return await retry_on_transient_error(
                lambda: func(self, *args, **kwargs),
                config,
                operation_name
            )
        
        return wrapper
    return decorator


def is_transient_milvus_error(exception: Exception) -> bool:
    """
    Determine if a Milvus exception represents a transient condition.
    
    This function checks if an exception from PyMilvus represents a
    transient condition that warrants a retry. This includes conditions
    like temporary schema unavailability or collection locks, but NOT
    network errors (handled by ConnectionManager).
    
    Args:
        exception: Exception to check
    
    Returns:
        True if the exception represents a transient condition
    
    Note:
        This function examines exception messages as PyMilvus doesn't
        provide granular exception types for all transient conditions.
    """
    if not isinstance(exception, Exception):
        return False
    
    # Checklist of transient error patterns in Milvus
    transient_patterns = [
        "schema not ready",
        "collection is being loaded",
        "collection is loading",
        "temporary unavailable",
        "temporarily unavailable",
        "rate limit exceeded",
        "too many requests"
    ]
    
    error_message = str(exception).lower()
    
    return any(pattern in error_message for pattern in transient_patterns)

