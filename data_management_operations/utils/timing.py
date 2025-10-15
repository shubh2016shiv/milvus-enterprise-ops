"""
Performance Timing Utilities

Provides utilities for measuring and tracking the performance of operations.
"""

import time
import logging
from typing import Optional, Dict, List, Any
from contextlib import asynccontextmanager
from datetime import datetime
from pydantic import BaseModel, Field
import statistics

logger = logging.getLogger(__name__)


class TimingResult(BaseModel):
    """
    Result of a timed operation.
    
    Attributes:
        operation_name: Name of the operation that was timed
        execution_time: Time taken to execute the operation in seconds
        timestamp: When the operation was executed
        success: Whether the operation completed successfully
        metadata: Additional metadata about the operation
    """
    operation_name: str
    execution_time: float = Field(description="Execution time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class BatchTimingResult(BaseModel):
    """
    Aggregated timing statistics for multiple operations of the same type.
    
    Attributes:
        operation_name: Name of the operation type
        total_operations: Total number of operations
        successful_operations: Number of successful operations
        failed_operations: Number of failed operations
        average_execution_time: Average execution time in seconds
        median_execution_time: Median execution time in seconds
        min_execution_time: Minimum execution time in seconds
        max_execution_time: Maximum execution time in seconds
        p95_execution_time: 95th percentile execution time in seconds
    """
    operation_name: str
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    average_execution_time: float = 0.0
    median_execution_time: float = 0.0
    min_execution_time: float = 0.0
    max_execution_time: float = 0.0
    p95_execution_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.total_operations == 0:
            return 0.0
        return (self.successful_operations / self.total_operations) * 100.0


class PerformanceTimer:
    """
    Context manager for timing operations and collecting performance statistics.
    """
    
    def __init__(self, enable_logging: bool = True):
        """
        Initialize the performance timer.
        
        Args:
            enable_logging: Whether to log timing results
        """
        self._enable_logging = enable_logging
        self._timing_history: List[TimingResult] = []
    
    @asynccontextmanager
    async def time_operation(
        self,
        operation_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Async context manager for timing an operation.
        
        Args:
            operation_name: Name of the operation being timed
            metadata: Additional metadata to store with the timing result
            
        Yields:
            TimingResult object that will be populated with timing data
        """
        start_time = time.time()
        result = TimingResult(
            operation_name=operation_name,
            execution_time=0.0,
            metadata=metadata or {}
        )
        
        try:
            yield result
            result.success = True
        except Exception as e:
            result.success = False
            raise
        finally:
            result.execution_time = time.time() - start_time
            
            # Store result
            self._timing_history.append(result)
            
            # Log if enabled
            if self._enable_logging:
                status = "succeeded" if result.success else "failed"
                logger.debug(
                    f"Operation '{operation_name}' {status} in {result.execution_time*1000:.2f}ms"
                )
    
    def get_timing_history(self) -> List[TimingResult]:
        """Get the complete timing history."""
        return self._timing_history.copy()
    
    def get_operation_stats(self, operation_name: str) -> Optional[BatchTimingResult]:
        """
        Get aggregated statistics for a specific operation type.
        
        Args:
            operation_name: Name of the operation to get stats for
            
        Returns:
            BatchTimingResult with statistics, or None if no operations found
        """
        # Filter timing results for this operation
        operation_results = [r for r in self._timing_history if r.operation_name == operation_name]
        
        if not operation_results:
            return None
        
        # Calculate statistics
        execution_times = [r.execution_time for r in operation_results]
        successful = [r for r in operation_results if r.success]
        failed = [r for r in operation_results if not r.success]
        
        return BatchTimingResult(
            operation_name=operation_name,
            total_operations=len(operation_results),
            successful_operations=len(successful),
            failed_operations=len(failed),
            average_execution_time=statistics.mean(execution_times),
            median_execution_time=statistics.median(execution_times),
            min_execution_time=min(execution_times),
            max_execution_time=max(execution_times),
            p95_execution_time=statistics.quantiles(execution_times, n=20)[18] if len(execution_times) > 1 else execution_times[0]
        )
    
    def get_summary(self) -> Dict[str, BatchTimingResult]:
        """
        Get a summary of all operations with their statistics.
        
        Returns:
            Dictionary mapping operation names to their statistics
        """
        # Get unique operation names
        operation_names = set(r.operation_name for r in self._timing_history)
        
        # Calculate stats for each operation
        summary = {}
        for op_name in operation_names:
            stats = self.get_operation_stats(op_name)
            if stats:
                summary[op_name] = stats
        
        return summary
    
    def clear_history(self):
        """Clear the timing history."""
        self._timing_history.clear()


def time_operation(operation_name: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator for timing synchronous functions.
    
    Args:
        operation_name: Name of the operation being timed
        metadata: Additional metadata to store with the timing result
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.debug(f"Operation '{operation_name}' completed in {execution_time*1000:.2f}ms")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.debug(f"Operation '{operation_name}' failed after {execution_time*1000:.2f}ms")
                raise
        return wrapper
    return decorator