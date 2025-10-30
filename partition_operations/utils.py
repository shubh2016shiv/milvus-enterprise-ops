"""
Utility functions for partition operations.

This module provides various utility functions and helpers for
partition management operations.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime, timedelta
from functools import wraps
import weakref

logger = logging.getLogger(__name__)


class PartitionProgressTracker:
    """
    Tracks progress of partition operations for monitoring and reporting.
    
    This class provides functionality to track the progress of long-running
    partition operations and generate progress reports.
    """
    
    def __init__(self, max_operations: int = 1000):
        """
        Initialize the progress tracker.
        
        Args:
            max_operations: Maximum number of operations to track
        """
        self.max_operations = max_operations
        self.operations: Dict[str, Dict[str, Any]] = {}
        self._weakref_dict = weakref.WeakValueDictionary()
    
    def start_operation(
        self, 
        operation_id: str, 
        operation_type: str, 
        collection_name: str,
        partition_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Start tracking a new operation.
        
        Args:
            operation_id: Unique identifier for the operation
            operation_type: Type of operation (create, drop, load, etc.)
            collection_name: Name of the collection
            partition_name: Optional name of the partition
            metadata: Optional additional metadata
        """
        self.operations[operation_id] = {
            'type': operation_type,
            'collection_name': collection_name,
            'partition_name': partition_name,
            'status': 'started',
            'start_time': time.time(),
            'end_time': None,
            'progress': 0.0,
            'metadata': metadata or {}
        }
        logger.debug(f"Started tracking operation {operation_id} ({operation_type})")
    
    def update_progress(self, operation_id: str, progress: float, status: str = None) -> None:
        """
        Update the progress of an operation.
        
        Args:
            operation_id: The operation identifier
            progress: Progress value between 0.0 and 1.0
            status: Optional status update
        """
        if operation_id in self.operations:
            self.operations[operation_id]['progress'] = max(0.0, min(1.0, progress))
            if status:
                self.operations[operation_id]['status'] = status
            logger.debug(f"Updated operation {operation_id} progress to {progress:.2%}")
    
    def complete_operation(self, operation_id: str, success: bool = True, error: str = None) -> None:
        """
        Mark an operation as completed.
        
        Args:
            operation_id: The operation identifier
            success: Whether the operation was successful
            error: Optional error message if operation failed
        """
        if operation_id in self.operations:
            self.operations[operation_id].update({
                'status': 'completed' if success else 'failed',
                'end_time': time.time(),
                'progress': 1.0 if success else self.operations[operation_id]['progress']
            })
            if error:
                self.operations[operation_id]['error'] = error
            logger.info(f"Completed operation {operation_id} - {'success' if success else 'failed'}")
    
    def get_operation_progress(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current progress of an operation.
        
        Args:
            operation_id: The operation identifier
            
        Returns:
            Dictionary with operation progress information
        """
        return self.operations.get(operation_id)
    
    def get_active_operations(self) -> List[Dict[str, Any]]:
        """
        Get all currently active operations.
        
        Returns:
            List of active operations
        """
        return [
            op for op in self.operations.values()
            if op['status'] in ['started', 'in_progress']
        ]
    
    def get_operation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all operations.
        
        Returns:
            Dictionary with operation statistics
        """
        total_ops = len(self.operations)
        active_ops = len(self.get_active_operations())
        completed_ops = len([op for op in self.operations.values() if op['status'] == 'completed'])
        failed_ops = len([op for op in self.operations.values() if op['status'] == 'failed'])
        
        return {
            'total_operations': total_ops,
            'active_operations': active_ops,
            'completed_operations': completed_ops,
            'failed_operations': failed_ops,
            'success_rate': completed_ops / total_ops if total_ops > 0 else 0.0
        }


class PartitionTimer:
    """
    Timer utility for measuring operation performance.
    
    This class provides functionality to time partition operations
    and collect performance metrics.
    """
    
    def __init__(self):
        """Initialize the timer."""
        self.operations: Dict[str, List[float]] = {}
    
    def start_timer(self, operation_name: str) -> str:
        """
        Start timing an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Timer identifier
        """
        timer_id = f"{operation_name}_{time.time()}"
        if operation_name not in self.operations:
            self.operations[operation_name] = []
        return timer_id
    
    def end_timer(self, timer_id: str) -> float:
        """
        End timing and return elapsed time.
        
        Args:
            timer_id: Timer identifier from start_timer
            
        Returns:
            Elapsed time in seconds
        """
        operation_name = timer_id.rsplit('_', 1)[0]
        elapsed_time = time.time() - float(timer_id.rsplit('_', 1)[1])
        
        if operation_name in self.operations:
            self.operations[operation_name].append(elapsed_time)
        
        return elapsed_time
    
    def time_operation(self, operation_name: str, func: Callable, *args, **kwargs) -> Any:
        """
        Time a function execution.
        
        Args:
            operation_name: Name of the operation
            func: Function to time
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        timer_id = self.start_timer(operation_name)
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = self.end_timer(timer_id)
            logger.debug(f"Operation {operation_name} took {elapsed:.3f}s")
    
    def get_operation_stats(self, operation_name: str) -> Optional[Dict[str, float]]:
        """
        Get statistics for an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Dictionary with operation statistics
        """
        if operation_name not in self.operations:
            return None
        
        times = self.operations[operation_name]
        if not times:
            return None
        
        return {
            'count': len(times),
            'min_time': min(times),
            'max_time': max(times),
            'avg_time': sum(times) / len(times),
            'total_time': sum(times)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all operations.
        
        Returns:
            Dictionary with statistics for all operations
        """
        return {op_name: self.get_operation_stats(op_name) 
                for op_name in self.operations}


# Essential utilities - removed over-engineered functions that weren't being used


# Global instances
_global_progress_tracker = None
_global_timer = None


def get_global_progress_tracker() -> PartitionProgressTracker:
    """Get global progress tracker instance."""
    global _global_progress_tracker
    if _global_progress_tracker is None:
        _global_progress_tracker = PartitionProgressTracker()
    return _global_progress_tracker


def get_global_timer() -> PartitionTimer:
    """Get global timer instance."""
    global _global_timer
    if _global_timer is None:
        _global_timer = PartitionTimer()
    return _global_timer