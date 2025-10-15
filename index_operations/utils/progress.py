"""
Index Build Progress Tracking

Provides utilities for tracking the progress of long-running index build operations,
including estimating completion time and remaining time.

Typical usage:
    from Milvus_Ops.index_operations.utils import IndexBuildTracker
    
    # Start tracking a build
    tracker = IndexBuildTracker(
        collection_name="documents",
        field_name="embedding",
        total_rows=1000000
    )
    tracker.start_tracking()
    
    # Update progress periodically
    tracker.update_progress(processed_rows=250000)
    
    # Get current progress
    progress = tracker.get_current_progress()
    print(f"Progress: {progress.percentage:.2f}%")
    print(f"ETA: {progress.formatted_eta}")
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta

from index_operations.models.entities import IndexState, IndexBuildProgress

logger = logging.getLogger(__name__)


class IndexBuildTracker:
    """
    Tracks the progress of an index build operation.
    
    This class provides methods for tracking the progress of a long-running
    index build operation, including estimating completion time and remaining time.
    It maintains a history of progress updates to improve estimation accuracy.
    
    Attributes:
        collection_name: Name of the collection
        field_name: Name of the field being indexed
        total_rows: Total number of rows to index
    """
    
    def __init__(
        self,
        collection_name: str,
        field_name: str,
        total_rows: Optional[int] = None
    ):
        """
        Initialize the tracker.
        
        Args:
            collection_name: Name of the collection
            field_name: Name of the field being indexed
            total_rows: Total number of rows to index (if known)
        """
        self.collection_name = collection_name
        self.field_name = field_name
        self.total_rows = total_rows
        
        self.state = IndexState.NONE
        self.start_time: Optional[datetime] = None
        self.last_update_time: Optional[datetime] = None
        self.processed_rows = 0
        self.percentage = 0.0
        self.history: List[Tuple[datetime, int, float]] = []  # [(time, rows, percentage)]
        self.failed_reason: Optional[str] = None
    
    def start_tracking(self) -> None:
        """
        Start tracking the build progress.
        
        This method initializes the tracker with the current time and
        sets the state to CREATING.
        """
        self.start_time = datetime.now()
        self.last_update_time = self.start_time
        self.state = IndexState.CREATING
        self.processed_rows = 0
        self.percentage = 0.0
        self.history = [(self.start_time, 0, 0.0)]
        logger.debug(
            f"Started tracking index build for {self.collection_name}.{self.field_name}"
        )
    
    def update_progress(
        self,
        processed_rows: Optional[int] = None,
        percentage: Optional[float] = None,
        state: Optional[IndexState] = None,
        failed_reason: Optional[str] = None
    ) -> None:
        """
        Update the build progress.
        
        This method updates the tracker with the latest progress information
        and adds an entry to the history.
        
        Args:
            processed_rows: Number of rows processed so far
            percentage: Completion percentage (0-100)
            state: Current state of the index build
            failed_reason: Reason for failure if state is FAILED
        """
        now = datetime.now()
        self.last_update_time = now
        
        # Update state if provided
        if state is not None:
            self.state = state
            
            if state == IndexState.FAILED and failed_reason:
                self.failed_reason = failed_reason
                logger.warning(
                    f"Index build for {self.collection_name}.{self.field_name} "
                    f"failed: {failed_reason}"
                )
            
            elif state == IndexState.CREATED:
                self.percentage = 100.0
                if self.total_rows:
                    self.processed_rows = self.total_rows
                logger.info(
                    f"Index build for {self.collection_name}.{self.field_name} completed"
                )
        
        # Update processed rows if provided
        if processed_rows is not None:
            self.processed_rows = processed_rows
            
            # Calculate percentage if total_rows is known
            if self.total_rows:
                self.percentage = min(100.0, (processed_rows / self.total_rows) * 100.0)
        
        # Update percentage if provided
        if percentage is not None:
            self.percentage = min(100.0, max(0.0, percentage))
            
            # Calculate processed_rows if total_rows is known
            if self.total_rows:
                self.processed_rows = int((self.percentage / 100.0) * self.total_rows)
        
        # Add to history
        self.history.append((now, self.processed_rows, self.percentage))
        
        # Trim history if it gets too large
        if len(self.history) > 100:
            # Keep first entry, last 50 entries
            self.history = [self.history[0]] + self.history[-50:]
    
    def estimate_completion_time(self) -> Tuple[Optional[datetime], Optional[float]]:
        """
        Estimate the completion time and remaining time.
        
        This method uses the history of progress updates to estimate
        when the index build will complete and how much time remains.
        
        Returns:
            Tuple of (estimated completion time, estimated remaining seconds)
        """
        if self.state == IndexState.CREATED:
            return None, 0.0
            
        if self.state == IndexState.FAILED:
            return None, None
            
        if len(self.history) < 2:
            return None, None
        
        # Use the last 10 history points or all if fewer
        history_points = min(10, len(self.history) - 1)
        start_idx = max(0, len(self.history) - 1 - history_points)
        
        start_time, start_rows, start_pct = self.history[start_idx]
        end_time, end_rows, end_pct = self.history[-1]
        
        # Calculate progress rate
        time_diff = (end_time - start_time).total_seconds()
        if time_diff <= 0:
            return None, None
        
        if self.total_rows:
            # Use rows for estimation if available
            rows_diff = end_rows - start_rows
            if rows_diff <= 0:
                return None, None
                
            rows_per_second = rows_diff / time_diff
            remaining_rows = self.total_rows - end_rows
            
            if rows_per_second <= 0:
                return None, None
                
            remaining_seconds = remaining_rows / rows_per_second
            
        else:
            # Use percentage for estimation
            pct_diff = end_pct - start_pct
            if pct_diff <= 0:
                return None, None
                
            pct_per_second = pct_diff / time_diff
            remaining_pct = 100.0 - end_pct
            
            if pct_per_second <= 0:
                return None, None
                
            remaining_seconds = remaining_pct / pct_per_second
        
        # Calculate estimated completion time
        completion_time = end_time + timedelta(seconds=remaining_seconds)
        
        return completion_time, remaining_seconds
    
    def get_current_progress(self) -> IndexBuildProgress:
        """
        Get the current progress information.
        
        Returns:
            IndexBuildProgress with current progress information
        """
        estimated_completion, remaining_seconds = self.estimate_completion_time()
        
        return IndexBuildProgress(
            collection_name=self.collection_name,
            field_name=self.field_name,
            state=self.state,
            percentage=self.percentage,
            start_time=self.start_time,
            current_time=datetime.now(),
            estimated_completion_time=estimated_completion,
            estimated_remaining_time_seconds=remaining_seconds,
            total_rows=self.total_rows,
            processed_rows=self.processed_rows,
            failed_reason=self.failed_reason
        )
    
    def is_complete(self) -> bool:
        """
        Check if the build is complete.
        
        Returns:
            True if the build is complete (success or failure)
        """
        return self.state in (IndexState.CREATED, IndexState.FAILED)


class IndexBuildTrackerRegistry:
    """
    Registry for tracking multiple index builds.
    
    This class maintains a registry of active index build trackers,
    allowing for centralized management of multiple concurrent builds.
    """
    
    def __init__(self):
        """Initialize the registry."""
        self._trackers: Dict[str, IndexBuildTracker] = {}
    
    def _get_key(self, collection_name: str, field_name: str) -> str:
        """Get a unique key for a collection and field."""
        return f"{collection_name}:{field_name}"
    
    def register_build(
        self,
        collection_name: str,
        field_name: str,
        total_rows: Optional[int] = None
    ) -> IndexBuildTracker:
        """
        Register a new index build.
        
        Args:
            collection_name: Name of the collection
            field_name: Name of the field being indexed
            total_rows: Total number of rows to index
            
        Returns:
            The created tracker
        """
        key = self._get_key(collection_name, field_name)
        tracker = IndexBuildTracker(collection_name, field_name, total_rows)
        tracker.start_tracking()
        self._trackers[key] = tracker
        return tracker
    
    def get_tracker(
        self,
        collection_name: str,
        field_name: str
    ) -> Optional[IndexBuildTracker]:
        """
        Get the tracker for a specific index build.
        
        Args:
            collection_name: Name of the collection
            field_name: Name of the field being indexed
            
        Returns:
            The tracker, or None if not found
        """
        key = self._get_key(collection_name, field_name)
        return self._trackers.get(key)
    
    def update_progress(
        self,
        collection_name: str,
        field_name: str,
        **kwargs
    ) -> None:
        """
        Update the progress of an index build.
        
        Args:
            collection_name: Name of the collection
            field_name: Name of the field being indexed
            **kwargs: Progress update parameters
        """
        tracker = self.get_tracker(collection_name, field_name)
        if tracker:
            tracker.update_progress(**kwargs)
    
    def get_progress(
        self,
        collection_name: str,
        field_name: str
    ) -> Optional[IndexBuildProgress]:
        """
        Get the progress of an index build.
        
        Args:
            collection_name: Name of the collection
            field_name: Name of the field being indexed
            
        Returns:
            IndexBuildProgress, or None if not found
        """
        tracker = self.get_tracker(collection_name, field_name)
        if tracker:
            return tracker.get_current_progress()
        return None
    
    def remove_tracker(
        self,
        collection_name: str,
        field_name: str
    ) -> None:
        """
        Remove a tracker from the registry.
        
        Args:
            collection_name: Name of the collection
            field_name: Name of the field being indexed
        """
        key = self._get_key(collection_name, field_name)
        if key in self._trackers:
            del self._trackers[key]
    
    def get_active_builds(self) -> List[IndexBuildProgress]:
        """
        Get all active index builds.
        
        Returns:
            List of IndexBuildProgress for all active builds
        """
        return [
            tracker.get_current_progress()
            for tracker in self._trackers.values()
            if tracker.state == IndexState.CREATING
        ]
    
    def get_all_builds(self) -> List[IndexBuildProgress]:
        """
        Get all index builds (active, completed, and failed).
        
        Returns:
            List of IndexBuildProgress for all builds
        """
        return [
            tracker.get_current_progress()
            for tracker in self._trackers.values()
        ]


# Global registry instance
_registry = IndexBuildTrackerRegistry()


def get_registry() -> IndexBuildTrackerRegistry:
    """
    Get the global registry instance.
    
    Returns:
        The global IndexBuildTrackerRegistry instance
    """
    return _registry
