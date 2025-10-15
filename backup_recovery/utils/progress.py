"""
Progress Tracking Utilities

Provides real-time progress tracking for backup and restore operations with
ETA estimation based on current speed and history.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict
from collections import deque

from ..models.entities import BackupState, BackupProgress

logger = logging.getLogger(__name__)


class BackupProgressTracker:
    """
    Track progress of backup operations with ETA estimation.
    
    This class monitors the progress of long-running backup operations,
    calculating percentage complete, current speed, and estimated time
    remaining based on historical speed data.
    
    Example:
        ```python
        tracker = BackupProgressTracker(
            backup_id="backup_123",
            collection_name="documents",
            total_bytes=1000000000
        )
        
        # Start tracking
        tracker.start_tracking()
        
        # Update progress
        tracker.update_progress(bytes_processed=500000000)
        
        # Get current progress
        progress = tracker.get_current_progress()
        print(f"Progress: {progress.percentage:.2f}%")
        print(f"ETA: {progress.formatted_eta}")
        ```
    """
    
    # Number of speed samples to keep for averaging
    SPEED_HISTORY_SIZE = 10
    
    # Minimum interval between speed calculations (seconds)
    MIN_SPEED_CALC_INTERVAL = 1.0
    
    def __init__(
        self,
        backup_id: str,
        collection_name: str,
        total_bytes: int = 0
    ):
        """
        Initialize progress tracker.
        
        Args:
            backup_id: Unique identifier for the backup
            collection_name: Name of the collection being backed up
            total_bytes: Total bytes to process (0 if unknown)
        """
        self.backup_id = backup_id
        self.collection_name = collection_name
        self.total_bytes = total_bytes
        
        self.state = BackupState.NONE
        self.bytes_processed = 0
        self.start_time: Optional[datetime] = None
        self.last_update_time: Optional[datetime] = None
        self.last_speed_calc_time: Optional[datetime] = None
        self.last_bytes_for_speed: int = 0
        
        # Speed history for smoothing ETA calculations
        self.speed_history: deque = deque(maxlen=self.SPEED_HISTORY_SIZE)
        
        self.error_message: Optional[str] = None
        
        logger.debug(
            f"Progress tracker initialized for {collection_name} "
            f"(backup_id={backup_id}, total_bytes={total_bytes})"
        )
    
    def start_tracking(self) -> None:
        """
        Start tracking progress.
        
        Sets the start time and initializes state.
        """
        self.start_time = datetime.now()
        self.last_update_time = self.start_time
        self.last_speed_calc_time = self.start_time
        self.state = BackupState.IN_PROGRESS
        self.bytes_processed = 0
        self.last_bytes_for_speed = 0
        
        logger.info(f"Started tracking progress for backup {self.backup_id}")
    
    def update_progress(
        self,
        bytes_processed: int,
        state: Optional[BackupState] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Update progress with new information.
        
        Args:
            bytes_processed: Total bytes processed so far
            state: Current state of the backup (optional)
            error_message: Error message if failed (optional)
        """
        now = datetime.now()
        self.bytes_processed = bytes_processed
        self.last_update_time = now
        
        if state is not None:
            self.state = state
        
        if error_message is not None:
            self.error_message = error_message
        
        # Update speed calculation
        self._update_speed(now)
        
        logger.debug(
            f"Progress updated: {self.bytes_processed}/{self.total_bytes} bytes "
            f"({self.percentage:.2f}%)"
        )
    
    def _update_speed(self, now: datetime) -> None:
        """
        Update speed calculation based on recent progress.
        
        Args:
            now: Current timestamp
        """
        if self.last_speed_calc_time is None:
            return
        
        # Only calculate speed if enough time has passed
        time_delta = (now - self.last_speed_calc_time).total_seconds()
        if time_delta < self.MIN_SPEED_CALC_INTERVAL:
            return
        
        # Calculate bytes processed since last speed calculation
        bytes_delta = self.bytes_processed - self.last_bytes_for_speed
        
        if bytes_delta > 0 and time_delta > 0:
            # Calculate speed in bytes per second
            speed_bps = bytes_delta / time_delta
            self.speed_history.append(speed_bps)
            
            # Update tracking variables
            self.last_speed_calc_time = now
            self.last_bytes_for_speed = self.bytes_processed
            
            logger.debug(f"Current speed: {speed_bps / (1024**2):.2f} MB/s")
    
    @property
    def percentage(self) -> float:
        """
        Calculate progress percentage.
        
        Returns:
            Progress percentage (0-100)
        """
        if self.total_bytes <= 0:
            return 0.0
        
        return min(100.0, (self.bytes_processed / self.total_bytes) * 100.0)
    
    @property
    def average_speed_bps(self) -> Optional[float]:
        """
        Get average processing speed in bytes per second.
        
        Returns:
            Average speed or None if not enough data
        """
        if not self.speed_history:
            return None
        
        return sum(self.speed_history) / len(self.speed_history)
    
    @property
    def average_speed_mbps(self) -> Optional[float]:
        """
        Get average processing speed in MB per second.
        
        Returns:
            Average speed in MB/s or None if not enough data
        """
        speed_bps = self.average_speed_bps
        if speed_bps is None:
            return None
        
        return speed_bps / (1024 ** 2)
    
    def estimate_completion(self) -> Optional[datetime]:
        """
        Estimate completion time based on current speed.
        
        Returns:
            Estimated completion datetime or None if cannot be estimated
        """
        if self.total_bytes <= 0 or self.bytes_processed >= self.total_bytes:
            return None
        
        speed = self.average_speed_bps
        if speed is None or speed <= 0:
            return None
        
        # Calculate remaining bytes
        remaining_bytes = self.total_bytes - self.bytes_processed
        
        # Estimate remaining time in seconds
        remaining_seconds = remaining_bytes / speed
        
        # Calculate estimated completion time
        now = datetime.now()
        estimated_completion = now + timedelta(seconds=remaining_seconds)
        
        return estimated_completion
    
    def estimate_remaining_seconds(self) -> Optional[float]:
        """
        Estimate remaining time in seconds.
        
        Returns:
            Estimated seconds remaining or None if cannot be estimated
        """
        estimated_completion = self.estimate_completion()
        if estimated_completion is None:
            return None
        
        now = datetime.now()
        remaining = (estimated_completion - now).total_seconds()
        
        return max(0.0, remaining)
    
    def get_current_progress(self) -> BackupProgress:
        """
        Get current progress snapshot.
        
        Returns:
            BackupProgress object with current state
        """
        now = datetime.now()
        estimated_completion = self.estimate_completion()
        remaining_seconds = self.estimate_remaining_seconds()
        
        return BackupProgress(
            backup_id=self.backup_id,
            collection_name=self.collection_name,
            state=self.state,
            bytes_processed=self.bytes_processed,
            total_bytes=self.total_bytes,
            percentage=self.percentage,
            start_time=self.start_time,
            current_time=now,
            estimated_completion_time=estimated_completion,
            estimated_remaining_time_seconds=remaining_seconds,
            current_speed_mbps=self.average_speed_mbps,
            error_message=self.error_message
        )
    
    def mark_complete(self, success: bool = True, error_message: Optional[str] = None) -> None:
        """
        Mark backup as complete.
        
        Args:
            success: Whether backup completed successfully
            error_message: Error message if failed
        """
        if success:
            self.state = BackupState.COMPLETED
            self.bytes_processed = self.total_bytes  # Ensure 100%
            logger.info(f"Backup {self.backup_id} completed successfully")
        else:
            self.state = BackupState.FAILED
            self.error_message = error_message
            logger.error(f"Backup {self.backup_id} failed: {error_message}")
        
        self.last_update_time = datetime.now()
    
    def mark_verifying(self) -> None:
        """Mark backup as being verified."""
        self.state = BackupState.VERIFYING
        logger.info(f"Backup {self.backup_id} verification started")
    
    def mark_verified(self) -> None:
        """Mark backup as verified."""
        self.state = BackupState.VERIFIED
        logger.info(f"Backup {self.backup_id} verified successfully")
    
    def is_complete(self) -> bool:
        """Check if backup is complete."""
        return self.state in (BackupState.COMPLETED, BackupState.VERIFIED)
    
    def is_failed(self) -> bool:
        """Check if backup failed."""
        return self.state == BackupState.FAILED
    
    @property
    def elapsed_time_seconds(self) -> float:
        """Get elapsed time since start in seconds."""
        if self.start_time is None:
            return 0.0
        
        now = datetime.now()
        return (now - self.start_time).total_seconds()


class BackupProgressTrackerRegistry:
    """
    Registry for managing multiple backup progress trackers.
    
    This class maintains a central registry of all active backup trackers,
    allowing lookup and management of progress for multiple concurrent backups.
    
    Example:
        ```python
        registry = BackupProgressTrackerRegistry()
        
        # Register a new tracker
        tracker = registry.register_backup(
            backup_id="backup_123",
            collection_name="documents",
            total_bytes=1000000000
        )
        
        # Get tracker later
        tracker = registry.get_tracker("backup_123")
        
        # Remove completed tracker
        registry.remove_tracker("backup_123")
        ```
    """
    
    def __init__(self):
        """Initialize registry."""
        self._trackers: Dict[str, BackupProgressTracker] = {}
        logger.debug("Progress tracker registry initialized")
    
    def register_backup(
        self,
        backup_id: str,
        collection_name: str,
        total_bytes: int = 0
    ) -> BackupProgressTracker:
        """
        Register a new backup tracker.
        
        Args:
            backup_id: Unique backup identifier
            collection_name: Collection name
            total_bytes: Total bytes to process
        
        Returns:
            New BackupProgressTracker instance
        """
        tracker = BackupProgressTracker(
            backup_id=backup_id,
            collection_name=collection_name,
            total_bytes=total_bytes
        )
        
        self._trackers[backup_id] = tracker
        logger.info(f"Registered progress tracker for backup {backup_id}")
        
        return tracker
    
    def get_tracker(self, backup_id: str) -> Optional[BackupProgressTracker]:
        """
        Get tracker by backup ID.
        
        Args:
            backup_id: Backup identifier
        
        Returns:
            BackupProgressTracker or None if not found
        """
        return self._trackers.get(backup_id)
    
    def remove_tracker(self, backup_id: str) -> bool:
        """
        Remove tracker from registry.
        
        Args:
            backup_id: Backup identifier
        
        Returns:
            True if removed, False if not found
        """
        if backup_id in self._trackers:
            del self._trackers[backup_id]
            logger.debug(f"Removed progress tracker for backup {backup_id}")
            return True
        return False
    
    def get_all_active_trackers(self) -> Dict[str, BackupProgressTracker]:
        """
        Get all active trackers.
        
        Returns:
            Dictionary of backup_id to tracker
        """
        return self._trackers.copy()
    
    def cleanup_completed(self) -> int:
        """
        Remove all completed or failed trackers.
        
        Returns:
            Number of trackers removed
        """
        to_remove = [
            backup_id
            for backup_id, tracker in self._trackers.items()
            if tracker.is_complete() or tracker.is_failed()
        ]
        
        for backup_id in to_remove:
            self.remove_tracker(backup_id)
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} completed trackers")
        
        return len(to_remove)


# Global registry instance
_global_registry: Optional[BackupProgressTrackerRegistry] = None


def get_registry() -> BackupProgressTrackerRegistry:
    """
    Get the global progress tracker registry.
    
    Returns:
        Global BackupProgressTrackerRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = BackupProgressTrackerRegistry()
    return _global_registry

