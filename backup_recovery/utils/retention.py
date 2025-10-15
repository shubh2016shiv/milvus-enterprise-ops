"""
Retention Policy Management

Provides retention policy management for backups with support for both
count-based and time-based retention strategies, while ensuring minimum
backup guarantees.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from collections import defaultdict

from ..models.entities import BackupVersion

logger = logging.getLogger(__name__)


class RetentionPolicyManager:
    """
    Manage backup retention policies.
    
    This class applies retention policies to determine which backups to keep
    and which to delete, supporting both count-based (keep N most recent) and
    time-based (keep backups newer than X days) strategies.
    
    The manager ensures a minimum number of backups are always retained,
    regardless of age, to prevent accidental deletion of all backups.
    
    Example:
        ```python
        manager = RetentionPolicyManager(
            retention_count=10,
            retention_days=30,
            min_backups_to_keep=3
        )
        
        # Apply policy to backup list
        to_keep, to_delete = manager.apply_retention_policy(all_backups)
        
        print(f"Keeping {len(to_keep)} backups")
        print(f"Deleting {len(to_delete)} backups")
        ```
    """
    
    def __init__(
        self,
        retention_count: int = 10,
        retention_days: int = 30,
        min_backups_to_keep: int = 3
    ):
        """
        Initialize retention policy manager.
        
        Args:
            retention_count: Maximum number of backups to keep per collection
            retention_days: Maximum age of backups to keep (in days)
            min_backups_to_keep: Minimum backups to retain regardless of age
        """
        if retention_count < 0:
            raise ValueError("retention_count cannot be negative")
        if retention_days < 0:
            raise ValueError("retention_days cannot be negative")
        if min_backups_to_keep < 0:
            raise ValueError("min_backups_to_keep cannot be negative")
        
        self.retention_count = retention_count
        self.retention_days = retention_days
        self.min_backups_to_keep = min_backups_to_keep
        
        logger.debug(
            f"Retention policy initialized: "
            f"count={retention_count}, days={retention_days}, min={min_backups_to_keep}"
        )
    
    def apply_retention_policy(
        self,
        backups: List[BackupVersion]
    ) -> Tuple[List[BackupVersion], List[BackupVersion]]:
        """
        Apply retention policy to a list of backups.
        
        This method determines which backups to keep and which to delete based
        on the configured retention policy. Backups are grouped by collection
        name, and policies are applied per collection.
        
        Args:
            backups: List of backup versions to evaluate
        
        Returns:
            Tuple of (backups_to_keep, backups_to_delete)
        
        Example:
            ```python
            to_keep, to_delete = manager.apply_retention_policy(all_backups)
            
            for backup in to_delete:
                print(f"Will delete backup: {backup.backup_name}")
            ```
        """
        if not backups:
            logger.debug("No backups to apply retention policy to")
            return [], []
        
        # Group backups by collection
        backups_by_collection = self._group_by_collection(backups)
        
        to_keep = []
        to_delete = []
        
        # Apply policy per collection
        for collection_name, collection_backups in backups_by_collection.items():
            keep, delete = self._apply_policy_to_collection(
                collection_name,
                collection_backups
            )
            to_keep.extend(keep)
            to_delete.extend(delete)
        
        logger.info(
            f"Retention policy applied: {len(to_keep)} to keep, {len(to_delete)} to delete"
        )
        
        return to_keep, to_delete
    
    def _group_by_collection(
        self,
        backups: List[BackupVersion]
    ) -> dict[str, List[BackupVersion]]:
        """
        Group backups by collection name.
        
        Args:
            backups: List of backup versions
        
        Returns:
            Dictionary mapping collection name to list of backups
        """
        grouped = defaultdict(list)
        for backup in backups:
            grouped[backup.collection_name].append(backup)
        return grouped
    
    def _apply_policy_to_collection(
        self,
        collection_name: str,
        backups: List[BackupVersion]
    ) -> Tuple[List[BackupVersion], List[BackupVersion]]:
        """
        Apply retention policy to backups for a single collection.
        
        Args:
            collection_name: Name of the collection
            backups: List of backup versions for this collection
        
        Returns:
            Tuple of (backups_to_keep, backups_to_delete)
        """
        if not backups:
            return [], []
        
        # Sort backups by creation time (newest first)
        sorted_backups = sorted(
            backups,
            key=lambda b: b.created_at,
            reverse=True
        )
        
        logger.debug(
            f"Applying retention policy to {len(sorted_backups)} backups "
            f"for collection '{collection_name}'"
        )
        
        # Start with all backups to keep
        to_keep = []
        to_delete = []
        
        # Calculate cutoff date for time-based retention
        cutoff_date = None
        if self.retention_days > 0:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        for i, backup in enumerate(sorted_backups):
            should_keep = False
            reason = None
            
            # Always keep minimum number of backups
            if i < self.min_backups_to_keep:
                should_keep = True
                reason = f"within minimum backup count ({self.min_backups_to_keep})"
            
            # Keep if within retention count
            elif self.retention_count > 0 and i < self.retention_count:
                should_keep = True
                reason = f"within retention count ({self.retention_count})"
            
            # Keep if within retention days
            elif cutoff_date and backup.created_at >= cutoff_date:
                should_keep = True
                reason = f"within retention period ({self.retention_days} days)"
            
            if should_keep:
                to_keep.append(backup)
                logger.debug(
                    f"Keeping backup {backup.backup_name} ({backup.age_days:.1f} days old): {reason}"
                )
            else:
                to_delete.append(backup)
                logger.debug(
                    f"Marking for deletion: {backup.backup_name} ({backup.age_days:.1f} days old)"
                )
        
        logger.info(
            f"Collection '{collection_name}': "
            f"{len(to_keep)} backups to keep, {len(to_delete)} to delete"
        )
        
        return to_keep, to_delete
    
    def get_backups_to_keep(self, backups: List[BackupVersion]) -> List[BackupVersion]:
        """
        Get list of backups that should be kept.
        
        Convenience method that returns only the backups to keep.
        
        Args:
            backups: List of backup versions
        
        Returns:
            List of backups to keep
        """
        to_keep, _ = self.apply_retention_policy(backups)
        return to_keep
    
    def get_backups_to_delete(self, backups: List[BackupVersion]) -> List[BackupVersion]:
        """
        Get list of backups that should be deleted.
        
        Convenience method that returns only the backups to delete.
        
        Args:
            backups: List of backup versions
        
        Returns:
            List of backups to delete
        """
        _, to_delete = self.apply_retention_policy(backups)
        return to_delete
    
    def should_keep_backup(
        self,
        backup: BackupVersion,
        all_backups: List[BackupVersion]
    ) -> bool:
        """
        Check if a specific backup should be kept.
        
        This method evaluates a single backup in the context of all backups
        for its collection to determine if it should be retained.
        
        Args:
            backup: Backup to evaluate
            all_backups: All backups for the same collection
        
        Returns:
            True if backup should be kept, False otherwise
        """
        # Get backups for the same collection
        collection_backups = [
            b for b in all_backups
            if b.collection_name == backup.collection_name
        ]
        
        to_keep, _ = self._apply_policy_to_collection(
            backup.collection_name,
            collection_backups
        )
        
        return backup in to_keep
    
    def get_policy_summary(self) -> dict:
        """
        Get a summary of the retention policy configuration.
        
        Returns:
            Dictionary with policy details
        """
        return {
            'retention_count': self.retention_count,
            'retention_days': self.retention_days,
            'min_backups_to_keep': self.min_backups_to_keep,
            'description': self._generate_policy_description()
        }
    
    def _generate_policy_description(self) -> str:
        """Generate human-readable policy description."""
        parts = []
        
        if self.retention_count > 0:
            parts.append(f"keep up to {self.retention_count} most recent backups")
        
        if self.retention_days > 0:
            parts.append(f"keep backups newer than {self.retention_days} days")
        
        if self.min_backups_to_keep > 0:
            parts.append(f"always keep at least {self.min_backups_to_keep} backups")
        
        if not parts:
            return "no retention policy configured"
        
        return "; ".join(parts)
    
    @staticmethod
    def calculate_storage_savings(
        backups_to_delete: List[BackupVersion]
    ) -> Tuple[int, float]:
        """
        Calculate storage that will be freed by deleting backups.
        
        Args:
            backups_to_delete: List of backups to be deleted
        
        Returns:
            Tuple of (total_bytes, total_gb)
        """
        total_bytes = sum(b.size_bytes for b in backups_to_delete)
        total_gb = total_bytes / (1024 ** 3)
        
        return total_bytes, total_gb
    
    def __repr__(self) -> str:
        """String representation of retention policy."""
        return (
            f"RetentionPolicyManager("
            f"count={self.retention_count}, "
            f"days={self.retention_days}, "
            f"min={self.min_backups_to_keep})"
        )

