"""
Milvus Native Backup Backend

Implements backup and restore operations using Milvus-Backup tool.
This is a wrapper around the Milvus-Backup command-line tool or API.

Note: This implementation assumes Milvus-Backup tool is installed and accessible.
"""

import json
import logging
import subprocess
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..config import BackupRecoveryConfig
from ..models.entities import (
    BackupMetadata,
    BackupStorageType,
    BackupState
)
from ..models.parameters import BackupParams
from ..exceptions import (
    BackupError,
    BackupNotFoundError,
    RestoreError
)
from ..utils.progress import BackupProgressTracker

logger = logging.getLogger(__name__)


class MilvusNativeBackupBackend:
    """
    Milvus native backup backend using Milvus-Backup tool.
    
    This backend wraps the Milvus-Backup command-line tool to provide
    backup and restore functionality using Milvus's native backup system.
    
    Note: Requires Milvus-Backup tool to be installed separately.
    See: https://github.com/zilliztech/milvus-backup
    
    Example:
        ```python
        backend = MilvusNativeBackupBackend(config)
        
        # Create backup
        metadata = await backend.create_backup(
            collection_name="documents",
            params=backup_params
        )
        
        # Restore backup
        await backend.restore_backup(
            backup_name="backup_2024_01_15",
            target_collection_name="documents_restored"
        )
        ```
    """
    
    def __init__(self, config: BackupRecoveryConfig):
        """
        Initialize Milvus native backup backend.
        
        Args:
            config: Backup recovery configuration
        """
        self.config = config
        self.milvus_backup_bin = self._find_milvus_backup_binary()
        
        logger.info("MilvusNativeBackupBackend initialized")
    
    def _find_milvus_backup_binary(self) -> Optional[str]:
        """
        Find Milvus-Backup binary in PATH.
        
        Returns:
            Path to milvus-backup binary or None
        """
        try:
            result = subprocess.run(
                ["which", "milvus-backup"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.warning(f"Failed to find milvus-backup binary: {e}")
        
        return None
    
    async def create_backup(
        self,
        collection_name: str,
        params: BackupParams,
        backup_name: Optional[str] = None,
        progress_tracker: Optional[BackupProgressTracker] = None
    ) -> BackupMetadata:
        """
        Create a backup using Milvus-Backup tool.
        
        Args:
            collection_name: Name of collection to backup
            params: Backup parameters
            backup_name: Optional backup name
            progress_tracker: Optional progress tracker
        
        Returns:
            BackupMetadata with backup information
        
        Raises:
            BackupError: If backup creation fails
        """
        if not self.milvus_backup_bin:
            raise BackupError(
                "Milvus-Backup tool not found. Please install milvus-backup first.",
                collection_name=collection_name
            )
        
        backup_name = backup_name or params.backup_name or f"{collection_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            if progress_tracker:
                progress_tracker.start_tracking()
            
            logger.info(f"Starting Milvus native backup for collection '{collection_name}'")
            
            # Build milvus-backup command
            cmd = [
                self.milvus_backup_bin,
                "create",
                "-n", backup_name,
                "-c", collection_name
            ]
            
            # Execute backup command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.default_backup_timeout
            )
            
            if result.returncode != 0:
                raise BackupError(
                    f"Milvus-Backup command failed: {result.stderr}",
                    collection_name=collection_name
                )
            
            # Parse output to get backup info
            # Note: This is simplified - actual implementation would parse JSON output
            metadata = BackupMetadata(
                backup_id=backup_name,
                backup_name=backup_name,
                collection_name=collection_name,
                partition_names=[],
                created_at=datetime.now(),
                storage_type=BackupStorageType.MILVUS_NATIVE,
                storage_path=self.config.milvus_backup_bucket or "milvus_backups",
                size_bytes=0,  # Would be parsed from output
                state=BackupState.COMPLETED,
                backup_type=params.backup_type,
                compression_enabled=params.compression_enabled,
                include_indexes=params.include_indexes
            )
            
            if progress_tracker:
                progress_tracker.mark_complete(success=True)
            
            logger.info(f"Milvus native backup completed: {backup_name}")
            return metadata
            
        except subprocess.TimeoutExpired:
            if progress_tracker:
                progress_tracker.mark_complete(success=False, error_message="Timeout")
            raise BackupError(
                f"Backup timed out after {self.config.default_backup_timeout} seconds",
                collection_name=collection_name
            )
        except Exception as e:
            if progress_tracker:
                progress_tracker.mark_complete(success=False, error_message=str(e))
            raise BackupError(
                f"Failed to create Milvus native backup: {e}",
                collection_name=collection_name
            )
    
    async def restore_backup(
        self,
        backup_name: str,
        target_collection_name: Optional[str] = None
    ) -> bool:
        """
        Restore a backup using Milvus-Backup tool.
        
        Args:
            backup_name: Name of backup to restore
            target_collection_name: Optional target collection name
        
        Returns:
            True if restore succeeded
        
        Raises:
            RestoreError: If restore fails
        """
        if not self.milvus_backup_bin:
            raise RestoreError(
                "Milvus-Backup tool not found",
                backup_id=backup_name
            )
        
        try:
            logger.info(f"Starting Milvus native restore for backup '{backup_name}'")
            
            # Build restore command
            cmd = [
                self.milvus_backup_bin,
                "restore",
                "-n", backup_name
            ]
            
            if target_collection_name:
                cmd.extend(["-t", target_collection_name])
            
            # Execute restore command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.default_restore_timeout
            )
            
            if result.returncode != 0:
                raise RestoreError(
                    f"Milvus-Backup restore failed: {result.stderr}",
                    backup_id=backup_name,
                    target_collection_name=target_collection_name
                )
            
            logger.info(f"Milvus native restore completed: {backup_name}")
            return True
            
        except subprocess.TimeoutExpired:
            raise RestoreError(
                f"Restore timed out after {self.config.default_restore_timeout} seconds",
                backup_id=backup_name
            )
        except Exception as e:
            raise RestoreError(
                f"Failed to restore Milvus native backup: {e}",
                backup_id=backup_name,
                target_collection_name=target_collection_name
            )
    
    async def list_backups(self) -> List[BackupMetadata]:
        """
        List available Milvus native backups.
        
        Returns:
            List of BackupMetadata
        """
        if not self.milvus_backup_bin:
            logger.warning("Milvus-Backup tool not found")
            return []
        
        try:
            # List backups command
            result = subprocess.run(
                [self.milvus_backup_bin, "list"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.warning(f"Failed to list backups: {result.stderr}")
                return []
            
            # Parse output (simplified - actual implementation would parse JSON)
            logger.debug("Milvus native backups listed")
            return []
            
        except Exception as e:
            logger.error(f"Failed to list Milvus native backups: {e}")
            return []
    
    async def delete_backup(self, backup_name: str) -> bool:
        """
        Delete a Milvus native backup.
        
        Args:
            backup_name: Name of backup to delete
        
        Returns:
            True if deleted successfully
        """
        if not self.milvus_backup_bin:
            return False
        
        try:
            result = subprocess.run(
                [self.milvus_backup_bin, "delete", "-n", backup_name],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Failed to delete Milvus native backup: {e}")
            return False

