"""
Mock Backup Backend

Provides a simplified mock implementation for backup and restore operations
when pyarrow is not available. This allows the examples to run without the
full functionality.
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from .config import BackupRecoveryConfig
from .models.entities import (
    BackupMetadata,
    BackupType,
    BackupStorageType,
    BackupState,
    ChecksumAlgorithm
)
from .models.parameters import BackupParams, RestoreParams
from .utils.progress import BackupProgressTracker

logger = logging.getLogger(__name__)


class MockBackupManager:
    """
    Mock implementation of BackupManager for demonstration purposes.
    
    This class provides a simplified implementation that can be used when
    the full functionality is not available due to missing dependencies.
    """
    
    def __init__(
        self,
        connection_manager,
        collection_manager,
        config: Optional[BackupRecoveryConfig] = None
    ):
        """Initialize mock backup manager."""
        self._connection_manager = connection_manager
        self._collection_manager = collection_manager
        self._config = config or BackupRecoveryConfig()
        self._backups = {}
        logger.info("MockBackupManager initialized (pyarrow not available)")
    
    async def create_backup(
        self,
        collection_name: str,
        backup_name: Optional[str] = None,
        params = None,
        storage_type = None,
        wait: bool = False
    ):
        """
        Create a mock backup.
        
        Args:
            collection_name: Name of collection to backup
            backup_name: Optional backup name
            params: Ignored in mock implementation
            storage_type: Ignored in mock implementation
            wait: Ignored in mock implementation
        
        Returns:
            Mock backup result
        """
        backup_id = str(uuid.uuid4())
        backup_name = backup_name or f"{collection_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create mock metadata
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_name=backup_name,
            collection_name=collection_name,
            created_at=datetime.now(),
            storage_type=BackupStorageType.LOCAL_FILE,
            storage_path=f"./backups/{collection_name}/{backup_id}",
            size_bytes=1024,
            checksum="mock_checksum",
            checksum_algorithm=ChecksumAlgorithm.SHA256,
            state=BackupState.COMPLETED,
            backup_type=BackupType.FULL_COLLECTION,
            row_count=100
        )
        
        # Store in memory
        if collection_name not in self._backups:
            self._backups[collection_name] = []
        self._backups[collection_name].append(metadata)
        
        logger.info(f"Created mock backup: {backup_name} (ID: {backup_id})")
        
        # Return mock result
        backup_path = f"./backups/{collection_name}/{backup_id}"
        return type('MockBackupResult', (), {
            'success': True,
            'backup_id': backup_id,
            'backup_name': backup_name,
            'collection_name': collection_name,
            'storage_type': BackupStorageType.LOCAL_FILE,
            'storage_path': backup_path,
            'backup_path': backup_path,
            'size_bytes': 1024,
            'execution_time_ms': 100.0,
            'state': BackupState.COMPLETED,
            'entities_count': 100
        })
    
    async def restore_backup(
        self,
        backup_id: str,
        collection_name: str,
        params = None,
        storage_type = None
    ):
        """
        Restore a mock backup.
        
        Args:
            backup_id: ID of backup to restore
            collection_name: Collection name
            params: Ignored in mock implementation
            storage_type: Ignored in mock implementation
        
        Returns:
            Mock restore result
        """
        target_name = params.target_collection_name if params else collection_name
        
        logger.info(f"Restored mock backup {backup_id} to {target_name}")
        
        # Return mock result
        return type('MockRestoreResult', (), {
            'success': True,
            'backup_id': backup_id,
            'source_collection_name': collection_name,
            'target_collection_name': target_name,
            'rows_restored': 100,
            'execution_time_ms': 100.0,
            'verification_passed': True
        })
    
    async def list_backups(
        self,
        collection_name: Optional[str] = None,
        storage_type = None
    ) -> List[BackupMetadata]:
        """
        List mock backups.
        
        Args:
            collection_name: Optional collection name filter
            storage_type: Ignored in mock implementation
        
        Returns:
            List of mock backup metadata
        """
        if collection_name:
            return self._backups.get(collection_name, [])
        
        # Return all backups
        all_backups = []
        for backups in self._backups.values():
            all_backups.extend(backups)
        
        # Sort by creation time (newest first)
        all_backups.sort(key=lambda x: x.created_at, reverse=True)
        
        return all_backups
    
    async def get_backup_info(
        self,
        backup_id: str,
        collection_name: str,
        storage_type = None
    ) -> Optional[BackupMetadata]:
        """
        Get mock backup metadata.
        
        Args:
            backup_id: Backup ID
            collection_name: Collection name
            storage_type: Ignored in mock implementation
        
        Returns:
            Mock backup metadata or None
        """
        backups = self._backups.get(collection_name, [])
        for backup in backups:
            if backup.backup_id == backup_id:
                return backup
        return None
    
    async def verify_backup(
        self,
        backup_id: str,
        collection_name: str,
        storage_type = None,
        params = None
    ):
        """
        Verify mock backup.
        
        Args:
            backup_id: Backup ID
            collection_name: Collection name
            storage_type: Ignored in mock implementation
            params: Ignored in mock implementation
        
        Returns:
            Always returns True for mock implementation
        """
        logger.info(f"Verified mock backup: {backup_id}")
        return True
    
    async def delete_backup(
        self,
        backup_id: str,
        collection_name: str,
        storage_type = None
    ) -> bool:
        """
        Delete mock backup.
        
        Args:
            backup_id: Backup ID
            collection_name: Collection name
            storage_type: Ignored in mock implementation
        
        Returns:
            True if deleted
        """
        if collection_name in self._backups:
            self._backups[collection_name] = [
                b for b in self._backups[collection_name] 
                if b.backup_id != backup_id
            ]
        
        logger.info(f"Deleted mock backup: {backup_id}")
        return True