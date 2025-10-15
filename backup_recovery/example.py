"""
Example Usage of Backup Recovery Module

Demonstrates comprehensive backup and recovery operations including:
- Creating local and Milvus native backups
- Partition-level backups
- Verification and restore operations
- Progress monitoring
- Retention policy management
"""

import asyncio
import logging
from datetime import datetime

from ..connection_management import ConnectionManager
from ..collection_operations import CollectionManager
from ..config import MilvusSettings

# Import backup recovery components
from . import (
    BackupManager,
    BackupRecoveryConfig,
    BackupParams,
    RestoreParams,
    VerificationParams,
    BackupType,
    BackupStorageType,
    VerificationType
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_basic_backup(backup_manager: BackupManager, collection_name: str):
    """Demonstrate basic full collection backup."""
    logger.info("\n=== Basic Full Collection Backup ===")
    
    try:
        # Create backup with default parameters
        result = await backup_manager.create_backup(
            collection_name=collection_name,
            params=BackupParams(
                backup_type=BackupType.FULL_COLLECTION,
                compression_enabled=True,
                include_indexes=True
            )
        )
        
        if result.success:
            logger.info(f"✓ Backup created successfully!")
            logger.info(f"  Backup ID: {result.backup_id}")
            logger.info(f"  Size: {result.size_mb:.2f} MB")
            logger.info(f"  Time: {result.execution_time_seconds:.2f} seconds")
            logger.info(f"  Storage: {result.storage_path}")
        else:
            logger.error(f"✗ Backup failed: {result.error_message}")
            
    except Exception as e:
        logger.error(f"Error during backup: {e}")


async def example_partition_backup(backup_manager: BackupManager, collection_name: str):
    """Demonstrate partition-level backup."""
    logger.info("\n=== Partition-Level Backup ===")
    
    try:
        # Create backup of specific partitions
        result = await backup_manager.create_backup(
            collection_name=collection_name,
            params=BackupParams(
                backup_type=BackupType.PARTITION,
                partition_names=["partition_2024_01", "partition_2024_02"],
                compression_enabled=True,
                compression_level=9  # Maximum compression
            )
        )
        
        if result.success:
            logger.info(f"✓ Partition backup created!")
            logger.info(f"  Partitions: {result.metadata.partition_names if result.metadata else 'N/A'}")
            logger.info(f"  Size: {result.size_mb:.2f} MB")
        else:
            logger.error(f"✗ Backup failed: {result.error_message}")
            
    except Exception as e:
        logger.error(f"Error during partition backup: {e}")


async def example_list_backups(backup_manager: BackupManager, collection_name: str):
    """Demonstrate listing backups."""
    logger.info("\n=== List Backups ===")
    
    try:
        backups = await backup_manager.list_backups(collection_name=collection_name)
        
        logger.info(f"Found {len(backups)} backup(s) for collection '{collection_name}':")
        
        for backup in backups:
            logger.info(f"\n  Backup: {backup.backup_name}")
            logger.info(f"    ID: {backup.backup_id}")
            logger.info(f"    Created: {backup.created_at}")
            logger.info(f"    Size: {backup.size_mb:.2f} MB")
            logger.info(f"    Type: {backup.backup_type.value}")
            logger.info(f"    State: {backup.state.value}")
            logger.info(f"    Verified: {'Yes' if backup.is_verified else 'No'}")
            
            if backup.compression_enabled:
                logger.info(f"    Compression: Level {backup.compression_level}")
                if backup.compression_ratio:
                    logger.info(f"    Compression ratio: {backup.compression_ratio * 100:.1f}%")
            
    except Exception as e:
        logger.error(f"Error listing backups: {e}")


async def example_verify_backup(backup_manager: BackupManager, backup_id: str, collection_name: str):
    """Demonstrate backup verification."""
    logger.info("\n=== Verify Backup ===")
    
    try:
        # Verify with checksum
        result = await backup_manager.verify_backup(
            backup_id=backup_id,
            collection_name=collection_name,
            params=VerificationParams(
                verification_type=VerificationType.CHECKSUM,
                fail_fast=True
            )
        )
        
        if result.success:
            logger.info(f"✓ Backup verification passed!")
            logger.info(f"  Files verified: {result.files_verified}")
            logger.info(f"  Checksum valid: {result.checksum_valid}")
            logger.info(f"  Time: {result.verification_time_seconds:.2f} seconds")
        else:
            logger.error(f"✗ Verification failed!")
            logger.error(f"  Files failed: {result.files_failed}")
            for error in result.errors:
                logger.error(f"    - {error}")
            
    except Exception as e:
        logger.error(f"Error during verification: {e}")


async def example_restore_backup(backup_manager: BackupManager, backup_id: str, collection_name: str):
    """Demonstrate backup restoration."""
    logger.info("\n=== Restore Backup ===")
    
    try:
        # Restore to a new collection name
        result = await backup_manager.restore_backup(
            backup_id=backup_id,
            collection_name=collection_name,
            params=RestoreParams(
                target_collection_name=f"{collection_name}_restored",
                verify_before_restore=True,
                drop_existing=False,
                load_after_restore=True
            )
        )
        
        if result.success:
            logger.info(f"✓ Backup restored successfully!")
            logger.info(f"  Source: {result.source_collection_name}")
            logger.info(f"  Target: {result.target_collection_name}")
            logger.info(f"  Rows restored: {result.rows_restored:,}")
            logger.info(f"  Time: {result.execution_time_seconds:.2f} seconds")
            logger.info(f"  Verification: {'Passed' if result.verification_passed else 'Failed'}")
        else:
            logger.error(f"✗ Restore failed: {result.error_message}")
            
    except Exception as e:
        logger.error(f"Error during restore: {e}")


async def example_apply_retention_policy(backup_manager: BackupManager, collection_name: str):
    """Demonstrate retention policy application."""
    logger.info("\n=== Apply Retention Policy ===")
    
    try:
        # Dry run first to see what would be deleted
        result = await backup_manager.apply_retention_policy(
            collection_name=collection_name,
            dry_run=True
        )
        
        logger.info(f"Retention policy analysis:")
        logger.info(f"  Total backups: {result['total_backups']}")
        logger.info(f"  To keep: {result['backups_to_keep']}")
        logger.info(f"  To delete: {result['backups_to_delete']}")
        logger.info(f"  Storage to free: {result['storage_to_free_gb']:.2f} GB")
        
        # Actual deletion (commented out for safety)
        # result = await backup_manager.apply_retention_policy(
        #     collection_name=collection_name,
        #     dry_run=False
        # )
        # logger.info(f"Deleted {len(result['deleted_backup_ids'])} backups")
        
    except Exception as e:
        logger.error(f"Error applying retention policy: {e}")


async def example_milvus_native_backup(backup_manager: BackupManager, collection_name: str):
    """Demonstrate Milvus native backup (requires milvus-backup tool)."""
    logger.info("\n=== Milvus Native Backup ===")
    
    try:
        result = await backup_manager.create_backup(
            collection_name=collection_name,
            params=BackupParams(
                backup_type=BackupType.FULL_COLLECTION,
                compression_enabled=True
            ),
            storage_type=BackupStorageType.MILVUS_NATIVE
        )
        
        if result.success:
            logger.info(f"✓ Milvus native backup created!")
            logger.info(f"  Backup ID: {result.backup_id}")
        else:
            logger.error(f"✗ Backup failed: {result.error_message}")
            
    except Exception as e:
        logger.error(f"Error during Milvus native backup: {e}")


async def main():
    """Main demonstration function."""
    logger.info("=== Backup Recovery Module Examples ===\n")
    
    try:
        # 1. Create configuration
        logger.info("1. Creating configuration...")
        config = BackupRecoveryConfig(
            local_backup_root_path="./milvus_backups",
            compression_enabled=True,
            compression_level=6,
            retention_count=10,
            retention_days=30,
            min_backups_to_keep=3,
            enable_timing=True,
            auto_verify_after_backup=True
        )
        logger.info(f"   Config: {config}")
        
        # 2. Initialize managers
        logger.info("\n2. Initializing managers...")
        milvus_settings = MilvusSettings()
        connection_manager = ConnectionManager(config=milvus_settings)
        collection_manager = CollectionManager(connection_manager=connection_manager)
        
        backup_manager = BackupManager(
            connection_manager=connection_manager,
            collection_manager=collection_manager,
            config=config
        )
        logger.info("   ✓ BackupManager initialized")
        
        # 3. Set collection name
        collection_name = "test_collection"
        logger.info(f"\n3. Working with collection: '{collection_name}'")
        logger.info("   Note: Collection must exist before running these examples")
        
        # 4. Run examples
        await example_basic_backup(backup_manager, collection_name)
        await example_list_backups(backup_manager, collection_name)
        
        # Get first backup ID for other examples
        backups = await backup_manager.list_backups(collection_name=collection_name)
        if backups:
            backup_id = backups[0].backup_id
            await example_verify_backup(backup_manager, backup_id, collection_name)
            # await example_restore_backup(backup_manager, backup_id, collection_name)  # Commented for safety
        
        await example_apply_retention_policy(backup_manager, collection_name)
        
        # Optional: Try Milvus native backup
        # await example_milvus_native_backup(backup_manager, collection_name)
        
        logger.info("\n=== Examples Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Example failed with error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())

