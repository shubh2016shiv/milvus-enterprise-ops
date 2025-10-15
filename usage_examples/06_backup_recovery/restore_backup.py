"""
Restore Backup Example

Demonstrates how to restore a collection from a backup.
Shows restoration process and verification of restored data.
"""

import sys
import os
import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("restore_example")

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from connection_management import ConnectionManager
from collection_operations import CollectionManager
from backup_recovery import BackupManager, BackupRecoveryConfig, BackupStorageType
from config import load_settings
# Import usage_examples utils (not the project's utils package)
import importlib.util
utils_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils.py'))
spec = importlib.util.spec_from_file_location("example_utils", utils_file_path)
example_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example_utils)
print_section = example_utils.print_section
print_step = example_utils.print_step
print_success = example_utils.print_success
print_info = example_utils.print_info
print_error = example_utils.print_error
print_warning = example_utils.print_warning


COLLECTION_NAME = "test_example_collection"
RESTORED_COLLECTION_NAME = "test_example_collection_restored"
BACKUP_DIR = "./backups"


async def main():
    """Main function to demonstrate backup restoration."""
    print_section("Restore Backup Example")
    
    print_warning("This will create a new collection from backup")
    print(f"  Restored collection name: {RESTORED_COLLECTION_NAME}\n")
    
    # Step 1: Initialize Managers
    print_step(1, "Initialize Managers")
    try:
        config = load_settings()
        conn_manager = ConnectionManager(config=config)
        coll_manager = CollectionManager(conn_manager)
        
        backup_config = BackupRecoveryConfig(
            default_storage_type=BackupStorageType.LOCAL_FILE,
            local_backup_root_path=BACKUP_DIR,
            compression_enabled=True,
            enable_checksum_verification=True
        )
        
        backup_manager = BackupManager(
            connection_manager=conn_manager,
            collection_manager=coll_manager,
            config=backup_config
        )
        
        print_success("Managers initialized")
    except Exception as e:
        print_error(f"Initialization failed: {e}")
        return
    
    # Step 2: List Available Backups
    print_step(2, "List Available Backups")
    try:
        logger.info("Listing available backups")
        backups = await backup_manager.list_backups()
        logger.info(f"Found {len(backups)} backups")
        
        if not backups:
            print_error("No backups found")
            print_info("Hint", "Run create_backup.py first")
            conn_manager.close()
            return
        
        print_success(f"Found {len(backups)} backup(s)")
        print("\n  Available Backups:\n")
        for i, backup in enumerate(backups, 1):
            print(f"  {i}. {backup.backup_name}")
            print(f"     Collection: {backup.collection_name}")
            print(f"     Created: {backup.created_at}")
            print(f"     Entities: {backup.row_count}")
            logger.info(f"Backup details: {backup.__dict__ if hasattr(backup, '__dict__') else backup}")
            print()
        
        # Select the most recent backup
        latest_backup = backups[-1]
        backup_name = latest_backup.backup_name
        
        print_info("Selected backup", backup_name)
    except Exception as e:
        print_error(f"Listing backups failed: {e}")
        conn_manager.close()
        return
    
    # Step 3: Verify Backup Integrity
    print_step(3, "Verify Backup Integrity")
    try:
        print_info("Action", "Verifying checksum...")
        
        logger.info(f"Verifying backup: {backup_name}")
        is_valid = await backup_manager.verify_backup(
            backup_id=backup_name,
            collection_name=latest_backup.collection_name
        )
        logger.info(f"Verification result: {is_valid}")
        
        if is_valid:
            print_success("Backup integrity verified")
        else:
            print_error("Backup integrity check failed")
            print_warning("Continuing with restore, but data may be corrupted")
    except Exception as e:
        print_error(f"Verification failed: {e}")
        print_warning("Continuing with restore...")
    
    # Step 4: Check if Restored Collection Exists
    print_step(4, "Check for Existing Restored Collection")
    try:
        if await coll_manager.has_collection(RESTORED_COLLECTION_NAME):
            print_warning(f"Collection '{RESTORED_COLLECTION_NAME}' already exists")
            print_info("Action", "Dropping existing collection...")
            
            await coll_manager.drop_collection(RESTORED_COLLECTION_NAME)
            print_success("Existing collection dropped")
        else:
            print_info("Status", "No existing collection, ready to restore")
    except Exception as e:
        print_error(f"Collection check failed: {e}")
    
    # Step 5: Restore Backup
    print_step(5, "Restore Collection from Backup")
    try:
        print_info("Backup", backup_name)
        print_info("Target collection", RESTORED_COLLECTION_NAME)
        print_info("Status", "Restoring...")
        
        from backup_recovery import RestoreParams
        restore_params = RestoreParams(target_collection_name=RESTORED_COLLECTION_NAME)
        logger.info(f"Restore params: {restore_params}")
        
        result = await backup_manager.restore_backup(
            backup_id=backup_name,
            collection_name=latest_backup.collection_name,
            params=restore_params
        )
        logger.info(f"Restore result: {result.__dict__ if hasattr(result, '__dict__') else result}")
        
        if result.success:
            print_success("Backup restored successfully")
            print_info("Entities restored", result.rows_restored)
        else:
            print_error(f"Restore failed: {result.message}")
            conn_manager.close()
            return
    except Exception as e:
        print_error(f"Restore failed: {e}")
        conn_manager.close()
        return
    
    # Step 6: Verify Restored Collection
    print_step(6, "Verify Restored Collection")
    try:
        if not await coll_manager.has_collection(RESTORED_COLLECTION_NAME):
            print_error("Restored collection not found")
            conn_manager.close()
            return
        
        stats = await coll_manager.get_collection_stats(RESTORED_COLLECTION_NAME)
        logger.info(f"Restored collection stats: {stats}")
        
        print_success("Restored collection verified")
        print("\n  Restored Collection:")
        print(f"    Name: {RESTORED_COLLECTION_NAME}")
        print(f"    Entity count: {stats.row_count}")
        print(f"    Segments: {len(stats.segments)}")
        print()
        
        # Compare with original
        if await coll_manager.has_collection(COLLECTION_NAME):
            original_stats = await coll_manager.get_collection_stats(COLLECTION_NAME)
            logger.info(f"Original collection stats: {original_stats}")
            print_info("Original collection entities", original_stats.row_count)
            print_info("Restored collection entities", stats.row_count)
            
            if stats.row_count == original_stats.row_count:
                print_success("Entity counts match!")
            else:
                print_warning("Entity counts differ")
    except Exception as e:
        print_error(f"Verification failed: {e}")
    
    # Step 7: Load and Sample Data
    print_step(7, "Load Collection and Sample Data")
    try:
        from pymilvus import Collection
        collection = Collection(RESTORED_COLLECTION_NAME)
        
        print_info("Action", "Loading collection...")
        collection.load()
        print_success("Collection loaded")
        
        # Query sample data
        sample_results = collection.query(
            expr="pk >= 0",
            output_fields=["pk", "text", "value", "category"],
            limit=5
        )
        
        if sample_results:
            print("\n  Sample Restored Data:\n")
            for doc in sample_results:
                print(f"  ID: {doc['pk']}")
                print(f"    Text: {doc.get('text', 'N/A')[:50]}")
                print(f"    Value: {doc.get('value', 'N/A')}")
                print(f"    Category: {doc.get('category', 'N/A')}")
                print()
            
            print_success("Sample data retrieved")
        else:
            print_warning("No data found in restored collection")
    except Exception as e:
        print_error(f"Data sampling failed: {e}")
    
    # Step 8: Restore Statistics
    print_step(8, "Restore Statistics")
    try:
        logger.info(f"Getting backup info for: {backup_name}")
        metadata = await backup_manager.get_backup_info(
            backup_id=backup_name,
            collection_name=latest_backup.collection_name
        )
        logger.info(f"Backup metadata: {metadata.__dict__ if hasattr(metadata, '__dict__') else metadata}")
        
        stats = await coll_manager.get_collection_stats(RESTORED_COLLECTION_NAME)
        logger.info(f"Final collection stats: {stats}")
        
        print("\n  Restore Summary:")
        print(f"    Backup name: {backup_name}")
        print(f"    Backup created: {metadata.created_at}")
        print(f"    Backup size: {metadata.size_bytes / (1024*1024):.2f} MB")
        print(f"    Expected entities: {metadata.row_count}")
        print(f"    Restored entities: {stats.row_count}")
        
        recovery_rate = (stats.row_count / metadata.row_count * 100) if metadata.row_count > 0 else 0
        print(f"    Recovery rate: {recovery_rate:.1f}%")
        
        if recovery_rate >= 99.9:
            print_success("Full recovery achieved")
        elif recovery_rate >= 95.0:
            print_warning("Partial recovery (>95%)")
        else:
            print_error("Incomplete recovery (<95%)")
    except Exception as e:
        print_error(f"Statistics failed: {e}")
    
    # Step 9: Cleanup
    print_step(9, "Close Connection")
    try:
        conn_manager.close()
        print_success("Connection closed")
    except Exception as e:
        print_error(f"Cleanup failed: {e}")
    
    print_section("Example Completed")
    print("\nKey Takeaways:")
    print("  • Verify backup integrity before restoring")
    print("  • Check restored entity count")
    print("  • Test restored data by sampling")
    print("  • Compare with original collection")
    print("\nRestore Scenarios:")
    print("  • Disaster recovery")
    print("  • Data migration")
    print("  • Creating test environments")
    print("  • Rollback after errors")
    print("\nNote:")
    print(f"  • Restored collection: '{RESTORED_COLLECTION_NAME}'")
    print("  • Original collection unchanged")
    print("  • You can drop the restored collection when done")


if __name__ == "__main__":
    asyncio.run(main())

