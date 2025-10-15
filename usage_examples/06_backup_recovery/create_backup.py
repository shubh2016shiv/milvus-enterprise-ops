"""
Create Backup Example

Demonstrates how to create a backup of a Milvus collection.
Shows backup configuration, execution, and verification.
"""

import sys
import os
import asyncio
import logging
from datetime import datetime

from loguru import logger

# Remove existing logging configuration
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger("backup_example")

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


COLLECTION_NAME = "test_example_collection"
BACKUP_DIR = "./backups"


async def main():
    """Main function to demonstrate backup creation."""
    print_section("Create Backup Example")
    
    # Step 1: Initialize Managers
    print_step(1, "Initialize Managers")
    try:
        config = load_settings()
        conn_manager = ConnectionManager(config=config)
        coll_manager = CollectionManager(conn_manager)
        
        # Configure backup
        backup_config = BackupRecoveryConfig(
            default_storage_type=BackupStorageType.LOCAL_FILE,
            local_backup_root_path=BACKUP_DIR,
            compression_enabled=True,
            enable_checksum_verification=True,
            compression_level=6
        )
        
        backup_manager = BackupManager(
            connection_manager=conn_manager,
            collection_manager=coll_manager,
            config=backup_config
        )
        
        print_success("Managers initialized")
        print_info("Backup directory", BACKUP_DIR)
        print_info("Compression", "Enabled")
        print_info("Checksum", "Enabled")
    except Exception as e:
        print_error(f"Initialization failed: {e}")
        return
    
    # Step 2: Verify Collection Exists
    print_step(2, "Verify Collection Exists")
    try:
        if not await coll_manager.has_collection(COLLECTION_NAME):
            print_error(f"Collection '{COLLECTION_NAME}' does not exist")
            print_info("Hint", "Run create_collection.py and insert_data.py first")
            conn_manager.close()
            return
        
        stats = await coll_manager.get_collection_stats(COLLECTION_NAME)
        logger.info(f"Collection stats: {stats}")
        print_success(f"Collection found with {stats.row_count} entities")
    except Exception as e:
        print_error(f"Collection check failed: {e}")
        conn_manager.close()
        return
    
    # Step 3: Prepare Backup Metadata
    print_step(3, "Prepare Backup Metadata")
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{COLLECTION_NAME}_backup_{timestamp}"
        
        print_info("Backup name", backup_name)
        print_info("Timestamp", timestamp)
        print_success("Backup metadata prepared")
    except Exception as e:
        print_error(f"Metadata preparation failed: {e}")
        conn_manager.close()
        return
    
    # Step 4: Create Backup Directory
    print_step(4, "Create Backup Directory")
    try:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        print_success(f"Backup directory ready: {BACKUP_DIR}")
    except Exception as e:
        print_error(f"Directory creation failed: {e}")
        conn_manager.close()
        return
    
    # Step 5: Create Backup
    print_step(5, "Create Collection Backup")
    try:
        print_info("Status", "Creating backup...")
        print_info("Collection", COLLECTION_NAME)
        
        result = await backup_manager.create_backup(
            collection_name=COLLECTION_NAME,
            backup_name=backup_name
        )
        
        if result.success:
            logger.info(f"Backup created successfully: {result.backup_id}")
            print_success(f"Backup created successfully with ID: {result.backup_id}")
            print_info("Backup path", result.storage_path)
            print_info("Entities backed up", result.row_count if hasattr(result, 'row_count') else 'N/A')
        else:
            print_error(f"Backup failed: {result.message}")
            conn_manager.close()
            return
    except Exception as e:
        print_error(f"Backup creation failed: {e}")
        conn_manager.close()
        return
    
    # Step 6: Verify Backup
    print_step(6, "Verify Backup Integrity")
    try:
        print_info("Action", "Verifying backup files...")
        
        # Check backup metadata
        logger.info(f"Getting backup info for ID: {result.backup_id}")
        metadata = await backup_manager.get_backup_info(
            backup_id=result.backup_id,
            collection_name=COLLECTION_NAME
        )
        logger.info(f"Backup metadata for ID {result.backup_id}: {metadata.model_dump_json(indent=2) if metadata else 'None'}")
        
        if metadata:
            print_success("Backup metadata verified")
            print("\n  Backup Metadata:")
            print(f"    Name: {metadata.backup_name}")
            print(f"    ID: {metadata.backup_id}")
            print(f"    Collection: {metadata.collection_name}")
            print(f"    Created: {metadata.created_at}")
            print(f"    Entities: {metadata.row_count}")
            print(f"    Size: {metadata.size_bytes / (1024*1024):.2f} MB")
            print(f"    Compressed: {metadata.compression_enabled}")
            print(f"    Checksum: {metadata.checksum[:16]}...")
            print()
        else:
            print_error("Could not retrieve backup metadata")
    except Exception as e:
        print_error(f"Verification failed: {e}")
    
    # Step 7: List All Backups
    print_step(7, "List All Available Backups")
    try:
        logger.info("Listing all backups")
        backups = await backup_manager.list_backups()
        logger.info(f"Found {len(backups)} backups: {[b.backup_id for b in backups]}")
        
        if backups:
            print_success(f"Found {len(backups)} backup(s)")
            print("\n  Available Backups:\n")
            for i, backup in enumerate(backups, 1):
                print(f"  {i}. {backup.backup_name}")
                print(f"     ID: {backup.backup_id}")
                print(f"     Collection: {backup.collection_name}")
                print(f"     Created: {backup.created_at}")
                print(f"     Size: {backup.size_bytes / (1024*1024):.2f} MB")
                print(f"     Entities: {backup.row_count}")
                print()
        else:
            print_info("Status", "No backups found")
    except Exception as e:
        print_error(f"Listing backups failed: {e}")
    
    # Step 8: Backup Statistics
    print_step(8, "Backup Statistics")
    try:
        stats = await coll_manager.get_collection_stats(COLLECTION_NAME)
        logger.info(f"Collection stats for summary: {stats}")
        
        print("\n  Backup Summary:")
        print(f"    Original collection size: {stats.row_count} entities")
        print(f"    Backed up: {result.row_count if hasattr(result, 'row_count') else 'N/A'} entities")
        print(f"    Backup ID: {result.backup_id}")
        print(f"    Backup location: {BACKUP_DIR}")
        print(f"    Compression: Enabled (level {backup_config.compression_level})")
        
        # Calculate compression ratio if available
        # Note: result.backup_path is not a direct file path, but a directory
        # We need to find the actual data files within the backup directory to get size
        if result.storage_path and os.path.exists(result.storage_path):
            total_backup_size = 0
            for dirpath, dirnames, filenames in os.walk(result.storage_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_backup_size += os.path.getsize(fp)
            print(f"    Total backup file size: {total_backup_size / (1024*1024):.2f} MB")
        
        print_success("Statistics calculated")
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
    print("  • Regular backups protect against data loss")
    print("  • Include metadata with backups")
    print("  • Enable compression to save storage")
    print("  • Use checksums to verify integrity")
    print("\nBackup Best Practices:")
    print("  • Schedule regular automated backups")
    print("  • Test restore procedures periodically")
    print("  • Store backups in multiple locations")
    print("  • Implement retention policies")
    print("\nNext Steps:")
    print("  • Run restore_backup.py to test recovery")
    print("  • Run verify_backup.py to check integrity")


if __name__ == "__main__":
    asyncio.run(main())

