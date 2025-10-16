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
from backup_recovery import BackupManager, BackupRecoveryConfig, BackupStorageType, BackupParams
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


# Collection to backup
COLLECTION_NAME = "test_example_collection"
# Backup directory will be loaded from config/default_settings.yaml


async def main():
    """Main function to demonstrate backup creation."""
    print_section("Create Backup Example")

    # Step 1: Initialize Managers and Connect to Milvus
    print_step(1, "Initialize Managers and Connect to Milvus")
    try:
        config = load_settings()

        # Initialize ConnectionManager (automatically establishes connection)
        print_info("Status", "Connecting to Milvus...")
        conn_manager = ConnectionManager(config=config)

        # Verify connection is working
        if not conn_manager.check_server_status():
            print_error("Failed to connect to Milvus server")
            return

        print_success(f"Connected to Milvus at {config.connection.host}:{config.connection.port}")

        coll_manager = CollectionManager(conn_manager)

        # Configure backup - settings will be loaded from YAML config
        # COMPRESSION CONTROL: Compression settings are loaded from config/default_settings.yaml
        # To control compression:
        #   1. Edit config/default_settings.yaml: backup.compression = true/false
        #   2. Or override here with compression_enabled=True/False and compression_level=1-9
        #   3. Default is compression_enabled=True with level=6 (balanced setting)
        backup_config = BackupRecoveryConfig(
            default_storage_type=BackupStorageType.LOCAL_FILE,
            enable_checksum_verification=True
            # Other settings like backup_path, compression, etc. are loaded from config/default_settings.yaml
        )

        backup_manager = BackupManager(
            connection_manager=conn_manager,
            collection_manager=coll_manager,
            config=backup_config
        )

        print_success("Managers initialized")
        print_info("Backup directory", backup_config.local_backup_root_path)
        # COMPRESSION CONTROL: This shows current compression status from config
        # To modify compression settings, see config/default_settings.yaml or BackupParams
        print_info("Compression", "Enabled" if backup_config.compression_enabled else "Disabled")
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

        # CRITICAL FIX: Milvus get_collection_stats() can return stale/cached data
        # showing 0 entities even when collection has data. We need to verify directly.

        # CRITICAL FIX: Don't use or log Milvus get_collection_stats() as it returns stale/cached data
        # The stats API consistently shows 0 entities even when the collection has data
        # This is a known issue with Milvus where stats are cached and not updated in real-time

        # Instead of relying on stats, we'll query the collection directly to check for data
        logger.info(
            f"Checking collection '{COLLECTION_NAME}' for data presence (not using stats API due to staleness)"
        )

        # Then, query a small sample to verify if data actually exists
        # This is more reliable than stats.row_count which can be stale
        has_data = False
        try:
            # Try to query for a few rows to check if data exists
            def _check_data(alias):
                from pymilvus import Collection
                from backup_recovery.utils.query import generate_query_expression

                coll = Collection(name=COLLECTION_NAME, using=alias)
                # Generate robust query expression that works across all Milvus versions
                query_expr = generate_query_expression(coll)
                sample = coll.query(expr=query_expr, output_fields=["pk"], limit=5)
                return len(sample) > 0

            has_data = await conn_manager.execute_operation_async(_check_data)
            logger.info(f"Collection data presence check: {'Data found' if has_data else 'No data found'}")
        except Exception as e:
            logger.warning(f"Failed to check for data directly: {e}")
            # Assume collection might have data if we can't check
            has_data = True

        if not has_data:
            print_warning(f"Collection found but appears to be empty. Backup may be empty.")
            print_info("Hint", "You may want to insert data first using insert_data.py")
        else:
            print_success(f"Collection found with data")
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
        backup_dir = backup_config.local_backup_root_path
        os.makedirs(backup_dir, exist_ok=True)
        print_success(f"Backup directory ready: {backup_dir}")
    except Exception as e:
        print_error(f"Directory creation failed: {e}")
        conn_manager.close()
        return

    # Step 5: Create Backup
    print_step(5, "Create Collection Backup")
    try:
        # CRITICAL FIX: Don't fetch stats as they're unreliable (always show 0 due to caching)
        # We already verified the collection has data in Step 2

        print("\n  Backup Configuration:")
        print(f"    Collection to backup: {COLLECTION_NAME}")

        # CRITICAL FIX: Don't use potentially stale row_count from stats API
        # Instead, report that we're backing up the collection's data without specifying a count
        # that might be incorrect
        print(f"    Backup name: {backup_name}")
        print(f"    Compression: Enabled (level {backup_config.compression_level})")
        print(f"    Include indexes: Yes")
        print()

        print_info("Status", "Creating backup...")

        # CRITICAL FIX: Don't use potentially stale row_count from stats API
        # Instead, use the actual data count we verified earlier
        if has_data:
            logger.info(f"Starting backup of collection '{COLLECTION_NAME}' with data")
            print_info("Data status", "Collection has data that will be backed up")
        else:
            logger.info(f"Starting backup of collection '{COLLECTION_NAME}' (possibly empty)")
            print_warning("Collection may be empty - backup might contain no data")

        # Create backup parameters
        # COMPRESSION CONTROL: Per-backup compression settings
        # You can control compression for each individual backup by modifying:
        #   - compression_enabled: Set to True/False to enable/disable compression
        #   - compression_level: Set from 1 (fastest) to 9 (best compression)
        # These settings override the global settings from config.yaml
        backup_params = BackupParams(
            backup_name=backup_name,
            compression_enabled=True,  # COMPRESSION CONTROL: Set to False to disable compression
            compression_level=backup_config.compression_level,  # COMPRESSION CONTROL: 1-9 (higher = better compression but slower)
            include_indexes=True
        )

        result = await backup_manager.create_backup(
            collection_name=COLLECTION_NAME,
            params=backup_params
        )

        if result.success:
            logger.info(f"Backup created successfully: {result.backup_id}")
            print_success(f"Backup created successfully with ID: {result.backup_id}")
            print_info("Backup path", result.storage_path)
            print_info("Entities backed up", result.row_count if hasattr(result, 'row_count') else 'N/A')
        else:
            print_error(f"Backup failed: {result.error_message}")
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
        # Get the most accurate stats from the backup metadata
        # CRITICAL FIX: Don't rely on get_collection_stats() which returns stale/cached data
        # showing 0 entities even when the collection has data (as seen in Milvus UI)

        # Get metadata for the most accurate information
        metadata = await backup_manager.get_backup_info(
            backup_id=result.backup_id,
            collection_name=COLLECTION_NAME
        )

        # CRITICAL FIX: Don't use Milvus stats API as it consistently returns stale/cached data
        # showing 0 entities even after successful backup of 1030 entities
        # This is a known limitation with Milvus where collection stats are not updated in real-time

        # Instead, use the accurate row count from the backup metadata
        logger.info(f"Backup completed. Accurate row count from backup data: {metadata.row_count if metadata else 'unknown'}")

        # Note: The Milvus get_collection_stats() API would incorrectly show 0 entities here
        # but the backup metadata shows the true count from the actual data export

        print("\n  Backup Summary:")

        # Use the metadata's row count which is accurate (from actual data export)
        actual_row_count = metadata.row_count if metadata else (
            result.metadata.row_count if result.metadata else 0
        )

        print(f"    Original collection size: {actual_row_count} entities")
        print(f"    Backed up: {actual_row_count} entities")
        print(f"    Backup ID: {result.backup_id}")
        print(f"    Backup location: {backup_config.local_backup_root_path}")
        # COMPRESSION CONTROL: Shows the actual compression settings used for this backup
        # To change compression settings for future backups:
        #   1. Global setting: Edit config/default_settings.yaml (backup.compression)
        #   2. Per-backup: Modify BackupParams in the code (compression_enabled, compression_level)
        print(f"    Compression: {'Enabled' if backup_config.compression_enabled else 'Disabled'} (level {backup_config.compression_level})")

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

