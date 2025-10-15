"""
Verify Backup Example

Demonstrates how to verify backup integrity and validity.
Shows checksum verification, metadata checks, and backup health assessment.
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
logger = logging.getLogger("verify_example")

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from connection_management import ConnectionManager
from collection_operations import CollectionManager
from backup_recovery import BackupManager, BackupRecoveryConfig, BackupStorageType, VerificationParams
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


BACKUP_DIR = "./backups"


async def main():
    """Main function to demonstrate backup verification."""
    print_section("Verify Backup Example")
    
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
    
    # Step 2: Check Backup Directory
    print_step(2, "Check Backup Directory")
    try:
        if not os.path.exists(BACKUP_DIR):
            print_error(f"Backup directory '{BACKUP_DIR}' does not exist")
            print_info("Hint", "Run create_backup.py first")
            conn_manager.close()
            return
        
        print_success(f"Backup directory found: {BACKUP_DIR}")
        
        # List files in backup directory
        files = os.listdir(BACKUP_DIR)
        print_info("Files found", len(files))
    except Exception as e:
        print_error(f"Directory check failed: {e}")
        conn_manager.close()
        return
    
    # Step 3: List All Backups
    print_step(3, "List All Available Backups")
    try:
        logger.info("Listing all backups")
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
            print(f"     Size: {backup.size_bytes / (1024*1024):.2f} MB")
            print(f"     Entities: {backup.row_count}")
            print(f"     Compressed: {backup.compression_enabled}")
            logger.info(f"Backup details: {backup.__dict__ if hasattr(backup, '__dict__') else backup}")
            print()
    except Exception as e:
        print_error(f"Listing backups failed: {e}")
        conn_manager.close()
        return
    
    # Step 4: Verify Each Backup
    print_step(4, "Verify Integrity of All Backups")
    try:
        verification_results = []
        
        for backup in backups:
            print(f"\n  Verifying: {backup.backup_name}")
            
            try:
                logger.info(f"Verifying backup: {backup.backup_name} (ID: {backup.backup_id})")
                is_valid = await backup_manager.verify_backup(
                    backup_id=backup.backup_id,
                    collection_name=backup.collection_name
                )
                logger.info(f"Verification result: {is_valid}")
                
                verification_results.append({
                    'name': backup.backup_name,
                    'valid': is_valid,
                    'size_mb': backup.size_bytes / (1024*1024)
                })
                
                if is_valid:
                    print_success(f"  ✓ Valid")
                else:
                    print_error(f"  ✗ Invalid")
            except Exception as e:
                print_error(f"  ✗ Verification failed: {e}")
                verification_results.append({
                    'name': backup.backup_name,
                    'valid': False,
                    'error': str(e)
                })
        
        # Summary
        valid_count = sum(1 for r in verification_results if r.get('valid', False))
        print(f"\n  Verification Summary:")
        print(f"    Total backups: {len(verification_results)}")
        print(f"    Valid: {valid_count}")
        print(f"    Invalid: {len(verification_results) - valid_count}")
        
        if valid_count == len(verification_results):
            print_success("All backups are valid")
        else:
            print_warning(f"{len(verification_results) - valid_count} backup(s) failed verification")
    except Exception as e:
        print_error(f"Verification process failed: {e}")
    
    # Step 5: Detailed Metadata Check
    print_step(5, "Detailed Metadata Inspection")
    try:
        # Select first backup for detailed inspection
        backup_to_inspect = backups[0]
        backup_name = backup_to_inspect.backup_name
        
        print_info("Inspecting", backup_name)
        
        logger.info(f"Getting detailed metadata for backup: {backup_name}")
        metadata = await backup_manager.get_backup_info(
            backup_id=backup_name,
            collection_name=backup_to_inspect.collection_name
        )
        logger.info(f"Metadata: {metadata.__dict__ if hasattr(metadata, '__dict__') else metadata}")
        
        if metadata:
            print("\n  Detailed Metadata:\n")
            print(f"    Backup Name: {metadata.backup_name}")
            print(f"    Collection Name: {metadata.collection_name}")
            print(f"    Created At: {metadata.created_at}")
            print(f"    Entities Count: {metadata.row_count}")
            print(f"    Size (bytes): {metadata.size_bytes:,}")
            print(f"    Size (MB): {metadata.size_bytes / (1024*1024):.2f}")
            print(f"    Compressed: {metadata.compression_enabled}")
            print(f"    Checksum: {metadata.checksum}")
            
            if hasattr(metadata, 'schema'):
                print(f"    Schema Fields: {len(metadata.schema.fields) if metadata.schema else 'N/A'}")
            
            print()
            print_success("Metadata inspection complete")
        else:
            print_error("Could not retrieve metadata")
    except Exception as e:
        print_error(f"Metadata inspection failed: {e}")
    
    # Step 6: Check File System Integrity
    print_step(6, "Check File System Integrity")
    try:
        for backup in backups:
            backup_path = os.path.join(BACKUP_DIR, backup.backup_name)
            
            if os.path.exists(backup_path):
                file_size = os.path.getsize(backup_path)
                
                # Check if file size matches metadata
                if abs(file_size - backup.size_bytes) < 1024:  # Allow 1KB tolerance
                    status = "✓"
                else:
                    status = "✗ Size mismatch"
                
                print(f"  {backup.backup_name}: {status}")
            else:
                print(f"  {backup.backup_name}: ✗ File not found")
        
        print_success("File system check complete")
    except Exception as e:
        print_error(f"File system check failed: {e}")
    
    # Step 7: Backup Health Assessment
    print_step(7, "Backup Health Assessment")
    try:
        print("\n  Health Assessment:\n")
        
        total_size = sum(b.size_bytes for b in backups)
        avg_size = total_size / len(backups) if backups else 0
        
        print(f"    Total backups: {len(backups)}")
        print(f"    Total storage used: {total_size / (1024*1024):.2f} MB")
        print(f"    Average backup size: {avg_size / (1024*1024):.2f} MB")
        
        # Check for old backups (older than 30 days)
        from datetime import datetime, timedelta
        now = datetime.now()
        old_backups = []
        
        for backup in backups:
            try:
                created = backup.created_at
                age_days = (now - created).days
                if age_days > 30:
                    old_backups.append(backup.backup_name)
            except:
                pass
        
        if old_backups:
            print_warning(f"    Old backups (>30 days): {len(old_backups)}")
            print(f"      Consider implementing retention policy")
        else:
            print_info("    All backups are recent (<30 days)", "")
        
        print_success("Health assessment complete")
    except Exception as e:
        print_error(f"Health assessment failed: {e}")
    
    # Step 8: Recommendations
    print_step(8, "Backup Recommendations")
    try:
        print("\n  Recommendations:\n")
        
        # Based on verification results
        if valid_count < len(backups):
            print("    ⚠  CRITICAL: Some backups are invalid")
            print("       → Create new backups immediately")
            print("       → Investigate corruption causes")
        
        # Based on backup count
        if len(backups) < 3:
            print("    ⚠  WARNING: Low backup count")
            print("       → Maintain at least 3 recent backups")
            print("       → Implement automated backup schedule")
        
        # Based on total size
        total_size_gb = total_size / (1024*1024*1024)
        if total_size_gb > 10:
            print("    ℹ  INFO: High storage usage")
            print("       → Consider compression tuning")
            print("       → Implement retention policy")
        
        # General recommendations
        print("\n    Best Practices:")
        print("      ✓ Test restore procedures regularly")
        print("      ✓ Store backups in multiple locations")
        print("      ✓ Monitor backup age and validity")
        print("      ✓ Document recovery procedures")
        
        print_success("Recommendations generated")
    except Exception as e:
        print_error(f"Recommendations failed: {e}")
    
    # Step 9: Cleanup
    print_step(9, "Close Connection")
    try:
        conn_manager.close()
        print_success("Connection closed")
    except Exception as e:
        print_error(f"Cleanup failed: {e}")
    
    print_section("Example Completed")
    print("\nKey Takeaways:")
    print("  • Regularly verify backup integrity")
    print("  • Monitor backup age and size")
    print("  • Test restore procedures")
    print("  • Implement retention policies")
    print("\nVerification Checks:")
    print("  • Checksum validation")
    print("  • Metadata consistency")
    print("  • File system integrity")
    print("  • Size verification")
    print("\nMaintenance Tasks:")
    print("  • Remove old backups")
    print("  • Verify backup validity weekly")
    print("  • Test restore monthly")
    print("  • Update documentation")


if __name__ == "__main__":
    asyncio.run(main())

