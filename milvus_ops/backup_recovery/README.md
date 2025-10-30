# Backup Recovery Module for Milvus

A professional, enterprise-grade backup and recovery solution for Milvus vector databases with support for both local file system and Milvus native backups.

## Features

- **Multiple Storage Backends**: Local file system (Parquet-based) and Milvus native backup
- **Backup Types**: Full collection and partition-level backups
- **Data Integrity**: Checksum verification at multiple stages
- **Compression**: Optional compression with configurable levels
- **Progress Tracking**: Real-time progress monitoring with ETA estimation
- **Retention Policies**: Flexible retention based on count and age
- **Verification**: Multiple verification strategies (checksum, deep, quick)
- **Fault Tolerance**: Automatic retries and partial failure handling
- **Scalability**: Chunking for very large collections
- **Type Safety**: Full type hinting with Pydantic models

## Installation

This module is part of the `Milvus_Ops` package. Install the package as a dependency in your project:

```bash
pip install milvus-ops
```

## Quick Start

```python
import asyncio
from Milvus_Ops.connection_management import ConnectionManager
from Milvus_Ops.collection_operations import CollectionManager
from Milvus_Ops.backup_recovery import (
    BackupManager,
    BackupRecoveryConfig,
    BackupParams,
    BackupType
)

async def main():
    # Initialize managers
    connection_manager = ConnectionManager()
    collection_manager = CollectionManager(connection_manager)
    
    # Configure backup settings
    config = BackupRecoveryConfig(
        local_backup_root_path="/path/to/backups",
        compression_enabled=True
    )
    
    # Create backup manager
    backup_manager = BackupManager(
        connection_manager=connection_manager,
        collection_manager=collection_manager,
        config=config
    )
    
    # Create a backup
    result = await backup_manager.create_backup(
        collection_name="my_collection",
        params=BackupParams(
            backup_type=BackupType.FULL_COLLECTION,
            compression_enabled=True
        )
    )
    
    if result.success:
        print(f"Backup created successfully: {result.backup_id}")
        print(f"Size: {result.size_mb:.2f} MB")
    else:
        print(f"Backup failed: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

The `BackupRecoveryConfig` class provides centralized configuration:

```python
config = BackupRecoveryConfig(
    # Storage settings
    default_storage_type=BackupStorageType.LOCAL_FILE,  # or MILVUS_NATIVE
    local_backup_root_path="./milvus_backups",
    milvus_backup_bucket=None,  # For Milvus native backup
    
    # Performance settings
    default_chunk_size_mb=256,
    max_concurrent_chunks=4,
    compression_enabled=True,
    compression_level=6,  # 1-9 (higher = better compression but slower)
    
    # Reliability settings
    enable_checksum_verification=True,
    checksum_algorithm=ChecksumAlgorithm.SHA256,  # or MD5, BLAKE2B
    auto_verify_after_backup=True,
    auto_verify_before_restore=True,
    
    # Timeout settings
    default_backup_timeout=3600.0,  # seconds
    default_restore_timeout=3600.0,  # seconds
    
    # Retention settings
    retention_count=10,  # Keep 10 most recent backups
    retention_days=30,   # Keep backups for 30 days
    min_backups_to_keep=3  # Always keep at least 3 backups
)
```

## Core Operations

### Creating Backups

#### Full Collection Backup

Use when you need to back up an entire collection:

```python
result = await backup_manager.create_backup(
    collection_name="my_collection",
    params=BackupParams(
        backup_type=BackupType.FULL_COLLECTION,
        compression_enabled=True,
        include_indexes=True  # Include index definitions
    )
)
```

#### Partition Backup

Use when you only need specific partitions:

```python
result = await backup_manager.create_backup(
    collection_name="my_collection",
    params=BackupParams(
        backup_type=BackupType.PARTITION,
        partition_names=["partition_2024_01", "partition_2024_02"],
        compression_enabled=True
    )
)
```

### Listing Backups

List all available backups for a collection:

```python
backups = await backup_manager.list_backups(collection_name="my_collection")

for backup in backups:
    print(f"Backup: {backup.backup_name}")
    print(f"  ID: {backup.backup_id}")
    print(f"  Created: {backup.created_at}")
    print(f"  Size: {backup.size_mb:.2f} MB")
    print(f"  Type: {backup.backup_type.value}")
    print(f"  State: {backup.state.value}")
```

### Verifying Backups

Verify backup integrity:

```python
result = await backup_manager.verify_backup(
    backup_id=backup_id,
    collection_name="my_collection",
    params=VerificationParams(
        verification_type=VerificationType.CHECKSUM,
        fail_fast=True
    )
)

if result.success:
    print("Verification passed!")
else:
    print(f"Verification failed: {result.errors}")
```

### Restoring Backups

Restore a backup:

```python
result = await backup_manager.restore_backup(
    backup_id=backup_id,
    collection_name="my_collection",
    params=RestoreParams(
        target_collection_name="my_collection_restored",  # Optional rename
        verify_before_restore=True,
        drop_existing=False,
        load_after_restore=True
    )
)

if result.success:
    print(f"Restored {result.rows_restored:,} rows successfully")
else:
    print(f"Restore failed: {result.error_message}")
```

### Deleting Backups

Delete a specific backup:

```python
success = await backup_manager.delete_backup(
    backup_id=backup_id,
    collection_name="my_collection"
)

if success:
    print("Backup deleted successfully")
```

### Applying Retention Policies

Clean up old backups based on retention policy:

```python
# Dry run first to see what would be deleted
result = await backup_manager.apply_retention_policy(
    collection_name="my_collection",
    dry_run=True
)

print(f"Would delete {result['backups_to_delete']} backups")
print(f"Would free {result['storage_to_free_gb']:.2f} GB")

# Actual deletion
if result['backups_to_delete'] > 0:
    result = await backup_manager.apply_retention_policy(
        collection_name="my_collection",
        dry_run=False
    )
    print(f"Deleted {len(result['deleted_backup_ids'])} backups")
```

## Storage Backends

### Local File System Backend

The local backend stores backups in a structured directory hierarchy on the local file system, using Parquet format for data storage.

**When to use:**
- When you need direct file access to backup data
- For smaller deployments without cloud storage
- When you want to manually inspect backup contents
- For easier integration with existing backup systems

**Directory structure:**
```
backup_root/
├── collection_name/
│   ├── backup_id/
│   │   ├── metadata.json      # Backup metadata
│   │   ├── schema.json        # Collection schema
│   │   ├── data/              # Data files
│   │   │   ├── partition_1.parquet
│   │   │   ├── partition_2.parquet
│   │   ├── checksums.json     # File checksums
│   │   └── indexes/           # Index definitions (optional)
```

### Milvus Native Backend

The Milvus native backend uses the Milvus-Backup tool for backup and restore operations.

**When to use:**
- When you need tighter integration with Milvus
- For compatibility with Milvus cloud deployments
- When you want to leverage Milvus's native backup capabilities
- For more efficient handling of very large collections

**Note:** Requires the `milvus-backup` tool to be installed and accessible.

## Backup Types

### Full Collection Backup

Creates a complete backup of the entire collection, including all partitions and optionally all indexes.

**When to use:**
- For comprehensive backups of your data
- When you need to restore the entire collection
- For regular scheduled backups
- When you need a consistent snapshot of the entire collection

### Partition Backup

Creates a backup of specific partitions within a collection.

**When to use:**
- When you only need to back up recent data (e.g., time-partitioned collections)
- To reduce backup size and time for very large collections
- For more granular backup strategies
- When different partitions have different backup requirements

## Verification Strategies

### Checksum Verification

Fast verification that ensures file integrity using cryptographic checksums.

**When to use:**
- For routine verification after every backup
- For quick integrity checks before restore
- When you need to verify many backups efficiently

### Deep Verification

Comprehensive verification that restores data to a temporary collection and compares it with the original.

**When to use:**
- For critical backups that require absolute certainty
- Periodically (e.g., weekly) for important collections
- Before major system changes or migrations
- When corruption is suspected

### Quick Verification

Sampling-based verification that checks a percentage of the data.

**When to use:**
- For very large backups where full verification is impractical
- As a compromise between speed and thoroughness
- For routine checks of non-critical backups

## Compression

The module supports compression to reduce backup size:

```python
params = BackupParams(
    backup_type=BackupType.FULL_COLLECTION,
    compression_enabled=True,
    compression_level=9  # Maximum compression (1-9)
)
```

**When to use:**
- **High compression (7-9)**: When storage space is limited and backup speed is less critical
- **Medium compression (4-6)**: For balanced performance and size reduction
- **Low compression (1-3)**: When backup speed is critical but some size reduction is desired
- **No compression**: When storage space is plentiful and maximum backup speed is required

## Retention Policies

The module supports flexible retention policies:

```python
config = BackupRecoveryConfig(
    retention_count=10,  # Keep 10 most recent backups
    retention_days=30,   # Keep backups for 30 days
    min_backups_to_keep=3  # Always keep at least 3 backups
)
```

**Policy behaviors:**
- Keeps backups that satisfy EITHER the count OR days condition
- Always keeps at least `min_backups_to_keep` backups, regardless of age
- Can be applied per collection or globally

## Error Handling

The module provides a granular exception hierarchy for precise error handling:

```python
try:
    result = await backup_manager.create_backup(...)
except BackupError as e:
    print(f"Backup failed: {e}")
    print(f"Collection: {e.collection_name}")
except InsufficientStorageError as e:
    print(f"Not enough disk space: {e}")
    print(f"Required: {e.required_gb:.2f} GB")
    print(f"Available: {e.available_gb:.2f} GB")
except PartitionNotFoundError as e:
    print(f"Partition not found: {e.partition_name}")
    print(f"Available partitions: {e.available_partitions}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

### For Large Collections

1. **Use chunking**: Set appropriate `default_chunk_size_mb` in config
2. **Use compression wisely**: Higher compression levels may slow down backups
3. **Consider partition backups**: Back up partitions individually for better manageability
4. **Monitor progress**: Use the progress tracking capabilities
5. **Verify selectively**: Use quick verification for routine checks, deep for critical data

### For Production Deployments

1. **Schedule regular backups**: Implement a backup rotation strategy
2. **Verify backups periodically**: Enable `auto_verify_after_backup`
3. **Apply retention policies**: Avoid unbounded storage growth
4. **Monitor backup sizes**: Track growth over time
5. **Test restore procedures**: Regularly verify that backups can be restored

### For Critical Data

1. **Use multiple storage backends**: Back up to both local and Milvus native
2. **Implement off-site storage**: Copy local backups to remote storage
3. **Use deep verification**: Periodically perform deep verification
4. **Keep longer retention**: Increase `min_backups_to_keep` and `retention_days`
5. **Document backup procedures**: Maintain clear recovery procedures

## Advanced Usage

### Custom Checksum Algorithm

```python
from Milvus_Ops.backup_recovery import ChecksumAlgorithm

config = BackupRecoveryConfig(
    checksum_algorithm=ChecksumAlgorithm.BLAKE2B  # Faster and more secure than SHA256
)
```

### Manual Checksum Verification

```python
from Milvus_Ops.backup_recovery import ChecksumCalculator

calculator = ChecksumCalculator(ChecksumAlgorithm.SHA256)
checksum = await calculator.calculate_file_checksum("/path/to/file")
is_valid = await calculator.verify_file_checksum(
    "/path/to/file",
    expected_checksum=checksum
)
```

### Custom Compression

```python
from Milvus_Ops.backup_recovery import CompressionHandler

handler = CompressionHandler(compression_level=9, prefer_zstd=True)
compressed_path = await handler.compress_file("/path/to/file")
decompressed_path = await handler.decompress_file(compressed_path)
```

## Integration Examples

### Regular Backup Schedule

```python
import asyncio
import schedule
import time
from datetime import datetime

async def perform_backup():
    result = await backup_manager.create_backup(
        collection_name="my_collection",
        params=BackupParams(
            backup_name=f"daily_{datetime.now().strftime('%Y%m%d')}",
            backup_type=BackupType.FULL_COLLECTION
        )
    )
    print(f"Backup completed: {result.success}")
    
    # Apply retention policy
    await backup_manager.apply_retention_policy(
        collection_name="my_collection",
        dry_run=False
    )

def scheduled_backup():
    asyncio.run(perform_backup())

# Schedule daily backup at 2 AM
schedule.every().day.at("02:00").do(scheduled_backup)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Backup Before Migration

```python
async def migrate_collection(collection_name, new_schema):
    # Create backup before migration
    backup_result = await backup_manager.create_backup(
        collection_name=collection_name,
        params=BackupParams(
            backup_name=f"pre_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            backup_type=BackupType.FULL_COLLECTION,
            compression_enabled=True
        )
    )
    
    if not backup_result.success:
        print(f"Pre-migration backup failed: {backup_result.error_message}")
        return False
    
    print(f"Pre-migration backup created: {backup_result.backup_id}")
    
    # Verify backup integrity
    verify_result = await backup_manager.verify_backup(
        backup_id=backup_result.backup_id,
        collection_name=collection_name
    )
    
    if not verify_result.success:
        print(f"Backup verification failed: {verify_result.errors}")
        return False
    
    # Proceed with migration...
    # ...
    
    return True
```

## Troubleshooting

### Backup Creation Fails

1. **Check storage space**: Ensure sufficient disk space is available
2. **Check permissions**: Verify write permissions on backup directory
3. **Check collection**: Ensure collection exists and is accessible
4. **Check partitions**: Verify partition names if using partition backup

### Restore Fails

1. **Check backup integrity**: Run verification on the backup
2. **Check target collection**: Ensure no conflicts with existing collections
3. **Check schema compatibility**: Verify schema compatibility if restoring to existing collection
4. **Check error message**: Look for specific error details in the result

### Verification Fails

1. **Check checksum file**: Ensure checksums.json exists and is valid
2. **Check for corruption**: Verify backup files are not corrupted
3. **Try different verification**: If checksum fails, try quick verification
4. **Check storage**: Ensure backup storage is healthy

## License

This module is part of the Milvus_Ops package and is licensed under [LICENSE TERMS].

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for details.
