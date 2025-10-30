# Partition Operations Module for Milvus

A professional, production-ready partition management module for Milvus vector databases. The Partition Operations module provides an intuitive interface for creating, managing, and monitoring partitions with built-in concurrency control, robust error handling, and scalable operations.

## Overview

The Partition Operations module is designed to simplify partition operations in Milvus while maintaining enterprise-grade reliability and performance. It handles the complexity of partition lifecycle management, providing developers with a clean, easy-to-use API that abstracts away low-level details while offering full control when needed.

Partitions in Milvus allow you to organize your vector data into logical segments within a collection. This organization enables more efficient queries by allowing you to target specific subsets of data, improves load balancing, and provides better resource management. The Partition Operations module makes working with partitions straightforward, whether you're creating a single partition or managing hundreds across multiple collections.

## Key Features

**Simple and Intuitive**: The API is designed for ease of use, with sensible defaults and clear method names. You can perform common operations with minimal code while still having access to advanced features when needed.

**Robust and Reliable**: Built-in concurrency control prevents race conditions during partition operations. Automatic retries through the connection manager ensure operations complete successfully even in the face of transient failures.

**Type-Safe**: Full Pydantic v2 integration provides compile-time type checking and runtime validation, reducing bugs and improving code quality.

**Asynchronous**: All operations are async/await compatible, enabling high-performance applications that can handle multiple partition operations concurrently without blocking.

**Production-Ready**: Comprehensive error handling with specific exception types, configurable timeouts, and detailed logging make this module suitable for production deployments.

## Architecture

The module follows a clean, layered architecture:

```
partition_manager/
├── __init__.py              # Module exports and public API
├── manager.py               # PartitionManager class - core functionality
├── validator.py             # Partition name validation logic
├── config.py                # Configuration management
├── exceptions.py            # Custom exception hierarchy
└── models/
    └── entities.py          # Pydantic models for data structures
```

The `PartitionManager` class is the main entry point for all partition operations. It uses the `ConnectionManager` from your connection management module to handle communication with Milvus, ensuring robust connection handling and automatic retries. The validator ensures partition names meet Milvus requirements before operations are attempted, providing fast feedback without requiring network calls.

## Getting Started

### Installation and Setup

The Partition Operations module requires a properly configured connection to your Milvus instance. Begin by ensuring you have the necessary dependencies installed and your connection manager configured.

```python
import asyncio
from connection_management import ConnectionManager
from partition_operations import PartitionManager

async def initialize():
    # Create a connection manager with your Milvus configuration
    connection_manager = ConnectionManager(
        host="localhost",
        port="19530",
        user="username",
        password="password"
    )
    
    # Initialize the Partition Operations module
    partition_manager = PartitionManager(connection_manager)
    
    return partition_manager

# Initialize and use
manager = asyncio.run(initialize())
```

### Understanding Partition Basics

Before diving into operations, it's important to understand how partitions work in Milvus. Every collection in Milvus automatically has a default partition named `_default`. When you insert data without specifying a partition, it goes into this default partition. You can create additional partitions to organize your data based on any criteria that makes sense for your application - such as user segments, time periods, geographic regions, or data types.

Partitions are permanent structures within a collection. Once created, a partition exists until explicitly deleted. Data inserted into a partition remains there unless you delete the entire partition. You can load and unload partitions from memory independently, which is useful for managing memory usage when working with large datasets.

## Core Operations

### Creating Partitions

Creating partitions is straightforward with the Partition Operations module. The module provides two methods for partition creation, each suited to different use cases.

```python
async def create_partitions_example(manager, collection_name):
    """
    Demonstrates partition creation with proper error handling.
    """
    
    # Method 1: Standard creation (fails if partition exists)
    try:
        partition = await manager.create_partition(
            collection_name=collection_name,
            partition_name="users_2024_q1",
            timeout=30.0
        )
        print(f"Created partition: {partition.name}")
        print(f"Partition ID: {partition.partition_id}")
        print(f"Created at: {partition.created_at}")
    except CollectionNotFoundError:
        print("Collection doesn't exist - create it first")
    except InvalidPartitionNameError as e:
        print(f"Invalid partition name: {e}")
    
    # Method 2: Idempotent creation (recommended)
    # This is the safer approach for most use cases
    partition = await manager.create_partition_if_not_exists(
        collection_name=collection_name,
        partition_name="users_2024_q1",
        timeout=30.0
    )
    # Returns existing partition info if it already exists
    print(f"Partition ready: {partition.name}")
```

The `create_partition_if_not_exists` method is particularly useful in production environments where you want to ensure a partition exists without worrying about whether it was created in a previous run. This method is idempotent - calling it multiple times with the same parameters is safe and will always result in the partition existing.

### Listing and Checking Partitions

Before performing operations on partitions, you often need to know what partitions exist in a collection. The Partition Operations module provides efficient methods for checking partition existence and listing all partitions.

```python
async def query_partitions_example(manager, collection_name):
    """
    Demonstrates how to query and check for partitions.
    """
    
    # Get all partitions in a collection
    partitions = await manager.list_partitions(
        collection_name=collection_name,
        timeout=10.0
    )
    print(f"Found {len(partitions)} partitions:")
    for partition_name in partitions:
        print(f"  - {partition_name}")
    
    # Check if a specific partition exists
    exists = await manager.partition_exists(
        collection_name=collection_name,
        partition_name="users_2024_q1",
        timeout=5.0
    )
    
    if exists:
        # Get detailed information about the partition
        info = await manager.get_partition_info(
            collection_name=collection_name,
            partition_name="users_2024_q1",
            timeout=10.0
        )
        print(f"Partition: {info.name}")
        print(f"State: {info.state}")
        print(f"Load State: {info.load_state}")
        print(f"Collection: {info.collection_name}")
```

The `list_partitions` method returns a list of partition names, including the default `_default` partition. This method is fast and efficient, making it suitable for frequent calls. The `partition_exists` method performs a targeted check for a specific partition, returning a boolean result. Both methods handle errors gracefully - if the collection doesn't exist, they return empty results rather than raising exceptions, unless you explicitly request strict mode.

### Loading and Unloading Partitions

In Milvus, partitions must be loaded into memory before you can query them. The Partition Operations module provides comprehensive support for loading operations, including progress monitoring for long-running loads.

```python
async def load_partition_example(manager, collection_name):
    """
    Demonstrates partition loading with progress monitoring.
    """
    
    partition_name = "users_2024_q1"
    
    # Simple load - returns immediately
    success = await manager.load_partition(
        collection_name=collection_name,
        partition_name=partition_name,
        wait=False,
        timeout=120.0
    )
    print(f"Load initiated: {success}")
    
    # Load and wait for completion
    # This is useful when you need to ensure the partition
    # is ready before proceeding with queries
    progress = await manager.load_partition(
        collection_name=collection_name,
        partition_name=partition_name,
        wait=True,
        timeout=300.0
    )
    
    if progress.is_successful:
        print(f"Partition loaded successfully")
        print(f"Progress: {progress.percentage_complete}")
        print(f"Loaded segments: {progress.loaded_segments}/{progress.total_segments}")
    else:
        print(f"Loading failed: {progress.error_message}")
    
    # Check load progress at any time
    current_progress = await manager.get_load_progress(
        collection_name=collection_name,
        partition_name=partition_name,
        timeout=10.0
    )
    print(f"Current state: {current_progress.state}")
    print(f"Progress: {current_progress.progress:.1%}")
```

When you load a partition with `wait=True`, the operation uses exponential backoff to poll for completion status. This means it checks frequently at first, then gradually increases the interval between checks. This approach is efficient and responsive - you get quick feedback when operations complete fast, but don't overwhelm the system with requests for long-running operations.

Unloading partitions is equally straightforward and is important for managing memory resources:

```python
async def unload_partition_example(manager, collection_name):
    """
    Demonstrates partition unloading to free memory.
    """
    
    # Release a partition from memory
    success = await manager.release_partition(
        collection_name=collection_name,
        partition_name="users_2024_q1",
        timeout=30.0
    )
    
    if success:
        print("Partition unloaded - memory freed")
        
        # Verify the partition is unloaded
        info = await manager.get_partition_info(
            collection_name=collection_name,
            partition_name="users_2024_q1"
        )
        print(f"Load state: {info.load_state}")  # Should be UNLOADED
```

### Getting Partition Statistics

Understanding your partition's characteristics is crucial for optimization and capacity planning. The Partition Operations module provides detailed statistics about partition size, entity counts, and resource usage.

```python
async def analyze_partition_example(manager, collection_name):
    """
    Demonstrates retrieving and analyzing partition statistics.
    """
    
    stats = await manager.get_partition_stats(
        collection_name=collection_name,
        partition_name="users_2024_q1",
        timeout=30.0
    )
    
    print(f"Partition: {stats.name}")
    print(f"Entity count: {stats.row_count:,}")
    print(f"Memory size: {stats.memory_size / (1024**2):.2f} MB")
    print(f"Disk size: {stats.disk_size / (1024**2):.2f} MB")
    print(f"Index size: {stats.index_size / (1024**2):.2f} MB")
    print(f"Total size: {stats.total_size / (1024**2):.2f} MB")
    print(f"Segments: {stats.num_segments}")
    
    # Check if partition needs optimization
    if stats.is_empty:
        print("Warning: Partition is empty")
    
    if stats.num_segments > 0:
        avg_size = stats.average_segment_size / (1024**2)
        print(f"Average segment size: {avg_size:.2f} MB")
```

These statistics are invaluable for understanding how your data is distributed and how much resources each partition consumes. You can use this information to make informed decisions about when to create new partitions, when to compact existing ones, and how to allocate resources.

### Deleting Partitions

When you no longer need a partition, you can delete it permanently. This operation is irreversible and removes all data in the partition, so use it with caution.

```python
async def delete_partition_example(manager, collection_name):
    """
    Demonstrates safe partition deletion.
    """
    
    partition_name = "users_2024_q1"
    
    # Verify the partition exists before deletion
    exists = await manager.partition_exists(
        collection_name=collection_name,
        partition_name=partition_name
    )
    
    if not exists:
        print(f"Partition {partition_name} doesn't exist")
        return
    
    # Get statistics before deletion (for logging/auditing)
    stats = await manager.get_partition_stats(
        collection_name=collection_name,
        partition_name=partition_name
    )
    print(f"About to delete partition with {stats.row_count} entities")
    
    try:
        # Delete the partition
        success = await manager.delete_partition(
            collection_name=collection_name,
            partition_name=partition_name,
            timeout=60.0
        )
        
        if success:
            print(f"Partition {partition_name} deleted successfully")
    except ValueError as e:
        # Protection against deleting the default partition
        print(f"Cannot delete partition: {e}")
```

The Partition Operations module includes safety features to prevent accidental deletion of critical partitions. By default, it prevents deletion of the `_default` partition, as this could cause issues with data insertion. You can configure this behavior through the configuration system if needed.

## Batch Operations

When working with multiple partitions, batch operations can significantly improve efficiency and simplify your code. The Partition Operations module provides convenient methods for creating and deleting multiple partitions.

```python
async def batch_operations_example(manager, collection_name):
    """
    Demonstrates efficient batch partition operations.
    """
    
    # Create multiple partitions at once
    # This is useful for setting up time-based partitions
    partition_names = [
        "users_2024_q1",
        "users_2024_q2",
        "users_2024_q3",
        "users_2024_q4"
    ]
    
    created_partitions = await manager.create_multiple_partitions(
        collection_name=collection_name,
        partition_names=partition_names,
        timeout=30.0,
        continue_on_error=True  # Keep going if one fails
    )
    
    print(f"Created {len(created_partitions)} partitions:")
    for partition in created_partitions:
        print(f"  - {partition.name} (ID: {partition.partition_id})")
    
    # Later, delete old partitions in batch
    old_partitions = ["users_2023_q1", "users_2023_q2"]
    
    deleted = await manager.delete_multiple_partitions(
        collection_name=collection_name,
        partition_names=old_partitions,
        timeout=30.0,
        continue_on_error=True
    )
    
    print(f"Deleted {len(deleted)} partitions: {', '.join(deleted)}")
```

The `continue_on_error` parameter is particularly useful in batch operations. When set to `True`, the operation continues processing remaining partitions even if one fails. This is ideal for scenarios where you want to make progress on as many partitions as possible. When set to `False`, the operation stops at the first error, which is appropriate when all partitions must succeed.

## Configuration

The Partition Operations module uses a flexible configuration system that supports both environment variables and programmatic configuration. This allows you to set defaults globally while overriding them per-operation when needed.

### Environment-Based Configuration

The simplest way to configure the Partition Operations module is through environment variables. This approach is ideal for containerized deployments and follows the twelve-factor app methodology.

```bash
# Set default timeouts (in seconds)
export MILVUS_PARTITION_DEFAULT_TIMEOUT=30.0
export MILVUS_PARTITION_CREATE_TIMEOUT=60.0
export MILVUS_PARTITION_DROP_TIMEOUT=30.0
export MILVUS_PARTITION_LOAD_TIMEOUT=120.0

# Enable partition name validation
export MILVUS_PARTITION_VALIDATE_NAMES=true
export MILVUS_PARTITION_MAX_NAME_LENGTH=255

# Safety features
export MILVUS_PARTITION_PREVENT_DEFAULT_DELETION=true
```

These environment variables are read when the configuration object is first created. You can then use the Partition Operations module without any additional configuration:

```python
from partition_operations import PartitionManager

# Configuration is automatically loaded from environment
manager = PartitionManager(connection_manager)
```

### Programmatic Configuration

For more complex scenarios or when you need different configurations in different parts of your application, you can configure the Partition Operations module programmatically.

```python
from partition_operations.config import PartitionConfig, set_partition_config

# Create a custom configuration
config = PartitionConfig(
    default_operation_timeout=45.0,
    create_partition_timeout=90.0,
    drop_partition_timeout=45.0,
    load_partition_timeout=180.0,
    validate_partition_names=True,
    max_partition_name_length=200,
    prevent_default_partition_deletion=True
)

# Set as global configuration
set_partition_config(config)

# Now all PartitionManager instances will use this configuration
manager = PartitionManager(connection_manager)
```

You can also retrieve the current configuration to check settings or modify them:

```python
from partition_operations.config import get_partition_config

config = get_partition_config()
print(f"Default timeout: {config.default_operation_timeout}s")
print(f"Validation enabled: {config.validate_partition_names}")
```

## Validation and Naming Conventions

The Partition Operations module enforces naming conventions to ensure partition names are valid and won't cause issues with Milvus. Understanding these rules helps you choose good partition names from the start.

### Partition Naming Rules

Partition names in Milvus have specific requirements:

**Length**: Names must be between 1 and 255 characters. While Milvus supports up to 255 characters, shorter names (20-50 characters) are recommended for readability.

**Characters**: Only alphanumeric characters (A-Z, a-z, 0-9), underscores (_), and hyphens (-) are allowed. Spaces and special characters will cause validation errors.

**Reserved Names**: You cannot use `_default` or `default` as partition names, as these are reserved by Milvus.

**Best Practices**: Names should not start or end with underscores or hyphens, and should not contain consecutive underscores. While these patterns are technically valid, they can cause confusion and should be avoided.

### Validation Examples

```python
from partition_operations.core.validator import PartitionValidator

async def validate_names_example():
    """
    Demonstrates partition name validation.
    """
    
    # Valid partition names
    valid_names = [
        "users_2024",
        "user-segments",
        "partition123",
        "p1"
    ]
    
    for name in valid_names:
        is_valid, errors = await PartitionValidator.validate_partition_name(name)
        print(f"{name}: {'✓ Valid' if is_valid else '✗ Invalid'}")
    
    # Invalid partition names and why they fail
    invalid_examples = {
        "": "Empty name",
        "_default": "Reserved name",
        "user data": "Contains space",
        "user/data": "Contains invalid character",
        "user__data": "Consecutive underscores",
        "_users": "Starts with underscore",
        "users_": "Ends with underscore",
        "a" * 256: "Exceeds maximum length"
    }
    
    for name, reason in invalid_examples.items():
        is_valid, errors = await PartitionValidator.validate_partition_name(name)
        if not is_valid:
            print(f"❌ {name[:20]}... : {reason}")
            print(f"   Errors: {', '.join(errors)}")
```

The validator also provides a sanitization method that attempts to fix common naming issues:

```python
from partition_operations.core.validator import PartitionValidator

# Sanitize problematic names
problematic_name = "  User Data 2024!  "
sanitized = PartitionValidator.sanitize_partition_name(problematic_name)
print(f"Original: '{problematic_name}'")
print(f"Sanitized: '{sanitized}'")  # Output: "User_Data_2024"

# Always validate after sanitization
is_valid, errors = await PartitionValidator.validate_partition_name(sanitized)
if is_valid:
    print("✓ Name is now valid")
```

## Error Handling

Robust error handling is crucial for production applications. The Partition Operations module provides a hierarchy of specific exception types that help you handle different error scenarios appropriately.

### Exception Hierarchy

```python
from partition_operations.exceptions import (
    PartitionError,              # Base exception
    PartitionNotFoundError,      # Partition doesn't exist
    PartitionAlreadyExistsError, # Partition already exists
    InvalidPartitionNameError,   # Name validation failed
    PartitionOperationError      # Operation failed
)
```

### Comprehensive Error Handling Example

```python
async def robust_partition_creation(manager, collection_name, partition_name):
    """
    Demonstrates comprehensive error handling for partition operations.
    """
    
    try:
        partition = await manager.create_partition(
            collection_name=collection_name,
            partition_name=partition_name,
            timeout=60.0
        )
        print(f"✓ Created partition: {partition.name}")
        return partition
        
    except InvalidPartitionNameError as e:
        # Handle validation errors - these are client-side errors
        # that don't require a retry
        print(f"❌ Invalid partition name: {e}")
        print("   Please choose a name with only letters, numbers, and underscores")
        
        # Suggest a sanitized version
        from partition_operations.core.validator import PartitionValidator
        suggested = PartitionValidator.sanitize_partition_name(partition_name)
        print(f"   Suggested name: {suggested}")
        return None
        
    except CollectionNotFoundError as e:
        # Handle missing collection - need to create collection first
        print(f"❌ Collection doesn't exist: {e}")
        print("   Create the collection before creating partitions")
        return None
        
    except PartitionAlreadyExistsError as e:
        # Partition exists - decide if this is an error or expected
        print(f"ℹ Partition already exists: {e}")
        # Get existing partition info
        partition = await manager.get_partition_info(
            collection_name=collection_name,
            partition_name=partition_name
        )
        return partition
        
    except OperationTimeoutError as e:
        # Timeout - operation may have succeeded
        print(f"⏱ Operation timed out: {e}")
        print("   Checking if partition was created...")
        
        # Verify if partition exists
        exists = await manager.partition_exists(
            collection_name=collection_name,
            partition_name=partition_name
        )
        
        if exists:
            print("   ✓ Partition was created despite timeout")
            return await manager.get_partition_info(
                collection_name=collection_name,
                partition_name=partition_name
            )
        else:
            print("   ✗ Partition was not created - retry recommended")
            return None
            
    except ConnectionError as e:
        # Connection issues - retry may help
        print(f"🔌 Connection error: {e}")
        print("   Check Milvus connection and retry")
        return None
        
    except PartitionOperationError as e:
        # General operation error - could be various causes
        print(f"❌ Operation failed: {e}")
        return None
        
    except Exception as e:
        # Unexpected error - log for investigation
        print(f"⚠ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None
```

## Real-World Scenarios

### Scenario 1: Time-Based Partition Management

A common use case is organizing data by time periods. This example shows how to manage monthly partitions for a user activity collection.

```python
from datetime import datetime, timedelta
from partition_operations import PartitionManager

async def manage_monthly_partitions(manager, collection_name):
    """
    Create and manage monthly partitions for user activity data.
    """
    
    # Generate partition names for the next 12 months
    current_date = datetime.now()
    partition_names = []
    
    for i in range(12):
        future_date = current_date + timedelta(days=30 * i)
        partition_name = f"activity_{future_date.strftime('%Y_%m')}"
        partition_names.append(partition_name)
    
    print(f"Creating partitions for: {', '.join(partition_names)}")
    
    # Create all partitions at once
    created = await manager.create_multiple_partitions(
        collection_name=collection_name,
        partition_names=partition_names,
        timeout=30.0,
        continue_on_error=True
    )
    
    print(f"Created {len(created)} partitions")
    
    # Load the current month's partition
    current_partition = f"activity_{current_date.strftime('%Y_%m')}"
    await manager.load_partition(
        collection_name=collection_name,
        partition_name=current_partition,
        wait=True,
        timeout=120.0
    )
    
    print(f"Loaded partition: {current_partition}")
    
    # Archive old partitions (older than 6 months)
    cutoff_date = current_date - timedelta(days=180)
    all_partitions = await manager.list_partitions(collection_name)
    
    old_partitions = []
    for partition_name in all_partitions:
        if partition_name.startswith("activity_"):
            try:
                # Extract date from partition name
                date_str = partition_name.split("_", 1)[1]
                partition_date = datetime.strptime(date_str, "%Y_%m")
                
                if partition_date < cutoff_date:
                    old_partitions.append(partition_name)
            except ValueError:
                continue
    
    if old_partitions:
        print(f"Archiving {len(old_partitions)} old partitions")
        # In production, you'd backup data before deletion
        deleted = await manager.delete_multiple_partitions(
            collection_name=collection_name,
            partition_names=old_partitions,
            timeout=60.0,
            continue_on_error=True
        )
        print(f"Archived {len(deleted)} partitions")
```

### Scenario 2: User Segment Management

Another common pattern is creating partitions based on user segments for targeted queries and better query performance.

```python
async def manage_user_segments(manager, collection_name):
    """
    Create and manage partitions for different user segments.
    """
    
    segments = {
        "premium_users": {"load": True, "priority": "high"},
        "standard_users": {"load": True, "priority": "medium"},
        "trial_users": {"load": False, "priority": "low"},
        "inactive_users": {"load": False, "priority": "low"}
    }
    
    # Create partitions for each segment
    for segment_name, config in segments.items():
        partition = await manager.create_partition_if_not_exists(
            collection_name=collection_name,
            partition_name=segment_name,
            timeout=30.0
        )
        
        print(f"Partition '{segment_name}' ready")
        
        # Load high-priority partitions
        if config["load"]:
            await manager.load_partition(
                collection_name=collection_name,
                partition_name=segment_name,
                wait=True,
                timeout=120.0
            )
            print(f"  ✓ Loaded (priority: {config['priority']})")
        else:
            print(f"  ○ Not loaded (priority: {config['priority']})")
    
    # Monitor segment sizes
    print("\nSegment Statistics:")
    for segment_name in segments.keys():
        stats = await manager.get_partition_stats(
            collection_name=collection_name,
            partition_name=segment_name,
            timeout=30.0
        )
        
        print(f"\n{segment_name}:")
        print(f"  Users: {stats.row_count:,}")
        print(f"  Size: {stats.total_size / (1024**2):.2f} MB")
        print(f"  Load state: {stats.load_state if hasattr(stats, 'load_state') else 'unknown'}")
```

### Scenario 3: Dynamic Partition Creation Based on Data

Sometimes you need to create partitions dynamically as data arrives, such as when handling multi-tenant applications.

```python
async def dynamic_tenant_partitions(manager, collection_name, tenant_id):
    """
    Create tenant-specific partitions on-demand.
    """
    
    # Generate partition name based on tenant ID
    partition_name = f"tenant_{tenant_id}"
    
    # Validate the partition name
    from partition_operations.core.validator import PartitionValidator
    is_valid, errors = await PartitionValidator.validate_partition_name(partition_name)
    
    if not is_valid:
        # Sanitize and retry
        partition_name = PartitionValidator.sanitize_partition_name(partition_name)
        print(f"Using sanitized name: {partition_name}")
    
    # Create partition if it doesn't exist
    partition = await manager.create_partition_if_not_exists(
        collection_name=collection_name,
        partition_name=partition_name,
        timeout=30.0
    )
    
    # Load the partition for immediate use
    await manager.load_partition(
        collection_name=collection_name,
        partition_name=partition_name,
        wait=True,
        timeout=120.0
    )
    
    print(f"Tenant partition '{partition_name}' is ready")
    
    return partition_name
```

## Best Practices

### Connection Management

Always use a properly configured connection manager and reuse the same PartitionManager instance within your application. Creating multiple PartitionManager instances can lead to unnecessary overhead and connection pool exhaustion.

```python
# Good: Single instance, reused
class Application:
    def __init__(self):
        self.connection_manager = ConnectionManager(...)
        self.partition_manager = PartitionManager(self.connection_manager)
    
    async def handle_request(self, collection_name):
        return await self.partition_manager.list_partitions(collection_name)

# Avoid: Creating new instances repeatedly
async def handle_request(collection_name):
    connection_manager = ConnectionManager(...)  # Don't do this
    partition_manager = PartitionManager(connection_manager)
    return await partition_manager.list_partitions(collection_name)
```

### Timeout Configuration

Set timeouts appropriately based on your partition sizes and network latency. Loading large partitions can take considerable time, so use longer timeouts for load operations than for simple metadata queries.

```python
# Quick operations - short timeout
exists = await manager.partition_exists(collection_name, partition_name, timeout=5.0)

# Metadata operations - moderate timeout
info = await manager.get_partition_info(collection_name, partition_name, timeout=30.0)

# Data operations - longer timeout
await manager.load_partition(collection_name, partition_name, wait=True, timeout=300.0)
```

### Error Recovery

Implement retry logic for transient failures, but avoid retrying validation errors or permanent failures.

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ConnectionError, OperationTimeoutError))
)
async def create_partition_with_retry(manager, collection_name, partition_name):
    """Creates a partition with automatic retry on transient failures."""
    return await manager.create_partition(
        collection_name=collection_name,
        partition_name=partition_name,
        timeout=60.0
    )
```

### Resource Management

Monitor partition memory usage and unload partitions that aren't actively being queried. This is especially important when working with many partitions or large datasets.

```