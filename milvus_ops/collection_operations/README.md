# Collection Operations Module for Milvus

A professional, enterprise-grade collection management solution for Milvus vector databases with comprehensive schema validation, lifecycle management, and production-ready resiliency patterns.

## Features

- **Collection Lifecycle Management**: This module provides complete CRUD operations for collections, ensuring proper validation and error handling. This is crucial for maintaining data integrity and consistency across operations.
- **Schema Definition**: Utilizes Pydantic models for strongly-typed schema definitions, offering comprehensive validation to prevent runtime errors and ensure data consistency. This ensures type safety by providing robust schema definitions and reducing the risk of type-related errors.
- **Load Management**: Supports efficient loading and unloading of collections, with progress tracking and memory management to optimize resource usage.
- **Statistics & Monitoring**: Offers detailed collection statistics, including memory usage and performance metrics, to aid in monitoring and optimizing collection performance.
- **Schema Validation**: Implements pre-flight validation with detailed error reporting and compatibility checking to ensure schemas are correct before operations.
- **Concurrency Control**: Provides thread-safe operations with collection-specific locking to prevent race conditions, ensuring data integrity in concurrent environments.
- **Fault Tolerance**: Incorporates circuit breaker patterns, connection pooling, and automatic retry mechanisms to enhance reliability and resilience.
- **Type Safety**: Ensures full type hinting with Pydantic models, providing robust schema definitions and reducing the risk of type-related errors.
- **Idempotent Operations**: Supports safe re-execution of operations with schema compatibility checking, allowing for reliable and predictable behavior. This feature ensures that operations can be safely repeated without unintended side effects, enhancing the module's reliability.

## Installation

This module is part of the `Milvus_Ops` package. Install the package as a dependency in your project:

```bash
pip install milvus-ops
```

## Quick Start

```python
import asyncio
from connection_management import ConnectionManager
from milvus_ops.collection_operations import CollectionManager, CollectionSchema, FieldSchema, DataType


# Ensure all necessary imports are correctly listed and explained

async def main():
    # Initialize managers
    connection_manager = ConnectionManager()
    collection_manager = CollectionManager(connection_manager)

    # Define collection schema
    schema = CollectionSchema(
        fields=[
            FieldSchema(
                name="pk",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=False,
                description="Primary key field"
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=128,
                description="Vector embedding field"
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=512,
                description="Text content"
            )
        ],
        description="Example collection for vector search"
    )

    # Create collection
    await collection_manager.create_collection(
        collection_name="my_collection",
        schema=schema
    )

    # Load collection for search operations
    await collection_manager.load_collection(
        collection_name="my_collection",
        wait=True
    )

    print("Collection created and loaded successfully!")


if __name__ == "__main__":
    asyncio.run(main())
```

## Core Operations

### Creating Collections

#### Basic Collection Creation

Create a collection with a defined schema:

```python
# Define fields
id_field = FieldSchema(
    name="id",
    dtype=DataType.INT64,
    is_primary=True,
    auto_id=False,
    description="Primary key"
)

vector_field = FieldSchema(
    name="vector",
    dtype=DataType.FLOAT_VECTOR,
    dim=384,  # Dimension for embeddings
    description="Vector embeddings"
)

text_field = FieldSchema(
    name="text",
    dtype=DataType.VARCHAR,
    max_length=1000,
    description="Text content"
)

# Create schema
schema = CollectionSchema(
    fields=[id_field, vector_field, text_field],
    description="Document collection with embeddings",
    enable_dynamic_field=False
)

# Create collection
await collection_manager.create_collection(
    collection_name="documents",
    schema=schema
)
```

#### Collection with Partitioning

Create a collection with partition keys for better query performance:

```python
partition_field = FieldSchema(
    name="category",
    dtype=DataType.VARCHAR,
    max_length=50,
    is_partition_key=True,  # Enable partitioning
    description="Content category"
)

schema = CollectionSchema(
    fields=[id_field, vector_field, text_field, partition_field],
    description="Partitioned document collection"
)

await collection_manager.create_collection(
    collection_name="partitioned_docs",
    schema=schema
)
```

### Loading Collections

Load collections into memory for search operations:

```python
# Load collection with progress tracking
result = await collection_manager.load_collection(
    collection_name="documents",
    wait=True  # Wait for loading to complete
)

if hasattr(result, 'state') and result.state.value == "Loaded":
    print("Collection loaded successfully")
else:
    print(f"Loading failed: {result}")

# Monitor loading progress
progress = await collection_manager.get_load_progress("documents")
print(f"Progress: {progress.progress:.1f}%")
print(f"Loaded segments: {progress.loaded_segments}/{progress.total_segments}")
```

### Collection Statistics

Retrieve comprehensive statistics about collections:

```python
# Get collection description
description = await collection_manager.describe_collection("documents")
print(f"Collection: {description.name}")
print(f"Created: {description.created_at}")
print(f"State: {description.state.value}")
print(f"Load State: {description.load_state.value}")

# Get detailed statistics
stats = await collection_manager.get_collection_stats("documents")
print(f"Entity count: {stats.row_count}")
print(f"Memory size: {stats.memory_size / 1024 / 1024:.2f} MB")
print(f"Disk size: {stats.disk_size / 1024 / 1024:.2f} MB")
print(f"Number of segments: {stats.num_segments}")
```

### Listing Collections

List all collections in the database:

```python
collections = await collection_manager.list_collections()

for collection_name in collections:
    exists = await collection_manager.has_collection(collection_name)
    if exists:
        desc = await collection_manager.describe_collection(collection_name)
        print(f"- {collection_name}: {desc.load_state.value}")
```

### Releasing Collections

Unload collections from memory to free resources:

```python
success = await collection_manager.release_collection("documents")

if success:
    print("Collection released successfully")
else:
    print("Failed to release collection")
```

### Dropping Collections

Permanently delete collections (use with caution):

```python
# Check if collection exists before dropping
exists = await collection_manager.has_collection("documents")

if exists:
    # Release first if loaded
    progress = await collection_manager.get_load_progress("documents")
    if progress.state.value == "Loaded":
        await collection_manager.release_collection("documents")

    # Drop collection
    success = await collection_manager.drop_collection("documents")
    if success:
        print("Collection dropped successfully")
else:
    print("Collection does not exist")
```

## Schema Definition and Management

### Data Types

The module supports all Milvus data types:

```python
from milvus_ops.collection_operations import DataType

# Numeric types
INT64 = DataType.INT64
FLOAT = DataType.FLOAT
DOUBLE = DataType.DOUBLE

# String types
VARCHAR = DataType.VARCHAR  # Requires max_length

# Vector types
FLOAT_VECTOR = DataType.FLOAT_VECTOR  # Requires dim
BINARY_VECTOR = DataType.BINARY_VECTOR  # Requires dim
SPARSE_FLOAT_VECTOR = DataType.SPARSE_FLOAT_VECTOR  # Requires dim

# Other types
BOOL = DataType.BOOL
JSON = DataType.JSON
ARRAY = DataType.ARRAY  # Requires element_type
```

### Field Schema

Define individual fields with proper constraints:

```python
# Primary key field
pk_field = FieldSchema(
    name="id",
    dtype=DataType.INT64,
    is_primary=True,      # Exactly one primary key required
    auto_id=False,        # Manual ID assignment
    description="Unique identifier"
)

# Vector field
vector_field = FieldSchema(
    name="embedding",
    dtype=DataType.FLOAT_VECTOR,
    dim=768,              # Required for vector fields
    description="Text embeddings"
)

# Text field
text_field = FieldSchema(
    name="content",
    dtype=DataType.VARCHAR,
    max_length=2048,      # Required for VARCHAR fields
    description="Document content"
)

# Partition key field
category_field = FieldSchema(
    name="category",
    dtype=DataType.VARCHAR,
    max_length=100,
    is_partition_key=True,  # Enables partitioning
    description="Content category"
)

# Array field
tags_field = FieldSchema(
    name="tags",
    dtype=DataType.ARRAY,
    element_type=DataType.VARCHAR,  # Required for ARRAY fields
    description="Content tags"
)
```

### Collection Schema

Combine fields into a complete collection schema:

```python
schema = CollectionSchema(
    fields=[pk_field, vector_field, text_field, category_field],
    description="Production document collection",
    enable_dynamic_field=False,  # Strict schema enforcement
    shard_num=2                  # Number of shards (optional)
)
```

### Schema Validation

Schemas are validated before collection creation:

```python
from milvus_ops.collection_operations import SchemaValidator

# Validate schema
is_valid, errors = await SchemaValidator.validate_schema(schema)

if not is_valid:
    for error in errors:
        print(f"Validation error: {error}")
else:
    print("Schema is valid")
```

## Configuration

The CollectionManager integrates with the ConnectionManager for resilient operations. Configuration is handled through the ConnectionManager's settings.

### Connection Manager Configuration

```python
from connection_management import ConnectionManager
from config import load_settings

# Load configuration from environment/settings
config = load_settings()

# Configure connection with retry and pooling
connection_manager = ConnectionManager(
    config=config,
    # Additional connection parameters can be passed here
)

# Create collection manager
collection_manager = CollectionManager(connection_manager)
```

### Timeout Configuration

Set operation timeouts for different scenarios:

```python
# Create collection with timeout
await collection_manager.create_collection(
    collection_name="my_collection",
    schema=schema,
    timeout=300.0  # 5 minutes
)

# Load collection with timeout
await collection_manager.load_collection(
    collection_name="my_collection",
    wait=True,
    timeout=600.0  # 10 minutes for large collections
)
```

## Error Handling

The module provides comprehensive error handling with specific exception types:

```python
from milvus_ops.collection_operations import (
    CollectionError,
    CollectionNotFoundError,
    SchemaError
)
from milvus_ops.milvus_ops_exceptions import (
    ConnectionError,
    OperationTimeoutError
)

try:
    await collection_manager.create_collection("my_collection", schema)
except CollectionNotFoundError as e:
    print(f"Collection not found: {e}")
except SchemaError as e:
    print(f"Schema validation failed: {e}")
    print(f"Details: {e.details}")
except ConnectionError as e:
    print(f"Connection failed: {e}")
except OperationTimeoutError as e:
    print(f"Operation timed out: {e}")
except CollectionError as e:
    print(f"Collection operation failed: {e}")
```

### Idempotent Operations

Operations are designed to be idempotent where possible:

```python
# Safe to call multiple times - will check compatibility
await collection_manager.create_collection(
    collection_name="my_collection",
    schema=schema
)

# Will succeed if collection exists and is compatible
# Will fail if collection exists but schema is incompatible
```

## Advanced Usage

### Custom Schema Validation

Implement custom validation logic:

```python
class CustomSchemaValidator(SchemaValidator):
    @classmethod
    async def validate_custom_rules(cls, schema: CollectionSchema) -> List[str]:
        errors = []

        # Custom business rules
        vector_fields = [f for f in schema.fields if f.dtype == DataType.FLOAT_VECTOR]
        if len(vector_fields) > 1:
            errors.append("Multiple vector fields not allowed in this schema")

        # Domain-specific constraints
        for field in schema.fields:
            if field.name.startswith('internal_') and not field.description:
                errors.append(f"Internal field {field.name} must have description")

        return errors

# Use custom validator
is_valid, errors = await CustomSchemaValidator.validate_schema(schema)
custom_errors = await CustomSchemaValidator.validate_custom_rules(schema)
errors.extend(custom_errors)
```

### Collection Migration

Safely migrate collections with schema changes:

```python
async def migrate_collection(old_name: str, new_name: str, new_schema: CollectionSchema):
    # Create new collection with updated schema
    await collection_manager.create_collection(
        collection_name=new_name,
        schema=new_schema
    )

    # Load new collection
    await collection_manager.load_collection(new_name, wait=True)

    # TODO: Implement data migration logic here
    # This would involve querying data from old collection
    # and inserting into new collection

    # Drop old collection after successful migration
    await collection_manager.drop_collection(old_name)

    print(f"Successfully migrated {old_name} to {new_name}")
```

### Memory Management

Monitor and manage collection memory usage:

```python
# Get memory usage statistics
stats = await collection_manager.get_collection_stats("large_collection")

memory_mb = stats.memory_size / 1024 / 1024
disk_mb = stats.disk_size / 1024 / 1024

print(f"Memory usage: {memory_mb:.2f} MB")
print(f"Disk usage: {disk_mb:.2f} MB")

# Implement memory-based decisions
if memory_mb > 1000:  # 1GB threshold
    print("Collection using significant memory, consider partitioning")
    await collection_manager.release_collection("large_collection")
```

## Best Practices

### For Development

1. **Use descriptive field names**: Avoid generic names like "field1", "field2"
2. **Always specify descriptions**: Document the purpose of each field
3. **Validate schemas early**: Test schema validation before deployment
4. **Use consistent naming**: Follow naming conventions across collections

### For Production Deployments

1. **Implement proper logging**: Monitor collection operations for debugging
2. **Set appropriate timeouts**: Configure timeouts based on collection size
3. **Handle errors gracefully**: Implement retry logic for transient failures
4. **Monitor resource usage**: Track memory and disk usage over time
5. **Use partitioning strategically**: Partition large collections by logical groups
6. **Backup before destructive operations**: Always backup before dropping collections

### For Large Collections

1. **Use sharding**: Set appropriate `shard_num` for distributed processing
2. **Monitor load progress**: Track loading progress for large collections
3. **Implement batch operations**: Process large collections in manageable chunks
4. **Consider memory limits**: Monitor memory usage and implement release strategies

### For High-Throughput Scenarios

1. **Reuse connections**: Use the ConnectionManager's pooling capabilities
2. **Implement concurrency control**: Use the built-in locking mechanisms
3. **Batch schema validations**: Validate multiple schemas together when possible
4. **Monitor operation latency**: Track and optimize slow operations

## Troubleshooting

### Collection Creation Fails

1. **Check schema validation**:
   ```python
   is_valid, errors = await SchemaValidator.validate_schema(schema)
   print("Validation errors:", errors)
   ```

2. **Verify field constraints**:
   - Vector fields must have `dim` specified
   - VARCHAR fields must have `max_length`
   - ARRAY fields must have `element_type`
   - Exactly one primary key field required

3. **Check reserved names**: Avoid using Milvus reserved field names

### Loading Fails

1. **Check collection state**: Ensure collection exists and is not already loaded
2. **Verify index existence**: Collections need indexes before loading
3. **Check memory availability**: Ensure sufficient memory for loading
4. **Monitor progress**: Use `get_load_progress` to track loading status

### Memory Issues

1. **Monitor memory usage**: Use `get_collection_stats` to track memory consumption
2. **Implement release strategies**: Release unused collections to free memory
3. **Check for memory leaks**: Monitor memory growth over time
4. **Consider partitioning**: Split large collections into smaller partitions

### Connection Issues

1. **Check ConnectionManager status**: Verify connection pool health
2. **Implement retry logic**: Handle transient connection failures
3. **Monitor circuit breaker**: Check if circuit breaker is in OPEN state
4. **Review network configuration**: Ensure proper network connectivity

## Integration Examples

### Automated Collection Management

```python
import asyncio
import schedule
from datetime import datetime

class CollectionMaintainer:
    def __init__(self, collection_manager: CollectionManager):
        self.collection_manager = collection_manager

    async def ensure_collection_exists(self, name: str, schema: CollectionSchema):
        """Ensure a collection exists with the correct schema."""
        if not await self.collection_manager.has_collection(name):
            await self.collection_manager.create_collection(name, schema)
            print(f"Created collection: {name}")
        else:
            print(f"Collection {name} already exists")

    async def maintain_collections(self):
        """Periodic maintenance of collections."""
        collections = await self.collection_manager.list_collections()

        for collection_name in collections:
            try:
                # Check load state
                desc = await self.collection_manager.describe_collection(collection_name)

                # Auto-load critical collections
                if desc.load_state.value != "Loaded":
                    critical_collections = ["user_vectors", "product_catalog"]
                    if collection_name in critical_collections:
                        await self.collection_manager.load_collection(collection_name)
                        print(f"Auto-loaded critical collection: {collection_name}")

                # Clean up old collections (example logic)
                if "temp_" in collection_name:
                    # Implement cleanup logic based on your requirements
                    pass

            except Exception as e:
                print(f"Error maintaining {collection_name}: {e}")

# Usage
maintainer = CollectionMaintainer(collection_manager)

# Schedule periodic maintenance
def maintenance_job():
    asyncio.run(maintainer.maintain_collections())

schedule.every(1).hours.do(maintenance_job)
```

### Schema Evolution Helper

```python
class SchemaEvolutionManager:
    def __init__(self, collection_manager: CollectionManager):
        self.collection_manager = collection_manager

    async def evolve_schema(self, collection_name: str, new_schema: CollectionSchema):
        """Safely evolve a collection's schema."""

        # Get current schema
        current_desc = await self.collection_manager.describe_collection(collection_name)
        current_schema = current_desc.collection_schema

        # Check compatibility
        if self._is_compatible(current_schema, new_schema):
            print("Schema is compatible, no migration needed")
            return True

        # Create new collection with evolved schema
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"{collection_name}_v2_{timestamp}"

        await self.collection_manager.create_collection(new_name, new_schema)

        # TODO: Implement data migration
        # await self._migrate_data(collection_name, new_name)

        # Swap collections (rename operations)
        # await self._swap_collections(collection_name, new_name)

        return True

    def _is_compatible(self, old_schema: CollectionSchema, new_schema: CollectionSchema) -> bool:
        """Check if schema evolution is backward compatible."""
        # Implement compatibility checking logic
        # Return True if compatible, False if migration needed
        return False
```

## License

This module is part of the Milvus_Ops package and is licensed under [LICENSE TERMS].

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for details.
