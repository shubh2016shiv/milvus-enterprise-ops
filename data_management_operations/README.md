# Data Management Operations Module

Professional, enterprise-grade module for managing data operations in Milvus collections with robust error handling, configurable parameters, and performance monitoring.

## Features

- **Document Insertion**: Batch insertion with automatic batching and validation
- **Upsert Operations**: Insert or update documents atomically
- **Delete Operations**: Expression-based document deletion
- **Schema Validation**: Pre-flight validation against collection schemas
- **Configurable Parameters**: Externalized configuration for batch sizes, timeouts, and retries
- **Performance Timing**: Built-in performance monitoring and statistics
- **Robust Error Handling**: Granular exception hierarchy for precise error handling
- **Transient Error Retry**: Automatic retry for transient failures
- **Thread-Safe Operations**: Asyncio-based locking for concurrent operations
- **Type-Safe**: Comprehensive type hinting throughout

## Installation

This module is part of the `Milvus_Ops` package. Install the package as a dependency in your project.

## Quick Start

```python
from Milvus_Ops.data_management_operations import (
    DataManager,
    DataOperationConfig,
    Document
)
from Milvus_Ops.connection_management import ConnectionManager
from Milvus_Ops.collection_operations import CollectionManager

# Create custom configuration
config = DataOperationConfig(
    default_batch_size=500,
    retry_transient_errors=True,
    default_operation_timeout=60.0
)

# Initialize managers
connection_mgr = ConnectionManager(...)
collection_mgr = CollectionManager(...)

# Create data manager with configuration
data_manager = DataManager(
    connection_manager=connection_mgr,
    collection_manager=collection_mgr,
    config=config
)

# Insert documents
documents = [
    Document(id=1, vector=[0.1, 0.2, 0.3], text="example"),
    Document(id=2, vector=[0.4, 0.5, 0.6], text="another")
]

result = await data_manager.insert(
    collection_name="my_collection",
    documents=documents,
    batch_size=1000
)

print(f"Inserted {result.successful_count} documents")
```

## Module Structure

```
data_management_operations/
├── __init__.py              # Public API exports
├── config.py                # Configuration dataclass
├── exceptions.py            # Custom exceptions
├── example.py               # Usage examples
├── README.md                # This file
├── core/                    # Core functionality
│   ├── __init__.py
│   ├── manager.py           # DataManager class
│   └── validator.py         # Data validation
├── models/                  # Data models
│   ├── __init__.py
│   └── entities.py          # Pydantic models
└── utils/                   # Utilities
    ├── __init__.py
    ├── timing.py            # Performance timing
    └── retry.py             # Retry logic
```

## Configuration

The `DataOperationConfig` class provides centralized configuration:

```python
config = DataOperationConfig(
    # Batching
    default_batch_size=1000,
    max_batch_size=10000,
    min_batch_size=100,
    
    # Timeouts (seconds)
    default_operation_timeout=30.0,
    health_check_timeout=5.0,
    
    # Retry settings
    retry_transient_errors=True,
    max_transient_retries=3,
    transient_retry_delay=0.5,
    
    # Performance
    enable_timing=True,
    
    # Validation
    strict_validation=True
)
```

Configuration can also be loaded from a dictionary:

```python
config_dict = {
    'default_batch_size': 500,
    'retry_transient_errors': True
}
config = DataOperationConfig.from_dict(config_dict)
```

## Operations

### Insert Documents

```python
result = await data_manager.insert(
    collection_name="my_collection",
    documents=documents,
    batch_size=1000,
    validate=True,
    partition_key=None,
    auto_create_collection=False
)
```

### Upsert Documents

```python
result = await data_manager.upsert(
    collection_name="my_collection",
    documents=documents,
    batch_size=1000,
    validate=True
)
```

### Delete Documents

```python
result = await data_manager.delete(
    collection_name="my_collection",
    expr="id in [1, 2, 3]",
    partition_key=None
)
```

## Error Handling

The module provides granular exception types for precise error handling:

```python
from Milvus_Ops.data_management_operations import (
    BatchPartialFailureError,
    SchemaValidationError,
    CollectionOperationError,
    TransientOperationError
)

try:
    result = await data_manager.insert(...)
except BatchPartialFailureError as e:
    # Handle partial batch failure
    print(f"Succeeded: {e.successful_count}")
    print(f"Failed: {e.failed_count}")
    print(f"Success rate: {e.success_rate:.2f}%")
    for doc_id, error in e.error_details.items():
        print(f"Document {doc_id} failed: {error}")

except SchemaValidationError as e:
    # Handle validation errors
    for doc_id, errors in e.validation_errors.items():
        print(f"Document {doc_id} errors: {errors}")

except CollectionOperationError as e:
    # Handle collection-level errors
    print(f"Collection error: {e}")

except TransientOperationError as e:
    # Handle transient errors (already retried)
    print(f"Transient error persisted: {e}")
```

## Performance Monitoring

Built-in performance timing tracks all operations:

```python
# Get timing history
history = data_manager.get_timing_history()
for timing in history:
    print(f"{timing.operation_name}: {timing.execution_time_ms:.2f}ms")

# Get statistics for specific operation
stats = data_manager.get_operation_stats("insert_documents")
if stats:
    print(f"Average time: {stats.average_execution_time*1000:.2f}ms")
    print(f"P95 time: {stats.p95_execution_time*1000:.2f}ms")
    print(f"Success rate: {stats.success_rate:.2f}%")

# Get summary of all operations
summary = data_manager.get_performance_summary()
for op_name, stats in summary.items():
    print(f"{op_name}: {stats.total_operations} operations")
```

## Document Models

### Document

Standard document model with vector and metadata:

```python
doc = Document(
    id=1,
    vector=[0.1, 0.2, 0.3],  # Can also be dict of named vectors
    text="Example document",
    metadata={"category": "test"}
)
```

### DocumentBase

Base class for custom document models:

```python
from Milvus_Ops.data_management_operations import DocumentBase

class MyDocument(DocumentBase):
    custom_field: str
    another_field: int
```

## Best Practices

1. **Use Configuration**: Externalize all tunable parameters in `DataOperationConfig`
2. **Handle Partial Failures**: Always check for `BatchPartialFailureError` in batch operations
3. **Monitor Performance**: Use timing utilities to track and optimize performance
4. **Validate Early**: Enable `validate=True` to catch schema mismatches before insertion
5. **Configure Batching**: Adjust `batch_size` based on your document size and available memory
6. **Use Type Hints**: Leverage the generic type system for type-safe document handling

## Scaling Considerations

- **Batch Size**: Larger batches are more efficient but use more memory
- **Concurrent Operations**: The manager uses per-collection locks for thread safety
- **Timeout Configuration**: Set appropriate timeouts based on your network latency
- **Retry Configuration**: Enable retries for transient errors in unreliable networks

## Integration with External Projects

This module is designed as a reusable package. In your project:

1. Add `Milvus_Ops` as a dependency
2. Import and configure as needed
3. Use the public API from `data_management_operations`
4. Customize configuration via `DataOperationConfig`

```python
# your_project/data_layer.py
from Milvus_Ops.data_management_operations import (
    DataManager,
    DataOperationConfig,
    Document,
    BatchPartialFailureError
)

# Load config from your project's settings
config = DataOperationConfig.from_dict(your_settings['milvus_data_ops'])

# Use in your application
manager = DataManager(conn_mgr, coll_mgr, config=config)
```

## License

Part of the Milvus_Ops package.

