# Index Operations Module

Professional, enterprise-grade module for managing index operations in Milvus collections with robust error handling, type-safe parameters, and progress monitoring.

## Features

- **Index Creation**: Create indexes with type-specific parameters and validation
- **Progress Monitoring**: Track long-running index builds with ETA estimation
- **Index Management**: Describe, list, and drop indexes
- **Type-Safe Parameters**: Strongly-typed parameter classes for each index type
- **Parameter Validation**: Automatic validation of parameters for compatibility
- **Robust Error Handling**: Granular exception hierarchy for precise error handling
- **Performance Tracking**: Built-in timing for operations
- **Async Operations**: Non-blocking operations for high concurrency
- **Multiple Index Types**: Support for HNSW, IVF_FLAT, IVF_SQ8, IVF_PQ, SCANN, and more

## Installation

This module is part of the `Milvus_Ops` package. Install the package as a dependency in your project.

## Quick Start

```python
from Milvus_Ops.index_operations import (
    IndexManager,
    IndexOperationConfig,
    IndexType,
    MetricType,
    HNSWParams
)

# Create custom configuration
config = IndexOperationConfig(
    default_timeout=120.0,
    enable_timing=True
)

# Initialize managers
connection_mgr = ConnectionManager(...)
collection_mgr = CollectionManager(...)

# Create index manager
index_manager = IndexManager(
    connection_manager=connection_mgr,
    collection_manager=collection_mgr,
    config=config
)

# Create index with type-safe parameters
result = await index_manager.create_index(
    collection_name="documents",
    field_name="embedding",
    index_type=IndexType.HNSW,
    metric_type=MetricType.COSINE,
    index_params=HNSWParams(M=16, efConstruction=200),
    wait=True
)

if result.success:
    print("Index created successfully!")
```

## Module Structure

```
index_operations/
├── __init__.py              # Public API exports
├── config.py                # IndexOperationConfig
├── exceptions.py            # Index-specific exceptions
├── README.md                # This file
├── example.py               # Usage examples
├── core/                    # Core functionality
│   ├── __init__.py
│   ├── manager.py           # IndexManager class
│   └── validator.py         # Parameter validation
├── models/                  # Data models
│   ├── __init__.py
│   ├── entities.py          # Index models and results
│   └── parameters.py        # Type-specific parameters
└── utils/                   # Utilities
    ├── __init__.py
    └── progress.py          # Progress tracking
```

## Configuration

The `IndexOperationConfig` class provides centralized configuration:

```python
config = IndexOperationConfig(
    # Timeout settings
    default_timeout=60.0,  # seconds
    
    # Progress monitoring
    build_progress_poll_interval=2.0,  # seconds
    
    # Concurrency
    max_concurrent_builds=0,  # 0 = no limit
    
    # Performance
    enable_timing=True,
    
    # Optimization
    auto_optimize_params=False,
    resource_monitoring=False,
    
    # Retry settings
    retry_transient_errors=True,
    max_transient_retries=3,
    transient_retry_delay=0.5
)
```

Configuration can also be loaded from a dictionary:

```python
config_dict = {
    'default_timeout': 120.0,
    'enable_timing': True
}
config = IndexOperationConfig.from_dict(config_dict)
```

## Operations

### Create Index

```python
# Create index with default parameters
result = await index_manager.create_index(
    collection_name="documents",
    field_name="embedding",
    index_type=IndexType.IVF_FLAT,
    metric_type=MetricType.L2,
    wait=True
)

# Create index with custom parameters
hnsw_params = HNSWParams(M=16, efConstruction=200)
result = await index_manager.create_index(
    collection_name="documents",
    field_name="embedding",
    index_type=IndexType.HNSW,
    metric_type=MetricType.COSINE,
    index_params=hnsw_params,
    wait=False  # Don't wait for completion
)
```

### Monitor Progress

```python
# Create index without waiting
result = await index_manager.create_index(..., wait=False)

# Monitor progress
while True:
    progress = await index_manager.get_index_build_progress(
        collection_name="documents",
        field_name="embedding"
    )
    
    print(f"Progress: {progress.percentage:.2f}%")
    
    if progress.formatted_eta:
        print(f"ETA: {progress.formatted_eta}")
    
    if progress.state == IndexState.CREATED:
        print("Build complete!")
        break
    
    await asyncio.sleep(5)
```

### Describe Index

```python
index_info = await index_manager.describe_index(
    collection_name="documents",
    field_name="embedding"
)

print(f"Type: {index_info.index_type}")
print(f"Metric: {index_info.metric_type}")
print(f"State: {index_info.state}")
print(f"Parameters: {index_info.params}")
```

### List Indexes

```python
indexes = await index_manager.list_indexes(
    collection_name="documents"
)

for index in indexes:
    print(f"Field: {index.field_name}, Type: {index.index_type}")
```

### Check if Index Exists

```python
has_index = await index_manager.has_index(
    collection_name="documents",
    field_name="embedding"
)

if has_index:
    print("Field has an index")
```

### Drop Index

```python
result = await index_manager.drop_index(
    collection_name="documents",
    field_name="embedding"
)

if result.success:
    print("Index dropped successfully")
```

## Index Types and Parameters

### HNSW (Hierarchical Navigable Small World)

Best for: High-dimensional vectors, high query performance

```python
from Milvus_Ops.index_operations import HNSWParams

params = HNSWParams(
    M=16,              # Connections per layer (4-64)
    efConstruction=200  # Build quality (higher = better but slower)
)
```

### IVF_FLAT

Best for: Balance between performance and recall

```python
from Milvus_Ops.index_operations import IvfFlatParams

params = IvfFlatParams(
    nlist=1024  # Number of clusters
)
```

### IVF_SQ8

Best for: Memory-constrained scenarios

```python
from Milvus_Ops.index_operations import IvfSQ8Params

params = IvfSQ8Params(
    nlist=1024  # Number of clusters
)
```

### IVF_PQ

Best for: Very large datasets with memory constraints

```python
from Milvus_Ops.index_operations import IvfPQParams

params = IvfPQParams(
    nlist=1024,  # Number of clusters
    m=8,         # Number of subdivisions (must divide dimension)
    nbits=8      # Bits per subdivision
)
```

### SCANN

Best for: Large-scale similarity search

```python
from Milvus_Ops.index_operations import SCANNParams

params = SCANNParams(
    nlist=1024,  # Number of clusters
    nprobe=16    # Clusters to search
)
```

## Error Handling

The module provides granular exception types for precise error handling:

```python
from Milvus_Ops.index_operations import (
    IndexParameterError,
    IndexBuildError,
    IndexNotFoundError,
    IndexBuildInProgressError,
    IndexTimeoutError
)

try:
    result = await index_manager.create_index(...)
except IndexParameterError as e:
    # Handle invalid parameters
    print(f"Parameter error: {e}")
    for param, error in e.parameter_errors.items():
        print(f"  {param}: {error}")

except IndexBuildError as e:
    # Handle build failure
    print(f"Build failed: {e}")
    print(f"Collection: {e.collection_name}")
    print(f"Field: {e.field_name}")

except IndexNotFoundError as e:
    # Handle missing index
    print(f"Index not found: {e}")

except IndexBuildInProgressError as e:
    # Handle concurrent build attempt
    print(f"Build in progress: {e.progress:.2f}%")

except IndexTimeoutError as e:
    # Handle timeout
    print(f"Operation timed out after {e.timeout_seconds}s")
```

## Parameter Validation

The module automatically validates parameters for compatibility:

```python
from Milvus_Ops.index_operations import IndexValidator

validator = IndexValidator()

# Validate parameters before creation
try:
    validated_params = validator.validate_index_params(
        index_type=IndexType.HNSW,
        metric_type=MetricType.COSINE,
        dimension=128,
        params={"M": 16, "efConstruction": 200}
    )
except IndexParameterError as e:
    print(f"Validation failed: {e}")

# Get optimized parameters
optimal_params = validator.optimize_parameters(
    index_type=IndexType.IVF_FLAT,
    dimension=128,
    row_count=1000000
)

# Estimate memory usage
memory_bytes = validator.estimate_memory_usage(
    index_type=IndexType.HNSW,
    dimension=128,
    row_count=1000000,
    params=HNSWParams(M=16, efConstruction=200)
)
print(f"Estimated memory: {memory_bytes / (1024**3):.2f} GB")
```

## Best Practices

1. **Use Type-Safe Parameters**: Always use the parameter classes (e.g., `HNSWParams`) instead of dictionaries
2. **Monitor Long Builds**: For large datasets, use `wait=False` and monitor progress
3. **Handle Errors Gracefully**: Use specific exception types for targeted error handling
4. **Validate Parameters**: The module validates automatically, but you can pre-validate if needed
5. **Configure Appropriately**: Set timeouts and retry settings based on your environment
6. **Consider Index Type**: Choose the index type based on your performance and memory requirements

## Scaling Considerations

- **Build Time**: Index building time scales with dataset size and index complexity
- **Memory Usage**: Different index types have different memory requirements
- **Concurrent Builds**: Configure `max_concurrent_builds` based on available resources
- **Timeout Configuration**: Set appropriate timeouts for large datasets
- **Progress Polling**: Adjust `build_progress_poll_interval` to balance responsiveness and overhead

## Integration with External Projects

This module is designed as a reusable package. In your project:

1. Add `Milvus_Ops` as a dependency
2. Import and configure as needed
3. Use the public API from `index_operations`
4. Customize configuration via `IndexOperationConfig`

```python
# your_project/index_layer.py
from Milvus_Ops.index_operations import (
    IndexManager,
    IndexOperationConfig,
    IndexType,
    MetricType,
    HNSWParams
)

# Load config from your project's settings
config = IndexOperationConfig.from_dict(your_settings['milvus_index_ops'])

# Use in your application
index_manager = IndexManager(conn_mgr, coll_mgr, config=config)
```

## License

Part of the Milvus_Ops package.
