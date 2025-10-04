# Milvus_Ops

A professional, enterprise-grade package for Milvus vector database operations.

## Overview

Milvus_Ops provides a comprehensive toolkit for interacting with Milvus vector database in production environments. It includes optimized implementations for all common Milvus operations with a focus on performance, reliability, and ease of use.

## Features

- **Connection Management**: Optimized connection handling with pooling and retry mechanisms
- **Collection Operations**: Create, manage, and optimize Milvus collections
- **Index Operations**: Create and tune various index types (HNSW, IVF, FLAT, SPARSE)
- **Data Operations**: Efficient insertion, querying, and modification of vector data
- **Search Optimization**: Advanced techniques for optimizing vector search performance
- **Partition Management**: Effective data partitioning strategies
- **Monitoring**: Comprehensive performance monitoring and metrics collection
- **Backup & Recovery**: Enterprise-grade backup and recovery solutions

## Installation

```bash
pip install milvus-ops
```

## Quick Start

```python
from milvus_ops import MilvusClient
from milvus_ops.config import load_settings

# Load configuration from YAML file
settings = load_settings("config.yaml")

# Initialize client
client = MilvusClient(settings)

# Create collection
client.collection.create(
    name="my_collection",
    fields=[
        {"name": "id", "type": "int64", "is_primary": True},
        {"name": "vector", "type": "float_vector", "dim": 128}
    ]
)

# Insert data
client.insert(
    collection_name="my_collection",
    data={
        "id": [1, 2, 3],
        "vector": [[...], [...], [...]]  # Your vectors here
    }
)

# Search
results = client.query.search(
    collection_name="my_collection",
    query_vectors=[[...]],  # Your query vector
    limit=10
)
```

## Configuration

Milvus_Ops uses Pydantic for configuration management, providing:

1. Type validation and enforcement
2. Environment variable support
3. YAML configuration files
4. Nested configuration with proper validation

Example configuration (YAML):

```yaml
connection:
  host: milvus.example.com
  port: "19530"
  user: "milvus_user"
  password: "milvus_password"

index:
  hnsw_params:
    M: 32
    efConstruction: 300

search:
  hybrid_sparse_weight: 0.4
  hybrid_dense_weight: 0.6
```

Environment variables are also supported:

```bash
export MILVUS_HOST=milvus.example.com
export MILVUS_PORT=19530
export MILVUS_HNSW_M=32
```

## Package Structure

```
Milvus_Ops/
├── connection_management/     # Connection handling and pooling
├── collection_operations/     # Collection creation and management
├── index_operations/          # Index creation and optimization
├── data_insertion_operations/ # Data insertion utilities
├── data_query_operations/     # Vector search and retrieval
├── data_modification_operations/ # Update and delete operations
├── partition_operations/      # Partition management
├── search_optimization/       # Search parameter tuning
├── monitoring/                # Performance monitoring
├── backup_recovery/           # Backup and recovery utilities
├── config/                    # Pydantic-based configuration
└── utils/                     # Common utilities
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.