# Milvus Operations Examples

This directory contains example scripts for testing and demonstrating the Milvus Operations modules.

## Collection Operations Test

The `test_collection_ops.py` script provides a simple, professional test of the collection operations module.

## Running the Test

### Prerequisites

1. **Milvus Server**: Ensure Milvus is running and accessible
2. **Python Dependencies**: Install required packages from requirements.txt

### Running the Test

From the project root directory:

```bash
python examples/test_collection_ops.py
```

The script will test all core collection operations against a real Milvus instance.

### Configuration

Uses default configuration from `config/default_settings.yaml`. Modify as needed for your environment.
