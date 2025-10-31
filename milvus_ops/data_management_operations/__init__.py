"""
Data Management Operations Module

Provides comprehensive functionality for managing data in Milvus collections:
- Document insertion with batching and validation
- Upsert operations (insert or update)
- Delete operations with expression-based filtering
- Data validation with schema compatibility checks
- Robust error handling and reporting
- Thread-safe operations with asyncio locks
- Support for partitioned collections
- Configurable operation parameters (batch size, timeouts, retries)
- Performance timing and monitoring

Implements best practices for high-throughput, reliable data operations
in production environments with proper error handling and validation.

Typical usage from external projects:

    from Milvus_Ops.data_management_operations import (
        DataManager,
        DataOperationConfig,
        Document,
        BatchPartialFailureError
    )

    # Create custom configuration
    config = DataOperationConfig(
        default_batch_size=500,
        retry_transient_errors=True,
        default_operation_timeout=60.0
    )

    # Initialize manager with configuration
    manager = DataManager(connection_mgr, collection_mgr, config=config)

    # Insert documents with error handling
    try:
        result = await manager.insert_documents(
            collection_name="my_collection",
            documents=docs,
            batch_size=1000
        )
        print(f"Successfully inserted {result.successful_count} documents")
    except BatchPartialFailureError as e:
        print(f"Partial failure: {e.successful_count} succeeded, {e.failed_count} failed")
        for doc_id in e.failed_ids:
            print(f"Failed document: {doc_id}")
"""

# Core manager (primary interface)
from .core.manager import DataManager

# Validation
from .core.validator import DataValidator

# Configuration
from .data_ops_config import DataOperationConfig

# Exceptions
from .data_ops_exceptions import (
    BatchPartialFailureError,
    CollectionOperationError,
    DataOperationError,
    DeleteOperationError,
    DocumentPreparationError,
    InsertionError,  # For backward compatibility
    SchemaValidationError,
    TransientOperationError,
)

# Data models
from .models.entities import (
    BatchOperationResult,
    DataValidationResult,
    DeleteResult,
    Document,
    DocumentBase,
    OperationStatus,
)

# Timing utilities
from .utils.timing import (
    BatchTimingResult,
    PerformanceTimer,
    TimingResult,
    time_operation,
)

__all__ = [
    # Primary interface
    "DataManager",
    "DataOperationConfig",
    # Models
    "Document",
    "DocumentBase",
    "BatchOperationResult",
    "DeleteResult",
    "DataValidationResult",
    "OperationStatus",
    # Utilities
    "DataValidator",
    "PerformanceTimer",
    "TimingResult",
    "BatchTimingResult",
    "time_operation",
    # Exceptions
    "DataOperationError",
    "BatchPartialFailureError",
    "TransientOperationError",
    "SchemaValidationError",
    "DocumentPreparationError",
    "CollectionOperationError",
    "DeleteOperationError",
    "InsertionError",
]
