"""
Index Operations Module

Provides comprehensive functionality for managing indexes in Milvus collections:
- Index creation with parameter validation
- Index build progress monitoring
- Index dropping with safety checks
- Support for various index types (IVF_FLAT, IVF_SQ8, HNSW, etc.)
- Optimized index parameters for different use cases
- Build progress tracking and monitoring
- Comprehensive error handling and reporting

Implements best practices for index management in production environments
with proper validation, error handling, and progress tracking.

Typical usage from external projects:

    from Milvus_Ops.index_operations import (
        IndexManager,
        IndexConfig,
        IndexParams,
        VectorIndexType,
        ScalarIndexType
    )
    
    # Create custom configuration
    config = IndexConfig(
        default_build_timeout=600.0,
        enable_progress_tracking=True
    )
    
    # Initialize manager with configuration
    index_manager = IndexManager(connection_mgr, collection_mgr, config=config)
    
    # Create index with error handling
    try:
        result = await index_manager.create_index(
            collection_name="my_collection",
            field_name="embedding",
            params=params
        )
        print(f"Index created successfully: {result}")
    except IndexBuildError as e:
        print(f"Index build failed: {e}")
"""

# Core manager (primary interface)
from .core.manager import IndexManager

# Configuration
from .config import IndexOperationConfig

# Models
from .models.entities import (
    IndexState,
    IndexBuildProgress,
    IndexDescription,
    IndexStats,
    IndexResult
)

from .models.parameters import (
    IndexParams,
    IvfFlatParams,
    IvfSQ8Params,
    IvfPQParams,
    HNSWParams,
    ANNOYParams,
    create_index_params,
    get_default_params
)
from collection_operations.schema import IndexType, MetricType

# Validation
from .core.validator import IndexValidator

# Utilities
from .utils.progress import (
    IndexBuildTracker,
    get_registry
)

# Exceptions
from .index_ops_exceptions import (
    IndexOperationError,
    IndexBuildError,
    IndexNotFoundError,
    IndexParameterError,
    IndexTypeError,
    IndexBuildInProgressError,
    IndexResourceError,
    IndexTimeoutError
)

__all__ = [
    # Primary interface
    'IndexManager',
    'IndexOperationConfig',
    
    # Models - Entities
    'IndexState',
    'IndexBuildProgress',
    'IndexDescription',
    'IndexStats',
    'IndexResult',
    
    # Models - Parameters
    'IndexType',
    'MetricType',
    'IndexParams',
    'IvfFlatParams',
    'IvfSQ8Params',
    'IvfPQParams',
    'HNSWParams',
    'ANNOYParams',
    'create_index_params',
    'get_default_params',
    
    # Utilities
    'IndexValidator',
    'IndexBuildTracker',
    'get_registry',
    
    # Exceptions
    'IndexOperationError',
    'IndexBuildError',
    'IndexNotFoundError',
    'IndexParameterError',
    'IndexTypeError',
    'IndexBuildInProgressError',
    'IndexResourceError',
    'IndexTimeoutError'
]

