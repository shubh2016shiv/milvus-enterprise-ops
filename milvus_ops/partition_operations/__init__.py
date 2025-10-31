"""
Partition Operations Module for Milvus Enterprise Ops

This module provides comprehensive partition management functionality for Milvus
vector databases, following enterprise-grade patterns and best practices.

Key Features:
- Async/await support for high-performance operations
- Comprehensive validation and error handling
- Progress tracking and monitoring capabilities
- Batch operations support
- Configuration-driven behavior
- Integration with existing Milvus enterprise ops infrastructure

Basic Usage:
    from partition_operations import PartitionManager
    from connection_management import ConnectionManager

    # Initialize managers
    connection_manager = ConnectionManager()
    partition_manager = PartitionManager(connection_manager)

    # Create a partition
    partition = await partition_manager.create_partition(
        collection_name="my_collection",
        partition_name="my_partition"
    )

    # List partitions
    partitions = await partition_manager.list_partitions(
        collection_name="my_collection"
    )

    # Load partition for querying
    await partition_manager.load_partition(
        collection_name="my_collection",
        partition_name="my_partition"
    )

Architecture:
- core/manager.py: Main PartitionManager class with async operations
- core/validator.py: Validation rules and naming conventions
- models/entities.py: Pydantic models for type safety
- config.py: Configuration management and settings
- utils.py: Utility functions and helpers
- example.py: Comprehensive usage examples

Integration:
This module integrates seamlessly with other Milvus enterprise ops modules:
- Uses ConnectionManager for robust connection handling
- Follows same patterns as collection_operations module
- Compatible with existing error handling and logging
- Supports same configuration and monitoring patterns
"""

# Core components
# Configuration
from .config import PartitionConfig, get_partition_config, set_partition_config
from .core.manager import PartitionManager
from .core.validator import PartitionValidator

# Exceptions
from .exceptions import (
    InvalidPartitionNameError,
    PartitionAlreadyExistsError,
    PartitionError,
    PartitionNotFoundError,
    PartitionOperationError,
)

# Models and entities
from .models.entities import (
    LoadProgress,
    PartitionDescription,
    PartitionLoadState,
    PartitionState,
    PartitionStats,
)

# Utilities
from .utils import (
    PartitionProgressTracker,
    PartitionTimer,
    get_global_progress_tracker,
    get_global_timer,
)

__version__ = "1.0.0"
__author__ = "Milvus Enterprise Ops Team"

# Public API
__all__ = [
    # Core components
    "PartitionManager",
    "PartitionValidator",
    # Models and entities
    "PartitionDescription",
    "PartitionStats",
    "LoadProgress",
    "PartitionLoadState",
    "PartitionState",
    # Configuration
    "get_partition_config",
    "set_partition_config",
    "PartitionConfig",
    # Utilities
    "PartitionProgressTracker",
    "PartitionTimer",
    "get_global_progress_tracker",
    "get_global_timer",
    # Exceptions
    "PartitionError",
    "PartitionNotFoundError",
    "PartitionAlreadyExistsError",
    "PartitionOperationError",
    "InvalidPartitionNameError",
    # Factory functions
    "get_partition_operations",
]

# Module metadata
__all__.extend(["__version__", "__author__"])


# Quick start function for convenience
async def quick_create_partition(
    collection_name: str, partition_name: str, connection_manager=None, **kwargs
) -> PartitionDescription:
    """
    Quick function to create a partition with minimal setup.

    Args:
        collection_name: Name of the collection
        partition_name: Name of the partition to create
        connection_manager: Optional connection manager (creates one if not provided)
        **kwargs: Additional parameters passed to create_partition

    Returns:
        PartitionDescription object

    Example:
        partition = await quick_create_partition(
            collection_name="my_collection",
            partition_name="user_data"
        )
    """
    from connection_management import ConnectionManager

    if connection_manager is None:
        connection_manager = ConnectionManager()
        should_close = True
    else:
        should_close = False

    try:
        manager = PartitionManager(connection_manager)
        return await manager.create_partition(
            collection_name=collection_name, partition_name=partition_name, **kwargs
        )
    finally:
        if should_close:
            connection_manager.close()


# Quick start function for convenience
def get_partition_operations(connection_manager=None):
    """
    Factory function to create a PartitionManager instance.

    Args:
        connection_manager: ConnectionManager instance

    Returns:
        PartitionManager instance

    Example:
        manager = get_partition_operations(connection_manager)
        partitions = await manager.list_partitions("my_collection")
    """
    if connection_manager is None:
        from connection_management import ConnectionManager

        connection_manager = ConnectionManager()

    return PartitionManager(connection_manager)


# Version compatibility
SUPPORTED_PYMILVUS_VERSIONS = ["2.0.0", "2.1.0", "2.2.0", "2.3.0", "2.4.0"]
