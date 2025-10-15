"""
Example Usage of Index Operations

Demonstrates how to use the IndexManager class with custom configuration,
error handling, and progress monitoring in a production-like scenario.
"""

import asyncio
import logging
from typing import List

from connection_management import ConnectionManager
from collection_operations import CollectionManager, CollectionSchema, FieldSchema, DataType, MetricType
from config import MilvusSettings

# Import from the module (using the public API)
from . import (
    IndexManager,
    IndexOperationConfig,
    IndexType,
    IndexState,
    HNSWParams,
    IvfFlatParams,
    IndexParameterError,
    IndexBuildError,
    IndexNotFoundError
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demonstrate_basic_index_creation(
    index_manager: IndexManager,
    collection_name: str
):
    """Demonstrate basic index creation."""
    logger.info("\n=== Basic Index Creation ===")
    
    try:
        # Create a simple FLAT index
        result = await index_manager.create_index(
            collection_name=collection_name,
            field_name="embedding",
            index_type=IndexType.FLAT,
            metric_type=MetricType.COSINE,
            wait=True
        )
        
        if result.success:
            logger.info(f"Index created successfully!")
            logger.info(f"Execution time: {result.execution_time_ms:.2f}ms")
        else:
            logger.error(f"Index creation failed: {result.error_message}")
            
    except IndexParameterError as e:
        logger.error(f"Invalid parameters: {e}")
    except IndexBuildError as e:
        logger.error(f"Build failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


async def demonstrate_custom_parameters(
    index_manager: IndexManager,
    collection_name: str
):
    """Demonstrate index creation with custom parameters."""
    logger.info("\n=== Index Creation with Custom Parameters ===")
    
    try:
        # Create HNSW index with custom parameters
        hnsw_params = HNSWParams(
            M=16,
            efConstruction=200
        )
        
        result = await index_manager.create_index(
            collection_name=collection_name,
            field_name="embedding",
            index_type=IndexType.HNSW,
            metric_type=MetricType.COSINE,
            index_params=hnsw_params,
            wait=False  # Don't wait, we'll monitor progress
        )
        
        if result.success:
            logger.info(f"Index creation initiated")
            logger.info(f"State: {result.state}")
        
    except IndexParameterError as e:
        logger.error(f"Invalid parameters: {e}")
        for param, error in e.parameter_errors.items():
            logger.error(f"  {param}: {error}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


async def demonstrate_progress_monitoring(
    index_manager: IndexManager,
    collection_name: str
):
    """Demonstrate progress monitoring."""
    logger.info("\n=== Progress Monitoring ===")
    
    try:
        # Create IVF_FLAT index without waiting
        ivf_params = IvfFlatParams(nlist=1024)
        
        result = await index_manager.create_index(
            collection_name=collection_name,
            field_name="embedding",
            index_type=IndexType.IVF_FLAT,
            metric_type=MetricType.L2,
            index_params=ivf_params,
            wait=False
        )
        
        if not result.success:
            logger.error(f"Failed to initiate index creation: {result.error_message}")
            return
        
        logger.info(f"Index creation initiated, monitoring progress...")
        
        # Monitor progress
        max_polls = 60  # Maximum number of polls
        poll_count = 0
        
        while poll_count < max_polls:
            progress = await index_manager.get_index_build_progress(
                collection_name=collection_name,
                field_name="embedding"
            )
            
            logger.info(f"Progress: {progress.percentage:.2f}% complete")
            
            if progress.formatted_eta:
                logger.info(f"ETA: {progress.formatted_eta}")
            
            if progress.state == IndexState.CREATED:
                logger.info("Index build complete!")
                break
            elif progress.state == IndexState.FAILED:
                logger.error(f"Index build failed: {progress.failed_reason}")
                break
            
            await asyncio.sleep(2)
            poll_count += 1
        
        if poll_count >= max_polls:
            logger.warning("Reached maximum poll count, stopping monitoring")
        
    except Exception as e:
        logger.error(f"Error during progress monitoring: {e}")


async def demonstrate_index_describe(
    index_manager: IndexManager,
    collection_name: str
):
    """Demonstrate getting index information."""
    logger.info("\n=== Describe Index ===")
    
    try:
        index_info = await index_manager.describe_index(
            collection_name=collection_name,
            field_name="embedding"
        )
        
        logger.info(f"Index Information:")
        logger.info(f"  Name: {index_info.index_name}")
        logger.info(f"  Type: {index_info.index_type}")
        logger.info(f"  Metric: {index_info.metric_type}")
        logger.info(f"  State: {index_info.state}")
        logger.info(f"  Parameters: {index_info.params}")
        
        if index_info.index_size_mb:
            logger.info(f"  Size: {index_info.index_size_mb:.2f} MB")
        
        if index_info.is_available:
            logger.info("  Status: Index is available for use")
        
    except IndexNotFoundError as e:
        logger.error(f"Index not found: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


async def demonstrate_list_indexes(
    index_manager: IndexManager,
    collection_name: str
):
    """Demonstrate listing all indexes in a collection."""
    logger.info("\n=== List All Indexes ===")
    
    try:
        indexes = await index_manager.list_indexes(collection_name=collection_name)
        
        logger.info(f"Found {len(indexes)} index(es) in collection '{collection_name}':")
        
        for index in indexes:
            logger.info(f"  Field: {index.field_name}")
            logger.info(f"    Type: {index.index_type}")
            logger.info(f"    Metric: {index.metric_type}")
            logger.info(f"    State: {index.state}")
        
        if not indexes:
            logger.info(f"  No indexes found")
        
    except Exception as e:
        logger.error(f"Failed to list indexes: {e}")


async def demonstrate_check_index(
    index_manager: IndexManager,
    collection_name: str
):
    """Demonstrate checking if a field has an index."""
    logger.info("\n=== Check if Field has Index ===")
    
    field_name = "embedding"
    
    try:
        has_index = await index_manager.has_index(
            collection_name=collection_name,
            field_name=field_name
        )
        
        if has_index:
            logger.info(f"Field '{field_name}' has an index")
        else:
            logger.info(f"Field '{field_name}' does not have an index")
        
    except Exception as e:
        logger.error(f"Error checking index: {e}")


async def demonstrate_drop_index(
    index_manager: IndexManager,
    collection_name: str
):
    """Demonstrate dropping an index."""
    logger.info("\n=== Drop Index ===")
    
    try:
        result = await index_manager.drop_index(
            collection_name=collection_name,
            field_name="embedding"
        )
        
        if result.success:
            logger.info(f"Index dropped successfully")
            logger.info(f"Execution time: {result.execution_time_ms:.2f}ms")
        else:
            logger.error(f"Failed to drop index: {result.error_message}")
        
    except IndexNotFoundError as e:
        logger.error(f"Index not found: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


async def main():
    """Main demonstration function."""
    logger.info("Starting Index Operations Example")
    
    try:
        # 1. Create custom configuration
        logger.info("\n=== Configuration ===")
        config = IndexOperationConfig(
            default_timeout=120.0,
            build_progress_poll_interval=2.0,
            enable_timing=True,
            retry_transient_errors=True
        )
        logger.info(f"Configuration created: timeout={config.default_timeout}s, "
                   f"timing_enabled={config.enable_timing}")
        
        # 2. Initialize connections (adjust connection details as needed)
        milvus_settings = MilvusSettings()
        connection_manager = ConnectionManager(config=milvus_settings)
        collection_manager = CollectionManager(connection_manager=connection_manager)
        
        # 3. Initialize IndexManager with custom configuration
        index_manager = IndexManager(
            connection_manager=connection_manager,
            collection_manager=collection_manager,
            config=config
        )
        logger.info("IndexManager initialized with custom configuration")
        
        # 4. Set collection name
        collection_name = "test_collection"
        
        # Note: Ensure the collection exists before running these examples
        # You may need to create it first using CollectionManager
        
        # 5. Run demonstrations
        await demonstrate_basic_index_creation(index_manager, collection_name)
        await demonstrate_custom_parameters(index_manager, collection_name)
        await demonstrate_progress_monitoring(index_manager, collection_name)
        await demonstrate_index_describe(index_manager, collection_name)
        await demonstrate_list_indexes(index_manager, collection_name)
        await demonstrate_check_index(index_manager, collection_name)
        # await demonstrate_drop_index(index_manager, collection_name)  # Uncomment to test
        
        logger.info("\n=== Example Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Example failed with error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
