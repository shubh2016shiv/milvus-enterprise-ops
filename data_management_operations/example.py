"""
Example Usage of Data Management Operations

Demonstrates how to use the DataManager class with custom configuration,
error handling, and performance monitoring in a production-like scenario.
"""

import asyncio
import logging
from typing import List
import random
import sys
import os

# CRITICAL: Add parent directory to path to avoid circular imports
# This ensures we import from the root modules, not local ones
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import after path is fixed
from connection_management import ConnectionManager
from collection_operations import CollectionManager, CollectionSchema, FieldSchema, DataType, MetricType
from config import MilvusSettings

# Import from the module (using the public API)
from data_management_operations import (
    DataManager,
    DataOperationConfig,
    Document,
    BatchOperationResult,
    DeleteResult,
    OperationStatus,
    BatchPartialFailureError,
    SchemaValidationError,
    CollectionOperationError
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def create_test_collection(collection_manager: CollectionManager) -> None:
    """Create a test collection for demonstration."""
    logger.info("Creating test collection...")
    
    # Define schema
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ],
        description="Test collection for data management operations",
        enable_dynamic_field=False,
        shard_num=2
    )
    
    # Check if collection already exists
    if await collection_manager.has_collection("test_collection"):
        logger.info("Collection 'test_collection' already exists")
        return
    
    # Create collection
    await collection_manager.create_collection(
        collection_name="test_collection",
        schema=schema
    )
    
    # Load collection
    await collection_manager.load_collection(
        collection_name="test_collection",
        wait=True
    )
    
    logger.info("Test collection created and loaded successfully")


def generate_test_documents(count: int, start_id: int = 0) -> List[Document]:
    """Generate test documents for insertion."""
    documents = []
    for i in range(count):
        # Generate random vector
        vector = [random.random() for _ in range(128)]
        
        # Create document
        document = Document(
            id=start_id + i,
            vector=vector,
            text=f"This is test document {start_id + i}",
            metadata={"index": start_id + i, "type": "test"}
        )
        documents.append(document)
    
    return documents


async def demonstrate_basic_insertion(
    data_manager: DataManager,
    collection_name: str
):
    """Demonstrate basic document insertion."""
    logger.info("\n=== Basic Document Insertion ===")
    
    # Generate test documents
    documents = generate_test_documents(count=10)
    
    try:
        # Insert documents
        result = await data_manager.insert(
            collection_name=collection_name,
            documents=documents,
            validate=True
        )
        
        logger.info(f"Inserted {result.successful_count}/{len(documents)} documents")
        logger.info(f"Success rate: {result.success_rate:.2f}%")
        logger.info(f"Inserted IDs: {result.inserted_ids}")
        
    except SchemaValidationError as e:
        logger.error(f"Schema validation failed: {e}")
        for doc_id, errors in e.validation_errors.items():
            logger.error(f"Document {doc_id} errors: {errors}")
    except CollectionOperationError as e:
        logger.error(f"Collection operation failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)


async def demonstrate_batched_insertion(
    data_manager: DataManager,
    collection_name: str
):
    """Demonstrate batched insertion with custom batch size."""
    logger.info("\n=== Batched Document Insertion ===")
    
    # Generate larger set of documents
    documents = generate_test_documents(count=50, start_id=100)
    
    try:
        # Insert with custom batch size
        result = await data_manager.insert(
            collection_name=collection_name,
            documents=documents,
            batch_size=10,  # Process in batches of 10
            validate=True
        )
        
        logger.info(f"Batch insertion completed")
        logger.info(f"Total documents: {len(documents)}")
        logger.info(f"Successful: {result.successful_count}")
        logger.info(f"Failed: {result.failed_count}")
        logger.info(f"Success rate: {result.success_rate:.2f}%")
        
        if result.failed_count > 0:
            logger.warning(f"Failed documents: {list(result.error_messages.keys())}")
        
    except BatchPartialFailureError as e:
        logger.warning(f"Partial failure in batch insertion")
        logger.warning(f"Succeeded: {e.successful_count}")
        logger.warning(f"Failed: {e.failed_count}")
        logger.warning(f"Success rate: {e.success_rate:.2f}%")
        for doc_id, error in e.error_details.items():
            logger.error(f"Document {doc_id} failed: {error}")
    except Exception as e:
        logger.error(f"Batch insertion failed: {e}", exc_info=True)


async def demonstrate_upsert_operation(
    data_manager: DataManager,
    collection_name: str
):
    """Demonstrate upsert (insert or update) operation."""
    logger.info("\n=== Upsert Operation ===")
    
    # Generate documents (some may already exist)
    documents = generate_test_documents(count=5, start_id=5)
    
    try:
        # Upsert documents
        result = await data_manager.upsert(
            collection_name=collection_name,
            documents=documents,
            validate=True
        )
        
        logger.info(f"Upserted {result.successful_count} documents")
        logger.info(f"Operation status: {result.status}")
        
    except Exception as e:
        logger.error(f"Upsert failed: {e}", exc_info=True)


async def demonstrate_delete_operation(
    data_manager: DataManager,
    collection_name: str
):
    """Demonstrate delete operation."""
    logger.info("\n=== Delete Operation ===")
    
    try:
        # Delete documents with IDs >= 100 and < 110
        result = await data_manager.delete(
            collection_name=collection_name,
            expr="id >= 100 and id < 110"
        )
        
        logger.info(f"Deleted {result.deleted_count} documents")
        logger.info(f"Operation status: {result.status}")
        
    except Exception as e:
        logger.error(f"Delete failed: {e}", exc_info=True)


async def demonstrate_performance_monitoring(data_manager: DataManager):
    """Demonstrate performance monitoring features."""
    logger.info("\n=== Performance Monitoring ===")
    
    # Get timing history
    timing_history = data_manager.get_timing_history()
    logger.info(f"Total operations tracked: {len(timing_history)}")
    
    # Get stats for specific operations
    insert_stats = data_manager.get_operation_stats("insert_documents")
    if insert_stats:
        logger.info(f"\nInsert operations statistics:")
        logger.info(f"  Total operations: {insert_stats.total_operations}")
        logger.info(f"  Success rate: {insert_stats.success_rate:.2f}%")
        logger.info(f"  Average time: {insert_stats.average_execution_time*1000:.2f}ms")
        logger.info(f"  Median time: {insert_stats.median_execution_time*1000:.2f}ms")
        logger.info(f"  P95 time: {insert_stats.p95_execution_time*1000:.2f}ms")
    
    # Get summary of all operations
    summary = data_manager.get_performance_summary()
    logger.info(f"\nPerformance summary ({len(summary)} operation types):")
    for op_name, stats in summary.items():
        logger.info(f"  {op_name}: {stats.total_operations} ops, "
                   f"avg={stats.average_execution_time*1000:.2f}ms")


async def main():
    """Main demonstration function."""
    print("=" * 60)
    print("Starting Data Management Operations Example")
    print("=" * 60)
    logger.info("Starting Data Management Operations Example")
    
    try:
        # 1. Create custom configuration
        logger.info("\n=== Configuration ===")
        config = DataOperationConfig(
            default_batch_size=500,
            retry_transient_errors=True,
            default_operation_timeout=60.0,
            enable_timing=True,
            strict_validation=True
        )
        logger.info(f"Configuration created: batch_size={config.default_batch_size}, "
                   f"timing_enabled={config.enable_timing}")
        
        # 2. Initialize connections (adjust connection details as needed)
        logger.info("\n=== Initializing Connections ===")
        from config.settings import ConnectionSettings
        connection_settings = ConnectionSettings(host="localhost", port="19530")
        milvus_settings = MilvusSettings(connection=connection_settings)
        connection_manager = ConnectionManager(config=milvus_settings)
        collection_manager = CollectionManager(connection_manager=connection_manager)
        logger.info("Connection and Collection managers initialized")
        
        # 3. Initialize DataManager with custom configuration
        data_manager = DataManager(
            connection_manager=connection_manager,
            collection_manager=collection_manager,
            config=config
        )
        logger.info("DataManager initialized with custom configuration")
        
        # 4. Test connection to Milvus
        logger.info("\n=== Testing Milvus Connection ===")
        is_connected = connection_manager.check_server_status()
        if not is_connected:
            logger.error("❌ Failed to connect to Milvus server. Please ensure Milvus is running.")
            logger.error("   To start Milvus, run: docker-compose up -d")
            return
        logger.info("✓ Successfully connected to Milvus server")
        
        # 5. Create test collection
        await create_test_collection(collection_manager)
        
        # 6. Run demonstrations
        collection_name = "test_collection"
        
        await demonstrate_basic_insertion(data_manager, collection_name)
        await demonstrate_batched_insertion(data_manager, collection_name)
        await demonstrate_upsert_operation(data_manager, collection_name)
        await demonstrate_delete_operation(data_manager, collection_name)
        await demonstrate_performance_monitoring(data_manager)
        
        print("\n" + "=" * 60)
        logger.info("=== Example Completed Successfully ===")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Example failed with error: {e}", exc_info=True)
        print("\n" + "=" * 60)
        print("Example failed - see logs above for details")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())