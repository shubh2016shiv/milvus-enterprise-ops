#!/usr/bin/env python3
"""
Simple test script for collection operations module.

Tests real Milvus operations without mocking - professional, modular, efficient.
"""

import asyncio
import logging
import numpy as np
from collection_operations import CollectionManager, CollectionSchema, FieldSchema, DataType
from connection_management import ConnectionManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_collection_creation(manager: CollectionManager) -> str:
    """Test collection creation with realistic schema."""
    logger.info("Testing collection creation...")
    
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ],
        description="Test collection for document embeddings"
    )
    
    collection = await manager.create_collection("test_docs", schema)
    logger.info(f"Created collection: {collection.name} (ID: {collection.id})")
    return collection.name


async def test_index_creation(manager: CollectionManager, name: str) -> None:
    """Test index creation on vector field."""
    logger.info(f"Testing index creation on: {name}")
    
    # This is a simplified example - in a real application, you would use
    # the index_operations module instead of direct pymilvus calls
    try:
        # Use ConnectionManager to execute the index creation in the correct thread
        await manager._connection_manager.execute_operation_async(
            lambda alias: _create_index_internal(alias, name)
        )
        logger.info(f"Created index on collection: {name}")
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        raise


def _create_index_internal(alias: str, collection_name: str) -> None:
    """Internal helper to create an index via the PyMilvus SDK."""
    from pymilvus import Collection
    
    # Get collection with the correct connection alias
    collection = Collection(name=collection_name, using=alias)
    
    # Define index parameters - using a simple IVF_FLAT index for testing
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    
    # Create index on the embedding field
    collection.create_index("embedding", index_params)
    logger.info(f"Index created on 'embedding' field")


async def test_collection_describe(manager: CollectionManager, name: str) -> None:
    """Test collection description retrieval."""
    logger.info(f"Testing collection description: {name}")
    
    desc = await manager.describe_collection(name)
    logger.info(f"Collection: {desc.name}, State: {desc.state}, Fields: {len(desc.schema.fields)}")


async def test_collection_load(manager: CollectionManager, name: str) -> None:
    """Test collection loading."""
    logger.info(f"Testing collection loading: {name}")
    
    # Non-blocking load with index error handling
    await manager.load_collection(name, wait=False, ignore_index_errors=True)
    
    # Check progress
    for _ in range(5):
        progress = await manager.get_load_progress(name)
        logger.info(f"Load progress: {progress.progress:.1%}")
        if progress.is_complete:
            break
        await asyncio.sleep(1)


async def test_data_insertion(manager: CollectionManager, name: str) -> None:
    """Test data insertion into collection."""
    logger.info(f"Testing data insertion into: {name}")
    
    # Prepare some test data
    num_entities = 5
    data = [
        # Auto-generated primary key (pk) is not provided
        # Title field
        ["Document " + str(i) for i in range(num_entities)],
        # Embedding field - random vectors
        [np.random.random(384).tolist() for _ in range(num_entities)],
        # Metadata field
        [{"source": "test", "timestamp": 1698765432} for _ in range(num_entities)]
    ]
    
    # Insert data using the manager
    insert_result = await manager.insert(name, data)
    logger.info(f"Inserted {num_entities} entities, IDs: {insert_result.primary_keys}")
    
    return insert_result.primary_keys


async def test_collection_stats(manager: CollectionManager, name: str) -> None:
    """Test collection statistics."""
    logger.info(f"Testing collection stats: {name}")
    
    stats = await manager.get_collection_stats(name)
    logger.info(f"Stats - Rows: {stats.row_count}, Memory: {stats.memory_size/1024/1024:.1f}MB")


async def test_collection_list(manager: CollectionManager) -> None:
    """Test collection listing."""
    logger.info("Testing collection listing...")
    
    # Explicitly await the result
    collections = await manager.list_collections()
    # Convert to list if needed and handle potential None result
    if collections is None:
        collections = []
    elif asyncio.iscoroutine(collections):
        collections = await collections
    logger.info(f"Found {len(collections)} collections: {collections}")


async def cleanup_collection(manager: CollectionManager, name: str) -> None:
    """Clean up test collection."""
    logger.info(f"Cleaning up collection: {name}")
    
    await manager.release_collection(name)
    logger.info(f"Released collection: {name}")


async def test_drop_collection(manager: CollectionManager, name: str) -> None:
    """Test collection dropping."""
    logger.info(f"Testing collection dropping: {name}")
    
    # Drop the collection
    result = await manager.drop_collection(name, safe=False)
    logger.info(f"Dropped collection: {name}, result: {result}")


async def main():
    """Main test execution."""
    logger.info("Starting collection operations test...")
    
    try:
        # Initialize connection
        conn_manager = ConnectionManager()
        collection_manager = CollectionManager(conn_manager)
        
        # First ensure clean state by dropping any existing test collection
        try:
            collections = await collection_manager.list_collections()
            if collections is not None and "test_docs" in collections:
                logger.info("Found existing test_docs collection, dropping it first...")
                await test_drop_collection(collection_manager, "test_docs")
        except Exception as e:
            logger.warning(f"Error during initial cleanup: {e}")
        
        # Run tests
        await test_collection_list(collection_manager)
        
        collection_name = await test_collection_creation(collection_manager)
        await test_collection_describe(collection_manager, collection_name)
        
        # Create an index on the vector field before loading
        # This is a critical step for Milvus to properly load the collection
        await test_index_creation(collection_manager, collection_name)
        
        # Now the collection can be properly loaded with an index
        await test_collection_load(collection_manager, collection_name)
        
        # Insert test data
        await test_data_insertion(collection_manager, collection_name)
        
        # Check stats after data insertion
        await test_collection_stats(collection_manager, collection_name)
        
        # Cleanup
        await cleanup_collection(collection_manager, collection_name)
        
        # Final cleanup - drop the collection completely
        await test_drop_collection(collection_manager, collection_name)
        
        # Verify collection is gone
        collections = await collection_manager.list_collections()
        if collections is None:
            collections = []
        elif asyncio.iscoroutine(collections):
            collections = await collections
        
        if collection_name in collections:
            logger.warning(f"Collection {collection_name} still exists after dropping!")
        else:
            logger.info(f"Verified collection {collection_name} was properly dropped")
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
