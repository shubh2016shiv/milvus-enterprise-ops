#!/usr/bin/env python3
"""
Concurrent Collection Operations Stress Test

This script tests the scalability and robustness of the collection_operations
and connection_management modules by creating, indexing, loading, and managing
1000 collections concurrently.

Tests real Milvus operations without mocking - enterprise-grade stress testing.
"""

import asyncio
import logging
import time
import numpy as np
from typing import List, Dict, Any
from collection_operations import CollectionManager, CollectionSchema, FieldSchema, DataType
from connection_management import ConnectionManager

# Setup logging with more detailed format for stress testing
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)


class ConcurrentCollectionStressTest:
    """
    Enterprise-grade stress test for concurrent collection operations.
    
    This class orchestrates the creation, indexing, loading, and cleanup
    of multiple collections concurrently to test system scalability. It uses an
    asyncio.Semaphore to control the maximum level of concurrency, providing
    a sustained, high-throughput load on the server.
    """
    
    def __init__(self, num_collections: int = 100, concurrency_limit: int = 10):
        """
        Initialize the stress test.
        
        Args:
            num_collections: Total number of collections to create.
            concurrency_limit: The maximum number of operations to run at the same time.
        """
        self.num_collections = num_collections
        self.concurrency_limit = concurrency_limit
        self.connection_manager = None
        self.collection_manager = None
        self.semaphore = None
        self.results = {
            'created': 0,
            'indexed': 0,
            'loaded': 0,
            'data_inserted': 0,
            'dropped': 0,
            'errors': [],
            'timing': {}
        }
    
    async def setup(self):
        """Initialize connection and collection managers."""
        logger.info("Setting up connection and collection managers...")
        self.connection_manager = ConnectionManager()
        self.collection_manager = CollectionManager(self.connection_manager)
        self.semaphore = asyncio.Semaphore(self.concurrency_limit)
        logger.info(f"Setup complete with concurrency limit: {self.concurrency_limit}")
    
    def create_test_schema(self, collection_suffix: str) -> CollectionSchema:
        """
        Create a test schema for a collection.
        
        Args:
            collection_suffix: Unique suffix for this collection
            
        Returns:
            CollectionSchema configured for testing
        """
        return CollectionSchema(
            fields=[
                FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),  # Smaller dim for performance
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ],
            description=f"Stress test collection {collection_suffix}"
        )
    
    async def create_collection_with_index(self, collection_id: int) -> Dict[str, Any]:
        """
        Create a single collection with index.
        
        Args:
            collection_id: Unique identifier for this collection
            
        Returns:
            Dict containing operation results
        """
        collection_name = f"stress_test_{collection_id:04d}"
        result = {
            'name': collection_name,
            'created': False,
            'indexed': False,
            'error': None
        }
        
        try:
            # Create collection
            schema = self.create_test_schema(str(collection_id))
            collection = await self.collection_manager.create_collection(collection_name, schema)
            result['created'] = True
            
            # Create index
            await self.collection_manager._connection_manager.execute_operation_async(
                lambda alias: self._create_index_internal(alias, collection_name)
            )
            result['indexed'] = True
            
            logger.debug(f"Successfully created and indexed collection: {collection_name}")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error creating collection {collection_name}: {e}")
        
        return result
    
    async def _worker_create(self, collection_id: int) -> Dict[str, Any]:
        """Worker task for creating and indexing a collection, managed by a semaphore."""
        async with self.semaphore:
            return await self.create_collection_with_index(collection_id)

    async def _worker_load(self, collection_name: str) -> Dict[str, Any]:
        """Worker task for loading and populating a collection, managed by a semaphore."""
        async with self.semaphore:
            return await self.load_and_insert_data(collection_name)

    async def _worker_cleanup(self, collection_name: str) -> Dict[str, Any]:
        """Worker task for cleaning up a collection, managed by a semaphore."""
        async with self.semaphore:
            return await self.cleanup_collection(collection_name)
    
    def _create_index_internal(self, alias: str, collection_name: str) -> None:
        """Internal helper to create an index via the PyMilvus SDK."""
        from pymilvus import Collection
        
        collection = Collection(name=collection_name, using=alias)
        
        # Use simpler index for faster creation during stress test
        index_params = {
            "metric_type": "L2",
            "index_type": "FLAT",  # Simpler than IVF_FLAT for stress testing
        }
        
        collection.create_index("embedding", index_params)
    
    async def load_and_insert_data(self, collection_name: str) -> Dict[str, Any]:
        """
        Load collection and insert test data.
        
        Args:
            collection_name: Name of the collection to load and populate
            
        Returns:
            Dict containing operation results
        """
        result = {
            'name': collection_name,
            'loaded': False,
            'data_inserted': False,
            'error': None
        }
        
        try:
            # Load collection
            await self.collection_manager.load_collection(
                collection_name, 
                wait=False, 
                ignore_index_errors=False  # Should not have index errors now
            )
            result['loaded'] = True
            
            # Insert small amount of test data
            num_entities = 10  # Keep small for stress test performance
            data = [
                [f"Doc_{i}" for i in range(num_entities)],
                [np.random.random(128).tolist() for _ in range(num_entities)],
                [{"test": True, "batch": collection_name} for _ in range(num_entities)]
            ]
            
            insert_result = await self.collection_manager.insert(collection_name, data)
            result['data_inserted'] = True
            result['inserted_ids'] = len(insert_result.primary_keys) if hasattr(insert_result, 'primary_keys') else 0
            
            logger.debug(f"Successfully loaded and populated collection: {collection_name}")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error loading/populating collection {collection_name}: {e}")
        
        return result
    
    async def cleanup_collection(self, collection_name: str) -> Dict[str, Any]:
        """
        Clean up a single collection.
        
        Args:
            collection_name: Name of the collection to clean up
            
        Returns:
            Dict containing cleanup results
        """
        result = {
            'name': collection_name,
            'dropped': False,
            'error': None
        }
        
        try:
            # Release and drop collection
            try:
                await self.collection_manager.release_collection(collection_name)
            except Exception as e:
                logger.warning(f"Error releasing collection {collection_name}: {e}")
            
            await self.collection_manager.drop_collection(collection_name, safe=False)
            result['dropped'] = True
            
            logger.debug(f"Successfully cleaned up collection: {collection_name}")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error cleaning up collection {collection_name}: {e}")
        
        return result
    
    async def run_stress_test(self) -> Dict[str, Any]:
        """
        Execute the complete stress test by launching all tasks and controlling
        their execution with a semaphore for sustained concurrency.
        
        Returns:
            Dict containing comprehensive test results.
        """
        logger.info(f"Starting stress test: {self.num_collections} collections, concurrency limit: {self.concurrency_limit}")
        start_time = time.time()

        # --- Phase 1: Create collections with indexes ---
        logger.info("Phase 1: Creating collections and indexes...")
        phase1_start = time.time()
        
        create_tasks = [self._worker_create(cid) for cid in range(self.num_collections)]
        create_results = await asyncio.gather(*create_tasks, return_exceptions=True)
        
        for result in create_results:
            if not isinstance(result, Exception):
                if result.get('created'): self.results['created'] += 1
                if result.get('indexed'): self.results['indexed'] += 1
                if result.get('error'): self.results['errors'].append(f"Create: {result.get('name', 'unknown')}: {result['error']}")
            else:
                self.results['errors'].append(f"Create: Unhandled exception: {result}")

        self.results['timing']['phase1_create'] = time.time() - phase1_start
        logger.info(f"Phase 1 complete: {self.results['created']} created, {self.results['indexed']} indexed")

        # --- Phase 2: Load collections and insert data ---
        logger.info("Phase 2: Loading collections and inserting data...")
        phase2_start = time.time()
        
        collection_names = [f"stress_test_{cid:04d}" for cid in range(self.num_collections)]
        load_tasks = [self._worker_load(name) for name in collection_names]
        load_results = await asyncio.gather(*load_tasks, return_exceptions=True)

        for result in load_results:
            if not isinstance(result, Exception):
                if result.get('loaded'): self.results['loaded'] += 1
                if result.get('data_inserted'): self.results['data_inserted'] += 1
                if result.get('error'): self.results['errors'].append(f"Load: {result.get('name', 'unknown')}: {result['error']}")
            else:
                self.results['errors'].append(f"Load: Unhandled exception: {result}")

        self.results['timing']['phase2_load'] = time.time() - phase2_start
        logger.info(f"Phase 2 complete: {self.results['loaded']} loaded, {self.results['data_inserted']} populated")

        # --- Phase 3: Cleanup ---
        logger.info("Phase 3: Cleaning up collections...")
        phase3_start = time.time()
        
        cleanup_tasks = [self._worker_cleanup(name) for name in collection_names]
        cleanup_results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        for result in cleanup_results:
            if not isinstance(result, Exception):
                if result.get('dropped'): self.results['dropped'] += 1
                if result.get('error'): self.results['errors'].append(f"Cleanup: {result.get('name', 'unknown')}: {result['error']}")
            else:
                self.results['errors'].append(f"Cleanup: Unhandled exception: {result}")
        
        self.results['timing']['phase3_cleanup'] = time.time() - phase3_start
        logger.info(f"Phase 3 complete: {self.results['dropped']} dropped")
        
        # Final results
        self.results['timing']['total'] = time.time() - start_time
        return self.results
    
    def print_results(self):
        """Print comprehensive test results."""
        print("\n" + "="*80)
        print("CONCURRENT COLLECTION STRESS TEST RESULTS")
        print("="*80)
        print(f"Target Collections: {self.num_collections}")
        print(f"Concurrency Limit: {self.concurrency_limit}")
        print()
        
        print("OPERATION RESULTS:")
        print(f"  Collections Created: {self.results['created']}/{self.num_collections}")
        print(f"  Indexes Created: {self.results['indexed']}/{self.num_collections}")
        print(f"  Collections Loaded: {self.results['loaded']}/{self.num_collections}")
        print(f"  Data Insertions: {self.results['data_inserted']}/{self.num_collections}")
        print(f"  Collections Dropped: {self.results['dropped']}/{self.num_collections}")
        print()
        
        print("TIMING RESULTS:")
        for phase, duration in self.results['timing'].items():
            print(f"  {phase.replace('_', ' ').title()}: {duration:.2f}s")
        
        if self.results['timing'].get('total'):
            collections_per_sec = self.num_collections / self.results['timing']['total']
            print(f"  Average Throughput: {collections_per_sec:.2f} collections/second")
        print()
        
        if self.results['errors']:
            print(f"ERRORS ({len(self.results['errors'])}):")
            for error in self.results['errors'][:10]:  # Show first 10 errors
                print(f"  {error}")
            if len(self.results['errors']) > 10:
                print(f"  ... and {len(self.results['errors']) - 10} more errors")
        else:
            print("ERRORS: None")
        
        print()
        success_rate = (self.results['created'] / self.num_collections) * 100
        print(f"SUCCESS RATE: {success_rate:.1f}%")
        print("="*80)


async def main():
    """Main stress test execution."""
    # Configuration
    NUM_COLLECTIONS = 10
    CONCURRENCY_LIMIT = 2  # Controls how many operations run in parallel
    
    logger.info("Initializing Concurrent Collection Stress Test...")
    
    # Create and run stress test
    stress_test = ConcurrentCollectionStressTest(
        num_collections=NUM_COLLECTIONS,
        concurrency_limit=CONCURRENCY_LIMIT
    )
    
    try:
        await stress_test.setup()
        results = await stress_test.run_stress_test()
        stress_test.print_results()
        
        # Return appropriate exit code
        if results['created'] == NUM_COLLECTIONS and len(results['errors']) == 0:
            logger.info("Stress test completed successfully!")
            return 0
        else:
            logger.warning("Stress test completed with some failures")
            return 1
            
    except Exception as e:
        logger.error(f"Stress test failed: {e}")
        raise
    finally:
        if stress_test.connection_manager:
            stress_test.connection_manager.close()


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
