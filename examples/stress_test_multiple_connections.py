#!/usr/bin/env python3
"""
Multi-Connection Concurrent Stress Test

This script tests the system's ability to handle multiple independent clients,
each with its own connection manager, operating on Milvus concurrently. This
simulates a real-world microservices architecture where many applications
interact with the same database.
"""

import asyncio
import logging
import time
import numpy as np
from typing import List, Dict, Any
from collection_operations import CollectionManager, CollectionSchema, FieldSchema, DataType
from connection_management import ConnectionManager

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiConnectionStressTest:
    """
    Simulates multiple independent clients connecting and operating on Milvus.
    """
    
    def __init__(self, num_clients: int = 10, collections_per_client: int = 2):
        """
        Initialize the multi-connection stress test.
        
        Args:
            num_clients: The number of concurrent clients to simulate.
            collections_per_client: The number of collections each client will create.
        """
        self.num_clients = num_clients
        self.collections_per_client = collections_per_client
        self.results = {
            'total_collections_target': num_clients * collections_per_client,
            'success_count': 0,
            'failure_count': 0,
            'errors': [],
            'timing': {}
        }

    async def client_worker(self, client_id: int) -> Dict[str, Any]:
        """
        Represents a single, independent client application.
        
        This worker will create its own ConnectionManager and CollectionManager,
        then perform the full lifecycle of operations for its assigned collections.
        
        Args:
            client_id: The unique ID for this simulated client.
            
        Returns:
            A dictionary summarizing the results for this client.
        """
        logger.info(f"[Client-{client_id:02d}] Starting...")
        
        client_conn_manager = None
        client_results = {
            'client_id': client_id,
            'collections_created': 0,
            'collections_indexed': 0,
            'collections_loaded': 0,
            'collections_populated': 0,
            'collections_dropped': 0,
            'errors': []
        }
        
        try:
            # Each client gets its own independent managers
            client_conn_manager = ConnectionManager()
            client_collection_manager = CollectionManager(client_conn_manager)
            
            collection_names = [f"mc_test_{client_id:02d}_{i:02d}" for i in range(self.collections_per_client)]
            
            # CONCURRENCY FIX: Run complete lifecycle for each collection concurrently
            # This allows different collections to be in different phases simultaneously
            collection_tasks = []
            for name in collection_names:
                schema = self._create_schema(name)
                collection_tasks.append(
                    self._run_collection_lifecycle(client_collection_manager, name, schema)
                )
            
            # Run all collection lifecycles concurrently
            lifecycle_results = await asyncio.gather(*collection_tasks, return_exceptions=True)
            
            # Process results
            for res in lifecycle_results:
                if isinstance(res, dict) and res.get('success'):
                    client_results['collections_created'] += 1
                    client_results['collections_indexed'] += 1
                    client_results['collections_loaded'] += 1
                    client_results['collections_populated'] += 1
                    client_results['collections_dropped'] += 1
                else:
                    client_results['errors'].append(f"Collection lifecycle failed: {res}")

        except Exception as e:
            logger.error(f"[Client-{client_id:02d}] A fatal error occurred: {e}")
            client_results['errors'].append(f"Fatal client error: {e}")
        finally:
            if client_conn_manager:
                client_conn_manager.close()
            logger.info(f"[Client-{client_id:02d}] Finished.")
        
        return client_results

    def _create_schema(self, collection_name: str) -> CollectionSchema:
        """Creates a standard schema for a test collection."""
        return CollectionSchema(
            fields=[
                FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=8),
            ],
            description=f"Multi-client test for {collection_name}"
        )

    async def _run_collection_lifecycle(self, manager: CollectionManager, name: str, schema: CollectionSchema) -> Dict[str, Any]:
        """
        Run the complete lifecycle for a single collection: create → index → load → insert → cleanup.
        
        This method represents the full lifecycle of a collection and allows each collection
        to progress through all phases independently, maximizing concurrency.
        """
        try:
            # Phase 1: Create and Index
            await self._create_and_index_collection(manager, name, schema)
            
            # Phase 2: Load and Insert Data  
            await self._load_and_insert_data(manager, name)
            
            # Phase 3: Cleanup
            await self._cleanup_collection(manager, name)
            
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _create_and_index_collection(self, manager: CollectionManager, name: str, schema: CollectionSchema) -> Dict[str, Any]:
        """Helper to create a collection and then index it."""
        from pymilvus import Collection
        
        # CONCURRENCY FIX: Create collection first, then index it
        # These must be sequential because indexing requires the collection to exist
        await manager.create_collection(name, schema)
        
        index_params = {"metric_type": "L2", "index_type": "FLAT", "params": {}}
        
        # We need to execute the raw pymilvus call through the manager
        await manager._connection_manager.execute_operation_async(
            lambda alias: Collection(name, using=alias).create_index("embedding", index_params)
        )
        return {'success': True}

    async def _load_and_insert_data(self, manager: CollectionManager, name: str) -> Dict[str, Any]:
        """Helper to load a collection and insert data."""
        # CONCURRENCY FIX: Load and insert can run concurrently for different collections
        # but for a single collection, load must complete before insert
        await manager.load_collection(name, wait=True)
        
        data = [
            [np.random.random(8).tolist() for _ in range(10)]
        ]
        await manager.insert(name, data)
        return {'success': True}

    async def _cleanup_collection(self, manager: CollectionManager, name: str) -> Dict[str, Any]:
        """Helper to release and drop a collection."""
        try:
            # First, release the collection
            logger.info(f"Releasing collection: {name}")
            await manager.release_collection(name)
            
            # Then drop it with safe=False to force deletion even if there are issues
            logger.info(f"Dropping collection: {name}")
            result = await manager.drop_collection(name, safe=False)
            
            if result:
                logger.info(f"Successfully dropped collection: {name}")
                return {'success': True}
            else:
                logger.error(f"Failed to drop collection: {name} - drop_collection returned False")
                return {'success': False, 'error': "drop_collection returned False"}
        except Exception as e:
            logger.error(f"Error during cleanup of collection {name}: {e}")
            return {'success': False, 'error': str(e)}

    async def run_test(self):
        """Run the multi-client stress test."""
        logger.info(
            f"Starting multi-connection stress test with {self.num_clients} clients, "
            f"{self.collections_per_client} collections each."
        )
        start_time = time.time()

        try:
            # Create and run all client workers concurrently
            tasks = [self.client_worker(i) for i in range(self.num_clients)]
            all_client_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            self.results['timing']['total_duration'] = time.time() - start_time
            
            # Process results
            for res in all_client_results:
                if isinstance(res, Exception):
                    self.results['failure_count'] += self.collections_per_client
                    self.results['errors'].append(f"Client worker failed entirely: {res}")
                else:
                    successful_collections_in_client = res.get('collections_dropped', 0)
                    self.results['success_count'] += successful_collections_in_client
                    
                    failed_collections_in_client = self.collections_per_client - successful_collections_in_client
                    self.results['failure_count'] += failed_collections_in_client
                    
                    if res.get('errors'):
                        self.results['errors'].extend([f"[Client-{res['client_id']:02d}] {e}" for e in res['errors']])
        finally:
            # Ensure we clean up any leftover collections, even if the test fails
            logger.info("Running final cleanup to ensure no collections are left behind...")
            await self._emergency_cleanup()
            
        self.print_results()
        
    async def _emergency_cleanup(self):
        """Emergency cleanup to ensure no collections are left behind, even if the test fails."""
        try:
            # Create a temporary connection manager and collection manager
            conn_manager = ConnectionManager()
            coll_manager = CollectionManager(conn_manager)
            
            # List all collections
            all_collections = await coll_manager.list_collections()
            test_collections = [name for name in all_collections if name.startswith("mc_test_")]
            
            if test_collections:
                logger.warning(f"Found {len(test_collections)} leftover test collections. Cleaning up...")
                
                # Create cleanup tasks
                cleanup_tasks = []
                for name in test_collections:
                    cleanup_tasks.append(self._cleanup_collection(coll_manager, name))
                
                # Execute cleanup tasks
                if cleanup_tasks:
                    await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                    logger.info(f"Emergency cleanup completed for {len(cleanup_tasks)} collections.")
            else:
                logger.info("No leftover test collections found.")
                
        except Exception as e:
            logger.error(f"Error during emergency cleanup: {e}")
        finally:
            # Close the temporary connection manager
            if 'conn_manager' in locals():
                conn_manager.close()

    def print_results(self):
        """Print comprehensive test results."""
        print("\n" + "="*80)
        print("MULTI-CONNECTION STRESS TEST RESULTS")
        print("="*80)
        print(f"Simulated Clients: {self.num_clients}")
        print(f"Collections per Client: {self.collections_per_client}")
        print(f"Total Collections: {self.results['total_collections_target']}")
        print(f"Total Duration: {self.results['timing']['total_duration']:.2f}s")
        print()
        
        print("OPERATION RESULTS:")
        print(f"  Successful Collection Lifecycles: {self.results['success_count']}/{self.results['total_collections_target']}")
        print(f"  Failed Collection Lifecycles: {self.results['failure_count']}/{self.results['total_collections_target']}")
        
        if self.results['timing']['total_duration'] > 0:
            throughput = self.results['total_collections_target'] / self.results['timing']['total_duration']
            print(f"  Throughput: {throughput:.2f} collections/second")
        print()
        
        if self.results['errors']:
            print(f"ERRORS ({len(self.results['errors'])}):")
            for error in self.results['errors'][:15]:  # Show first 15 errors
                print(f"  {error}")
            if len(self.results['errors']) > 15:
                print(f"  ... and {len(self.results['errors']) - 15} more errors")
        else:
            print("ERRORS: None")
        
        print("="*80)

async def pre_cleanup(num_clients: int, collections_per_client: int):
    """Clean up any leftover collections from previous runs."""
    logger.info("Running pre-test cleanup...")
    manager = None
    try:
        conn_manager = ConnectionManager()
        manager = CollectionManager(conn_manager)
        all_collections = await manager.list_collections()
        
        if all_collections:
            tasks = []
            for i in range(num_clients):
                for j in range(collections_per_client):
                    name = f"mc_test_{i:02d}_{j:02d}"
                    if name in all_collections:
                        tasks.append(manager.drop_collection(name, safe=False))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                logger.info(f"Pre-cleaned {len(tasks)} leftover collections.")
    except Exception as e:
        logger.warning(f"Pre-cleanup failed, which is okay if it's the first run: {e}")
    finally:
        if manager and manager._connection_manager:
            manager._connection_manager.close()


async def main():
    """Main execution function."""
    
    # Configuration
    NUM_CLIENTS = 10
    COLLECTIONS_PER_CLIENT = 2

    await pre_cleanup(NUM_CLIENTS, COLLECTIONS_PER_CLIENT)
    
    test = MultiConnectionStressTest(
        num_clients=NUM_CLIENTS,
        collections_per_client=COLLECTIONS_PER_CLIENT
    )
    await test.run_test()

if __name__ == "__main__":
    asyncio.run(main())
