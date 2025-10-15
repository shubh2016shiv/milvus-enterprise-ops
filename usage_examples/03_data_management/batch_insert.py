"""
Batch Insert Example

Demonstrates how to insert large amounts of data efficiently using batching.
Shows progress monitoring and handling of partial failures.
"""

import sys
import os
import asyncio
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from connection_management import ConnectionManager
from collection_operations import CollectionManager
from data_management_operations import DataManager, DataOperationConfig
from config import load_settings
# Import usage_examples utils (not the project's utils package)
import importlib.util
utils_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils.py'))
spec = importlib.util.spec_from_file_location("example_utils", utils_file_path)
example_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example_utils)
print_section = example_utils.print_section
print_step = example_utils.print_step
print_success = example_utils.print_success
print_info = example_utils.print_info
print_error = example_utils.print_error
generate_test_data = example_utils.generate_test_data


COLLECTION_NAME = "test_example_collection"
VECTOR_DIM = 128
TOTAL_ENTITIES = 1000
BATCH_SIZE = 100


def _flush_collection(alias: str, collection_name: str):
    """Helper function to flush collection."""
    from pymilvus import Collection
    collection = Collection(collection_name, using=alias)
    collection.flush()

def _query_collection(alias: str, collection_name: str, expr: str, output_fields: list, limit: int = 16384):
    """
    Helper function to query collection.

    CRITICAL FIX: Added 'limit' parameter with Milvus maximum (16,384) to prevent
    incomplete results. PyMilvus defaults to a lower limit causing incorrect entity
    counts and missing IDs in existence checks. For counting all entities, use
    get_collection_stats() API instead - querying "pk >= 0" is unreliable.
    """
    from pymilvus import Collection
    collection = Collection(collection_name, using=alias)
    return collection.query(expr=expr, output_fields=output_fields, limit=limit)


async def main():
    """Main function to demonstrate batch insertion."""
    print_section("Batch Insert Data Example")
    
    # Step 1: Initialize Managers
    print_step(1, "Initialize Managers")
    try:
        config = load_settings()
        conn_manager = ConnectionManager(config=config)
        coll_manager = CollectionManager(conn_manager)
        data_config = DataOperationConfig(default_batch_size=BATCH_SIZE)
        data_manager = DataManager(conn_manager, coll_manager, config=data_config)
        print_success("Managers initialized")
        print_info("Batch size", BATCH_SIZE)
    except Exception as e:
        print_error(f"Initialization failed: {e}")
        return
    
    # Step 2: Verify Collection Exists
    print_step(2, "Verify Collection Exists")
    try:
        if not await coll_manager.has_collection(COLLECTION_NAME):
            print_error(f"Collection '{COLLECTION_NAME}' does not exist")
            print_info("Hint", "Run create_collection.py first")
            conn_manager.close()
            return
        print_success(f"Collection '{COLLECTION_NAME}' found")
    except Exception as e:
        print_error(f"Collection check failed: {e}")
        conn_manager.close()
        return
    
    # Step 3: Get Initial Count
    print_step(3, "Get Initial Entity Count")
    try:
        # Load collection to ensure it's ready for querying
        description = await coll_manager.describe_collection(COLLECTION_NAME)
        if description.load_state.value != "Loaded":
            await coll_manager.load_collection(COLLECTION_NAME, wait=True)
        
        # Use a direct query to count entities
        query_results = await conn_manager.execute_operation_async(
            lambda alias: _query_collection(
                alias,
                COLLECTION_NAME,
                "pk >= 0",  # Match all entities
                ["pk"]
            )
        )
        initial_count = len(query_results)
        print_info("Initial entity count", initial_count)
    except Exception as e:
        print_error(f"Count retrieval failed: {e}")
        initial_count = 0
    
    # Step 4: Generate Large Dataset
    print_step(4, "Generate Test Data")
    try:
        print_info("Generating", f"{TOTAL_ENTITIES} entities...")
        start_time = time.time()
        
        data = generate_test_data(TOTAL_ENTITIES, VECTOR_DIM, include_metadata=True)
        
        # Adjust IDs to start after existing entities
        data['id'] = [i + initial_count + 1 for i in range(TOTAL_ENTITIES)]
        
        generation_time = time.time() - start_time
        print_success(f"Generated {TOTAL_ENTITIES} entities in {generation_time:.2f}s")
    except Exception as e:
        print_error(f"Data generation failed: {e}")
        conn_manager.close()
        return
    
    # Step 5: Batch Insert with Progress
    print_step(5, "Insert Data in Batches")
    try:
        # Convert to documents
        # Note: The collection uses 'pk' as the primary key field name
        documents = []
        for i in range(TOTAL_ENTITIES):
            doc = {
                "pk": data["id"][i],  # Map 'id' from test data to 'pk' for the collection
                "vector": data["vector"][i],
                "text": data["text"][i],
                "value": data["value"][i],
                "category": data["category"][i]
            }
            documents.append(doc)
        
        print_info("Total batches", TOTAL_ENTITIES // BATCH_SIZE)
        start_time = time.time()
        
        result = await data_manager.insert(
            collection_name=COLLECTION_NAME,
            documents=documents,
            batch_size=BATCH_SIZE
        )
        
        insertion_time = time.time() - start_time
        
        print_success(f"Inserted {result.successful_count} entities")
        print_info("Failed", result.failed_count)
        print_info("Total time", f"{insertion_time:.2f}s")
        print_info("Throughput", f"{result.successful_count / insertion_time:.0f} entities/sec")
        
        if result.failed_count > 0:
            print_error(f"{result.failed_count} entities failed")
            print_info("Failed IDs", str(result.failed_ids[:10]))
    except Exception as e:
        print_error(f"Batch insertion failed: {e}")
        conn_manager.close()
        return
    
    # Step 6: Flush and Verify
    print_step(6, "Flush and Verify Data")
    try:
        print_info("Flushing", "Ensuring data is persisted...")
        
        # Use ConnectionManager to execute flush operation
        await conn_manager.execute_operation_async(
            lambda alias: _flush_collection(alias, COLLECTION_NAME)
        )
        print_success("Flush completed")

        # Load collection to ensure it's ready for querying
        description = await coll_manager.describe_collection(COLLECTION_NAME)
        if description.load_state.value != "Loaded":
            print_info("Loading", "Collection needs to be loaded for verification")
            await coll_manager.load_collection(COLLECTION_NAME, wait=True)

        # ============================================================================
        # CRITICAL FIX: Verify sample of inserted IDs instead of counting all rows
        # ============================================================================
        # THREE PROBLEMS WE'RE AVOIDING:
        # 1. Stats API (get_collection_stats) returns STALE/CACHED data after flush
        # 2. Querying "pk >= 0" hits 16,384 result limit - fails for large collections
        # 3. For 1000 inserts, we can't verify all if collection already has >15,384 rows
        #
        # SOLUTION: Query a sample of inserted IDs directly (reliable & efficient)
        # ============================================================================
        sample_size = min(100, len(result.inserted_ids))
        sample_ids_str = ', '.join(map(str, result.inserted_ids[:sample_size]))
        verify_results = await conn_manager.execute_operation_async(
            lambda alias: _query_collection(
                alias,
                COLLECTION_NAME,
                f"pk in [{sample_ids_str}]",
                ["pk"],
                limit=sample_size + 10  # Small buffer for safety
            )
        )

        verified_count = len(verify_results) if verify_results else 0

        print_info("Sample size", sample_size)
        print_info("Verified", verified_count)
        print_info("Total inserted", result.successful_count)

        if verified_count >= sample_size:
            print_success("Sample verification passed - data successfully persisted")
        else:
            print_error(f"Only {verified_count} of {sample_size} sample entities verified")
    except Exception as e:
        print_error(f"Verification failed: {e}")
    
    # Step 7: Performance Summary
    print_step(7, "Performance Summary")
    try:
        print("\n  Performance Metrics:")
        print(f"    Entities/Second: {result.successful_count / insertion_time:.0f}")
        print(f"    Time per Batch: {insertion_time / (TOTAL_ENTITIES // BATCH_SIZE):.3f}s")
        print(f"    Success Rate: {(result.successful_count / TOTAL_ENTITIES) * 100:.1f}%")
        
        print_success("Performance metrics calculated")
    except Exception as e:
        print_error(f"Performance calculation failed: {e}")
    
    # Step 8: Cleanup
    print_step(8, "Close Connection")
    try:
        conn_manager.close()
        print_success("Connection closed")
    except Exception as e:
        print_error(f"Cleanup failed: {e}")
    
    print_section("Example Completed")
    print("\nKey Takeaways:")
    print("  • Batching improves insertion performance")
    print("  • Monitor success rate for large insertions")
    print("  • Flush ensures data persistence")
    print("  • Handle partial failures gracefully")
    print("\nBest Practices:")
    print("  • Use appropriate batch sizes (100-1000)")
    print("  • Monitor memory usage for large batches")
    print("  • Always verify final entity count")


if __name__ == "__main__":
    asyncio.run(main())
