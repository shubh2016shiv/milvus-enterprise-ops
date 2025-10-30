"""
Load and Unload Partitions Example

Demonstrates how to load and unload partitions in Milvus collections using the
Partition Operations module. Shows both synchronous and asynchronous loading
with progress monitoring.
"""

import sys
import os
import asyncio
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from connection_management import ConnectionManager
from partition_operations import PartitionManager
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
print_note = example_utils.print_note


COLLECTION_NAME = "test_partition_collection"
VECTOR_DIM = 128


async def insert_test_data_simple(partition_name: str, count: int = 100):
    """Insert test data into a specific partition using pymilvus directly."""
    try:
        from pymilvus import Collection
        import numpy as np

        # Generate test data
        vectors = np.random.rand(count, VECTOR_DIM).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # Normalize

        data = {
            "pk": list(range(1, count + 1)),
            "vector": vectors.tolist(),
            "text": [f"Sample text {i} in {partition_name}" for i in range(1, count + 1)]
        }

        # Insert directly using pymilvus
        collection = Collection(COLLECTION_NAME)
        collection.insert(data, partition_name=partition_name, timeout=60.0)

        print_success(f"Inserted {count} entities into partition '{partition_name}'")
        return True

    except Exception as e:
        print_error(f"Failed to insert data into '{partition_name}': {e}")
        return False


async def monitor_load_progress(partition_manager: PartitionManager, collection_name: str, partition_name: str):
    """Monitor load progress with real-time updates."""
    print(f"\n  Monitoring load progress for '{partition_name}':")

    start_time = time.time()
    last_progress = -1

    while True:
        try:
            progress = await partition_manager.get_load_progress(
                collection_name=collection_name,
                partition_name=partition_name,
                timeout=10.0
            )

            current_progress = int(progress.percentage_complete * 100)

            # Only print if progress changed significantly
            if current_progress != last_progress:
                elapsed = time.time() - start_time
                print(f"    Progress: {current_progress:3d}% "
                      f"(segments: {progress.loaded_segments}/{progress.total_segments}) "
                      ".1f")

                if progress.state in ["LOADED", "FAILED"]:
                    break

                last_progress = current_progress

            await asyncio.sleep(0.5)  # Check every 0.5 seconds

        except Exception as e:
            print_error(f"Failed to get load progress: {e}")
            break

    return progress


async def main():
    """Main function to demonstrate partition loading and unloading."""
    print_section("Load and Unload Partitions Example")

    # Step 1: Initialize Managers
    print_step(1, "Initialize Managers")
    try:
        config = load_settings()
        conn_manager = ConnectionManager(config=config)
        coll_manager = CollectionManager(conn_manager)
        partition_manager = PartitionManager(conn_manager)
        data_config = DataOperationConfig(default_batch_size=100)
        data_manager = DataManager(conn_manager, coll_manager, config=data_config)
        print_success("Managers initialized")
    except Exception as e:
        print_error(f"Initialization failed: {e}")
        return

    # Step 2: Verify Collection Exists
    print_step(2, "Verify Collection Exists")
    try:
        exists = await coll_manager.has_collection(COLLECTION_NAME)
        if not exists:
            print_error(f"Collection '{COLLECTION_NAME}' does not exist")
            print_note("Run create_partition.py first to create the test collection")
            conn_manager.close()
            return
        print_success(f"Collection '{COLLECTION_NAME}' exists")
    except Exception as e:
        print_error(f"Failed to check collection: {e}")
        conn_manager.close()
        return

    # Step 3: Setup Test Partition with Data
    print_step(3, "Setup Test Partition with Data")
    test_partition = "load_test_partition"

    try:
        # Create partition if it doesn't exist
        partition = await partition_manager.create_partition_if_not_exists(
            collection_name=COLLECTION_NAME,
            partition_name=test_partition,
            timeout=30.0
        )
        print_success(f"Partition '{test_partition}' is ready")

        # Insert test data
        await insert_test_data_simple(test_partition, count=500)

        # Load the collection first (required for partition operations)
        await coll_manager.load_collection(COLLECTION_NAME)
        print_info("Collection loaded", "Ready for partition operations")

    except Exception as e:
        print_error(f"Failed to setup test partition: {e}")
        conn_manager.close()
        return

    # Step 4: Check Initial Load State
    print_step(4, "Check Initial Load State")
    try:
        info = await partition_manager.get_partition_info(COLLECTION_NAME, test_partition)
        print_info("Initial load state", info.load_state)
    except Exception as e:
        print_error(f"Failed to get partition info: {e}")

    # Step 5: Load Partition (Asynchronous)
    print_step(5, "Load Partition (Asynchronous)")
    try:
        success = await partition_manager.load_partition(
            collection_name=COLLECTION_NAME,
            partition_name=test_partition,
            wait=False,  # Don't wait for completion
            timeout=120.0
        )

        if success:
            print_success("Load operation initiated successfully")

            # Monitor progress manually
            final_progress = await monitor_load_progress(
                partition_manager, COLLECTION_NAME, test_partition
            )

            if final_progress and final_progress.is_successful:
                print_success("Partition loaded successfully")
            else:
                print_error(f"Load failed: {final_progress.error_message if final_progress else 'Unknown error'}")
        else:
            print_error("Failed to initiate load operation")

    except Exception as e:
        print_error(f"Load operation failed: {e}")

    # Step 6: Load Partition (Synchronous)
    print_step(6, "Load Partition (Synchronous)")
    try:
        # First unload to test loading again
        await partition_manager.release_partition(COLLECTION_NAME, test_partition)
        print_info("Partition unloaded", "for re-loading test")

        # Load with wait=True
        print_info("Loading", "with synchronous wait...")
        progress = await partition_manager.load_partition(
            collection_name=COLLECTION_NAME,
            partition_name=test_partition,
            wait=True,  # Wait for completion
            timeout=300.0
        )

        if progress.is_successful:
            print_success("Partition loaded synchronously")
            print_info("Final progress", f"{progress.percentage_complete:.1%}")
            print_info("Loaded segments", f"{progress.loaded_segments}/{progress.total_segments}")
        else:
            print_error(f"Synchronous load failed: {progress.error_message}")

    except Exception as e:
        print_error(f"Synchronous load failed: {e}")

    # Step 7: Load Multiple Partitions
    print_step(7, "Load Multiple Partitions")
    try:
        # Create a few more partitions with data
        partitions_to_load = ["multi_load_1", "multi_load_2"]

        for partition_name in partitions_to_load:
            await partition_manager.create_partition_if_not_exists(COLLECTION_NAME, partition_name)
            await insert_test_data_simple(partition_name, count=100)

        print_info("Created partitions", f"{', '.join(partitions_to_load)}")

        # Load multiple partitions
        load_tasks = []
        for partition_name in partitions_to_load:
            task = partition_manager.load_partition(
                collection_name=COLLECTION_NAME,
                partition_name=partition_name,
                wait=True,
                timeout=120.0
            )
            load_tasks.append(task)

        # Execute all loads concurrently
        results = await asyncio.gather(*load_tasks, return_exceptions=True)

        success_count = 0
        for i, result in enumerate(results):
            partition_name = partitions_to_load[i]
            if isinstance(result, Exception):
                print_error(f"Failed to load '{partition_name}': {result}")
            else:
                success_count += 1
                print_success(f"Loaded '{partition_name}'")

        print_info("Multi-load result", f"{success_count}/{len(partitions_to_load)} successful")

    except Exception as e:
        print_error(f"Multi-partition load failed: {e}")

    # Step 8: Check Load States
    print_step(8, "Check Load States")
    try:
        all_partitions = ["_default", test_partition] + partitions_to_load

        print("\n  Load states for all partitions:")
        for partition_name in all_partitions:
            try:
                info = await partition_manager.get_partition_info(COLLECTION_NAME, partition_name)
                state = info.load_state
                print(f"    - {partition_name}: {state}")
            except Exception as e:
                print_error(f"Failed to get info for '{partition_name}': {e}")

    except Exception as e:
        print_error(f"Failed to check load states: {e}")

    # Step 9: Unload Partitions
    print_step(9, "Unload Partitions")
    try:
        partitions_to_unload = [test_partition] + partitions_to_load

        for partition_name in partitions_to_unload:
            try:
                success = await partition_manager.release_partition(
                    collection_name=COLLECTION_NAME,
                    partition_name=partition_name,
                    timeout=30.0
                )

                if success:
                    print_success(f"Unloaded partition '{partition_name}'")
                else:
                    print_error(f"Failed to unload '{partition_name}'")
            except Exception as e:
                print_error(f"Error unloading '{partition_name}': {e}")

        # Verify unload states
        print("\n  Verifying unload states:")
        for partition_name in partitions_to_unload:
            try:
                info = await partition_manager.get_partition_info(COLLECTION_NAME, partition_name)
                print_info(f"'{partition_name}' load state", info.load_state)
            except Exception as e:
                print_error(f"Failed to verify '{partition_name}': {e}")

    except Exception as e:
        print_error(f"Unload operations failed: {e}")

    # Step 10: Demonstrate Error Handling
    print_step(10, "Demonstrate Error Handling")
    try:
        # Try to load non-existent partition
        success = await partition_manager.load_partition(
            collection_name=COLLECTION_NAME,
            partition_name="non_existent_partition",
            wait=False
        )
        print_info("Non-existent partition", f"Load attempt returned: {success}")
    except Exception as e:
        print_info("Expected error", f"Loading non-existent partition: {type(e).__name__}")

    try:
        # Try to load partition from non-existent collection
        success = await partition_manager.load_partition(
            collection_name="non_existent_collection",
            partition_name="test",
            wait=False
        )
        print_info("Non-existent collection", f"Load attempt returned: {success}")
    except Exception as e:
        print_info("Expected error", f"Loading from non-existent collection: {type(e).__name__}")

    # Step 11: Cleanup
    print_step(11, "Close Connection")
    try:
        conn_manager.close()
        print_success("Connection closed")
    except Exception as e:
        print_error(f"Cleanup failed: {e}")

    print_section("Example Completed")
    print("\nKey Takeaways:")
    print("  • Partitions must be loaded before querying data")
    print("  • Use load_partition(wait=False) for async operations")
    print("  • Use load_partition(wait=True) when you need immediate readiness")
    print("  • Monitor progress with get_load_progress()")
    print("  • release_partition() frees memory resources")
    print("  • Load operations can be performed concurrently")
    print("  • Check load_state to verify partition status")


if __name__ == "__main__":
    asyncio.run(main())
