"""
List and Check Partitions Example

Demonstrates how to list partitions in a collection and check for partition
existence using the Partition Operations module.
"""

import sys
import os
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from connection_management import ConnectionManager
from partition_operations import PartitionManager
from collection_operations import CollectionManager
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


async def setup_test_partitions(partition_manager: PartitionManager):
    """Set up some test partitions for demonstration."""
    test_partitions = ["users_q1_2024", "users_q2_2024", "products_electronics", "products_books"]

    print_info("Setup", "Creating test partitions for demonstration")
    created = []

    for partition_name in test_partitions:
        try:
            exists = await partition_manager.partition_exists(COLLECTION_NAME, partition_name)
            if not exists:
                partition = await partition_manager.create_partition_if_not_exists(
                    collection_name=COLLECTION_NAME,
                    partition_name=partition_name,
                    timeout=30.0
                )
                created.append(partition_name)
                print_info("Created", f"'{partition_name}'")
            else:
                print_info("Exists", f"'{partition_name}'")
        except Exception as e:
            print_error(f"Failed to create '{partition_name}': {e}")

    if created:
        print_success(f"Created {len(created)} new partitions")
    else:
        print_info("All test partitions", "already exist")

    return test_partitions


async def main():
    """Main function to demonstrate partition listing and checking."""
    print_section("List and Check Partitions Example")

    # Step 1: Initialize Managers
    print_step(1, "Initialize Managers")
    try:
        config = load_settings()
        conn_manager = ConnectionManager(config=config)
        coll_manager = CollectionManager(conn_manager)
        partition_manager = PartitionManager(conn_manager)
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

    # Step 3: Setup Test Partitions
    print_step(3, "Setup Test Partitions")
    await setup_test_partitions(partition_manager)

    # Step 4: List All Partitions
    print_step(4, "List All Partitions")
    try:
        partitions = await partition_manager.list_partitions(
            collection_name=COLLECTION_NAME,
            timeout=10.0
        )
        print_success(f"Found {len(partitions)} partitions")

        print("\n  Partition List:")
        for i, partition_name in enumerate(sorted(partitions), 1):
            is_default = " (DEFAULT)" if partition_name == "_default" else ""
            print(f"    {i:2d}. {partition_name}{is_default}")
    except Exception as e:
        print_error(f"Failed to list partitions: {e}")

    # Step 5: Check Specific Partition Existence
    print_step(5, "Check Specific Partition Existence")
    test_checks = [
        ("_default", "Should always exist"),
        ("users_q1_2024", "Created in setup"),
        ("non_existent_partition", "Should not exist"),
        ("users_q2_2024", "Created in setup")
    ]

    print("\n  Partition Existence Checks:")
    for partition_name, description in test_checks:
        try:
            exists = await partition_manager.partition_exists(
                collection_name=COLLECTION_NAME,
                partition_name=partition_name,
                timeout=5.0
            )
            status = "EXISTS" if exists else "NOT FOUND"
            print(f"    - {partition_name}: {status} ({description})")
        except Exception as e:
            print_error(f"Failed to check '{partition_name}': {e}")

    # Step 6: Get Detailed Partition Information
    print_step(6, "Get Detailed Partition Information")
    partitions_to_check = ["_default", "users_q1_2024", "users_q2_2024"]

    for partition_name in partitions_to_check:
        try:
            exists = await partition_manager.partition_exists(COLLECTION_NAME, partition_name)
            if exists:
                print(f"\n  Details for partition '{partition_name}':")
                info = await partition_manager.get_partition_info(
                    collection_name=COLLECTION_NAME,
                    partition_name=partition_name,
                    timeout=10.0
                )
                print(f"    Name: {info.name}")
                print(f"    Collection: {info.collection_name}")
                print(f"    State: {info.state}")
                print(f"    Load State: {info.load_state}")
                print(f"    Created Time: {info.created_at.isoformat() if info.created_at else 'N/A'}")
            else:
                print_info(f"Partition '{partition_name}'", "does not exist")
        except Exception as e:
            print_error(f"Failed to get info for '{partition_name}': {e}")

    # Step 7: Demonstrate Error Handling
    print_step(7, "Demonstrate Error Handling")
    try:
        # Try to list partitions from non-existent collection
        partitions = await partition_manager.list_partitions("non_existent_collection")
        print_info("Non-existent collection", f"Returned {len(partitions)} partitions")
    except Exception as e:
        print_info("Error handling", f"Expected error: {type(e).__name__}")

    try:
        # Try to check partition in non-existent collection
        exists = await partition_manager.partition_exists("non_existent_collection", "test")
        print_info("Non-existent collection check", f"Returned: {exists}")
    except Exception as e:
        print_info("Error handling", f"Expected error: {type(e).__name__}")

    # Step 8: Performance Comparison
    print_step(8, "Performance Comparison")
    import time

    # Test list_partitions performance
    start_time = time.time()
    for _ in range(10):
        await partition_manager.list_partitions(COLLECTION_NAME)
    list_time = time.time() - start_time

    # Test individual partition_exists calls
    partitions_to_check = ["_default", "users_q1_2024", "users_q2_2024", "products_electronics"]
    start_time = time.time()
    for partition_name in partitions_to_check:
        await partition_manager.partition_exists(COLLECTION_NAME, partition_name)
    exists_time = time.time() - start_time

    print(".3f")
    print(".3f")
    print_note("list_partitions() is more efficient for bulk checks")

    # Step 9: Cleanup
    print_step(9, "Close Connection")
    try:
        conn_manager.close()
        print_success("Connection closed")
    except Exception as e:
        print_error(f"Cleanup failed: {e}")

    print_section("Example Completed")
    print("\nKey Takeaways:")
    print("  • Use list_partitions() to get all partitions in a collection")
    print("  • Use partition_exists() to check specific partition existence")
    print("  • get_partition_info() provides detailed partition metadata")
    print("  • The '_default' partition always exists in every collection")
    print("  • list_partitions() is more efficient than multiple exists() calls")
    print("  • Operations handle missing collections gracefully")


if __name__ == "__main__":
    asyncio.run(main())
