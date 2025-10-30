"""
Delete Partitions Example

Demonstrates how to safely delete partitions using the Partition Operations module.
Shows both single partition deletion and batch deletion operations with proper
safety checks and error handling.
"""

import sys
import os
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from connection_management import ConnectionManager
from partition_operations import PartitionManager
from milvus_ops.collection_operations import CollectionManager
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


async def create_test_partitions_for_deletion(partition_manager: PartitionManager):
    """Create several test partitions that can be safely deleted."""
    partitions_to_create = [
        "temp_partition_1",
        "temp_partition_2",
        "temp_partition_3",
        "disposable_data"
    ]

    print_info("Setup", "Creating test partitions for deletion demonstration")

    for partition_name in partitions_to_create:
        try:
            # Create partition
            await partition_manager.create_partition_if_not_exists(
                collection_name=COLLECTION_NAME,
                partition_name=partition_name,
                timeout=30.0
            )

            print_success(f"Created partition '{partition_name}'")

        except Exception as e:
            print_error(f"Failed to create '{partition_name}': {e}")

    return partitions_to_create


async def demonstrate_safe_deletion(partition_manager: PartitionManager, partition_name: str):
    """Demonstrate safe partition deletion with all safety checks."""
    print(f"\n  === Safe Deletion Process for '{partition_name}' ===")

    try:
        # Step 1: Check if partition exists
        print("    1. Checking if partition exists...")
        exists = await partition_manager.partition_exists(
            collection_name=COLLECTION_NAME,
            partition_name=partition_name,
            timeout=10.0
        )

        if not exists:
            print_info("Result", f"Partition '{partition_name}' does not exist")
            return False

        print_success("Partition exists")

        # Step 2: Get statistics before deletion
        print("    2. Gathering statistics before deletion...")
        stats = await partition_manager.get_partition_stats(
            collection_name=COLLECTION_NAME,
            partition_name=partition_name,
            timeout=30.0
        )

        print_info("Entities to delete", f"{stats.row_count:,}")
        print_info("Total size", f"{stats.total_size / (1024**2):.2f} MB")

        # Step 3: Confirm deletion (in real scenarios, you'd have user confirmation)
        print("    3. Confirming deletion...")
        print_note("In production code, add user confirmation here")

        # Step 4: Perform deletion
        print("    4. Deleting partition...")
        success = await partition_manager.delete_partition(
            collection_name=COLLECTION_NAME,
            partition_name=partition_name,
            timeout=60.0
        )

        if success:
            print_success(f"Partition '{partition_name}' deleted successfully")
            return True
        else:
            print_error(f"Deletion of '{partition_name}' failed")
            return False

    except Exception as e:
        print_error(f"Error during deletion process: {e}")
        return False


async def main():
    """Main function to demonstrate partition deletion."""
    print_section("Delete Partitions Example")

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

    # Step 3: Create Test Partitions for Deletion
    print_step(3, "Create Test Partitions for Deletion")
    test_partitions = await create_test_partitions_for_deletion(partition_manager)

    # Step 4: Demonstrate Safe Single Partition Deletion
    print_step(4, "Demonstrate Safe Single Partition Deletion")
    target_partition = "temp_partition_1"

    success = await demonstrate_safe_deletion(partition_manager, target_partition)

    if success:
        # Verify deletion
        exists = await partition_manager.partition_exists(COLLECTION_NAME, target_partition)
        status = "NOT FOUND" if not exists else "STILL EXISTS"
        print_info("Verification", f"Partition '{target_partition}': {status}")

    # Step 5: Attempt to Delete Default Partition (Should Fail)
    print_step(5, "Attempt to Delete Default Partition")
    try:
        print_info("Attempting", "to delete _default partition (should be protected)...")
        success = await partition_manager.delete_partition(
            collection_name=COLLECTION_NAME,
            partition_name="_default",
            timeout=30.0
        )

        if success:
            print_error("Unexpected: Default partition was deleted!")
        else:
            print_success("Default partition deletion was blocked (as expected)")

    except Exception as e:
        print_info("Expected protection", f"Cannot delete default partition: {type(e).__name__}")

    # Step 6: Delete Multiple Partitions (Batch Operation)
    print_step(6, "Delete Multiple Partitions (Batch Operation)")
    remaining_partitions = [p for p in test_partitions if p != target_partition]

    if remaining_partitions:
        print_info("Batch deletion", f"Deleting {len(remaining_partitions)} partitions: {', '.join(remaining_partitions)}")

        try:
            # Get statistics before batch deletion
            total_entities = 0
            for partition_name in remaining_partitions:
                stats = await partition_manager.get_partition_stats(COLLECTION_NAME, partition_name)
                total_entities += stats.row_count

            print_info("Total entities to delete", f"{total_entities:,}")

            # Perform batch deletion
            deleted = await partition_manager.delete_multiple_partitions(
                collection_name=COLLECTION_NAME,
                partition_names=remaining_partitions,
                timeout=60.0,
                continue_on_error=True  # Continue if one fails
            )

            print_success(f"Successfully deleted {len(deleted)} partitions: {', '.join(deleted)}")

            # Check for failures
            failed = set(remaining_partitions) - set(deleted)
            if failed:
                print_error(f"Failed to delete: {', '.join(failed)}")

        except Exception as e:
            print_error(f"Batch deletion failed: {e}")
    else:
        print_info("No partitions", "available for batch deletion")

    # Step 7: Verify All Deletions
    print_step(7, "Verify All Deletions")
    try:
        all_partitions = await partition_manager.list_partitions(COLLECTION_NAME)
        deleted_partitions = set(test_partitions)
        remaining_test_partitions = deleted_partitions.intersection(set(all_partitions))

        print(f"\n  Original test partitions: {len(test_partitions)}")
        print(f"  Remaining test partitions: {len(remaining_test_partitions)}")

        if remaining_test_partitions:
            print("  Still exist:", ", ".join(sorted(remaining_test_partitions)))
        else:
            print_success("All test partitions successfully deleted")

        print(f"\n  Total partitions in collection: {len(all_partitions)}")
        print("  All partitions:", ", ".join(sorted(all_partitions)))

    except Exception as e:
        print_error(f"Failed to verify deletions: {e}")

    # Step 8: Demonstrate Error Handling
    print_step(8, "Demonstrate Error Handling")
    try:
        # Try to delete non-existent partition
        success = await partition_manager.delete_partition(
            collection_name=COLLECTION_NAME,
            partition_name="already_deleted_partition",
            timeout=30.0
        )
        print_info("Non-existent partition", f"Deletion attempt returned: {success}")
    except Exception as e:
        print_info("Expected error", f"Deleting non-existent partition: {type(e).__name__}")

    try:
        # Try to delete partition from non-existent collection
        success = await partition_manager.delete_partition(
            collection_name="non_existent_collection",
            partition_name="test",
            timeout=30.0
        )
        print_info("Non-existent collection", f"Deletion attempt returned: {success}")
    except Exception as e:
        print_info("Expected error", f"Deleting from non-existent collection: {type(e).__name__}")

    # Step 9: Demonstrate Batch Deletion with Errors
    print_step(9, "Demonstrate Batch Deletion with Errors")
    try:
        # Mix of existing and non-existing partitions
        mixed_partitions = ["non_existent_1", "non_existent_2", "_default"]

        print_info("Mixed batch", "Deleting mix of existing/non-existing partitions")

        deleted = await partition_manager.delete_multiple_partitions(
            collection_name=COLLECTION_NAME,
            partition_names=mixed_partitions,
            timeout=30.0,
            continue_on_error=True
        )

        print_info("Successfully deleted", f"{len(deleted)} partitions")
        if deleted:
            print("  Deleted:", ", ".join(deleted))

        expected_failures = len(mixed_partitions) - len(deleted)
        print_info("Expected failures", f"{expected_failures} (non-existent + protected partitions)")

    except Exception as e:
        print_error(f"Mixed batch deletion failed: {e}")

    # Step 10: Cleanup
    print_step(10, "Close Connection")
    try:
        conn_manager.close()
        print_success("Connection closed")
    except Exception as e:
        print_error(f"Cleanup failed: {e}")

    print_section("Example Completed")
    print("\nKey Takeaways:")
    print("  • Always verify partition exists before deletion")
    print("  • Get statistics before deletion for auditing/logging")
    print("  • Default partition (_default) cannot be deleted")
    print("  • Deletion is irreversible - implement confirmation in production")
    print("  • Use delete_multiple_partitions() for bulk operations")
    print("  • continue_on_error=True allows partial success in batch operations")
    print("  • Always verify deletion success")
    print("  • Handle errors gracefully in production code")


if __name__ == "__main__":
    asyncio.run(main())
