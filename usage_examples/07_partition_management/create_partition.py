"""
Create Partition Example

Demonstrates how to create partitions in a Milvus collection using the
Partition Operations module. Shows both standard creation and idempotent creation.
"""

import sys
import os
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from connection_management import ConnectionManager
from partition_operations import PartitionManager
from milvus_ops.collection_operations import CollectionManager, CollectionSchema, FieldSchema, DataType
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


async def create_test_collection(collection_manager: CollectionManager):
    """Create a test collection for partition operations."""
    try:
        # Define schema
        pk_field = FieldSchema(
            name="pk",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=False,
            description="Primary key"
        )

        vector_field = FieldSchema(
            name="vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=VECTOR_DIM,
            description="Vector embeddings"
        )

        text_field = FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="Text content"
        )

        # Create collection
        schema = CollectionSchema(
            fields=[pk_field, vector_field, text_field],
            description="Test collection for partition operations"
        )

        await collection_manager.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema
        )

        print_success(f"Created test collection '{COLLECTION_NAME}'")
        return True

    except Exception as e:
        print_error(f"Failed to create test collection: {e}")
        return False


async def main():
    """Main function to demonstrate partition creation."""
    print_section("Create Partition Example")

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

    # Step 2: Create Test Collection
    print_step(2, "Create Test Collection")
    if not await coll_manager.has_collection(COLLECTION_NAME):
        success = await create_test_collection(coll_manager)
        if not success:
            conn_manager.close()
            return
    else:
        print_info("Status", f"Collection '{COLLECTION_NAME}' already exists")

    # Step 3: Create Partition (Standard Method)
    print_step(3, "Create Partition (Standard Method)")
    partition_name_1 = "users_q1_2024"
    try:
        partition = await partition_manager.create_partition(
            collection_name=COLLECTION_NAME,
            partition_name=partition_name_1,
            timeout=30.0
        )
        print_success(f"Created partition '{partition.name}'")
        print_info("Partition ID", partition.partition_id)
        print_info("Created at", partition.created_at.isoformat() if partition.created_at else "N/A")
    except Exception as e:
        print_error(f"Failed to create partition: {e}")
        # Continue to show other methods

    # Step 4: Create Partition (Idempotent Method)
    print_step(4, "Create Partition (Idempotent Method)")
    partition_name_2 = "users_q2_2024"
    try:
        # This method won't fail if partition already exists
        partition = await partition_manager.create_partition_if_not_exists(
            collection_name=COLLECTION_NAME,
            partition_name=partition_name_2,
            timeout=30.0
        )
        print_success(f"Partition '{partition.name}' is ready")
        print_info("Partition ID", partition.partition_id)
        print_info("Created at", partition.created_at.isoformat() if partition.created_at else "N/A")
    except Exception as e:
        print_error(f"Failed to create partition: {e}")

    # Step 5: Verify Partitions Exist
    print_step(5, "Verify Partitions Exist")
    try:
        partitions = await partition_manager.list_partitions(COLLECTION_NAME)
        print_info("Total partitions", len(partitions))

        for partition_name in partitions:
            if partition_name in [partition_name_1, partition_name_2]:
                exists = await partition_manager.partition_exists(
                    COLLECTION_NAME, partition_name
                )
                status = "EXISTS" if exists else "NOT FOUND"
                print_info(f"Partition '{partition_name}'", status)
    except Exception as e:
        print_error(f"Failed to verify partitions: {e}")

    # Step 6: Demonstrate Error Handling
    print_step(6, "Demonstrate Error Handling")
    try:
        # Try to create partition with invalid name
        await partition_manager.create_partition(
            collection_name=COLLECTION_NAME,
            partition_name="invalid name with spaces",
            timeout=30.0
        )
    except Exception as e:
        print_info("Expected error", f"Invalid partition name: {type(e).__name__}")

    try:
        # Try to create partition in non-existent collection
        await partition_manager.create_partition(
            collection_name="non_existent_collection",
            partition_name="test_partition",
            timeout=30.0
        )
    except Exception as e:
        print_info("Expected error", f"Collection not found: {type(e).__name__}")

    # Step 7: Create Multiple Partitions
    print_step(7, "Create Multiple Partitions")
    try:
        partition_names = ["products_electronics", "products_clothing", "products_books"]
        results = await partition_manager.create_multiple_partitions(
            collection_name=COLLECTION_NAME,
            partition_names=partition_names,
            timeout=30.0
        )

        print_success(f"Created {len(results)}/{len(partition_names)} partitions")

        for result in results:
            print_info(f"Created", f"'{result.name}' (ID: {result.partition_id})")
    except Exception as e:
        print_error(f"Failed to create multiple partitions: {e}")

    # Step 8: List All Partitions
    print_step(8, "List All Partitions")
    try:
        partitions = await partition_manager.list_partitions(COLLECTION_NAME)
        print_info("Total partitions", len(partitions))
        print("\n  All partitions:")
        for partition_name in sorted(partitions):
            is_default = "(default)" if partition_name == "_default" else ""
            print(f"    - {partition_name} {is_default}")
    except Exception as e:
        print_error(f"Failed to list partitions: {e}")

    # Step 9: Cleanup
    print_step(9, "Close Connection")
    try:
        conn_manager.close()
        print_success("Connection closed")
    except Exception as e:
        print_error(f"Cleanup failed: {e}")

    print_section("Example Completed")
    print("\nKey Takeaways:")
    print("  • Use create_partition_if_not_exists() for production code")
    print("  • Partition names must follow Milvus naming rules")
    print("  • Always verify collection exists before partition operations")
    print("  • Use create_multiple_partitions() for bulk operations")
    print("  • The '_default' partition exists automatically in every collection")


if __name__ == "__main__":
    asyncio.run(main())
