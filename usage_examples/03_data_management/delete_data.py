"""
Delete Data Example

Demonstrates how to delete entities from a collection using IDs
and filter expressions.
"""

import sys
import os
import asyncio

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


COLLECTION_NAME = "test_example_collection"


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
    """Main function to demonstrate deletion operations."""
    print_section("Delete Data Example")
    
    # Step 1: Initialize Managers
    print_step(1, "Initialize Managers")
    try:
        config = load_settings()
        conn_manager = ConnectionManager(config=config)
        coll_manager = CollectionManager(conn_manager)
        data_manager = DataManager(conn_manager, coll_manager, config=DataOperationConfig())
        print_success("Managers initialized")
    except Exception as e:
        print_error(f"Initialization failed: {e}")
        return
    
    # Step 2: Verify Collection Exists
    print_step(2, "Verify Collection Exists and Has Data")
    try:
        if not await coll_manager.has_collection(COLLECTION_NAME):
            print_error(f"Collection '{COLLECTION_NAME}' does not exist")
            print_info("Hint", "Run create_collection.py and insert_data.py first")
            conn_manager.close()
            return
        
        # Load collection to enable queries
        description = await coll_manager.describe_collection(COLLECTION_NAME)
        if description.load_state.value != "Loaded":
            await coll_manager.load_collection(COLLECTION_NAME, wait=True)
        
        # Get actual count by querying (stats may be cached)
        results = await conn_manager.execute_operation_async(
            lambda alias: _query_collection(alias, COLLECTION_NAME, "pk >= 0", ["pk"])
        )
        initial_count = len(results) if results else 0
        
        if initial_count == 0:
            print_error("Collection is empty, nothing to delete")
            conn_manager.close()
            return
        
        print_success(f"Collection has {initial_count} entities")
    except Exception as e:
        print_error(f"Collection check failed: {e}")
        conn_manager.close()
        return
    
    # Step 3: Delete by IDs
    print_step(3, "Delete Entities by IDs")
    try:
        # Delete entities with PKs 1, 2, 3
        ids_to_delete = [1, 2, 3]
        expr = f"pk in [{', '.join(map(str, ids_to_delete))}]"
        
        result = await data_manager.delete(
            collection_name=COLLECTION_NAME,
            expr=expr
        )
        
        print_success(f"Deleted {result.deleted_count} entities by IDs")
        print_info("Deleted IDs", ids_to_delete)
        print_info("Status", result.status)
    except Exception as e:
        print_error(f"Delete by IDs failed: {e}")
    
    # Step 4: Delete by PK List Expression
    print_step(4, "Delete Entities by PK List Expression")
    try:
        # Delete entities with PKs 10-15
        # Note: Milvus only supports 'pk in [...]' format for deletion
        pks_to_delete = list(range(10, 16))  # [10, 11, 12, 13, 14, 15]
        expression = f"pk in [{', '.join(map(str, pks_to_delete))}]"
        
        result = await data_manager.delete(
            collection_name=COLLECTION_NAME,
            expr=expression
        )
        
        print_success(f"Deleted {result.deleted_count} entities by PK list")
        print_info("Expression", expression)
        print_info("Status", result.status)
    except Exception as e:
        print_error(f"Delete by expression failed: {e}")
    
    # Step 5: Flush and Verify
    print_step(5, "Flush and Verify Deletions")
    try:
        # Use ConnectionManager to execute flush operation
        await conn_manager.execute_operation_async(
            lambda alias: _flush_collection(alias, COLLECTION_NAME)
        )
        
        # Get actual count by querying
        results = await conn_manager.execute_operation_async(
            lambda alias: _query_collection(alias, COLLECTION_NAME, "pk >= 0", ["pk"])
        )
        final_count = len(results) if results else 0
        
        print_info("Initial count", initial_count)
        print_info("Final count", final_count)
        print_info("Total deleted", initial_count - final_count)
        print_success("Deletions persisted")
    except Exception as e:
        print_error(f"Verification failed: {e}")
    
    # Step 6: Verify Specific Entity is Deleted
    print_step(6, "Verify Specific Entity is Deleted")
    try:
        # Ensure collection is loaded
        description = await coll_manager.describe_collection(COLLECTION_NAME)
        if description.load_state.value != "Loaded":
            await coll_manager.load_collection(COLLECTION_NAME, wait=True)
        
        # Try to query deleted entity (PK=1)
        results = await conn_manager.execute_operation_async(
            lambda alias: _query_collection(alias, COLLECTION_NAME, "pk == 1", ["pk"])
        )
        
        if not results:
            print_success("Confirmed: Entity with PK=1 is deleted")
        else:
            print_error("Entity with PK=1 still exists")
    except Exception as e:
        print_error(f"Verification query failed: {e}")
    
    # Step 7: Delete by PK List (Another Expression Example)
    print_step(7, "Delete by PK List (Expression Example)")
    try:
        # Delete entities with specific PKs
        # Note: Milvus only supports deletion by primary key expressions
        pks_to_delete = [20, 21, 22, 23, 24]
        expression = f"pk in [{', '.join(map(str, pks_to_delete))}]"
        
        result = await data_manager.delete(
            collection_name=COLLECTION_NAME,
            expr=expression
        )
        
        print_success(f"Deleted {result.deleted_count} entities by PK list")
        print_info("Expression", expression)
    except Exception as e:
        print_error(f"PK list deletion failed: {e}")
    
    # Step 8: Final Statistics
    print_step(8, "Final Statistics")
    try:
        # Use ConnectionManager to execute flush operation
        await conn_manager.execute_operation_async(
            lambda alias: _flush_collection(alias, COLLECTION_NAME)
        )
        
        # Get actual count by querying
        results = await conn_manager.execute_operation_async(
            lambda alias: _query_collection(alias, COLLECTION_NAME, "pk >= 0", ["pk"])
        )
        final_count = len(results) if results else 0
        
        print("\n  Deletion Summary:")
        print(f"    Starting entities: {initial_count}")
        print(f"    Ending entities: {final_count}")
        print(f"    Total deleted: {initial_count - final_count}")
        print(f"    Retention rate: {(final_count / initial_count) * 100:.1f}%")
        
        print_success("Deletion operations completed")
    except Exception as e:
        print_error(f"Statistics retrieval failed: {e}")
    
    # Step 9: Cleanup
    print_step(9, "Close Connection")
    try:
        conn_manager.close()
        print_success("Connection closed")
    except Exception as e:
        print_error(f"Cleanup failed: {e}")
    
    print_section("Example Completed")
    print("\nKey Takeaways:")
    print("  • Delete by IDs for specific entities")
    print("  • Delete by expression for conditional deletion")
    print("  • Always flush() to persist deletions")
    print("  • Verify deletions with queries")
    print("\nExpression Examples (Primary Key Only):")
    print("  • 'pk in [1, 2, 3]' - Delete specific PKs")
    print("  • 'pk in [10, 11, 12, 13, 14, 15]' - Delete multiple PKs")
    print("\nNote: Milvus only supports 'pk in [...]' format for deletion.")
    print("Range expressions (pk >= X) are NOT supported for deletion.")
    print("To delete by other fields, query for PKs first, then delete by PK.")


if __name__ == "__main__":
    asyncio.run(main())

