"""
Drop Collection Example

Demonstrates how to safely drop (delete) a Milvus collection.
This operation is irreversible and removes all data.
"""

import sys
import os
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from connection_management import ConnectionManager
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


COLLECTION_NAME = "test_example_collection"


async def main():
    """Main function to demonstrate collection dropping."""
    print_section("Drop Milvus Collection Example")
    
    print("\n⚠️  WARNING: This operation will permanently delete the collection!")
    print(f"    Collection to delete: {COLLECTION_NAME}")
    print("    This action cannot be undone.\n")
    
    # Step 1: Initialize Managers
    print_step(1, "Initialize Managers")
    try:
        config = load_settings()
        conn_manager = ConnectionManager(config=config)
        coll_manager = CollectionManager(conn_manager)
        print_success("Managers initialized")
    except Exception as e:
        print_error(f"Initialization failed: {e}")
        return
    
    # Step 2: Verify Collection Exists
    print_step(2, "Verify Collection Exists")
    try:
        if not await coll_manager.has_collection(COLLECTION_NAME):
            print_error(f"Collection '{COLLECTION_NAME}' does not exist")
            print_info("Status", "Nothing to drop")
            conn_manager.close()
            return
        
        # Get stats before dropping
        stats = await coll_manager.get_collection_stats(COLLECTION_NAME)
        print_success(f"Collection '{COLLECTION_NAME}' found")
        print_info("Entity count", stats.row_count)
        
        # Check if collection is loaded
        try:
            from pymilvus import Collection
            collection = Collection(name=COLLECTION_NAME)
            is_loaded = getattr(collection, "is_loaded", False)
            print_info("Loaded", is_loaded)
        except Exception:
            print_info("Loaded", "Unknown")
    except Exception as e:
        print_error(f"Collection check failed: {e}")
        conn_manager.close()
        return
    
    # Step 3: Release Collection (if loaded)
    print_step(3, "Release Collection from Memory (if loaded)")
    try:
        # Check if collection is loaded
        from pymilvus import Collection
        collection = Collection(name=COLLECTION_NAME)
        is_loaded = getattr(collection, "is_loaded", False)
        
        if is_loaded:
            print_info("Status", "Collection is loaded, releasing first...")
            await coll_manager.release_collection(COLLECTION_NAME)
            print_success("Collection released from memory")
        else:
            print_info("Status", "Collection not loaded, skipping release")
    except Exception as e:
        # Continue even if release fails
        print_error(f"Release failed: {e} (continuing...)")
    
    # Step 4: Drop Collection
    print_step(4, "Drop Collection")
    try:
        print_info("Action", f"Dropping collection '{COLLECTION_NAME}'...")
        
        await coll_manager.drop_collection(COLLECTION_NAME)
        
        print_success(f"Collection '{COLLECTION_NAME}' dropped successfully")
    except Exception as e:
        print_error(f"Drop operation failed: {e}")
        conn_manager.close()
        return
    
    # Step 5: Verify Deletion
    print_step(5, "Verify Collection is Deleted")
    try:
        exists = await coll_manager.has_collection(COLLECTION_NAME)
        
        if not exists:
            print_success(f"Confirmed: Collection '{COLLECTION_NAME}' no longer exists")
        else:
            print_error("Collection still exists after drop operation")
    except Exception as e:
        print_error(f"Verification failed: {e}")
    
    # Step 6: List Remaining Collections
    print_step(6, "List Remaining Collections")
    try:
        remaining = await coll_manager.list_collections()
        
        print_info("Remaining collections", len(remaining))
        if remaining:
            for coll in remaining:
                print(f"    • {coll}")
        else:
            print("    (No collections)")
        
        print_success("Listed remaining collections")
    except Exception as e:
        print_error(f"Listing failed: {e}")
    
    # Step 7: Cleanup
    print_step(7, "Close Connection")
    try:
        conn_manager.close()
        print_success("Connection closed")
    except Exception as e:
        print_error(f"Cleanup failed: {e}")
    
    print_section("Example Completed")
    print("\nKey Takeaways:")
    print("  • Dropping a collection is permanent and irreversible")
    print("  • Release loaded collections before dropping")
    print("  • Always verify collection existence before dropping")
    print("  • Use with caution in production environments")
    print("\n⚠️  Best Practices:")
    print("  • Always backup important data before dropping")
    print("  • Use meaningful collection names to avoid mistakes")
    print("  • Consider using a 'deleted' flag instead of dropping")


if __name__ == "__main__":
    asyncio.run(main())


