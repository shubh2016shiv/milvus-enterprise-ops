"""
Load Collection Example

Demonstrates how to load a collection into memory and monitor the
loading progress. Collections must be loaded before search operations.
"""

import sys
import os
import asyncio
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from connection_management import ConnectionManager
from collection_operations import CollectionManager, LoadState
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
print_warning = example_utils.print_warning
print_note = example_utils.print_note


COLLECTION_NAME = "test_example_collection"


async def main():
    """Main function to demonstrate collection loading."""
    print_section("Load Milvus Collection Example")
    
    # Step 1: Initialize Managers
    print_step(1, "Initialize Connection and Collection Managers")
    try:
        config = load_settings()
        conn_manager = ConnectionManager(config=config)
        coll_manager = CollectionManager(conn_manager)
        print_success("Managers initialized")
    except Exception as e:
        print_error(f"Initialization failed: {e}")
        return
    
    # Step 2: Check Collection Exists
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
    
    # Step 3: Load Collection into Memory
    print_step(3, "Load Collection into Memory")
    try:
        start_time = time.time()
        
        result = await coll_manager.load_collection(
            collection_name=COLLECTION_NAME,
            wait=True
        )
        
        load_time = time.time() - start_time
        
        # We expect a LoadProgress object, not a boolean
        if hasattr(result, 'state') and result.state == LoadState.LOADED:
            print_success(f"Collection loaded successfully in {load_time:.2f} seconds")
            print_info("Load state", result.state.value)
        else:
            # This path should ideally not be taken now
            print_error("Collection failed to load even with an index.")
            print_info("Result", str(result))

    except Exception as e:
        print_error(f"Loading failed: {e}")
        conn_manager.close()
        return
    
    # Step 4: Verify Loading Progress
    print_step(4, "Verify Loading Progress")
    try:
        progress = await coll_manager.get_load_progress(COLLECTION_NAME)
        
        print_info("Progress", f"{progress.progress:.1f}%")
        
        if progress.state == LoadState.LOADING:
            print_info("Status", "Still loading...")
        else:
            print_success("Collection fully loaded")
        
        if progress.total_segments > 0:
            print_info("Loaded segments", 
                       f"{progress.loaded_segments}/{progress.total_segments}")
        
    except Exception as e:
        print_error(f"Progress check failed: {e}")
    
    # Step 5: Confirm Collection is Ready for Search
    print_step(5, "Confirm Collection is Ready for Search")
    try:
        # Check if collection is loaded BEFORE releasing it
        description = await coll_manager.describe_collection(COLLECTION_NAME)
        stats = await coll_manager.get_collection_stats(COLLECTION_NAME)
        
        if description.load_state == LoadState.LOADED:
            print_success("Collection is loaded and ready for search operations")
            print_info("Entities available", stats.row_count)
            print_info("Memory status", "In-memory")
        else:
            print_error(f"Collection is not fully loaded (state: {description.load_state.value})")
    except Exception as e:
        print_error(f"Confirmation failed: {e}")
    
    # Step 6: Close Connection (Keep Collection Loaded)
    print_step(6, "Close Connection")
    try:
        conn_manager.close()
        print_success("Connection closed")
        print_note("Collection remains loaded in Milvus for search operations")
    except Exception as e:
        print_error(f"Cleanup failed: {e}")
    
    print_section("Example Completed")
    print("\nKey Takeaways:")
    print("  • Collections must be loaded before search operations")
    print("  • Loading brings data into memory for fast access")
    print("  • An index must exist before loading (created during collection setup)")
    print("  • Monitor load progress for large collections")
    print("  • Collection remains loaded after script ends")
    print("\nNext Steps:")
    print("  • Collection is LOADED and ready for search operations")
    print("  • Run search examples to query the loaded data")
    print("  • To unload: run release_collection.py or manually release")
    print("\nNote:")
    print("  • If you get an index error, run create_collection.py first")
    print("  • The create_collection.py script creates both collection and index")


if __name__ == "__main__":
    asyncio.run(main())
