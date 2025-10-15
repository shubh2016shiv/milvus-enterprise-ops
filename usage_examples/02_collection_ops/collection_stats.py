"""
Collection Statistics Example

Demonstrates how to retrieve and display comprehensive statistics
about a Milvus collection including entity counts, memory usage, and schema.
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
format_bytes = example_utils.format_bytes


COLLECTION_NAME = "test_example_collection"


async def main():
    """Main function to demonstrate collection statistics retrieval."""
    print_section("Collection Statistics Example")
    
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
            print_info("Hint", "Run create_collection.py first")
            conn_manager.close()
            return
        print_success(f"Collection '{COLLECTION_NAME}' found")
    except Exception as e:
        print_error(f"Collection check failed: {e}")
        conn_manager.close()
        return
    
    # Step 3: Get Collection Description
    print_step(3, "Get Collection Description")
    try:
        desc = await coll_manager.describe_collection(COLLECTION_NAME)
        
        print("\n  Collection Information:")
        print(f"    Name: {desc.name}")
        print(f"    Description: {desc.collection_schema.description or 'N/A'}")
        print(f"    Created: {desc.created_at}")
        print(f"    State: {desc.state.value}")
        print(f"    Load State: {desc.load_state.value}")
        
        print_success("Retrieved collection description")
    except Exception as e:
        print_error(f"Description retrieval failed: {e}")
    
    # Step 4: Get Collection Statistics
    print_step(4, "Get Collection Statistics")
    try:
        stats = await coll_manager.get_collection_stats(COLLECTION_NAME)
        
        print("\n  Statistics:")
        print(f"    Entity Count: {stats.row_count}")
        print(f"    Memory Size: {format_bytes(stats.memory_size)}")
        print(f"    Disk Size: {format_bytes(stats.disk_size)}")
        print(f"    Index Size: {format_bytes(stats.index_size)}")
        print(f"    Partitions: {stats.num_partitions}")
        print(f"    Segments: {stats.num_segments}")
        
        print_success("Retrieved collection statistics")
    except Exception as e:
        print_error(f"Statistics retrieval failed: {e}")
    
    # Step 5: Display Schema Details
    print_step(5, "Display Schema Details")
    try:
        desc = await coll_manager.describe_collection(COLLECTION_NAME)
        schema = desc.collection_schema
        
        print(f"\n  Schema ({len(schema.fields)} fields):")
        for i, field in enumerate(schema.fields, 1):
            field_type = field.dtype
            
            print(f"\n  Field {i}: {field.name}")
            print(f"    Type: {field_type}")
            print(f"    Primary: {field.is_primary}")
            
            if field.dim:
                print(f"    Dimension: {field.dim}")
            
            if field.max_length:
                print(f"    Max Length: {field.max_length}")
            
            if hasattr(field, 'auto_id'):
                print(f"    Auto ID: {field.auto_id}")
            
            if field.description:
                print(f"    Description: {field.description}")
        
        print_success("Schema details displayed")
    except Exception as e:
        print_error(f"Schema display failed: {e}")
    
    # Step 6: List All Collections
    print_step(6, "List All Collections in Database")
    try:
        all_collections = await coll_manager.list_collections()
        
        print(f"\n  Total Collections: {len(all_collections)}")
        for coll_name in all_collections:
            marker = " ← Current" if coll_name == COLLECTION_NAME else ""
            print(f"    • {coll_name}{marker}")
        
        print_success("Listed all collections")
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
    print("  • Collection statistics provide entity counts and load state")
    print("  • Schema describes field types and configurations")
    print("  • Monitor loaded state before search operations")
    print("  • Use stats for capacity planning and monitoring")


if __name__ == "__main__":
    asyncio.run(main())


