"""
Filtered Search Example

Demonstrates how to perform vector searches with metadata filters.
Shows how to combine semantic similarity with structured queries.
"""

import sys
import os
import asyncio
import numpy as np

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
VECTOR_DIM = 128


async def main():
    """Main function to demonstrate filtered search."""
    print_section("Filtered Search Example")
    
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
    
    # Step 2: Verify Collection is Ready
    print_step(2, "Verify Collection is Ready")
    try:
        if not await coll_manager.has_collection(COLLECTION_NAME):
            print_error(f"Collection '{COLLECTION_NAME}' does not exist")
            conn_manager.close()
            return
        
        # Check if collection is loaded
        description = await coll_manager.describe_collection(COLLECTION_NAME)
        if description.load_state.value != "Loaded":
            print_info("Loading collection", "Collection not loaded, loading now...")
            await coll_manager.load_collection(COLLECTION_NAME, wait=True)
        
        # Check if collection has data using direct query
        def _query_collection(alias, collection_name):
            from pymilvus import Collection
            collection = Collection(collection_name, using=alias)
            return collection.query(expr="pk >= 0", output_fields=["pk"], limit=1)
        
        results = await conn_manager.execute_operation_async(
            lambda alias: _query_collection(alias, COLLECTION_NAME)
        )
        
        if not results:
            print_error("Collection is empty")
            conn_manager.close()
            return
        
        print_success(f"Collection ready for search")
    except Exception as e:
        print_error(f"Collection check failed: {e}")
        conn_manager.close()
        return
    
    # Step 3: Generate Query Vector
    print_step(3, "Generate Query Vector")
    try:
        query_vector = np.random.rand(VECTOR_DIM).tolist()
        print_success("Query vector generated")
    except Exception as e:
        print_error(f"Query generation failed: {e}")
        conn_manager.close()
        return
    
    # Step 4: Search with Numeric Filter
    print_step(4, "Search with Numeric Filter (value > 50)")
    try:
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        # Filter by value field
        filter_expr = "value > 50"
        
        print_info("Filter expression", filter_expr)
        
        # Define search function to be executed with connection
        def _search_with_filter(alias):
            from pymilvus import Collection
            collection = Collection(COLLECTION_NAME, using=alias)
            return collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=10,
                expr=filter_expr,
                output_fields=["pk", "text", "value", "category"]
            )
        
        # Execute search using connection manager
        results = await conn_manager.execute_operation_async(_search_with_filter)
        
        print_success(f"Found {len(results[0])} results matching filter")
        
        if results[0]:
            print("\n  Top 5 Results:\n")
            for i, hit in enumerate(results[0][:5], 1):
                print(f"  {i}. ID: {hit.entity.get('pk')}, Value: {hit.entity.get('value'):.2f}, Distance: {hit.distance:.6f}")
            print()
    except Exception as e:
        print_error(f"Numeric filter search failed: {e}")
    
    # Step 5: Search with Category Filter
    print_step(5, "Search with Category Filter (category == 'A')")
    try:
        filter_expr = 'category == "A"'
        
        print_info("Filter expression", filter_expr)
        
        # Define search function to be executed with connection
        def _search_with_category(alias):
            from pymilvus import Collection
            collection = Collection(COLLECTION_NAME, using=alias)
            return collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=10,
                expr=filter_expr,
                output_fields=["pk", "text", "value", "category"]
            )
        
        # Execute search using connection manager
        results = await conn_manager.execute_operation_async(_search_with_category)
        
        print_success(f"Found {len(results[0])} results in category A")
        
        if results[0]:
            print("\n  Category A Results:\n")
            for i, hit in enumerate(results[0][:5], 1):
                print(f"  {i}. ID: {hit.entity.get('pk')}, Category: {hit.entity.get('category')}, Distance: {hit.distance:.6f}")
            print()
    except Exception as e:
        print_error(f"Category filter search failed: {e}")
    
    # Step 6: Search with Combined Filters
    print_step(6, "Search with Combined Filters (AND condition)")
    try:
        filter_expr = 'value > 30 and category == "B"'
        
        print_info("Filter expression", filter_expr)
        
        # Define search function to be executed with connection
        def _search_with_combined(alias):
            from pymilvus import Collection
            collection = Collection(COLLECTION_NAME, using=alias)
            return collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=10,
                expr=filter_expr,
                output_fields=["pk", "text", "value", "category"]
            )
        
        # Execute search using connection manager
        results = await conn_manager.execute_operation_async(_search_with_combined)
        
        print_success(f"Found {len(results[0])} results matching combined filter")
        
        if results[0]:
            print("\n  Combined Filter Results:\n")
            for i, hit in enumerate(results[0][:5], 1):
                print(f"  {i}. ID: {hit.entity.get('pk')}, Value: {hit.entity.get('value'):.2f}, Category: {hit.entity.get('category')}, Distance: {hit.distance:.6f}")
            print()
    except Exception as e:
        print_error(f"Combined filter search failed: {e}")
    
    # Step 7: Search with OR Condition
    print_step(7, "Search with OR Condition")
    try:
        filter_expr = 'category == "A" or category == "C"'
        
        print_info("Filter expression", filter_expr)
        
        # Define search function to be executed with connection
        def _search_with_or(alias):
            from pymilvus import Collection
            collection = Collection(COLLECTION_NAME, using=alias)
            return collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=10,
                expr=filter_expr,
                output_fields=["pk", "category"]
            )
        
        # Execute search using connection manager
        results = await conn_manager.execute_operation_async(_search_with_or)
        
        print_success(f"Found {len(results[0])} results in categories A or C")
        
        if results[0]:
            categories = [hit.entity.get('category') for hit in results[0][:10]]
            print_info("Result categories", categories)
    except Exception as e:
        print_error(f"OR filter search failed: {e}")
    
    # Step 8: Search with Range Filter
    print_step(8, "Search with Range Filter (20 <= value <= 60)")
    try:
        filter_expr = "value >= 20 and value <= 60"
        
        print_info("Filter expression", filter_expr)
        
        # Define search function to be executed with connection
        def _search_with_range(alias):
            from pymilvus import Collection
            collection = Collection(COLLECTION_NAME, using=alias)
            return collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=10,
                expr=filter_expr,
                output_fields=["pk", "value"]
            )
        
        # Execute search using connection manager
        results = await conn_manager.execute_operation_async(_search_with_range)
        
        print_success(f"Found {len(results[0])} results in range")
        
        if results[0]:
            values = [hit.entity.get('value') for hit in results[0][:10]]
            print_info("Result values", [f"{v:.2f}" for v in values])
    except Exception as e:
        print_error(f"Range filter search failed: {e}")
    
    # Step 9: Search with IN Clause
    print_step(9, "Search with IN Clause")
    try:
        filter_expr = 'category in ["A", "B", "C"]'
        
        print_info("Filter expression", filter_expr)
        
        # Define search function to be executed with connection
        def _search_with_in(alias):
            from pymilvus import Collection
            collection = Collection(COLLECTION_NAME, using=alias)
            return collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=10,
                expr=filter_expr,
                output_fields=["pk", "category"]
            )
        
        # Execute search using connection manager
        results = await conn_manager.execute_operation_async(_search_with_in)
        
        print_success(f"Found {len(results[0])} results in specified categories")
    except Exception as e:
        print_error(f"IN clause search failed: {e}")
    
    # Step 10: Cleanup
    print_step(10, "Close Connection")
    try:
        conn_manager.close()
        print_success("Connection closed")
    except Exception as e:
        print_error(f"Cleanup failed: {e}")
    
    print_section("Example Completed")
    print("\nKey Takeaways:")
    print("  • Filtered search combines vector similarity with metadata")
    print("  • Use 'expr' parameter to specify filter conditions")
    print("  • Supports AND, OR, IN operators")
    print("  • Works with numeric and string fields")
    print("\nFilter Expression Syntax:")
    print("  • Comparison: ==, !=, >, <, >=, <=")
    print("  • Logical: and, or, not")
    print("  • Membership: in [list]")
    print("  • String: Use double quotes \"value\"")
    print("\nCommon Use Cases:")
    print("  • Search within specific categories")
    print("  • Filter by price range")
    print("  • Time-based filtering")
    print("  • Multi-criteria search")


if __name__ == "__main__":
    asyncio.run(main())

