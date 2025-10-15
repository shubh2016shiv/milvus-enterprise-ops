"""
Semantic Search Example

Demonstrates basic vector similarity search using a sample query vector.
Shows how to find nearest neighbors in the vector space.
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
    """Main function to demonstrate semantic search."""
    print_section("Semantic Search Example")
    
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
    print_step(2, "Verify Collection is Ready for Search")
    try:
        if not await coll_manager.has_collection(COLLECTION_NAME):
            print_error(f"Collection '{COLLECTION_NAME}' does not exist")
            conn_manager.close()
            return
        
        # Check if collection is loaded
        description = await coll_manager.describe_collection(COLLECTION_NAME)
        print_info("Loaded", "Yes" if description.load_state.value == "Loaded" else "No")
        
        # Load if not loaded
        if description.load_state.value != "Loaded":
            print_info("Action", "Loading collection into memory...")
            await coll_manager.load_collection(COLLECTION_NAME, wait=True)
            print_success("Collection loaded")
        
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
        # Generate a random query vector
        query_vector = np.random.rand(VECTOR_DIM).tolist()
        
        print_info("Query vector dimension", VECTOR_DIM)
        print_info("Sample values", f"[{query_vector[0]:.4f}, {query_vector[1]:.4f}, ...]")
        print_success("Query vector generated")
    except Exception as e:
        print_error(f"Query vector generation failed: {e}")
        conn_manager.close()
        return
    
    # Step 4: Perform Basic Search
    print_step(4, "Perform Basic Vector Search")
    try:
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}  # For IVF indexes
        }
        
        top_k = 10  # Return top 10 results
        
        print_info("Search params", search_params)
        print_info("Top K", top_k)
        
        # Define search function to be executed with connection
        def _basic_search(alias):
            from pymilvus import Collection
            collection = Collection(COLLECTION_NAME, using=alias)
            return collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["pk", "text", "value", "category"]
            )
        
        # Execute search using connection manager
        results = await conn_manager.execute_operation_async(_basic_search)
        
        print_success(f"Search completed, found {len(results[0])} results")
    except Exception as e:
        print_error(f"Search failed: {e}")
        conn_manager.close()
        return
    
    # Step 5: Display Results
    print_step(5, "Display Search Results")
    try:
        if results and len(results[0]) > 0:
            print(f"\n  Top {len(results[0])} Results (Closest Vectors):\n")
            
            for i, hit in enumerate(results[0], 1):
                print(f"  {i}. ID: {hit.entity.get('pk')}")
                print(f"     Distance: {hit.distance:.6f}")
                print(f"     Text: {hit.entity.get('text', 'N/A')}")
                print(f"     Value: {hit.entity.get('value', 'N/A')}")
                print(f"     Category: {hit.entity.get('category', 'N/A')}")
                print()
            
            print_success("Results displayed")
        else:
            print_error("No results found")
    except Exception as e:
        print_error(f"Result display failed: {e}")
    
    # Step 6: Search with Different Top K
    print_step(6, "Search with Different Top K Values")
    try:
        for k in [3, 5, 20]:
            # Define search function to be executed with connection
            def _search_with_k(alias, k_value):
                from pymilvus import Collection
                collection = Collection(COLLECTION_NAME, using=alias)
                return collection.search(
                    data=[query_vector],
                    anns_field="vector",
                    param=search_params,
                    limit=k_value,
                    output_fields=["pk"]
                )
            
            # Execute search using connection manager
            k_results = await conn_manager.execute_operation_async(
                lambda alias: _search_with_k(alias, k)
            )
            
            print_info(f"Top-{k} search", f"Returned {len(k_results[0])} results")
        
        print_success("Multiple searches completed")
    except Exception as e:
        print_error(f"Multi-search failed: {e}")
    
    # Step 7: Batch Search (Multiple Queries)
    print_step(7, "Batch Search with Multiple Query Vectors")
    try:
        # Generate 3 query vectors
        num_queries = 3
        query_vectors = [np.random.rand(VECTOR_DIM).tolist() for _ in range(num_queries)]
        
        print_info("Number of queries", num_queries)
        
        # Define batch search function to be executed with connection
        def _batch_search(alias):
            from pymilvus import Collection
            collection = Collection(COLLECTION_NAME, using=alias)
            return collection.search(
                data=query_vectors,
                anns_field="vector",
                param=search_params,
                limit=5,
                output_fields=["pk", "text"]
            )
        
        # Execute batch search using connection manager
        results = await conn_manager.execute_operation_async(_batch_search)
        
        print(f"\n  Batch Search Results:\n")
        for i, query_result in enumerate(results, 1):
            print(f"  Query {i}: Found {len(query_result)} results")
            top_ids = [hit.entity.get('pk') for hit in query_result[:3]]
            print(f"    Top 3 IDs: {top_ids}\n")
        
        print_success("Batch search completed")
    except Exception as e:
        print_error(f"Batch search failed: {e}")
    
    # Step 8: Cleanup
    print_step(8, "Close Connection")
    try:
        conn_manager.close()
        print_success("Connection closed")
    except Exception as e:
        print_error(f"Cleanup failed: {e}")
    
    print_section("Example Completed")
    print("\nKey Takeaways:")
    print("  • Vector search finds semantically similar vectors")
    print("  • Lower distance = more similar")
    print("  • Top-K controls number of results")
    print("  • Batch search processes multiple queries efficiently")
    print("\nSearch Parameters:")
    print("  • metric_type: L2 (Euclidean), IP (Inner Product), COSINE")
    print("  • nprobe: Number of clusters to search (IVF indexes)")
    print("  • ef: Search depth (HNSW indexes)")
    print("\nPerformance Tips:")
    print("  • Index the collection for faster searches")
    print("  • Use appropriate metric type for your data")
    print("  • Batch queries for better throughput")
    print("  • Load collection before searching")


if __name__ == "__main__":
    asyncio.run(main())

