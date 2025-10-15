"""
Hybrid Search Example

Demonstrates hybrid search combining dense (vector) and sparse (BM25) retrieval.
Shows how to use multiple search strategies for better results.
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
print_warning = example_utils.print_warning


COLLECTION_NAME = "test_example_collection"
VECTOR_DIM = 128


async def main():
    """Main function to demonstrate hybrid search."""
    print_section("Hybrid Search Example")
    
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
    
    # Step 2: Verify Collection
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
    
    # Step 3: Prepare Query
    print_step(3, "Prepare Hybrid Query")
    try:
        # Dense vector for semantic search
        query_vector = np.random.rand(VECTOR_DIM).tolist()
        
        # Text query for keyword-based search
        query_text = "test data sample"
        
        print_info("Dense query", f"Vector of dim {VECTOR_DIM}")
        print_info("Sparse query", f"Text: '{query_text}'")
        print_success("Hybrid query prepared")
    except Exception as e:
        print_error(f"Query preparation failed: {e}")
        conn_manager.close()
        return
    
    # Step 4: Perform Dense Vector Search
    print_step(4, "Perform Dense Vector Search")
    try:
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        # Define search function to be executed with connection
        def _dense_search(alias):
            from pymilvus import Collection
            collection = Collection(COLLECTION_NAME, using=alias)
            return collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=20,  # Retrieve more for fusion
                output_fields=["pk", "text", "value"]
            )
        
        # Execute search using connection manager
        dense_results = await conn_manager.execute_operation_async(_dense_search)
        
        dense_ids = [hit.entity.get('pk') for hit in dense_results[0]]
        dense_scores = {hit.entity.get('pk'): 1.0 / (1.0 + hit.distance) for hit in dense_results[0]}
        
        print_success(f"Dense search: Found {len(dense_results[0])} results")
        print_info("Top 3 IDs", dense_ids[:3])
    except Exception as e:
        print_error(f"Dense search failed: {e}")
        conn_manager.close()
        return
    
    # Step 5: Simulate Sparse (Keyword) Search
    print_step(5, "Simulate Sparse (Keyword) Search")
    try:
        # Define query function to be executed with connection
        def _sparse_search(alias):
            from pymilvus import Collection
            collection = Collection(COLLECTION_NAME, using=alias)
            # Simulate keyword matching with filter
            # In production, you would use BM25 or full-text search
            return collection.query(
                expr=f'text like "{query_text.split()[0]}%"',
                output_fields=["pk", "text", "value"],
                limit=20
            )
        
        # Execute query using connection manager
        sparse_results = await conn_manager.execute_operation_async(_sparse_search)
        
        if not sparse_results:
            # Fallback: get random subset for demonstration
            print_warning("No keyword matches, using subset for demonstration")
            
            def _fallback_search(alias):
                from pymilvus import Collection
                collection = Collection(COLLECTION_NAME, using=alias)
                return collection.query(
                    expr="pk >= 0",
                    output_fields=["pk", "text", "value"],
                    limit=20
                )
            
            all_results = await conn_manager.execute_operation_async(_fallback_search)
            sparse_results = all_results[:10] if all_results else []
        
        sparse_ids = [doc['pk'] for doc in sparse_results]
        # Assign simple relevance scores
        sparse_scores = {doc['pk']: 1.0 - (i * 0.05) for i, doc in enumerate(sparse_results)}
        
        print_success(f"Sparse search: Found {len(sparse_results)} results")
        print_info("Top 3 IDs", sparse_ids[:3])
    except Exception as e:
        print_error(f"Sparse search failed: {e}")
        # Continue with just dense results
        sparse_ids = []
        sparse_scores = {}
    
    # Step 6: Fusion - Combine Results
    print_step(6, "Fusion: Combine Dense and Sparse Results")
    try:
        # Reciprocal Rank Fusion (RRF)
        k = 60  # RRF parameter
        fused_scores = {}
        
        # Add scores from dense search
        for rank, doc_id in enumerate(dense_ids, 1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.0 / (k + rank)
        
        # Add scores from sparse search
        for rank, doc_id in enumerate(sparse_ids, 1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.0 / (k + rank)
        
        # Sort by combined score
        fused_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        print_success(f"Fused {len(fused_results)} unique results")
        print_info("Fusion method", "Reciprocal Rank Fusion (RRF)")
        print_info("k parameter", k)
    except Exception as e:
        print_error(f"Fusion failed: {e}")
        fused_results = []
    
    # Step 7: Display Fused Results
    print_step(7, "Display Top Fused Results")
    try:
        if fused_results:
            top_10_ids = [doc_id for doc_id, score in fused_results[:10]]
            
            # Define query function to be executed with connection
            def _get_details(alias):
                from pymilvus import Collection
                collection = Collection(COLLECTION_NAME, using=alias)
                return collection.query(
                    expr=f"pk in {top_10_ids}",
                    output_fields=["pk", "text", "value", "category"]
                )
            
            # Execute query using connection manager
            details = await conn_manager.execute_operation_async(_get_details)
            
            # Create lookup dict
            details_dict = {doc['pk']: doc for doc in details}
            
            print("\n  Top 10 Hybrid Search Results:\n")
            for i, (doc_id, score) in enumerate(fused_results[:10], 1):
                doc = details_dict.get(doc_id, {})
                in_dense = "Yes" if doc_id in dense_ids else "No"
                in_sparse = "Yes" if doc_id in sparse_ids else "No"
                
                print(f"  {i}. ID: {doc_id} (Score: {score:.4f})")
                print(f"     Dense: {in_dense}  Sparse: {in_sparse}")
                print(f"     Text: {doc.get('text', 'N/A')[:50]}")
                print(f"     Value: {doc.get('value', 'N/A')}")
                print()
            
            print_success("Hybrid results displayed")
        else:
            print_error("No fused results to display")
    except Exception as e:
        print_error(f"Result display failed: {e}")
    
    # Step 8: Compare Rankings
    print_step(8, "Compare Rankings: Dense vs Sparse vs Hybrid")
    try:
        print("\n  Ranking Comparison:\n")
        print(f"  {'Rank':<6} {'Dense':<10} {'Sparse':<10} {'Hybrid':<10}")
        print("  " + "-" * 40)
        
        for rank in range(1, min(11, len(fused_results) + 1)):
            dense_id = dense_ids[rank - 1] if rank <= len(dense_ids) else "-"
            sparse_id = sparse_ids[rank - 1] if rank <= len(sparse_ids) else "-"
            hybrid_id = fused_results[rank - 1][0] if rank <= len(fused_results) else "-"
            
            print(f"  {rank:<6} {str(dense_id):<10} {str(sparse_id):<10} {str(hybrid_id):<10}")
        
        print()
        print_success("Ranking comparison displayed")
    except Exception as e:
        print_error(f"Comparison failed: {e}")
    
    # Step 9: Cleanup
    print_step(9, "Close Connection")
    try:
        conn_manager.close()
        print_success("Connection closed")
    except Exception as e:
        print_error(f"Cleanup failed: {e}")
    
    print_section("Example Completed")
    print("\nKey Takeaways:")
    print("  • Hybrid search combines semantic and keyword matching")
    print("  • Reciprocal Rank Fusion (RRF) is a simple, effective fusion method")
    print("  • Hybrid often outperforms single-method search")
    print("  • Results appear in both dense and sparse are boosted")
    print("\nFusion Methods:")
    print("  • RRF: Simple, parameter-free (just k)")
    print("  • Weighted Sum: Control importance of each method")
    print("  • Borda Count: Rank-based voting")
    print("\nUse Cases:")
    print("  • E-commerce search (semantic + keyword)")
    print("  • Document retrieval (meaning + exact terms)")
    print("  • Question answering (context + keywords)")
    print("\nNote:")
    print("  • This example simulates sparse search")
    print("  • For production, integrate true BM25 or full-text search")
    print("  • Consider using Milvus sparse vector support")


if __name__ == "__main__":
    asyncio.run(main())

