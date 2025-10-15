"""
Reranking Search Example

Demonstrates reranking search results to improve relevance.
Shows how to retrieve more results then rerank for better accuracy.
"""

import sys
import os
import asyncio
import numpy as np
import random

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


def simple_reranker(query_vector, results, output_k=10):
    """
    Simple reranking function using cosine similarity.
    
    In production, you would use:
    - Cross-encoder models
    - LLM-based reranking
    - Custom business logic
    """
    reranked = []
    
    for hit in results:
        # Simulate getting the vector (in practice, fetch from collection)
        # For this example, we'll use a random vector
        doc_vector = np.random.rand(VECTOR_DIM)
        
        # Calculate cosine similarity
        query_np = np.array(query_vector)
        doc_np = np.array(doc_vector)
        
        cosine_sim = np.dot(query_np, doc_np) / (np.linalg.norm(query_np) * np.linalg.norm(doc_np))
        
        reranked.append({
            'id': hit.entity.get('pk'),
            'original_distance': hit.distance,
            'rerank_score': cosine_sim,
            'entity': hit.entity
        })
    
    # Sort by rerank score (descending)
    reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
    
    return reranked[:output_k]


async def main():
    """Main function to demonstrate reranking search."""
    print_section("Reranking Search Example")
    
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
    
    # Step 3: Generate Query Vector
    print_step(3, "Generate Query Vector")
    try:
        query_vector = np.random.rand(VECTOR_DIM).tolist()
        print_success("Query vector generated")
    except Exception as e:
        print_error(f"Query generation failed: {e}")
        conn_manager.close()
        return
    
    # Step 4: Initial Search (Retrieve More)
    print_step(4, "Initial Search with Larger K")
    try:
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        # Retrieve more results (e.g., 50) for reranking
        initial_k = 50
        final_k = 10
        
        print_info("Initial K", initial_k)
        print_info("Final K (after reranking)", final_k)
        
        # Define search function to be executed with connection
        def _initial_search(alias):
            from pymilvus import Collection
            collection = Collection(COLLECTION_NAME, using=alias)
            return collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=initial_k,
                output_fields=["pk", "text", "value", "category"]
            )
        
        # Execute search using connection manager
        results = await conn_manager.execute_operation_async(_initial_search)
        
        print_success(f"Retrieved {len(results[0])} candidates for reranking")
    except Exception as e:
        print_error(f"Initial search failed: {e}")
        conn_manager.close()
        return
    
    # Step 5: Display Original Top 10
    print_step(5, "Original Top 10 Results (Before Reranking)")
    try:
        print("\n  Original Ranking:\n")
        for i, hit in enumerate(results[0][:10], 1):
            print(f"  {i}. ID: {hit.entity.get('pk'):5d} | Distance: {hit.distance:.6f} | Text: {hit.entity.get('text', 'N/A')[:30]}")
        print()
        print_success("Original results displayed")
    except Exception as e:
        print_error(f"Display failed: {e}")
    
    # Step 6: Rerank Results
    print_step(6, "Rerank Using Advanced Similarity")
    try:
        print_info("Reranking", f"Processing {len(results[0])} results...")
        print_info("Method", "Cosine similarity (simulated)")
        
        reranked_results = simple_reranker(query_vector, results[0], output_k=final_k)
        
        print_success(f"Reranked to top {len(reranked_results)} results")
    except Exception as e:
        print_error(f"Reranking failed: {e}")
        conn_manager.close()
        return
    
    # Step 7: Display Reranked Top 10
    print_step(7, "Reranked Top 10 Results")
    try:
        print("\n  Reranked Results:\n")
        for i, result in enumerate(reranked_results, 1):
            print(f"  {i}. ID: {result['id']:5d} | Rerank Score: {result['rerank_score']:.6f} | Text: {result['entity'].get('text', 'N/A')[:30]}")
        print()
        print_success("Reranked results displayed")
    except Exception as e:
        print_error(f"Display failed: {e}")
    
    # Step 8: Compare Rankings
    print_step(8, "Compare Original vs Reranked")
    try:
        original_top10_ids = [hit.entity.get('pk') for hit in results[0][:10]]
        reranked_top10_ids = [r['id'] for r in reranked_results]
        
        print("\n  Ranking Changes:\n")
        print(f"  {'Rank':<6} {'Original':<12} {'Reranked':<12} {'Change':<10}")
        print("  " + "-" * 50)
        
        for rank in range(1, 11):
            orig_id = original_top10_ids[rank - 1] if rank <= len(original_top10_ids) else "-"
            rerank_id = reranked_top10_ids[rank - 1] if rank <= len(reranked_top10_ids) else "-"
            
            # Check if IDs are different
            change = ""
            if orig_id != rerank_id and orig_id != "-" and rerank_id != "-":
                # Find where the reranked ID was in original
                if rerank_id in original_top10_ids:
                    orig_pos = original_top10_ids.index(rerank_id) + 1
                    change = f"^ from #{orig_pos}"
                else:
                    change = "^ new"
            elif orig_id == rerank_id:
                change = "="
            
            print(f"  {rank:<6} {str(orig_id):<12} {str(rerank_id):<12} {change:<10}")
        
        print()
        
        # Calculate overlap
        overlap = len(set(original_top10_ids) & set(reranked_top10_ids))
        print_info("Overlap", f"{overlap}/10 documents remain in top 10")
        print_success("Comparison completed")
    except Exception as e:
        print_error(f"Comparison failed: {e}")
    
    # Step 9: Metrics
    print_step(9, "Reranking Metrics")
    try:
        original_top10_ids = [hit.entity.get('pk') for hit in results[0][:10]]
        reranked_top10_ids = [r['id'] for r in reranked_results]
        
        # Calculate metrics
        overlap = len(set(original_top10_ids) & set(reranked_top10_ids))
        new_entries = len([id for id in reranked_top10_ids if id not in original_top10_ids])
        
        print("\n  Reranking Impact:")
        print(f"    Documents preserved: {overlap}/10")
        print(f"    New documents in top-10: {new_entries}")
        print(f"    Reranking effectiveness: {(10 - overlap) / 10 * 100:.1f}% change")
        
        print_success("Metrics calculated")
    except Exception as e:
        print_error(f"Metrics calculation failed: {e}")
    
    # Step 10: Cleanup
    print_step(10, "Close Connection")
    try:
        conn_manager.close()
        print_success("Connection closed")
    except Exception as e:
        print_error(f"Cleanup failed: {e}")
    
    print_section("Example Completed")
    print("\nKey Takeaways:")
    print("  • Retrieve more candidates (e.g., 50-100)")
    print("  • Apply advanced reranking model")
    print("  • Return top-k from reranked results")
    print("  • Improves relevance at slight latency cost")
    print("\nReranking Methods:")
    print("  • Cross-encoders (BERT, T5)")
    print("  • LLM-based reranking (GPT, Claude)")
    print("  • Custom business logic")
    print("  • Learned-to-rank models")
    print("\nWhen to Use Reranking:")
    print("  • When precision is critical")
    print("  • Multi-stage retrieval pipelines")
    print("  • Combining multiple signals")
    print("  • Fine-tuning search relevance")
    print("\nProduction Considerations:")
    print("  • Balance initial_k vs rerank_cost")
    print("  • Cache reranking results")
    print("  • Use async reranking for speed")
    print("  • Monitor latency impact")


if __name__ == "__main__":
    asyncio.run(main())

