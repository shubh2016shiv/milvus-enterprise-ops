"""
Upsert Data Example

Demonstrates upsert operations - updating existing entities and inserting
new ones in a single operation based on primary key.
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
generate_test_data = example_utils.generate_test_data


COLLECTION_NAME = "test_example_collection"
VECTOR_DIM = 128


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
    """Main function to demonstrate upsert operations."""
    print_section("Upsert Data Example")
    
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
    
    # Step 2: Verify Collection Exists and Has Data
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

        # ============================================================================
        # CRITICAL FIX: Query sample entities to verify collection has data
        # ============================================================================
        # PROBLEM WE'RE AVOIDING:
        # Stats API (get_collection_stats) returns STALE/CACHED data showing 0 entities
        # even right after successful insert+flush operations (e.g., shows 0 when 1020 exist)
        #
        # SOLUTION: Query first 10 entities to check if collection has data (real-time check)
        # ============================================================================
        sample_check = await conn_manager.execute_operation_async(
            lambda alias: _query_collection(
                alias,
                COLLECTION_NAME,
                "pk >= 0",
                ["pk"],
                limit=10  # Just need to know if ANY data exists
            )
        )

        if not sample_check or len(sample_check) == 0:
            print_error("Collection is empty")
            print_info("Hint", "Run insert_data.py first")
            conn_manager.close()
            return

        entity_count = len(sample_check)  # At least this many
        print_success(f"Collection has data (found at least {entity_count} entities)")
    except Exception as e:
        print_error(f"Collection check failed: {e}")
        conn_manager.close()
        return

    # Step 3: Prepare Upsert Data
    print_step(3, "Prepare Upsert Data (Mix of Updates and Inserts)")
    try:
        # Check which entities actually exist (IDs 1-10)
        potential_update_ids = list(range(1, 11))
        existing_results = await conn_manager.execute_operation_async(
            lambda alias: _query_collection(
                alias,
                COLLECTION_NAME,
                f"pk in [{', '.join(map(str, potential_update_ids))}]",
                ["pk"]
            )
        )
        existing_ids = set([r['pk'] for r in existing_results]) if existing_results else set()

        # Determine actual updates vs inserts
        update_ids = [id for id in potential_update_ids if id in existing_ids]
        insert_from_expected_updates = [id for id in potential_update_ids if id not in existing_ids]
        new_insert_ids = list(range(5001, 5011))
        all_ids = potential_update_ids + new_insert_ids

        data = generate_test_data(len(all_ids), VECTOR_DIM, include_metadata=True)
        data['id'] = all_ids

        print_info("Entities to update (existing)", len(update_ids))
        print_info("Entities to insert (new)", len(insert_from_expected_updates) + len(new_insert_ids))
        print_info("Total upsert", len(all_ids))
        
        # Convert to documents
        # Note: The collection uses 'pk' as the primary key field name
        documents = []
        for i in range(len(all_ids)):
            doc = {
                "pk": data["id"][i],  # Map 'id' from test data to 'pk' for the collection
                "vector": data["vector"][i],
                "text": f"UPDATED: {data['text'][i]}" if data["id"][i] in update_ids else data["text"][i],
                "value": data["value"][i] + 1000,  # Different values to show update
                "category": data["category"][i]
            }
            documents.append(doc)
        
        print_success("Upsert data prepared")
    except Exception as e:
        print_error(f"Data preparation failed: {e}")
        conn_manager.close()
        return
    
    # Step 4: Perform Upsert
    print_step(4, "Perform Upsert Operation")
    try:
        result = await data_manager.upsert(
            collection_name=COLLECTION_NAME,
            documents=documents
        )
        
        print_success(f"Upserted {result.successful_count} entities")
        print_info("Actual updates", len(update_ids))
        print_info("Actual inserts", len(insert_from_expected_updates) + len(new_insert_ids))
        print_info("Status", result.status)

        if result.failed_count > 0:
            print_error(f"{result.failed_count} entities failed")
    except Exception as e:
        error_msg = str(e)
        if "unknown method Upsert" in error_msg or "UNIMPLEMENTED" in error_msg:
            print_error("Upsert is not supported by this Milvus version")
            print_info("Note", "Upsert requires Milvus 2.3.0 or higher")
            print_info("Workaround", "Using delete + insert for update operations")
            
            # Demonstrate workaround: delete existing entities, then insert all
            print("\n  Demonstrating Workaround (Delete + Insert):")
            try:
                # Delete existing entities
                delete_expr = f"pk in [{', '.join(map(str, update_ids))}]"
                delete_result = await data_manager.delete(
                    collection_name=COLLECTION_NAME,
                    expr=delete_expr
                )
                print_info("Deleted existing", delete_result.deleted_count)
                
                # Insert all documents (updates + new)
                insert_result = await data_manager.insert(
                    collection_name=COLLECTION_NAME,
                    documents=documents
                )
                print_success(f"Inserted {insert_result.successful_count} entities (updates + new)")
                result = insert_result  # Use insert result for subsequent steps
            except Exception as workaround_error:
                print_error(f"Workaround failed: {workaround_error}")
                conn_manager.close()
                return
        else:
            print_error(f"Upsert failed: {e}")
            conn_manager.close()
            return
    
    # Step 5: Flush and Verify
    print_step(5, "Flush and Verify Changes")
    try:
        # Use ConnectionManager to execute flush operation
        await conn_manager.execute_operation_async(
            lambda alias: _flush_collection(alias, COLLECTION_NAME)
        )

        # ============================================================================
        # CRITICAL FIX: Verify upserted IDs directly instead of counting all rows
        # ============================================================================
        # TWO PROBLEMS WE'RE AVOIDING:
        # 1. Stats API (get_collection_stats) returns STALE/CACHED data after flush
        #    - Shows old count (e.g., 1010) instead of new count (1024)
        # 2. Querying "pk >= 0" hits 16,384 result limit - fails for large collections
        #
        # SOLUTION: Query the specific IDs we just upserted (only 20 IDs = fast & accurate)
        # ============================================================================
        all_upserted_ids = potential_update_ids + new_insert_ids
        upserted_ids_str = ', '.join(map(str, all_upserted_ids))
        verify_results = await conn_manager.execute_operation_async(
            lambda alias: _query_collection(
                alias,
                COLLECTION_NAME,
                f"pk in [{upserted_ids_str}]",
                ["pk"],
                limit=len(all_upserted_ids) + 5  # Small buffer
            )
        )

        verified_count = len(verify_results) if verify_results else 0
        print_info("Upserted IDs", len(all_upserted_ids))
        print_info("Verified IDs", verified_count)

        if verified_count == len(all_upserted_ids):
            print_success("All upserted entities verified successfully")
        else:
            missing = len(all_upserted_ids) - verified_count
            print_error(f"Verification failed! {missing} entities not found")
    except Exception as e:
        print_error(f"Verification failed: {e}")

    # Step 6: Query an Upserted Entity to Verify
    print_step(6, "Query an Upserted Entity to Verify")
    try:
        # Ensure collection is loaded
        description = await coll_manager.describe_collection(COLLECTION_NAME)
        if description.load_state.value != "Loaded":
            await coll_manager.load_collection(COLLECTION_NAME, wait=True)

        # Query entity with PK=1 (either updated or inserted)
        results = await conn_manager.execute_operation_async(
            lambda alias: _query_collection(alias, COLLECTION_NAME, "pk == 1", ["pk", "text", "value", "category"])
        )
        
        if results:
            entity_type = "Updated" if 1 in existing_ids else "Inserted"
            print(f"\n  {entity_type} Entity (PK=1):")
            for field, value in results[0].items():
                if field != "vector":
                    print(f"    {field}: {value}")
            if 1 in existing_ids:
                print_success("Update verified - text should start with 'UPDATED:'")
            else:
                print_success("Insert verified - entity was newly created")
        else:
            print_error("Could not retrieve entity")
    except Exception as e:
        print_error(f"Query failed: {e}")
    
    # Step 7: Cleanup
    print_step(7, "Close Connection")
    try:
        conn_manager.close()
        print_success("Connection closed")
    except Exception as e:
        print_error(f"Cleanup failed: {e}")
    
    print_section("Example Completed")
    print("\nKey Takeaways:")
    print("  • Upsert updates existing entities and inserts new ones")
    print("  • Based on primary key matching")
    print("  • Atomic operation for both insert and update")
    print("  • More efficient than separate insert/update operations")
    print("\nUse Cases:")
    print("  • Incremental data updates")
    print("  • Synchronizing external data sources")
    print("  • Handling duplicate prevention")


if __name__ == "__main__":
    asyncio.run(main())

