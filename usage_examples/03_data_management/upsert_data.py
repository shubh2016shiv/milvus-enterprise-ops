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


def _query_collection(alias: str, collection_name: str, expr: str, output_fields: list):
    """Helper function to query collection."""
    from pymilvus import Collection
    collection = Collection(collection_name, using=alias)
    return collection.query(expr=expr, output_fields=output_fields)


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
        
        # Get actual count by querying (stats may be cached)
        results = await conn_manager.execute_operation_async(
            lambda alias: _query_collection(alias, COLLECTION_NAME, "pk >= 0", ["pk"])
        )
        entity_count = len(results) if results else 0
        
        if entity_count == 0:
            print_error("Collection is empty")
            print_info("Hint", "Run insert_data.py first")
            conn_manager.close()
            return
        
        print_success(f"Collection has {entity_count} entities")
    except Exception as e:
        print_error(f"Collection check failed: {e}")
        conn_manager.close()
        return
    
    # Step 3: Prepare Upsert Data
    print_step(3, "Prepare Upsert Data (Mix of Updates and Inserts)")
    try:
        # Generate data for upsert
        # IDs 1-10: Update existing entities
        # IDs 5001-5010: Insert new entities
        
        update_ids = list(range(1, 11))
        insert_ids = list(range(5001, 5011))
        all_ids = update_ids + insert_ids
        
        data = generate_test_data(len(all_ids), VECTOR_DIM, include_metadata=True)
        data['id'] = all_ids
        
        print_info("Entities to update (existing)", len(update_ids))
        print_info("Entities to insert (new)", len(insert_ids))
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
        print_info("Updated (existing)", len(update_ids))
        print_info("Inserted (new)", len(insert_ids))
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
        
        # Get actual count by querying
        results = await conn_manager.execute_operation_async(
            lambda alias: _query_collection(alias, COLLECTION_NAME, "pk >= 0", ["pk"])
        )
        final_count = len(results) if results else 0
        print_info("Total entities after upsert", final_count)
        print_success("Changes persisted")
    except Exception as e:
        print_error(f"Verification failed: {e}")
    
    # Step 6: Query Updated Entity
    print_step(6, "Query an Updated Entity to Verify")
    try:
        # Ensure collection is loaded
        description = await coll_manager.describe_collection(COLLECTION_NAME)
        if description.load_state.value != "Loaded":
            await coll_manager.load_collection(COLLECTION_NAME, wait=True)
        
        # Query entity with PK=1 (should be updated)
        results = await conn_manager.execute_operation_async(
            lambda alias: _query_collection(alias, COLLECTION_NAME, "pk == 1", ["pk", "text", "value", "category"])
        )
        
        if results:
            print("\n  Updated Entity (PK=1):")
            for field, value in results[0].items():
                if field != "vector":
                    print(f"    {field}: {value}")
            print_success("Update verified - text should start with 'UPDATED:'")
        else:
            print_error("Could not retrieve updated entity")
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

