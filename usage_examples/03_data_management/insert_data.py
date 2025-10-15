"""
Insert Data Example

Demonstrates how to insert sample vectors and metadata into a Milvus collection.
Shows basic data preparation, insertion, and verification.
"""

import sys
import os
import asyncio
from data_management_operations.models.entities import OperationStatus

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
NUM_ENTITIES = 20


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
    """Main function to demonstrate data insertion."""
    print_section("Insert Data Example")
    
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
    
    # Step 3: Generate Test Data
    print_step(3, "Generate Test Data")
    try:
        print_info("Generating", f"{NUM_ENTITIES} entities")
        
        data = generate_test_data(NUM_ENTITIES, VECTOR_DIM, include_metadata=True)
        
        print_info("Vector dimension", VECTOR_DIM)
        print_info("Fields", list(data.keys()))
        print_success("Test data generated")
    except Exception as e:
        print_error(f"Data generation failed: {e}")
        conn_manager.close()
        return
    
    # Step 4: Prepare Documents for Insertion
    print_step(4, "Prepare Documents for Insertion")
    try:
        # Convert data dict to list of document dicts
        # Note: The collection uses 'pk' as the primary key field name
        documents = []
        for i in range(NUM_ENTITIES):
            doc = {
                "pk": data["id"][i],  # Map 'id' from test data to 'pk' for the collection
                "vector": data["vector"][i],
                "text": data["text"][i],
                "value": data["value"][i],
                "category": data["category"][i]
            }
            documents.append(doc)

        print_info("Documents prepared", len(documents))
        print_success("Documents ready for insertion")
    except Exception as e:
        print_error(f"Document preparation failed: {e}")
        conn_manager.close()
        return
    
    # Step 5: Insert Data
    print_step(5, "Insert Data into Collection")
    try:
        print_info("Collection", COLLECTION_NAME)
        print_info("Entities to insert", len(documents))

        result = await data_manager.insert(
            collection_name=COLLECTION_NAME,
            documents=documents
        )

        if result.status == OperationStatus.SUCCESS:
            print_success(f"Inserted {result.successful_count} entities")
            print_info("Insert IDs", result.inserted_ids[:5] if len(result.inserted_ids) > 5 else result.inserted_ids)
        else:
            print_error(f"Insertion failed with status: {result.status}")
            if result.failed_count > 0:
                print_info("Failed count", result.failed_count)
            conn_manager.close()
            return
    except Exception as e:
        print_error(f"Insertion failed: {e}")
        conn_manager.close()
        return
    
    # Step 6: Flush Data to Persist
    print_step(6, "Flush Data to Ensure Persistence")
    try:
        print_info("Flushing", "Writing data to disk...")
        
        # Use ConnectionManager to execute flush operation
        await conn_manager.execute_operation_async(
            lambda alias: _flush_collection(alias, COLLECTION_NAME)
        )
        
        print_success("Data flushed successfully")
    except Exception as e:
        print_error(f"Flush failed: {e}")
    
    # Step 7: Verify Insertion
    print_step(7, "Verify Data Insertion")
    try:
        # Load collection to ensure it's ready for querying
        description = await coll_manager.describe_collection(COLLECTION_NAME)
        if description.load_state.value != "Loaded":
            print_info("Loading", "Collection needs to be loaded for verification")
            await coll_manager.load_collection(COLLECTION_NAME, wait=True)
        
        # Use a direct query to count entities
        query_results = await conn_manager.execute_operation_async(
            lambda alias: _query_collection(
                alias,
                COLLECTION_NAME,
                "pk >= 0",  # Match all entities
                ["pk"]
            )
        )
        
        actual_count = len(query_results)
        print_info("Total entities in collection", actual_count)
        print_info("Newly inserted", result.successful_count)

        if actual_count >= result.successful_count:
            print_success("Data successfully persisted")
        else:
            print_error("Some data may not have been persisted")
    except Exception as e:
        print_error(f"Verification failed: {e}")
    
    # Step 8: Sample Inserted Data
    print_step(8, "Sample Inserted Data")
    try:
        # Check if collection is loaded
        description = await coll_manager.describe_collection(COLLECTION_NAME)
        if description.load_state.value != "Loaded":
            print_info("Loading", "Collection needs to be loaded for queries")
            await coll_manager.load_collection(COLLECTION_NAME, wait=True)

        # Query first 3 inserted entities
        first_ids = result.inserted_ids[:3]
        
        # Use ConnectionManager to execute query
        query_results = await conn_manager.execute_operation_async(
            lambda alias: _query_collection(
                alias,
                COLLECTION_NAME,
                f"pk in {first_ids}",
                ["pk", "text", "value", "category"]
            )
        )

        if query_results:
            print("\n  Sample Inserted Data:\n")
            for doc in query_results:
                print(f"  PK: {doc['pk']}")
                print(f"    Text: {doc.get('text', 'N/A')}")
                print(f"    Value: {doc.get('value', 'N/A')}")
                print(f"    Category: {doc.get('category', 'N/A')}")
                print()

            print_success("Sample data retrieved successfully")
        else:
            print_error("Could not retrieve sample data")
    except Exception as e:
        print_error(f"Sampling failed: {e}")
    
    # Step 9: Cleanup
    print_step(9, "Close Connection")
    try:
        conn_manager.close()
        print_success("Connection closed")
    except Exception as e:
        print_error(f"Cleanup failed: {e}")
    
    print_section("Example Completed")
    print("\nKey Takeaways:")
    print("  • Prepare data as list of documents")
    print("  • Use DataManager for insertion")
    print("  • Flush to ensure persistence")
    print("  • Verify insertion success")
    print("\nNext Steps:")
    print("  • Run batch_insert.py for large-scale insertion")
    print("  • Run upsert_data.py to update data")
    print("  • Run delete_data.py to remove data")


if __name__ == "__main__":
    asyncio.run(main())
