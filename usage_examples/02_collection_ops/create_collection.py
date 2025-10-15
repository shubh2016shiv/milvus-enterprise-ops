"""
Create Collection Example

Demonstrates how to create a Milvus collection with a schema including
vector fields and metadata fields.
"""

import sys
import os
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from connection_management import ConnectionManager
from collection_operations import CollectionManager, CollectionSchema, FieldSchema, DataType
from index_operations import IndexManager, IndexType, MetricType
from index_operations.models.parameters import IvfFlatParams
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
print_note = example_utils.print_note
cleanup_collection = example_utils.cleanup_collection


COLLECTION_NAME = "test_example_collection"
VECTOR_DIM = 128


async def main():
    """Main function to demonstrate collection creation."""
    print_section("Create Milvus Collection Example")
    
    # Step 1: Initialize Managers
    print_step(1, "Initialize Managers")
    try:
        config = load_settings()
        conn_manager = ConnectionManager(config=config)
        coll_manager = CollectionManager(conn_manager)
        index_manager = IndexManager(conn_manager, coll_manager)
        print_success("Managers initialized")
    except Exception as e:
        print_error(f"Initialization failed: {e}")
        return
    
    # Step 2: Clean Up Existing Collection
    print_step(2, "Clean Up Existing Collection (if any)")
    try:
        if await coll_manager.has_collection(COLLECTION_NAME):
            await coll_manager.drop_collection(COLLECTION_NAME)
            print_info("Cleanup", f"Dropped existing collection '{COLLECTION_NAME}'")
        else:
            print_info("Status", "No existing collection to clean up")
    except Exception as e:
        print_error(f"Cleanup failed: {e}")
    
    # Step 3: Define Collection Schema
    print_step(3, "Define Collection Schema")
    try:
        # Define fields
        id_field = FieldSchema(
            name="pk",  # Changed from 'id' as it's reserved in Milvus
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=False,
            description="Primary key field"
        )
        
        vector_field = FieldSchema(
            name="vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=VECTOR_DIM,
            description="Vector embedding field"
        )
        
        text_field = FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="Text content"
        )
        
        value_field = FieldSchema(
            name="value",
            dtype=DataType.INT64,
            description="Numeric value"
        )
        
        category_field = FieldSchema(
            name="category",
            dtype=DataType.VARCHAR,
            max_length=50,
            description="Category label"
        )
        
        # Create schema
        schema = CollectionSchema(
            fields=[id_field, vector_field, text_field, value_field, category_field],
            description="Example collection for testing",
            enable_dynamic_field=False
        )
        
        print_info("Fields defined", 5)
        print_info("Vector dimension", VECTOR_DIM)
        print_info("Primary key", "pk (manual)")
        print_success("Schema defined")
    except Exception as e:
        print_error(f"Schema definition failed: {e}")
        conn_manager.close()
        return
    
    # Step 4: Create Collection
    print_step(4, "Create Collection")
    try:
        await coll_manager.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema
        )
        print_success(f"Collection '{COLLECTION_NAME}' created")
    except Exception as e:
        print_error(f"Collection creation failed: {e}")
        conn_manager.close()
        return
    
    # Step 5: Verify Collection
    print_step(5, "Verify Collection")
    try:
        exists = await coll_manager.has_collection(COLLECTION_NAME)
        if exists:
            print_success("Collection exists in Milvus")
            
            # Get collection info
            info = await coll_manager.describe_collection(COLLECTION_NAME)
            print_info("Collection name", info.name)
            print_info("Number of fields", len(info.collection_schema.fields))
            print_info("Description", info.collection_schema.description or "N/A")
            
            print("\n  Fields:")
            for field in info.collection_schema.fields:
                field_type = field.dtype.name if hasattr(field.dtype, 'name') else str(field.dtype)
                print(f"    - {field.name}: {field_type}", end="")
                if field.is_primary:
                    print(" (PRIMARY KEY)", end="")
                if hasattr(field, 'dim') and field.dim:
                    print(f" [dim={field.dim}]", end="")
                print()
        else:
            print_error("Collection was not created")
    except Exception as e:
        print_error(f"Verification failed: {e}")
    
    # Step 6: Create Index on Vector Field
    print_step(6, "Create Index on Vector Field")
    try:
        print_info("Creating", "IVF_FLAT index with L2 metric")
        print_note("Index creation on empty collection is fast")
        
        # Define index parameters
        index_params = IvfFlatParams(nlist=128)
        
        # Create the index (don't wait since collection is empty)
        result = await index_manager.create_index(
            collection_name=COLLECTION_NAME,
            field_name="vector",
            index_type=IndexType.IVF_FLAT,
            metric_type=MetricType.L2,
            index_params=index_params,
            wait=False  # Don't wait for empty collection
        )
        
        if result.success:
            print_success(f"Index '{result.index_name}' created successfully")
            print_info("Index type", "IVF_FLAT")
            print_info("Metric type", "L2")
            print_note("Index will be built when data is inserted")
        else:
            print_error(f"Index creation failed: {result.error_message}")
    except Exception as e:
        print_error(f"Index creation failed: {e}")
    
    # Step 7: Cleanup
    print_step(7, "Close Connection")
    try:
        conn_manager.close()
        print_success("Connection closed")
    except Exception as e:
        print_error(f"Cleanup failed: {e}")
    
    print_section("Example Completed")
    print("\nKey Takeaways:")
    print("  • Define schema with fields before creation")
    print("  • Specify vector dimensions for FLOAT_VECTOR fields")
    print("  • Set max_length for VARCHAR fields")
    print("  • Primary key can be manual (auto_id=False) or auto-generated")
    print("  • Create index immediately after collection creation")
    print(f"\nCollection '{COLLECTION_NAME}' is ready for data insertion and loading!")


if __name__ == "__main__":
    asyncio.run(main())


