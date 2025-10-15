"""
Data Validator

Provides validation utilities for documents before insertion into Milvus collections.
Ensures data integrity by validating against collection schemas, checking field types,
vector dimensions, and other schema constraints.

Typical usage from external projects:

    from Milvus_Ops.data_management_operations import DataValidator
    from Milvus_Ops.collection_operations import CollectionSchema
    
    # Validate documents against schema
    validation_result = await DataValidator.validate_documents(documents, schema)
    
    if not validation_result.is_valid:
        for doc_id, errors in validation_result.errors.items():
            print(f"Document {doc_id} validation errors: {errors}")
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import numpy as np

from collection_operations import CollectionSchema, FieldSchema, DataType
from data_management_operations.models.entities import Document, DocumentBase, DataValidationResult

logger = logging.getLogger(__name__)


# Mapping from Milvus DataType enum to Python types for validation
_DATA_TYPE_TO_PYTHON_TYPE = {
    DataType.BOOL: bool,
    DataType.INT8: int,
    DataType.INT16: int,
    DataType.INT32: int,
    DataType.INT64: int,
    DataType.FLOAT: float,
    DataType.DOUBLE: float,
    DataType.VARCHAR: str,
    DataType.STRING: str,
    DataType.JSON: dict,
    # Vectors are handled separately
}


class DataValidator:
    """
    Validates data before insertion into Milvus collections.
    
    This class provides methods for validating documents against a collection's
    schema, ensuring that all required fields are present and correctly typed.
    It helps prevent errors during insertion and maintains data integrity.
    """
    
    @classmethod
    async def validate_documents(
        cls,
        documents: List[Union[DocumentBase, Dict[str, Any]]],
        schema: CollectionSchema
    ) -> DataValidationResult:
        """
        Validate a list of documents against a collection schema.
        
        This method performs comprehensive validation of documents, checking for:
        - Presence of all required fields defined in the schema.
        - Correct data types for each field.
        - Vector dimension matching for all vector fields.
        
        Args:
            documents: List of documents to validate. The documents should be instances
                       of a model inheriting from DocumentBase.
            schema: The collection schema to validate against.
            
        Returns:
            A DataValidationResult with validation status and a dictionary of any errors.
        """
        errors: Dict[Union[int, str], List[str]] = {}
        schema_fields = {f.name: f for f in schema.fields}

        # Pre-calculate vector fields for efficiency
        vector_fields = schema.get_vector_fields()
        vector_field_names = {f.name for f in vector_fields}

        for i, doc in enumerate(documents):
            doc_errors: List[str] = []
            # Use the document's ID if available, otherwise use its index in the list.
            if hasattr(doc, 'id'):
                doc_id = doc.id if doc.id is not None else i
            else:
                doc_id = doc.get('id', i)

            # Convert document to dictionary
            if hasattr(doc, 'dict'):
                # Pydantic model
                doc_dict = doc.dict()
            else:
                # Already a dictionary
                doc_dict = doc

            # 1. Check for missing required fields and correct types
            for field_name, schema_field in schema_fields.items():
                # Skip auto-id primary keys if no ID is provided in the document
                if schema_field.auto_id and schema_field.is_primary and doc_dict.get(field_name) is None:
                    continue

                if field_name not in doc_dict:
                    doc_errors.append(f"Missing required field '{field_name}'.")
                    continue  # Skip further checks for this missing field

                # Check type if a mapping exists
                python_type = _DATA_TYPE_TO_PYTHON_TYPE.get(schema_field.dtype)
                if python_type:
                    if not isinstance(doc_dict[field_name], python_type):
                        doc_errors.append(
                            f"Invalid type for field '{field_name}'. "
                            f"Expected {python_type.__name__}, got {type(doc_dict[field_name]).__name__}."
                        )

            # 2. Validate vector fields
            for vector_field in vector_fields:
                field_name = vector_field.name
                vector_value = doc_dict.get(field_name)

                if vector_value is None:
                    doc_errors.append(f"Missing vector data for field '{field_name}'.")
                    continue

                if not isinstance(vector_value, list):
                    doc_errors.append(f"Vector field '{field_name}' must be a list, but got {type(vector_value).__name__}.")
                    continue
                
                if len(vector_value) != vector_field.dim:
                    doc_errors.append(
                        f"Vector dimension mismatch for '{field_name}'. "
                        f"Expected {vector_field.dim}, got {len(vector_value)}."
                    )
            
            # 3. Check for extraneous fields not defined in the schema if dynamic fields are disabled
            if not schema.enable_dynamic_field:
                for doc_field_name in doc_dict:
                    if doc_field_name not in schema_fields:
                        doc_errors.append(f"Unexpected field '{doc_field_name}' found in document for a non-dynamic schema.")

            if doc_errors:
                errors[doc_id] = doc_errors
        
        is_valid = not errors
        if not is_valid:
            logger.warning(f"Data validation failed for {len(errors)} documents. Errors: {errors}")
            
        return DataValidationResult(is_valid=is_valid, errors=errors)
    
    @classmethod
    def prepare_documents_for_insertion(
        cls,
        documents: List[Union[DocumentBase, Dict[str, Any]]],
        schema: CollectionSchema
    ) -> List[Dict[str, Any]]:
        """
        Prepare documents for insertion by converting them to the format expected by Milvus.

        This method converts a list of Pydantic models or dictionaries into the list of entity
        dictionaries format that the PyMilvus client expects for `insert` and `upsert` operations.
        It ensures that all fields defined in the schema are present for each document, inserting
        `None` for any missing values, which PyMilvus handles correctly.

        Args:
            documents: List of documents to prepare (can be DocumentBase models or dicts).
            schema: The collection schema to prepare against

        Returns:
            A list of dictionaries, where each dictionary represents one entity
        """
        field_names = [field.name for field in schema.fields]
        prepared_documents: List[Dict[str, Any]] = []

        # Process each document
        for doc in documents:
            # Convert document to dictionary
            if hasattr(doc, 'dict'):
                # Pydantic model
                doc_dict = doc.dict()
            else:
                # Already a dictionary
                doc_dict = doc

            # Create a new document with all schema fields
            prepared_doc = {}
            for field_name in field_names:
                prepared_doc[field_name] = doc_dict.get(field_name)
            
            prepared_documents.append(prepared_doc)

        return prepared_documents
