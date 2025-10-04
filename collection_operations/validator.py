"""
Schema validator for Milvus collections.

This module provides validation logic for Milvus collection schemas,
ensuring they meet all Milvus requirements before attempting to create
or modify collections.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

from .schema import CollectionSchema, FieldSchema, DataType

logger = logging.getLogger(__name__)


class SchemaValidator:
    """
    Provides a centralized service for validating Milvus collection schemas.
    
    This class encapsulates all the rules and constraints that a Milvus schema
    must adhere to. By performing these "pre-flight" checks locally, it allows
    for rapid validation and clear error feedback without needing to make a
    network request to the Milvus server, which would be slower and could
    return less specific error messages.
    
    Note on validation responsibilities:
    - CollectionSchema validators (@validator, @root_validator) handle basic structural
      validation like ensuring one primary key and valid field types. These are enforced
      at model creation time.
    - SchemaValidator adds additional Milvus-specific rules that go beyond basic structure,
      such as reserved name checks, advanced vector constraints, and cross-field validations.
      These are applied before operations like create_collection.
    
    This separation keeps model definitions clean while allowing for more complex
    validation logic that may evolve over time with Milvus server requirements.
    """
    
    # Mapping of data type aliases to their canonical forms
    DTYPE_ALIASES = {
        "STRING": "VARCHAR"  # STRING is an alias for VARCHAR
    }
    
    @classmethod
    def normalize_dtype(cls, dtype: str) -> str:
        """
        Normalizes a data type string to its canonical form.
        
        This ensures that aliases like "STRING" are consistently mapped to
        their canonical form "VARCHAR" throughout the system.
        
        Args:
            dtype: The data type string to normalize.
            
        Returns:
            The normalized data type string.
        """
        return cls.DTYPE_ALIASES.get(dtype, dtype)
    
    # Maximum allowed dimensions for different vector types
    MAX_FLOAT_VECTOR_DIM = 32768
    MAX_BINARY_VECTOR_DIM = 32768
    MAX_SPARSE_VECTOR_DIM = 1000000
    
    # Maximum allowed length for string fields
    MAX_VARCHAR_LENGTH = 65535
    
    # Reserved field names in Milvus
    RESERVED_FIELD_NAMES = {
        "id", "collection_name", "timestamp", "distance", "count", "score"
    }
    
    @classmethod
    async def validate_schema(cls, schema: CollectionSchema) -> Tuple[bool, List[str]]:
        """
        Performs a comprehensive validation of a `CollectionSchema` object.
        
        This method aggregates all individual validation checks into a single
        entry point. It checks for everything from reserved names and primary
        key constraints to vector dimensions and field type requirements.

        Args:
            schema: The `CollectionSchema` object to be validated.
            
        Returns:
            A tuple containing:
            - A boolean indicating if the schema is valid (`True`) or not (`False`).
            - A list of strings, where each string is a detailed error message
              describing a validation failure. The list is empty if the schema
              is valid.
        """
        errors = []
        
        # Check for reserved field names
        cls._validate_field_names(schema, errors)
        
        # Validate primary key
        cls._validate_primary_key(schema, errors)
        
        # Validate vector fields
        cls._validate_vector_fields(schema, errors)
        
        # Validate field type constraints
        cls._validate_field_constraints(schema, errors)
        
        # Check for duplicate field names
        cls._check_duplicate_fields(schema, errors)
        
        return len(errors) == 0, errors
    
    @classmethod
    def _validate_field_names(cls, schema: CollectionSchema, errors: List[str]) -> None:
        """
        Checks that no field names conflict with Milvus's reserved keywords.
        
        Milvus uses certain field names internally (e.g., in query results).
        This validation prevents the creation of schemas that would cause
        downstream conflicts.
        """
        for field in schema.fields:
            if field.name.lower() in cls.RESERVED_FIELD_NAMES:
                errors.append(f"Field name '{field.name}' is reserved in Milvus")
    
    @classmethod
    def _validate_primary_key(cls, schema: CollectionSchema, errors: List[str]) -> None:
        """
        Ensures the schema has exactly one primary key and it is of a supported type.
        
        A valid primary key is mandatory for every Milvus collection for data
        identification and retrieval.
        """
        primary_keys = [f for f in schema.fields if f.is_primary]
        
        if not primary_keys:
            errors.append("Schema must define exactly one primary key field")
            return
        
        if len(primary_keys) > 1:
            errors.append(f"Schema defines multiple primary keys: {[pk.name for pk in primary_keys]}")
            return
        
        pk = primary_keys[0]
        if pk.dtype not in [DataType.INT64, DataType.VARCHAR]:
            errors.append(f"Primary key field '{pk.name}' must be INT64 or VARCHAR, got {pk.dtype}")
    
    @classmethod
    def _validate_vector_fields(cls, schema: CollectionSchema, errors: List[str]) -> None:
        """
        Validates constraints specific to vector fields, such as dimension limits.
        
        Correct vector dimensions are critical for the functioning of Milvus,
        as they define the vector space for similarity searches.
        """
        for field in schema.fields:
            if field.dtype == DataType.FLOAT_VECTOR:
                if field.dim is None:
                    errors.append(f"Float vector field '{field.name}' must specify dimension")
                elif field.dim <= 0 or field.dim > cls.MAX_FLOAT_VECTOR_DIM:
                    errors.append(
                        f"Float vector field '{field.name}' dimension must be between 1 and "
                        f"{cls.MAX_FLOAT_VECTOR_DIM}, got {field.dim}"
                    )
            
            elif field.dtype == DataType.BINARY_VECTOR:
                if field.dim is None:
                    errors.append(f"Binary vector field '{field.name}' must specify dimension")
                elif field.dim <= 0 or field.dim > cls.MAX_BINARY_VECTOR_DIM:
                    errors.append(
                        f"Binary vector field '{field.name}' dimension must be between 1 and "
                        f"{cls.MAX_BINARY_VECTOR_DIM}, got {field.dim}"
                    )
                # Binary vector dimensions must be a multiple of 8
                elif field.dim % 8 != 0:
                    errors.append(
                        f"Binary vector field '{field.name}' dimension must be a multiple of 8, "
                        f"got {field.dim}"
                    )
            
            elif field.dtype == DataType.SPARSE_FLOAT_VECTOR:
                if field.dim is None:
                    errors.append(f"Sparse vector field '{field.name}' must specify dimension")
                elif field.dim <= 0 or field.dim > cls.MAX_SPARSE_VECTOR_DIM:
                    errors.append(
                        f"Sparse vector field '{field.name}' dimension must be between 1 and "
                        f"{cls.MAX_SPARSE_VECTOR_DIM}, got {field.dim}"
                    )
    
    @classmethod
    def _validate_field_constraints(cls, schema: CollectionSchema, errors: List[str]) -> None:
        """
        Validates constraints for non-vector field types, like `VARCHAR` and `ARRAY`.
        
        This ensures that fields requiring additional parameters (e.g., `max_length`
        for `VARCHAR`) are correctly defined.
        """
        for field in schema.fields:
            # VARCHAR fields need max_length
            if field.dtype == DataType.VARCHAR:
                if field.max_length is None:
                    errors.append(f"VARCHAR field '{field.name}' must specify max_length")
                elif field.max_length <= 0 or field.max_length > cls.MAX_VARCHAR_LENGTH:
                    errors.append(
                        f"VARCHAR field '{field.name}' max_length must be between 1 and "
                        f"{cls.MAX_VARCHAR_LENGTH}, got {field.max_length}"
                    )
            
            # ARRAY fields need element_type
            if field.dtype == DataType.ARRAY:
                if field.element_type is None:
                    errors.append(f"ARRAY field '{field.name}' must specify element_type")
    
    @classmethod
    def _check_duplicate_fields(cls, schema: CollectionSchema, errors: List[str]) -> None:
        """
        Ensures that all field names within the schema are unique.
        
        Duplicate field names are not allowed and would lead to an error from
        the Milvus server.
        """
        field_names = [field.name for field in schema.fields]
        if len(field_names) != len(set(field_names)):
            # Find duplicates
            seen = set()
            duplicates = []
            for name in field_names:
                if name in seen:
                    duplicates.append(name)
                seen.add(name)
            errors.append(f"Schema contains duplicate field names: {duplicates}")

    @classmethod
    async def compare_schemas(
        cls, 
        schema1: CollectionSchema, 
        schema2: CollectionSchema
    ) -> Tuple[bool, List[str]]:
        """
        Compares two `CollectionSchema` objects to determine if they are compatible.
        
        "Compatibility" here means that the schemas are functionally identical,
        ignoring non-functional properties like descriptions. This is crucial for
        idempotent operations, allowing the system to verify if an existing
        collection matches a newly requested one.

        Args:
            schema1: The first `CollectionSchema` to compare.
            schema2: The second `CollectionSchema` to compare.
            
        Returns:
            A tuple containing:
            - A boolean indicating if the schemas are compatible (`True`) or not (`False`).
            - A list of strings detailing the reasons for incompatibility.
        """
        incompatibilities = []
        
        # Compare fields
        fields1 = {f.name: f for f in schema1.fields}
        fields2 = {f.name: f for f in schema2.fields}
        
        # Check for missing fields
        for name, field in fields1.items():
            if name not in fields2:
                incompatibilities.append(f"Field '{name}' exists in schema1 but not in schema2")
        
        for name, field in fields2.items():
            if name not in fields1:
                incompatibilities.append(f"Field '{name}' exists in schema2 but not in schema1")
        
        # Check for field type mismatches
        for name, field1 in fields1.items():
            if name in fields2:
                field2 = fields2[name]
                # Normalize data types before comparison
                dtype1 = cls.normalize_dtype(field1.dtype.value if hasattr(field1.dtype, 'value') else str(field1.dtype))
                dtype2 = cls.normalize_dtype(field2.dtype.value if hasattr(field2.dtype, 'value') else str(field2.dtype))
                
                # Special case: STRING and VARCHAR are aliases and should be treated as equal
                if dtype1 != dtype2:
                    incompatibilities.append(
                        f"Field '{name}' has different types: {field1.dtype} vs {field2.dtype}"
                    )
                
                # For vector fields, check dimensions
                if field1.dtype in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR, DataType.SPARSE_FLOAT_VECTOR]:
                    if field1.dim != field2.dim:
                        incompatibilities.append(
                            f"Vector field '{name}' has different dimensions: {field1.dim} vs {field2.dim}"
                        )
                
                # Check primary key consistency
                if field1.is_primary != field2.is_primary:
                    incompatibilities.append(
                        f"Field '{name}' has different primary key status: {field1.is_primary} vs {field2.is_primary}"
                    )
        
        # Check shard number
        if schema1.shard_num != schema2.shard_num:
            incompatibilities.append(
                f"Schemas have different shard numbers: {schema1.shard_num} vs {schema2.shard_num}"
            )
        
        return len(incompatibilities) == 0, incompatibilities
