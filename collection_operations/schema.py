"""
Schema models for Milvus collections.

This module defines Pydantic models for representing Milvus collection schemas
in a strongly-typed manner. These models provide validation, serialization,
and a clean interface for defining collection structures.

Note: This module uses Pydantic v1 style validators (@validator, @root_validator).
If upgrading to Pydantic v2, these would need to be replaced with @field_validator
and @model_validator respectively. The current implementation is compatible with
Pydantic v1.x.
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class DataType(str, Enum):
    """
    Enumeration of supported Milvus data types for collection fields.
    
    This enum provides a strongly-typed way to specify data types, preventing
    common errors from using incorrect or unsupported string values. It maps
    directly to the data types supported by the Milvus SDK.
    """
    BOOL = "BOOL"
    INT8 = "INT8"
    INT16 = "INT16"
    INT32 = "INT32"
    INT64 = "INT64"
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"
    STRING = "VARCHAR"  # Note: This is serialized as "VARCHAR" in Milvus for compatibility
    VARCHAR = "VARCHAR"  # Canonical name in Milvus
    BINARY_VECTOR = "BINARY_VECTOR"
    FLOAT_VECTOR = "FLOAT_VECTOR"
    SPARSE_FLOAT_VECTOR = "SPARSE_FLOAT_VECTOR"
    JSON = "JSON"
    ARRAY = "ARRAY"


class IndexType(str, Enum):
    """
    Enumeration of supported Milvus index types.
    
    Milvus supports various index types optimized for different use cases and
    performance characteristics. This enum provides a comprehensive list of
    available index types, ensuring that only valid indexes are specified.
    """
    FLAT = "FLAT"
    IVF_FLAT = "IVF_FLAT"
    IVF_SQ8 = "IVF_SQ8"
    IVF_PQ = "IVF_PQ"
    HNSW = "HNSW"
    ANNOY = "ANNOY"
    RHNSW_FLAT = "RHNSW_FLAT"
    RHNSW_SQ = "RHNSW_SQ"
    RHNSW_PQ = "RHNSW_PQ"
    BIN_FLAT = "BIN_FLAT"
    BIN_IVF_FLAT = "BIN_IVF_FLAT"
    SPARSE_INVERTED_INDEX = "SPARSE_INVERTED_INDEX"
    AUTOINDEX = "AUTOINDEX"


class MetricType(str, Enum):
    """
    Enumeration of supported Milvus distance metrics for vector similarity search.
    
    The choice of metric is crucial for the performance and accuracy of vector
    searches. This enum ensures that a valid and appropriate metric is chosen
    for the vector field's index.
    """
    L2 = "L2"
    IP = "IP"
    COSINE = "COSINE"
    HAMMING = "HAMMING"
    JACCARD = "JACCARD"
    TANIMOTO = "TANIMOTO"
    SUBSTRUCTURE = "SUBSTRUCTURE"
    SUPERSTRUCTURE = "SUPERSTRUCTURE"
    SPARSE_IP = "SPARSE_IP"
    SPARSE_COSINE = "SPARSE_COSINE"


class IndexParams(BaseModel):
    """
    A Pydantic model representing index parameters for a vector field.
    
    This model defines the index type and parameters that should be used for
    a vector field. While not directly used for collection creation, it serves
    as metadata for future index creation operations.

    Attributes:
        index_type: The type of index to create.
        metric_type: The distance metric to use for similarity search.
        params: Additional parameters specific to the index type.
    """
    index_type: IndexType = Field(..., description="The type of index to use.")
    metric_type: MetricType = Field(..., description="The distance metric to use for similarity search.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters for the index.")


class FieldSchema(BaseModel):
    """
    A Pydantic model representing the schema for a single field in a Milvus collection.
    
    This class provides a structured and validated way to define the properties of
    a field, such as its name, data type, and other constraints. It serves as the
    building block for the overall collection schema.

    Attributes:
        name: The name of the field, which must be unique within the collection.
        dtype: The data type of the field, chosen from the `DataType` enum.
        description: An optional description for the field, useful for documentation.
        is_primary: A boolean indicating if this field is the primary key. Each collection
                    must have exactly one primary key.
        is_partition_key: A boolean indicating if this field is used for partitioning.
                          Partitioning can improve query performance by scoping searches.
        auto_id: If `True` for a primary key field, Milvus will automatically generate
                 unique IDs for inserted entities.
        dim: The dimension of the vector, required for `FLOAT_VECTOR`, `BINARY_VECTOR`,
             and `SPARSE_FLOAT_VECTOR` fields.
        max_length: The maximum length for `VARCHAR` fields, required to pre-allocate
                    resources.
        element_type: The data type of elements in an `ARRAY` field.
        index_params: Optional index parameters for vector fields. This is metadata
                      only and does not affect collection creation.
    """
    name: str = Field(..., description="The unique name of the field within the collection.")
    dtype: DataType = Field(..., description="The data type of the field.")
    description: Optional[str] = Field(None, description="An optional description for the field.")
    is_primary: bool = Field(False, description="Indicates if this field is the primary key.")
    is_partition_key: bool = Field(False, description="Indicates if this field is a partition key.")
    auto_id: bool = Field(False, description="If True, Milvus automatically generates IDs for this primary key field.")
    dim: Optional[int] = Field(None, description="The dimension of the vector field.")
    max_length: Optional[int] = Field(None, description="The maximum length for a VARCHAR field.")
    element_type: Optional[DataType] = Field(None, description="The data type of elements in an ARRAY field.")
    index_params: Optional[IndexParams] = Field(None, description="Optional index parameters for vector fields (metadata only).")
    
    @field_validator('dim')
    def validate_vector_dim(cls, v, info):
        """Validate that vector fields have dimensions specified."""
        dtype = info.data.get('dtype')
        if dtype in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR, DataType.SPARSE_FLOAT_VECTOR] and v is None:
            raise ValueError(f"Field with dtype {dtype} must specify 'dim'")
        return v
    
    @field_validator('max_length')
    def validate_string_length(cls, v, info):
        """Validate that string fields have max_length specified."""
        dtype = info.data.get('dtype')
        if dtype == DataType.VARCHAR and v is None:
            raise ValueError("VARCHAR fields must specify 'max_length'")
        return v
    
    @field_validator('element_type')
    def validate_array_element_type(cls, v, info):
        """Validate that array fields have element_type specified."""
        dtype = info.data.get('dtype')
        if dtype == DataType.ARRAY and v is None:
            raise ValueError("ARRAY fields must specify 'element_type'")
        return v
    
    @model_validator(mode='after')
    def validate_primary_key(cls, model):
        """Validate primary key constraints."""
        if model.is_primary:
            allowed_pk_types = [DataType.INT64, DataType.VARCHAR]
            if model.dtype not in allowed_pk_types:
                raise ValueError(f"Primary key must be one of {allowed_pk_types}, got {model.dtype}")
        return model


class CollectionSchema(BaseModel):
    """
    A Pydantic model representing the schema for a complete Milvus collection.
    
    This class aggregates a list of `FieldSchema` objects and defines collection-level
    properties. It provides a single, validated object to define the entire structure
    of a Milvus collection, ensuring correctness before it's sent to the server.
    
    Note on validation responsibilities:
    - The validators in this class (@validator, @root_validator) handle basic structural
      validation that can be enforced at model creation time, like ensuring exactly one
      primary key field and valid partition key types.
    - More complex Milvus-specific validations are handled by the SchemaValidator class,
      which performs additional checks before operations like create_collection.

    Attributes:
        fields: A list of `FieldSchema` objects that define the columns of the collection.
        description: An optional description for the collection.
        enable_dynamic_field: If `True`, allows insertion of data into fields that are
                              not explicitly defined in the schema. This is useful for
                              unstructured or evolving data.
        shard_num: The number of shards the collection will be divided into. Sharding
                   is a key mechanism for scaling and distributing data in Milvus.
    """
    fields: List[FieldSchema] = Field(..., description="A list of fields defining the collection's structure.")
    description: Optional[str] = Field(None, description="An optional description for the collection.")
    enable_dynamic_field: bool = Field(False, description="Allows dynamic fields if set to True. When enabled, fields not defined in the schema can be inserted.")
    shard_num: int = Field(2, description="The number of shards to distribute the collection data into. Default is 2. Higher values improve write throughput for large collections.")
    
    @field_validator('shard_num')
    def validate_shard_num(cls, value):
        """Validate that shard_num is at least 1."""
        if value < 1:
            raise ValueError("shard_num must be at least 1")
        return value
    
    @field_validator('fields')
    def validate_fields(cls, fields):
        """Validate that the collection has at least one field and one primary key."""
        if not fields:
            raise ValueError("Collection must have at least one field")
        
        # Check for primary key
        primary_keys = [f for f in fields if f.is_primary]
        if not primary_keys:
            raise ValueError("Collection must have exactly one primary key field")
        if len(primary_keys) > 1:
            raise ValueError(f"Collection has multiple primary keys: {[pk.name for pk in primary_keys]}")
        
        # Check for partition keys
        partition_keys = [f for f in fields if f.is_partition_key]
        if len(partition_keys) > 1:
            raise ValueError(f"Collection has multiple partition keys: {[pk.name for pk in partition_keys]}")
        
        # If there is a partition key, validate its type
        if partition_keys:
            pk = partition_keys[0]
            allowed_types = [DataType.INT64, DataType.VARCHAR, DataType.INT32, DataType.INT16, DataType.INT8]
            if pk.dtype not in allowed_types:
                raise ValueError(f"Partition key must be one of {allowed_types}, got {pk.dtype}")
        
        return fields
    
    def get_field_by_name(self, name: str) -> Optional[FieldSchema]:
        """
        Retrieves a field from the schema by its name.

        Args:
            name: The name of the field to retrieve.

        Returns:
            The `FieldSchema` object if found, otherwise `None`.
        """
        for field in self.fields:
            if field.name == name:
                return field
        return None
    
    def get_primary_key_field(self) -> FieldSchema:
        """
        Retrieves the primary key field from the schema.

        This is a convenience method to easily access the primary key, which is a
        mandatory component of any collection schema.

        Returns:
            The `FieldSchema` object for the primary key.

        Raises:
            ValueError: If no primary key field is found in the schema.
        """
        for field in self.fields:
            if field.is_primary:
                return field
        # This should never happen due to the validator
        raise ValueError("No primary key field found")
    
    def get_vector_fields(self) -> List[FieldSchema]:
        """
        Retrieves all vector fields from the schema.

        Vector fields are the core of Milvus and are used for similarity searches.
        This method helps in identifying them for indexing or querying purposes.

        Returns:
            A list of `FieldSchema` objects for all vector fields.
        """
        return [f for f in self.fields if f.dtype in 
                [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR, DataType.SPARSE_FLOAT_VECTOR]]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the schema object to a dictionary.

        This is useful for converting the structured Pydantic model into a format
        that can be used by the Milvus SDK or for other serialization purposes.

        Returns:
            A dictionary representation of the schema.
        """
        return self.dict(exclude_none=True, by_alias=True)
    
    def compute_hash(self) -> str:
        """
        Compute a deterministic hash of the schema for compatibility checks.
        
        This hash can be used to detect schema changes and prevent accidental
        mismatches between application code and the actual schema in Milvus.
        
        It only includes "functional" fields that affect data compatibility (like
        data types, dimensions, and keys), while ignoring non-functional fields
        like descriptions to prevent unnecessary hash changes.

        Returns:
            A SHA-256 hash string representing the functional schema.
        """
        import hashlib
        import json
        
        # Create a normalized representation for hashing with only functional fields
        functional_schema = {
            "fields": [],
            "enable_dynamic_field": self.enable_dynamic_field,
            "shard_num": self.shard_num
        }
        
        # Include only functional field properties
        for field in self.fields:
            functional_field = {
                "name": field.name,
                "dtype": field.dtype.value if hasattr(field.dtype, 'value') else str(field.dtype),
                "is_primary": field.is_primary,
                "auto_id": field.auto_id,
                "is_partition_key": field.is_partition_key
            }
            
            # Include type-specific properties
            if field.dim is not None:
                functional_field["dim"] = field.dim
                
            if field.max_length is not None:
                functional_field["max_length"] = field.max_length
                
            if field.element_type is not None:
                # Convert enum to string to ensure JSON serialization works
                functional_field["element_type"] = field.element_type.value if hasattr(field.element_type, 'value') else str(field.element_type)
                
            functional_schema["fields"].append(functional_field)
        
        # Sort fields by name for deterministic ordering
        functional_schema["fields"] = sorted(functional_schema["fields"], key=lambda x: x["name"])
        
        # Convert to a canonical JSON string
        canonical = json.dumps(functional_schema, sort_keys=True)
        
        # Compute SHA-256 hash
        return hashlib.sha256(canonical.encode()).hexdigest()
