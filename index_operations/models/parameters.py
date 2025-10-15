"""
Index Parameters

This module defines parameter classes for different index types in Milvus.
Each index type has specific parameters that affect its performance and accuracy.
These classes provide type-safe parameter validation and sensible defaults.

Typical usage:
    from Milvus_Ops.index_operations import HNSWParams, IndexType
    
    # Create index with type-specific parameters
    params = HNSWParams(M=16, efConstruction=200)
    
    result = await index_manager.create_index(
        collection_name="documents",
        field_name="embedding",
        index_type=IndexType.HNSW,
        metric_type=MetricType.COSINE,
        index_params=params
    )
"""

from enum import Enum
from typing import Dict, Any, Optional, Union, ClassVar, Type, List, cast
from pydantic import BaseModel, Field, validator, field_validator, model_validator

from collection_operations.schema import IndexType, MetricType


class IndexParams(BaseModel):
    """
    Base class for index parameters.
    
    This class provides common functionality for all index parameter types,
    including validation and conversion to/from dictionaries.
    
    Attributes:
        index_type: The type of index these parameters are for
    """
    index_type: ClassVar[IndexType]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a dictionary for Milvus API."""
        return self.dict(exclude={"index_type"})
    
    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'IndexParams':
        """Create parameters from a dictionary."""
        return cls(**params)


class IvfFlatParams(IndexParams):
    """
    Parameters for IVF_FLAT index.
    
    IVF_FLAT is a basic clustering-based index that divides vectors into clusters
    and performs exact search within selected clusters.
    
    Attributes:
        nlist: Number of clusters (higher values provide higher accuracy but slower performance)
    """
    index_type: ClassVar[IndexType] = IndexType.IVF_FLAT
    nlist: int = Field(1024, ge=1, description="Number of clusters")
    
    @validator('nlist')
    def validate_nlist(cls, v):
        """Validate nlist is a reasonable value."""
        if v < 1:
            raise ValueError("nlist must be at least 1")
        if v > 65536:
            raise ValueError("nlist is very large (>65536), which may cause performance issues")
        return v


class IvfSQ8Params(IndexParams):
    """
    Parameters for IVF_SQ8 index.
    
    IVF_SQ8 is a quantization-based index that reduces memory usage by using
    scalar quantization to compress vectors.
    
    Attributes:
        nlist: Number of clusters
    """
    index_type: ClassVar[IndexType] = IndexType.IVF_SQ8
    nlist: int = Field(1024, ge=1, description="Number of clusters")
    
    @validator('nlist')
    def validate_nlist(cls, v):
        """Validate nlist is a reasonable value."""
        if v < 1:
            raise ValueError("nlist must be at least 1")
        if v > 65536:
            raise ValueError("nlist is very large (>65536), which may cause performance issues")
        return v


class IvfPQParams(IndexParams):
    """
    Parameters for IVF_PQ index.
    
    IVF_PQ is a product quantization index that significantly reduces memory usage
    by compressing vectors using product quantization.
    
    Attributes:
        nlist: Number of clusters
        m: Number of vector subdivisions (must be a divisor of the dimension)
        nbits: Number of bits for each sub-vector (typically 8)
    """
    index_type: ClassVar[IndexType] = IndexType.IVF_PQ
    nlist: int = Field(1024, ge=1, description="Number of clusters")
    m: int = Field(8, ge=1, description="Number of vector subdivisions")
    nbits: int = Field(8, ge=1, le=8, description="Number of bits for each sub-vector")
    
    @model_validator(mode='after')
    def validate_parameters(self):
        """Validate parameter combinations."""
        if self.nbits > 8:
            raise ValueError("nbits must be <= 8")
        return self


class HNSWParams(IndexParams):
    """
    Parameters for HNSW index.
    
    HNSW (Hierarchical Navigable Small World) is a graph-based index that offers
    high performance for approximate nearest neighbor search.
    
    Attributes:
        M: Number of connections per layer (higher values provide higher accuracy but use more memory)
        efConstruction: Size of the dynamic candidate list during construction
    """
    index_type: ClassVar[IndexType] = IndexType.HNSW
    M: int = Field(16, ge=4, le=64, description="Number of connections per layer")
    efConstruction: int = Field(200, ge=8, description="Size of dynamic candidate list during construction")
    
    @validator('M')
    def validate_m(cls, v):
        """Validate M is within reasonable bounds."""
        if v < 4:
            raise ValueError("M must be at least 4")
        if v > 64:
            raise ValueError("M is very large (>64), which may cause memory issues")
        return v
    
    @validator('efConstruction')
    def validate_ef_construction(cls, v):
        """Validate efConstruction is within reasonable bounds."""
        if v < 8:
            raise ValueError("efConstruction must be at least 8")
        if v < 40:
            raise ValueError("efConstruction < 40 may result in lower accuracy")
        return v


class ANNOYParams(IndexParams):
    """
    Parameters for ANNOY index.
    
    ANNOY (Approximate Nearest Neighbors Oh Yeah) is a fast approximate nearest neighbor
    search algorithm optimized for static datasets.
    
    Attributes:
        n_trees: Number of trees to build (more trees give higher accuracy but slower build time)
    """
    index_type: ClassVar[IndexType] = IndexType.ANNOY
    n_trees: int = Field(8, ge=1, description="Number of trees to build")


# Factory function to create appropriate params from type
def create_index_params(
    index_type: Union[str, IndexType],
    **kwargs
) -> IndexParams:
    """
    Create index parameters based on index type.
    
    Args:
        index_type: Type of index
        **kwargs: Parameters for the index
        
    Returns:
        IndexParams instance of the appropriate type
        
    Raises:
        ValueError: If index_type is not supported
    """
    # Convert string to enum if needed
    if isinstance(index_type, str):
        try:
            index_type = IndexType(index_type.upper())
        except ValueError:
            raise ValueError(f"Unsupported index type: {index_type}")
    
    # Map index types to parameter classes
    params_map: Dict[IndexType, Type[IndexParams]] = {
        IndexType.IVF_FLAT: IvfFlatParams,
        IndexType.IVF_SQ8: IvfSQ8Params,
        IndexType.IVF_PQ: IvfPQParams,
        IndexType.HNSW: HNSWParams,
        IndexType.ANNOY: ANNOYParams
    }
    
    # Get the appropriate class
    params_class = params_map.get(index_type)
    if not params_class:
        raise ValueError(f"No parameter class defined for index type: {index_type}")
    
    # Create and return parameters
    return params_class(**kwargs)


# Map of index types to their parameter classes for type checking
INDEX_PARAMS_MAP: Dict[IndexType, Type[IndexParams]] = {
    IndexType.IVF_FLAT: IvfFlatParams,
    IndexType.IVF_SQ8: IvfSQ8Params,
    IndexType.IVF_PQ: IvfPQParams,
    IndexType.HNSW: HNSWParams,
    IndexType.ANNOY: ANNOYParams
}


def get_default_params(index_type: Union[str, IndexType]) -> IndexParams:
    """
    Get default parameters for an index type.
    
    Args:
        index_type: Type of index
        
    Returns:
        IndexParams with default values
        
    Raises:
        ValueError: If index_type is not supported
    """
    return create_index_params(index_type)
