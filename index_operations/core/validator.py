"""
Index Parameter Validator

Provides validation utilities for index parameters to ensure they are compatible
with the index type, metric type, and vector dimensions. This helps prevent
errors during index creation and provides helpful error messages.

Typical usage:
    from Milvus_Ops.index_operations import IndexValidator
    
    # Validate parameters
    validator = IndexValidator()
    validator.validate_index_params(
        index_type=IndexType.HNSW,
        metric_type=MetricType.COSINE,
        dimension=128,
        params={"M": 16, "efConstruction": 200}
    )
"""

import logging
from typing import Dict, Any, Optional, Union, List, Tuple, Set
import math

from collection_operations.schema import IndexType, MetricType, FieldSchema, DataType
from index_operations.models.parameters import (
    IndexParams,
    IvfFlatParams,
    IvfSQ8Params,
    IvfPQParams,
    HNSWParams,
    ANNOYParams,
    create_index_params
)
from index_operations.index_ops_exceptions import IndexParameterError, IndexTypeError

logger = logging.getLogger(__name__)


class IndexValidator:
    """
    Validates index parameters for compatibility and correctness.
    
    This class provides methods for validating that index parameters
    are compatible with the index type, metric type, and vector dimensions.
    It also provides methods for optimizing parameters based on data
    characteristics.
    """
    
    # Map of index types to supported metric types
    _METRIC_COMPATIBILITY: Dict[IndexType, Set[MetricType]] = {
        IndexType.FLAT: {MetricType.L2, MetricType.IP, MetricType.COSINE},
        IndexType.IVF_FLAT: {MetricType.L2, MetricType.IP, MetricType.COSINE},
        IndexType.IVF_SQ8: {MetricType.L2, MetricType.IP, MetricType.COSINE},
        IndexType.IVF_PQ: {MetricType.L2, MetricType.IP, MetricType.COSINE},
        IndexType.HNSW: {MetricType.L2, MetricType.IP, MetricType.COSINE},
        IndexType.ANNOY: {MetricType.L2, MetricType.IP, MetricType.COSINE},
        IndexType.RHNSW_FLAT: {MetricType.L2, MetricType.IP, MetricType.COSINE},
        IndexType.RHNSW_SQ: {MetricType.L2, MetricType.IP, MetricType.COSINE},
        IndexType.RHNSW_PQ: {MetricType.L2, MetricType.IP, MetricType.COSINE},
        IndexType.BIN_FLAT: {MetricType.HAMMING, MetricType.JACCARD, MetricType.TANIMOTO},
        IndexType.BIN_IVF_FLAT: {MetricType.HAMMING, MetricType.JACCARD, MetricType.TANIMOTO},
        IndexType.SPARSE_INVERTED_INDEX: {MetricType.SPARSE_IP, MetricType.SPARSE_COSINE},
        IndexType.ANNOY: {MetricType.L2, MetricType.IP, MetricType.COSINE}
    }
    
    # Map of index types to supported vector types
    _VECTOR_TYPE_COMPATIBILITY: Dict[IndexType, Set[DataType]] = {
        IndexType.FLAT: {DataType.FLOAT_VECTOR},
        IndexType.IVF_FLAT: {DataType.FLOAT_VECTOR},
        IndexType.IVF_SQ8: {DataType.FLOAT_VECTOR},
        IndexType.IVF_PQ: {DataType.FLOAT_VECTOR},
        IndexType.HNSW: {DataType.FLOAT_VECTOR},
        IndexType.ANNOY: {DataType.FLOAT_VECTOR},
        IndexType.RHNSW_FLAT: {DataType.FLOAT_VECTOR},
        IndexType.RHNSW_SQ: {DataType.FLOAT_VECTOR},
        IndexType.RHNSW_PQ: {DataType.FLOAT_VECTOR},
        IndexType.BIN_FLAT: {DataType.BINARY_VECTOR},
        IndexType.BIN_IVF_FLAT: {DataType.BINARY_VECTOR},
        IndexType.SPARSE_INVERTED_INDEX: {DataType.SPARSE_FLOAT_VECTOR},
        IndexType.ANNOY: {DataType.FLOAT_VECTOR}
    }
    
    # Dimension constraints for specific index types
    _DIMENSION_CONSTRAINTS: Dict[IndexType, Dict[str, Any]] = {
        IndexType.IVF_PQ: {
            "divisible_by": 1,  # Must be divisible by this number
            "min": 1,
            "max": None
        },
        IndexType.BIN_FLAT: {
            "divisible_by": 8,  # Binary vectors must be divisible by 8
            "min": 8,
            "max": None
        }
    }
    
    def validate_index_params(
        self,
        index_type: Union[str, IndexType],
        metric_type: Union[str, MetricType],
        dimension: int,
        params: Optional[Union[Dict[str, Any], IndexParams]] = None,
        field_type: Optional[Union[str, DataType]] = None
    ) -> IndexParams:
        """
        Validate index parameters for compatibility and correctness.
        
        This method performs comprehensive validation of index parameters,
        checking for compatibility with the index type, metric type, and
        vector dimensions. It also converts the parameters to the appropriate
        type-specific parameter class.
        
        Args:
            index_type: Type of index
            metric_type: Metric type for similarity search
            dimension: Dimension of the vector field
            params: Parameters for the index (dict or IndexParams)
            field_type: Type of the field (e.g., FLOAT_VECTOR, BINARY_VECTOR)
            
        Returns:
            Validated IndexParams instance of the appropriate type
            
        Raises:
            IndexTypeError: If the index type is not supported
            IndexParameterError: If the parameters are invalid or incompatible
        """
        # Convert string types to enums if needed
        if isinstance(index_type, str):
            try:
                index_type = IndexType(index_type.upper())
            except ValueError:
                raise IndexTypeError(
                    f"Unsupported index type: {index_type}",
                    index_type=index_type,
                    supported_types=[t.value for t in IndexType]
                )
        
        if isinstance(metric_type, str):
            try:
                metric_type = MetricType(metric_type.upper())
            except ValueError:
                raise IndexParameterError(
                    f"Unsupported metric type: {metric_type}",
                    index_type=str(index_type),
                    parameter_errors={"metric_type": f"Unsupported metric type: {metric_type}"}
                )
        
        if field_type is not None and isinstance(field_type, str):
            try:
                field_type = DataType(field_type.upper())
            except ValueError:
                raise IndexParameterError(
                    f"Unsupported field type: {field_type}",
                    index_type=str(index_type),
                    parameter_errors={"field_type": f"Unsupported field type: {field_type}"}
                )
        
        # Validate metric type compatibility
        self.validate_metric_compatibility(index_type, metric_type)
        
        # Validate field type compatibility if provided
        if field_type is not None:
            self.validate_field_type_compatibility(index_type, field_type)
        
        # Validate dimension compatibility
        self.validate_dimension_compatibility(index_type, dimension)
        
        # Convert params to appropriate type if needed
        if params is None:
            # Use default parameters
            params_dict = {}
        elif isinstance(params, dict):
            params_dict = params
        elif isinstance(params, IndexParams):
            if params.index_type != index_type:
                raise IndexParameterError(
                    f"Index parameters are for {params.index_type}, but index type is {index_type}",
                    index_type=str(index_type),
                    parameter_errors={"index_type": f"Parameter mismatch: {params.index_type} != {index_type}"}
                )
            return params
        else:
            raise IndexParameterError(
                f"Invalid index parameters type: {type(params)}",
                index_type=str(index_type)
            )
        
        # Create and validate type-specific parameters
        try:
            index_params = create_index_params(index_type, **params_dict)
            return index_params
        except ValueError as e:
            raise IndexParameterError(
                f"Invalid index parameters: {str(e)}",
                index_type=str(index_type),
                parameter_errors={"params": str(e)}
            )
    
    def validate_metric_compatibility(
        self,
        index_type: IndexType,
        metric_type: MetricType
    ) -> None:
        """
        Validate that the metric type is compatible with the index type.
        
        Args:
            index_type: Type of index
            metric_type: Metric type for similarity search
            
        Raises:
            IndexParameterError: If the metric type is incompatible with the index type
        """
        compatible_metrics = self._METRIC_COMPATIBILITY.get(index_type, set())
        if metric_type not in compatible_metrics:
            raise IndexParameterError(
                f"Metric type {metric_type} is not compatible with index type {index_type}",
                index_type=str(index_type),
                parameter_errors={
                    "metric_type": f"Incompatible with {index_type}. "
                                  f"Supported metrics: {', '.join(str(m) for m in compatible_metrics)}"
                }
            )
    
    def validate_field_type_compatibility(
        self,
        index_type: IndexType,
        field_type: DataType
    ) -> None:
        """
        Validate that the field type is compatible with the index type.
        
        Args:
            index_type: Type of index
            field_type: Type of the field
            
        Raises:
            IndexParameterError: If the field type is incompatible with the index type
        """
        compatible_types = self._VECTOR_TYPE_COMPATIBILITY.get(index_type, set())
        if field_type not in compatible_types:
            raise IndexParameterError(
                f"Field type {field_type} is not compatible with index type {index_type}",
                index_type=str(index_type),
                parameter_errors={
                    "field_type": f"Incompatible with {index_type}. "
                                 f"Supported types: {', '.join(str(t) for t in compatible_types)}"
                }
            )
    
    def validate_dimension_compatibility(
        self,
        index_type: IndexType,
        dimension: int
    ) -> None:
        """
        Validate that the dimension is compatible with the index type.
        
        Args:
            index_type: Type of index
            dimension: Dimension of the vector field
            
        Raises:
            IndexParameterError: If the dimension is incompatible with the index type
        """
        constraints = self._DIMENSION_CONSTRAINTS.get(index_type)
        if constraints is None:
            # No specific constraints for this index type
            return
        
        errors = []
        
        # Check minimum dimension
        min_dim = constraints.get("min")
        if min_dim is not None and dimension < min_dim:
            errors.append(f"Dimension must be at least {min_dim}")
        
        # Check maximum dimension
        max_dim = constraints.get("max")
        if max_dim is not None and dimension > max_dim:
            errors.append(f"Dimension must be at most {max_dim}")
        
        # Check divisibility
        divisible_by = constraints.get("divisible_by")
        if divisible_by is not None and dimension % divisible_by != 0:
            errors.append(f"Dimension must be divisible by {divisible_by}")
        
        if errors:
            raise IndexParameterError(
                f"Invalid dimension {dimension} for index type {index_type}: {', '.join(errors)}",
                index_type=str(index_type),
                parameter_errors={"dimension": ', '.join(errors)}
            )
    
    def optimize_parameters(
        self,
        index_type: IndexType,
        dimension: int,
        row_count: Optional[int] = None
    ) -> IndexParams:
        """
        Suggest optimal parameters based on data characteristics.
        
        This method provides optimized parameters for the given index type
        based on the vector dimension and dataset size.
        
        Args:
            index_type: Type of index
            dimension: Dimension of the vector field
            row_count: Number of vectors in the dataset
            
        Returns:
            Optimized IndexParams instance
        """
        # Default parameters for each index type
        if index_type == IndexType.IVF_FLAT:
            # For IVF indexes, nlist should scale with sqrt(row_count)
            nlist = 1024
            if row_count:
                nlist = min(16384, max(1, int(4 * math.sqrt(row_count))))
            return IvfFlatParams(nlist=nlist)
            
        elif index_type == IndexType.IVF_SQ8:
            nlist = 1024
            if row_count:
                nlist = min(16384, max(1, int(4 * math.sqrt(row_count))))
            return IvfSQ8Params(nlist=nlist)
            
        elif index_type == IndexType.IVF_PQ:
            nlist = 1024
            if row_count:
                nlist = min(16384, max(1, int(4 * math.sqrt(row_count))))
            
            # For PQ, m should divide the dimension
            m = 8
            for divisor in [16, 12, 8, 4]:
                if dimension % divisor == 0:
                    m = divisor
                    break
            
            return IvfPQParams(nlist=nlist, m=m, nbits=8)
            
        elif index_type == IndexType.HNSW:
            # For HNSW, M and efConstruction affect index quality and build time
            M = 16
            efConstruction = 200
            
            # For larger dimensions, reduce M slightly
            if dimension > 1000:
                M = 12
            
            # For larger datasets, increase efConstruction
            if row_count and row_count > 1000000:
                efConstruction = 400
            
            return HNSWParams(M=M, efConstruction=efConstruction)
            
        elif index_type == IndexType.ANNOY:
            nlist = 1024
            if row_count:
                nlist = min(16384, max(1, int(4 * math.sqrt(row_count))))
            
            return ANNOYParams(n_trees=8)
            
        else:
            # For other index types, use default parameters
            return create_index_params(index_type)
    
    def estimate_memory_usage(
        self,
        index_type: IndexType,
        dimension: int,
        row_count: int,
        params: Optional[IndexParams] = None
    ) -> int:
        """
        Estimate memory usage for an index.
        
        This method provides a rough estimate of the memory usage for an index
        based on the index type, vector dimension, dataset size, and parameters.
        
        Args:
            index_type: Type of index
            dimension: Dimension of the vector field
            row_count: Number of vectors in the dataset
            params: Index parameters
            
        Returns:
            Estimated memory usage in bytes
        """
        # Base memory usage for vectors (assuming float32)
        base_memory = row_count * dimension * 4
        
        # Index-specific overhead
        if index_type == IndexType.FLAT:
            # FLAT index stores vectors as-is
            return base_memory
            
        elif index_type == IndexType.IVF_FLAT:
            # IVF_FLAT adds clustering overhead
            return int(base_memory * 1.05)
            
        elif index_type == IndexType.IVF_SQ8:
            # IVF_SQ8 compresses vectors to 1 byte per dimension
            return int(row_count * dimension + base_memory * 0.05)
            
        elif index_type == IndexType.IVF_PQ:
            # IVF_PQ has significant compression
            if params and isinstance(params, IvfPQParams):
                m = params.m
                compression_ratio = m / dimension
                return int(base_memory * compression_ratio * 1.1)
            return int(base_memory * 0.25)  # Rough estimate without params
            
        elif index_type == IndexType.HNSW:
            # HNSW has significant overhead for graph connections
            if params and isinstance(params, HNSWParams):
                M = params.M
                # Each vector has M connections on average
                return int(base_memory + row_count * M * 8)
            return int(base_memory * 1.5)  # Rough estimate without params
            
        else:
            # Default estimate for other index types
            return int(base_memory * 1.2)
