"""
Hybrid Search Configuration

This module defines configuration for hybrid search operations.
"""

from dataclasses import dataclass
import math

from .base import BaseSearchConfig


@dataclass
class HybridSearchConfig(BaseSearchConfig):
    """
    Configuration for hybrid search.

    This configuration is used for hybrid search combining
    dense vector search with sparse vector or keyword search.
    Supports vector-only mode when no sparse/keyword fields are specified.
    """

    vector_field: str = "vector"  # Dense vector field
    sparse_field: str | None = None  # Sparse vector field
    keyword_field: str | None = None  # Keyword field
    expr: str | None = None  # Filter expression

    # Weights for hybrid search (flexible, will be normalized)
    vector_weight: float = 0.7
    sparse_weight: float = 0.3

    def __post_init__(self):
        """Validate hybrid search configuration"""
        super().__post_init__()

        # Note: sparse_field and keyword_field are now truly optional
        # Vector-only mode is supported when neither is specified

        # Validate weights are non-negative and not NaN
        if (
            self.vector_weight < 0
            or self.sparse_weight < 0
            or math.isnan(self.vector_weight)
            or math.isnan(self.sparse_weight)
        ):
            raise ValueError("Weights must be non-negative and not NaN")

        # Ensure at least one weight is positive
        if self.vector_weight <= 0 and self.sparse_weight <= 0:
            raise ValueError("At least one weight must be positive")
