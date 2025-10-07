"""
Hybrid Search Configuration

This module defines configuration for hybrid search operations.
"""

from typing import Optional
from dataclasses import dataclass

from .base import BaseSearchConfig


@dataclass
class HybridSearchConfig(BaseSearchConfig):
    """
    Configuration for hybrid search.
    
    This configuration is used for hybrid search combining
    dense vector search with sparse vector or keyword search.
    """
    vector_field: str = "vector"        # Dense vector field
    sparse_field: Optional[str] = None  # Sparse vector field
    keyword_field: Optional[str] = None # Keyword field
    
    # Weights for hybrid search
    vector_weight: float = 0.7
    sparse_weight: float = 0.3
    
    def __post_init__(self):
        """Validate hybrid search configuration"""
        super().__post_init__()
        
        # Ensure at least one additional field beyond vector is specified
        if not self.sparse_field and not self.keyword_field:
            raise ValueError("Hybrid search requires at least one of sparse_field or keyword_field")
        
        # Validate weights sum to 1.0
        total_weight = self.vector_weight
        if self.sparse_field:
            total_weight += self.sparse_weight
            
        if abs(total_weight - 1.0) > 0.001:  # Allow small floating point error
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

