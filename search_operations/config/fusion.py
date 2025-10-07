"""
Fusion Search Configuration

This module defines configuration for fusion search operations.
"""

from typing import List, Optional, Union
from dataclasses import dataclass, field

from .base import BaseSearchConfig, FusionMethod
from .semantic import SemanticSearchConfig
from .hybrid import HybridSearchConfig


@dataclass
class FusionSearchConfig(BaseSearchConfig):
    """
    Configuration for fusion search.
    
    This configuration is used for combining results from
    multiple search methods using fusion algorithms.
    """
    method: FusionMethod = FusionMethod.RRF
    search_configs: List[Union[SemanticSearchConfig, HybridSearchConfig]] = field(default_factory=list)
    weights: Optional[List[float]] = None
    
    def __post_init__(self):
        """Validate fusion search configuration"""
        super().__post_init__()
        
        if not self.search_configs or len(self.search_configs) < 2:
            raise ValueError("Fusion search requires at least two search configurations")
        
        if self.method == FusionMethod.WEIGHTED:
            if not self.weights:
                raise ValueError("Weights must be provided for weighted fusion")
            
            if len(self.weights) != len(self.search_configs):
                raise ValueError(f"Number of weights ({len(self.weights)}) must match number of search configs ({len(self.search_configs)})")
            
            if abs(sum(self.weights) - 1.0) > 0.001:  # Allow small floating point error
                raise ValueError(f"Weights must sum to 1.0, got {sum(self.weights)}")

