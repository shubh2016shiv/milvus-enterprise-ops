"""
Re-ranking Configuration

This module defines configuration for re-ranking search results.
"""

from typing import Dict, Any
from dataclasses import dataclass, field

from .base import ReRankingMethod


@dataclass
class ReRankingConfig:
    """
    Configuration for re-ranking search results.
    
    This configuration is used to specify how search results
    should be re-ranked for improved relevance using Milvus's
    native re-ranking capabilities.
    """
    enabled: bool = False
    method: ReRankingMethod = ReRankingMethod.NONE
    
    # Optional parameters for specific re-ranking methods
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate re-ranking configuration"""
        if self.enabled and self.method == ReRankingMethod.NONE:
            raise ValueError("Re-ranking is enabled but method is NONE")
        
        if self.enabled and self.method == ReRankingMethod.WEIGHTED:
            weights = self.params.get("weights")
            if not weights:
                self.params["weights"] = [0.5, 0.5]  # Default equal weights
            elif sum(weights) != 1.0:
                raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")
        
        if self.enabled and self.method == ReRankingMethod.RRF:
            if "k" not in self.params:
                self.params["k"] = 60  # Default RRF constant

