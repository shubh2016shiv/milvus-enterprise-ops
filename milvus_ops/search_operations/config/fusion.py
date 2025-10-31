"""
Fusion Search Configuration

This module defines configuration for fusion search operations.
"""

from dataclasses import dataclass, field
import math

from .base import BaseSearchConfig, FusionMethod
from .hybrid import HybridSearchConfig
from .semantic import SemanticSearchConfig


@dataclass
class FusionSearchConfig(BaseSearchConfig):
    """
    Configuration for fusion search.

    This configuration is used for combining results from
    multiple search methods using fusion algorithms.
    """

    method: FusionMethod = FusionMethod.RRF
    search_configs: list[SemanticSearchConfig | HybridSearchConfig] = field(default_factory=list)
    weights: list[float] | None = None

    def __post_init__(self):
        """Validate fusion search configuration"""
        super().__post_init__()

        if not self.search_configs or len(self.search_configs) < 2:
            raise ValueError("Fusion search requires at least two search configurations")

        if self.method == FusionMethod.WEIGHTED:
            if not self.weights:
                raise ValueError("Weights must be provided for weighted fusion")

            if len(self.weights) != len(self.search_configs):
                raise ValueError(
                    f"Number of weights ({len(self.weights)}) must match "
                    f"number of search configs ({len(self.search_configs)})"
                )

            if any(math.isnan(w) for w in self.weights):
                raise ValueError("Weights cannot contain NaN values")

            if abs(sum(self.weights) - 1.0) > 0.001:  # Allow small floating point error
                raise ValueError(f"Weights must sum to 1.0, got {sum(self.weights)}")
