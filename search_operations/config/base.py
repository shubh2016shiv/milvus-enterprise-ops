"""
Base Search Configuration

This module defines base configuration classes and enums for search operations.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


class SearchType(str, Enum):
    """Enumeration of supported search types"""
    SEMANTIC = "semantic"  # Dense vector search
    HYBRID = "hybrid"      # Combined dense + sparse search
    FUSION = "fusion"      # Multiple search results fused together


class MetricType(str, Enum):
    """Enumeration of supported distance metrics"""
    L2 = "L2"             # Euclidean distance
    IP = "IP"             # Inner product
    COSINE = "COSINE"     # Cosine similarity
    HAMMING = "HAMMING"   # Hamming distance


class ReRankingMethod(str, Enum):
    """Enumeration of supported re-ranking methods"""
    NONE = "none"         # No re-ranking
    WEIGHTED = "weighted" # Weighted re-ranking (Milvus native)
    RRF = "rrf"           # Reciprocal Rank Fusion (Milvus native)


class FusionMethod(str, Enum):
    """Enumeration of supported fusion methods"""
    RRF = "rrf"           # Reciprocal Rank Fusion
    WEIGHTED = "weighted" # Weighted score fusion
    MAX = "max"           # Maximum score
    MEAN = "mean"         # Mean score


@dataclass
class BaseSearchConfig:
    """
    Base configuration for all search types.
    
    This class provides common parameters used across all search types,
    serving as a foundation for specialized search configurations.
    """
    top_k: int = 10
    timeout: float = 30.0
    metric_type: MetricType = MetricType.COSINE
    params: Dict[str, Any] = field(default_factory=dict)
    output_fields: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        
        # Initialize default params if not provided
        if not self.params:
            self.params = {"nprobe": 10, "ef": 64}

