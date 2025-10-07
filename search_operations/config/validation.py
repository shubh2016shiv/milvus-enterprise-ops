"""
Search Parameters Validation

This module defines Pydantic models for API-level parameter validation.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator

from .base import SearchType, MetricType, ReRankingMethod, FusionMethod


class SearchParams(BaseModel):
    """
    Pydantic model for search parameters validation.
    
    This provides a clean interface for API-level parameter validation
    before converting to the appropriate dataclass configuration.
    """
    search_type: SearchType = Field(SearchType.SEMANTIC, description="Type of search to perform")
    top_k: int = Field(10, gt=0, description="Number of results to return")
    metric_type: MetricType = Field(MetricType.COSINE, description="Distance metric to use")
    timeout: float = Field(30.0, gt=0, description="Search timeout in seconds")
    
    # Fields for semantic search
    vector_field: str = Field("vector", description="Vector field name")
    expr: Optional[str] = Field(None, description="Filter expression")
    
    # Fields for hybrid search
    sparse_field: Optional[str] = Field(None, description="Sparse vector field name")
    keyword_field: Optional[str] = Field(None, description="Keyword field name")
    vector_weight: float = Field(0.7, ge=0, le=1, description="Weight for vector search")
    sparse_weight: float = Field(0.3, ge=0, le=1, description="Weight for sparse search")
    
    # Fields for re-ranking
    rerank: bool = Field(False, description="Whether to apply re-ranking")
    rerank_method: ReRankingMethod = Field(ReRankingMethod.NONE, description="Re-ranking method")
    rerank_weights: Optional[List[float]] = Field(None, description="Weights for weighted re-ranking")
    rerank_k: int = Field(60, gt=0, description="RRF constant for RRF re-ranking")
    
    # Fields for fusion search
    fusion_method: FusionMethod = Field(FusionMethod.RRF, description="Fusion method")
    fusion_weights: Optional[List[float]] = Field(None, description="Weights for fusion")
    
    # Advanced parameters
    params: Dict[str, Any] = Field(default_factory=dict, description="Additional search parameters")
    output_fields: Optional[List[str]] = Field(None, description="Fields to include in results")
    
    @validator('fusion_weights')
    def validate_fusion_weights(cls, v, values):
        """Validate fusion weights if provided"""
        if v is not None:
            if 'fusion_method' in values and values['fusion_method'] == FusionMethod.WEIGHTED:
                if abs(sum(v) - 1.0) > 0.001:
                    raise ValueError(f"Fusion weights must sum to 1.0, got {sum(v)}")
        return v
    
    @validator('vector_weight', 'sparse_weight')
    def validate_hybrid_weights(cls, v, values):
        """Validate hybrid search weights"""
        if 'search_type' in values and values['search_type'] == SearchType.HYBRID:
            # We'll check the sum in a root validator
            pass
        return v
    
    @validator('params')
    def validate_params(cls, v):
        """Ensure params has required fields"""
        if 'nprobe' not in v:
            v['nprobe'] = 10
        if 'ef' not in v:
            v['ef'] = 64
        return v
    
    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"  # Prevent extra fields

