"""
Query Optimization Module

This module provides intelligent query optimization for semantic search operations,
including automatic parameter tuning and search strategy selection.
"""

from typing import Dict, Any, Optional
from ...config.semantic import SemanticSearchConfig


class QueryOptimizer:
    """
    Optimizes search queries and parameters for better performance.
    
    Features:
    - Automatic index parameter tuning
    - Search strategy selection
    - Performance-based parameter adjustment
    - Collection-aware optimization
    """
    
    def __init__(self, enable_auto_tuning: bool = True):
        """
        Initialize query optimizer.
        
        Args:
            enable_auto_tuning: Enable automatic parameter tuning
        """
        self.enable_auto_tuning = enable_auto_tuning
    
    def optimize_search_params(
        self,
        config: SemanticSearchConfig,
        collection_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize search parameters based on configuration and collection metadata.
        
        Args:
            config: Search configuration
            collection_info: Optional collection metadata
            
        Returns:
            Optimized search parameters dictionary
        """
        params = config.params.copy() if config.params else {}
        
        if not self.enable_auto_tuning:
            return params
        
        # Optimize based on top_k
        params = self._optimize_for_top_k(params, config.top_k)
        
        # Optimize based on index type if collection info available
        if collection_info:
            params = self._optimize_for_index_type(params, collection_info)
        
        # Optimize based on metric type
        params = self._optimize_for_metric_type(params, config.metric_type)
        
        return params
    
    def _optimize_for_top_k(
        self,
        params: Dict[str, Any],
        top_k: int
    ) -> Dict[str, Any]:
        """
        Optimize parameters based on requested top_k value.
        
        Args:
            params: Current parameters
            top_k: Number of results requested
            
        Returns:
            Optimized parameters
        """
        optimized = params.copy()
        
        # IVF index optimization
        if "nprobe" not in optimized:
            if top_k <= 10:
                optimized["nprobe"] = 16
            elif top_k <= 50:
                optimized["nprobe"] = 32
            elif top_k <= 100:
                optimized["nprobe"] = 64
            else:
                optimized["nprobe"] = 128
        
        # HNSW index optimization
        if "ef" not in optimized:
            # ef should be at least top_k
            optimized["ef"] = max(top_k * 2, 64)
        
        return optimized
    
    def _optimize_for_index_type(
        self,
        params: Dict[str, Any],
        collection_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize parameters based on index type.
        
        Args:
            params: Current parameters
            collection_info: Collection metadata
            
        Returns:
            Optimized parameters
        """
        optimized = params.copy()
        
        index_type = collection_info.get("index_type", "").upper()
        
        if "IVF" in index_type:
            # IVF-specific optimizations
            if "nprobe" not in optimized:
                nlist = collection_info.get("nlist", 2048)
                # nprobe typically 5-10% of nlist for good recall
                optimized["nprobe"] = max(int(nlist * 0.08), 16)
        
        elif "HNSW" in index_type:
            # HNSW-specific optimizations
            if "ef" not in optimized:
                m = collection_info.get("M", 16)
                optimized["ef"] = max(m * 4, 64)
        
        elif "ANNOY" in index_type:
            # ANNOY-specific optimizations
            if "search_k" not in optimized:
                optimized["search_k"] = 100
        
        return optimized
    
    def _optimize_for_metric_type(
        self,
        params: Dict[str, Any],
        metric_type: Optional[str]
    ) -> Dict[str, Any]:
        """
        Optimize parameters based on metric type.
        
        Args:
            params: Current parameters
            metric_type: Distance metric type
            
        Returns:
            Optimized parameters
        """
        optimized = params.copy()
        
        if not metric_type:
            return optimized
        
        metric = metric_type.upper()
        
        # Cosine similarity often benefits from higher precision
        if metric == "COSINE":
            if "ef" in optimized:
                optimized["ef"] = max(optimized["ef"], 128)
            if "nprobe" in optimized:
                optimized["nprobe"] = max(optimized["nprobe"], 32)
        
        return optimized
    
    def suggest_consistency_level(
        self,
        top_k: int,
        timeout: Optional[float]
    ) -> str:
        """
        Suggest appropriate consistency level based on query characteristics.
        
        Args:
            top_k: Number of results requested
            timeout: Query timeout
            
        Returns:
            Suggested consistency level
        """
        # For large top_k or tight timeout, suggest eventual consistency
        if top_k > 100 or (timeout and timeout < 1.0):
            return "Eventually"
        
        # For small top_k, strong consistency is acceptable
        if top_k <= 10:
            return "Strong"
        
        # Default to bounded consistency
        return "Bounded"
    
    def estimate_search_complexity(
        self,
        config: SemanticSearchConfig,
        collection_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Estimate search complexity and provide recommendations.
        
        Args:
            config: Search configuration
            collection_size: Optional collection size
            
        Returns:
            Dictionary with complexity estimation and recommendations
        """
        complexity = {
            "top_k_complexity": "low" if config.top_k <= 10 else "medium" if config.top_k <= 100 else "high",
            "filter_complexity": "low" if not config.expr else "medium",
            "overall_complexity": "medium",
            "recommendations": []
        }
        
        # Add recommendations based on analysis
        if config.top_k > 1000:
            complexity["recommendations"].append(
                "Consider reducing top_k for better performance"
            )
        
        if config.expr and len(config.expr) > 100:
            complexity["recommendations"].append(
                "Complex filter expression may impact performance"
            )
        
        if collection_size and collection_size > 10_000_000:
            complexity["recommendations"].append(
                "Large collection - ensure proper index configuration"
            )
        
        # Determine overall complexity
        if complexity["top_k_complexity"] == "high" or complexity["filter_complexity"] == "medium":
            complexity["overall_complexity"] = "high"
        elif complexity["top_k_complexity"] == "low" and complexity["filter_complexity"] == "low":
            complexity["overall_complexity"] = "low"
        
        return complexity


class SearchParamsBuilder:
    """
    Builds optimized search parameters for Milvus operations.
    
    Provides a fluent interface for constructing search parameters
    with automatic validation and optimization.
    """
    
    def __init__(self, config: SemanticSearchConfig):
        """
        Initialize search params builder.
        
        Args:
            config: Base search configuration
        """
        self.config = config
        self.optimizer = QueryOptimizer()
    
    def build(
        self,
        query_vector: list,
        collection_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build complete search parameters.
        
        Args:
            query_vector: Query embedding vector
            collection_info: Optional collection metadata
            
        Returns:
            Complete search parameters dictionary
        """
        # Optimize parameters
        optimized_params = self.optimizer.optimize_search_params(
            self.config,
            collection_info
        )
        
        # Suggest consistency level
        consistency_level = self.optimizer.suggest_consistency_level(
            self.config.top_k,
            self.config.timeout
        )
        
        # Build complete parameters
        search_params = {
            "data": [query_vector],
            "anns_field": self.config.search_field,
            "param": optimized_params,
            "limit": self.config.top_k,
            "consistency_level": consistency_level
        }
        
        # Add optional parameters
        if self.config.expr:
            search_params["expr"] = self.config.expr
        
        if self.config.output_fields:
            search_params["output_fields"] = self.config.output_fields
        
        return search_params

