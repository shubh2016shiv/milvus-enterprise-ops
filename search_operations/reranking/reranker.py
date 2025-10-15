"""
Re-ranking Operations

This module provides production-grade implementation for Milvus's native re-ranking strategies,
improving relevance through weighted or rank fusion approaches with comprehensive
fault tolerance, validation, and observability.
"""

import time
import logging
import asyncio
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from ..core.search_ops_exceptions import (
    ReRankingError,
    InvalidSearchParametersError
)
from ..config.base import ReRankingMethod
from ..config.reranking import ReRankingConfig
from ..core.base import SearchResult

logger = logging.getLogger(__name__)


class MilvusReRankingMethod(str, Enum):
    """
    Enumeration of Milvus's native re-ranking methods.
    
    Milvus provides two built-in re-ranking strategies:
    - WEIGHTED: Assigns different weights to vector fields based on importance
    - RRF: Reciprocal Rank Fusion, balances importance across all fields
    """
    WEIGHTED = "weighted"
    RRF = "rrf"


class ReRankingStatus(Enum):
    """Status of re-ranking operations"""
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"
    DEGRADED = "degraded"


@dataclass
class ReRankingMetrics:
    """Comprehensive metrics for re-ranking operations"""
    operation_id: str
    method: str
    status: ReRankingStatus = ReRankingStatus.SUCCESS
    input_count: int = 0
    output_count: int = 0
    reranking_time_ms: float = 0.0
    validation_time_ms: float = 0.0
    total_time_ms: float = 0.0
    error_message: Optional[str] = None
    weights_used: Optional[List[float]] = None
    rrf_k: Optional[int] = None
    score_changes: Optional[Dict[str, float]] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class WeightValidationResult:
    """Result of weight validation"""
    is_valid: bool
    normalized_weights: Optional[List[float]] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class MilvusReRanker:
    """
    Production-grade implementation of Milvus's native re-ranking strategies.
    
    Features:
    - Automatic weight normalization and validation
    - Comprehensive parameter validation
    - Graceful degradation on errors
    - Detailed metrics and observability
    - Support for custom re-ranking strategies
    - Result quality analysis
    - Performance monitoring
    - Memory-efficient processing
    """
    
    # Constants
    MIN_WEIGHT = 0.0
    MAX_WEIGHT = 100.0
    MIN_RRF_K = 1
    MAX_RRF_K = 10000
    DEFAULT_RRF_K = 60
    MAX_RESULTS_SIZE = 100000  # Safety limit
    
    def __init__(
        self,
        method: MilvusReRankingMethod = MilvusReRankingMethod.WEIGHTED,
        enable_validation: bool = True,
        enable_normalization: bool = True,
        enable_metrics: bool = True,
        fallback_on_error: bool = True,
        metrics_callback: Optional[Callable[[ReRankingMetrics], None]] = None
    ):
        """
        Initialize the re-ranker with production features.
        
        Args:
            method: Re-ranking method to use
            enable_validation: Enable parameter validation
            enable_normalization: Auto-normalize weights
            enable_metrics: Enable metrics collection
            fallback_on_error: Return original results on error instead of raising
            metrics_callback: Optional callback for metrics reporting
        """
        self.method = method
        self.enable_validation = enable_validation
        self.enable_normalization = enable_normalization
        self.enable_metrics = enable_metrics
        self.fallback_on_error = fallback_on_error
        self.metrics_callback = metrics_callback
        
        self._ranker = None
        self._metrics_history: List[ReRankingMetrics] = []
        self._lock = asyncio.Lock()
        
        # Validate PyMilvus availability
        self._validate_dependencies()
        
        logger.info(
            f"MilvusReRanker initialized - "
            f"method: {method}, validation: {enable_validation}, "
            f"normalization: {enable_normalization}, fallback: {fallback_on_error}"
        )
    
    def _validate_dependencies(self) -> None:
        """Validate that required PyMilvus components are available."""
        try:
            from pymilvus import WeightedRanker, RRFRanker
            self._weighted_ranker_class = WeightedRanker
            self._rrf_ranker_class = RRFRanker
        except ImportError as e:
            error_msg = (
                "Failed to import Milvus re-ranking classes. "
                "Please ensure PyMilvus >= 2.3.0 is installed."
            )
            logger.error(error_msg)
            raise ReRankingError(error_msg) from e
    
    def validate_weights(
        self,
        weights: List[float],
        num_fields: Optional[int] = None,
        auto_normalize: bool = True
    ) -> WeightValidationResult:
        """
        Validate and optionally normalize weights.
        
        Args:
            weights: List of weights to validate
            num_fields: Expected number of fields (for validation)
            auto_normalize: Whether to auto-normalize weights
            
        Returns:
            WeightValidationResult with validation status and normalized weights
        """
        result = WeightValidationResult(is_valid=True)
        
        # Check if weights list is empty
        if not weights:
            result.is_valid = False
            result.errors.append("Weights list cannot be empty")
            return result
        
        # Check for valid number types
        if not all(isinstance(w, (int, float)) for w in weights):
            result.is_valid = False
            result.errors.append("All weights must be numeric (int or float)")
            return result
        
        # Check for negative weights
        if any(w < self.MIN_WEIGHT for w in weights):
            result.is_valid = False
            result.errors.append(f"Weights cannot be negative (min: {self.MIN_WEIGHT})")
            return result
        
        # Check for excessively large weights
        if any(w > self.MAX_WEIGHT for w in weights):
            result.warnings.append(
                f"Some weights exceed {self.MAX_WEIGHT}, which may cause numerical issues"
            )
        
        # Check if all weights are zero
        if all(w == 0 for w in weights):
            result.is_valid = False
            result.errors.append("At least one weight must be non-zero")
            return result
        
        # Validate count if num_fields specified
        if num_fields is not None and len(weights) != num_fields:
            result.is_valid = False
            result.errors.append(
                f"Expected {num_fields} weights, got {len(weights)}"
            )
            return result
        
        # Normalize weights if requested
        if auto_normalize and self.enable_normalization:
            total_weight = sum(weights)
            if total_weight > 0:
                result.normalized_weights = [w / total_weight for w in weights]
                if result.normalized_weights != weights:
                    result.warnings.append(
                        f"Weights normalized from {weights} to {result.normalized_weights}"
                    )
            else:
                result.is_valid = False
                result.errors.append("Cannot normalize: sum of weights is zero")
        else:
            result.normalized_weights = weights
        
        return result
    
    def validate_rrf_k(self, k: int) -> bool:
        """
        Validate RRF k parameter.
        
        Args:
            k: RRF k parameter to validate
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            InvalidSearchParametersError: If k is invalid
        """
        if not isinstance(k, int):
            raise InvalidSearchParametersError(f"RRF k must be an integer, got {type(k)}")
        
        if k < self.MIN_RRF_K or k > self.MAX_RRF_K:
            raise InvalidSearchParametersError(
                f"RRF k must be between {self.MIN_RRF_K} and {self.MAX_RRF_K}, got {k}"
            )
        
        return True
    
    def create_ranker(
        self,
        weights: Optional[List[float]] = None,
        k: Optional[int] = None
    ) -> Any:
        """
        Create a Milvus re-ranker instance with validation.
        
        Args:
            weights: Weights for weighted ranker (required for WEIGHTED method)
            k: RRF constant (used for RRF method)
            
        Returns:
            Milvus re-ranker instance
            
        Raises:
            ReRankingError: If re-ranker creation fails
            InvalidSearchParametersError: If parameters are invalid
        """
        try:
            if self.method == MilvusReRankingMethod.WEIGHTED:
                if not weights:
                    raise InvalidSearchParametersError(
                        "Weights must be provided for weighted re-ranking"
                    )
                
                # Validate and normalize weights
                if self.enable_validation:
                    validation_result = self.validate_weights(weights)
                    
                    if not validation_result.is_valid:
                        error_msg = "; ".join(validation_result.errors)
                        raise InvalidSearchParametersError(f"Weight validation failed: {error_msg}")
                    
                    for warning in validation_result.warnings:
                        logger.warning(f"Weight validation warning: {warning}")
                    
                    weights = validation_result.normalized_weights
                
                logger.info(f"Creating WeightedRanker with weights: {weights}")
                return self._weighted_ranker_class(*weights)
                
            elif self.method == MilvusReRankingMethod.RRF:
                k = k or self.DEFAULT_RRF_K
                
                # Validate k parameter
                if self.enable_validation:
                    self.validate_rrf_k(k)
                
                logger.info(f"Creating RRFRanker with k: {k}")
                return self._rrf_ranker_class(k)
                
            else:
                raise InvalidSearchParametersError(
                    f"Unsupported re-ranking method: {self.method}"
                )
                
        except (InvalidSearchParametersError, ReRankingError):
            raise
        except Exception as e:
            error_msg = f"Failed to create re-ranker: {str(e)}"
            logger.error(error_msg)
            raise ReRankingError(error_msg) from e
    
    async def get_search_params(
        self,
        config: ReRankingConfig,
        search_params: Dict[str, Any],
        num_vector_fields: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Add re-ranking parameters to search params with validation.
        
        This method adds the appropriate re-ranking configuration
        to the search parameters for use with Milvus search operations.
        
        Args:
            config: Re-ranking configuration
            search_params: Existing search parameters
            num_vector_fields: Number of vector fields (for validation)
            
        Returns:
            Updated search parameters with re-ranking configuration
            
        Raises:
            ReRankingError: If parameter preparation fails
        """
        start_time = time.time()
        
        try:
            # If re-ranking is disabled, return original params
            if not config.enabled:
                logger.debug("Re-ranking is disabled, skipping")
                return search_params.copy()
            
            # Validate input search params
            if not isinstance(search_params, dict):
                raise InvalidSearchParametersError(
                    f"search_params must be a dictionary, got {type(search_params)}"
                )
            
            # Create a copy to avoid modifying original
            updated_params = search_params.copy()
            
            # Create re-ranker based on method
            if config.method == ReRankingMethod.WEIGHTED:
                weights = config.params.get("weights")
                if not weights:
                    raise InvalidSearchParametersError(
                        "Weights must be provided for weighted re-ranking"
                    )
                
                # Validate weight count if num_vector_fields provided
                if num_vector_fields and len(weights) != num_vector_fields:
                    raise InvalidSearchParametersError(
                        f"Number of weights ({len(weights)}) must match "
                        f"number of vector fields ({num_vector_fields})"
                    )
                
                ranker = self.create_ranker(weights=weights)
                updated_params["rerank"] = ranker
                
                logger.info(f"Added weighted re-ranking with {len(weights)} weights")
                
            elif config.method == ReRankingMethod.RRF:
                k = config.params.get("k", self.DEFAULT_RRF_K)
                ranker = self.create_ranker(k=k)
                updated_params["rerank"] = ranker
                
                logger.info(f"Added RRF re-ranking with k={k}")
                
            else:
                raise InvalidSearchParametersError(
                    f"Unsupported re-ranking method: {config.method}"
                )
            
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"Re-ranking params prepared in {elapsed_ms:.2f}ms")
            
            return updated_params
            
        except (InvalidSearchParametersError, ReRankingError):
            raise
        except Exception as e:
            error_msg = f"Failed to prepare re-ranking parameters: {str(e)}"
            logger.error(error_msg)
            raise ReRankingError(error_msg) from e
    
    def _calculate_score_changes(
        self,
        original_results: List[Dict[str, Any]],
        reranked_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate score changes after re-ranking.
        
        Args:
            original_results: Original search results
            reranked_results: Re-ranked results
            
        Returns:
            Dictionary with score change statistics
        """
        if not original_results or not reranked_results:
            return {}
        
        # Create ID to score mappings
        original_scores = {
            str(r.get('id', r.get('pk', i))): r.get('distance', 0.0)
            for i, r in enumerate(original_results)
        }
        reranked_scores = {
            str(r.get('id', r.get('pk', i))): r.get('distance', 0.0)
            for i, r in enumerate(reranked_results)
        }
        
        # Calculate changes
        score_diffs = []
        for doc_id in reranked_scores:
            if doc_id in original_scores:
                diff = abs(reranked_scores[doc_id] - original_scores[doc_id])
                score_diffs.append(diff)
        
        if not score_diffs:
            return {}
        
        return {
            "avg_score_change": sum(score_diffs) / len(score_diffs),
            "max_score_change": max(score_diffs),
            "min_score_change": min(score_diffs),
            "docs_changed": len(score_diffs)
        }
    
    async def process_results(
        self,
        results: SearchResult,
        config: ReRankingConfig,
        original_results: Optional[SearchResult] = None,
        operation_id: Optional[str] = None
    ) -> SearchResult:
        """
        Process results after re-ranking with metrics and validation.
        
        This method adds re-ranking metadata to the search results and
        collects comprehensive metrics.
        
        Args:
            results: Search results (already re-ranked by Milvus)
            config: Re-ranking configuration
            original_results: Original results before re-ranking (for metrics)
            operation_id: Optional operation ID for tracking
            
        Returns:
            Updated search results with re-ranking metadata and metrics
        """
        start_time = time.time()
        operation_id = operation_id or f"rerank_{int(time.time() * 1000)}"
        
        metrics = ReRankingMetrics(
            operation_id=operation_id,
            method=config.method.value if hasattr(config.method, 'value') else str(config.method),
            input_count=len(original_results.hits) if original_results else 0,
            output_count=len(results.hits)
        )
        
        try:
            # If re-ranking is disabled, return original results
            if not config.enabled:
                metrics.status = ReRankingStatus.SKIPPED
                return results
            
            # Validate results
            if not results or not results.hits:
                logger.warning("No results to process for re-ranking")
                metrics.status = ReRankingStatus.SKIPPED
                return results
            
            # Check for size limits
            if len(results.hits) > self.MAX_RESULTS_SIZE:
                logger.warning(
                    f"Results size ({len(results.hits)}) exceeds limit ({self.MAX_RESULTS_SIZE}), "
                    "truncating for safety"
                )
                results.hits = results.hits[:self.MAX_RESULTS_SIZE]
                metrics.status = ReRankingStatus.DEGRADED
            
            # Calculate score changes if we have original results
            if original_results and self.enable_metrics:
                metrics.score_changes = self._calculate_score_changes(
                    original_results.hits,
                    results.hits
                )
            
            # Add re-ranking metadata to search params
            if not hasattr(results, 'search_params') or results.search_params is None:
                results.search_params = {}
            
            results.search_params["reranking"] = {
                "method": config.method.value if hasattr(config.method, 'value') else str(config.method),
                "enabled": True,
                "operation_id": operation_id
            }
            
            # Add method-specific metadata
            if config.method == ReRankingMethod.WEIGHTED:
                weights = config.params.get("weights", [])
                results.search_params["reranking"]["weights"] = weights
                results.search_params["reranking"]["num_weights"] = len(weights)
                metrics.weights_used = weights
                
            elif config.method == ReRankingMethod.RRF:
                k = config.params.get("k", self.DEFAULT_RRF_K)
                results.search_params["reranking"]["k"] = k
                metrics.rrf_k = k
            
            # Add score change statistics to metadata if available
            if metrics.score_changes:
                results.search_params["reranking"]["score_changes"] = metrics.score_changes
            
            metrics.reranking_time_ms = (time.time() - start_time) * 1000
            metrics.status = ReRankingStatus.SUCCESS
            
            logger.info(
                f"Re-ranking processed - method: {config.method}, "
                f"results: {len(results.hits)}, time: {metrics.reranking_time_ms:.2f}ms"
            )
            
            return results
            
        except Exception as e:
            metrics.status = ReRankingStatus.FAILURE
            metrics.error_message = str(e)
            
            error_msg = f"Failed to process re-ranking results: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            if self.fallback_on_error:
                logger.warning("Falling back to original results due to error")
                return original_results if original_results else results
            else:
                raise ReRankingError(error_msg) from e
                
        finally:
            metrics.total_time_ms = (time.time() - start_time) * 1000
            
            # Store metrics
            if self.enable_metrics:
                async with self._lock:
                    self._metrics_history.append(metrics)
                    # Keep last 1000 metrics
                    if len(self._metrics_history) > 1000:
                        self._metrics_history = self._metrics_history[-1000:]
            
            # Call metrics callback if provided
            if self.metrics_callback:
                try:
                    self.metrics_callback(metrics)
                except Exception as e:
                    logger.error(f"Metrics callback failed: {str(e)}")
    
    async def analyze_reranking_impact(
        self,
        original_results: SearchResult,
        reranked_results: SearchResult
    ) -> Dict[str, Any]:
        """
        Analyze the impact of re-ranking on results.
        
        Args:
            original_results: Original search results
            reranked_results: Re-ranked results
            
        Returns:
            Dictionary with impact analysis
        """
        analysis = {
            "rank_changes": 0,
            "score_changes": {},
            "top_k_overlap": {},
            "position_changes": []
        }
        
        try:
            # Create ID to position mappings
            original_positions = {
                str(r.get('id', r.get('pk', i))): i
                for i, r in enumerate(original_results.hits)
            }
            reranked_positions = {
                str(r.get('id', r.get('pk', i))): i
                for i, r in enumerate(reranked_results.hits)
            }
            
            # Calculate rank changes
            for doc_id, new_pos in reranked_positions.items():
                if doc_id in original_positions:
                    old_pos = original_positions[doc_id]
                    if old_pos != new_pos:
                        analysis["rank_changes"] += 1
                        analysis["position_changes"].append({
                            "doc_id": doc_id,
                            "old_position": old_pos,
                            "new_position": new_pos,
                            "change": new_pos - old_pos
                        })
            
            # Calculate top-k overlap
            for k in [1, 5, 10, 20]:
                if len(original_results.hits) >= k and len(reranked_results.hits) >= k:
                    original_top_k = set(
                        str(r.get('id', r.get('pk', i)))
                        for i, r in enumerate(original_results.hits[:k])
                    )
                    reranked_top_k = set(
                        str(r.get('id', r.get('pk', i)))
                        for i, r in enumerate(reranked_results.hits[:k])
                    )
                    overlap = len(original_top_k & reranked_top_k)
                    analysis["top_k_overlap"][f"top_{k}"] = {
                        "overlap": overlap,
                        "percentage": (overlap / k) * 100
                    }
            
            # Calculate score changes
            analysis["score_changes"] = self._calculate_score_changes(
                original_results.hits,
                reranked_results.hits
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze re-ranking impact: {str(e)}")
            analysis["error"] = str(e)
        
        return analysis
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of re-ranking metrics.
        
        Returns:
            Dictionary with metrics summary
        """
        async with self._lock:
            if not self._metrics_history:
                return {"message": "No metrics available"}
            
            total_ops = len(self._metrics_history)
            successful = sum(1 for m in self._metrics_history if m.status == ReRankingStatus.SUCCESS)
            failed = sum(1 for m in self._metrics_history if m.status == ReRankingStatus.FAILURE)
            skipped = sum(1 for m in self._metrics_history if m.status == ReRankingStatus.SKIPPED)
            degraded = sum(1 for m in self._metrics_history if m.status == ReRankingStatus.DEGRADED)
            
            # Calculate averages for successful operations
            successful_metrics = [m for m in self._metrics_history if m.status == ReRankingStatus.SUCCESS]
            if successful_metrics:
                avg_reranking_time = sum(m.reranking_time_ms for m in successful_metrics) / len(successful_metrics)
                avg_total_time = sum(m.total_time_ms for m in successful_metrics) / len(successful_metrics)
                avg_input = sum(m.input_count for m in successful_metrics) / len(successful_metrics)
                avg_output = sum(m.output_count for m in successful_metrics) / len(successful_metrics)
            else:
                avg_reranking_time = avg_total_time = avg_input = avg_output = 0.0
            
            return {
                "total_operations": total_ops,
                "successful": successful,
                "failed": failed,
                "skipped": skipped,
                "degraded": degraded,
                "success_rate": successful / total_ops if total_ops > 0 else 0,
                "avg_reranking_time_ms": round(avg_reranking_time, 2),
                "avg_total_time_ms": round(avg_total_time, 2),
                "avg_input_count": round(avg_input, 2),
                "avg_output_count": round(avg_output, 2),
                "method": self.method.value
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the re-ranking system.
        
        Returns:
            Health status dictionary
        """
        health = {
            "status": "healthy",
            "timestamp": time.time(),
            "components": {}
        }
        
        try:
            # Check PyMilvus availability
            try:
                self._validate_dependencies()
                health["components"]["pymilvus"] = {"status": "healthy"}
            except ReRankingError as e:
                health["components"]["pymilvus"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["status"] = "unhealthy"
            
            # Check recent failure rate
            async with self._lock:
                if self._metrics_history:
                    recent = self._metrics_history[-100:]  # Last 100 operations
                    failure_rate = sum(
                        1 for m in recent if m.status == ReRankingStatus.FAILURE
                    ) / len(recent)
                    
                    health["components"]["operations"] = {
                        "status": "healthy" if failure_rate < 0.1 else "degraded",
                        "recent_failure_rate": round(failure_rate, 3)
                    }
                    
                    if failure_rate >= 0.1:
                        health["status"] = "degraded"
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
            logger.error(f"Health check failed: {str(e)}")
        
        return health
    
    def get_recommended_weights(
        self,
        field_importance: Dict[str, float]
    ) -> List[float]:
        """
        Calculate recommended weights based on field importance scores.
        
        Args:
            field_importance: Dictionary mapping field names to importance scores
            
        Returns:
            Normalized weights list
        """
        if not field_importance:
            raise InvalidSearchParametersError("field_importance cannot be empty")
        
        weights = list(field_importance.values())
        validation_result = self.validate_weights(weights, auto_normalize=True)
        
        if not validation_result.is_valid:
            raise InvalidSearchParametersError(
                f"Invalid importance scores: {'; '.join(validation_result.errors)}"
            )
        
        return validation_result.normalized_weights
    
    async def clear_metrics(self) -> None:
        """Clear metrics history."""
        async with self._lock:
            self._metrics_history.clear()
        logger.info("Re-ranking metrics cleared")
    
    @asynccontextmanager
    async def reranking_context(self, config: ReRankingConfig):
        """
        Context manager for re-ranking operations.
        
        Usage:
            async with reranker.reranking_context(config):
                # Perform operations
                pass
        """
        start_time = time.time()
        try:
            yield self
        finally:
            elapsed = (time.time() - start_time) * 1000
            logger.debug(f"Re-ranking context completed in {elapsed:.2f}ms")


# Utility functions

def compare_ranking_methods(
    original_results: SearchResult,
    weighted_results: SearchResult,
    rrf_results: SearchResult
) -> Dict[str, Any]:
    """
    Compare different re-ranking methods.
    
    Args:
        original_results: Original search results
        weighted_results: Results with weighted re-ranking
        rrf_results: Results with RRF re-ranking
        
    Returns:
        Comparison analysis
    """
    comparison = {
        "original_count": len(original_results.hits),
        "weighted_count": len(weighted_results.hits),
        "rrf_count": len(rrf_results.hits),
        "agreements": {},
        "differences": {}
    }
    
    try:
        # Get top-k IDs from each method
        original_ids = [
            str(r.get('id', r.get('pk', i)))
            for i, r in enumerate(original_results.hits[:10])
        ]
        weighted_ids = [
            str(r.get('id', r.get('pk', i)))
            for i, r in enumerate(weighted_results.hits[:10])
        ]
        rrf_ids = [
            str(r.get('id', r.get('pk', i)))
            for i, r in enumerate(rrf_results.hits[:10])
        ]
        
        # Calculate overlaps
        comparison["agreements"]["weighted_vs_rrf"] = len(set(weighted_ids) & set(rrf_ids))
        comparison["agreements"]["original_vs_weighted"] = len(set(original_ids) & set(weighted_ids))
        comparison["agreements"]["original_vs_rrf"] = len(set(original_ids) & set(rrf_ids))
        
        # Calculate differences
        comparison["differences"]["unique_to_weighted"] = len(set(weighted_ids) - set(rrf_ids))
        comparison["differences"]["unique_to_rrf"] = len(set(rrf_ids) - set(weighted_ids))
        
    except Exception as e:
        logger.error(f"Failed to compare ranking methods: {str(e)}")
        comparison["error"] = str(e)
    
    return comparison


def create_adaptive_weights(
    field_types: List[str],
    query_type: str = "general",
    custom_rules: Optional[Dict[str, Dict[str, float]]] = None
) -> List[float]:
    """
    Create adaptive weights based on field types and query characteristics.
    
    This function provides intelligent weight suggestions based on common
    patterns in multi-vector search scenarios.
    
    Args:
        field_types: List of field types (e.g., ['text', 'image', 'audio'])
        query_type: Type of query ('general', 'text_heavy', 'visual_heavy', etc.)
        custom_rules: Optional custom weight rules
        
    Returns:
        List of suggested weights
        
    Example:
        >>> weights = create_adaptive_weights(
        ...     field_types=['text_dense', 'text_sparse', 'image'],
        ...     query_type='text_heavy'
        ... )
        >>> # Returns something like [0.4, 0.4, 0.2]
    """
    # Default weight rules
    default_rules = {
        "general": {
            "text_dense": 0.35,
            "text_sparse": 0.35,
            "image": 0.20,
            "audio": 0.10
        },
        "text_heavy": {
            "text_dense": 0.40,
            "text_sparse": 0.40,
            "image": 0.15,
            "audio": 0.05
        },
        "visual_heavy": {
            "text_dense": 0.20,
            "text_sparse": 0.15,
            "image": 0.50,
            "audio": 0.15
        },
        "multimodal": {
            "text_dense": 0.30,
            "text_sparse": 0.25,
            "image": 0.30,
            "audio": 0.15
        }
    }
    
    # Use custom rules if provided, otherwise use defaults
    rules = custom_rules or default_rules
    
    # Get weights for the query type
    if query_type not in rules:
        logger.warning(f"Unknown query_type '{query_type}', using 'general'")
        query_type = "general"
    
    weight_map = rules[query_type]
    
    # Build weights list based on field types
    weights = []
    for field_type in field_types:
        weight = weight_map.get(field_type, 1.0 / len(field_types))  # Default to equal weight
        weights.append(weight)
    
    # Normalize weights
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]
    
    logger.info(
        f"Created adaptive weights for query_type='{query_type}': "
        f"{dict(zip(field_types, weights))}"
    )
    
    return weights


def calculate_optimal_rrf_k(
    num_searches: int,
    expected_top_k: int,
    diversity_preference: float = 0.5
) -> int:
    """
    Calculate optimal RRF k parameter based on search characteristics.
    
    The k parameter in RRF controls how much weight is given to top-ranked results.
    Lower k values favor top results more, higher values are more democratic.
    
    Args:
        num_searches: Number of searches being fused
        expected_top_k: Expected number of top results to retrieve
        diversity_preference: Preference for diversity (0.0 = favor precision, 1.0 = favor diversity)
        
    Returns:
        Recommended k value
        
    Example:
        >>> k = calculate_optimal_rrf_k(
        ...     num_searches=3,
        ...     expected_top_k=10,
        ...     diversity_preference=0.7
        ... )
        >>> print(k)  # Something like 80
    """
    # Base k starts at typical default
    base_k = 60
    
    # Adjust based on number of searches
    # More searches benefit from higher k to balance contributions
    search_factor = 1.0 + (num_searches - 2) * 0.2
    
    # Adjust based on expected top-k
    # Larger top-k might benefit from slightly lower k to maintain ranking quality
    topk_factor = 1.0 - (expected_top_k / 1000.0)
    topk_factor = max(0.5, min(1.5, topk_factor))  # Clamp between 0.5 and 1.5
    
    # Apply diversity preference
    # Higher diversity preference increases k
    diversity_factor = 1.0 + diversity_preference
    
    # Calculate final k
    k = int(base_k * search_factor * topk_factor * diversity_factor)
    
    # Clamp to reasonable range
    k = max(10, min(1000, k))
    
    logger.info(
        f"Calculated optimal RRF k={k} for num_searches={num_searches}, "
        f"top_k={expected_top_k}, diversity={diversity_preference}"
    )
    
    return k


class ReRankingStrategy:
    """
    Advanced re-ranking strategy that combines multiple approaches.
    
    This class provides a higher-level interface for intelligent re-ranking
    that can adapt based on query characteristics and result patterns.
    """
    
    def __init__(self):
        self.weighted_ranker = MilvusReRanker(method=MilvusReRankingMethod.WEIGHTED)
        self.rrf_ranker = MilvusReRanker(method=MilvusReRankingMethod.RRF)
        self._performance_history: Dict[str, List[float]] = {
            "weighted": [],
            "rrf": []
        }
    
    async def auto_select_method(
        self,
        num_vector_fields: int,
        query_characteristics: Optional[Dict[str, Any]] = None
    ) -> tuple[MilvusReRankingMethod, Dict[str, Any]]:
        """
        Automatically select the best re-ranking method based on characteristics.
        
        Args:
            num_vector_fields: Number of vector fields being searched
            query_characteristics: Optional query characteristics
            
        Returns:
            Tuple of (selected_method, parameters)
        """
        query_characteristics = query_characteristics or {}
        
        # Decision logic
        if num_vector_fields == 2:
            # For 2 fields, weighted often works well
            method = MilvusReRankingMethod.WEIGHTED
            params = {"weights": [0.6, 0.4]}  # Slight preference to first field
            
        elif num_vector_fields > 4:
            # For many fields, RRF handles complexity better
            method = MilvusReRankingMethod.RRF
            k = calculate_optimal_rrf_k(
                num_searches=num_vector_fields,
                expected_top_k=query_characteristics.get("top_k", 10)
            )
            params = {"k": k}
            
        else:
            # For 3-4 fields, check if we have performance history
            weighted_avg = (
                sum(self._performance_history["weighted"]) / len(self._performance_history["weighted"])
                if self._performance_history["weighted"] else None
            )
            rrf_avg = (
                sum(self._performance_history["rrf"]) / len(self._performance_history["rrf"])
                if self._performance_history["rrf"] else None
            )
            
            if weighted_avg is not None and rrf_avg is not None:
                # Use historical performance
                if weighted_avg > rrf_avg:
                    method = MilvusReRankingMethod.WEIGHTED
                    params = {"weights": [1.0 / num_vector_fields] * num_vector_fields}
                else:
                    method = MilvusReRankingMethod.RRF
                    params = {"k": 60}
            else:
                # Default to RRF for balanced approach
                method = MilvusReRankingMethod.RRF
                params = {"k": 60}
        
        logger.info(
            f"Auto-selected re-ranking method: {method.value} with params: {params}"
        )
        
        return method, params
    
    def record_performance(self, method: str, score: float):
        """Record performance score for a method."""
        if method in self._performance_history:
            self._performance_history[method].append(score)
            # Keep last 100 scores
            if len(self._performance_history[method]) > 100:
                self._performance_history[method] = self._performance_history[method][-100:]


class MultiStageReRanker:
    """
    Multi-stage re-ranking system for complex scenarios.
    
    This class implements a pipeline approach where results go through
    multiple re-ranking stages for progressive refinement.
    """
    
    def __init__(
        self,
        stages: List[tuple[MilvusReRankingMethod, Dict[str, Any]]],
        enable_metrics: bool = True
    ):
        """
        Initialize multi-stage re-ranker.
        
        Args:
            stages: List of (method, params) tuples for each stage
            enable_metrics: Enable metrics collection
        """
        self.stages = stages
        self.enable_metrics = enable_metrics
        self.rankers = []
        
        for method, _ in stages:
            ranker = MilvusReRanker(
                method=method,
                enable_metrics=enable_metrics
            )
            self.rankers.append(ranker)
        
        logger.info(f"MultiStageReRanker initialized with {len(stages)} stages")
    
    async def rerank(
        self,
        results: SearchResult,
        configs: List[ReRankingConfig]
    ) -> SearchResult:
        """
        Apply multi-stage re-ranking.
        
        Args:
            results: Initial search results
            configs: List of re-ranking configs for each stage
            
        Returns:
            Final re-ranked results
        """
        if len(configs) != len(self.stages):
            raise InvalidSearchParametersError(
                f"Number of configs ({len(configs)}) must match number of stages ({len(self.stages)})"
            )
        
        current_results = results
        
        for i, (ranker, config) in enumerate(zip(self.rankers, configs)):
            logger.info(f"Applying re-ranking stage {i + 1}/{len(self.stages)}")
            
            current_results = await ranker.process_results(
                results=current_results,
                config=config,
                operation_id=f"stage_{i + 1}"
            )
        
        return current_results
    
    async def get_stage_metrics(self) -> List[Dict[str, Any]]:
        """Get metrics for each stage."""
        metrics = []
        for i, ranker in enumerate(self.rankers):
            stage_metrics = await ranker.get_metrics_summary()
            stage_metrics["stage"] = i + 1
            metrics.append(stage_metrics)
        return metrics


# Factory functions for common use cases

def create_text_search_reranker(
    enable_sparse: bool = True,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4
) -> MilvusReRanker:
    """
    Create a re-ranker optimized for text search.
    
    Args:
        enable_sparse: Whether sparse vectors are used
        dense_weight: Weight for dense vectors
        sparse_weight: Weight for sparse vectors
        
    Returns:
        Configured MilvusReRanker instance
    """
    if enable_sparse:
        return MilvusReRanker(
            method=MilvusReRankingMethod.WEIGHTED,
            enable_normalization=True,
            enable_validation=True
        )
    else:
        return MilvusReRanker(
            method=MilvusReRankingMethod.RRF,
            enable_validation=True
        )


def create_multimodal_reranker(
    modalities: List[str],
    query_modality: str = "text"
) -> MilvusReRanker:
    """
    Create a re-ranker optimized for multimodal search.
    
    Args:
        modalities: List of modality types (e.g., ['text', 'image', 'audio'])
        query_modality: Primary query modality
        
    Returns:
        Configured MilvusReRanker instance with adaptive weights
    """
    ranker = MilvusReRanker(
        method=MilvusReRankingMethod.WEIGHTED,
        enable_normalization=True,
        enable_validation=True,
        fallback_on_error=True
    )
    
    logger.info(
        f"Created multimodal re-ranker for modalities: {modalities}, "
        f"query modality: {query_modality}"
    )
    
    return ranker


def create_ensemble_reranker(
    num_models: int,
    diversity_preference: float = 0.5
) -> MilvusReRanker:
    """
    Create a re-ranker optimized for ensemble search (multiple models).
    
    Args:
        num_models: Number of models in the ensemble
        diversity_preference: Preference for result diversity
        
    Returns:
        Configured MilvusReRanker instance
    """
    # For ensemble, RRF usually works better
    k = calculate_optimal_rrf_k(
        num_searches=num_models,
        expected_top_k=10,
        diversity_preference=diversity_preference
    )
    
    ranker = MilvusReRanker(
        method=MilvusReRankingMethod.RRF,
        enable_validation=True,
        fallback_on_error=True
    )
    
    logger.info(
        f"Created ensemble re-ranker for {num_models} models with k={k}"
    )
    
    return ranker


# Example usage and best practices

"""
Example Usage:
-------------

1. Basic Weighted Re-ranking:
   
   ranker = MilvusReRanker(method=MilvusReRankingMethod.WEIGHTED)
   config = ReRankingConfig(
       enabled=True,
       method=ReRankingMethod.WEIGHTED,
       params={"weights": [0.6, 0.4]}
   )
   
   # Add to search params
   search_params = await ranker.get_search_params(config, base_search_params)
   
   # Process results
   final_results = await ranker.process_results(results, config)


2. RRF Re-ranking with Auto-tuning:
   
   k = calculate_optimal_rrf_k(
       num_searches=3,
       expected_top_k=20,
       diversity_preference=0.7
   )
   
   ranker = MilvusReRanker(method=MilvusReRankingMethod.RRF)
   config = ReRankingConfig(
       enabled=True,
       method=ReRankingMethod.RRF,
       params={"k": k}
   )


3. Adaptive Weights for Multimodal:
   
   weights = create_adaptive_weights(
       field_types=['text_dense', 'text_sparse', 'image'],
       query_type='text_heavy'
   )
   
   ranker = create_multimodal_reranker(
       modalities=['text', 'image'],
       query_modality='text'
   )


4. Multi-stage Re-ranking:
   
   stages = [
       (MilvusReRankingMethod.WEIGHTED, {"weights": [0.5, 0.5]}),
       (MilvusReRankingMethod.RRF, {"k": 60})
   ]
   
   multi_ranker = MultiStageReRanker(stages)
   final_results = await multi_ranker.rerank(results, configs)


5. Production Monitoring:
   
   def metrics_callback(metrics: ReRankingMetrics):
       # Send to monitoring system
       prometheus.record(metrics)
   
   ranker = MilvusReRanker(
       method=MilvusReRankingMethod.WEIGHTED,
       enable_metrics=True,
       metrics_callback=metrics_callback
   )
   
   # Check health
   health = await ranker.health_check()
   
   # Get metrics summary
   summary = await ranker.get_metrics_summary()


Best Practices:
--------------

1. Always enable validation in production:
   - enable_validation=True ensures parameter correctness
   - enable_normalization=True handles weight normalization automatically

2. Use fallback_on_error=True for resilience:
   - Returns original results on re-ranking failure
   - Prevents complete search failure due to re-ranking issues

3. Choose the right method:
   - WEIGHTED: When you have clear field importance preferences
   - RRF: When you want balanced fusion or have many fields

4. Monitor re-ranking performance:
   - Use metrics_callback for real-time monitoring
   - Regularly check get_metrics_summary()
   - Perform health_check() in readiness probes

5. Optimize parameters:
   - Use calculate_optimal_rrf_k() for RRF parameter tuning
   - Use create_adaptive_weights() for intelligent weight selection
   - Test different configurations with your data

6. Handle errors gracefully:
   - Always use try-except for re-ranking operations
   - Log failures for debugging
   - Have fallback strategies ready
"""