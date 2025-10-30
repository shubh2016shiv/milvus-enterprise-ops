"""
Fallback Module

This module provides graceful degradation logic for handling partial
failures in hybrid search operations.
"""

import logging
from typing import Optional, List, Dict, Any

from ..utils.metrics import SearchStatus

logger = logging.getLogger(__name__)


async def handle_fallback(
    vector_results: Optional[List[Dict[str, Any]]],
    sparse_results: Optional[List[Dict[str, Any]]],
    fallback_to_vector: bool = True
) -> tuple[List[Dict[str, Any]], SearchStatus]:
    """
    Handle fallback logic when one search method fails.
    
    This function implements graceful degradation by returning available
    results when not all search methods succeed. If sparse search fails
    but vector search succeeds, it can fall back to vector-only results.
    
    Args:
        vector_results: Results from vector search (None if failed)
        sparse_results: Results from sparse search (None if failed)
        fallback_to_vector: Whether to allow fallback to vector-only
        
    Returns:
        Tuple of (combined_results, status)
        
    Raises:
        Exception: If both searches failed or fallback is not allowed
    """
    # Both successful - no fallback needed
    if vector_results is not None and sparse_results is not None:
        logger.debug("Both vector and sparse searches successful")
        return (vector_results, sparse_results), SearchStatus.SUCCESS
    
    # Both failed - cannot recover
    if vector_results is None and sparse_results is None:
        logger.error("Both vector and sparse searches failed")
        raise Exception("All search methods failed - cannot provide results")
    
    # Sparse failed, vector succeeded
    if sparse_results is None and vector_results is not None:
        if fallback_to_vector:
            logger.warning(
                "Sparse search failed, falling back to vector-only results "
                f"(count: {len(vector_results)})"
            )
            return (vector_results, None), SearchStatus.DEGRADED
        else:
            logger.error("Sparse search failed and fallback is disabled")
            raise Exception("Sparse search failed and fallback to vector is disabled")
    
    # Vector failed, sparse succeeded (rare case)
    if vector_results is None and sparse_results is not None:
        logger.warning(
            "Vector search failed, using sparse-only results "
            f"(count: {len(sparse_results)})"
        )
        return (None, sparse_results), SearchStatus.DEGRADED
    
    # Should not reach here
    return (vector_results or [], sparse_results or []), SearchStatus.PARTIAL


def should_attempt_fallback(
    error: Exception,
    fallback_enabled: bool
) -> bool:
    """
    Determine if fallback should be attempted for a given error.
    
    Args:
        error: The exception that occurred
        fallback_enabled: Whether fallback is globally enabled
        
    Returns:
        True if fallback should be attempted
    """
    if not fallback_enabled:
        return False
    
    # List of error types that should trigger fallback
    fallback_errors = (
        TimeoutError,
        ConnectionError,
        Exception,  # Generic fallback for most errors
    )
    
    return isinstance(error, fallback_errors)


class FallbackManager:
    """
    Manager for tracking and handling fallback operations.
    
    This class maintains statistics about fallback occurrences and
    provides methods for managing degraded operation states.
    """
    
    def __init__(self, enable_fallback: bool = True):
        """
        Initialize fallback manager.
        
        Args:
            enable_fallback: Whether to enable fallback operations
        """
        self.enable_fallback = enable_fallback
        self.fallback_count = 0
        self.total_operations = 0
        self.degraded_state = False
    
    async def attempt_fallback(
        self,
        vector_results: Optional[List[Dict[str, Any]]],
        sparse_results: Optional[List[Dict[str, Any]]],
        error: Optional[Exception] = None
    ) -> tuple[List[Dict[str, Any]], SearchStatus]:
        """
        Attempt fallback operation.
        
        Args:
            vector_results: Results from vector search
            sparse_results: Results from sparse search
            error: Optional error that triggered fallback
            
        Returns:
            Tuple of (results, status)
        """
        self.total_operations += 1
        
        if not self.enable_fallback:
            logger.error("Fallback is disabled")
            if error:
                raise error
            raise Exception("Fallback is disabled")
        
        try:
            results, status = await handle_fallback(
                vector_results,
                sparse_results,
                self.enable_fallback
            )
            
            if status == SearchStatus.DEGRADED:
                self.fallback_count += 1
                self.degraded_state = True
                logger.warning(
                    f"Operating in degraded mode - "
                    f"fallback rate: {self.get_fallback_rate():.2%}"
                )
            
            return results, status
            
        except Exception as e:
            logger.error(f"Fallback failed: {str(e)}")
            raise
    
    def get_fallback_rate(self) -> float:
        """
        Get the rate of fallback operations.
        
        Returns:
            Fallback rate as a float between 0 and 1
        """
        if self.total_operations == 0:
            return 0.0
        return self.fallback_count / self.total_operations
    
    def is_degraded(self) -> bool:
        """
        Check if system is in degraded state.
        
        Returns:
            True if operating in degraded mode
        """
        return self.degraded_state
    
    def reset_degraded_state(self) -> None:
        """Reset degraded state flag."""
        self.degraded_state = False
        logger.info("Degraded state reset")
    
    def get_stats(self) -> dict:
        """
        Get fallback statistics.
        
        Returns:
            Dictionary with fallback statistics
        """
        return {
            "enable_fallback": self.enable_fallback,
            "fallback_count": self.fallback_count,
            "total_operations": self.total_operations,
            "fallback_rate": self.get_fallback_rate(),
            "degraded_state": self.degraded_state
        }

