"""
Validation Module

This module provides parameter validation and query sanitization utilities
for hybrid search operations to ensure data integrity and security.
"""

import logging
from typing import Any
from ....core.search_ops_exceptions import InvalidSearchParametersError
from ....config.hybrid import HybridSearchConfig

logger = logging.getLogger(__name__)


def validate_search_params(
    collection_name: str,
    query: str,
    config: HybridSearchConfig
) -> None:
    """
    Validate search parameters for hybrid search operations.
    
    This function performs comprehensive validation of all search parameters
    to ensure they meet requirements and constraints before search execution.
    
    Args:
        collection_name: Name of the collection to search
        query: Query text
        config: Hybrid search configuration
        
    Raises:
        InvalidSearchParametersError: If any parameter is invalid
    """
    # Validate collection name
    if not collection_name or not collection_name.strip():
        raise InvalidSearchParametersError("Collection name cannot be empty")
    
    # Validate query
    if not query or not query.strip():
        raise InvalidSearchParametersError("Query cannot be empty")
    
    # Validate top_k
    if config.top_k <= 0:
        raise InvalidSearchParametersError(
            f"top_k must be positive, got {config.top_k}"
        )
    
    if config.top_k > 16384:
        raise InvalidSearchParametersError(
            f"top_k exceeds maximum limit (16384), got {config.top_k}"
        )
    
    # Validate timeout
    if config.timeout <= 0:
        raise InvalidSearchParametersError(
            f"timeout must be positive, got {config.timeout}"
        )
    
    # Validate weights
    if config.vector_weight < 0:
        raise InvalidSearchParametersError(
            f"vector_weight must be non-negative, got {config.vector_weight}"
        )
    
    if config.sparse_weight < 0:
        raise InvalidSearchParametersError(
            f"sparse_weight must be non-negative, got {config.sparse_weight}"
        )
    
    # Ensure at least one weight is positive
    total_weight = config.vector_weight + config.sparse_weight
    if total_weight <= 0:
        raise InvalidSearchParametersError(
            "At least one weight (vector_weight or sparse_weight) must be positive"
        )
    
    logger.debug(f"Search parameters validated successfully for collection: {collection_name}")


def sanitize_query(query: str, max_length: int = 10000) -> str:
    """
    Sanitize query string to remove control characters and enforce length limits.
    
    This function cleans the input query by removing control characters,
    normalizing whitespace, and enforcing a maximum length to prevent
    potential issues with extremely long queries.
    
    Args:
        query: Raw query string
        max_length: Maximum allowed query length (default: 10000)
        
    Returns:
        Sanitized query string
        
    Raises:
        InvalidSearchParametersError: If query is empty after sanitization
    """
    if not query:
        raise InvalidSearchParametersError("Query cannot be empty")
    
    # Remove control characters (except newlines which we'll handle separately)
    # Keep characters with ASCII value >= 32 or newlines
    sanitized = "".join(
        char for char in query 
        if ord(char) >= 32 or char == "\n"
    )
    
    # Normalize whitespace (collapse multiple spaces into one)
    sanitized = " ".join(sanitized.split())
    
    # Enforce maximum length
    if len(sanitized) > max_length:
        logger.warning(
            f"Query truncated from {len(sanitized)} to {max_length} characters"
        )
        sanitized = sanitized[:max_length]
    
    # Validate result
    if not sanitized or not sanitized.strip():
        raise InvalidSearchParametersError("Query is empty after sanitization")
    
    return sanitized


def validate_fusion_weights(vector_weight: float, sparse_weight: float) -> None:
    """
    Validate fusion weights for result combination.
    
    Args:
        vector_weight: Weight for vector search results
        sparse_weight: Weight for sparse search results
        
    Raises:
        InvalidSearchParametersError: If weights are invalid
    """
    if vector_weight < 0 or sparse_weight < 0:
        raise InvalidSearchParametersError(
            f"Weights must be non-negative, got vector_weight={vector_weight}, "
            f"sparse_weight={sparse_weight}"
        )
    
    if vector_weight == 0 and sparse_weight == 0:
        raise InvalidSearchParametersError(
            "At least one weight must be positive"
        )


def validate_batch_size(batch_size: int, max_batch_size: int = 100) -> None:
    """
    Validate batch size for batch search operations.
    
    Args:
        batch_size: Requested batch size
        max_batch_size: Maximum allowed batch size
        
    Raises:
        InvalidSearchParametersError: If batch size is invalid
    """
    if batch_size <= 0:
        raise InvalidSearchParametersError(
            f"Batch size must be positive, got {batch_size}"
        )
    
    if batch_size > max_batch_size:
        raise InvalidSearchParametersError(
            f"Batch size ({batch_size}) exceeds maximum ({max_batch_size})"
        )

