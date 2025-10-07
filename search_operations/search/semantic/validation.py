"""
Validation Module

This module provides parameter validation and query sanitization
for semantic search operations, ensuring data integrity and security.
"""

from typing import Optional
from ...core.exceptions import InvalidSearchParametersError
from ...config.semantic import SemanticSearchConfig


class SearchValidator:
    """
    Validates search parameters and sanitizes query inputs.
    
    Features:
    - Parameter validation against Milvus constraints
    - Query sanitization to prevent injection attacks
    - Collection name validation
    - Configuration validation
    """
    
    # Milvus constraints
    MAX_TOP_K = 16384
    MAX_QUERY_LENGTH = 10000
    MIN_TIMEOUT = 0.1
    
    # Valid Milvus metric types
    VALID_METRIC_TYPES = ["L2", "IP", "COSINE", "HAMMING", "JACCARD", "TANIMOTO"]
    
    @staticmethod
    def validate_collection_name(collection_name: str) -> None:
        """
        Validate collection name.
        
        Args:
            collection_name: Collection name to validate
            
        Raises:
            InvalidSearchParametersError: If collection name is invalid
        """
        if not collection_name or not collection_name.strip():
            raise InvalidSearchParametersError("Collection name cannot be empty")
        
        if len(collection_name) > 255:
            raise InvalidSearchParametersError(
                f"Collection name exceeds maximum length (255), got {len(collection_name)}"
            )
        
        # Check for invalid characters
        if not collection_name.replace("_", "").replace("-", "").isalnum():
            raise InvalidSearchParametersError(
                "Collection name can only contain alphanumeric characters, underscores, and hyphens"
            )
    
    @staticmethod
    def validate_query(query: str) -> None:
        """
        Validate query string.
        
        Args:
            query: Query string to validate
            
        Raises:
            InvalidSearchParametersError: If query is invalid
        """
        if not query or not query.strip():
            raise InvalidSearchParametersError("Query cannot be empty")
        
        if len(query) > SearchValidator.MAX_QUERY_LENGTH:
            raise InvalidSearchParametersError(
                f"Query exceeds maximum length ({SearchValidator.MAX_QUERY_LENGTH}), "
                f"got {len(query)}"
            )
    
    @staticmethod
    def validate_top_k(top_k: int) -> None:
        """
        Validate top_k parameter.
        
        Args:
            top_k: Number of results to return
            
        Raises:
            InvalidSearchParametersError: If top_k is invalid
        """
        if top_k <= 0:
            raise InvalidSearchParametersError(f"top_k must be positive, got {top_k}")
        
        if top_k > SearchValidator.MAX_TOP_K:
            raise InvalidSearchParametersError(
                f"top_k exceeds maximum allowed value ({SearchValidator.MAX_TOP_K}), "
                f"got {top_k}"
            )
    
    @staticmethod
    def validate_timeout(timeout: Optional[float]) -> None:
        """
        Validate timeout parameter.
        
        Args:
            timeout: Timeout in seconds
            
        Raises:
            InvalidSearchParametersError: If timeout is invalid
        """
        if timeout is not None:
            if timeout < SearchValidator.MIN_TIMEOUT:
                raise InvalidSearchParametersError(
                    f"timeout must be at least {SearchValidator.MIN_TIMEOUT}s, got {timeout}"
                )
            
            if timeout > 300:  # 5 minutes max
                raise InvalidSearchParametersError(
                    f"timeout exceeds maximum allowed (300s), got {timeout}"
                )
    
    @staticmethod
    def validate_metric_type(metric_type: Optional[str]) -> None:
        """
        Validate metric type.
        
        Args:
            metric_type: Distance metric type
            
        Raises:
            InvalidSearchParametersError: If metric type is invalid
        """
        if metric_type and metric_type.upper() not in SearchValidator.VALID_METRIC_TYPES:
            raise InvalidSearchParametersError(
                f"Invalid metric_type: {metric_type}. "
                f"Must be one of {SearchValidator.VALID_METRIC_TYPES}"
            )
    
    @staticmethod
    def validate_config(config: SemanticSearchConfig) -> None:
        """
        Validate complete search configuration.
        
        Args:
            config: Search configuration to validate
            
        Raises:
            InvalidSearchParametersError: If configuration is invalid
        """
        SearchValidator.validate_top_k(config.top_k)
        SearchValidator.validate_timeout(config.timeout)
        SearchValidator.validate_metric_type(config.metric_type)
        
        # Validate search field
        if not config.search_field or not config.search_field.strip():
            raise InvalidSearchParametersError("search_field cannot be empty")
        
        # Validate output fields if provided
        if config.output_fields is not None:
            if not isinstance(config.output_fields, list):
                raise InvalidSearchParametersError("output_fields must be a list")
            
            for field in config.output_fields:
                if not isinstance(field, str):
                    raise InvalidSearchParametersError(
                        f"output_fields must contain strings, got {type(field)}"
                    )
    
    @staticmethod
    def validate_search_params(
        collection_name: str,
        query: str,
        config: SemanticSearchConfig
    ) -> None:
        """
        Validate all search parameters together.
        
        Args:
            collection_name: Collection name
            query: Query string
            config: Search configuration
            
        Raises:
            InvalidSearchParametersError: If any parameter is invalid
        """
        SearchValidator.validate_collection_name(collection_name)
        SearchValidator.validate_query(query)
        SearchValidator.validate_config(config)


class QuerySanitizer:
    """
    Sanitizes query strings to prevent injection attacks and ensure data integrity.
    
    Features:
    - Control character removal
    - Whitespace normalization
    - Length enforcement
    - Special character handling
    """
    
    @staticmethod
    def sanitize_query(query: str, max_length: int = 10000) -> str:
        """
        Sanitize query string for safe processing.
        
        Args:
            query: Raw query string
            max_length: Maximum allowed query length
            
        Returns:
            Sanitized query string
        """
        # Remove control characters (except newline and tab)
        sanitized = "".join(
            char for char in query
            if ord(char) >= 32 or char in ["\n", "\t"]
        )
        
        # Normalize whitespace
        sanitized = " ".join(sanitized.split())
        
        # Enforce length limit
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized
    
    @staticmethod
    def sanitize_collection_name(collection_name: str) -> str:
        """
        Sanitize collection name.
        
        Args:
            collection_name: Raw collection name
            
        Returns:
            Sanitized collection name
        """
        # Remove whitespace
        sanitized = collection_name.strip()
        
        # Convert to lowercase for consistency
        sanitized = sanitized.lower()
        
        return sanitized
    
    @staticmethod
    def sanitize_field_name(field_name: str) -> str:
        """
        Sanitize field name.
        
        Args:
            field_name: Raw field name
            
        Returns:
            Sanitized field name
        """
        # Remove whitespace and special characters
        sanitized = field_name.strip()
        
        # Replace spaces with underscores
        sanitized = sanitized.replace(" ", "_")
        
        return sanitized

