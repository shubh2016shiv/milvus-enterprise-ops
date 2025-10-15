"""
Data Management Operations Configuration

Centralized configuration for data management operations, providing
a single source of truth for all tunable parameters related to batching,
timeouts, retries, and validation.

This configuration can be customized by external projects to match their
specific requirements and deployment environments.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataOperationConfig:
    """
    Configuration for data management operations in Milvus.
    
    This class centralizes all configurable parameters for data operations,
    making it easy for external projects to customize behavior without
    modifying the core implementation.
    
    Attributes:
        default_batch_size: Default number of documents to process in a single batch.
                           Used when batch_size is not specified in operation calls.
        max_batch_size: Maximum allowed batch size to prevent memory issues.
        min_batch_size: Minimum allowed batch size for efficiency.
        default_operation_timeout: Default timeout in seconds for data operations.
                                  None means no timeout.
        health_check_timeout: Timeout in seconds for connection health checks.
        retry_transient_errors: Whether to retry operations that fail with transient errors.
        max_transient_retries: Maximum number of retry attempts for transient errors.
        transient_retry_delay: Base delay in seconds between retry attempts.
                              Actual delay increases linearly with attempt number.
        enable_timing: Whether to enable performance timing for operations.
        strict_validation: Whether to perform strict schema validation before operations.
    
    Example:
        ```python
        # Create custom configuration
        config = DataOperationConfig(
            default_batch_size=500,
            retry_transient_errors=True,
            default_operation_timeout=60.0
        )
        
        # Use with DataManager
        manager = DataManager(conn_mgr, coll_mgr, config=config)
        ```
    """
    
    # Batching settings
    default_batch_size: int = 1000
    max_batch_size: int = 10000
    min_batch_size: int = 100
    
    # Timeout settings (seconds, None means no timeout)
    default_operation_timeout: Optional[float] = 30.0
    health_check_timeout: float = 5.0
    
    # Retry settings for transient failures not handled by ConnectionManager
    # These apply only to specific transient errors like temporary schema
    # unavailability or collection locks
    retry_transient_errors: bool = True
    max_transient_retries: int = 3
    transient_retry_delay: float = 0.5
    
    # Performance monitoring
    enable_timing: bool = True
    
    # Validation settings
    strict_validation: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.default_batch_size < self.min_batch_size:
            logger.warning(
                f"default_batch_size ({self.default_batch_size}) is less than "
                f"min_batch_size ({self.min_batch_size}). Setting to min_batch_size."
            )
            self.default_batch_size = self.min_batch_size
        
        if self.default_batch_size > self.max_batch_size:
            logger.warning(
                f"default_batch_size ({self.default_batch_size}) exceeds "
                f"max_batch_size ({self.max_batch_size}). Setting to max_batch_size."
            )
            self.default_batch_size = self.max_batch_size
        
        if self.max_transient_retries < 0:
            raise ValueError("max_transient_retries must be non-negative")
        
        if self.transient_retry_delay < 0:
            raise ValueError("transient_retry_delay must be non-negative")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataOperationConfig':
        """
        Create configuration from a dictionary.
        
        This method allows external projects to provide configuration
        via dictionary (e.g., from YAML, JSON, or environment variables).
        
        Args:
            config_dict: Dictionary containing configuration parameters.
                        Keys should match the dataclass field names.
        
        Returns:
            DataOperationConfig instance with specified parameters.
        
        Example:
            ```python
            config_dict = {
                'default_batch_size': 500,
                'retry_transient_errors': True,
                'default_operation_timeout': 60.0
            }
            config = DataOperationConfig.from_dict(config_dict)
            ```
        """
        # Filter to only include valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        return cls(**filtered_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration.
        """
        return {
            'default_batch_size': self.default_batch_size,
            'max_batch_size': self.max_batch_size,
            'min_batch_size': self.min_batch_size,
            'default_operation_timeout': self.default_operation_timeout,
            'health_check_timeout': self.health_check_timeout,
            'retry_transient_errors': self.retry_transient_errors,
            'max_transient_retries': self.max_transient_retries,
            'transient_retry_delay': self.transient_retry_delay,
            'enable_timing': self.enable_timing,
            'strict_validation': self.strict_validation
        }
    
    def validate_batch_size(self, batch_size: Optional[int]) -> int:
        """
        Validate and normalize a batch size parameter.
        
        Args:
            batch_size: Requested batch size, or None to use default.
        
        Returns:
            Validated batch size within configured bounds.
        
        Raises:
            ValueError: If batch_size is outside allowed bounds.
        """
        if batch_size is None:
            return self.default_batch_size
        
        if batch_size < self.min_batch_size:
            raise ValueError(
                f"Batch size {batch_size} is below minimum ({self.min_batch_size})"
            )
        
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds maximum ({self.max_batch_size})"
            )
        
        return batch_size

