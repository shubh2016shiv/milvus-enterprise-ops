"""
Index Operations Configuration

Centralized configuration for index operations, providing a single source
of truth for all tunable parameters related to index creation, monitoring,
and optimization.

This configuration can be customized by external projects to match their
specific requirements and deployment environments.

Typical usage:
    from Milvus_Ops.index_operations import IndexOperationConfig
    
    # Create custom configuration
    config = IndexOperationConfig(
        default_timeout=120.0,
        build_progress_poll_interval=5.0,
        enable_timing=True
    )
    
    # Use with IndexManager
    index_manager = IndexManager(
        connection_manager=conn_mgr,
        collection_manager=coll_mgr,
        config=config
    )
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class IndexOperationConfig:
    """
    Configuration for index operations in Milvus.
    
    This class centralizes all configurable parameters for index operations,
    making it easy for external projects to customize behavior without
    modifying the core implementation.
    
    Attributes:
        default_timeout: Default timeout in seconds for index operations.
                        None means no timeout.
        build_progress_poll_interval: Interval in seconds for polling build progress.
        max_concurrent_builds: Maximum number of concurrent index builds.
                              0 means no limit.
        enable_timing: Whether to enable performance timing for operations.
        auto_optimize_params: Whether to automatically optimize index parameters
                             based on data characteristics.
        resource_monitoring: Whether to monitor resource usage during index operations.
        retry_transient_errors: Whether to retry operations that fail with transient errors.
        max_transient_retries: Maximum number of retry attempts for transient errors.
        transient_retry_delay: Base delay in seconds between retry attempts.
                              Actual delay increases linearly with attempt number.
    
    Example:
        ```python
        # Create custom configuration
        config = IndexOperationConfig(
            default_timeout=120.0,
            build_progress_poll_interval=5.0,
            enable_timing=True
        )
        
        # Use with IndexManager
        index_manager = IndexManager(conn_mgr, coll_mgr, config=config)
        ```
    """
    
    # Timeout settings (seconds)
    default_timeout: Optional[float] = 60.0
    
    # Progress monitoring
    build_progress_poll_interval: float = 2.0
    
    # Concurrency settings
    max_concurrent_builds: int = 0  # 0 means no limit
    
    # Performance monitoring
    enable_timing: bool = True
    
    # Optimization settings
    auto_optimize_params: bool = False
    resource_monitoring: bool = False
    
    # Retry settings for transient failures
    retry_transient_errors: bool = True
    max_transient_retries: int = 3
    transient_retry_delay: float = 0.5
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.build_progress_poll_interval <= 0:
            logger.warning(
                f"build_progress_poll_interval ({self.build_progress_poll_interval}) "
                f"must be positive. Setting to 2.0."
            )
            self.build_progress_poll_interval = 2.0
        
        if self.max_concurrent_builds < 0:
            logger.warning(
                f"max_concurrent_builds ({self.max_concurrent_builds}) "
                f"cannot be negative. Setting to 0 (no limit)."
            )
            self.max_concurrent_builds = 0
        
        if self.max_transient_retries < 0:
            logger.warning(
                f"max_transient_retries ({self.max_transient_retries}) "
                f"cannot be negative. Setting to 0 (no retries)."
            )
            self.max_transient_retries = 0
        
        if self.transient_retry_delay < 0:
            logger.warning(
                f"transient_retry_delay ({self.transient_retry_delay}) "
                f"cannot be negative. Setting to 0.5."
            )
            self.transient_retry_delay = 0.5
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'IndexOperationConfig':
        """
        Create configuration from a dictionary.
        
        This method allows external projects to provide configuration
        via dictionary (e.g., from YAML, JSON, or environment variables).
        
        Args:
            config_dict: Dictionary containing configuration parameters.
                        Keys should match the dataclass field names.
        
        Returns:
            IndexOperationConfig instance with specified parameters.
        
        Example:
            ```python
            config_dict = {
                'default_timeout': 120.0,
                'build_progress_poll_interval': 5.0,
                'enable_timing': True
            }
            config = IndexOperationConfig.from_dict(config_dict)
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
            'default_timeout': self.default_timeout,
            'build_progress_poll_interval': self.build_progress_poll_interval,
            'max_concurrent_builds': self.max_concurrent_builds,
            'enable_timing': self.enable_timing,
            'auto_optimize_params': self.auto_optimize_params,
            'resource_monitoring': self.resource_monitoring,
            'retry_transient_errors': self.retry_transient_errors,
            'max_transient_retries': self.max_transient_retries,
            'transient_retry_delay': self.transient_retry_delay
        }
