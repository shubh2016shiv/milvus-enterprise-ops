"""
Configuration module for partition operations.

Provides essential configuration settings with sensible defaults
and environment variable support.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PartitionConfig:
    """
    Essential configuration for partition operations.

    Simple, focused configuration with only the settings that matter.
    """

    def __init__(self, **kwargs):
        """Initialize with environment variables and provided values."""

        # Core timeout settings (in seconds)
        self.default_operation_timeout = float(
            os.getenv("MILVUS_PARTITION_DEFAULT_TIMEOUT", "30.0")
        )
        self.create_partition_timeout = float(
            os.getenv("MILVUS_PARTITION_CREATE_TIMEOUT", "60.0")
        )
        self.drop_partition_timeout = float(
            os.getenv("MILVUS_PARTITION_DROP_TIMEOUT", "30.0")
        )
        self.load_partition_timeout = float(
            os.getenv("MILVUS_PARTITION_LOAD_TIMEOUT", "120.0")
        )

        # Validation settings
        self.validate_partition_names = (
            os.getenv("MILVUS_PARTITION_VALIDATE_NAMES", "true").lower() == "true"
        )
        self.max_partition_name_length = int(
            os.getenv("MILVUS_PARTITION_MAX_NAME_LENGTH", "255")
        )

        # Safety settings
        self.prevent_default_partition_deletion = (
            os.getenv("MILVUS_PARTITION_PREVENT_DEFAULT_DELETION", "true").lower() == "true"
        )

        # Override with provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")

        # Validate configuration
        self._validate()

    def _validate(self):
        """Validate configuration values."""
        if self.max_partition_name_length <= 0 or self.max_partition_name_length > 1000:
            raise ValueError("max_partition_name_length must be between 1 and 1000")

        if self.default_operation_timeout <= 0:
            raise ValueError("default_operation_timeout must be greater than 0")


# Global configuration instance
_config: Optional[PartitionConfig] = None


def get_partition_config() -> PartitionConfig:
    """
    Get the global partition configuration.

    Returns:
        The global configuration instance
    """
    global _config
    if _config is None:
        _config = PartitionConfig()
    return _config


def set_partition_config(config: PartitionConfig) -> None:
    """
    Set the global partition configuration.

    Args:
        config: The configuration to set
    """
    global _config
    _config = config
    logger.info("Updated partition configuration")