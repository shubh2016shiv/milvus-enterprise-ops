"""
Backup Recovery Configuration

Centralized configuration for backup and recovery operations, providing
a single source of truth for all tunable parameters related to storage,
performance, reliability, and retention.

This configuration can be customized by external projects to match their
specific requirements and deployment environments.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from .models.entities import BackupStorageType, ChecksumAlgorithm

logger = logging.getLogger(__name__)


@dataclass
class BackupRecoveryConfig:
    """
    Configuration for backup and recovery operations.
    
    This class centralizes all configurable parameters for backup operations,
    making it easy for external projects to customize behavior without
    modifying the core implementation.
    
    Storage Settings:
        default_storage_type: Default backend for backups (LOCAL_FILE or MILVUS_NATIVE)
        local_backup_root_path: Root directory for local file backups
        milvus_backup_bucket: Bucket/path for Milvus native backups
    
    Performance Settings:
        default_chunk_size_mb: Size of chunks for large backups (default: 256MB)
        max_concurrent_chunks: Maximum parallel chunk processing (default: 4)
        compression_enabled: Enable compression by default
        compression_level: Compression level 1-9 (default: 6)
    
    Reliability Settings:
        enable_checksum_verification: Always verify checksums (default: True)
        checksum_algorithm: Algorithm to use (SHA256, MD5, BLAKE2B)
        deep_verification_interval_days: Run deep verification every N days (default: 7)
        auto_verify_after_backup: Verify immediately after creating backup
        auto_verify_before_restore: Verify before restoring backup
    
    Timeout Settings:
        default_backup_timeout: Maximum time for backup operations in seconds
        default_restore_timeout: Maximum time for restore operations in seconds
        verification_timeout: Maximum time for verification in seconds
    
    Retention Settings:
        retention_count: Keep N most recent backups (default: 10)
        retention_days: Keep backups for N days (default: 30)
        min_backups_to_keep: Minimum backups to retain regardless of age (default: 3)
    
    Retry Settings:
        retry_transient_errors: Retry on transient failures
        max_retries: Maximum retry attempts (default: 3)
        retry_delay_seconds: Base delay between retries (default: 5)
    
    Monitoring Settings:
        enable_timing: Track performance metrics
        progress_poll_interval: Progress update interval in seconds (default: 2.0)
    
    Example:
        ```python
        # Create custom configuration
        config = BackupRecoveryConfig(
            local_backup_root_path="/mnt/backups",
            compression_enabled=True,
            compression_level=9,
            retention_count=20,
            enable_timing=True
        )
        
        # Use with backup manager
        backup_manager = BackupManager(
            connection_mgr=conn_mgr,
            collection_mgr=coll_mgr,
            config=config
        )
        ```
    """
    
    # Storage Settings
    default_storage_type: BackupStorageType = BackupStorageType.LOCAL_FILE
    local_backup_root_path: str = "./milvus_backups"
    milvus_backup_bucket: Optional[str] = None
    
    # Performance Settings
    default_chunk_size_mb: int = 256
    max_concurrent_chunks: int = 4
    compression_enabled: bool = True
    compression_level: int = 6
    
    # Reliability Settings
    enable_checksum_verification: bool = True
    checksum_algorithm: ChecksumAlgorithm = ChecksumAlgorithm.SHA256
    deep_verification_interval_days: int = 7
    auto_verify_after_backup: bool = True
    auto_verify_before_restore: bool = True
    
    # Timeout Settings (in seconds)
    default_backup_timeout: float = 3600.0  # 1 hour
    default_restore_timeout: float = 3600.0  # 1 hour
    verification_timeout: float = 1800.0  # 30 minutes
    
    # Retention Settings
    retention_count: int = 10
    retention_days: int = 30
    min_backups_to_keep: int = 3
    
    # Retry Settings
    retry_transient_errors: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    
    # Monitoring Settings
    enable_timing: bool = True
    progress_poll_interval: float = 2.0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Ensures all configuration values are within acceptable ranges and
        are logically consistent.
        
        Raises:
            ValueError: If any configuration parameter is invalid
        """
        # Validate chunk size
        if self.default_chunk_size_mb <= 0:
            raise ValueError("default_chunk_size_mb must be positive")
        if self.default_chunk_size_mb > 2048:
            logger.warning(f"Large chunk size ({self.default_chunk_size_mb}MB) may cause memory issues")
        
        # Validate concurrency
        if self.max_concurrent_chunks <= 0:
            raise ValueError("max_concurrent_chunks must be positive")
        if self.max_concurrent_chunks > 16:
            logger.warning(f"High concurrency ({self.max_concurrent_chunks}) may strain resources")
        
        # Validate compression level
        if not 1 <= self.compression_level <= 9:
            raise ValueError("compression_level must be between 1 and 9")
        
        # Validate timeouts
        if self.default_backup_timeout <= 0:
            raise ValueError("default_backup_timeout must be positive")
        if self.default_restore_timeout <= 0:
            raise ValueError("default_restore_timeout must be positive")
        if self.verification_timeout <= 0:
            raise ValueError("verification_timeout must be positive")
        
        # Validate retention settings
        if self.retention_count < 0:
            raise ValueError("retention_count cannot be negative")
        if self.retention_days < 0:
            raise ValueError("retention_days cannot be negative")
        if self.min_backups_to_keep < 0:
            raise ValueError("min_backups_to_keep cannot be negative")
        
        # Warn if retention settings might delete all backups
        if self.retention_count < self.min_backups_to_keep:
            logger.warning(
                f"retention_count ({self.retention_count}) is less than "
                f"min_backups_to_keep ({self.min_backups_to_keep}). "
                f"min_backups_to_keep will take precedence."
            )
        
        # Validate retry settings
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds cannot be negative")
        
        # Validate monitoring settings
        if self.progress_poll_interval <= 0:
            raise ValueError("progress_poll_interval must be positive")
        
        # Validate storage path
        if self.default_storage_type == BackupStorageType.LOCAL_FILE:
            if not self.local_backup_root_path:
                raise ValueError("local_backup_root_path must be set for LOCAL_FILE storage")
        
        # Validate deep verification interval
        if self.deep_verification_interval_days < 0:
            raise ValueError("deep_verification_interval_days cannot be negative")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BackupRecoveryConfig':
        """
        Create configuration from a dictionary.
        
        This method allows loading configuration from external sources like
        JSON files, YAML files, or environment variables.
        
        Args:
            config_dict: Dictionary containing configuration parameters
        
        Returns:
            BackupRecoveryConfig instance
        
        Example:
            ```python
            config_dict = {
                'local_backup_root_path': '/mnt/backups',
                'compression_enabled': True,
                'retention_count': 20
            }
            config = BackupRecoveryConfig.from_dict(config_dict)
            ```
        """
        # Handle enum conversions
        if 'default_storage_type' in config_dict:
            if isinstance(config_dict['default_storage_type'], str):
                config_dict['default_storage_type'] = BackupStorageType(
                    config_dict['default_storage_type']
                )
        
        if 'checksum_algorithm' in config_dict:
            if isinstance(config_dict['checksum_algorithm'], str):
                config_dict['checksum_algorithm'] = ChecksumAlgorithm(
                    config_dict['checksum_algorithm']
                )
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        This method allows serializing configuration for storage or transmission.
        Enum values are converted to their string representations.
        
        Returns:
            Dictionary representation of the configuration
        
        Example:
            ```python
            config = BackupRecoveryConfig()
            config_dict = config.to_dict()
            # Save to JSON
            with open('config.json', 'w') as f:
                json.dump(config_dict, f)
            ```
        """
        result = asdict(self)
        
        # Convert enums to strings
        if isinstance(result['default_storage_type'], BackupStorageType):
            result['default_storage_type'] = result['default_storage_type'].value
        if isinstance(result['checksum_algorithm'], ChecksumAlgorithm):
            result['checksum_algorithm'] = result['checksum_algorithm'].value
        
        return result
    
    def get_backup_root_path(self) -> Path:
        """
        Get the backup root path as a Path object.
        
        Returns:
            Path object for the backup root directory
        """
        return Path(self.local_backup_root_path)
    
    def ensure_backup_directory_exists(self) -> None:
        """
        Ensure the backup root directory exists.
        
        Creates the directory if it doesn't exist. Only applicable for
        local file storage.
        
        Raises:
            OSError: If directory cannot be created
        """
        if self.default_storage_type == BackupStorageType.LOCAL_FILE:
            backup_path = self.get_backup_root_path()
            backup_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured backup directory exists: {backup_path}")
    
    @property
    def chunk_size_bytes(self) -> int:
        """Get chunk size in bytes."""
        return self.default_chunk_size_mb * 1024 * 1024
    
    @property
    def is_local_storage(self) -> bool:
        """Check if using local file storage."""
        return self.default_storage_type == BackupStorageType.LOCAL_FILE
    
    @property
    def is_milvus_native_storage(self) -> bool:
        """Check if using Milvus native storage."""
        return self.default_storage_type == BackupStorageType.MILVUS_NATIVE
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"BackupRecoveryConfig("
            f"storage_type={self.default_storage_type.value}, "
            f"backup_path={self.local_backup_root_path}, "
            f"compression={'enabled' if self.compression_enabled else 'disabled'}, "
            f"retention={self.retention_count} backups / {self.retention_days} days"
            f")"
        )

