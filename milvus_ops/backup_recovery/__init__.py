"""
Backup and Recovery Module

Provides comprehensive functionality for reliable backup and recovery of Milvus
collections with support for multiple storage backends, robust verification,
and flexible retention policies.

Features:
- Local file system and Milvus native backup backends
- Full collection and partition-level backups
- Checksum-based integrity verification with periodic deep verification
- Configurable compression with multiple algorithms
- Progress tracking for long-running operations
- Flexible retention policies (count-based and time-based)
- Comprehensive error handling and recovery
- Type-safe parameters and configuration

Typical usage from external projects:

    from Milvus_Ops.backup_recovery import (
        BackupRecoveryConfig,
        BackupParams,
        BackupStorageType,
        ChecksumCalculator
    )

    # Create configuration
    config = BackupRecoveryConfig(
        local_backup_root_path="/mnt/backups",
        compression_enabled=True,
        retention_count=20
    )

    # Create backup parameters
    params = BackupParams(
        backup_type=BackupType.FULL_COLLECTION,
        compression_enabled=True
    )

    # Use with backup manager (when implemented)
    # backup_manager = BackupManager(conn_mgr, coll_mgr, config=config)
    # result = await backup_manager.create_backup("documents", params)
"""

import importlib.util
import logging

from .config import BackupRecoveryConfig
from .exceptions import (
    BackupAlreadyExistsError,
    BackupCorruptedError,
    BackupError,
    BackupInProgressError,
    BackupNotFoundError,
    BackupRecoveryError,
    BackupStorageError,
    InsufficientStorageError,
    PartitionNotFoundError,
    RestoreError,
    RestoreValidationError,
    SchemaIncompatibleError,
)
from .models.entities import (
    BackupMetadata,
    BackupProgress,
    BackupResult,
    BackupState,
    BackupStorageType,
    BackupType,
    BackupVersion,
    ChecksumAlgorithm,
    RestoreResult,
    VerificationResult,
    VerificationType,
)
from .models.parameters import BackupParams, RestoreParams, VerificationParams
from .utils import (
    BackupProgressTracker,
    BackupProgressTrackerRegistry,
    ChecksumCalculator,
    CompressionHandler,
    RetentionPolicyManager,
    get_registry,
)

# Try to import pyarrow for Parquet support
PARQUET_AVAILABLE = importlib.util.find_spec("pyarrow") is not None
if not PARQUET_AVAILABLE:
    logging.warning("pyarrow not available, backup functionality will be limited")

# Core manager or mock implementation
if PARQUET_AVAILABLE:
    from .core import BackupManager
else:
    from .mock_backend import MockBackupManager as BackupManager

__all__ = [
    # Core manager
    "BackupManager",
    # Configuration
    "BackupRecoveryConfig",
    # Enums
    "BackupState",
    "BackupType",
    "BackupStorageType",
    "ChecksumAlgorithm",
    "VerificationType",
    # Entities
    "BackupMetadata",
    "BackupResult",
    "RestoreResult",
    "BackupProgress",
    "VerificationResult",
    "BackupVersion",
    # Parameters
    "BackupParams",
    "RestoreParams",
    "VerificationParams",
    # Exceptions
    "BackupRecoveryError",
    "BackupError",
    "RestoreError",
    "BackupNotFoundError",
    "BackupCorruptedError",
    "BackupStorageError",
    "BackupAlreadyExistsError",
    "RestoreValidationError",
    "InsufficientStorageError",
    "BackupInProgressError",
    "PartitionNotFoundError",
    "SchemaIncompatibleError",
    # Utilities
    "ChecksumCalculator",
    "CompressionHandler",
    "BackupProgressTracker",
    "BackupProgressTrackerRegistry",
    "get_registry",
    "RetentionPolicyManager",
]
