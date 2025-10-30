"""
Backup Recovery Exceptions

Defines a granular exception hierarchy for backup and restore operations,
providing specific exception types for different failure scenarios to enable
targeted error handling and recovery strategies.
"""

from typing import Optional, Dict, Any, List


class BackupRecoveryError(Exception):
    """
    Base exception for all backup and recovery operations.
    
    This is the parent class for all backup-related exceptions, providing
    common context attributes that are useful for debugging and error reporting.
    
    Attributes:
        message: Human-readable error message
        collection_name: Name of the collection involved (if applicable)
        backup_id: Identifier of the backup involved (if applicable)
        context: Additional context information as key-value pairs
    
    Example:
        ```python
        try:
            # Backup operation
            pass
        except BackupRecoveryError as e:
            logger.error(f"Backup error for {e.collection_name}: {e.message}")
            logger.error(f"Context: {e.context}")
        ```
    """
    
    def __init__(
        self,
        message: str,
        collection_name: Optional[str] = None,
        backup_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.collection_name = collection_name
        self.backup_id = backup_id
        self.context = context or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.collection_name:
            parts.append(f"Collection: {self.collection_name}")
        if self.backup_id:
            parts.append(f"Backup ID: {self.backup_id}")
        if self.context:
            parts.append(f"Context: {self.context}")
        return " | ".join(parts)


class BackupError(BackupRecoveryError):
    """
    Failed to create backup.
    
    Raised when a backup creation operation fails due to issues such as
    storage problems, data access errors, or Milvus operation failures.
    
    Additional Attributes:
        storage_path: Path where backup was being created
        storage_type: Type of storage backend being used
    
    Example:
        ```python
        raise BackupError(
            message="Failed to write backup data",
            collection_name="documents",
            storage_path="/backups/documents/backup_123",
            storage_type="LOCAL_FILE"
        )
        ```
    """
    
    def __init__(
        self,
        message: str,
        collection_name: Optional[str] = None,
        backup_id: Optional[str] = None,
        storage_path: Optional[str] = None,
        storage_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, collection_name, backup_id, context)
        self.storage_path = storage_path
        self.storage_type = storage_type


class RestoreError(BackupRecoveryError):
    """
    Failed to restore backup.
    
    Raised when a restore operation fails, which could be due to corrupted
    backup data, incompatible schemas, or target collection issues.
    
    Additional Attributes:
        target_collection_name: Name of the collection being restored to
        source_backup_id: Identifier of the backup being restored from
    
    Example:
        ```python
        raise RestoreError(
            message="Schema incompatibility detected",
            collection_name="documents",
            backup_id="backup_123",
            target_collection_name="documents_restored"
        )
        ```
    """
    
    def __init__(
        self,
        message: str,
        backup_id: Optional[str] = None,
        target_collection_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, target_collection_name, backup_id, context)
        self.target_collection_name = target_collection_name


class BackupNotFoundError(BackupRecoveryError):
    """
    Backup does not exist.
    
    Raised when attempting to access, restore, or verify a backup that
    cannot be found in the storage backend.
    
    Additional Attributes:
        backup_name: Name of the backup that was not found
        storage_path: Path where backup was expected
    
    Example:
        ```python
        raise BackupNotFoundError(
            message="Backup not found in storage",
            backup_id="backup_123",
            backup_name="daily_backup_2024_01_15",
            storage_path="/backups/documents"
        )
        ```
    """
    
    def __init__(
        self,
        message: str,
        backup_id: Optional[str] = None,
        backup_name: Optional[str] = None,
        storage_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, None, backup_id, context)
        self.backup_name = backup_name
        self.storage_path = storage_path


class BackupCorruptedError(BackupRecoveryError):
    """
    Backup failed verification or is corrupted.
    
    Raised when backup integrity checks fail, indicating the backup data
    may be corrupted or tampered with.
    
    Additional Attributes:
        expected_checksum: Expected checksum value
        actual_checksum: Actual checksum value computed
        checksum_algorithm: Algorithm used for checksum
        failed_files: List of files that failed verification
    
    Example:
        ```python
        raise BackupCorruptedError(
            message="Checksum mismatch detected",
            backup_id="backup_123",
            expected_checksum="abc123...",
            actual_checksum="def456...",
            checksum_algorithm="SHA256"
        )
        ```
    """
    
    def __init__(
        self,
        message: str,
        backup_id: Optional[str] = None,
        expected_checksum: Optional[str] = None,
        actual_checksum: Optional[str] = None,
        checksum_algorithm: Optional[str] = None,
        failed_files: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, None, backup_id, context)
        self.expected_checksum = expected_checksum
        self.actual_checksum = actual_checksum
        self.checksum_algorithm = checksum_algorithm
        self.failed_files = failed_files or []


class BackupStorageError(BackupRecoveryError):
    """
    Storage access failure.
    
    Raised when there are issues accessing the storage backend, such as
    permission errors, disk full, or network issues for remote storage.
    
    Additional Attributes:
        storage_path: Path that caused the error
        error_code: System error code (if available)
        permissions_issue: Whether this is a permissions-related error
    
    Example:
        ```python
        raise BackupStorageError(
            message="Permission denied when writing backup",
            storage_path="/backups/documents",
            permissions_issue=True,
            error_code="EACCES"
        )
        ```
    """
    
    def __init__(
        self,
        message: str,
        storage_path: Optional[str] = None,
        error_code: Optional[str] = None,
        permissions_issue: bool = False,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, None, None, context)
        self.storage_path = storage_path
        self.error_code = error_code
        self.permissions_issue = permissions_issue


class BackupAlreadyExistsError(BackupRecoveryError):
    """
    Backup name conflict - backup already exists.
    
    Raised when attempting to create a backup with a name that already
    exists in the storage backend.
    
    Additional Attributes:
        backup_name: Name of the conflicting backup
        existing_backup_id: ID of the existing backup
        storage_path: Path where existing backup is located
    
    Example:
        ```python
        raise BackupAlreadyExistsError(
            message="Backup with this name already exists",
            backup_name="daily_backup",
            existing_backup_id="backup_122",
            collection_name="documents"
        )
        ```
    """
    
    def __init__(
        self,
        message: str,
        backup_name: Optional[str] = None,
        existing_backup_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        storage_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, collection_name, existing_backup_id, context)
        self.backup_name = backup_name
        self.storage_path = storage_path


class RestoreValidationError(BackupRecoveryError):
    """
    Pre-restore validation failed.
    
    Raised when validation checks before restore operation fail, such as
    schema compatibility issues or missing prerequisites.
    
    Additional Attributes:
        validation_errors: List of specific validation errors
        target_collection_name: Name of target collection
    
    Example:
        ```python
        raise RestoreValidationError(
            message="Schema validation failed",
            backup_id="backup_123",
            target_collection_name="documents",
            validation_errors=["Field 'embedding' dimension mismatch"]
        )
        ```
    """
    
    def __init__(
        self,
        message: str,
        backup_id: Optional[str] = None,
        target_collection_name: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, target_collection_name, backup_id, context)
        self.validation_errors = validation_errors or []


class InsufficientStorageError(BackupRecoveryError):
    """
    Not enough disk space for backup operation.
    
    Raised when there is insufficient storage space available to complete
    a backup or restore operation.
    
    Additional Attributes:
        required_bytes: Number of bytes required
        available_bytes: Number of bytes available
        storage_path: Path where space is needed
    
    Example:
        ```python
        raise InsufficientStorageError(
            message="Insufficient disk space for backup",
            required_bytes=10737418240,  # 10 GB
            available_bytes=5368709120,   # 5 GB
            storage_path="/backups"
        )
        ```
    """
    
    def __init__(
        self,
        message: str,
        required_bytes: Optional[int] = None,
        available_bytes: Optional[int] = None,
        storage_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, None, None, context)
        self.required_bytes = required_bytes
        self.available_bytes = available_bytes
        self.storage_path = storage_path
    
    @property
    def required_gb(self) -> Optional[float]:
        """Get required space in GB."""
        if self.required_bytes:
            return self.required_bytes / (1024 ** 3)
        return None
    
    @property
    def available_gb(self) -> Optional[float]:
        """Get available space in GB."""
        if self.available_bytes:
            return self.available_bytes / (1024 ** 3)
        return None


class BackupInProgressError(BackupRecoveryError):
    """
    Operation conflicts with ongoing backup.
    
    Raised when attempting an operation that conflicts with a backup
    currently in progress.
    
    Additional Attributes:
        in_progress_backup_id: ID of the backup in progress
        progress_percentage: Current progress of the backup
        operation_attempted: The operation that was attempted
    
    Example:
        ```python
        raise BackupInProgressError(
            message="Cannot perform operation while backup is in progress",
            collection_name="documents",
            in_progress_backup_id="backup_123",
            progress_percentage=45.0,
            operation_attempted="delete_collection"
        )
        ```
    """
    
    def __init__(
        self,
        message: str,
        collection_name: Optional[str] = None,
        in_progress_backup_id: Optional[str] = None,
        progress_percentage: Optional[float] = None,
        operation_attempted: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, collection_name, in_progress_backup_id, context)
        self.progress_percentage = progress_percentage
        self.operation_attempted = operation_attempted


class PartitionNotFoundError(BackupRecoveryError):
    """
    Specified partition does not exist.
    
    Raised when attempting to backup or restore a partition that doesn't
    exist in the collection.
    
    Additional Attributes:
        partition_name: Name of the partition that was not found
        available_partitions: List of available partitions
    
    Example:
        ```python
        raise PartitionNotFoundError(
            message="Partition does not exist",
            collection_name="documents",
            partition_name="partition_2024_13",
            available_partitions=["partition_2024_01", "partition_2024_02"]
        )
        ```
    """
    
    def __init__(
        self,
        message: str,
        collection_name: Optional[str] = None,
        partition_name: Optional[str] = None,
        available_partitions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, collection_name, None, context)
        self.partition_name = partition_name
        self.available_partitions = available_partitions or []


class SchemaIncompatibleError(BackupRecoveryError):
    """
    Schema mismatch during restore.
    
    Raised when the schema of a backup is incompatible with the target
    collection or environment.
    
    Additional Attributes:
        backup_schema: Schema from the backup
        target_schema: Schema of the target (if exists)
        incompatibilities: List of specific schema differences
    
    Example:
        ```python
        raise SchemaIncompatibleError(
            message="Schema incompatibility detected",
            backup_id="backup_123",
            collection_name="documents",
            incompatibilities=[
                "Field 'embedding' dimension: backup=128, target=256",
                "Field 'metadata' type mismatch"
            ]
        )
        ```
    """
    
    def __init__(
        self,
        message: str,
        backup_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        backup_schema: Optional[Dict[str, Any]] = None,
        target_schema: Optional[Dict[str, Any]] = None,
        incompatibilities: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, collection_name, backup_id, context)
        self.backup_schema = backup_schema
        self.target_schema = target_schema
        self.incompatibilities = incompatibilities or []

