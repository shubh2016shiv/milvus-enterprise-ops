"""
Backup Recovery Entities

Defines data models for backup and restore operations, including metadata,
progress tracking, verification results, and operation outcomes.

These models use Pydantic for validation and provide a type-safe interface
for backup operations.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class BackupState(str, Enum):
    """
    Represents the current state of a backup operation.
    
    States:
        NONE: No backup exists or operation not started
        IN_PROGRESS: Backup is currently being created
        COMPLETED: Backup successfully completed
        FAILED: Backup operation failed
        VERIFYING: Backup is being verified
        VERIFIED: Backup has been verified and is ready for use
    """
    NONE = "NONE"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    VERIFYING = "VERIFYING"
    VERIFIED = "VERIFIED"


class BackupType(str, Enum):
    """
    Type of backup operation performed.
    
    Types:
        FULL_COLLECTION: Complete backup of entire collection
        PARTITION: Backup of specific partitions only
        INCREMENTAL: Incremental backup (future enhancement)
    """
    FULL_COLLECTION = "FULL_COLLECTION"
    PARTITION = "PARTITION"
    INCREMENTAL = "INCREMENTAL"


class BackupStorageType(str, Enum):
    """
    Storage backend used for the backup.
    
    Types:
        LOCAL_FILE: Backup stored in local file system
        MILVUS_NATIVE: Backup using Milvus-Backup tool
    """
    LOCAL_FILE = "LOCAL_FILE"
    MILVUS_NATIVE = "MILVUS_NATIVE"


class ChecksumAlgorithm(str, Enum):
    """
    Supported checksum algorithms for data integrity verification.
    
    Algorithms:
        SHA256: SHA-256 hash (recommended for security)
        MD5: MD5 hash (faster, less secure)
        BLAKE2B: BLAKE2b hash (fast and secure)
    """
    SHA256 = "SHA256"
    MD5 = "MD5"
    BLAKE2B = "BLAKE2B"


class VerificationType(str, Enum):
    """
    Type of backup verification to perform.
    
    Types:
        CHECKSUM: Fast checksum-based verification
        DEEP: Full restore to temp collection and validate data
        QUICK: Sample-based verification for large backups
    """
    CHECKSUM = "CHECKSUM"
    DEEP = "DEEP"
    QUICK = "QUICK"


class BackupMetadata(BaseModel):
    """
    Comprehensive metadata about a backup.
    
    This model stores all information necessary to identify, manage, and restore
    a backup, including integrity information, storage details, and recovery metadata.
    
    Attributes:
        backup_id: Unique identifier for the backup (UUID format)
        backup_name: Human-readable name for the backup
        collection_name: Name of the collection that was backed up
        partition_names: List of partition names included in backup (empty for full backup)
        created_at: Timestamp when backup was created
        storage_type: Type of storage backend used (LOCAL_FILE or MILVUS_NATIVE)
        storage_path: Path or location where backup is stored
        size_bytes: Total size of backup data in bytes
        compressed_size_bytes: Size after compression (if compression enabled)
        checksum: Checksum value for integrity verification
        checksum_algorithm: Algorithm used to calculate checksum
        is_verified: Whether backup has been verified for integrity
        last_verified_at: Timestamp of most recent verification
        state: Current state of the backup
        backup_type: Type of backup (FULL_COLLECTION or PARTITION)
        error_message: Error message if backup failed
        milvus_version: Version of Milvus when backup was created
        schema_snapshot: JSON representation of collection schema
        row_count: Total number of rows in the backup
        segment_count: Number of segments in the backup
        compression_enabled: Whether compression was used
        compression_level: Compression level used (1-9)
        include_indexes: Whether indexes were included in backup
        metadata_version: Version of metadata format (for future compatibility)
    
    Example:
        ```python
        metadata = BackupMetadata(
            backup_id="550e8400-e29b-41d4-a716-446655440000",
            backup_name="daily_backup_2024_01_15",
            collection_name="documents",
            partition_names=[],
            created_at=datetime.now(),
            storage_type=BackupStorageType.LOCAL_FILE,
            storage_path="/backups/documents/backup_id",
            size_bytes=1024000000,
            checksum="abc123...",
            checksum_algorithm=ChecksumAlgorithm.SHA256,
            state=BackupState.VERIFIED
        )
        ```
    """
    # Basic information
    backup_id: str = Field(..., description="Unique identifier for the backup")
    backup_name: str = Field(..., description="Human-readable backup name")
    collection_name: str = Field(..., description="Name of the backed up collection")
    partition_names: List[str] = Field(default_factory=list, description="List of partitions in backup")
    created_at: datetime = Field(..., description="Backup creation timestamp")
    
    # Storage information
    storage_type: BackupStorageType = Field(..., description="Storage backend type")
    storage_path: str = Field(..., description="Path to backup location")
    size_bytes: int = Field(default=0, ge=0, description="Total backup size in bytes")
    compressed_size_bytes: Optional[int] = Field(default=None, ge=0, description="Compressed size if applicable")
    
    # Integrity information
    checksum: str = Field(default="", description="Checksum for integrity verification")
    checksum_algorithm: ChecksumAlgorithm = Field(default=ChecksumAlgorithm.SHA256, description="Checksum algorithm used")
    is_verified: bool = Field(default=False, description="Whether backup has been verified")
    last_verified_at: Optional[datetime] = Field(default=None, description="Last verification timestamp")
    
    # State information
    state: BackupState = Field(default=BackupState.NONE, description="Current backup state")
    backup_type: BackupType = Field(..., description="Type of backup")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    
    # Recovery metadata
    milvus_version: Optional[str] = Field(default=None, description="Milvus version at backup time")
    schema_snapshot: Optional[Dict[str, Any]] = Field(default=None, description="Collection schema snapshot")
    row_count: int = Field(default=0, ge=0, description="Total number of rows")
    segment_count: int = Field(default=0, ge=0, description="Number of segments")
    
    # Backup parameters
    compression_enabled: bool = Field(default=False, description="Whether compression was used")
    compression_level: int = Field(default=6, ge=1, le=9, description="Compression level (1-9)")
    include_indexes: bool = Field(default=True, description="Whether indexes were backed up")
    
    # Metadata versioning
    metadata_version: str = Field(default="1.0", description="Metadata format version")
    
    @property
    def is_full_backup(self) -> bool:
        """Check if this is a full collection backup."""
        return self.backup_type == BackupType.FULL_COLLECTION
    
    @property
    def is_partition_backup(self) -> bool:
        """Check if this is a partition-level backup."""
        return self.backup_type == BackupType.PARTITION
    
    @property
    def is_completed(self) -> bool:
        """Check if backup completed successfully."""
        return self.state in (BackupState.COMPLETED, BackupState.VERIFIED)
    
    @property
    def compression_ratio(self) -> Optional[float]:
        """Calculate compression ratio if compression was used."""
        if self.compression_enabled and self.compressed_size_bytes and self.size_bytes > 0:
            return self.compressed_size_bytes / self.size_bytes
        return None
    
    @property
    def size_mb(self) -> float:
        """Get size in megabytes."""
        return self.size_bytes / (1024 * 1024)
    
    @property
    def compressed_size_mb(self) -> Optional[float]:
        """Get compressed size in megabytes."""
        if self.compressed_size_bytes:
            return self.compressed_size_bytes / (1024 * 1024)
        return None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BackupResult(BaseModel):
    """
    Result of a backup operation.
    
    Contains information about the outcome of a backup creation, including
    success status, metadata, timing information, and any errors encountered.
    
    Attributes:
        success: Whether the backup operation succeeded
        backup_id: Unique identifier of the created backup
        backup_name: Name of the created backup
        collection_name: Name of the collection that was backed up
        storage_type: Type of storage used
        storage_path: Path where backup was stored
        size_bytes: Size of the backup in bytes
        execution_time_ms: Time taken to create backup in milliseconds
        state: Final state of the backup
        error_message: Error message if operation failed
        metadata: Full backup metadata (optional, for detailed info)
    
    Example:
        ```python
        result = BackupResult(
            success=True,
            backup_id="550e8400-e29b-41d4-a716-446655440000",
            backup_name="daily_backup",
            collection_name="documents",
            storage_type=BackupStorageType.LOCAL_FILE,
            execution_time_ms=15000.5
        )
        ```
    """
    success: bool = Field(..., description="Whether operation succeeded")
    backup_id: str = Field(..., description="Backup unique identifier")
    backup_name: str = Field(..., description="Backup name")
    collection_name: str = Field(..., description="Collection name")
    storage_type: BackupStorageType = Field(..., description="Storage backend used")
    storage_path: Optional[str] = Field(default=None, description="Path to backup")
    size_bytes: int = Field(default=0, ge=0, description="Backup size in bytes")
    execution_time_ms: float = Field(default=0.0, ge=0.0, description="Execution time in milliseconds")
    state: BackupState = Field(default=BackupState.COMPLETED, description="Final backup state")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Optional[BackupMetadata] = Field(default=None, description="Full backup metadata")
    
    @property
    def size_mb(self) -> float:
        """Get size in megabytes."""
        return self.size_bytes / (1024 * 1024)
    
    @property
    def execution_time_seconds(self) -> float:
        """Get execution time in seconds."""
        return self.execution_time_ms / 1000.0


class RestoreResult(BaseModel):
    """
    Result of a restore operation.
    
    Contains information about the outcome of a restore from backup, including
    success status, validation information, and timing data.
    
    Attributes:
        success: Whether the restore operation succeeded
        backup_id: Identifier of the backup that was restored
        source_collection_name: Original collection name in backup
        target_collection_name: Name of restored collection (may differ from source)
        rows_restored: Number of rows successfully restored
        partitions_restored: List of partition names that were restored
        execution_time_ms: Time taken to restore in milliseconds
        verification_passed: Whether post-restore verification passed
        verification_details: Details from verification process
        error_message: Error message if operation failed
    
    Example:
        ```python
        result = RestoreResult(
            success=True,
            backup_id="550e8400-e29b-41d4-a716-446655440000",
            source_collection_name="documents",
            target_collection_name="documents_restored",
            rows_restored=1000000,
            execution_time_ms=30000.0
        )
        ```
    """
    success: bool = Field(..., description="Whether restore succeeded")
    backup_id: str = Field(..., description="Backup identifier")
    source_collection_name: str = Field(..., description="Original collection name")
    target_collection_name: str = Field(..., description="Target collection name")
    rows_restored: int = Field(default=0, ge=0, description="Number of rows restored")
    partitions_restored: List[str] = Field(default_factory=list, description="Restored partition names")
    execution_time_ms: float = Field(default=0.0, ge=0.0, description="Execution time in milliseconds")
    verification_passed: bool = Field(default=False, description="Whether verification passed")
    verification_details: Optional[Dict[str, Any]] = Field(default=None, description="Verification details")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    
    @property
    def execution_time_seconds(self) -> float:
        """Get execution time in seconds."""
        return self.execution_time_ms / 1000.0


@dataclass
class BackupProgress:
    """
    Real-time progress tracking for backup operations.
    
    Tracks the progress of long-running backup operations, providing information
    about bytes processed, estimated completion time, and current state.
    
    Attributes:
        backup_id: Identifier of the backup being created
        collection_name: Name of the collection being backed up
        state: Current state of the backup operation
        bytes_processed: Number of bytes processed so far
        total_bytes: Total bytes to process (0 if unknown)
        percentage: Progress percentage (0-100)
        start_time: When the backup started
        current_time: Current time
        estimated_completion_time: Estimated time when backup will complete
        estimated_remaining_time_seconds: Estimated seconds remaining
        current_speed_mbps: Current processing speed in MB/s
        error_message: Error message if failed
    
    Example:
        ```python
        progress = BackupProgress(
            backup_id="550e8400-e29b-41d4-a716-446655440000",
            collection_name="documents",
            state=BackupState.IN_PROGRESS,
            bytes_processed=500000000,
            total_bytes=1000000000,
            percentage=50.0
        )
        ```
    """
    backup_id: str
    collection_name: str
    state: BackupState
    bytes_processed: int = 0
    total_bytes: int = 0
    percentage: float = 0.0
    start_time: Optional[datetime] = None
    current_time: Optional[datetime] = None
    estimated_completion_time: Optional[datetime] = None
    estimated_remaining_time_seconds: Optional[float] = None
    current_speed_mbps: Optional[float] = None
    error_message: Optional[str] = None
    
    @property
    def formatted_eta(self) -> Optional[str]:
        """Get formatted ETA string."""
        if self.estimated_remaining_time_seconds is None:
            return None
        
        seconds = int(self.estimated_remaining_time_seconds)
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    @property
    def is_complete(self) -> bool:
        """Check if backup is complete."""
        return self.state in (BackupState.COMPLETED, BackupState.VERIFIED)
    
    @property
    def is_failed(self) -> bool:
        """Check if backup failed."""
        return self.state == BackupState.FAILED


class VerificationResult(BaseModel):
    """
    Result of backup verification operation.
    
    Contains detailed information about backup integrity verification, including
    the verification type, outcome, and any errors or warnings encountered.
    
    Attributes:
        success: Whether verification passed
        backup_id: Identifier of verified backup
        verification_type: Type of verification performed
        checksum_valid: Whether checksum verification passed
        deep_verification_passed: Whether deep verification passed (if performed)
        files_verified: Number of files verified
        files_failed: Number of files that failed verification
        errors: List of errors encountered
        warnings: List of warnings encountered
        verification_time_ms: Time taken to verify in milliseconds
        verified_at: Timestamp when verification was performed
        details: Additional verification details
    
    Example:
        ```python
        result = VerificationResult(
            success=True,
            backup_id="550e8400-e29b-41d4-a716-446655440000",
            verification_type=VerificationType.CHECKSUM,
            checksum_valid=True,
            files_verified=10,
            files_failed=0
        )
        ```
    """
    success: bool = Field(..., description="Whether verification passed")
    backup_id: str = Field(..., description="Backup identifier")
    verification_type: VerificationType = Field(..., description="Type of verification")
    checksum_valid: bool = Field(default=False, description="Checksum verification result")
    deep_verification_passed: Optional[bool] = Field(default=None, description="Deep verification result")
    files_verified: int = Field(default=0, ge=0, description="Number of files verified")
    files_failed: int = Field(default=0, ge=0, description="Number of failed files")
    errors: List[str] = Field(default_factory=list, description="List of errors")
    warnings: List[str] = Field(default_factory=list, description="List of warnings")
    verification_time_ms: float = Field(default=0.0, ge=0.0, description="Verification time in milliseconds")
    verified_at: datetime = Field(default_factory=datetime.now, description="Verification timestamp")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")
    
    @property
    def verification_time_seconds(self) -> float:
        """Get verification time in seconds."""
        return self.verification_time_ms / 1000.0
    
    @property
    def has_errors(self) -> bool:
        """Check if any errors were encountered."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if any warnings were encountered."""
        return len(self.warnings) > 0


class BackupVersion(BaseModel):
    """
    Version information for backup retention management.
    
    Used to track backup versions and determine which backups to retain
    or delete based on retention policies.
    
    Attributes:
        backup_id: Unique backup identifier
        backup_name: Backup name
        collection_name: Collection name
        created_at: Creation timestamp
        size_bytes: Backup size in bytes
        is_verified: Whether backup is verified
        version_number: Sequential version number for this collection
        tags: Optional tags for categorization
    
    Example:
        ```python
        version = BackupVersion(
            backup_id="550e8400-e29b-41d4-a716-446655440000",
            backup_name="daily_backup_001",
            collection_name="documents",
            created_at=datetime.now(),
            size_bytes=1024000000,
            version_number=1
        )
        ```
    """
    backup_id: str = Field(..., description="Backup identifier")
    backup_name: str = Field(..., description="Backup name")
    collection_name: str = Field(..., description="Collection name")
    created_at: datetime = Field(..., description="Creation timestamp")
    size_bytes: int = Field(default=0, ge=0, description="Backup size")
    is_verified: bool = Field(default=False, description="Verification status")
    version_number: int = Field(default=1, ge=1, description="Version number")
    tags: List[str] = Field(default_factory=list, description="Optional tags")
    
    @property
    def age_days(self) -> float:
        """Calculate age of backup in days."""
        now = datetime.now()
        delta = now - self.created_at
        return delta.total_seconds() / 86400.0
    
    @property
    def size_mb(self) -> float:
        """Get size in megabytes."""
        return self.size_bytes / (1024 * 1024)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

