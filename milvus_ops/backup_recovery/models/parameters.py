"""
Backup Recovery Parameters

Defines parameter classes for backup and restore operations, providing
type-safe configuration for various backup scenarios.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from .entities import BackupType, VerificationType


class BackupParams(BaseModel):
    """
    Parameters for creating a backup.
    
    Configures how a backup should be created, including what to backup,
    how to compress it, and what additional metadata to include.
    
    Attributes:
        backup_type: Type of backup to create (FULL_COLLECTION or PARTITION)
        partition_names: List of partition names to backup (required if backup_type is PARTITION)
        compression_enabled: Whether to compress the backup data
        compression_level: Compression level from 1 (fastest) to 9 (best compression)
        chunk_size_mb: Size of chunks for splitting large backups (in MB)
        include_indexes: Whether to include index definitions in the backup
        backup_name: Optional custom name for the backup (auto-generated if not provided)
        tags: Optional tags for categorizing the backup
    
    Example:
        ```python
        # Full collection backup with compression
        params = BackupParams(
            backup_type=BackupType.FULL_COLLECTION,
            compression_enabled=True,
            compression_level=6,
            include_indexes=True
        )
        
        # Partition backup
        params = BackupParams(
            backup_type=BackupType.PARTITION,
            partition_names=["partition_2024_01", "partition_2024_02"],
            compression_enabled=True
        )
        ```
    """
    backup_type: BackupType = Field(
        default=BackupType.FULL_COLLECTION,
        description="Type of backup to create"
    )
    partition_names: List[str] = Field(
        default_factory=list,
        description="List of partitions to backup (for PARTITION backup type)"
    )
    compression_enabled: bool = Field(
        default=True,
        description="Whether to compress backup data"
    )
    compression_level: int = Field(
        default=6,
        ge=1,
        le=9,
        description="Compression level (1=fastest, 9=best compression)"
    )
    chunk_size_mb: int = Field(
        default=256,
        gt=0,
        description="Chunk size in MB for splitting large backups"
    )
    include_indexes: bool = Field(
        default=True,
        description="Whether to include index definitions"
    )
    backup_name: Optional[str] = Field(
        default=None,
        description="Custom name for the backup (auto-generated if not provided)"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Optional tags for categorization"
    )
    
    @field_validator("partition_names")
    @classmethod
    def validate_partition_names(cls, v, info):
        """Ensure partition names are provided for partition backups."""
        backup_type = info.data.get("backup_type")
        if backup_type == BackupType.PARTITION and not v:
            raise ValueError("partition_names must be provided for PARTITION backup type")
        return v
    
    @property
    def is_full_backup(self) -> bool:
        """Check if this is a full collection backup."""
        return self.backup_type == BackupType.FULL_COLLECTION
    
    @property
    def is_partition_backup(self) -> bool:
        """Check if this is a partition backup."""
        return self.backup_type == BackupType.PARTITION
    
    @property
    def chunk_size_bytes(self) -> int:
        """Get chunk size in bytes."""
        return self.chunk_size_mb * 1024 * 1024


class RestoreParams(BaseModel):
    """
    Parameters for restoring from a backup.
    
    Configures how a backup should be restored, including target collection name,
    which partitions to restore, and verification options.
    
    Attributes:
        target_collection_name: Name for the restored collection (uses original name if None)
        partition_names: Specific partitions to restore (all if empty for full backup)
        verify_before_restore: Whether to verify backup integrity before restoring
        drop_existing: Whether to drop existing collection with same name
        load_after_restore: Whether to load collection into memory after restore
        restore_indexes: Whether to restore indexes (if they were backed up)
        skip_failed_partitions: Continue restoring other partitions if one fails
    
    Example:
        ```python
        # Restore with verification
        params = RestoreParams(
            verify_before_restore=True,
            drop_existing=False,
            load_after_restore=True
        )
        
        # Restore to different collection name
        params = RestoreParams(
            target_collection_name="documents_restored",
            verify_before_restore=True,
            drop_existing=False
        )
        
        # Restore specific partitions only
        params = RestoreParams(
            partition_names=["partition_2024_01"],
            verify_before_restore=True
        )
        ```
    """
    target_collection_name: Optional[str] = Field(
        default=None,
        description="Target collection name (uses original if None)"
    )
    partition_names: List[str] = Field(
        default_factory=list,
        description="Specific partitions to restore (empty = all)"
    )
    verify_before_restore: bool = Field(
        default=True,
        description="Verify backup integrity before restoring"
    )
    drop_existing: bool = Field(
        default=False,
        description="Drop existing collection if it exists"
    )
    load_after_restore: bool = Field(
        default=False,
        description="Load collection into memory after restore"
    )
    restore_indexes: bool = Field(
        default=True,
        description="Restore index definitions (if backed up)"
    )
    skip_failed_partitions: bool = Field(
        default=False,
        description="Continue with other partitions if one fails"
    )
    
    @property
    def restore_all_partitions(self) -> bool:
        """Check if all partitions should be restored."""
        return len(self.partition_names) == 0


class VerificationParams(BaseModel):
    """
    Parameters for backup verification.
    
    Configures how backup integrity should be verified, including the type
    of verification and error handling behavior.
    
    Attributes:
        verification_type: Type of verification to perform (CHECKSUM, DEEP, or QUICK)
        sample_size: Number of records to sample for QUICK verification (percentage of total)
        fail_fast: Stop verification on first error encountered
        verify_checksums: Whether to verify file checksums
        verify_schema: Whether to verify schema integrity
        verify_data_integrity: Whether to verify data can be read correctly
        deep_verify_row_count: For deep verification, compare row counts
    
    Example:
        ```python
        # Fast checksum verification
        params = VerificationParams(
            verification_type=VerificationType.CHECKSUM,
            fail_fast=True
        )
        
        # Deep verification with full data check
        params = VerificationParams(
            verification_type=VerificationType.DEEP,
            verify_checksums=True,
            verify_schema=True,
            verify_data_integrity=True
        )
        
        # Quick sampling-based verification
        params = VerificationParams(
            verification_type=VerificationType.QUICK,
            sample_size=10.0,  # Verify 10% of data
            fail_fast=False
        )
        ```
    """
    verification_type: VerificationType = Field(
        default=VerificationType.CHECKSUM,
        description="Type of verification to perform"
    )
    sample_size: float = Field(
        default=10.0,
        gt=0.0,
        le=100.0,
        description="Sample size for QUICK verification (percentage)"
    )
    fail_fast: bool = Field(
        default=True,
        description="Stop on first error"
    )
    verify_checksums: bool = Field(
        default=True,
        description="Verify file checksums"
    )
    verify_schema: bool = Field(
        default=True,
        description="Verify schema integrity"
    )
    verify_data_integrity: bool = Field(
        default=True,
        description="Verify data can be read"
    )
    deep_verify_row_count: bool = Field(
        default=True,
        description="Verify row counts match (for deep verification)"
    )
    
    @property
    def is_checksum_only(self) -> bool:
        """Check if only checksum verification is requested."""
        return self.verification_type == VerificationType.CHECKSUM
    
    @property
    def is_deep_verification(self) -> bool:
        """Check if deep verification is requested."""
        return self.verification_type == VerificationType.DEEP
    
    @property
    def is_quick_verification(self) -> bool:
        """Check if quick verification is requested."""
        return self.verification_type == VerificationType.QUICK

