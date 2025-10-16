"""
Backup Validator

Provides validation for backup parameters, storage space, and backup integrity.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any

from pymilvus import Collection, utility
from pymilvus.exceptions import MilvusException

from ..config import BackupRecoveryConfig
from ..models.parameters import BackupParams, RestoreParams
from ..models.entities import BackupMetadata
from ..exceptions import (
    RestoreValidationError,
    InsufficientStorageError,
    PartitionNotFoundError,
    SchemaIncompatibleError
)

logger = logging.getLogger(__name__)


class BackupValidator:
    """
    Validator for backup operations.
    
    Provides validation methods for:
    - Backup parameters before creation
    - Storage space availability
    - Partition existence
    - Schema compatibility for restore
    - Backup integrity
    
    Example:
        ```python
        validator = BackupValidator(config)
        
        # Validate backup parameters
        validator.validate_backup_params(collection, params)
        
        # Check storage space
        validator.validate_storage_space(required_bytes=1000000000)
        
        # Validate restore compatibility
        validator.validate_schema_compatibility(backup_metadata, target_collection)
        ```
    """
    
    def __init__(self, config: BackupRecoveryConfig):
        """
        Initialize backup validator.
        
        Args:
            config: Backup recovery configuration
        """
        self.config = config
        logger.debug("BackupValidator initialized")
    
    def validate_backup_params(
        self,
        collection: Collection,
        params: BackupParams
    ) -> None:
        """
        Validate backup parameters before creating backup.
        
        Args:
            collection: Collection to backup
            params: Backup parameters
        
        Raises:
            ValueError: If parameters are invalid
            PartitionNotFoundError: If specified partitions don't exist
        """
        # Validate partition names if partition backup
        if params.is_partition_backup:
            if not params.partition_names:
                raise ValueError("partition_names must be provided for PARTITION backup")
            
            self.validate_partition_names(collection, params.partition_names)
        
        # Validate compression level
        if not 1 <= params.compression_level <= 9:
            raise ValueError(f"compression_level must be between 1 and 9, got {params.compression_level}")
        
        # Validate chunk size
        if params.chunk_size_mb <= 0:
            raise ValueError(f"chunk_size_mb must be positive, got {params.chunk_size_mb}")
        
        logger.debug(f"Backup parameters validated for collection '{collection.name}'")
    
    def validate_restore_params(self, params: RestoreParams) -> None:
        """
        Validate restore parameters.
        
        Args:
            params: Restore parameters
        
        Raises:
            ValueError: If parameters are invalid
        """
        # Basic validation
        if params.target_collection_name and not params.target_collection_name.strip():
            raise ValueError("target_collection_name cannot be empty")
        
        logger.debug("Restore parameters validated")
    
    def validate_collection_state(self, collection: Collection) -> None:
        """
        Ensure collection is in valid state for backup.
        
        Args:
            collection: Collection to check
        
        Raises:
            ValueError: If collection state is invalid
        """
        try:
            # CRITICAL FIX: Use the collection's connection alias for utility calls
            # The connection pool uses aliases like "conn_0", not "default"
            using_alias = collection._using if hasattr(collection, '_using') else "default"

            # Check if collection exists using the same connection
            if not utility.has_collection(collection.name, using=using_alias):
                raise ValueError(f"Collection '{collection.name}' does not exist")

            # Check if collection has data
            stats = collection.num_entities
            logger.debug(f"Collection '{collection.name}' has {stats} entities (using alias '{using_alias}')")

        except MilvusException as e:
            raise ValueError(f"Failed to validate collection state: {e}")

    def validate_storage_space(
        self,
        required_bytes: int,
        storage_path: Optional[Path] = None
    ) -> None:
        """
        Check if sufficient disk space is available.

        Args:
            required_bytes: Required space in bytes
            storage_path: Path to check (uses config default if None)

        Raises:
            InsufficientStorageError: If not enough space available
        """
        if storage_path is None:
            storage_path = Path(self.config.local_backup_root_path)

        try:
            stat = shutil.disk_usage(storage_path)
            available_bytes = stat.free

            if available_bytes < required_bytes:
                raise InsufficientStorageError(
                    f"Insufficient storage space: need {required_bytes / (1024**3):.2f} GB, "
                    f"available {available_bytes / (1024**3):.2f} GB",
                    required_bytes=required_bytes,
                    available_bytes=available_bytes,
                    storage_path=str(storage_path)
                )

            logger.debug(
                f"Storage validation passed: {available_bytes / (1024**3):.2f} GB available, "
                f"{required_bytes / (1024**3):.2f} GB required"
            )

        except OSError as e:
            logger.warning(f"Failed to check disk space: {e}")
            # Don't fail the operation, just log warning

    def validate_partition_names(
        self,
        collection: Collection,
        partition_names: List[str]
    ) -> None:
        """
        Verify that specified partitions exist in collection.

        Args:
            collection: Collection to check
            partition_names: List of partition names to validate

        Raises:
            PartitionNotFoundError: If any partition doesn't exist
        """
        try:
            # CRITICAL FIX: Use the collection's connection alias for utility calls
            using_alias = collection._using if hasattr(collection, '_using') else "default"
            logger.debug(f"Validating partitions using connection alias: {using_alias}")

            # Get existing partitions using the same connection
            try:
                existing_partitions = [p.name for p in collection.partitions]
            except Exception as e:
                logger.error(f"Failed to get partitions: {e}")
                raise PartitionNotFoundError(
                    f"Failed to get partitions: {e}",
                    collection_name=collection.name
                )

            for partition_name in partition_names:
                if partition_name not in existing_partitions:
                    raise PartitionNotFoundError(
                        f"Partition '{partition_name}' not found in collection '{collection.name}'",
                        collection_name=collection.name,
                        partition_name=partition_name,
                        available_partitions=existing_partitions
                    )

            logger.debug(f"All {len(partition_names)} partitions validated")

        except MilvusException as e:
            raise PartitionNotFoundError(
                f"Failed to validate partitions: {e}",
                collection_name=collection.name
            )

    def validate_schema_compatibility(
        self,
        backup_metadata: BackupMetadata,
        target_schema: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Check schema compatibility between backup and target.

        Args:
            backup_metadata: Backup metadata with source schema
            target_schema: Target collection schema (if exists)

        Raises:
            SchemaIncompatibleError: If schemas are incompatible
        """
        if target_schema is None:
            # No target schema, assume compatible
            return

        backup_schema = backup_metadata.schema_snapshot
        if not backup_schema:
            logger.warning("Backup has no schema snapshot, skipping validation")
            return

        try:
            incompatibilities = []

            # Compare field count
            backup_fields = backup_schema.get("fields", [])
            target_fields = target_schema.get("fields", [])

            if len(backup_fields) != len(target_fields):
                incompatibilities.append(
                    f"Field count mismatch: backup has {len(backup_fields)}, "
                    f"target has {len(target_fields)}"
                )

            # Compare field definitions
            backup_field_dict = {f["name"]: f for f in backup_fields}
            target_field_dict = {f["name"]: f for f in target_fields}

            for field_name in backup_field_dict.keys():
                if field_name not in target_field_dict:
                    incompatibilities.append(f"Field '{field_name}' missing in target")
                    continue

                backup_field = backup_field_dict[field_name]
                target_field = target_field_dict[field_name]

                # Check data type
                if backup_field.get("dtype") != target_field.get("dtype"):
                    incompatibilities.append(
                        f"Field '{field_name}' type mismatch: "
                        f"backup={backup_field.get('dtype')}, target={target_field.get('dtype')}"
                    )

                # Check vector dimension
                if "dim" in backup_field and "dim" in target_field:
                    if backup_field["dim"] != target_field["dim"]:
                        incompatibilities.append(
                            f"Field '{field_name}' dimension mismatch: "
                            f"backup={backup_field['dim']}, target={target_field['dim']}"
                        )

            if incompatibilities:
                raise SchemaIncompatibleError(
                    "Schema compatibility validation failed",
                    backup_id=backup_metadata.backup_id,
                    collection_name=backup_metadata.collection_name,
                    backup_schema=backup_schema,
                    target_schema=target_schema,
                    incompatibilities=incompatibilities
                )

            logger.debug("Schema compatibility validated successfully")

        except SchemaIncompatibleError:
            raise
        except Exception as e:
            logger.warning(f"Schema validation failed with error: {e}")
            # Don't fail the operation, schemas might still be compatible

    def validate_backup_integrity(
        self,
        backup_metadata: BackupMetadata
    ) -> bool:
        """
        Perform basic integrity validation on backup metadata.

        Args:
            backup_metadata: Backup metadata to validate

        Returns:
            True if basic validation passes
        """
        try:
            # Check required fields
            if not backup_metadata.backup_id:
                logger.error("Backup ID is missing")
                return False

            if not backup_metadata.collection_name:
                logger.error("Collection name is missing")
                return False

            if not backup_metadata.storage_path:
                logger.error("Storage path is missing")
                return False

            # Check if backup is verified
            if not backup_metadata.is_verified:
                logger.warning(f"Backup {backup_metadata.backup_id} has not been verified")

            logger.debug(f"Backup integrity validation passed for {backup_metadata.backup_id}")
            return True

        except Exception as e:
            logger.error(f"Backup integrity validation failed: {e}")
            return False

    def estimate_backup_size(self, collection: Collection) -> int:
        """
        Estimate the size of a backup in bytes.

        This is a rough estimate based on entity count and schema.
        Actual size may vary significantly based on data characteristics.

        Args:
            collection: Collection to estimate

        Returns:
            Estimated size in bytes
        """
        try:
            # CRITICAL FIX: Use the collection's connection alias for utility calls
            using_alias = collection._using if hasattr(collection, '_using') else "default"
            logger.debug(f"Estimating backup size using connection alias: {using_alias}")

            # Get entity count safely
            try:
                num_entities = collection.num_entities
            except Exception as e:
                logger.warning(f"Failed to get entity count: {e}, using collection stats")
                try:
                    # Try to get collection stats
                    stats = collection.get_collection_stats()
                    num_entities = stats.get("row_count", 0)
                except Exception as e2:
                    logger.warning(f"Failed to get collection stats: {e2}, assuming 0 entities")
                    num_entities = 0

            # Rough estimate: 1KB per entity (very conservative)
            # Actual size depends on field types, dimensions, etc.
            estimated_size = max(num_entities * 1024, 1024 * 1024)  # At least 1MB
            
            logger.debug(
                f"Estimated backup size for {collection.name}: "
                f"{estimated_size / (1024**2):.2f} MB ({num_entities} entities)"
            )
            
            return estimated_size
            
        except Exception as e:
            logger.warning(f"Failed to estimate backup size: {e}")
            # Return a conservative estimate
            return 1024 * 1024 * 1024  # 1 GB

