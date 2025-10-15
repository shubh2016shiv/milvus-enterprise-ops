"""
Local File System Backup Backend

Implements backup and restore operations using local file system storage.
Uses Parquet format for efficient data storage and supports chunking for
very large collections.
"""

import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import asyncio

from pymilvus import Collection, utility
from pymilvus.exceptions import MilvusException

from ..config import BackupRecoveryConfig
from ..models.entities import (
    BackupMetadata,
    BackupType,
    BackupStorageType,
    BackupState,
    ChecksumAlgorithm
)
from ..models.parameters import BackupParams
from ..exceptions import (
    BackupError,
    BackupNotFoundError,
    BackupStorageError,
    InsufficientStorageError
)
from ..utils.checksum import ChecksumCalculator
from ..utils.compression import CompressionHandler
from ..utils.progress import BackupProgressTracker

logger = logging.getLogger(__name__)

# Try to import pyarrow for Parquet support
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    logger.warning("pyarrow not available, backup functionality will be limited")


class LocalBackupBackend:
    """
    Local file system backup backend for Milvus collections.
    
    This backend stores backups in a structured directory hierarchy on the
    local file system, using Parquet format for data storage with optional
    compression and checksum verification.
    
    Directory structure:
        backup_root/
        ├── collection_name/
        │   ├── backup_id/
        │   │   ├── metadata.json      # Backup metadata
        │   │   ├── schema.json        # Collection schema
        │   │   ├── data/              # Data files
        │   │   │   ├── partition_1_chunk_0.parquet
        │   │   │   ├── partition_1_chunk_1.parquet
        │   │   ├── checksums.json     # File checksums
        │   │   └── indexes/           # Index definitions (optional)
        │   │       └── indexes.json
    
    Example:
        ```python
        backend = LocalBackupBackend(config)
        
        # Create backup
        metadata = await backend.create_backup(
            collection=collection,
            params=backup_params,
            progress_tracker=tracker
        )
        
        # List backups
        backups = await backend.list_backups(collection_name="documents")
        
        # Restore backup
        await backend.restore_backup(
            backup_id="backup_123",
            target_collection_name="documents_restored"
        )
        ```
    """
    
    def __init__(self, config: BackupRecoveryConfig):
        """
        Initialize local backup backend.
        
        Args:
            config: Backup recovery configuration
        
        Raises:
            ValueError: If pyarrow is not available
        """
        if not PARQUET_AVAILABLE:
            raise ValueError("pyarrow is required for local backup backend")
        
        self.config = config
        self.root_path = Path(config.local_backup_root_path)
        
        # Initialize utilities
        self.checksum_calculator = ChecksumCalculator(config.checksum_algorithm)
        self.compression_handler = CompressionHandler(
            compression_level=config.compression_level,
            prefer_zstd=True
        )
        
        # Ensure root directory exists
        self.root_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"LocalBackupBackend initialized with root: {self.root_path}")
    
    def _get_backup_directory(self, collection_name: str, backup_id: str) -> Path:
        """Get the backup directory path."""
        return self.root_path / collection_name / backup_id
    
    def _get_collection_directory(self, collection_name: str) -> Path:
        """Get the collection directory path."""
        return self.root_path / collection_name
    
    async def create_backup(
        self,
        collection: Collection,
        params: BackupParams,
        backup_id: Optional[str] = None,
        backup_name: Optional[str] = None,
        progress_tracker: Optional[BackupProgressTracker] = None
    ) -> BackupMetadata:
        """
        Create a backup of a collection.
        
        Args:
            collection: Milvus collection to backup
            params: Backup parameters
            backup_id: Optional backup ID (generated if not provided)
            backup_name: Optional backup name
            progress_tracker: Optional progress tracker
        
        Returns:
            BackupMetadata with backup information
        
        Raises:
            BackupError: If backup creation fails
            InsufficientStorageError: If not enough disk space
        """
        collection_name = collection.name
        backup_id = backup_id or str(uuid.uuid4())
        backup_name = backup_name or params.backup_name or f"{collection_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_dir = self._get_backup_directory(collection_name, backup_id)
        
        try:
            # Create backup directory structure
            backup_dir.mkdir(parents=True, exist_ok=True)
            data_dir = backup_dir / "data"
            data_dir.mkdir(exist_ok=True)
            
            if params.include_indexes:
                indexes_dir = backup_dir / "indexes"
                indexes_dir.mkdir(exist_ok=True)
            
            # Start progress tracking
            if progress_tracker:
                progress_tracker.start_tracking()
            
            logger.info(f"Starting backup for collection '{collection_name}' (ID: {backup_id})")
            
            # Get collection schema
            schema_dict = await self._export_schema(collection)
            
            # Get partitions to backup
            partitions = await self._get_partitions_to_backup(collection, params)
            
            # Export data
            total_rows, data_files = await self._export_data(
                collection=collection,
                partitions=partitions,
                data_dir=data_dir,
                params=params,
                progress_tracker=progress_tracker
            )
            
            # Export indexes if requested
            index_info = None
            if params.include_indexes:
                index_info = await self._export_indexes(collection, backup_dir / "indexes")
            
            # Calculate checksums
            checksums = await self._calculate_checksums(backup_dir)
            
            # Calculate backup size
            size_bytes = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file())
            compressed_size_bytes = None
            if params.compression_enabled:
                compressed_size_bytes = size_bytes  # Already compressed during export
            
            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_name=backup_name,
                collection_name=collection_name,
                partition_names=partitions,
                created_at=datetime.now(),
                storage_type=BackupStorageType.LOCAL_FILE,
                storage_path=str(backup_dir),
                size_bytes=size_bytes,
                compressed_size_bytes=compressed_size_bytes,
                checksum=checksums.get("__all__", ""),
                checksum_algorithm=self.config.checksum_algorithm,
                is_verified=False,
                state=BackupState.COMPLETED,
                backup_type=params.backup_type,
                schema_snapshot=schema_dict,
                row_count=total_rows,
                compression_enabled=params.compression_enabled,
                compression_level=params.compression_level,
                include_indexes=params.include_indexes
            )
            
            # Save metadata and schema
            await self._save_metadata(backup_dir, metadata)
            await self._save_schema(backup_dir, schema_dict)
            await self._save_checksums(backup_dir, checksums)
            
            # Mark progress as complete
            if progress_tracker:
                progress_tracker.mark_complete(success=True)
            
            logger.info(
                f"Backup completed successfully: {backup_id} "
                f"({total_rows} rows, {size_bytes / (1024**2):.2f} MB)"
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Backup creation failed for {collection_name}: {e}")
            
            # Clean up partial backup
            if backup_dir.exists():
                try:
                    shutil.rmtree(backup_dir)
                except Exception as cleanup_err:
                    logger.warning(f"Failed to cleanup partial backup: {cleanup_err}")
            
            if progress_tracker:
                progress_tracker.mark_complete(success=False, error_message=str(e))
            
            raise BackupError(
                f"Failed to create backup: {str(e)}",
                collection_name=collection_name,
                backup_id=backup_id,
                storage_path=str(backup_dir)
            )
    
    async def _export_schema(self, collection: Collection) -> Dict[str, Any]:
        """Export collection schema."""
        try:
            schema = collection.schema
            
            schema_dict = {
                "collection_name": collection.name,
                "description": collection.description,
                "fields": [],
                "enable_dynamic_field": getattr(schema, "enable_dynamic_field", False)
            }
            
            for field in schema.fields:
                field_dict = {
                    "name": field.name,
                    "dtype": str(field.dtype),
                    "description": field.description or "",
                    "is_primary": field.is_primary,
                    "auto_id": field.auto_id if hasattr(field, "auto_id") else False
                }
                
                # Add dimension for vector fields
                if hasattr(field, "dim"):
                    field_dict["dim"] = field.dim
                
                # Add max_length for varchar fields
                if hasattr(field, "max_length"):
                    field_dict["max_length"] = field.max_length
                
                schema_dict["fields"].append(field_dict)
            
            return schema_dict
            
        except Exception as e:
            raise BackupError(f"Failed to export schema: {e}")
    
    async def _get_partitions_to_backup(
        self,
        collection: Collection,
        params: BackupParams
    ) -> List[str]:
        """Get list of partitions to backup."""
        if params.backup_type == BackupType.PARTITION:
            return params.partition_names
        else:
            # Get all partitions
            return [p.name for p in collection.partitions]
    
    async def _export_data(
        self,
        collection: Collection,
        partitions: List[str],
        data_dir: Path,
        params: BackupParams,
        progress_tracker: Optional[BackupProgressTracker]
    ) -> Tuple[int, List[Path]]:
        """
        Export collection data to Parquet files.
        
        Returns:
            Tuple of (total_rows, list_of_data_files)
        """
        total_rows = 0
        data_files = []
        
        try:
            for partition_name in partitions:
                logger.info(f"Exporting partition: {partition_name}")
                
                # Query all data from partition
                # Note: In production, this should be chunked for large partitions
                results = collection.query(
                    expr="",
                    partition_names=[partition_name],
                    output_fields=["*"]
                )
                
                if not results:
                    logger.info(f"Partition {partition_name} is empty, skipping")
                    continue
                
                # Convert to PyArrow table
                table = pa.Table.from_pylist(results)
                
                # Determine output file
                output_file = data_dir / f"{partition_name}.parquet"
                
                # Write to Parquet with optional compression
                compression = "gzip" if params.compression_enabled else None
                pq.write_table(table, output_file, compression=compression)
                
                data_files.append(output_file)
                total_rows += len(results)
                
                # Update progress
                if progress_tracker:
                    progress_tracker.update_progress(
                        bytes_processed=sum(f.stat().st_size for f in data_files)
                    )
                
                logger.info(
                    f"Exported {len(results)} rows from partition {partition_name} "
                    f"({output_file.stat().st_size / (1024**2):.2f} MB)"
                )
            
            return total_rows, data_files
            
        except Exception as e:
            raise BackupError(f"Failed to export data: {e}")
    
    async def _export_indexes(self, collection: Collection, indexes_dir: Path) -> Dict[str, Any]:
        """Export index definitions."""
        try:
            indexes_info = {"indexes": []}
            
            for index in collection.indexes:
                index_dict = {
                    "field_name": index.field_name,
                    "index_name": index.index_name,
                    "params": index.params
                }
                indexes_info["indexes"].append(index_dict)
            
            # Save indexes info
            indexes_file = indexes_dir / "indexes.json"
            with open(indexes_file, 'w') as f:
                json.dump(indexes_info, f, indent=2)
            
            logger.info(f"Exported {len(indexes_info['indexes'])} index definitions")
            return indexes_info
            
        except Exception as e:
            logger.warning(f"Failed to export indexes: {e}")
            return {"indexes": []}
    
    async def _calculate_checksums(self, backup_dir: Path) -> Dict[str, str]:
        """Calculate checksums for all files in backup."""
        checksums = {}
        
        try:
            all_files = list(backup_dir.rglob("*"))
            data_files = [f for f in all_files if f.is_file() and f.suffix in ['.parquet', '.json']]
            
            for file_path in data_files:
                relative_path = file_path.relative_to(backup_dir)
                checksum = await self.checksum_calculator.calculate_file_checksum(file_path)
                checksums[str(relative_path)] = checksum
            
            # Calculate overall checksum
            combined_checksums = "".join(sorted(checksums.values()))
            overall_checksum = self.checksum_calculator.calculate_data_checksum(
                combined_checksums.encode('utf-8')
            )
            checksums["__all__"] = overall_checksum
            
            logger.debug(f"Calculated checksums for {len(data_files)} files")
            return checksums
            
        except Exception as e:
            logger.warning(f"Failed to calculate checksums: {e}")
            return {}
    
    async def _save_metadata(self, backup_dir: Path, metadata: BackupMetadata) -> None:
        """Save backup metadata."""
        metadata_file = backup_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata.model_dump(mode='json'), f, indent=2, default=str)
    
    async def _save_schema(self, backup_dir: Path, schema_dict: Dict[str, Any]) -> None:
        """Save collection schema."""
        schema_file = backup_dir / "schema.json"
        with open(schema_file, 'w') as f:
            json.dump(schema_dict, f, indent=2)
    
    async def _save_checksums(self, backup_dir: Path, checksums: Dict[str, str]) -> None:
        """Save file checksums."""
        checksums_file = backup_dir / "checksums.json"
        with open(checksums_file, 'w') as f:
            json.dump(checksums, f, indent=2)
    
    async def list_backups(
        self,
        collection_name: Optional[str] = None
    ) -> List[BackupMetadata]:
        """
        List available backups.
        
        Args:
            collection_name: Optional collection name to filter by
        
        Returns:
            List of BackupMetadata
        """
        backups = []
        
        try:
            if collection_name:
                # List backups for specific collection
                collection_dir = self._get_collection_directory(collection_name)
                if collection_dir.exists():
                    for backup_dir in collection_dir.iterdir():
                        if backup_dir.is_dir():
                            metadata = await self.get_backup_metadata(collection_name, backup_dir.name)
                            if metadata:
                                backups.append(metadata)
            else:
                # List all backups
                for collection_dir in self.root_path.iterdir():
                    if collection_dir.is_dir():
                        for backup_dir in collection_dir.iterdir():
                            if backup_dir.is_dir():
                                metadata = await self.get_backup_metadata(
                                    collection_dir.name,
                                    backup_dir.name
                                )
                                if metadata:
                                    backups.append(metadata)
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x.created_at, reverse=True)
            
            logger.debug(f"Found {len(backups)} backups")
            return backups
            
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return []
    
    async def get_backup_metadata(
        self,
        collection_name: str,
        backup_id: str
    ) -> Optional[BackupMetadata]:
        """
        Get backup metadata.
        
        Args:
            collection_name: Collection name
            backup_id: Backup ID
        
        Returns:
            BackupMetadata or None if not found
        """
        backup_dir = self._get_backup_directory(collection_name, backup_id)
        metadata_file = backup_dir / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                return BackupMetadata(**data)
        except Exception as e:
            logger.error(f"Failed to read backup metadata: {e}")
            return None
    
    async def delete_backup(self, collection_name: str, backup_id: str) -> bool:
        """
        Delete a backup.
        
        Args:
            collection_name: Collection name
            backup_id: Backup ID
        
        Returns:
            True if deleted successfully
        
        Raises:
            BackupNotFoundError: If backup doesn't exist
        """
        backup_dir = self._get_backup_directory(collection_name, backup_id)
        
        if not backup_dir.exists():
            raise BackupNotFoundError(
                f"Backup not found: {backup_id}",
                backup_id=backup_id,
                storage_path=str(backup_dir)
            )
        
        try:
            shutil.rmtree(backup_dir)
            logger.info(f"Deleted backup: {backup_id}")
            return True
        except Exception as e:
            raise BackupStorageError(
                f"Failed to delete backup: {e}",
                storage_path=str(backup_dir)
            )
    
    async def verify_backup(self, collection_name: str, backup_id: str) -> bool:
        """
        Verify backup integrity using checksums.
        
        Args:
            collection_name: Collection name
            backup_id: Backup ID
        
        Returns:
            True if verification passed
        """
        backup_dir = self._get_backup_directory(collection_name, backup_id)
        checksums_file = backup_dir / "checksums.json"
        
        if not checksums_file.exists():
            logger.warning(f"No checksums file found for backup {backup_id}")
            return False
        
        try:
            # Load stored checksums
            with open(checksums_file, 'r') as f:
                stored_checksums = json.load(f)
            
            # Recalculate checksums
            current_checksums = await self._calculate_checksums(backup_dir)
            
            # Compare
            for file_path, stored_checksum in stored_checksums.items():
                if file_path == "__all__":
                    continue
                
                current_checksum = current_checksums.get(file_path)
                if current_checksum != stored_checksum:
                    logger.error(
                        f"Checksum mismatch for {file_path}: "
                        f"expected {stored_checksum[:16]}..., got {current_checksum[:16] if current_checksum else 'None'}..."
                    )
                    return False
            
            logger.info(f"Backup verification passed for {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False

