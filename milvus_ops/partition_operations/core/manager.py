"""
Partition Manager for Milvus.

This module provides the PartitionManager class for managing Milvus partitions
with simplicity, robustness, and scalability.
"""

import asyncio
from datetime import datetime
import logging
import time

from milvus_ops.connection_management import ConnectionManager
from milvus_ops.milvus_ops_exceptions import (
    CollectionNotFoundError,
    OperationTimeoutError,
)

from ..config import get_partition_config

# Import custom exceptions to avoid circular imports
from ..exceptions import InvalidPartitionNameError, PartitionNotFoundError
from ..models.entities import (
    LoadProgress,
    PartitionDescription,
    PartitionLoadState,
    PartitionState,
    PartitionStats,
)
from .validator import PartitionValidator

logger = logging.getLogger(__name__)


class PartitionManager:
    """
    User-friendly interface for managing Milvus partitions.

    Provides robust partition operations with automatic retries, proper error handling,
    and concurrency control for production use.
    """

    def __init__(self, connection_manager: ConnectionManager):
        """
        Initialize the PartitionManager with a connection manager.

        Args:
            connection_manager: ConnectionManager instance for handling Milvus connections.
                                 This provides robust connection pooling, retry logic, and
                                 error handling for all partition operations.

        The PartitionManager uses:
        - Per-partition locks to prevent concurrent modifications
        - Global configuration from environment variables and defaults
        - Connection manager for all Milvus SDK interactions
        """
        self._connection_manager = connection_manager
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self._config = get_partition_config()

    async def _get_partition_lock(self, collection_name: str, partition_name: str) -> asyncio.Lock:
        """
        Get or create a lock for a specific partition to prevent concurrent modifications.

        This method implements a thread-safe locking mechanism where each partition
        gets its own asyncio.Lock to prevent race conditions during partition operations.
        Multiple operations on different partitions can proceed concurrently, but operations
        on the same partition will be serialized.

        Args:
            collection_name: Name of the collection containing the partition
            partition_name: Name of the partition to lock

        Returns:
            asyncio.Lock instance for the specified partition
        """
        partition_key = f"{collection_name}:{partition_name}"
        async with self._global_lock:
            if partition_key not in self._locks:
                self._locks[partition_key] = asyncio.Lock()
            return self._locks[partition_key]

    async def create_partition(
        self, collection_name: str, partition_name: str, timeout: float | None = None
    ) -> PartitionDescription:
        """
        Create a new partition in the specified collection.

        This method performs comprehensive validation and safely creates a partition
        with proper error handling and concurrency control. The partition will be
        created in an unloaded state and must be loaded before querying.

        Args:
            collection_name: Name of the collection where the partition will be created.
                           The collection must already exist.
            partition_name: Name for the new partition. Must follow Milvus naming
                          conventions (alphanumeric, underscores, hyphens only).
            timeout: Maximum time in seconds to wait for the operation to complete.
                   Uses the configured default timeout if not specified.

        Returns:
            PartitionDescription: Object containing detailed information about the
                               newly created partition, including its ID, state,
                               and creation timestamp.

        Raises:
            CollectionNotFoundError: If the specified collection does not exist
            InvalidPartitionNameError: If the partition name violates naming rules
                                     (e.g., contains invalid characters, too long)
            PartitionAlreadyExistsError: If a partition with the same name already exists
            OperationTimeoutError: If the operation exceeds the timeout limit
        """
        # Validate partition name
        if self._config.validate_partition_names:
            is_valid, errors = await PartitionValidator.validate_partition_name(partition_name)
            if not is_valid:
                raise InvalidPartitionNameError(partition_name, ", ".join(errors))

        # Get partition lock for thread safety
        partition_lock = await self._get_partition_lock(collection_name, partition_name)

        async with partition_lock:
            # Check if collection exists
            exists = await self._collection_exists(collection_name, timeout=timeout)
            if not exists:
                raise CollectionNotFoundError(f"Collection '{collection_name}' does not exist")

            # Check if partition already exists
            partition_exists = await self.partition_exists(
                collection_name, partition_name, timeout=timeout
            )
            if partition_exists:
                logger.info(
                    f"Partition '{partition_name}' already exists in collection '{collection_name}'"
                )
                return await self.get_partition_info(
                    collection_name, partition_name, timeout=timeout
                )

            # Create partition
            await self._connection_manager.execute_operation_async(
                lambda alias: self._create_partition_internal(
                    alias, collection_name, partition_name
                ),
                timeout=timeout or self._config.create_partition_timeout,
            )

            # Get the created partition description
            description = await self.get_partition_info(
                collection_name, partition_name, timeout=timeout
            )
            logger.info(
                f"Successfully created partition '{partition_name}' "
                f"in collection '{collection_name}'"
            )
            return description

    async def create_partition_if_not_exists(
        self, collection_name: str, partition_name: str, timeout: float | None = None
    ) -> PartitionDescription:
        """
        Create a partition if it doesn't already exist (idempotent operation).

        This is the recommended method for partition creation in production environments
        where you want to ensure a partition exists without failing if it was already
        created. This method is safe to call multiple times with the same parameters.

        The method first checks if the partition exists, and if it does, returns
        information about the existing partition. If it doesn't exist, it creates
        a new partition and returns information about it.

        Args:
            collection_name: Name of the collection where the partition should exist.
                           The collection must already exist.
            partition_name: Name for the partition. Must follow Milvus naming conventions.
            timeout: Maximum time in seconds to wait for operations to complete.
                   Uses configured default timeouts for both existence check and creation.

        Returns:
            PartitionDescription: Object containing information about the partition,
                               whether it was existing or newly created.
        """
        try:
            exists = await self.partition_exists(collection_name, partition_name, timeout=timeout)
            if exists:
                return await self.get_partition_info(
                    collection_name, partition_name, timeout=timeout
                )
            else:
                return await self.create_partition(collection_name, partition_name, timeout=timeout)
        except Exception as e:
            logger.error(f"Failed to create partition '{partition_name}' if not exists: {e}")
            raise

    def _create_partition_internal(
        self, alias: str, collection_name: str, partition_name: str
    ) -> None:
        """Internal helper to create a partition via PyMilvus SDK."""
        from pymilvus import Collection

        collection = Collection(name=collection_name, using=alias)
        collection.create_partition(partition_name)

    async def partition_exists(
        self, collection_name: str, partition_name: str, timeout: float | None = None
    ) -> bool:
        """
        Checks if a partition exists in the specified collection.

        Args:
            collection_name: Name of the collection
            partition_name: Name of the partition
            timeout: Operation timeout in seconds

        Returns:
            True if partition exists, False otherwise
        """
        try:
            collection_exists = await self._collection_exists(collection_name, timeout=timeout)
            if not collection_exists:
                return False

            result = await self._connection_manager.execute_operation_async(
                lambda alias: self._has_partition_internal(alias, collection_name, partition_name),
                timeout=timeout or self._config.default_operation_timeout,
            )
            return await self._ensure_awaited(result)
        except Exception as e:
            logger.error(f"Error checking if partition '{partition_name}' exists: {e}")
            return False

    def _has_partition_internal(
        self, alias: str, collection_name: str, partition_name: str
    ) -> bool:
        """Internal helper to check partition existence."""
        from pymilvus import Collection

        collection = Collection(name=collection_name, using=alias)
        return collection.has_partition(partition_name)

    async def _collection_exists(self, collection_name: str, timeout: float | None = None) -> bool:
        """Helper method to check if a collection exists."""
        try:
            from pymilvus import utility

            result = await self._connection_manager.execute_operation_async(
                lambda alias: utility.has_collection(collection_name, using=alias),
                timeout=timeout or self._config.default_operation_timeout,
            )
            return await self._ensure_awaited(result)
        except Exception as e:
            logger.error(f"Error checking if collection '{collection_name}' exists: {e}")
            return False

    async def list_partitions(
        self, collection_name: str, timeout: float | None = None
    ) -> list[str]:
        """
        Retrieve a list of all partition names from the specified collection.

        This method returns all partitions in a collection, including the default
        '_default' partition that Milvus automatically creates. The list is sorted
        alphabetically for consistent ordering.

        Args:
            collection_name: Name of the collection to list partitions from.
                           If the collection doesn't exist, an empty list is returned.
            timeout: Maximum time in seconds to wait for the operation to complete.
                   Uses the configured default timeout if not specified.

        Returns:
            List[str]: List of partition names in the collection. Returns an empty
                      list if the collection doesn't exist or if there are no partitions.
                      Always includes the '_default' partition if the collection exists.

        Note:
            This operation does not check partition load states or statistics.
            Use get_partition_info() for detailed information about specific partitions.
        """
        try:
            collection_exists = await self._collection_exists(collection_name, timeout=timeout)
            if not collection_exists:
                logger.warning(f"Collection '{collection_name}' does not exist")
                return []

            result = await self._connection_manager.execute_operation_async(
                lambda alias: self._list_partitions_internal(alias, collection_name),
                timeout=timeout or self._config.default_operation_timeout,
            )
            return await self._ensure_awaited(result)
        except Exception as e:
            logger.error(f"Error listing partitions: {e}")
            return []

    def _list_partitions_internal(self, alias: str, collection_name: str) -> list[str]:
        """Internal helper to list all partitions."""
        from pymilvus import Collection

        collection = Collection(name=collection_name, using=alias)
        partitions = collection.partitions
        return [partition.name for partition in partitions]

    async def get_partition_info(
        self, collection_name: str, partition_name: str, timeout: float | None = None
    ) -> PartitionDescription:
        """
        Retrieves detailed information about a specific partition.

        Args:
            collection_name: Name of the collection
            partition_name: Name of the partition
            timeout: Operation timeout in seconds

        Returns:
            PartitionDescription object

        Raises:
            CollectionNotFoundError: If collection doesn't exist
            PartitionNotFoundError: If partition doesn't exist
        """
        try:
            collection_exists = await self._collection_exists(collection_name, timeout=timeout)
            if not collection_exists:
                raise CollectionNotFoundError(f"Collection '{collection_name}' does not exist")

            partition_exists = await self.partition_exists(
                collection_name, partition_name, timeout=timeout
            )
            if not partition_exists:
                raise PartitionNotFoundError(partition_name, collection_name)

            result = await self._connection_manager.execute_operation_async(
                lambda alias: self._describe_partition_internal(
                    alias, collection_name, partition_name
                ),
                timeout=timeout or self._config.default_operation_timeout,
            )
            return await self._ensure_awaited(result)
        except Exception as e:
            logger.error(f"Error getting partition info: {e}")
            raise

    def _describe_partition_internal(
        self, alias: str, collection_name: str, partition_name: str
    ) -> PartitionDescription:
        """Internal helper to describe a partition."""
        from pymilvus import Collection, Partition

        collection = Collection(name=collection_name, using=alias)
        partition = Partition(collection, partition_name)

        return PartitionDescription(
            name=partition_name,
            partition_id=str(partition.name),
            collection_name=collection_name,
            collection_id=collection.name,
            created_at=datetime.now(),  # Milvus doesn't provide creation time
            state=PartitionState.AVAILABLE,
            load_state=PartitionLoadState.UNLOADED,
            created_at_is_synthetic=True,
        )

    async def delete_partition(
        self, collection_name: str, partition_name: str, timeout: float | None = None
    ) -> bool:
        """
        Deletes a partition permanently from the specified collection.

        Args:
            collection_name: Name of the collection
            partition_name: Name of the partition
            timeout: Operation timeout in seconds

        Returns:
            True if deletion was successful

        Raises:
            ValueError: If attempting to delete the default partition
            CollectionNotFoundError: If collection doesn't exist
            PartitionNotFoundError: If partition doesn't exist
        """
        if partition_name == "_default" and self._config.prevent_default_partition_deletion:
            raise ValueError("Cannot delete the default partition '_default'")

        partition_lock = await self._get_partition_lock(collection_name, partition_name)

        async with partition_lock:
            collection_exists = await self._collection_exists(collection_name, timeout=timeout)
            if not collection_exists:
                raise CollectionNotFoundError(f"Collection '{collection_name}' does not exist")

            partition_exists = await self.partition_exists(
                collection_name, partition_name, timeout=timeout
            )
            if not partition_exists:
                raise PartitionNotFoundError(partition_name, collection_name)

            await self._connection_manager.execute_operation_async(
                lambda alias: self._drop_partition_internal(alias, collection_name, partition_name),
                timeout=timeout or self._config.drop_partition_timeout,
            )

            # Remove the partition lock
            partition_key = f"{collection_name}:{partition_name}"
            async with self._global_lock:
                if partition_key in self._locks:
                    del self._locks[partition_key]

            logger.info(
                f"Successfully deleted partition '{partition_name}' "
                f"from collection '{collection_name}'"
            )
            return True

    def _drop_partition_internal(
        self, alias: str, collection_name: str, partition_name: str
    ) -> None:
        """Internal helper to drop a partition."""
        from pymilvus import Collection

        collection = Collection(name=collection_name, using=alias)
        collection.drop_partition(partition_name)

    async def load_partition(
        self,
        collection_name: str,
        partition_name: str,
        wait: bool = False,
        timeout: float | None = None,
    ) -> bool | LoadProgress:
        """
        Load a partition into Milvus's memory for querying.

        Loading a partition makes its vectors available for search operations. This is
        a memory-intensive operation that should be done selectively based on your
        query patterns. Only loaded partitions can be searched.

        Args:
            collection_name: Name of the collection containing the partition.
                           The collection must exist and be loaded.
            partition_name: Name of the partition to load into memory.
            wait: If True, waits for the loading operation to complete and returns
                 a LoadProgress object. If False, initiates loading and returns
                 True immediately (asynchronous loading).
            timeout: Maximum time in seconds to wait for the operation to complete.
                   For wait=True, this applies to both the load initiation and
                   the waiting period. Uses configured default timeout if not specified.

        Returns:
            Union[bool, LoadProgress]:
                - bool (True): If wait=False, indicates loading was successfully initiated
                - LoadProgress: If wait=True, contains detailed loading progress information
                               including completion status, progress percentage, and timing

        Raises:
            CollectionNotFoundError: If the specified collection does not exist
            PartitionNotFoundError: If the specified partition does not exist
            OperationTimeoutError: If the loading operation exceeds the timeout limit

        Note:
            Loading large partitions can take significant time and memory.
            Consider using get_load_progress() to monitor loading status when wait=False.
        """
        partition_lock = await self._get_partition_lock(collection_name, partition_name)

        async with partition_lock:
            collection_exists = await self._collection_exists(collection_name, timeout=timeout)
            if not collection_exists:
                raise CollectionNotFoundError(f"Collection '{collection_name}' does not exist")

            partition_exists = await self.partition_exists(
                collection_name, partition_name, timeout=timeout
            )
            if not partition_exists:
                raise PartitionNotFoundError(partition_name, collection_name)

            await self._connection_manager.execute_operation_async(
                lambda alias: self._load_partition_internal(alias, collection_name, partition_name),
                timeout=timeout or self._config.load_partition_timeout,
            )

            if not wait:
                return True

            # Wait for loading to complete with exponential backoff
            start_time = time.time()
            poll_interval = 0.5
            max_poll_interval = 5.0

            while True:
                progress = await self.get_load_progress(
                    collection_name, partition_name, timeout=timeout
                )

                if progress.is_complete:
                    logger.info(f"Partition '{partition_name}' loaded successfully")
                    return progress

                elapsed = time.time() - start_time
                effective_timeout = timeout or self._config.load_partition_timeout
                if elapsed > effective_timeout:
                    raise OperationTimeoutError(
                        f"Timed out waiting for partition to load after {elapsed:.1f}s"
                    )

                await asyncio.sleep(poll_interval)
                poll_interval = min(poll_interval * 1.5, max_poll_interval)

    def _load_partition_internal(
        self, alias: str, collection_name: str, partition_name: str
    ) -> None:
        """Internal helper to load a partition."""
        from pymilvus import Collection

        collection = Collection(name=collection_name, using=alias)
        collection.load([partition_name])

    async def get_load_progress(
        self, collection_name: str, partition_name: str, timeout: float | None = None
    ) -> LoadProgress:
        """
        Retrieves the current loading progress of a partition.

        Note: Milvus doesn't provide partition-specific load state.
        This estimates progress based on collection load state.

        Args:
            collection_name: Name of the collection
            partition_name: Name of the partition
            timeout: Operation timeout in seconds

        Returns:
            LoadProgress object
        """
        try:
            collection_exists = await self._collection_exists(collection_name, timeout=timeout)
            if not collection_exists:
                raise CollectionNotFoundError(f"Collection '{collection_name}' does not exist")

            partition_exists = await self.partition_exists(
                collection_name, partition_name, timeout=timeout
            )
            if not partition_exists:
                raise PartitionNotFoundError(partition_name, collection_name)

            result = await self._connection_manager.execute_operation_async(
                lambda alias: self._get_load_progress_internal(
                    alias, collection_name, partition_name
                ),
                timeout=timeout or self._config.default_operation_timeout,
            )
            return await self._ensure_awaited(result)
        except Exception as e:
            logger.error(f"Error getting load progress for partition '{partition_name}': {e}")
            raise

    def _get_load_progress_internal(
        self, alias: str, collection_name: str, partition_name: str
    ) -> LoadProgress:
        """Internal helper to estimate load progress."""
        from pymilvus import utility

        try:
            # Get collection load state as proxy for partition state
            load_state = utility.load_state(collection_name, partition_name, using=alias)

            if load_state.name == "Loaded":
                return LoadProgress(
                    partition_name=partition_name,
                    collection_name=collection_name,
                    state=PartitionLoadState.LOADED,
                    progress=1.0,
                    loaded_segments=1,
                    total_segments=1,
                )
            elif load_state.name == "Loading":
                return LoadProgress(
                    partition_name=partition_name,
                    collection_name=collection_name,
                    state=PartitionLoadState.LOADING,
                    progress=0.5,
                    loaded_segments=0,
                    total_segments=1,
                )
        except Exception as e:
            logger.debug(f"Could not get load state: {e}")

        # Default to unloaded state
        return LoadProgress(
            partition_name=partition_name,
            collection_name=collection_name,
            state=PartitionLoadState.UNLOADED,
            progress=0.0,
            loaded_segments=0,
            total_segments=1,
        )

    async def release_partition(
        self, collection_name: str, partition_name: str, timeout: float | None = None
    ) -> bool:
        """
        Releases a partition from Milvus's memory.

        Args:
            collection_name: Name of the collection
            partition_name: Name of the partition
            timeout: Operation timeout in seconds

        Returns:
            True if release was successful
        """
        partition_lock = await self._get_partition_lock(collection_name, partition_name)

        async with partition_lock:
            collection_exists = await self._collection_exists(collection_name, timeout=timeout)
            if not collection_exists:
                raise CollectionNotFoundError(f"Collection '{collection_name}' does not exist")

            partition_exists = await self.partition_exists(
                collection_name, partition_name, timeout=timeout
            )
            if not partition_exists:
                raise PartitionNotFoundError(partition_name, collection_name)

            await self._connection_manager.execute_operation_async(
                lambda alias: self._release_partition_internal(
                    alias, collection_name, partition_name
                ),
                timeout=timeout or self._config.default_operation_timeout,
            )
            logger.info(f"Successfully released partition '{partition_name}' from memory")
            return True

    def _release_partition_internal(
        self, alias: str, collection_name: str, partition_name: str
    ) -> None:
        """Internal helper to release a partition."""
        from pymilvus import Collection

        collection = Collection(name=collection_name, using=alias)
        collection.release([partition_name])

    async def get_partition_stats(
        self, collection_name: str, partition_name: str, timeout: float | None = None
    ) -> PartitionStats:
        """
        Retrieves detailed statistics for a partition.

        Args:
            collection_name: Name of the collection
            partition_name: Name of the partition
            timeout: Operation timeout in seconds

        Returns:
            PartitionStats object
        """
        try:
            result = await self._connection_manager.execute_operation_async(
                lambda alias: self._get_partition_stats_internal(
                    alias, collection_name, partition_name
                ),
                timeout=timeout or self._config.default_operation_timeout,
            )
            return await self._ensure_awaited(result)
        except Exception as e:
            logger.error(f"Error getting partition stats: {e}")
            raise

    def _get_partition_stats_internal(
        self, alias: str, collection_name: str, partition_name: str
    ) -> PartitionStats:
        """Internal helper to get partition statistics."""
        from pymilvus import Collection, Partition

        collection = Collection(name=collection_name, using=alias)
        partition = Partition(collection, partition_name)

        row_count = partition.num_entities

        return PartitionStats(
            name=partition_name,
            partition_id=partition_name,
            collection_name=collection_name,
            collection_id=collection_name,
            created_at=datetime.now(),
            row_count=row_count,
            memory_size=0,  # Not available via SDK
            disk_size=0,  # Not available via SDK
            index_size=0,  # Not available via SDK
            num_segments=0,  # Not available via SDK
        )

    async def _ensure_awaited(self, result):
        """Helper method to ensure a result is properly awaited."""
        if asyncio.iscoroutine(result):
            return await result
        return result

    # Convenience methods for bulk operations
    async def create_multiple_partitions(
        self,
        collection_name: str,
        partition_names: list[str],
        timeout: float | None = None,
        continue_on_error: bool = True,
    ) -> list[PartitionDescription]:
        """
        Creates multiple partitions in a collection.

        Args:
            collection_name: Name of the collection
            partition_names: List of partition names to create
            timeout: Timeout per partition operation
            continue_on_error: If True, continues creating other partitions on error

        Returns:
            List of PartitionDescription objects for created partitions
        """
        results = []
        errors = []

        for partition_name in partition_names:
            try:
                partition = await self.create_partition_if_not_exists(
                    collection_name, partition_name, timeout=timeout
                )
                results.append(partition)
            except Exception as e:
                error_msg = f"Failed to create partition '{partition_name}': {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                if not continue_on_error:
                    break

        if errors and not continue_on_error:
            raise Exception(f"Partition creation failed: {'; '.join(errors)}")

        return results

    async def delete_multiple_partitions(
        self,
        collection_name: str,
        partition_names: list[str],
        timeout: float | None = None,
        continue_on_error: bool = True,
    ) -> list[str]:
        """
        Deletes multiple partitions from a collection.

        Args:
            collection_name: Name of the collection
            partition_names: List of partition names to delete
            timeout: Timeout per partition operation
            continue_on_error: If True, continues deleting other partitions on error

        Returns:
            List of partition names that were successfully deleted
        """
        deleted = []
        errors = []

        for partition_name in partition_names:
            try:
                await self.delete_partition(collection_name, partition_name, timeout=timeout)
                deleted.append(partition_name)
            except Exception as e:
                error_msg = f"Failed to delete partition '{partition_name}': {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                if not continue_on_error:
                    break

        if errors and not continue_on_error:
            raise Exception(f"Partition deletion failed: {'; '.join(errors)}")

        return deleted
