"""
Core Index Manager

Provides the primary interface for index operations in Milvus collections.
Handles index creation, management, and monitoring with automatic parameter
validation, progress tracking, and fault tolerance.

Typical usage from external projects:

    from Milvus_Ops.index_operations import IndexManager, IndexOperationConfig, HNSWParams
    
    # Create custom configuration
    config = IndexOperationConfig(default_timeout=120.0)
    
    # Initialize manager
    index_manager = IndexManager(conn_mgr, coll_mgr, config=config)
    
    # Create index with type-safe parameters
    result = await index_manager.create_index(
        collection_name="documents",
        field_name="embedding",
        index_type=IndexType.HNSW,
        metric_type=MetricType.COSINE,
        index_params=HNSWParams(M=16, efConstruction=200)
    )
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple

from pymilvus import Collection
from pymilvus.exceptions import MilvusException

from milvus_ops_exceptions import (
    CollectionNotFoundError as BaseCollectionNotFoundError,
    OperationTimeoutError,
    ConnectionError,
    SchemaError
)
from collection_operations.schema import IndexType, MetricType, DataType, FieldSchema, CollectionSchema
from connection_management import ConnectionManager
from collection_operations import CollectionManager as OpsCollectionManager
from index_operations.config import IndexOperationConfig
from index_operations.index_ops_exceptions import (
    IndexOperationError,
    IndexBuildError,
    IndexNotFoundError,
    IndexParameterError,
    IndexTypeError,
    IndexBuildInProgressError,
    IndexResourceError,
    IndexTimeoutError
)
from index_operations.models.entities import (
    IndexState,
    IndexDescription,
    IndexBuildProgress,
    IndexStats,
    IndexResult
)
from index_operations.models.parameters import (
    IndexParams,
    create_index_params,
    get_default_params
)
from index_operations.core.validator import IndexValidator
from index_operations.utils.progress import IndexBuildTracker, get_registry

logger = logging.getLogger(__name__)


class IndexManager:
    """
    Provides a high-level, asynchronous interface for managing indexes in Milvus collections.
    
    This class serves as the primary entry point for all index-related operations,
    including creation, monitoring, and management. It is designed to be robust,
    thread-safe, and scalable, using the ConnectionManager for resilient
    communication with the Milvus server.
    
    All public methods are asynchronous and designed to be used in an asyncio
    event loop, making it suitable for high-concurrency applications.
    
    Example:
        ```python
        # Initialize with custom configuration
        config = IndexOperationConfig(default_timeout=120.0)
        index_manager = IndexManager(conn_mgr, coll_mgr, config=config)
        
        # Create an index
        result = await index_manager.create_index(
            collection_name="documents",
            field_name="embedding",
            index_type=IndexType.HNSW,
            metric_type=MetricType.COSINE,
            index_params=HNSWParams(M=16, efConstruction=200),
            wait=True
        )
        
        if result.success:
            print("Index created successfully!")
        ```
    """
    
    def __init__(
        self,
        connection_manager: ConnectionManager,
        collection_manager: OpsCollectionManager,
        config: Optional[IndexOperationConfig] = None
    ):
        """
        Initialize IndexManager with injected dependencies.
        
        This constructor accepts all required dependencies via dependency injection,
        making the class testable and flexible for different deployment scenarios.
        
        Args:
            connection_manager: Manages Milvus connections with retry and circuit breaker logic
            collection_manager: Handles collection metadata and schema operations
            config: Configuration for index operations. If None, uses default settings.
        
        Example:
            ```python
            # With default configuration
            index_manager = IndexManager(conn_mgr, coll_mgr)
            
            # With custom configuration
            config = IndexOperationConfig(default_timeout=120.0, enable_timing=True)
            index_manager = IndexManager(conn_mgr, coll_mgr, config=config)
            ```
        """
        self._connection_manager = connection_manager
        self._collection_manager = collection_manager
        self._config = config or IndexOperationConfig()
        
        # Initialize validator
        self._validator = IndexValidator()
        
        # Initialize locks for thread safety
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        
        # Get build tracker registry
        self._tracker_registry = get_registry()
        
        logger.debug(
            f"IndexManager initialized with timeout={self._config.default_timeout}, "
            f"timing_enabled={self._config.enable_timing}"
        )

    async def _ensure_awaited(self, result):
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def _acquire_collection_lock(self, collection_name: str) -> asyncio.Lock:
        """
        Acquire a lock for a specific collection.
        
        This ensures that concurrent operations on the same collection from within
        the same application process are properly serialized, preventing race
        conditions. It uses asyncio.Lock to be compatible with the async nature
        of the manager.
        
        Note: This is in-process locking only. For distributed locking across
        multiple application instances, external coordination (e.g., Redis, etcd)
        would be required.
        
        Args:
            collection_name: The name of the collection for which to acquire a lock
            
        Returns:
            An asyncio.Lock instance specific to the collection name
        """
        async with self._global_lock:
            if collection_name not in self._locks:
                self._locks[collection_name] = asyncio.Lock()
            return self._locks[collection_name]
    
    async def _verify_connection_health(self, timeout: Optional[float] = None) -> None:
        """
        Verify Milvus connection is healthy before operations.
        
        This method performs a health check on the Milvus connection and
        raises an error if the connection is unhealthy. It provides a safety
        mechanism to prevent operations from failing due to stale connections.
        
        Args:
            timeout: Maximum time in seconds to wait for connection health check
            
        Raises:
            ConnectionError: If connection cannot be verified or is unhealthy
        """
        try:
            # Check server status
            if not self._connection_manager.check_server_status():
                logger.warning("Milvus server appears unhealthy")
                raise ConnectionError("Milvus server is not responding. Please reinitialize the connection.")
                
        except Exception as e:
            logger.error(f"Connection health check failed: {e}")
            raise ConnectionError(f"Failed to verify Milvus connection health: {e}")
    
    async def _verify_collection_exists(
        self,
        collection_name: str,
        timeout: Optional[float] = None
    ) -> CollectionSchema:
        """
        Verify collection exists and return its schema.
        
        This method checks if the specified collection exists and retrieves
        its schema for validation purposes.
        
        Args:
            collection_name: Name of the collection to check
            timeout: Maximum time in seconds to wait for collection check
            
        Returns:
            CollectionSchema for the collection
            
        Raises:
            BaseCollectionNotFoundError: If collection doesn't exist
        """
        try:
            # Check if collection exists
            exists = await self._collection_manager.has_collection(
                collection_name=collection_name,
                strict=True,
                timeout=timeout
            )
            
            if not exists:
                raise BaseCollectionNotFoundError(f"Collection '{collection_name}' does not exist")
            
            # Get collection schema
            description = await self._collection_manager.describe_collection(
                collection_name=collection_name,
                timeout=timeout
            )
            
            return description.collection_schema
            
        except BaseCollectionNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error checking collection '{collection_name}': {e}")
            raise BaseCollectionNotFoundError(f"Failed to verify collection existence: {e}")
    
    async def create_index(
        self,
        collection_name: str,
        field_name: str,
        index_type: Union[str, IndexType],
        metric_type: Union[str, MetricType],
        index_params: Optional[Union[Dict[str, Any], IndexParams]] = None,
        index_name: Optional[str] = None,
        timeout: Optional[float] = None,
        wait: bool = False,
        **kwargs
    ) -> IndexResult:
        """
        Create an index on a field in a collection.
        
        This method creates an index on the specified field using the given
        index type and metric. It validates all parameters before creation
        and optionally waits for the build to complete.
        
        Args:
            collection_name: Name of the collection
            field_name: Name of the field to index
            index_type: Type of index to create (e.g., HNSW, IVF_FLAT)
            metric_type: Metric type for similarity search (e.g., L2, COSINE)
            index_params: Parameters for the index (dict or IndexParams)
            index_name: Optional custom name for the index
            timeout: Operation timeout in seconds (uses config default if None)
            wait: Whether to wait for index building to complete
            **kwargs: Additional parameters for the index
            
        Returns:
            IndexResult with operation status
            
        Raises:
            IndexParameterError: If parameters are invalid
            IndexBuildError: If index creation fails
            BaseCollectionNotFoundError: If collection doesn't exist
            ConnectionError: If connection is unhealthy
            IndexBuildInProgressError: If an index is already being built
        
        Example:
            ```python
            # Create HNSW index with custom parameters
            result = await index_manager.create_index(
                collection_name="documents",
                field_name="embedding",
                index_type=IndexType.HNSW,
                metric_type=MetricType.COSINE,
                index_params=HNSWParams(M=16, efConstruction=200),
                wait=True
            )
            
            if result.success:
                print("Index created successfully!")
            ```
        """
        start_time = time.time()
        timeout = timeout or self._config.default_timeout
        
        try:
            # Verify connection health
            await self._verify_connection_health(timeout)
            
            # Acquire collection lock
            collection_lock = await self._acquire_collection_lock(collection_name)
            
            async with collection_lock:
                # Verify collection exists and get schema
                schema = await self._verify_collection_exists(collection_name, timeout)
                
                # Find the field in schema
                field_schema = None
                for field in schema.fields:
                    if field.name == field_name:
                        field_schema = field
                        break
                
                if field_schema is None:
                    raise IndexParameterError(
                        f"Field '{field_name}' does not exist in collection '{collection_name}'",
                        collection_name=collection_name,
                        field_name=field_name
                    )
                
                # Check if field is a vector field
                if field_schema.dtype not in (DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR, DataType.SPARSE_FLOAT_VECTOR):
                    raise IndexParameterError(
                        f"Field '{field_name}' is not a vector field (type: {field_schema.dtype})",
                        collection_name=collection_name,
                        field_name=field_name,
                        parameter_errors={"field_type": f"Not a vector field: {field_schema.dtype}"}
                    )
                
                # Get dimension
                dimension = field_schema.dim if hasattr(field_schema, 'dim') else None
                if dimension is None and field_schema.dtype == DataType.FLOAT_VECTOR:
                    raise IndexParameterError(
                        f"Cannot determine dimension for field '{field_name}'",
                        collection_name=collection_name,
                        field_name=field_name
                    )
                
                # Validate and prepare index parameters
                if index_params is None and kwargs:
                    index_params = kwargs
                
                validated_params = self._validator.validate_index_params(
                    index_type=index_type,
                    metric_type=metric_type,
                    dimension=dimension or 128,  # Use default if sparse vector
                    params=index_params,
                    field_type=field_schema.dtype
                )
                
                # Convert to index type enum if string
                if isinstance(index_type, str):
                    index_type = IndexType(index_type.upper())
                if isinstance(metric_type, str):
                    metric_type = MetricType(metric_type.upper())
                
                # Prepare index parameters for Milvus
                params_dict = validated_params.to_dict() if isinstance(validated_params, IndexParams) else validated_params
                
                # Create index using connection manager
                await self._connection_manager.execute_operation_async(
                    lambda alias: self._create_index_internal(
                        alias=alias,
                        collection_name=collection_name,
                        field_name=field_name,
                        index_type=index_type,
                        metric_type=metric_type,
                        index_name=index_name,
                        params=params_dict
                    ),
                    timeout=timeout
                )
                
                # Register build tracker if we need to monitor progress
                if wait or self._config.enable_timing:
                    tracker = self._tracker_registry.register_build(
                        collection_name=collection_name,
                        field_name=field_name
                    )
                
                # Wait for completion if requested
                if wait:
                    await self._wait_for_index_build(
                        collection_name=collection_name,
                        field_name=field_name,
                        timeout=timeout
                    )
                    state = IndexState.CREATED
                else:
                    state = IndexState.CREATING
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                logger.info(
                    f"Index creation initiated for {collection_name}.{field_name} "
                    f"(type={index_type}, metric={metric_type})"
                )
                
                return IndexResult(
                    success=True,
                    collection_name=collection_name,
                    field_name=field_name,
                    index_name=index_name or f"{field_name}_index",
                    operation="create",
                    state=state,
                    execution_time_ms=execution_time_ms
                )
                
        except (IndexParameterError, IndexBuildError, BaseCollectionNotFoundError, ConnectionError):
            raise
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Failed to create index on {collection_name}.{field_name}: {e}")
            
            return IndexResult(
                success=False,
                collection_name=collection_name,
                field_name=field_name,
                index_name=index_name or f"{field_name}_index",
                operation="create",
                state=IndexState.FAILED,
                error_message=str(e),
                execution_time_ms=execution_time_ms
            )
    
    def _create_index_internal(
        self,
        alias: str,
        collection_name: str,
        field_name: str,
        index_type: IndexType,
        metric_type: MetricType,
        index_name: Optional[str],
        params: Dict[str, Any]
    ) -> None:
        """
        Internal helper method to execute index creation via PyMilvus SDK.
        
        Args:
            alias: Connection alias to use
            collection_name: Name of the collection
            field_name: Name of the field to index
            index_type: Type of index
            metric_type: Metric type
            index_name: Optional index name
            params: Index parameters
        
        Raises:
            IndexBuildError: If index creation fails
        """
        try:
            collection = Collection(name=collection_name, using=alias)
            
            # Prepare index parameters for Milvus
            index_params = {
                "metric_type": metric_type.value,
                "index_type": index_type.value,
                "params": params
            }
            
            # Create the index
            collection.create_index(
                field_name=field_name,
                index_params=index_params,
                index_name=index_name
            )
            
            logger.debug(f"Index created on {collection_name}.{field_name}")
            
        except MilvusException as e:
            logger.error(f"Milvus error during index creation: {e}")
            raise IndexBuildError(
                f"Failed to create index: {str(e)}",
                collection_name=collection_name,
                field_name=field_name,
                index_type=str(index_type)
            )
        except Exception as e:
            logger.error(f"Unexpected error during index creation: {e}")
            raise IndexBuildError(
                f"Unexpected error creating index: {str(e)}",
                collection_name=collection_name,
                field_name=field_name,
                index_type=str(index_type)
            )
    
    async def _wait_for_index_build(
        self,
        collection_name: str,
        field_name: str,
        timeout: Optional[float] = None
    ) -> None:
        """
        Wait for index build to complete.
        
        This method polls the index build progress until it completes
        or times out.
        
        Args:
            collection_name: Name of the collection
            field_name: Name of the field
            timeout: Maximum time to wait in seconds
            
        Raises:
            IndexTimeoutError: If build doesn't complete within timeout
            IndexBuildError: If build fails
        """
        start_time = time.time()
        poll_interval = self._config.build_progress_poll_interval
        
        while True:
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise IndexTimeoutError(
                    f"Index build timed out after {timeout} seconds",
                    operation="create_index",
                    timeout_seconds=timeout
                )
            
            # Get progress
            progress = await self.get_index_build_progress(
                collection_name=collection_name,
                field_name=field_name
            )
            
            # Check if complete
            if progress.state == IndexState.CREATED:
                logger.info(f"Index build completed for {collection_name}.{field_name}")
                return
            
            # Check if failed
            if progress.state == IndexState.FAILED:
                raise IndexBuildError(
                    f"Index build failed: {progress.failed_reason}",
                    collection_name=collection_name,
                    field_name=field_name
                )
            
            # Wait before next poll
            await asyncio.sleep(poll_interval)
    
    async def describe_index(
        self,
        collection_name: str,
        field_name: str,
        timeout: Optional[float] = None
    ) -> Optional[IndexDescription]:
        """
        Get detailed information about an index.
        
        This method retrieves comprehensive information about an index,
        including its configuration, state, and statistics.
        
        Args:
            collection_name: Name of the collection
            field_name: Name of the field
            timeout: Operation timeout in seconds
            
        Returns:
            IndexDescription with comprehensive index information, or None if no index exists.
            
        Raises:
            BaseCollectionNotFoundError: If collection doesn't exist
            ConnectionError: If connection is unhealthy
        
        Example:
            ```python
            index_info = await index_manager.describe_index(
                collection_name="documents",
                field_name="embedding"
            )
            if index_info:
                print(f"Index type: {index_info.index_type}")
                print(f"State: {index_info.state}")
                print(f"Size: {index_info.index_size_mb:.2f} MB")
            ```
        """
        timeout = timeout or self._config.default_timeout
        
        try:
            await self._verify_connection_health(timeout)
            await self._verify_collection_exists(collection_name, timeout)
            
            # The lambda now awaits the async helper function
            index_info = await self._connection_manager.execute_operation_async(
                lambda alias: self._describe_index_internal(
                    alias=alias,
                    collection_name=collection_name,
                    field_name=field_name
                ),
                timeout=timeout
            )
            
            return await self._ensure_awaited(index_info)
            
        except (BaseCollectionNotFoundError, ConnectionError):
            raise
        except Exception as e:
            logger.error(f"Failed to describe index on {collection_name}.{field_name}: {e}")
            raise IndexOperationError(f"Failed to describe index: {str(e)}")
    
    def _describe_index_internal(
        self,
        alias: str,
        collection_name: str,
        field_name: str
    ) -> Optional[IndexDescription]:
        """
        Internal helper to get index information.
        
        Args:
            alias: Connection alias
            collection_name: Name of the collection
            field_name: Name of the field
            
        Returns:
            IndexDescription with index information, or None if no index exists
        """
        try:
            collection = Collection(name=collection_name, using=alias)
            
            # Get index information
            indexes = collection.indexes
            
            # Find index for the field
            field_index = None
            for idx in indexes:
                if idx.field_name == field_name:
                    field_index = idx
                    break
            
            if field_index is None:
                return None
            
            # Extract index information
            index_params = field_index.params
            
            return IndexDescription(
                collection_name=collection_name,
                field_name=field_name,
                index_name=field_index.index_name,
                index_type=index_params.get("index_type", "UNKNOWN"),
                metric_type=index_params.get("metric_type", "UNKNOWN"),
                params=index_params.get("params", {}),
                state=IndexState.CREATED,  # If we can describe it, it exists
                created_at=None,  # Milvus doesn't provide creation time
                indexed_rows=None,  # Would need to query collection stats
                index_size_bytes=None  # Would need to query collection stats
            )
            
        except MilvusException as e:
            if "index not found" in str(e).lower():
                return None
            raise IndexOperationError(f"Failed to describe index: {str(e)}")
    
    async def drop_index(
        self,
        collection_name: str,
        field_name: str,
        timeout: Optional[float] = None
    ) -> IndexResult:
        """
        Drop an index from a field.
        
        This method removes an index from the specified field. The operation
        is irreversible, so use with caution.
        
        Args:
            collection_name: Name of the collection
            field_name: Name of the field
            timeout: Operation timeout in seconds
            
        Returns:
            IndexResult with operation status
            
        Raises:
            IndexNotFoundError: If index doesn't exist
            BaseCollectionNotFoundError: If collection doesn't exist
            ConnectionError: If connection is unhealthy
        
        Example:
            ```python
            result = await index_manager.drop_index(
                collection_name="documents",
                field_name="embedding"
            )
            
            if result.success:
                print("Index dropped successfully")
            ```
        """
        start_time = time.time()
        timeout = timeout or self._config.default_timeout
        
        try:
            # Verify connection health
            await self._verify_connection_health(timeout)
            
            # Acquire collection lock
            collection_lock = await self._acquire_collection_lock(collection_name)
            
            async with collection_lock:
                # Verify collection exists
                await self._verify_collection_exists(collection_name, timeout)
                
                # Drop the index
                await self._connection_manager.execute_operation_async(
                    lambda alias: self._drop_index_internal(
                        alias=alias,
                        collection_name=collection_name,
                        field_name=field_name
                    ),
                    timeout=timeout
                )
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                logger.info(f"Index dropped from {collection_name}.{field_name}")
                
                return IndexResult(
                    success=True,
                    collection_name=collection_name,
                    field_name=field_name,
                    operation="drop",
                    state=IndexState.NONE,
                    execution_time_ms=execution_time_ms
                )
                
        except (IndexNotFoundError, BaseCollectionNotFoundError, ConnectionError):
            raise
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Failed to drop index from {collection_name}.{field_name}: {e}")
            
            return IndexResult(
                success=False,
                collection_name=collection_name,
                field_name=field_name,
                operation="drop",
                state=IndexState.FAILED,
                error_message=str(e),
                execution_time_ms=execution_time_ms
            )
    
    async def _drop_index_internal(
        self,
        alias: str,
        collection_name: str,
        field_name: str
    ) -> None:
        """
        Internal helper to drop an index.
        
        Args:
            alias: Connection alias
            collection_name: Name of the collection
            field_name: Name of the field
            
        Raises:
            IndexNotFoundError: If index doesn't exist
        """
        try:
            collection = Collection(name=collection_name, using=alias)
            collection.drop_index(field_name=field_name)
            
        except MilvusException as e:
            if "index not found" in str(e).lower():
                raise IndexNotFoundError(
                    f"Index not found on {collection_name}.{field_name}",
                    collection_name=collection_name,
                    field_name=field_name
                )
            raise IndexOperationError(f"Failed to drop index: {str(e)}")
    
    async def get_index_build_progress(
        self,
        collection_name: str,
        field_name: str,
        timeout: Optional[float] = None
    ) -> IndexBuildProgress:
        """
        Get the progress of an index build operation.
        
        This method retrieves the current progress of an ongoing or completed
        index build, including percentage complete and estimated time remaining.
        
        Args:
            collection_name: Name of the collection
            field_name: Name of the field being indexed
            timeout: Operation timeout in seconds
            
        Returns:
            IndexBuildProgress with progress information and ETA
            
        Raises:
            IndexNotFoundError: If index doesn't exist
            BaseCollectionNotFoundError: If collection doesn't exist
        
        Example:
            ```python
            progress = await index_manager.get_index_build_progress(
                collection_name="documents",
                field_name="embedding"
            )
            
            print(f"Progress: {progress.percentage:.2f}%")
            if progress.formatted_eta:
                print(f"ETA: {progress.formatted_eta}")
            ```
        """
        timeout = timeout or self._config.default_timeout
        
        try:
            # Check if we have a tracker for this build
            tracker = self._tracker_registry.get_tracker(collection_name, field_name)
            if tracker and not tracker.is_complete():
                return tracker.get_current_progress()
            
            # Otherwise, check index state from Milvus
            try:
                index_desc = await self.describe_index(collection_name, field_name, timeout)
                
                # If index exists and is complete, return 100% progress
                if index_desc and index_desc.state == IndexState.CREATED:
                    return IndexBuildProgress(
                        collection_name=collection_name,
                        field_name=field_name,
                        state=IndexState.CREATED,
                        percentage=100.0,
                        start_time=None,
                        current_time=datetime.now(),
                        estimated_completion_time=None,
                        estimated_remaining_time_seconds=0.0
                    )
            except IndexNotFoundError:
                # This should not happen if _verify_collection_exists is called first
                pass

            # Default to unknown progress if index does not exist or state is not CREATED
            return IndexBuildProgress(
                collection_name=collection_name,
                field_name=field_name,
                state=IndexState.NONE,
                percentage=0.0,
                start_time=None,
                current_time=datetime.now()
            )
            
        except BaseCollectionNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get index build progress for {collection_name}.{field_name}: {e}")
            raise IndexOperationError(f"Failed to get build progress: {str(e)}")
    
    async def list_indexes(
        self,
        collection_name: str,
        timeout: Optional[float] = None
    ) -> List[IndexDescription]:
        """
        List all indexes in a collection.
        
        This method retrieves information about all indexes defined
        on the specified collection.
        
        Args:
            collection_name: Name of the collection
            timeout: Operation timeout in seconds
            
        Returns:
            List of IndexDescription for all indexes in the collection
            
        Raises:
            BaseCollectionNotFoundError: If collection doesn't exist
            ConnectionError: If connection is unhealthy
        
        Example:
            ```python
            indexes = await index_manager.list_indexes(collection_name="documents")
            
            for index in indexes:
                print(f"Field: {index.field_name}, Type: {index.index_type}")
            ```
        """
        timeout = timeout or self._config.default_timeout
        
        try:
            # Verify connection health
            await self._verify_connection_health(timeout)
            
            # Verify collection exists
            schema = await self._verify_collection_exists(collection_name, timeout)
            
            # Get all indexes
            indexes = []
            for field in schema.fields:
                # Only check vector fields
                if field.dtype in (DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR, DataType.SPARSE_FLOAT_VECTOR):
                    try:
                        index_desc = await self.describe_index(collection_name, field.name, timeout)
                        if index_desc:
                            indexes.append(index_desc)
                    except IndexNotFoundError:
                        # Field doesn't have an index, skip it
                        continue
            
            return indexes
            
        except BaseCollectionNotFoundError:
            raise
        except ConnectionError:
            raise
        except Exception as e:
            logger.error(f"Failed to list indexes for {collection_name}: {e}")
            raise IndexOperationError(f"Failed to list indexes: {str(e)}")
    
    async def has_index(
        self,
        collection_name: str,
        field_name: str,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Check if a field has an index.
        
        This is a convenience method that returns True if the field
        has an index, False otherwise.
        
        Args:
            collection_name: Name of the collection
            field_name: Name of the field
            timeout: Operation timeout in seconds
            
        Returns:
            True if field has an index, False otherwise
        
        Example:
            ```python
            if await index_manager.has_index("documents", "embedding"):
                print("Field has an index")
            else:
                print("Field does not have an index")
            ```
        """
        try:
            index_info = await self.describe_index(collection_name, field_name, timeout)
            return index_info is not None
        except BaseCollectionNotFoundError:
             # If collection doesn't exist, it can't have an index
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred in has_index for {collection_name}.{field_name}: {e}")
            # For other errors, it's safer to return False or re-raise
            # depending on desired behavior. Re-raising is safer.
            raise
