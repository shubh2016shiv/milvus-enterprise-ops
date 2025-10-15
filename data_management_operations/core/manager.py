"""
Core Data Manager

Provides the primary interface for data operations in Milvus collections.
Handles document insertion, updates, and deletion with automatic batching,
validation, and fault tolerance.

Typical usage from external projects:

    from Milvus_Ops.data_management_operations import DataManager, DataOperationConfig
    
    # Create custom configuration
    config = DataOperationConfig(default_batch_size=500)
    
    # Initialize manager
    manager = DataManager(conn_mgr, coll_mgr, config=config)
    
    # Insert documents
    result = await manager.insert_documents(
        collection_name="my_collection",
        documents=docs,
        batch_size=1000
    )
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Set, TypeVar, Generic
import math

from pymilvus.exceptions import MilvusException

# Import from root exceptions using absolute path to avoid circular imports
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from milvus_ops_exceptions import (
    CollectionNotFoundError as BaseCollectionNotFoundError,
    OperationTimeoutError,
    ConnectionError,
    SchemaError
)
from data_management_operations.data_ops_exceptions import (
    InsertionError,
    DataOperationError,
    BatchPartialFailureError,
    TransientOperationError,
    SchemaValidationError,
    DocumentPreparationError,
    CollectionOperationError,
    DeleteOperationError
)
from data_management_operations.data_ops_config import DataOperationConfig
from connection_management import ConnectionManager
from collection_operations import CollectionManager, CollectionSchema
from data_management_operations.models.entities import Document, DocumentBase, BatchOperationResult, DeleteResult, OperationStatus
from data_management_operations.core.validator import DataValidator
from data_management_operations.utils.timing import PerformanceTimer, TimingResult, BatchTimingResult, time_operation
from data_management_operations.utils.retry import retry_on_transient_error, is_transient_milvus_error

logger = logging.getLogger(__name__)

# Type variable for document type, bound to the base document model
T = TypeVar('T', bound=DocumentBase)


class DataManager(Generic[T]):
    """
    Provides a high-level, asynchronous interface for managing data in Milvus collections.
    
    This class serves as the primary entry point for all data-related operations,
    including insertion, upsert, and deletion. It is designed to be robust,
    thread-safe, and scalable, using the ConnectionManager for resilient
    communication with the Milvus server.
    
    All public methods are asynchronous and designed to be used in an asyncio
    event loop, making it suitable for high-concurrency applications.
    """
    
    def __init__(
        self,
        connection_manager: ConnectionManager,
        collection_manager: CollectionManager,
        config: Optional[DataOperationConfig] = None
    ):
        """
        Initialize DataManager with injected dependencies.
        
        This constructor accepts all required dependencies via dependency injection,
        making the class testable and flexible for different deployment scenarios.
        
        Args:
            connection_manager: Manages Milvus connections with retry and circuit breaker logic
            collection_manager: Handles collection metadata and schema operations
            config: Configuration for data operations. If None, uses default settings.
        
        Example:
            ```python
            # With default configuration
            manager = DataManager(conn_mgr, coll_mgr)
            
            # With custom configuration
            config = DataOperationConfig(
                default_batch_size=500,
                retry_transient_errors=True
            )
            manager = DataManager(conn_mgr, coll_mgr, config=config)
            ```
        """
        self._connection_manager = connection_manager
        self._collection_manager = collection_manager
        self._config = config or DataOperationConfig()
        
        # Initialize based on config
        self._default_batch_size = self._config.default_batch_size
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        
        # Initialize performance timer
        self._timer = PerformanceTimer(enable_logging=self._config.enable_timing)
        
        logger.debug(
            f"DataManager initialized with batch_size={self._config.default_batch_size}, "
            f"timing_enabled={self._config.enable_timing}"
        )
    
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
            # Use configured health check timeout if no timeout specified
            check_timeout = timeout or self._config.health_check_timeout
            
            # Check server status
            if not self._connection_manager.check_server_status():
                logger.warning("Milvus server appears unhealthy")
                raise ConnectionError("Milvus server is not responding. Please reinitialize the connection.")
                
        except Exception as e:
            logger.error(f"Connection health check failed: {e}")
            raise ConnectionError(f"Failed to verify Milvus connection health: {e}")
    
    async def _verify_collection_exists_or_create(
        self, 
        collection_name: str, 
        timeout: Optional[float] = None,
        auto_create: bool = False,
        schema: Optional[CollectionSchema] = None
    ) -> None:
        """
        Verify collection exists before data operations, optionally creating it.
        
        This method checks if the specified collection exists and optionally
        creates it if it doesn't exist. It provides a safety mechanism to
        prevent data operations from failing due to missing collections.
        
        Args:
            collection_name: Name of the collection to check
            timeout: Maximum time in seconds to wait for collection check
            auto_create: Whether to automatically create the collection if it doesn't exist
            schema: Schema to use for collection creation (required if auto_create=True)
            
        Raises:
            CollectionOperationError: If collection doesn't exist and auto_create=False
            SchemaError: If auto_create=True but no schema provided
            TransientOperationError: If collection check fails transiently
        """
        try:
            # Check if collection exists
            exists = await self._collection_manager.has_collection(
                collection_name=collection_name,
                strict=True,
                timeout=timeout
            )
            
            if not exists:
                if auto_create:
                    if schema is None:
                        raise SchemaError("Schema is required for automatic collection creation")
                    
                    logger.info(f"Creating collection '{collection_name}' automatically")
                    
                    # Create the collection
                    await self._collection_manager.create_collection(
                        collection_name=collection_name,
                        schema=schema,
                        timeout=timeout
                    )
                    
                    # Load the collection
                    await self._collection_manager.load_collection(
                        collection_name=collection_name,
                        wait=True,
                        timeout=timeout
                    )
                    
                    logger.info(f"Successfully created and loaded collection '{collection_name}'")
                else:
                    raise CollectionOperationError(f"Collection '{collection_name}' does not exist")
                    
        except CollectionOperationError:
            raise  # Re-raise as-is
        except SchemaError:
            raise  # Re-raise as-is
        except Exception as e:
            # Check if this is a transient error
            if is_transient_milvus_error(e):
                raise TransientOperationError(f"Transient error checking collection: {e}")
            
            logger.error(f"Error checking collection '{collection_name}': {e}")
            raise CollectionOperationError(f"Failed to verify collection existence: {e}")

    async def insert(
        self,
        collection_name: str,
        documents: List[Union[T, Dict[str, Any]]],
        batch_size: Optional[int] = None,
        timeout: Optional[float] = None,
        validate: bool = True,
        partition_key: Optional[str] = None,
        auto_create_collection: bool = False,
        collection_schema: Optional[CollectionSchema] = None
    ) -> BatchOperationResult:
        """
        Insert documents into a collection.
        
        This method inserts a list of documents into the specified collection.
        For large lists, documents are inserted in batches to optimize performance
        and memory usage. Each batch is validated against the collection schema
        before insertion to ensure data integrity.
        
        Args:
            collection_name: The name of the collection to insert into
            documents: List of documents to insert
            batch_size: Size of each batch for insertion (default: self._default_batch_size)
            timeout: Maximum time in seconds to wait for each batch insertion
            validate: Whether to validate documents against the collection schema
            partition_key: Optional partition key for routing documents to a specific partition
            auto_create_collection: Whether to automatically create the collection if it doesn't exist
            collection_schema: Schema to use for collection creation (required if auto_create_collection=True)
            
        Returns:
            A BatchOperationResult with details of the insertion operation
            
        Raises:
            CollectionNotFoundError: If the specified collection does not exist
            DataValidationError: If document validation fails and cannot proceed
            InsertionError: If the insertion operation fails
            OperationTimeoutError: If the operation times out
            ConnectionError: If Milvus connection is unhealthy
        """
        if not documents:
            logger.warning(f"No documents provided for insertion into collection '{collection_name}'")
            return BatchOperationResult(
                status=OperationStatus.SUCCESS,
                successful_count=0,
                failed_count=0
            )
        
        # Time the entire insert operation
        async with self._timer.time_operation(
            operation_name="insert_documents",
            metadata={
                "collection_name": collection_name,
                "document_count": len(documents),
                "batch_size": batch_size or self._default_batch_size,
                "validate": validate,
                "partition_key": partition_key
            }
        ) as timing_result:
            # Safety check: Verify Milvus connection is healthy
            await self._verify_connection_health(timeout)
            
            # Validate and normalize batch size
            batch_size = self._config.validate_batch_size(batch_size)
            
            # Acquire collection lock to ensure thread safety
            collection_lock = await self._acquire_collection_lock(collection_name)
            
            # Safety check: Verify collection exists (with optional auto-creation)
            await self._verify_collection_exists_or_create(
                collection_name=collection_name,
                timeout=timeout,
                auto_create=auto_create_collection,
                schema=collection_schema
            )
            
            # Get collection schema for validation
            try:
                description = await self._collection_manager.describe_collection(
                    collection_name=collection_name,
                    timeout=timeout
                )
                schema = description.collection_schema
            except (BaseCollectionNotFoundError, CollectionOperationError) as e:
                error_msg = f"Collection '{collection_name}' does not exist"
                logger.error(f"Error retrieving schema: {error_msg}")
                raise CollectionOperationError(error_msg)
            
            # Validate documents if requested
            if validate:
                validation_result = await DataValidator.validate_documents(documents, schema)
                if not validation_result.is_valid:
                    error_msg = f"Document validation failed for collection '{collection_name}'"
                    logger.error(f"{error_msg}: {validation_result.errors}")
                    raise SchemaValidationError(error_msg, validation_errors=validation_result.errors)
            
            # Calculate number of batches
            num_documents = len(documents)
            num_batches = math.ceil(num_documents / batch_size)
            
            # Initialize result
            result = BatchOperationResult(
                status=OperationStatus.SUCCESS,
                successful_count=0,
                failed_count=0
            )
            
            # Process each batch
            async with collection_lock:
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, num_documents)
                    batch = documents[start_idx:end_idx]
                    
                    try:
                        # Prepare batch for insertion
                        insert_data = DataValidator.prepare_documents_for_insertion(batch, schema)
                        
                        # Execute insertion operation
                        batch_result = await self._connection_manager.execute_operation_async(
                            lambda alias: self._insert_internal(
                                alias=alias,
                                collection_name=collection_name,
                                data=insert_data,
                                partition_key=partition_key
                            ),
                            timeout=timeout
                        )
                        
                        # Update result with batch results
                        result.successful_count += len(batch)
                        result.inserted_ids.extend(batch_result)
                        
                    except Exception as e:
                        logger.error(
                            f"[insert] Error inserting batch {i+1}/{num_batches} into collection '{collection_name}': {e}"
                        )
                        
                        # Update result with failed batch
                        result.failed_count += len(batch)
                        
                        # Add error messages for each document in the batch
                        for j, doc in enumerate(batch):
                            doc_id = getattr(doc, "id", f"{start_idx + j}")
                            result.error_messages[doc_id] = str(e)
                        
                        # Set status to PARTIAL if some documents were successfully inserted
                        if result.successful_count > 0:
                            result.status = OperationStatus.PARTIAL
                        else:
                            result.status = OperationStatus.FAILED
                            if isinstance(e, (InsertionError, OperationTimeoutError)):
                                raise
                            raise InsertionError(f"Failed to insert documents: {e}")
            
            # Set final status
            if result.failed_count > 0:
                result.status = OperationStatus.PARTIAL
            
            logger.info(
                f"[insert] Inserted {result.successful_count}/{num_documents} documents into collection '{collection_name}'"
            )
            return result
    
    def _insert_internal(
        self,
        alias: str,
        collection_name: str,
        data: List[Dict[str, Any]],
        partition_key: Optional[str] = None
    ) -> List[Any]:
        """
        Internal helper method to execute the insertion via the PyMilvus SDK.
        
        Args:
            alias: The connection alias to use
            collection_name: The name of the collection to insert into
            data: List of entity dictionaries to insert
            partition_key: Optional partition key for routing
            
        Returns:
            List of inserted IDs
        """
        from pymilvus import Collection
        
        # Get the collection
        collection = Collection(name=collection_name, using=alias)
        
        # Insert data
        partition_name = None
        if partition_key is not None:
            partition_name = partition_key
        
        try:
            insert_result = collection.insert(data, partition_name=partition_name)
            return insert_result.primary_keys
        except MilvusException as e:
            logger.error(f"[_insert_internal] Milvus error during insertion in '{collection_name}': {e}")
            raise InsertionError(f"Milvus error during insertion: {e}")
        except Exception as e:
            logger.error(f"[_insert_internal] Unexpected error during insertion in '{collection_name}': {e}")
            raise InsertionError(f"Unexpected error during insertion: {e}")
    
    async def upsert(
        self,
        collection_name: str,
        documents: List[T],
        batch_size: Optional[int] = None,
        timeout: Optional[float] = None,
        validate: bool = True,
        partition_key: Optional[str] = None,
        auto_create_collection: bool = False,
        collection_schema: Optional[CollectionSchema] = None
    ) -> BatchOperationResult:
        """
        Insert or update documents in a collection.
        
        This method performs an upsert operation, which inserts documents if they
        don't exist or updates them if they do. Like insert, documents are processed
        in batches for large lists to optimize performance and memory usage.
        
        Args:
            collection_name: The name of the collection to upsert into
            documents: List of documents to upsert
            batch_size: Size of each batch for upsert (default: self._default_batch_size)
            timeout: Maximum time in seconds to wait for each batch upsert
            validate: Whether to validate documents against the collection schema
            partition_key: Optional partition key for routing documents to a specific partition
            auto_create_collection: Whether to automatically create the collection if it doesn't exist
            collection_schema: Schema to use for collection creation (required if auto_create_collection=True)
            
        Returns:
            A BatchOperationResult with details of the upsert operation
            
        Raises:
            CollectionNotFoundError: If the specified collection does not exist
            DataValidationError: If document validation fails and cannot proceed
            InsertionError: If the upsert operation fails
            OperationTimeoutError: If the operation times out
            ConnectionError: If Milvus connection is unhealthy
        """
        if not documents:
            logger.warning(f"No documents provided for upsert into collection '{collection_name}'")
            return BatchOperationResult(
                status=OperationStatus.SUCCESS,
                successful_count=0,
                failed_count=0
            )
        
        # Time the entire upsert operation
        async with self._timer.time_operation(
            operation_name="upsert_documents",
            metadata={
                "collection_name": collection_name,
                "document_count": len(documents),
                "batch_size": batch_size or self._default_batch_size,
                "validate": validate,
                "partition_key": partition_key
            }
        ) as timing_result:
            # Safety check: Verify Milvus connection is healthy
            await self._verify_connection_health(timeout)
            
            # Validate and normalize batch size
            batch_size = self._config.validate_batch_size(batch_size)
            
            # Acquire collection lock to ensure thread safety
            collection_lock = await self._acquire_collection_lock(collection_name)
            
            # Safety check: Verify collection exists (with optional auto-creation)
            await self._verify_collection_exists_or_create(
                collection_name=collection_name,
                timeout=timeout,
                auto_create=auto_create_collection,
                schema=collection_schema
            )
            
            # Get collection schema for validation
            try:
                description = await self._collection_manager.describe_collection(
                    collection_name=collection_name,
                    timeout=timeout
                )
                schema = description.collection_schema
            except (BaseCollectionNotFoundError, CollectionOperationError) as e:
                error_msg = f"Collection '{collection_name}' does not exist"
                logger.error(f"Error retrieving schema: {error_msg}")
                raise CollectionOperationError(error_msg)
            
            # Validate documents if requested
            if validate:
                validation_result = await DataValidator.validate_documents(documents, schema)
                if not validation_result.is_valid:
                    error_msg = f"Document validation failed for collection '{collection_name}'"
                    logger.error(f"{error_msg}: {validation_result.errors}")
                    raise SchemaValidationError(error_msg, validation_errors=validation_result.errors)
            
            # Calculate number of batches
            num_documents = len(documents)
            num_batches = math.ceil(num_documents / batch_size)
            
            # Initialize result
            result = BatchOperationResult(
                status=OperationStatus.SUCCESS,
                successful_count=0,
                failed_count=0
            )
            
            # Process each batch
            async with collection_lock:
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, num_documents)
                    batch = documents[start_idx:end_idx]
                    
                    try:
                        # Prepare batch for upsert
                        upsert_data = DataValidator.prepare_documents_for_insertion(batch, schema)
                        
                        # Execute upsert operation
                        batch_result = await self._connection_manager.execute_operation_async(
                            lambda alias: self._upsert_internal(
                                alias=alias,
                                collection_name=collection_name,
                                data=upsert_data,
                                partition_key=partition_key
                            ),
                            timeout=timeout
                        )
                        
                        # Update result with batch results
                        result.successful_count += len(batch)
                        result.inserted_ids.extend(batch_result)
                        
                    except Exception as e:
                        logger.error(
                            f"[upsert] Error upserting batch {i+1}/{num_batches} into collection '{collection_name}': {e}"
                        )
                        
                        # Update result with failed batch
                        result.failed_count += len(batch)
                        
                        # Add error messages for each document in the batch
                        for j, doc in enumerate(batch):
                            doc_id = getattr(doc, "id", f"{start_idx + j}")
                            result.error_messages[doc_id] = str(e)
                        
                        # Set status to PARTIAL if some documents were successfully upserted
                        if result.successful_count > 0:
                            result.status = OperationStatus.PARTIAL
                        else:
                            result.status = OperationStatus.FAILED
                            if isinstance(e, (InsertionError, OperationTimeoutError)):
                                raise
                            raise InsertionError(f"Failed to upsert documents: {e}")
            
            # Set final status
            if result.failed_count > 0:
                result.status = OperationStatus.PARTIAL
            
            logger.info(
                f"[upsert] Upserted {result.successful_count}/{num_documents} documents into collection '{collection_name}'"
            )
            return result
    
    def _upsert_internal(
        self,
        alias: str,
        collection_name: str,
        data: List[Dict[str, Any]],
        partition_key: Optional[str] = None
    ) -> List[Any]:
        """
        Internal helper method to execute the upsert via the PyMilvus SDK.
        
        Args:
            alias: The connection alias to use
            collection_name: The name of the collection to upsert into
            data: List of entity dictionaries to upsert
            partition_key: Optional partition key for routing
            
        Returns:
            List of upserted IDs
        """
        from pymilvus import Collection
        
        # Get the collection
        collection = Collection(name=collection_name, using=alias)
        
        # Upsert data
        partition_name = None
        if partition_key is not None:
            partition_name = partition_key
        
        try:
            upsert_result = collection.upsert(data, partition_name=partition_name)
            return upsert_result.primary_keys
        except MilvusException as e:
            logger.error(f"[_upsert_internal] Milvus error during upsert in '{collection_name}': {e}")
            raise InsertionError(f"Milvus error during upsert: {e}")
        except Exception as e:
            logger.error(f"[_upsert_internal] Unexpected error during upsert in '{collection_name}': {e}")
            raise InsertionError(f"Unexpected error during upsert: {e}")
    
    async def delete(
        self,
        collection_name: str,
        expr: str,
        timeout: Optional[float] = None,
        partition_key: Optional[str] = None,
        auto_create_collection: bool = False,
        collection_schema: Optional[CollectionSchema] = None
    ) -> DeleteResult:
        """
        Delete documents from a collection based on an expression.
        
        This method deletes documents that match the provided expression from
        the specified collection.
        
        Args:
            collection_name: The name of the collection to delete from
            expr: Expression to match documents to delete (e.g., "id in [1, 2, 3]")
            timeout: Maximum time in seconds to wait for the deletion
            partition_key: Optional partition key to limit deletion to a specific partition
            auto_create_collection: Whether to automatically create the collection if it doesn't exist
            collection_schema: Schema to use for collection creation (required if auto_create_collection=True)
            
        Returns:
            A DeleteResult with details of the deletion operation
            
        Raises:
            CollectionOperationError: If the specified collection does not exist
            DeleteOperationError: If the deletion operation fails
            OperationTimeoutError: If the operation times out
            ConnectionError: If Milvus connection is unhealthy
        """
        # Time the entire delete operation
        async with self._timer.time_operation(
            operation_name="delete_documents",
            metadata={
                "collection_name": collection_name,
                "expression": expr,
                "partition_key": partition_key
            }
        ) as timing_result:
            # Safety check: Verify Milvus connection is healthy
            await self._verify_connection_health(timeout)
            
            # Acquire collection lock to ensure thread safety
            collection_lock = await self._acquire_collection_lock(collection_name)
            
            # Safety check: Verify collection exists (with optional auto-creation)
            await self._verify_collection_exists_or_create(
                collection_name=collection_name,
                timeout=timeout,
                auto_create=auto_create_collection,
                schema=collection_schema
            )
        
            # Execute deletion
            async with collection_lock:
                try:
                    result = await self._connection_manager.execute_operation_async(
                        lambda alias: self._delete_internal(
                            alias=alias,
                            collection_name=collection_name,
                            expr=expr,
                            partition_key=partition_key
                        ),
                        timeout=timeout
                    )
                    
                    logger.info(
                        f"[delete] Deleted {result} documents from collection '{collection_name}' with expression '{expr}'"
                    )
                    
                    return DeleteResult(
                        status=OperationStatus.SUCCESS,
                        deleted_count=result
                    )
                    
                except Exception as e:
                    logger.error(
                        f"Error deleting documents from collection '{collection_name}' with expression '{expr}': {e}"
                    )
                    
                    if isinstance(e, (CollectionOperationError, OperationTimeoutError)):
                        raise
                    
                    raise DeleteOperationError(f"Failed to delete documents: {e}")
    
    def get_timing_history(self) -> List[TimingResult]:
        """
        Get the history of all timing measurements.
        
        Returns:
            List of TimingResult objects for all operations performed
        """
        return self._timer.get_timing_history()
    
    def get_operation_stats(self, operation_name: str) -> Optional[BatchTimingResult]:
        """
        Get statistics for a specific operation type.
        
        Args:
            operation_name: Name of the operation to get stats for
            
        Returns:
            BatchTimingResult with statistics, or None if no operations found
        """
        return self._timer.get_operation_stats(operation_name)
    
    def get_performance_summary(self) -> Dict[str, BatchTimingResult]:
        """
        Get a summary of all operations with their performance statistics.
        
        Returns:
            Dictionary mapping operation names to their statistics
        """
        return self._timer.get_summary()
    
    def clear_timing_history(self):
        """Clear the timing history."""
        self._timer.clear_history()
    
    def _delete_internal(
        self,
        alias: str,
        collection_name: str,
        expr: str,
        partition_key: Optional[str] = None
    ) -> int:
        """
        Internal helper method to execute the deletion via the PyMilvus SDK.
        
        Args:
            alias: The connection alias to use
            collection_name: The name of the collection to delete from
            expr: Expression to match documents to delete
            partition_key: Optional partition key for routing
            
        Returns:
            Number of deleted documents
        """
        from pymilvus import Collection
        
        # Get the collection
        collection = Collection(name=collection_name, using=alias)
        
        # Delete data
        partition_name = None
        if partition_key is not None:
            partition_name = partition_key
        
        try:
            if partition_name:
                delete_result = collection.delete(expr, partition_name=partition_name)
            else:
                delete_result = collection.delete(expr)
            
            return delete_result.delete_count
        except MilvusException as e:
            logger.error(f"Milvus error during deletion in '{collection_name}': {e}")
            raise DeleteOperationError(f"Milvus error during deletion: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during deletion in '{collection_name}': {e}")
            raise DeleteOperationError(f"Unexpected error during deletion: {e}")
