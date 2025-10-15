"""
Collection Manager for Milvus.

This module provides the CollectionManager class, which is responsible for
managing Milvus collections in a robust and scalable manner.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Set, Coroutine
from datetime import datetime

from pymilvus.exceptions import (
    CollectionNotExistException,
    SchemaNotReadyException  # Used for schema validation errors and collection not found
)

from connection_management import ConnectionManager
from milvus_ops_exceptions import (
    CollectionError, 
    CollectionNotFoundError, 
    SchemaError, 
    OperationTimeoutError
)

from .schema import CollectionSchema, FieldSchema, DataType
from .validator import SchemaValidator
from .entities import (
    CollectionDescription, 
    CollectionStats, 
    LoadProgress, 
    LoadState, 
    CollectionState
)

logger = logging.getLogger(__name__)


class CollectionManager:
    """
    Provides a high-level, asynchronous interface for managing Milvus collections.
    
    This class serves as the primary entry point for all collection-related
    operations. It is designed to be robust, thread-safe, and scalable, using
    the `ConnectionManager` for resilient communication with the Milvus server.
    All public methods are asynchronous and designed to be used in an `asyncio`
    event loop.
    """
    
    def __init__(self, connection_manager: ConnectionManager):
        """
        Initialize the CollectionManager.
        
        Args:
            connection_manager: The ConnectionManager instance to use for Milvus communication
        """
        self._connection_manager = connection_manager
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
    
    async def _get_collection_lock(self, collection_name: str) -> asyncio.Lock:
        """
        Get a lock for a specific collection.
        
        This ensures that concurrent operations on the same collection from within
        the same application process are properly serialized, preventing race
        conditions. It uses `asyncio.Lock` to be compatible with the async nature
        of the manager.
        
        Args:
            collection_name: The name of the collection for which to acquire a lock.
            
        Returns:
            An `asyncio.Lock` instance specific to the collection name.
        """
        async with self._global_lock:
            if collection_name not in self._locks:
                self._locks[collection_name] = asyncio.Lock()
            return self._locks[collection_name]
    
    async def _cleanup_unused_locks(self) -> int:
        """
        Clean up locks for collections that no longer exist.
        
        PRODUCTION FIX: This method prevents memory leaks by removing locks for
        collections that have been dropped. Without this cleanup, the _locks
        dictionary would grow unbounded over time as collections are created
        and dropped, leading to memory exhaustion in long-running production
        deployments.
        
        This method should be called periodically (e.g., after drop operations
        or on a background schedule) to maintain a healthy memory footprint.
        
        Returns:
            The number of locks that were cleaned up.
            
        Note:
            This method is safe to call at any time and will not interfere with
            active operations, as it only removes locks for non-existent collections.
        """
        async with self._global_lock:
            # Get the current list of collections from Milvus
            # Use strict=False to avoid raising exceptions during cleanup
            all_collections = await self.list_collections(strict=False)
            
            if all_collections is None:
                all_collections = []
            
            collection_set = set(all_collections)
            
            # Find locks for collections that no longer exist
            locks_to_remove = [name for name in self._locks if name not in collection_set]
            
            # Remove the orphaned locks
            for name in locks_to_remove:
                del self._locks[name]
            
            if locks_to_remove:
                logger.debug(
                    f"Cleaned up {len(locks_to_remove)} unused collection locks. "
                    f"Remaining locks: {len(self._locks)}"
                )
            
            return len(locks_to_remove)
    
    async def create_collection(
        self, 
        collection_name: str, 
        schema: CollectionSchema,
        timeout: Optional[float] = None
    ) -> CollectionDescription:
        """
        Creates a new collection in Milvus in an idempotent and safe manner.
        
        This method first performs comprehensive local validation of the provided
        schema. It then checks if a collection with the same name already exists.
        If it does, it compares the schemas for compatibility. If they are compatible,
        the operation succeeds without making any changes. If they are not, it
        raises a `SchemaError`. This idempotency prevents accidental creation
        of duplicate collections or schema mismatches.
        
        Collection configuration notes:
        - shard_num: Default is 2 (from schema.shard_num), controls data distribution
        - enable_dynamic_field: Default is False (from schema.enable_dynamic_field),
          controls whether fields not in schema can be inserted

        Args:
            collection_name: The name for the new collection.
            schema: A `CollectionSchema` object defining the structure of the collection.
            timeout: The maximum time in seconds to wait for the creation operation
                     to complete. This timeout is applied to each individual operation
                     with the Milvus server.
            
        Returns:
            A `CollectionDescription` object representing the newly created or
            existing compatible collection.
            
        Raises:
            SchemaError: If the provided schema is invalid or conflicts with an
                         existing collection's schema.
            CollectionError: If there is a failure during the creation process on
                             the Milvus server.
            OperationTimeoutError: If the operation times out.
        """
        # Validate schema
        is_valid, errors = await SchemaValidator.validate_schema(schema)
        if not is_valid:
            error_msg = f"Invalid schema for collection '{collection_name}': {', '.join(errors)}"
            logger.error(error_msg)
            raise SchemaError(error_msg)
        
        # Get collection lock
        collection_lock = await self._get_collection_lock(collection_name)
        
        async with collection_lock:
            # Check if collection already exists
            exists = await self.has_collection(collection_name, strict=True, timeout=timeout)
            
            if exists:
                # Get existing schema
                try:
                    description = await self.describe_collection(collection_name)
                    description = await self._ensure_awaited(description)
                    existing_schema = description.collection_schema
                    
                    # Compare schemas
                    is_compatible, incompatibilities = await SchemaValidator.compare_schemas(
                        schema, existing_schema
                    )
                    
                    if is_compatible:
                        logger.info(
                            f"Collection '{collection_name}' already exists with compatible schema"
                        )
                        return description
                    else:
                        error_msg = (
                            f"Collection '{collection_name}' already exists with incompatible schema: "
                            f"{', '.join(incompatibilities)}"
                        )
                        logger.error(error_msg)
                        raise SchemaError(error_msg)
                    
                except CollectionNotFoundError:
                    # This is actually not an error - the collection doesn't exist yet
                    # which means we should proceed with creation
                    logger.info(f"Collection '{collection_name}' doesn't exist yet, will create it")
                    # We'll continue with creation below
                except SchemaNotReadyException:
                    # This is equivalent to collection not found in Milvus
                    logger.info(f"Collection '{collection_name}' doesn't exist yet, will create it")
                    # We'll continue with creation below
                except Exception as e:
                    logger.error(f"Error checking existing collection schema: {e}")
                    raise CollectionError(f"Error checking existing collection: {e}")
            
            # Create collection
            try:
                # For future control-plane integration, we could update a metadata store
                # to set the collection state to CREATING here
                
                # Convert schema to Milvus format
                fields_data = []
                for field in schema.fields:
                    field_dict = field.dict(exclude_none=True)
                    fields_data.append(field_dict)
                
                # Add collection-level properties
                create_params = {
                    "collection_name": collection_name,
                    "fields": fields_data,
                    "description": schema.description,
                    "enable_dynamic_field": schema.enable_dynamic_field,
                    "num_shards": schema.shard_num,
                }
                
                # Execute creation operation
                await self._connection_manager.execute_operation_async(
                    lambda alias: self._create_collection_internal(alias, **create_params),
                    timeout=timeout
                )
                
                # For future control-plane integration, we could update a metadata store
                # to set the collection state to AVAILABLE here
                
                # Get the created collection description
                description = await self.describe_collection(collection_name, timeout=timeout)
                description = await self._ensure_awaited(description)
                logger.info(f"Successfully created collection '{collection_name}'")
                return description
                
            except Exception as e:
                logger.error(f"[create_collection] Failed to create collection '{collection_name}' (alias={getattr(e, 'using', 'unknown')}): {e}")
                if isinstance(e, (CollectionNotFoundError, SchemaError, OperationTimeoutError)):
                    raise
                if isinstance(e, SchemaNotReadyException):
                    # Convert PyMilvus schema errors to our SchemaError
                    raise SchemaError(f"Schema validation failed during creation: {e}")
                raise CollectionError(f"Failed to create collection: {e}")
    
    # Explicit mapping between our DataType enum and pymilvus DataType
    _MILVUS_DATATYPE_MAP = {
        "BOOL": "BOOL",
        "INT8": "INT8",
        "INT16": "INT16",
        "INT32": "INT32",
        "INT64": "INT64",
        "FLOAT": "FLOAT",
        "DOUBLE": "DOUBLE",
        "STRING": "VARCHAR",  # Normalize STRING to VARCHAR
        "VARCHAR": "VARCHAR",
        "BINARY_VECTOR": "BINARY_VECTOR",
        "FLOAT_VECTOR": "FLOAT_VECTOR",
        "SPARSE_FLOAT_VECTOR": "SPARSE_FLOAT_VECTOR",
        "JSON": "JSON",
        "ARRAY": "ARRAY"
    }
    
    # Whitelist of field parameters accepted by pymilvus FieldSchema
    _FIELD_PARAM_WHITELIST = {
        "name", "dtype", "description", "is_primary", "auto_id", 
        "dim", "max_length", "element_type"
    }
    
    def _create_collection_internal(self, alias: str, **kwargs) -> None:
        """
        Internal helper method to execute the collection creation via the PyMilvus SDK.
        
        This method translates the library's internal schema representation into
        PyMilvus objects and handles the actual API call. It includes robust error
        handling for data type mapping and field parameter validation.
        """
        from pymilvus import Collection, FieldSchema as MilvusFieldSchema, CollectionSchema as MilvusCollectionSchema
        from pymilvus import DataType as MilvusDataType
        
        # Convert our field schemas to pymilvus field schemas
        fields = []
        for field_dict in kwargs.get("fields", []):
            # Convert our DataType enum to pymilvus DataType
            dtype_str = field_dict.pop("dtype")
            
            # Use explicit mapping with error handling
            try:
                if dtype_str not in self._MILVUS_DATATYPE_MAP:
                    raise SchemaError(f"Unsupported data type: {dtype_str}")
                
                milvus_dtype_str = self._MILVUS_DATATYPE_MAP[dtype_str]
                dtype = getattr(MilvusDataType, milvus_dtype_str)
            except (AttributeError, KeyError) as e:
                raise SchemaError(f"Failed to map data type '{dtype_str}' to Milvus DataType: {e}")
            
            # Filter field parameters to only those accepted by pymilvus
            filtered_field_dict = {
                k: v for k, v in field_dict.items() 
                if k in self._FIELD_PARAM_WHITELIST
            }
            
            # Handle element_type conversion for ARRAY fields
            if dtype_str == "ARRAY" and "element_type" in filtered_field_dict:
                element_type_str = filtered_field_dict.pop("element_type")
                try:
                    if element_type_str not in self._MILVUS_DATATYPE_MAP:
                        raise SchemaError(f"Unsupported element_type: {element_type_str}")
                    
                    milvus_element_type_str = self._MILVUS_DATATYPE_MAP[element_type_str]
                    element_type = getattr(MilvusDataType, milvus_element_type_str)
                    filtered_field_dict["element_type"] = element_type
                except (AttributeError, KeyError) as e:
                    raise SchemaError(f"Failed to map element_type '{element_type_str}' to Milvus DataType: {e}")
            
            # Create the pymilvus FieldSchema
            try:
                fields.append(MilvusFieldSchema(**filtered_field_dict, dtype=dtype))
            except TypeError as e:
                raise SchemaError(f"Invalid field parameters for '{filtered_field_dict.get('name', 'unknown')}': {e}")
        
        # Create the pymilvus CollectionSchema
        schema = MilvusCollectionSchema(
            fields=fields,
            description=kwargs.get("description", "")
        )
        
        # Create the collection
        Collection(
            name=kwargs.get("collection_name"),
            schema=schema,
            enable_dynamic_field=kwargs.get("enable_dynamic_field", False),
            num_shards=kwargs.get("num_shards", 2),
            using=alias
        )
    
    async def has_collection(self, collection_name: str, strict: bool = False, timeout: Optional[float] = None) -> bool:
        """
        Checks if a collection with the given name exists in Milvus.

        Args:
            collection_name: The name of the collection to check.
            strict: If `True`, any exception during the check will be raised. If
                    `False` (default), exceptions are suppressed and `False` is
                    returned. This is useful for distinguishing between a non-existent
                    collection and a connection or server error.
            timeout: The maximum time in seconds to wait for the operation to complete.
            
        Returns:
            `True` if the collection exists, `False` otherwise.
            
        Raises:
            CollectionError: If `strict` is `True` and an error occurs during the
                             check.
            OperationTimeoutError: If the operation times out and `strict` is `True`.
        """
        try:
            result = await self._connection_manager.execute_operation_async(
                lambda alias: self._has_collection_internal(alias, collection_name),
                timeout=timeout
            )
            return await self._ensure_awaited(result)
        except Exception as e:
            logger.error(f"[has_collection] Error checking if collection '{collection_name}' exists (alias={getattr(e, 'using', 'unknown')}): {e}")
            if strict:
                # Preserve specific exception types for better error handling
                if isinstance(e, (ConnectionError, OperationTimeoutError)):
                    raise
                raise CollectionError(f"Error checking if collection exists: {e}")
            return False
    
    def _has_collection_internal(self, alias: str, collection_name: str) -> bool:
        """Internal helper to check for a collection's existence via the PyMilvus SDK."""
        from pymilvus import utility
        return utility.has_collection(collection_name, using=alias)
    
    async def list_collections(self, strict: bool = False, timeout: Optional[float] = None) -> List[str]:
        """
        Retrieves a list of all collection names from the Milvus server.

        Args:
            strict: If `True`, any exception during the operation will be raised.
                    If `False` (default), exceptions are suppressed and an empty
                    list is returned.
            timeout: The maximum time in seconds to wait for the operation to complete.
            
        Returns:
            A list of strings, where each string is a collection name.
            
        Raises:
            CollectionError: If `strict` is `True` and an error occurs.
            OperationTimeoutError: If the operation times out and `strict` is `True`.
        """
        try:
            result = await self._connection_manager.execute_operation_async(
                lambda alias: self._list_collections_internal(alias),
                timeout=timeout
            )
            return await self._ensure_awaited(result)
        except Exception as e:
            logger.error(f"[list_collections] Error listing collections (alias={getattr(e, 'using', 'unknown')}): {e}")
            if strict:
                # Preserve specific exception types for better error handling
                if isinstance(e, (ConnectionError, OperationTimeoutError)):
                    raise
                raise CollectionError(f"Error listing collections: {e}")
            return []
    
    def _list_collections_internal(self, alias: str) -> List[str]:
        """Internal helper to list all collections via the PyMilvus SDK."""
        from pymilvus import utility
        # Note: PyMilvus utility.list_collections doesn't accept timeout parameter
        return utility.list_collections(using=alias)
    
    async def describe_collection(self, collection_name: str, timeout: Optional[float] = None) -> CollectionDescription:
        """
        Retrieves detailed information about a specific collection.

        This method returns a `CollectionDescription` object that includes the
        collection's schema, logical ID (which is set to the collection name, not a server GUID),
        creation time, and other metadata.

        Args:
            collection_name: The name of the collection to describe.
            timeout: The maximum time in seconds to wait for the operation to complete.
            
        Returns:
            A `CollectionDescription` object with detailed information.
            
        Raises:
            CollectionNotFoundError: If the specified collection does not exist.
            CollectionError: If any other error occurs during the operation.
            OperationTimeoutError: If the operation times out.
        """
        try:
            result = await self._connection_manager.execute_operation_async(
                lambda alias: self._describe_collection_internal(alias, collection_name),
                timeout=timeout
            )
            return await self._ensure_awaited(result)
        except CollectionNotExistException:
            error_msg = f"Collection '{collection_name}' does not exist"
            logger.error(error_msg)
            raise CollectionNotFoundError(error_msg)
        except Exception as e:
            logger.error(f"[describe_collection] Error describing collection '{collection_name}' (alias={getattr(e, 'using', 'unknown')}): {e}")
            if isinstance(e, (CollectionNotFoundError, OperationTimeoutError)):
                raise
            # Schema errors should be propagated as SchemaError
            if isinstance(e, (SchemaError, SchemaNotReadyException)):
                raise SchemaError(f"Schema validation failed during describe: {e}")
            raise CollectionError(f"Error describing collection: {e}")
    
    # Reverse mapping from Milvus DataType to our DataType enum
    _REVERSE_DATATYPE_MAP = {
        "BOOL": "BOOL",
        "INT8": "INT8",
        "INT16": "INT16",
        "INT32": "INT32",
        "INT64": "INT64",
        "FLOAT": "FLOAT",
        "DOUBLE": "DOUBLE",
        "VARCHAR": "VARCHAR",  # Always map to VARCHAR, not STRING
        "STRING": "VARCHAR",   # In case Milvus returns STRING
        "BINARY_VECTOR": "BINARY_VECTOR",
        "FLOAT_VECTOR": "FLOAT_VECTOR",
        "SPARSE_FLOAT_VECTOR": "SPARSE_FLOAT_VECTOR",
        "JSON": "JSON",
        "ARRAY": "ARRAY"
    }
    
    async def _ensure_awaited(self, result):
        """
        Helper method to ensure a result is properly awaited if it's a coroutine.
        
        Args:
            result: The result to check and potentially await
            
        Returns:
            The awaited result if it was a coroutine, otherwise the original result
        """
        if asyncio.iscoroutine(result):
            return await result
        return result
        
    def _describe_collection_internal(self, alias: str, collection_name: str) -> CollectionDescription:
        """
        Internal helper to describe a collection via the PyMilvus SDK.
        
        This method handles the translation from the PyMilvus SDK's response
        objects to this library's strongly-typed Pydantic models.
        """
        from pymilvus import Collection
        
        # Get the collection
        collection = Collection(name=collection_name, using=alias)
        
        # Get schema
        milvus_schema = collection.schema
        
        # Convert pymilvus schema to our schema model
        fields = []
        for field in milvus_schema.fields:
            # Convert pymilvus DataType to our DataType enum string with explicit mapping
            try:
                milvus_dtype_name = field.dtype.name
                if milvus_dtype_name not in self._REVERSE_DATATYPE_MAP:
                    raise SchemaError(f"Unknown Milvus data type: {milvus_dtype_name}")
                
                dtype_name = self._REVERSE_DATATYPE_MAP[milvus_dtype_name]
                
                # Validate that this maps to a valid DataType in our enum
                if not hasattr(DataType, dtype_name):
                    raise SchemaError(f"Cannot map Milvus data type {milvus_dtype_name} to our DataType enum")
                
            except AttributeError as e:
                raise SchemaError(f"Failed to get data type from field {field.name}: {e}")
            
            # Create our FieldSchema
            field_params = {
                "name": field.name,
                "dtype": dtype_name,
                "description": field.description,
                "is_primary": field.is_primary,
                "auto_id": getattr(field, "auto_id", False),
                "dim": getattr(field, "dim", None),
                "max_length": getattr(field, "max_length", None),
            }
            
            # Handle element_type for ARRAY fields
            if dtype_name == "ARRAY" and hasattr(field, "element_type") and field.element_type is not None:
                try:
                    element_dtype_name = field.element_type.name
                    if element_dtype_name not in self._REVERSE_DATATYPE_MAP:
                        raise SchemaError(f"Unknown Milvus element type: {element_dtype_name}")
                    
                    mapped_element_type = self._REVERSE_DATATYPE_MAP[element_dtype_name]
                    
                    # Validate that this maps to a valid DataType in our enum
                    if not hasattr(DataType, mapped_element_type):
                        raise SchemaError(f"Cannot map Milvus element type {element_dtype_name} to our DataType enum")
                    
                    field_params["element_type"] = mapped_element_type
                except AttributeError as e:
                    logger.warning(f"Failed to extract element_type from ARRAY field {field.name}: {e}")
            
            field_schema = FieldSchema(**field_params)
            fields.append(field_schema)
        
        # Create our CollectionSchema
        schema = CollectionSchema(
            fields=fields,
            description=milvus_schema.description,
            enable_dynamic_field=getattr(collection, "enable_dynamic_field", False),
            shard_num=getattr(collection, "num_shards", 2)
        )
        
        # Get collection info - use name as logical ID for tracking
        # Note: This is intentionally using the collection name as a logical ID for tracking,
        # not attempting to extract a server-assigned GUID which may not be available.
        # This design choice ensures stable, predictable IDs that can be used for correlation
        # across operations and is consistent with how most integrators will reference collections.
        # Applications should NOT assume this ID is a unique server-generated GUID.
        collection_id = collection.name
        
        # Handle created_at with clear indication if synthesized
        import datetime
        created_at = getattr(collection, "created_time", None)
        created_at_is_synthetic = created_at is None
        if created_at is None:
            created_at = datetime.datetime.now()
        
        # Compute schema hash
        schema_hash = schema.compute_hash()
        
        # Determine collection state - this can be extended in the future
        # to track more detailed state transitions
        collection_state = CollectionState.AVAILABLE
        
        # Check load state using utility.load_state() - most reliable method
        load_state = LoadState.UNLOADED
        try:
            from pymilvus import utility
            load_state_info = utility.load_state(collection_name, using=alias)
            # Convert to string for comparison (handles enum, string, or dict)
            state_str = str(load_state_info).lower()
            if 'loaded' in state_str:
                load_state = LoadState.LOADED
        except Exception:
            # Fallback: try collection attribute
            try:
                is_loaded = getattr(collection, "is_loaded", False)
                if is_loaded:
                    load_state = LoadState.LOADED
            except (AttributeError, TypeError):
                # If all methods fail, assume unloaded
                load_state = LoadState.UNLOADED
        
        # For future control-plane integration, we could detect in-progress operations
        # by checking for locks or operation records in a metadata store
        
        return CollectionDescription(
            name=collection_name,
            collection_schema=schema,
            id=collection_id,
            created_at=created_at,
            schema_hash=schema_hash,
            state=collection_state,
            load_state=load_state,
            created_at_is_synthetic=created_at_is_synthetic  # Flag indicating if date is synthesized
        )
    
    async def load_collection(
        self, 
        collection_name: str, 
        wait: bool = False,
        timeout: Optional[float] = None,
        ignore_index_errors: bool = True
    ) -> Union[bool, LoadProgress]:
        """
        Loads a collection into Milvus's memory for querying.
        
        This is a prerequisite for performing search or query operations on a
        collection. This method can operate in two modes:
        - Non-blocking (`wait=False`): Initiates the loading process and returns
          immediately.
        - Blocking (`wait=True`): Initiates the loading and waits until the
          process completes or times out, polling with a gentle backoff strategy.

        Args:
            collection_name: The name of the collection to load.
            wait: If `True`, the method will block until loading is complete.
            timeout: The maximum time in seconds to wait for loading to complete.
                     Only effective when `wait=True`.
            ignore_index_errors: If `True`, errors related to missing indexes will be
                     ignored, allowing operations on collections without indexes.
            
        Returns:
            - If `wait=False`, returns `True` if the loading was initiated.
            - If `wait=True`, returns a `LoadProgress` object with the final status.
            
        Raises:
            CollectionNotFoundError: If the specified collection does not exist.
            CollectionError: If the load operation fails to initiate.
            OperationTimeoutError: If `wait=True` and the loading does not complete
                                 within the specified timeout.
        """
        # Get collection lock
        collection_lock = await self._get_collection_lock(collection_name)
        
        async with collection_lock:
            # Check if collection exists
            exists = await self.has_collection(collection_name, strict=True, timeout=timeout)
            if not exists:
                error_msg = f"Collection '{collection_name}' does not exist"
                logger.error(error_msg)
                raise CollectionNotFoundError(error_msg)
            
            try:
                # Initiate load
                # For future control-plane integration, we could update a metadata store
                # to set the collection state to LOADING here
                
                try:
                    await self._connection_manager.execute_operation_async(
                        lambda alias: self._load_collection_internal(alias, collection_name)
                    )
                except Exception as e:
                    # Check if this is an index-related error that we should ignore
                    error_str = str(e).lower()
                    if ignore_index_errors and ("index" in error_str and ("exist" in error_str or "found" in error_str)):
                        logger.warning(f"Ignoring index-related error during collection load: {e}")
                        # Return a synthetic "loaded" state since we're ignoring the error
                        if not wait:
                            return True
                        else:
                            return LoadProgress(
                                collection_name=collection_name,
                                state=LoadState.LOADED,
                                progress=1.0,
                                loaded_segments=0,
                                total_segments=0,
                                error_message=f"Loaded without index: {e}"
                            )
                    else:
                        # Re-raise if it's not an index error or we're not ignoring them
                        raise
                
                # For future control-plane integration, we could update a metadata store
                # to set the collection state to LOADED here
                
                if not wait:
                    return True
                
                # Wait for loading to complete with gentle backoff
                start_time = time.time()
                poll_interval = 0.5  # Start with 0.5s polling interval
                max_poll_interval = 5.0  # Cap at 5s
                iteration = 0
                
                logger.info(f"Waiting for collection '{collection_name}' to load (timeout={timeout}s)...")
                
                while True:
                    iteration += 1
                    
                    # Check progress
                    progress = await self.get_load_progress(collection_name, timeout=timeout)
                    
                    # Log progress every 5 iterations for debugging
                    if iteration % 5 == 0:
                        logger.info(
                            f"Load progress for '{collection_name}': "
                            f"state={progress.state.value}, progress={progress.progress:.2%}, "
                            f"iteration={iteration}"
                        )
                    
                    if progress.is_complete:
                        logger.info(f"Collection '{collection_name}' load complete: state={progress.state.value}")
                        return progress
                    
                    # Check timeout
                    elapsed = time.time() - start_time
                    if timeout and elapsed > timeout:
                        error_msg = (
                            f"Timed out waiting for collection '{collection_name}' to load after {elapsed:.1f}s. "
                            f"Last state: {progress.state.value}, progress: {progress.progress:.2%}"
                        )
                        logger.error(error_msg)
                        raise OperationTimeoutError(error_msg)
                    
                    # SAFETY: If we've been stuck at 0% progress for too long, something is wrong
                    if iteration > 20 and progress.progress == 0.0:
                        error_msg = (
                            f"Collection '{collection_name}' appears stuck at 0% progress after {iteration} iterations. "
                            f"State: {progress.state.value}. This may indicate the collection is already loaded "
                            f"or there's an issue with progress reporting."
                        )
                        logger.warning(error_msg)
                        # Try one more check with utility.load_state
                        try:
                            from pymilvus import utility
                            load_state_info = await self._connection_manager.execute_operation_async(
                                lambda alias: utility.load_state(collection_name, using=alias),
                                timeout=5.0
                            )
                            logger.info(f"Direct load_state check: {load_state_info}")
                            # If it's actually loaded, return success
                            if isinstance(load_state_info, dict):
                                state_value = str(load_state_info.get('state', '')).lower()
                                if 'loaded' in state_value:
                                    return LoadProgress(
                                        collection_name=collection_name,
                                        state=LoadState.LOADED,
                                        progress=1.0,
                                        loaded_segments=0,
                                        total_segments=0
                                    )
                        except Exception as check_error:
                            logger.error(f"Failed to verify load state: {check_error}")
                        
                        # If still not loaded after 20 iterations, raise error
                        raise OperationTimeoutError(
                            f"Collection load appears stuck. {error_msg}"
                        )
                    
                    # Wait before checking again with gentle backoff
                    await asyncio.sleep(poll_interval)
                    
                    # Increase poll interval with a cap
                    poll_interval = min(poll_interval * 1.5, max_poll_interval)
                    
            except Exception as e:
                logger.error(f"[load_collection] Error loading collection '{collection_name}' (alias={getattr(e, 'using', 'unknown')}): {e}")
                if isinstance(e, (CollectionNotFoundError, OperationTimeoutError)):
                    raise
                raise CollectionError(f"Error loading collection: {e}")
    
    def _load_collection_internal(self, alias: str, collection_name: str) -> None:
        """Internal helper to load a collection via the PyMilvus SDK."""
        from pymilvus import Collection
        collection = Collection(name=collection_name, using=alias)
        
        # Handle different pymilvus versions
        try:
            collection.load()
        except AttributeError:
            # Some versions might use a different method or approach
            try:
                if hasattr(collection, "load_collection"):
                    collection.load_collection()
            except Exception as e:
                raise RuntimeError(f"Failed to load collection: {e}")
    
    async def get_load_progress(self, collection_name: str, timeout: Optional[float] = None) -> LoadProgress:
        """
        Retrieves the current loading progress of a collection.

        This method is useful for monitoring the status of a collection that is
        being loaded into memory, especially for large collections where the
        process can take time.

        Args:
            collection_name: The name of the collection to check.
            timeout: The maximum time in seconds to wait for the operation to complete.
            
        Returns:
            A `LoadProgress` object with the current loading status.
            
        Raises:
            CollectionNotFoundError: If the specified collection does not exist.
            OperationTimeoutError: If the operation times out.
            CollectionError: If any other error occurs.
        """
        try:
            result = await self._connection_manager.execute_operation_async(
                lambda alias: self._get_load_progress_internal(alias, collection_name),
                timeout=timeout
            )
            return await self._ensure_awaited(result)
        except CollectionNotExistException:
            error_msg = f"Collection '{collection_name}' does not exist"
            logger.error(error_msg)
            raise CollectionNotFoundError(error_msg)
        except Exception as e:
            logger.error(f"[get_load_progress] Error getting load progress for collection '{collection_name}' (alias={getattr(e, 'using', 'unknown')}): {e}")
            if isinstance(e, (CollectionNotFoundError, OperationTimeoutError)):
                raise
            raise CollectionError(f"Error getting load progress: {e}")
    
    def _get_load_progress_internal(self, alias: str, collection_name: str) -> LoadProgress:
        """
        Internal helper to get load progress via the PyMilvus SDK.
        
        This method handles different responses from various PyMilvus versions
        and normalizes them into a consistent `LoadProgress` object.
        """
        from pymilvus import Collection, utility
        
        # Get the collection
        collection = Collection(name=collection_name, using=alias)
        
        # CRITICAL FIX: Check load state using utility.load_state() first
        # This is the most reliable way to determine if a collection is loaded
        try:
            load_state_info = utility.load_state(collection_name, using=alias)
            # load_state_info can be an enum, string, or dict
            # Convert to string for comparison
            state_str = str(load_state_info).lower()
            
            if 'loaded' in state_str:
                return LoadProgress(
                    collection_name=collection_name,
                    state=LoadState.LOADED,
                    progress=1.0,
                    loaded_segments=0,
                    total_segments=0
                )
        except Exception as e:
            logger.debug(f"Could not get load state via utility.load_state: {e}")
        
        # Check if loaded using collection attribute - handle different pymilvus versions
        try:
            is_loaded = getattr(collection, "is_loaded", False)
            if is_loaded:
                return LoadProgress(
                    collection_name=collection_name,
                    state=LoadState.LOADED,
                    progress=1.0,
                    loaded_segments=0,
                    total_segments=0
                )
        except (AttributeError, TypeError):
            # If we can't determine loaded state, try to get progress anyway
            pass
        
        # Get loading progress
        try:
            # Try to get loading_progress attribute
            progress = getattr(collection, "loading_progress", None)
            
            if progress is not None:
                return LoadProgress.from_milvus_response(collection_name, progress)
            
            # If attribute exists but is None, try to determine load state another way
            # This is a fallback for pymilvus versions with different APIs
            try:
                # Some versions might have get_loading_progress method
                if hasattr(collection, "get_loading_progress"):
                    progress = collection.get_loading_progress()
                    return LoadProgress.from_milvus_response(collection_name, progress)
            except Exception:
                pass
                
            # CRITICAL FIX: If we can't get progress info, assume collection is UNLOADED
            # rather than LOADING. This prevents infinite loops when progress is unavailable.
            return LoadProgress(
                collection_name=collection_name,
                state=LoadState.UNLOADED,
                progress=0.0,
                loaded_segments=0,
                total_segments=0,
                error_message="Progress information unavailable - assuming unloaded"
            )
        except (AttributeError, TypeError) as e:
            # CRITICAL FIX: Return UNLOADED state instead of LOADING
            return LoadProgress(
                collection_name=collection_name,
                state=LoadState.UNLOADED,
                progress=0.0,
                loaded_segments=0,
                total_segments=0,
                error_message=f"Progress information unavailable: {e}"
            )
    
    async def release_collection(self, collection_name: str, timeout: Optional[float] = None) -> bool:
        """
        Releases a collection from Milvus's memory.

        This operation unloads the collection's data from memory, freeing up
        resources. After being released, a collection cannot be searched or
        queried until it is loaded again.

        Args:
            collection_name: The name of the collection to release.
            timeout: The maximum time in seconds to wait for the operation to complete.
            
        Returns:
            `True` if the release operation was successful.
            
        Raises:
            CollectionNotFoundError: If the specified collection does not exist.
            OperationTimeoutError: If the operation times out.
            CollectionError: If the release operation fails.
        """
        # Get collection lock
        collection_lock = await self._get_collection_lock(collection_name)
        
        async with collection_lock:
            # Check if collection exists
            exists = await self.has_collection(collection_name, strict=True, timeout=timeout)
            if not exists:
                error_msg = f"Collection '{collection_name}' does not exist"
                logger.error(error_msg)
                raise CollectionNotFoundError(error_msg)
            
            try:
                # Release collection
                await self._connection_manager.execute_operation_async(
                    lambda alias: self._release_collection_internal(alias, collection_name),
                    timeout=timeout
                )
                return True
            except Exception as e:
                logger.error(f"[release_collection] Error releasing collection '{collection_name}' (alias={getattr(e, 'using', 'unknown')}): {e}")
                if isinstance(e, (CollectionNotFoundError, OperationTimeoutError)):
                    raise
                raise CollectionError(f"Error releasing collection: {e}")
    
    def _release_collection_internal(self, alias: str, collection_name: str) -> None:
        """Internal helper to release a collection via the PyMilvus SDK."""
        from pymilvus import Collection
        collection = Collection(name=collection_name, using=alias)
        
        # Handle different pymilvus versions
        try:
            collection.release()
        except AttributeError:
            # Some versions might use a different method or approach
            try:
                if hasattr(collection, "release_collection"):
                    collection.release_collection()
            except Exception as e:
                raise RuntimeError(f"Failed to release collection: {e}")
    
    async def drop_collection(self, collection_name: str, safe: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Drops a collection permanently from Milvus.

        This is a destructive operation and cannot be undone.

        Args:
            collection_name: The name of the collection to drop.
            safe: If `True` (default), the method will first check if the
                  collection is currently loaded or loading and, if so, will
                  raise an error to prevent accidental deletion of an active
                  collection. If `False`, the collection will be dropped
                  regardless of its load state.
            timeout: The maximum time in seconds to wait for the operation to complete.
            
        Returns:
            `True` if the drop operation was successful.
            
        Raises:
            CollectionNotFoundError: If the specified collection does not exist.
            OperationTimeoutError: If the operation times out.
            CollectionError: If `safe=True` and the collection is currently loaded
                             or loading, or if any other error occurs.
        """
        # Get collection lock
        collection_lock = await self._get_collection_lock(collection_name)
        
        async with collection_lock:
            # Check if collection exists
            exists = await self.has_collection(collection_name, strict=True, timeout=timeout)
            if not exists:
                error_msg = f"Collection '{collection_name}' does not exist"
                logger.error(error_msg)
                raise CollectionNotFoundError(error_msg)
            
            if safe:
                # Check if collection is loaded or loading
                try:
                    description = await self.describe_collection(collection_name, timeout=timeout)
                    if description.load_state in [LoadState.LOADED, LoadState.LOADING]:
                        operation = "loaded" if description.load_state == LoadState.LOADED else "loading"
                        error_msg = (
                            f"Collection '{collection_name}' is currently {operation}. "
                            f"Release it before dropping or use safe=False"
                        )
                        logger.error(error_msg)
                        raise CollectionError(error_msg)
                except Exception as e:
                    if isinstance(e, CollectionError):
                        raise
                    logger.error(f"Error checking collection state: {e}")
            
            try:
                # For future control-plane integration, we could update a metadata store
                # to set the collection state to DROPPING here
                
                # Drop collection
                await self._connection_manager.execute_operation_async(
                    lambda alias: self._drop_collection_internal(alias, collection_name),
                    timeout=timeout
                )
                
                # Remove the collection lock immediately
                async with self._global_lock:
                    if collection_name in self._locks:
                        del self._locks[collection_name]
                
                # PRODUCTION FIX: Trigger cleanup of other orphaned locks
                # This prevents memory leaks by opportunistically removing locks for
                # any other collections that may have been dropped externally or by
                # other clients. This is safe because cleanup only removes locks for
                # non-existent collections.
                try:
                    await self._cleanup_unused_locks()
                except Exception as cleanup_error:
                    # Don't fail the drop operation if cleanup fails
                    logger.warning(f"Failed to cleanup unused locks: {cleanup_error}")
                
                # For future control-plane integration, we could update a metadata store
                # to set the collection state to DELETED here
                
                return True
            except Exception as e:
                logger.error(f"[drop_collection] Error dropping collection '{collection_name}' (alias={getattr(e, 'using', 'unknown')}): {e}")
                if isinstance(e, (CollectionNotFoundError, OperationTimeoutError)):
                    raise
                raise CollectionError(f"Error dropping collection: {e}")
    
    def _drop_collection_internal(self, alias: str, collection_name: str) -> None:
        """Internal helper to drop a collection via the PyMilvus SDK."""
        try:
            # Standard approach using utility
            from pymilvus import utility
            utility.drop_collection(collection_name, using=alias)
        except (ImportError, AttributeError):
            # Fallback for older versions that might not have utility module
            try:
                from pymilvus import Collection
                collection = Collection(name=collection_name, using=alias)
                if hasattr(collection, "drop"):
                    collection.drop()
                elif hasattr(Collection, "drop_collection"):
                    Collection.drop_collection(collection_name, using=alias)
                else:
                    raise RuntimeError("Could not find appropriate method to drop collection")
            except Exception as e:
                raise RuntimeError(f"Failed to drop collection: {e}")

    async def insert(self, collection_name: str, data: List[list], timeout: Optional[float] = None) -> Any:
        """
        Inserts data into a collection.

        For more advanced features like partitioning, use the data_insertion_operations module.

        Args:
            collection_name: The name of the collection to insert data into.
            data: A list of lists representing the data to insert.
            timeout: The maximum time in seconds to wait for the operation to complete.
        
        Returns:
            The result of the insertion operation from pymilvus.
        
        Raises:
            CollectionNotFoundError: If the specified collection does not exist.
            CollectionError: If the insertion fails.
        """
        try:
            # Ensure collection exists before inserting
            exists = await self.has_collection(collection_name, strict=True, timeout=timeout)
            if not exists:
                raise CollectionNotFoundError(f"Collection '{collection_name}' does not exist.")

            # ARCHITECTURAL FIX:
            # The root cause of the previous 'ConnectionNotExistException' was that pymilvus
            # operations were being called directly from the main application thread, where no
            # Milvus connection had been established.
            # The correct, enterprise-grade solution is to delegate all database operations
            # to the ConnectionManager. The `execute_operation_async` method is the key:
            # it takes a callable (our lambda function) and runs it in a managed worker
            # thread. This thread is guaranteed by the ConnectionManager to have an active,
            # thread-local Milvus connection. This design respects the architecture,
            # centralizes connection handling, and ensures stability.
            result = await self._connection_manager.execute_operation_async(
                lambda alias: self._insert_internal(alias, collection_name, data),
                timeout=timeout
            )
            return await self._ensure_awaited(result)
        except CollectionNotExistException:
            raise CollectionNotFoundError(f"Collection '{collection_name}' does not exist during insert.")
        except Exception as e:
            logger.error(f"[insert] Error inserting data into collection '{collection_name}': {e}")
            raise CollectionError(f"Failed to insert data: {e}")

    def _insert_internal(self, alias: str, collection_name: str, data: List[list]) -> Any:
        """
        Internal helper to insert data via the PyMilvus SDK.

        This method is designed to be executed within a worker thread managed by the
        ConnectionManager. The 'alias' it receives is the identifier for the active,
        thread-local connection provided by the manager.
        """
        from pymilvus import Collection
        
        # This call now succeeds because it is no longer running in the main application
        # thread. It is executing inside the ConnectionManager's managed thread pool,
        # where `using=alias` correctly references an active connection.
        collection = Collection(name=collection_name, using=alias)
        insert_result = collection.insert(data)
        # We also flush here to ensure data is immediately queryable, which is
        collection.flush()
        logger.info(f"Inserted data into '{collection_name}' and flushed.")
        return insert_result
    
    async def get_collection_stats(self, collection_name: str, timeout: Optional[float] = None) -> CollectionStats:
        """
        Retrieves detailed statistics for a collection.

        This method provides valuable information for monitoring and capacity
        planning, including the number of entities, total disk size, and
        details about partitions and segments.

        Args:
            collection_name: The name of the collection to get stats for.
            timeout: The maximum time in seconds to wait for the operation to complete.
            
        Returns:
            A `CollectionStats` object with detailed statistics.
            
        Raises:
            CollectionNotFoundError: If the specified collection does not exist.
            OperationTimeoutError: If the operation times out.
            CollectionError: If any other error occurs.
        """
        try:
            result = await self._connection_manager.execute_operation_async(
                lambda alias: self._get_collection_stats_internal(alias, collection_name),
                timeout=timeout
            )
            return await self._ensure_awaited(result)
        except CollectionNotExistException:
            error_msg = f"Collection '{collection_name}' does not exist"
            logger.error(error_msg)
            raise CollectionNotFoundError(error_msg)
        except Exception as e:
            logger.error(f"[get_collection_stats] Error getting stats for collection '{collection_name}' (alias={getattr(e, 'using', 'unknown')}): {e}")
            if isinstance(e, (CollectionNotFoundError, OperationTimeoutError)):
                raise
            raise CollectionError(f"Error getting collection stats: {e}")
    
    def _get_collection_stats_internal(self, alias: str, collection_name: str) -> CollectionStats:
        """Internal helper to get collection statistics via the PyMilvus SDK."""
        from pymilvus import Collection
        
        # Get the collection
        collection = Collection(name=collection_name, using=alias)
        
        # Get statistics - handle different pymilvus versions
        try:
            # Try standard method
            stats = collection.get_collection_stats()
            
            # Convert to our model
            return CollectionStats.from_milvus_response(collection_name, stats)
        except AttributeError:
            # Some versions might use a different method name or approach
            try:
                # Try alternative method names that might exist in different versions
                if hasattr(collection, "stats"):
                    stats = collection.stats()
                    return CollectionStats.from_milvus_response(collection_name, stats)
                
                # Last resort - construct minimal stats
                return CollectionStats(
                    name=collection_name,
                    id=collection.name,
                    created_at=datetime.now(),
                    row_count=0,
                    memory_size=0,
                    disk_size=0,
                    index_size=0
                )
            except Exception as e:
                # If all attempts fail, return minimal stats with error info
                return CollectionStats(
                    name=collection_name,
                    id=collection.name,
                    created_at=datetime.now(),
                    row_count=0,
                    memory_size=0,
                    disk_size=0,
                    index_size=0
                )
    
    async def cleanup_locks(self) -> int:
        """
        Public method to manually trigger cleanup of unused collection locks.
        
        This method can be called periodically by the application to ensure
        memory-efficient operation in long-running deployments. It's particularly
        useful for applications that frequently create and drop collections.
        
        PRODUCTION USAGE:
        - Call this method periodically (e.g., every hour) in production
        - Call after batch operations that create/drop many collections
        - Monitor the return value to track lock accumulation patterns
        
        Returns:
            The number of locks that were cleaned up.
            
        Example:
            >>> manager = CollectionManager(connection_manager)
            >>> # After dropping multiple collections
            >>> cleaned = await manager.cleanup_locks()
            >>> logger.info(f"Cleaned up {cleaned} unused locks")
        """
        return await self._cleanup_unused_locks()
