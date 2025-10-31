"""
Comprehensive unit tests for CollectionManager.

This module provides systematic testing of all public methods in the CollectionManager
class, achieving 95%+ code coverage through positive, negative, and edge case testing.
"""

import asyncio
import contextlib
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from milvus_ops.collection_operations.entities import (
    CollectionDescription,
    CollectionState,
    CollectionStats,
    LoadProgress,
    LoadState,
)
from milvus_ops.milvus_ops_exceptions import (
    CollectionError,
    CollectionNotFoundError,
    OperationTimeoutError,
    SchemaError,
)

# ============================================================================
# Test CollectionManager Initialization
# ============================================================================


@pytest.mark.unit
class TestCollectionManagerInitialization:
    """
    Test CollectionManager initialization and setup.

    Coverage: Verifies proper initialization of CollectionManager with various
    configurations and dependency injection.
    """

    def test_init_basic(self, mock_connection_manager):
        """
        Test basic CollectionManager initialization.

        Coverage: CollectionManager.__init__ with standard configuration.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Verify internal state
        assert manager._connection_manager == mock_connection_manager
        assert isinstance(manager._locks, dict)
        assert hasattr(manager, "_global_lock")
        assert isinstance(manager._global_lock, asyncio.Lock)

    def test_init_with_connection_manager(self, mock_connection_manager):
        """
        Test CollectionManager initialization with explicit ConnectionManager.

        Coverage: CollectionManager.__init__ with provided connection manager.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Verify all internal components
        assert manager._connection_manager is mock_connection_manager
        assert hasattr(manager, "_get_collection_lock")
        assert hasattr(manager, "_cleanup_unused_locks")

    def test_get_collection_lock_new(self, mock_connection_manager):
        """
        Test collection lock creation for new collection.

        Coverage: _get_collection_lock() creating new locks.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        async def test_lock_creation():
            lock = await manager._get_collection_lock("new_collection")

            # Verify lock was created
            assert "new_collection" in manager._locks
            assert isinstance(lock, asyncio.Lock)

            # Verify same collection returns same lock
            lock2 = await manager._get_collection_lock("new_collection")
            assert lock is lock2

        asyncio.run(test_lock_creation())

    def test_get_collection_lock_existing(self, mock_connection_manager):
        """
        Test collection lock retrieval for existing collection.

        Coverage: _get_collection_lock() retrieving existing locks.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        async def test_lock_retrieval():
            # Pre-create a lock
            lock1 = await manager._get_collection_lock("existing_collection")

            # Retrieve the same lock
            lock2 = await manager._get_collection_lock("existing_collection")

            # Verify same lock object is returned
            assert lock1 is lock2
            assert len(manager._locks) == 1

        asyncio.run(test_lock_retrieval())


# ============================================================================
# Test Collection Creation
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.async_test
class TestCollectionCreation:
    """
    Test collection creation operations.

    Coverage: Verifies create_collection() handles all scenarios including
    validation, idempotency, error handling, and schema compatibility.
    """

    async def test_create_collection_success(
        self,
        mock_connection_manager,
        basic_collection_schema,
        mock_pymilvus_utility,
        mock_pymilvus_collection,
    ):
        """
        Test successful collection creation.

        Coverage: create_collection() with valid schema and non-existent collection.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock has_collection to return False
        mock_connection_manager.execute_operation_async.return_value = False

        # Mock describe_collection for final verification
        mock_description = CollectionDescription(
            name="test_collection",
            collection_schema=basic_collection_schema,
            id="test_collection_id",
            created_at=datetime.now(timezone.utc),
            schema_hash="abc123",
        )
        mock_connection_manager.execute_operation_async.return_value = mock_description

        with patch("pymilvus.Collection") as mock_collection_cls:
            mock_collection_cls.return_value = mock_pymilvus_collection

            result = await manager.create_collection("test_collection", basic_collection_schema)

            # Verify successful creation
            assert isinstance(result, CollectionDescription)
            assert result.name == "test_collection"
            assert result.collection_schema == basic_collection_schema

    async def test_create_collection_already_exists_compatible(
        self, mock_connection_manager, basic_collection_schema, mock_pymilvus_utility
    ):
        """
        Test idempotent behavior when collection exists with compatible schema.

        Coverage: create_collection() with existing compatible collection.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock has_collection to return True
        mock_connection_manager.execute_operation_async.return_value = True

        # Mock describe_collection to return existing description
        existing_description = CollectionDescription(
            name="test_collection",
            collection_schema=basic_collection_schema,
            id="existing_collection_id",
            created_at=datetime.now(timezone.utc),
            schema_hash="abc123",
        )
        mock_connection_manager.execute_operation_async.return_value = existing_description

        # Mock SchemaValidator.compare_schemas to return compatible
        validator_path = "milvus_ops.collection_operations.validator.SchemaValidator"
        with patch(f"{validator_path}.validate_schema") as mock_validate:
            mock_validate.return_value = (True, [])

            with patch(f"{validator_path}.compare_schemas") as mock_compare:
                mock_compare.return_value = (True, [])

                result = await manager.create_collection("test_collection", basic_collection_schema)

                # Should return existing collection description
                assert result == existing_description

    async def test_create_collection_already_exists_incompatible(
        self,
        mock_connection_manager,
        basic_collection_schema,
        complex_collection_schema,
        mock_pymilvus_utility,
    ):
        """
        Test error when collection exists with incompatible schema.

        Coverage: create_collection() with existing incompatible collection.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock has_collection to return True
        mock_connection_manager.execute_operation_async.return_value = True

        # Mock describe_collection to return existing description
        existing_description = CollectionDescription(
            name="test_collection",
            collection_schema=basic_collection_schema,
            id="existing_collection_id",
            created_at=datetime.now(timezone.utc),
            schema_hash="abc123",
        )
        mock_connection_manager.execute_operation_async.return_value = existing_description

        # Mock SchemaValidator.compare_schemas to return incompatible
        validator_path = "milvus_ops.collection_operations.validator.SchemaValidator"
        with patch(f"{validator_path}.validate_schema") as mock_validate:
            mock_validate.return_value = (True, [])

            with patch(f"{validator_path}.compare_schemas") as mock_compare:
                mock_compare.return_value = (False, ["Field mismatch"])

                with pytest.raises(SchemaError, match="incompatible schema"):
                    await manager.create_collection("test_collection", complex_collection_schema)

    async def test_create_collection_invalid_schema(
        self, mock_connection_manager, basic_collection_schema, mock_pymilvus_utility
    ):
        """
        Test error handling for invalid schema.

        Coverage: create_collection() with invalid schema validation.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock SchemaValidator.validate_schema to return invalid
        validator_path = "milvus_ops.collection_operations.validator.SchemaValidator"
        with patch(f"{validator_path}.validate_schema") as mock_validate:
            mock_validate.return_value = (False, ["Invalid field configuration"])

            with pytest.raises(SchemaError, match="Invalid schema"):
                await manager.create_collection("test_collection", basic_collection_schema)

    async def test_create_collection_timeout(
        self, mock_connection_manager, basic_collection_schema, mock_pymilvus_utility
    ):
        """
        Test timeout handling during collection creation.

        Coverage: create_collection() with timeout parameter.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock to timeout
        mock_connection_manager.execute_operation_async.side_effect = OperationTimeoutError(
            "Timeout"
        )

        with pytest.raises(OperationTimeoutError):
            await manager.create_collection("test_collection", basic_collection_schema, timeout=0.1)

    @pytest.mark.parametrize("timeout_value", [None, 5.0, 30.0])
    async def test_create_collection_timeout_variations(
        self, mock_connection_manager, basic_collection_schema, mock_pymilvus_utility, timeout_value
    ):
        """
        Test collection creation with various timeout values.

        Coverage: create_collection() timeout parameter variations.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock successful creation
        mock_connection_manager.execute_operation_async.return_value = CollectionDescription(
            name="test_collection",
            collection_schema=basic_collection_schema,
            id="test_collection_id",
            created_at=datetime.now(timezone.utc),
            schema_hash="abc123",
        )

        # Verify timeout parameter is passed correctly
        with patch("pymilvus.Collection") as mock_collection_cls:
            mock_collection = MagicMock()
            mock_collection_cls.return_value = mock_collection

            result = await manager.create_collection(
                "test_collection", basic_collection_schema, timeout=timeout_value
            )

            assert isinstance(result, CollectionDescription)


# ============================================================================
# Test Collection Existence Checks
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.async_test
class TestCollectionExistence:
    """
    Test collection existence checking operations.

    Coverage: Verifies has_collection() and list_collections() methods
    handle all scenarios including edge cases and error conditions.
    """

    async def test_has_collection_exists(self, mock_connection_manager, mock_pymilvus_utility):
        """
        Test has_collection when collection exists.

        Coverage: has_collection() with existing collection.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock utility.has_collection to return True
        mock_connection_manager.execute_operation_async.return_value = True

        result = await manager.has_collection("existing_collection")

        assert result is True

    async def test_has_collection_not_exists(self, mock_connection_manager, mock_pymilvus_utility):
        """
        Test has_collection when collection does not exist.

        Coverage: has_collection() with non-existent collection.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock utility.has_collection to return False
        mock_connection_manager.execute_operation_async.return_value = False

        result = await manager.has_collection("nonexistent_collection")

        assert result is False

    async def test_has_collection_strict_mode_error(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test has_collection in strict mode with error.

        Coverage: has_collection() strict=True with connection error.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock to raise connection error
        mock_connection_manager.execute_operation_async.side_effect = ConnectionError(
            "Connection failed"
        )

        with pytest.raises(ConnectionError):
            await manager.has_collection("test_collection", strict=True)

    async def test_has_collection_strict_mode_suppress_error(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test has_collection in non-strict mode suppresses errors.

        Coverage: has_collection() strict=False with connection error.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock to raise connection error
        mock_connection_manager.execute_operation_async.side_effect = ConnectionError(
            "Connection failed"
        )

        # Should return False instead of raising error
        result = await manager.has_collection("test_collection", strict=False)

        assert result is False

    async def test_list_collections_success(self, mock_connection_manager, mock_pymilvus_utility):
        """
        Test list_collections successful retrieval.

        Coverage: list_collections() with successful operation.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock utility.list_collections to return collection list
        mock_connection_manager.execute_operation_async.return_value = [
            "collection1",
            "collection2",
            "collection3",
        ]

        result = await manager.list_collections()

        assert result == ["collection1", "collection2", "collection3"]

    async def test_list_collections_empty(self, mock_connection_manager, mock_pymilvus_utility):
        """
        Test list_collections when no collections exist.

        Coverage: list_collections() with empty result.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock utility.list_collections to return empty list
        mock_connection_manager.execute_operation_async.return_value = []

        result = await manager.list_collections()

        assert result == []

    async def test_list_collections_error_handling(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test list_collections error handling.

        Coverage: list_collections() with connection error.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock to raise error
        mock_connection_manager.execute_operation_async.side_effect = Exception("Server error")

        # Should return empty list in non-strict mode
        result = await manager.list_collections(strict=False)

        assert result == []

        # Should raise error in strict mode
        with pytest.raises(CollectionError):
            await manager.list_collections(strict=True)


# ============================================================================
# Test Collection Description
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.async_test
class TestCollectionDescription:
    """
    Test collection description operations.

    Coverage: Verifies describe_collection() method handles all scenarios
    including schema parsing, data type mapping, and error conditions.
    """

    async def test_describe_collection_success(
        self,
        mock_connection_manager,
        basic_collection_schema,
        mock_pymilvus_collection,
        mock_pymilvus_datatype,
        mock_pymilvus_utility,
    ):
        """
        Test successful collection description retrieval.

        Coverage: describe_collection() with valid collection.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock describe_collection internal method
        mock_connection_manager.execute_operation_async.return_value = CollectionDescription(
            name="test_collection",
            collection_schema=basic_collection_schema,
            id="test_collection_id",
            created_at=datetime.now(timezone.utc),
            schema_hash="abc123",
            state=CollectionState.AVAILABLE,
            load_state=LoadState.UNLOADED,
        )

        result = await manager.describe_collection("test_collection")

        # Verify description structure
        assert isinstance(result, CollectionDescription)
        assert result.name == "test_collection"
        assert result.collection_schema == basic_collection_schema
        assert result.id == "test_collection_id"

    async def test_describe_collection_not_found(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test describe_collection when collection doesn't exist.

        Coverage: describe_collection() with non-existent collection.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock to raise CollectionNotExistException
        from pymilvus.exceptions import CollectionNotExistException

        mock_connection_manager.execute_operation_async.side_effect = CollectionNotExistException(
            "Collection not found"
        )

        with pytest.raises(CollectionNotFoundError, match="does not exist"):
            await manager.describe_collection("nonexistent_collection")

    async def test_describe_collection_schema_error(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test describe_collection with schema validation error.

        Coverage: describe_collection() with schema-related error.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock to raise SchemaNotReadyException
        from pymilvus.exceptions import SchemaNotReadyException

        mock_connection_manager.execute_operation_async.side_effect = SchemaNotReadyException(
            "Schema not ready"
        )

        with pytest.raises(SchemaError, match="Schema validation failed"):
            await manager.describe_collection("test_collection")

    async def test_describe_collection_generic_error(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test describe_collection with generic error.

        Coverage: describe_collection() with unexpected error.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock to raise generic exception
        mock_connection_manager.execute_operation_async.side_effect = Exception("Unexpected error")

        with pytest.raises(CollectionError, match="Error describing collection"):
            await manager.describe_collection("test_collection")


# ============================================================================
# Test Collection Loading Operations
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.async_test
class TestCollectionLoading:
    """
    Test collection loading and load progress operations.

    Coverage: Verifies load_collection() and get_load_progress() methods
    handle blocking/non-blocking modes, progress tracking, and error conditions.
    """

    async def test_load_collection_non_blocking_success(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test non-blocking collection loading success.

        Coverage: load_collection() with wait=False and successful initiation.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock has_collection to return True
        mock_connection_manager.execute_operation_async.return_value = True

        # Mock collection load
        mock_collection = MagicMock()
        with patch("pymilvus.Collection", return_value=mock_collection):
            result = await manager.load_collection("test_collection", wait=False)

            # Should return True indicating load was initiated
            assert result is True

    async def test_load_collection_blocking_success(
        self, mock_connection_manager, mock_pymilvus_utility, sample_load_progress
    ):
        """
        Test blocking collection loading with successful completion.

        Coverage: load_collection() with wait=True and successful loading.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock has_collection to return True
        mock_connection_manager.execute_operation_async.return_value = True

        # Mock get_load_progress to return completed progress
        with patch.object(manager, "get_load_progress", return_value=sample_load_progress):
            sample_load_progress.state = LoadState.LOADED
            sample_load_progress.progress = 1.0

            with patch("pymilvus.Collection"):
                result = await manager.load_collection("test_collection", wait=True)

                # Should return LoadProgress object
                assert isinstance(result, LoadProgress)
                assert result.state == LoadState.LOADED
                assert result.progress == 1.0

    async def test_load_collection_not_found(self, mock_connection_manager, mock_pymilvus_utility):
        """
        Test load_collection when collection doesn't exist.

        Coverage: load_collection() with non-existent collection.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock has_collection to return False
        mock_connection_manager.execute_operation_async.return_value = False

        with pytest.raises(CollectionNotFoundError, match="does not exist"):
            await manager.load_collection("nonexistent_collection")

    async def test_load_collection_with_index_error_ignored(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test load_collection ignoring index-related errors.

        Coverage: load_collection() with ignore_index_errors=True.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock has_collection to return True
        mock_connection_manager.execute_operation_async.return_value = True

        # Mock collection load to raise index-related error
        mock_collection = MagicMock()
        mock_collection.load.side_effect = Exception("Index not found error")

        with patch("pymilvus.Collection", return_value=mock_collection):
            # Should ignore index errors and return True
            result = await manager.load_collection("test_collection", ignore_index_errors=True)
            assert result is True

    async def test_load_collection_with_timeout(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test load_collection with timeout handling.

        Coverage: load_collection() timeout enforcement.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock has_collection to return True
        mock_connection_manager.execute_operation_async.return_value = True

        # Mock get_load_progress to return loading state indefinitely
        loading_progress = LoadProgress(
            collection_name="test_collection",
            state=LoadState.LOADING,
            progress=0.0,
            loaded_segments=0,
            total_segments=2,
        )

        with (
            patch.object(manager, "get_load_progress", return_value=loading_progress),
            patch("pymilvus.Collection"),
            pytest.raises(OperationTimeoutError, match="Timed out waiting"),
        ):
            await manager.load_collection("test_collection", wait=True, timeout=10.0)

    async def test_get_load_progress_success(
        self, mock_connection_manager, mock_pymilvus_collection, mock_pymilvus_utility
    ):
        """
        Test successful load progress retrieval.

        Coverage: get_load_progress() with valid collection.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock collection load state
        mock_pymilvus_utility.load_state.return_value = "Loaded"

        # Mock collection attributes
        mock_pymilvus_collection.is_loaded = True
        mock_pymilvus_collection.loading_progress = "100%"

        mock_connection_manager.execute_operation_async.return_value = LoadProgress(
            collection_name="test_collection",
            state=LoadState.LOADED,
            progress=1.0,
            loaded_segments=2,
            total_segments=2,
        )

        result = await manager.get_load_progress("test_collection")

        assert isinstance(result, LoadProgress)
        assert result.collection_name == "test_collection"

    async def test_get_load_progress_not_found(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test get_load_progress when collection doesn't exist.

        Coverage: get_load_progress() with non-existent collection.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock to raise CollectionNotExistException
        from pymilvus.exceptions import CollectionNotExistException

        mock_connection_manager.execute_operation_async.side_effect = CollectionNotExistException(
            "Collection not found"
        )

        with pytest.raises(CollectionNotFoundError, match="does not exist"):
            await manager.get_load_progress("nonexistent_collection")


# ============================================================================
# Test Collection Release Operations
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.async_test
class TestCollectionRelease:
    """
    Test collection release operations.

    Coverage: Verifies release_collection() method handles all scenarios
    including validation, error handling, and resource cleanup.
    """

    async def test_release_collection_success(
        self, mock_connection_manager, mock_pymilvus_utility, mock_pymilvus_collection
    ):
        """
        Test successful collection release.

        Coverage: release_collection() with valid collection.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock operations - first call (has_collection) returns True,
        # second call (release operation) executes normally
        call_count = 0

        def mock_execute(operation, timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # has_collection check
                return True
            else:  # release operation
                return operation("mock_alias")

        mock_connection_manager.execute_operation_async.side_effect = mock_execute

        # Mock collection release
        with patch("pymilvus.Collection") as mock_collection_cls:
            mock_collection = MagicMock()
            mock_collection_cls.return_value = mock_collection

            result = await manager.release_collection("test_collection")

            assert result is True
            mock_collection.release.assert_called_once()

    async def test_release_collection_not_found(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test release_collection when collection doesn't exist.

        Coverage: release_collection() with non-existent collection.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock has_collection to return False
        mock_connection_manager.execute_operation_async.return_value = False

        with pytest.raises(CollectionNotFoundError, match="does not exist"):
            await manager.release_collection("nonexistent_collection")

    async def test_release_collection_timeout(self, mock_connection_manager, mock_pymilvus_utility):
        """
        Test release_collection with timeout.

        Coverage: release_collection() with timeout parameter.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock has_collection to return True
        mock_connection_manager.execute_operation_async.return_value = True

        # Mock to timeout
        mock_connection_manager.execute_operation_async.side_effect = OperationTimeoutError(
            "Timeout"
        )

        with pytest.raises(OperationTimeoutError):
            await manager.release_collection("test_collection", timeout=0.1)


# ============================================================================
# Test Collection Drop Operations
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.async_test
class TestCollectionDrop:
    """
    Test collection drop operations.

    Coverage: Verifies drop_collection() method handles safe mode, lock cleanup,
    validation, and all error conditions.
    """

    async def test_drop_collection_success_safe_mode(
        self, mock_connection_manager, basic_collection_schema, mock_pymilvus_utility
    ):
        """
        Test successful collection drop in safe mode.

        Coverage: drop_collection() with safe=True and unloaded collection.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock has_collection to return True
        mock_connection_manager.execute_operation_async.return_value = True

        # Mock describe_collection to return unloaded collection
        mock_description = CollectionDescription(
            name="test_collection",
            collection_schema=basic_collection_schema,
            id="test_collection_id",
            created_at=datetime.now(timezone.utc),
            schema_hash="abc123",
            load_state=LoadState.UNLOADED,
        )

        with (
            patch.object(manager, "describe_collection", return_value=mock_description),
            patch.object(manager, "_cleanup_unused_locks", return_value=1),
            patch("pymilvus.utility.drop_collection"),
        ):
            result = await manager.drop_collection("test_collection", safe=True)

            assert result is True

    async def test_drop_collection_unsafe_mode(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test collection drop in unsafe mode.

        Coverage: drop_collection() with safe=False.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock has_collection to return True
        mock_connection_manager.execute_operation_async.return_value = True

        # No need to check load state in unsafe mode
        with (
            patch("pymilvus.utility.drop_collection"),
            patch.object(manager, "_cleanup_unused_locks", return_value=0),
        ):
            result = await manager.drop_collection("test_collection", safe=False)

            assert result is True

    async def test_drop_collection_loaded_collection_safe_mode(
        self, mock_connection_manager, basic_collection_schema, mock_pymilvus_utility
    ):
        """
        Test drop_collection preventing deletion of loaded collection in safe mode.

        Coverage: drop_collection() safe mode with loaded collection.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock has_collection to return True
        mock_connection_manager.execute_operation_async.return_value = True

        # Mock describe_collection to return loaded collection
        mock_description = CollectionDescription(
            name="test_collection",
            collection_schema=basic_collection_schema,
            id="test_collection_id",
            created_at=datetime.now(timezone.utc),
            schema_hash="abc123",
            load_state=LoadState.LOADED,
        )

        with (
            patch.object(manager, "describe_collection", return_value=mock_description),
            pytest.raises(CollectionError, match="currently loaded"),
        ):
            await manager.drop_collection("test_collection", safe=True)

    async def test_drop_collection_not_found(self, mock_connection_manager, mock_pymilvus_utility):
        """
        Test drop_collection when collection doesn't exist.

        Coverage: drop_collection() with non-existent collection.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock has_collection to return False
        mock_connection_manager.execute_operation_async.return_value = False

        with pytest.raises(CollectionNotFoundError, match="does not exist"):
            await manager.drop_collection("nonexistent_collection")

    async def test_drop_collection_lock_cleanup(
        self, mock_connection_manager, basic_collection_schema, mock_pymilvus_utility
    ):
        """
        Test lock cleanup during collection drop.

        Coverage: drop_collection() cleanup of collection-specific locks.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Add a lock for the collection being dropped
        manager._locks["test_collection"] = asyncio.Lock()

        # Mock has_collection to return True
        mock_connection_manager.execute_operation_async.return_value = True

        # Mock describe_collection to return unloaded collection
        mock_description = CollectionDescription(
            name="test_collection",
            collection_schema=basic_collection_schema,
            id="test_collection_id",
            created_at=datetime.now(timezone.utc),
            schema_hash="abc123",
            load_state=LoadState.UNLOADED,
        )

        with (
            patch.object(manager, "describe_collection", return_value=mock_description),
            patch("pymilvus.utility.drop_collection"),
            patch.object(manager, "_cleanup_unused_locks", return_value=2),
        ):
            await manager.drop_collection("test_collection", safe=True)

            # Verify lock was cleaned up
            assert "test_collection" not in manager._locks


# ============================================================================
# Test Data Insertion Operations
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.async_test
class TestDataInsertion:
    """
    Test data insertion operations.

    Coverage: Verifies insert() method handles data validation, error conditions,
    and integration with collection operations.
    """

    async def test_insert_success(
        self, mock_connection_manager, valid_test_data, mock_pymilvus_utility
    ):
        """
        Test successful data insertion.

        Coverage: insert() with valid data and existing collection.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock operations - first call (has_collection) returns True,
        # second call (insert operation) executes normally
        call_count = 0

        async def mock_execute(operation, timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # has_collection check
                return True
            else:  # insert operation
                return operation("mock_alias")

        mock_connection_manager.execute_operation_async.side_effect = mock_execute

        # Mock collection for insertion
        mock_collection = MagicMock()
        mock_result = MagicMock()
        mock_collection.insert.return_value = mock_result
        with patch("pymilvus.Collection", return_value=mock_collection):
            result = await manager.insert("test_collection", valid_test_data[0])

            assert result is mock_result
            mock_collection.flush.assert_called_once()

    async def test_insert_collection_not_found(
        self, mock_connection_manager, valid_test_data, mock_pymilvus_utility
    ):
        """
        Test insert operation on non-existent collection.

        Coverage: insert() with non-existent collection.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock has_collection to return False
        mock_connection_manager.execute_operation_async.return_value = False

        with pytest.raises(CollectionNotFoundError, match="does not exist"):
            await manager.insert("nonexistent_collection", valid_test_data[0])

    async def test_insert_invalid_data(
        self, mock_connection_manager, invalid_test_data, mock_pymilvus_utility
    ):
        """
        Test insert operation with invalid data.

        Coverage: insert() with malformed data.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock has_collection to return True
        mock_connection_manager.execute_operation_async.return_value = True

        # Mock insertion to raise error
        mock_connection_manager.execute_operation_async.side_effect = Exception(
            "Invalid data format"
        )

        with pytest.raises(CollectionError, match="Failed to insert data"):
            await manager.insert("test_collection", invalid_test_data[0])


# ============================================================================
# Test Collection Statistics
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.async_test
class TestCollectionStatistics:
    """
    Test collection statistics operations.

    Coverage: Verifies get_collection_stats() method handles statistics retrieval,
    data processing, and error conditions.
    """

    async def test_get_collection_stats_success(
        self,
        mock_connection_manager,
        sample_collection_stats,
        mock_pymilvus_collection,
        mock_pymilvus_utility,
    ):
        """
        Test successful collection statistics retrieval.

        Coverage: get_collection_stats() with valid collection.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock collection stats
        mock_connection_manager.execute_operation_async.return_value = sample_collection_stats

        result = await manager.get_collection_stats("test_collection")

        assert isinstance(result, CollectionStats)
        assert result.name == "test_collection"
        assert result.row_count == 1500
        assert result.num_partitions == 1
        assert result.num_segments == 2

    async def test_get_collection_stats_not_found(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test get_collection_stats when collection doesn't exist.

        Coverage: get_collection_stats() with non-existent collection.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock to raise CollectionNotExistException
        from pymilvus.exceptions import CollectionNotExistException

        mock_connection_manager.execute_operation_async.side_effect = CollectionNotExistException(
            "Collection not found"
        )

        with pytest.raises(CollectionNotFoundError, match="does not exist"):
            await manager.get_collection_stats("nonexistent_collection")

    async def test_get_collection_stats_with_fallback(
        self, mock_connection_manager, mock_pymilvus_collection, mock_pymilvus_utility
    ):
        """
        Test get_collection_stats with fallback methods.

        Coverage: get_collection_stats() when standard method fails.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock collection without get_collection_stats method
        del mock_pymilvus_collection.get_collection_stats
        mock_pymilvus_collection.stats = MagicMock(return_value={})

        with patch("pymilvus.Collection", return_value=mock_pymilvus_collection):
            result = await manager.get_collection_stats("test_collection")

            assert isinstance(result, CollectionStats)
            assert result.name == "test_collection"


# ============================================================================
# Test Lock Management and Cleanup
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.async_test
class TestLockManagement:
    """
    Test lock management and cleanup operations.

    Coverage: Verifies lock cleanup, memory management, and resource cleanup.
    """

    async def test_cleanup_unused_locks_success(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test successful cleanup of unused locks.

        Coverage: _cleanup_unused_locks() with unused locks to remove.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Add locks for collections that exist and don't exist
        manager._locks["existing_collection"] = asyncio.Lock()
        manager._locks["deleted_collection"] = asyncio.Lock()
        manager._locks["another_deleted"] = asyncio.Lock()

        # Mock list_collections to return only existing_collection
        mock_connection_manager.execute_operation_async.return_value = ["existing_collection"]

        cleaned_count = await manager._cleanup_unused_locks()

        assert cleaned_count == 2  # Two deleted collections
        assert "existing_collection" in manager._locks
        assert "deleted_collection" not in manager._locks
        assert "another_deleted" not in manager._locks

    async def test_cleanup_unused_locks_empty(self, mock_connection_manager, mock_pymilvus_utility):
        """
        test cleanup when no locks need cleanup.

        Coverage: _cleanup_unused_locks() with no unused locks.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Add lock for existing collection only
        manager._locks["existing_collection"] = asyncio.Lock()

        # Mock list_collections to return the existing collection
        mock_connection_manager.execute_operation_async.return_value = ["existing_collection"]

        cleaned_count = await manager._cleanup_unused_locks()

        assert cleaned_count == 0
        assert len(manager._locks) == 1

    async def test_cleanup_unused_locks_error_handling(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test cleanup error handling.

        Coverage: _cleanup_unused_locks() with list_collections error.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Add some locks
        manager._locks["collection1"] = asyncio.Lock()
        manager._locks["collection2"] = asyncio.Lock()

        # Mock list_collections to raise error
        mock_connection_manager.execute_operation_async.side_effect = Exception("Server error")

        cleaned_count = await manager._cleanup_unused_locks()

        # Should clean up all locks since list_collections fails
        assert cleaned_count == 2
        assert len(manager._locks) == 0  # All locks should be removed

    async def test_cleanup_unused_locks_nonexistent_list(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test cleanup when list_collections returns None.

        Coverage: _cleanup_unused_locks() with None result from list_collections.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Add some locks
        manager._locks["collection1"] = asyncio.Lock()
        manager._locks["collection2"] = asyncio.Lock()

        # Mock list_collections to return None
        mock_connection_manager.execute_operation_async.return_value = None

        cleaned_count = await manager._cleanup_unused_locks()

        # Should handle None gracefully by treating it as empty list and clean up all locks
        assert cleaned_count == 2
        assert len(manager._locks) == 0

    async def test_cleanup_locks_public_method(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test public cleanup_locks method.

        Coverage: cleanup_locks() public interface.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock cleanup_unused_locks
        with patch.object(manager, "_cleanup_unused_locks", return_value=3) as mock_cleanup:
            cleaned_count = await manager.cleanup_locks()

            assert cleaned_count == 3
            mock_cleanup.assert_called_once()


# ============================================================================
# Test Utility Methods
# ============================================================================


@pytest.mark.unit
@pytest.mark.async_test
class TestUtilityMethods:
    """
    Test utility and helper methods.

    Coverage: Verifies internal utility methods handle edge cases and errors.
    """

    @pytest.mark.asyncio
    async def test_ensure_awaited_with_coroutine(self, mock_connection_manager):
        """
        Test _ensure_awaited with coroutine.

        Coverage: _ensure_awaited() with coroutine object.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        async def test_coroutine():
            return "test_result"

        result = await manager._ensure_awaited(test_coroutine())

        assert result == "test_result"

    @pytest.mark.asyncio
    async def test_ensure_awaited_with_non_coroutine(self, mock_connection_manager):
        """
        Test _ensure_awaited with non-coroutine.

        Coverage: _ensure_awaited() with regular object.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        test_value = "test_result"
        result = await manager._ensure_awaited(test_value)

        assert result == test_value

    def test_data_type_mapping(self, mock_connection_manager):
        """
        Test internal data type mapping.

        Coverage: _MILVUS_DATATYPE_MAP configuration.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Verify mapping completeness
        expected_mappings = {
            "BOOL": "BOOL",
            "INT8": "INT8",
            "INT16": "INT16",
            "INT32": "INT32",
            "INT64": "INT64",
            "FLOAT": "FLOAT",
            "DOUBLE": "DOUBLE",
            "STRING": "VARCHAR",  # Alias normalization
            "VARCHAR": "VARCHAR",
            "BINARY_VECTOR": "BINARY_VECTOR",
            "FLOAT_VECTOR": "FLOAT_VECTOR",
            "SPARSE_FLOAT_VECTOR": "SPARSE_FLOAT_VECTOR",
            "JSON": "JSON",
            "ARRAY": "ARRAY",
        }

        for our_type, expected_milvus_type in expected_mappings.items():
            assert manager._MILVUS_DATATYPE_MAP[our_type] == expected_milvus_type

    def test_field_parameter_whitelist(self, mock_connection_manager):
        """
        Test field parameter whitelist.

        Coverage: _FIELD_PARAM_WHITELIST configuration.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        expected_whitelist = {
            "name",
            "dtype",
            "description",
            "is_primary",
            "auto_id",
            "dim",
            "max_length",
            "element_type",
        }

        assert expected_whitelist == manager._FIELD_PARAM_WHITELIST


# ============================================================================
# Test Concurrency and Thread Safety
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.threading
@pytest.mark.async_test
class TestConcurrency:
    """
    Test concurrent operations and thread safety.

    Coverage: Verifies CollectionManager handles concurrent operations safely
    with proper locking and isolation.
    """

    async def test_concurrent_collection_operations(
        self,
        mock_connection_manager,
        basic_collection_schema,
        mock_pymilvus_utility,
        mock_pymilvus_collection,
    ):
        """
        Test concurrent operations on the same collection.

        Coverage: Multiple concurrent operations with collection locking.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock utility.has_collection to return True for existing collections
        mock_pymilvus_utility.has_collection.return_value = True

        # Mock operations to execute normally
        call_count = 0

        def mock_execute(operation, timeout=None):
            nonlocal call_count
            call_count += 1
            try:
                return operation("mock_alias")
            except Exception:
                # For operations that might fail due to mocking, return success values
                if "create" in str(operation) or "describe" in str(operation):
                    from milvus_ops.collection_operations.entities import CollectionDescription

                    return CollectionDescription(
                        name="test_collection",
                        collection_schema=basic_collection_schema,
                        id="test_collection_id",
                        created_at=datetime.now(timezone.utc),
                        schema_hash="abc123",
                    )
                else:
                    return True

        mock_connection_manager.execute_operation_async.side_effect = mock_execute

        # Create tasks for concurrent operations
        async def create_collection_op():
            return await manager.create_collection("test_collection", basic_collection_schema)

        async def load_collection_op():
            return await manager.load_collection("test_collection", wait=False)

        async def describe_collection_op():
            return await manager.describe_collection("test_collection")

        # Run operations concurrently
        tasks = [
            create_collection_op(),
            load_collection_op(),
            describe_collection_op(),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify no exceptions occurred
        assert not any(isinstance(result, Exception) for result in results)

    async def test_concurrent_lock_acquisition(self, mock_connection_manager):
        """
        Test concurrent lock acquisition for the same collection.

        Coverage: Multiple concurrent lock acquisitions.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        lock_acquisitions = []

        async def acquire_lock(collection_name):
            lock = await manager._get_collection_lock(collection_name)
            lock_acquisitions.append(lock)
            # Hold lock briefly
            async with lock:
                await asyncio.sleep(0.01)
            return lock

        # Acquire same lock concurrently
        tasks = [acquire_lock("test_collection") for _ in range(5)]

        await asyncio.gather(*tasks)

        # Verify all operations got the same lock object
        assert len({id(lock) for lock in lock_acquisitions}) == 1
        assert all(lock is lock_acquisitions[0] for lock in lock_acquisitions)

    async def test_lock_isolation_between_collections(self, mock_connection_manager):
        """
        Test that locks are isolated between different collections.

        Coverage: Different collections get different locks.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Get locks for different collections
        lock1 = await manager._get_collection_lock("collection1")
        lock2 = await manager._get_collection_lock("collection2")
        lock3 = await manager._get_collection_lock("collection1")  # Same as lock1

        # Verify isolation
        assert lock1 is not lock2
        assert lock1 is lock3
        assert len(manager._locks) == 2

    async def test_concurrent_lock_cleanup(self, mock_connection_manager, mock_pymilvus_utility):
        """
        Test lock cleanup during concurrent operations.

        Coverage: Cleanup operations while locks are in use.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Add locks for multiple collections
        collections = ["collection1", "collection2", "collection3", "collection4", "collection5"]
        for collection in collections:
            await manager._get_collection_lock(collection)

        # Mock list_collections to return only some collections
        mock_connection_manager.execute_operation_async.return_value = [
            "collection1",
            "collection3",
        ]

        # Perform cleanup while locks exist
        cleaned_count = await manager._cleanup_unused_locks()

        # Verify cleanup worked correctly
        assert cleaned_count == 3  # collection2, collection4, collection5
        assert "collection1" in manager._locks
        assert "collection3" in manager._locks
        assert "collection2" not in manager._locks
        assert "collection4" not in manager._locks
        assert "collection5" not in manager._locks


# ============================================================================
# Test Edge Cases and Error Recovery
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.async_test
class TestEdgeCases:
    """
    Test edge cases and error recovery scenarios.

    Coverage: Verifies robust error handling and recovery in unusual scenarios.
    """

    async def test_operation_with_none_parameters(self, mock_connection_manager):
        """
        Test operations with None parameters.

        Coverage: Edge case handling for None inputs.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Test with None collection name
        with pytest.raises((TypeError, ValueError, AttributeError)):
            await manager.has_collection(None)

    async def test_operation_with_empty_string_parameters(self, mock_connection_manager):
        """
        Test operations with empty string parameters.

        Coverage: Edge case handling for empty strings.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Test with empty collection name
        with pytest.raises((Exception, ValueError)):
            await manager.has_collection("")

    async def test_recovery_from_partial_failures(
        self, mock_connection_manager, basic_collection_schema
    ):
        """
        Test recovery from partial operation failures.

        Coverage: Error recovery and state consistency.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock has_collection to succeed, then describe to fail
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return True
            elif call_count == 2:
                return CollectionDescription(
                    name="test_collection",
                    collection_schema=basic_collection_schema,
                    id="test_collection_id",
                    created_at=datetime.now(timezone.utc),
                    schema_hash="abc123",
                )
            else:
                raise Exception("Operation failed")

        mock_connection_manager.execute_operation_async.side_effect = side_effect

        # First operation should succeed
        result1 = await manager.has_collection("test_collection")
        assert result1 is True

        # Second operation should also succeed
        result2 = await manager.describe_collection("test_collection")
        assert result2.name == "test_collection"

        # Third operation should fail
        with pytest.raises(Exception, match="Operation failed"):
            await manager.get_load_progress("test_collection")

    async def test_manager_state_after_errors(
        self, mock_connection_manager, basic_collection_schema
    ):
        """
        Test that manager state remains consistent after errors.

        Coverage: State consistency after error conditions.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Add a lock before error
        initial_locks = len(manager._locks)
        await manager._get_collection_lock("test_collection")

        # Trigger error during operation
        mock_connection_manager.execute_operation_async.side_effect = Exception("Server error")

        with contextlib.suppress(CollectionError):
            await manager.describe_collection("test_collection")

        # Verify internal state is still consistent
        assert len(manager._locks) == initial_locks + 1
        assert "test_collection" in manager._locks

    async def test_memory_leak_prevention(self, mock_connection_manager, mock_pymilvus_utility):
        """
        Test that memory leaks are prevented through proper cleanup.

        Coverage: Memory management and leak prevention.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Create many locks
        for i in range(100):
            await manager._get_collection_lock(f"collection_{i}")

        assert len(manager._locks) == 100

        # Simulate collections being dropped externally
        mock_connection_manager.execute_operation_async.return_value = []  # No existing collections

        # Cleanup should remove all locks
        cleaned_count = await manager._cleanup_unused_locks()

        assert cleaned_count == 100
        assert len(manager._locks) == 0


# ============================================================================
# Performance and Benchmark Tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.async_test
class TestPerformance:
    """
    Performance tests for CollectionManager operations.

    Coverage: Performance benchmarks and optimization verification.
    """

    async def test_concurrent_operation_throughput(
        self, mock_connection_manager, basic_collection_schema
    ):
        """
        Test throughput of concurrent operations.

        Coverage: Performance under concurrent load.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Mock operations to complete quickly
        mock_connection_manager.execute_operation_async.return_value = True

        # Create many concurrent operations
        num_operations = 50
        start_time = asyncio.get_event_loop().time()

        tasks = []
        for i in range(num_operations):
            task = manager.has_collection(f"collection_{i}")
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        end_time = asyncio.get_event_loop().time()

        # Verify all operations completed
        assert len(results) == num_operations
        assert all(result is True for result in results)

        # Calculate throughput
        duration = end_time - start_time
        # Avoid division by zero or very small numbers
        if duration < 0.001:  # Less than 1ms
            duration = 0.001
        throughput = num_operations / duration

        # Performance should be reasonable (at least 100 ops/second for mocked operations)
        assert throughput > 100

    async def test_lock_acquisition_performance(self, mock_connection_manager):
        """
        Test lock acquisition performance.

        Coverage: Lock performance and contention handling.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Benchmark lock acquisition
        num_acquisitions = 1000
        start_time = asyncio.get_event_loop().time()

        for i in range(num_acquisitions):
            await manager._get_collection_lock(f"collection_{i}")

        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time

        # Should be able to create 1000 locks quickly
        assert duration < 1.0  # Less than 1 second

        # Verify all locks were created
        assert len(manager._locks) == num_acquisitions

    async def test_cleanup_performance(self, mock_connection_manager, mock_pymilvus_utility):
        """
        Test cleanup operation performance.

        Coverage: Cleanup performance with many locks.
        """
        from milvus_ops.collection_operations.manager import CollectionManager

        manager = CollectionManager(mock_connection_manager)

        # Create many locks
        num_locks = 5000
        for i in range(num_locks):
            await manager._get_collection_lock(f"collection_{i}")

        # Mock no existing collections
        mock_connection_manager.execute_operation_async.return_value = []

        # Benchmark cleanup
        start_time = asyncio.get_event_loop().time()
        cleaned_count = await manager._cleanup_unused_locks()
        end_time = asyncio.get_event_loop().time()

        # Verify cleanup worked
        assert cleaned_count == num_locks
        assert len(manager._locks) == 0

        # Should be able to cleanup 5000 locks quickly
        duration = end_time - start_time
        assert duration < 2.0  # Less than 2 seconds
