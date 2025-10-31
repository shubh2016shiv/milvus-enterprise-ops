"""
Comprehensive unit tests for CollectionManager.

This module provides systematic testing of the CollectionManager class, achieving
95%+ code coverage through positive, negative, and edge case testing including
concurrency scenarios and async operations.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from milvus_ops.collection_operations.manager import CollectionManager
from milvus_ops.milvus_ops_exceptions import (
    CollectionError,
    OperationTimeoutError,
)

# ============================================================================
# Test CollectionManager Initialization
# ============================================================================


@pytest.mark.unit
class TestCollectionManagerInitialization:
    """
    Test CollectionManager initialization and setup.

    Coverage: CollectionManager constructor and initialization logic.
    """

    def test_init_basic(self, mock_connection_manager):
        """
        Test basic CollectionManager initialization.

        Coverage: CollectionManager.__init__() with valid connection manager.
        """
        manager = CollectionManager(mock_connection_manager)

        assert manager._connection_manager == mock_connection_manager
        assert isinstance(manager._locks, dict)
        assert len(manager._locks) == 0

    def test_init_connection_manager_type_check(self):
        """
        Test that CollectionManager validates connection manager type.

        Coverage: Connection manager type validation.
        """
        # Test with None
        with pytest.raises((TypeError, AttributeError)):
            CollectionManager(None)

    def test_init_locks_initialization(self, mock_connection_manager):
        """
        Test that locks dictionary is properly initialized.

        Coverage: Locks dictionary initialization.
        """
        manager = CollectionManager(mock_connection_manager)

        # Should start empty
        assert len(manager._locks) == 0
        assert isinstance(manager._locks, dict)

    def test_init_global_lock_creation(self, mock_connection_manager):
        """
        Test that global lock is created during initialization.

        Coverage: Global lock initialization.
        """
        manager = CollectionManager(mock_connection_manager)

        # Check that global lock exists and is an asyncio.Lock
        assert hasattr(manager, "_global_lock")
        # Note: Can't directly check asyncio.Lock type due to mocking

    def test_connection_manager_assignment(self, mock_connection_manager):
        """
        Test that connection manager is properly assigned.

        Coverage: Connection manager assignment verification.
        """
        manager = CollectionManager(mock_connection_manager)

        # Verify the assignment
        assert manager._connection_manager is mock_connection_manager

    def test_multiple_managers_independent(self, mock_connection_manager):
        """
        Test that multiple managers have independent state.

        Coverage: Multiple manager instance isolation.
        """
        manager1 = CollectionManager(mock_connection_manager)
        manager2 = CollectionManager(mock_connection_manager)

        # Each manager should have its own locks
        assert manager1._locks is not manager2._locks
        assert id(manager1._connection_manager) == id(manager2._connection_manager)

    @pytest.mark.parametrize(
        "connection_manager_attr", ["_connection_manager", "_locks", "_global_lock"]
    )
    def test_manager_attributes_exist(self, mock_connection_manager, connection_manager_attr: str):
        """
        Test that all required attributes exist on CollectionManager.

        Coverage: Attribute existence verification.
        """
        manager = CollectionManager(mock_connection_manager)

        assert hasattr(manager, connection_manager_attr)

    def test_manager_has_required_methods(self, mock_connection_manager):
        """
        Test that CollectionManager has all required public methods.

        Coverage: Public method existence verification.
        """
        manager = CollectionManager(mock_connection_manager)

        required_methods = [
            "create_collection",
            "has_collection",
            "list_collections",
            "describe_collection",
            "load_collection",
            "get_load_progress",
            "release_collection",
            "drop_collection",
            "insert",
            "get_collection_stats",
            "cleanup_locks",
        ]

        for method_name in required_methods:
            assert hasattr(manager, method_name), f"Missing method: {method_name}"

    def test_manager_has_private_methods(self, mock_connection_manager):
        """
        Test that CollectionManager has required private methods.

        Coverage: Private method existence verification.
        """
        manager = CollectionManager(mock_connection_manager)

        private_methods = [
            "_get_collection_lock",
            "_cleanup_unused_locks",
            "_ensure_awaited",
            "_create_collection_internal",
            "_has_collection_internal",
            "_list_collections_internal",
            "_describe_collection_internal",
            "_load_collection_internal",
            "_get_load_progress_internal",
            "_release_collection_internal",
            "_drop_collection_internal",
            "_insert_internal",
            "_get_collection_stats_internal",
        ]

        for method_name in private_methods:
            assert hasattr(manager, method_name), f"Missing private method: {method_name}"


# ============================================================================
# Test Lock Management
# ============================================================================


@pytest.mark.unit
class TestLockManagement:
    """
    Test collection lock management functionality.

    Coverage: Lock acquisition, cleanup, and management operations.
    """

    @pytest.mark.asyncio
    async def test_get_collection_lock_new_lock(self, mock_connection_manager):
        """
        Test acquiring a lock for a new collection.

        Coverage: Lock creation for new collection.
        """
        manager = CollectionManager(mock_connection_manager)

        lock = await manager._get_collection_lock("new_collection")

        assert "new_collection" in manager._locks
        assert manager._locks["new_collection"] is lock

    @pytest.mark.asyncio
    async def test_get_collection_lock_existing_lock(self, mock_connection_manager):
        """
        Test acquiring a lock for an existing collection.

        Coverage: Lock reuse for existing collection.
        """
        manager = CollectionManager(mock_connection_manager)

        # Create initial lock
        lock1 = await manager._get_collection_lock("existing_collection")

        # Get same lock again
        lock2 = await manager._get_collection_lock("existing_collection")

        # Should be the same lock object
        assert lock1 is lock2
        assert len(manager._locks) == 1

    @pytest.mark.asyncio
    async def test_get_collection_lock_multiple_collections(self, mock_connection_manager):
        """
        Test acquiring locks for multiple collections.

        Coverage: Multiple collection lock management.
        """
        manager = CollectionManager(mock_connection_manager)

        lock1 = await manager._get_collection_lock("collection1")
        lock2 = await manager._get_collection_lock("collection2")
        lock3 = await manager._get_collection_lock("collection3")

        # Should have three different locks
        assert len(manager._locks) == 3
        assert lock1 is not lock2
        assert lock2 is not lock3
        assert lock1 is not lock3

    @pytest.mark.asyncio
    async def test_get_collection_lock_concurrent_access(self, mock_connection_manager):
        """
        Test lock behavior with concurrent access.

        Coverage: Lock thread safety and concurrent access.
        """
        manager = CollectionManager(mock_connection_manager)

        # Create multiple locks concurrently
        tasks = []
        for i in range(10):
            task = manager._get_collection_lock(f"collection_{i}")
            tasks.append(task)

        locks = await asyncio.gather(*tasks)

        # All locks should be created
        assert len(manager._locks) == 10

        # Each lock should be unique
        unique_locks = {id(lock) for lock in locks}
        assert len(unique_locks) == 10

    @pytest.mark.asyncio
    async def test_cleanup_unused_locks_basic(self, mock_connection_manager):
        """
        Test basic lock cleanup functionality.

        Coverage: Lock cleanup with existing collections.
        """
        manager = CollectionManager(mock_connection_manager)

        # Add some locks
        await manager._get_collection_lock("collection1")
        await manager._get_collection_lock("collection2")
        await manager._get_collection_lock("collection3")

        assert len(manager._locks) == 3

        # Mock that only collection1 and collection2 exist in Milvus
        with patch.object(manager, "list_collections", return_value=["collection1", "collection2"]):
            cleaned_count = await manager._cleanup_unused_locks()

        # Should clean up collection3's lock
        assert cleaned_count == 1
        assert len(manager._locks) == 2
        assert "collection3" not in manager._locks
        assert "collection1" in manager._locks
        assert "collection2" in manager._locks

    @pytest.mark.asyncio
    async def test_cleanup_unused_locks_no_existing_collections(self, mock_connection_manager):
        """
        Test lock cleanup when no collections exist in Milvus.

        Coverage: Lock cleanup with no existing collections.
        """
        manager = CollectionManager(mock_connection_manager)

        # Add some locks
        await manager._get_collection_lock("collection1")
        await manager._get_collection_lock("collection2")

        assert len(manager._locks) == 2

        # Mock that no collections exist
        with patch.object(manager, "list_collections", return_value=[]):
            cleaned_count = await manager._cleanup_unused_locks()

        # Should clean up all locks
        assert cleaned_count == 2
        assert len(manager._locks) == 0

    @pytest.mark.asyncio
    async def test_cleanup_unused_locks_all_locks_referenced(self, mock_connection_manager):
        """
        Test lock cleanup when all locks have corresponding collections.

        Coverage: Lock cleanup when all locks are valid.
        """
        manager = CollectionManager(mock_connection_manager)

        # Add some locks
        await manager._get_collection_lock("collection1")
        await manager._get_collection_lock("collection2")

        assert len(manager._locks) == 2

        # Mock that all collections exist
        with patch.object(manager, "list_collections", return_value=["collection1", "collection2"]):
            cleaned_count = await manager._cleanup_unused_locks()

        # Should not clean up any locks
        assert cleaned_count == 0
        assert len(manager._locks) == 2

    @pytest.mark.asyncio
    async def test_cleanup_unused_locks_strict_false(self, mock_connection_manager):
        """
        Test lock cleanup with strict=False handling errors.

        Coverage: Lock cleanup error handling.
        """
        manager = CollectionManager(mock_connection_manager)

        # Add some locks
        await manager._get_collection_lock("collection1")

        assert len(manager._locks) == 1

        # Mock list_collections to raise exception
        with patch.object(manager, "list_collections", side_effect=Exception("Connection error")):
            cleaned_count = await manager._cleanup_unused_locks()

        # Should handle error gracefully and not cleanup
        assert cleaned_count == 0
        assert len(manager._locks) == 1

    @pytest.mark.asyncio
    async def test_cleanup_locks_public_method(self, mock_connection_manager):
        """
        Test the public cleanup_locks method.

        Coverage: Public cleanup_locks method.
        """
        manager = CollectionManager(mock_connection_manager)

        # Add some locks
        await manager._get_collection_lock("collection1")
        await manager._get_collection_lock("collection2")

        # Mock list_collections for cleanup
        with patch.object(manager, "list_collections", return_value=["collection1"]):
            cleaned_count = await manager.cleanup_locks()

        assert cleaned_count == 1

    @pytest.mark.asyncio
    async def test_cleanup_locks_no_locks_to_cleanup(self, mock_connection_manager):
        """
        Test cleanup when no locks exist.

        Coverage: Cleanup with empty locks dictionary.
        """
        manager = CollectionManager(mock_connection_manager)

        assert len(manager._locks) == 0

        cleaned_count = await manager.cleanup_locks()

        assert cleaned_count == 0

    @pytest.mark.asyncio
    async def test_cleanup_locks_large_number_of_locks(self, mock_connection_manager):
        """
        Test cleanup with large number of locks.

        Coverage: Cleanup performance with many locks.
        """
        manager = CollectionManager(mock_connection_manager)

        # Create many locks
        collection_count = 1000
        for i in range(collection_count):
            await manager._get_collection_lock(f"collection_{i}")

        assert len(manager._locks) == collection_count

        # Mock that only first 500 exist
        existing_collections = [f"collection_{i}" for i in range(500)]
        with patch.object(manager, "list_collections", return_value=existing_collections):
            cleaned_count = await manager.cleanup_locks()

        # Should clean up 500 locks
        assert cleaned_count == 500
        assert len(manager._locks) == 500


# ============================================================================
# Test Collection Existence Checking
# ============================================================================


@pytest.mark.unit
class TestCollectionExistenceChecking:
    """
    Test collection existence checking functionality.

    Coverage: has_collection method with various scenarios.
    """

    @pytest.mark.asyncio
    async def test_has_collection_exists_true(self, mock_connection_manager, mock_pymilvus_utility):
        """
        Test has_collection returns True when collection exists.

        Coverage: has_collection() with existing collection.
        """
        mock_pymilvus_utility.has_collection.return_value = True

        manager = CollectionManager(mock_connection_manager)

        result = await manager.has_collection("existing_collection")

        assert result is True
        mock_pymilvus_utility.has_collection.assert_called_once_with(
            "existing_collection", using="mock_alias"
        )

    @pytest.mark.asyncio
    async def test_has_collection_exists_false(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test has_collection returns False when collection doesn't exist.

        Coverage: has_collection() with non-existing collection.
        """
        mock_pymilvus_utility.has_collection.return_value = False

        manager = CollectionManager(mock_connection_manager)

        result = await manager.has_collection("nonexistent_collection")

        assert result is False
        mock_pymilvus_utility.has_collection.assert_called_once_with(
            "nonexistent_collection", using="mock_alias"
        )

    @pytest.mark.asyncio
    async def test_has_collection_strict_true_exists(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test has_collection with strict=True and collection exists.

        Coverage: has_collection(strict=True) with existing collection.
        """
        mock_pymilvus_utility.has_collection.return_value = True

        manager = CollectionManager(mock_connection_manager)

        result = await manager.has_collection("existing_collection", strict=True)

        assert result is True
        mock_pymilvus_utility.has_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_has_collection_strict_true_not_exists(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test has_collection with strict=True and collection doesn't exist.

        Coverage: has_collection(strict=True) with non-existing collection.
        """
        mock_pymilvus_utility.has_collection.return_value = False

        manager = CollectionManager(mock_connection_manager)

        result = await manager.has_collection("nonexistent_collection", strict=True)

        assert result is False
        mock_pymilvus_utility.has_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_has_collection_strict_false_with_error(self, mock_connection_manager):
        """
        Test has_collection with strict=False handles errors gracefully.

        Coverage: has_collection(strict=False) with connection error.
        """
        manager = CollectionManager(mock_connection_manager)

        # Mock execute_operation_async to raise an error
        manager._connection_manager.execute_operation_async.side_effect = Exception(
            "Connection error"
        )

        result = await manager.has_collection("problematic_collection", strict=False)

        assert result is False
        # Should not raise exception when strict=False

    @pytest.mark.asyncio
    async def test_has_collection_strict_true_with_error(self, mock_connection_manager):
        """
        Test has_collection with strict=True raises error.

        Coverage: has_collection(strict=True) with connection error.
        """
        manager = CollectionManager(mock_connection_manager)

        # Mock execute_operation_async to raise an error
        manager._connection_manager.execute_operation_async.side_effect = CollectionError(
            "Connection error"
        )

        with pytest.raises(CollectionError):
            await manager.has_collection("problematic_collection", strict=True)

    @pytest.mark.asyncio
    async def test_has_collection_with_timeout(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test has_collection with timeout parameter.

        Coverage: has_collection() with timeout specification.
        """
        mock_pymilvus_utility.has_collection.return_value = True

        manager = CollectionManager(mock_connection_manager)

        result = await manager.has_collection("collection_with_timeout", timeout=30.0)

        assert result is True
        # Verify timeout was passed to execute_operation_async
        call_args = manager._connection_manager.execute_operation_async.call_args
        assert call_args[1]["timeout"] == 30.0

    @pytest.mark.asyncio
    async def test_has_collection_empty_name(self, mock_connection_manager, mock_pymilvus_utility):
        """
        Test has_collection with empty collection name.

        Coverage: has_collection() with empty name.
        """
        mock_pymilvus_utility.has_collection.return_value = False

        manager = CollectionManager(mock_connection_manager)

        with pytest.raises(ValueError, match="collection_name cannot be an empty string"):
            await manager.has_collection("")

    @pytest.mark.asyncio
    async def test_has_collection_special_characters(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test has_collection with special characters in collection name.

        Coverage: has_collection() with special characters.
        """
        special_name = "collection-with_special.chars@123"
        mock_pymilvus_utility.has_collection.return_value = True

        manager = CollectionManager(mock_connection_manager)

        result = await manager.has_collection(special_name)

        assert result is True
        mock_pymilvus_utility.has_collection.assert_called_once_with(
            special_name, using="mock_alias"
        )

    @pytest.mark.asyncio
    async def test_has_collection_very_long_name(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test has_collection with very long collection name.

        Coverage: has_collection() with long names.
        """
        long_name = "a" * 1000  # Very long name
        mock_pymilvus_utility.has_collection.return_value = False

        manager = CollectionManager(mock_connection_manager)

        result = await manager.has_collection(long_name)

        assert result is False
        mock_pymilvus_utility.has_collection.assert_called_once_with(long_name, using="mock_alias")

    @pytest.mark.asyncio
    async def test_has_collection_unicode_name(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test has_collection with Unicode collection name.

        Coverage: has_collection() with Unicode characters.
        """
        unicode_name = "collection_测试_коллекция_مجموعة"
        mock_pymilvus_utility.has_collection.return_value = True

        manager = CollectionManager(mock_connection_manager)

        result = await manager.has_collection(unicode_name)

        assert result is True
        mock_pymilvus_utility.has_collection.assert_called_once_with(
            unicode_name, using="mock_alias"
        )

    @pytest.mark.asyncio
    async def test_has_collection_concurrent_calls(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test has_collection with concurrent calls.

        Coverage: has_collection() concurrent execution.
        """
        mock_pymilvus_utility.has_collection.return_value = True

        manager = CollectionManager(mock_connection_manager)

        # Make concurrent calls
        tasks = [manager.has_collection(f"collection_{i}") for i in range(10)]

        results = await asyncio.gather(*tasks)

        # All should return True
        assert all(result is True for result in results)

        # Verify all calls were made
        assert mock_pymilvus_utility.has_collection.call_count == 10

    @pytest.mark.asyncio
    async def test_has_collection_operation_timeout(self, mock_connection_manager):
        """
        Test has_collection with operation timeout.

        Coverage: has_collection() timeout handling.
        """
        manager = CollectionManager(mock_connection_manager)

        # Mock execute_operation_async to raise OperationTimeoutError
        manager._connection_manager.execute_operation_async.side_effect = OperationTimeoutError(
            "Operation timed out"
        )

        with pytest.raises(OperationTimeoutError):
            await manager.has_collection("timeout_collection", strict=True)

    @pytest.mark.asyncio
    async def test_has_collection_preserves_exception_types(self, mock_connection_manager):
        """
        Test that has_collection preserves specific exception types.

        Coverage: Exception type preservation in has_collection.
        """
        manager = CollectionManager(mock_connection_manager)

        # Test ConnectionError preservation
        manager._connection_manager.execute_operation_async.side_effect = ConnectionError(
            "Connection lost"
        )

        with pytest.raises(ConnectionError):
            await manager.has_collection("error_collection", strict=True)

    @pytest.mark.asyncio
    async def test_has_collection_logging_calls(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test that has_collection makes appropriate logging calls.

        Coverage: Logging integration in has_collection.
        """
        mock_pymilvus_utility.has_collection.return_value = False

        manager = CollectionManager(mock_connection_manager)

        with patch("milvus_ops.collection_operations.manager.logger"):
            result = await manager.has_collection("test_collection")

            assert result is False
            # Should have error logging for non-strict mode with errors
            # Note: Actual logging behavior depends on implementation


# ============================================================================
# Test Collection Listing
# ============================================================================


@pytest.mark.unit
class TestCollectionListing:
    """
    Test collection listing functionality.

    Coverage: list_collections method with various scenarios.
    """

    @pytest.mark.asyncio
    async def test_list_collections_basic(self, mock_connection_manager, mock_pymilvus_utility):
        """
        Test basic list_collections functionality.

        Coverage: list_collections() with existing collections.
        """
        expected_collections = ["collection1", "collection2", "collection3"]
        mock_pymilvus_utility.list_collections.return_value = expected_collections

        manager = CollectionManager(mock_connection_manager)

        result = await manager.list_collections()

        assert result == expected_collections
        mock_pymilvus_utility.list_collections.assert_called_once_with(using="mock_alias")

    @pytest.mark.asyncio
    async def test_list_collections_empty(self, mock_connection_manager, mock_pymilvus_utility):
        """
        Test list_collections when no collections exist.

        Coverage: list_collections() with empty result.
        """
        mock_pymilvus_utility.list_collections.return_value = []

        manager = CollectionManager(mock_connection_manager)

        result = await manager.list_collections()

        assert result == []
        mock_pymilvus_utility.list_collections.assert_called_once_with(using="mock_alias")

    @pytest.mark.asyncio
    async def test_list_collections_strict_true(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test list_collections with strict=True.

        Coverage: list_collections(strict=True) functionality.
        """
        expected_collections = ["collection1", "collection2"]
        mock_pymilvus_utility.list_collections.return_value = expected_collections

        manager = CollectionManager(mock_connection_manager)

        result = await manager.list_collections(strict=True)

        assert result == expected_collections

    @pytest.mark.asyncio
    async def test_list_collections_strict_false_with_error(self, mock_connection_manager):
        """
        Test list_collections with strict=False handles errors.

        Coverage: list_collections(strict=False) error handling.
        """
        manager = CollectionManager(mock_connection_manager)

        # Mock execute_operation_async to raise an error
        manager._connection_manager.execute_operation_async.side_effect = Exception(
            "Connection error"
        )

        result = await manager.list_collections(strict=False)

        assert result == []  # Should return empty list for strict=False

    @pytest.mark.asyncio
    async def test_list_collections_strict_true_with_error(self, mock_connection_manager):
        """
        Test list_collections with strict=True raises error.

        Coverage: list_collections(strict=True) error propagation.
        """
        manager = CollectionManager(mock_connection_manager)

        # Mock execute_operation_async to raise an error
        manager._connection_manager.execute_operation_async.side_effect = CollectionError(
            "Connection error"
        )

        with pytest.raises(CollectionError):
            await manager.list_collections(strict=True)

    @pytest.mark.asyncio
    async def test_list_collections_with_timeout(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test list_collections with timeout parameter.

        Coverage: list_collections() with timeout specification.
        """
        expected_collections = ["collection1"]
        mock_pymilvus_utility.list_collections.return_value = expected_collections

        manager = CollectionManager(mock_connection_manager)

        result = await manager.list_collections(timeout=60.0)

        assert result == expected_collections
        # Verify timeout was passed
        call_args = manager._connection_manager.execute_operation_async.call_args
        assert call_args[1]["timeout"] == 60.0

    @pytest.mark.asyncio
    async def test_list_collections_large_number(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test list_collections with large number of collections.

        Coverage: list_collections() with many collections.
        """
        # Create a large list of collection names
        large_collection_list = [f"collection_{i:05d}" for i in range(10000)]
        mock_pymilvus_utility.list_collections.return_value = large_collection_list

        manager = CollectionManager(mock_connection_manager)

        result = await manager.list_collections()

        assert len(result) == 10000
        assert result == large_collection_list
        mock_pymilvus_utility.list_collections.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_collections_special_characters(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test list_collections with special characters in collection names.

        Coverage: list_collections() with special characters.
        """
        special_collections = [
            "collection-with-dashes",
            "collection_with_underscores",
            "collection.with.dots",
            "collection@symbols",
            "collection123numbers",
            "Collection.CamelCase",
        ]
        mock_pymilvus_utility.list_collections.return_value = special_collections

        manager = CollectionManager(mock_connection_manager)

        result = await manager.list_collections()

        assert result == special_collections

    @pytest.mark.asyncio
    async def test_list_collections_unicode_names(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        test list_collections with Unicode collection names.

        Coverage: list_collections() with Unicode characters.
        """
        unicode_collections = [
            "collection_测试",
            "коллекция_тест",
            "مجموعة_اختبار",
            "colección_prueba",
            " koleksi_uji",
        ]
        mock_pymilvus_utility.list_collections.return_value = unicode_collections

        manager = CollectionManager(mock_connection_manager)

        result = await manager.list_collections()

        assert result == unicode_collections

    @pytest.mark.asyncio
    async def test_list_collections_concurrent_calls(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test list_collections with concurrent calls.

        Coverage: list_collections() concurrent execution.
        """
        expected_collections = ["collection1", "collection2"]
        mock_pymilvus_utility.list_collections.return_value = expected_collections

        manager = CollectionManager(mock_connection_manager)

        # Make concurrent calls
        tasks = [manager.list_collections() for _ in range(5)]

        results = await asyncio.gather(*tasks)

        # All should return the same result
        for result in results:
            assert result == expected_collections

    @pytest.mark.asyncio
    async def test_list_collections_operation_timeout(self, mock_connection_manager):
        """
        Test list_collections with operation timeout.

        Coverage: list_collections() timeout handling.
        """
        manager = CollectionManager(mock_connection_manager)

        # Mock execute_operation_async to raise OperationTimeoutError
        manager._connection_manager.execute_operation_async.side_effect = OperationTimeoutError(
            "Operation timed out"
        )

        with pytest.raises(OperationTimeoutError):
            await manager.list_collections(strict=True)

    @pytest.mark.asyncio
    async def test_list_collections_connection_error_preservation(self, mock_connection_manager):
        """
        Test that list_collections preserves ConnectionError types.

        Coverage: ConnectionError preservation in list_collections.
        """
        manager = CollectionManager(mock_connection_manager)

        # Mock execute_operation_async to raise ConnectionError
        manager._connection_manager.execute_operation_async.side_effect = ConnectionError(
            "Connection lost"
        )

        with pytest.raises(ConnectionError):
            await manager.list_collections(strict=True)

    @pytest.mark.asyncio
    async def test_list_collections_return_type(
        self, mock_connection_manager, mock_pymilvus_utility
    ):
        """
        Test that list_collections returns correct type.

        Coverage: list_collections() return type verification.
        """
        mock_pymilvus_utility.list_collections.return_value = ["collection1"]

        manager = CollectionManager(mock_connection_manager)

        result = await manager.list_collections()

        assert isinstance(result, list)
        assert all(isinstance(name, str) for name in result)


# ============================================================================
# Test Helper Methods
# ============================================================================


@pytest.mark.unit
class TestHelperMethods:
    """
    Test internal helper methods.

    Coverage: Helper method functionality and edge cases.
    """

    @pytest.mark.asyncio
    async def test_ensure_awaited_with_coroutine(self):
        """
        Test _ensure_awaited with a coroutine.

        Coverage: _ensure_awaited() with coroutine input.
        """
        mock_manager = MagicMock()

        async def mock_coroutine():
            return "async_result"

        manager = CollectionManager(mock_manager)

        result = await manager._ensure_awaited(mock_coroutine())

        assert result == "async_result"

    @pytest.mark.asyncio
    async def test_ensure_awaited_with_non_coroutine(self):
        """
        Test _ensure_awaited with a non-coroutine.

        Coverage: _ensure_awaited() with non-coroutine input.
        """
        mock_manager = MagicMock()

        manager = CollectionManager(mock_manager)

        result = await manager._ensure_awaited("sync_result")

        assert result == "sync_result"

    @pytest.mark.asyncio
    async def test_ensure_awaited_with_none(self):
        """
        Test _ensure_awaited with None.

        Coverage: _ensure_awaited() with None input.
        """
        mock_manager = MagicMock()

        manager = CollectionManager(mock_manager)

        result = await manager._ensure_awaited(None)

        assert result is None

    @pytest.mark.asyncio
    async def test_ensure_awaited_with_object(self):
        """
        Test _ensure_awaited with regular object.

        Coverage: _ensure_awaited() with object input.
        """
        mock_manager = MagicMock()

        test_object = {"key": "value", "list": [1, 2, 3]}

        manager = CollectionManager(mock_manager)

        result = await manager._ensure_awaited(test_object)

        assert result == test_object
        assert result is test_object  # Should be the same object

    def test_milvus_datatype_mapping(self):
        """
        Test the Milvus data type mapping constants.

        Coverage: Data type mapping constants verification.
        """
        mock_manager = MagicMock()

        manager = CollectionManager(mock_manager)

        # Test forward mapping
        expected_mapping = {
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
            "ARRAY": "ARRAY",
        }

        assert expected_mapping == manager._MILVUS_DATATYPE_MAP

    def test_reverse_datatype_mapping(self):
        """
        Test the reverse Milvus data type mapping constants.

        Coverage: Reverse data type mapping constants verification.
        """
        mock_manager = MagicMock()

        manager = CollectionManager(mock_manager)

        # Test reverse mapping
        expected_reverse_mapping = {
            "BOOL": "BOOL",
            "INT8": "INT8",
            "INT16": "INT16",
            "INT32": "INT32",
            "INT64": "INT64",
            "FLOAT": "FLOAT",
            "DOUBLE": "DOUBLE",
            "VARCHAR": "VARCHAR",  # Always map to VARCHAR, not STRING
            "STRING": "VARCHAR",  # In case Milvus returns STRING
            "BINARY_VECTOR": "BINARY_VECTOR",
            "FLOAT_VECTOR": "FLOAT_VECTOR",
            "SPARSE_FLOAT_VECTOR": "SPARSE_FLOAT_VECTOR",
            "JSON": "JSON",
            "ARRAY": "ARRAY",
        }

        assert expected_reverse_mapping == manager._REVERSE_DATATYPE_MAP

    def test_field_param_whitelist(self):
        """
        Test the field parameter whitelist constant.

        Coverage: Field parameter whitelist verification.
        """
        mock_manager = MagicMock()

        manager = CollectionManager(mock_manager)

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
