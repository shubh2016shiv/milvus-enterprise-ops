"""
Comprehensive unit tests for DataManager.

This module provides systematic testing of the DataManager class,
including initialization, insert, upsert, delete operations, and timing methods.
"""

import asyncio
from datetime import datetime, timezone

import pytest

from milvus_ops.collection_operations.entities import (
    CollectionDescription,
    CollectionState,
    LoadState,
)
from milvus_ops.data_management_operations import DataManager, DataOperationConfig
from milvus_ops.data_management_operations.data_ops_exceptions import (
    CollectionOperationError,
    DeleteOperationError,
    InsertionError,
    SchemaValidationError,
)
from milvus_ops.data_management_operations.models.entities import OperationStatus
from milvus_ops.milvus_ops_exceptions import ConnectionError

# ============================================================================
# Test DataManager Initialization
# ============================================================================


@pytest.mark.unit
class TestDataManagerInitialization:
    """
    Test DataManager initialization.

    Coverage: DataManager constructor and initialization logic.
    """

    def test_init_basic(self, mock_connection_manager, mock_collection_manager):
        """
        Test basic DataManager initialization.

        Coverage: DataManager.__init__() with valid dependencies.
        """
        manager = DataManager(mock_connection_manager, mock_collection_manager)
        assert manager._connection_manager == mock_connection_manager
        assert manager._collection_manager == mock_collection_manager
        assert isinstance(manager._config, DataOperationConfig)
        assert manager._default_batch_size == 1000  # Default
        assert isinstance(manager._locks, dict)
        assert len(manager._locks) == 0

    def test_init_with_custom_config(self, mock_connection_manager, mock_collection_manager):
        """
        Test DataManager initialization with custom config.

        Coverage: DataManager.__init__() with custom DataOperationConfig.
        """
        config = DataOperationConfig(default_batch_size=500, enable_timing=False)
        manager = DataManager(mock_connection_manager, mock_collection_manager, config=config)
        assert manager._config == config
        assert manager._default_batch_size == 500
        assert manager._timer._enable_logging is False

    def test_init_with_none_config(self, mock_connection_manager, mock_collection_manager):
        """
        Test DataManager initialization with None config.

        Coverage: DataManager.__init__() uses default config when None provided.
        """
        manager = DataManager(mock_connection_manager, mock_collection_manager, config=None)
        assert isinstance(manager._config, DataOperationConfig)
        assert manager._default_batch_size == 1000  # Default

    def test_init_locks_initialization(self, mock_connection_manager, mock_collection_manager):
        """
        Test DataManager locks initialization.

        Coverage: DataManager initializes locks dictionary and global lock.
        """
        manager = DataManager(mock_connection_manager, mock_collection_manager)
        assert isinstance(manager._locks, dict)
        assert len(manager._locks) == 0
        assert hasattr(manager, "_global_lock")

    def test_init_timer_initialization(self, mock_connection_manager, mock_collection_manager):
        """
        Test DataManager timer initialization.

        Coverage: DataManager initializes PerformanceTimer.
        """
        manager = DataManager(mock_connection_manager, mock_collection_manager)
        assert hasattr(manager, "_timer")
        assert manager._timer._enable_logging is True  # Default

    def test_init_multiple_managers_independent(
        self, mock_connection_manager, mock_collection_manager
    ):
        """
        Test multiple DataManager instances are independent.

        Coverage: Multiple DataManager instances have independent state.
        """
        manager1 = DataManager(mock_connection_manager, mock_collection_manager)
        manager2 = DataManager(mock_connection_manager, mock_collection_manager)
        assert manager1._locks is not manager2._locks
        assert manager1._timer is not manager2._timer


# ============================================================================
# Test DataManager insert
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestDataManagerInsert:
    """
    Test DataManager.insert method.

    Coverage: Insert operations with various scenarios.
    """

    async def test_insert_success_single_document(
        self,
        mock_connection_manager,
        mock_collection_manager,
        basic_collection_schema,
    ):
        """
        Test insert with successful single document.

        Coverage: insert() successfully inserts a single document.
        """
        # Setup mocks
        mock_collection_manager.has_collection.return_value = True
        mock_collection_manager.describe_collection.return_value = CollectionDescription(
            name="test_collection",
            collection_schema=basic_collection_schema,
            id="test_collection_id",
            created_at=datetime.now(timezone.utc),
            schema_hash="abc123",
            state=CollectionState.AVAILABLE,
            load_state=LoadState.LOADED,
            created_at_is_synthetic=False,
        )
        mock_connection_manager.check_server_status.return_value = True
        mock_connection_manager.execute_operation_async.return_value = [1]  # Inserted ID

        manager = DataManager(mock_connection_manager, mock_collection_manager)
        # Use documents matching basic_collection_schema (entity_id and vector only)
        documents = [{"entity_id": 1, "vector": [0.1] * 128}]
        result = await manager.insert("test_collection", documents)

        assert result.status == OperationStatus.SUCCESS
        assert result.successful_count == 1
        assert result.failed_count == 0
        assert len(result.inserted_ids) == 1

    async def test_insert_success_batch(
        self,
        mock_connection_manager,
        mock_collection_manager,
        basic_collection_schema,
    ):
        """
        Test insert with successful batch.

        Coverage: insert() successfully inserts multiple documents in a batch.
        """
        # Setup mocks
        mock_collection_manager.has_collection.return_value = True
        mock_collection_manager.describe_collection.return_value = CollectionDescription(
            name="test_collection",
            collection_schema=basic_collection_schema,
            id="test_collection_id",
            created_at=datetime.now(timezone.utc),
            schema_hash="abc123",
            state=CollectionState.AVAILABLE,
            load_state=LoadState.LOADED,
            created_at_is_synthetic=False,
        )
        mock_connection_manager.check_server_status.return_value = True
        mock_connection_manager.execute_operation_async.return_value = [1, 2, 3]  # Inserted IDs

        manager = DataManager(mock_connection_manager, mock_collection_manager)
        # Use documents matching basic_collection_schema (entity_id and vector only)
        documents = [
            {"entity_id": 1, "vector": [0.1] * 128},
            {"entity_id": 2, "vector": [0.2] * 128},
            {"entity_id": 3, "vector": [0.3] * 128},
        ]
        result = await manager.insert("test_collection", documents)

        assert result.status == OperationStatus.SUCCESS
        assert result.successful_count == 3
        assert result.failed_count == 0
        assert len(result.inserted_ids) == 3

    async def test_insert_empty_documents(self, mock_connection_manager, mock_collection_manager):
        """
        Test insert with empty document list.

        Coverage: insert() handles empty document list gracefully.
        """
        manager = DataManager(mock_connection_manager, mock_collection_manager)
        result = await manager.insert("test_collection", [])

        assert result.status == OperationStatus.SUCCESS
        assert result.successful_count == 0
        assert result.failed_count == 0

    async def test_insert_with_validation_enabled(
        self,
        mock_connection_manager,
        mock_collection_manager,
        basic_collection_schema,
    ):
        """
        Test insert with validation enabled.

        Coverage: insert() validates documents when validate=True.
        """
        # Setup mocks
        mock_collection_manager.has_collection.return_value = True
        mock_collection_manager.describe_collection.return_value = CollectionDescription(
            name="test_collection",
            collection_schema=basic_collection_schema,
            id="test_collection_id",
            created_at=datetime.now(timezone.utc),
            schema_hash="abc123",
            state=CollectionState.AVAILABLE,
            load_state=LoadState.LOADED,
            created_at_is_synthetic=False,
        )
        mock_connection_manager.check_server_status.return_value = True
        mock_connection_manager.execute_operation_async.return_value = [1, 2, 3]

        manager = DataManager(mock_connection_manager, mock_collection_manager)
        # Use documents matching basic_collection_schema (entity_id and vector only)
        documents = [
            {"entity_id": 1, "vector": [0.1] * 128},
            {"entity_id": 2, "vector": [0.2] * 128},
            {"entity_id": 3, "vector": [0.3] * 128},
        ]
        result = await manager.insert("test_collection", documents, validate=True)

        assert result.status == OperationStatus.SUCCESS
        assert result.successful_count == 3

    async def test_insert_with_validation_disabled(
        self,
        mock_connection_manager,
        mock_collection_manager,
        basic_collection_schema,
        sample_documents,
    ):
        """
        Test insert with validation disabled.

        Coverage: insert() skips validation when validate=False.
        """
        # Setup mocks
        mock_collection_manager.has_collection.return_value = True
        mock_collection_manager.describe_collection.return_value = CollectionDescription(
            name="test_collection",
            collection_schema=basic_collection_schema,
            id="test_collection_id",
            created_at=datetime.now(timezone.utc),
            schema_hash="abc123",
            state=CollectionState.AVAILABLE,
            load_state=LoadState.LOADED,
            created_at_is_synthetic=False,
        )
        mock_connection_manager.check_server_status.return_value = True
        mock_connection_manager.execute_operation_async.return_value = [1, 2, 3]

        manager = DataManager(mock_connection_manager, mock_collection_manager)
        result = await manager.insert("test_collection", sample_documents, validate=False)

        assert result.status == OperationStatus.SUCCESS
        assert result.successful_count == 3

    async def test_insert_with_validation_error(
        self,
        mock_connection_manager,
        mock_collection_manager,
        basic_collection_schema,
    ):
        """
        Test insert with validation error.

        Coverage: insert() raises SchemaValidationError when validation fails.
        """
        # Setup mocks
        mock_collection_manager.has_collection.return_value = True
        mock_collection_manager.describe_collection.return_value = CollectionDescription(
            name="test_collection",
            collection_schema=basic_collection_schema,
            id="test_collection_id",
            created_at=datetime.now(timezone.utc),
            schema_hash="abc123",
            state=CollectionState.AVAILABLE,
            load_state=LoadState.LOADED,
            created_at_is_synthetic=False,
        )
        mock_connection_manager.check_server_status.return_value = True

        manager = DataManager(mock_connection_manager, mock_collection_manager)
        invalid_docs = [{"entity_id": 1, "vector": [0.1] * 64}]  # Wrong dimension

        with pytest.raises(SchemaValidationError):
            await manager.insert("test_collection", invalid_docs, validate=True)

    async def test_insert_collection_not_found(
        self,
        mock_connection_manager,
        mock_collection_manager,
    ):
        """
        Test insert with collection not found.

        Coverage: insert() raises CollectionOperationError when collection doesn't exist.
        """
        mock_collection_manager.has_collection.return_value = False
        mock_connection_manager.check_server_status.return_value = True

        manager = DataManager(mock_connection_manager, mock_collection_manager)

        with pytest.raises(CollectionOperationError, match="does not exist"):
            await manager.insert(
                "nonexistent_collection", [{"entity_id": 1, "vector": [0.1] * 128}]
            )

    async def test_insert_connection_health_failure(
        self, mock_connection_manager, mock_collection_manager
    ):
        """
        Test insert with connection health check failure.

        Coverage: insert() raises ConnectionError when connection is unhealthy.
        """
        mock_connection_manager.check_server_status.return_value = False

        manager = DataManager(mock_connection_manager, mock_collection_manager)

        with pytest.raises(ConnectionError, match="Milvus server is not responding"):
            await manager.insert("test_collection", [{"entity_id": 1, "vector": [0.1] * 128}])

    async def test_insert_with_partition_key(
        self,
        mock_connection_manager,
        mock_collection_manager,
        basic_collection_schema,
    ):
        """
        Test insert with partition key.

        Coverage: insert() handles partition_key parameter.
        """
        # Setup mocks
        mock_collection_manager.has_collection.return_value = True
        mock_collection_manager.describe_collection.return_value = CollectionDescription(
            name="test_collection",
            collection_schema=basic_collection_schema,
            id="test_collection_id",
            created_at=datetime.now(timezone.utc),
            schema_hash="abc123",
            state=CollectionState.AVAILABLE,
            load_state=LoadState.LOADED,
            created_at_is_synthetic=False,
        )
        mock_connection_manager.check_server_status.return_value = True
        mock_connection_manager.execute_operation_async.return_value = [1, 2, 3]

        manager = DataManager(mock_connection_manager, mock_collection_manager)
        # Use documents matching basic_collection_schema (entity_id and vector only)
        documents = [
            {"entity_id": 1, "vector": [0.1] * 128},
            {"entity_id": 2, "vector": [0.2] * 128},
            {"entity_id": 3, "vector": [0.3] * 128},
        ]
        result = await manager.insert("test_collection", documents, partition_key="partition1")

        assert result.status == OperationStatus.SUCCESS
        assert result.successful_count == 3

    async def test_insert_batch_partial_failure(
        self,
        mock_connection_manager,
        mock_collection_manager,
        basic_collection_schema,
    ):
        """
        Test insert with partial batch failure.

        Coverage: insert() handles partial batch failures correctly.
        """
        # Setup mocks
        mock_collection_manager.has_collection.return_value = True
        mock_collection_manager.describe_collection.return_value = CollectionDescription(
            name="test_collection",
            collection_schema=basic_collection_schema,
            id="test_collection_id",
            created_at=datetime.now(timezone.utc),
            schema_hash="abc123",
            state=CollectionState.AVAILABLE,
            load_state=LoadState.LOADED,
            created_at_is_synthetic=False,
        )
        mock_connection_manager.check_server_status.return_value = True

        # First batch succeeds, second batch fails
        call_count = 0

        async def mock_execute(operation, timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [1, 2]  # First batch succeeds
            else:
                raise InsertionError("Batch 2 failed")

        mock_connection_manager.execute_operation_async.side_effect = mock_execute

        manager = DataManager(
            mock_connection_manager,
            mock_collection_manager,
            config=DataOperationConfig(default_batch_size=2),
        )
        # Use documents matching basic_collection_schema (entity_id and vector only)
        # 6 documents, 3 batches of 2
        docs = [{"entity_id": i, "vector": [0.1] * 128} for i in range(1, 7)]

        result = await manager.insert("test_collection", docs)

        assert result.status == OperationStatus.PARTIAL
        assert result.successful_count > 0
        assert result.failed_count > 0

    @pytest.mark.parametrize(
        "batch_size",
        [100, 500, 1000, 2000],
    )
    async def test_insert_various_batch_sizes(
        self,
        batch_size,
        mock_connection_manager,
        mock_collection_manager,
        basic_collection_schema,
    ):
        """
        Test insert with various batch sizes.

        Coverage: insert() handles different batch sizes correctly.
        """
        # Setup mocks
        mock_collection_manager.has_collection.return_value = True
        mock_collection_manager.describe_collection.return_value = CollectionDescription(
            name="test_collection",
            collection_schema=basic_collection_schema,
            id="test_collection_id",
            created_at=datetime.now(timezone.utc),
            schema_hash="abc123",
            state=CollectionState.AVAILABLE,
            load_state=LoadState.LOADED,
            created_at_is_synthetic=False,
        )
        mock_connection_manager.check_server_status.return_value = True
        mock_connection_manager.execute_operation_async.return_value = list(
            range(1, batch_size + 1)
        )

        # Create documents
        documents = [{"entity_id": i, "vector": [0.1] * 128} for i in range(batch_size)]

        manager = DataManager(mock_connection_manager, mock_collection_manager)
        result = await manager.insert("test_collection", documents, batch_size=batch_size)

        assert result.status == OperationStatus.SUCCESS
        assert result.successful_count == batch_size


# ============================================================================
# Test DataManager upsert
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestDataManagerUpsert:
    """
    Test DataManager.upsert method.

    Coverage: Upsert operations with various scenarios.
    """

    async def test_upsert_success_single_document(
        self,
        mock_connection_manager,
        mock_collection_manager,
        basic_collection_schema,
    ):
        """
        Test upsert with successful single document.

        Coverage: upsert() successfully upserts a single document.
        """
        # Setup mocks
        mock_collection_manager.has_collection.return_value = True
        mock_collection_manager.describe_collection.return_value = CollectionDescription(
            name="test_collection",
            collection_schema=basic_collection_schema,
            id="test_collection_id",
            created_at=datetime.now(timezone.utc),
            schema_hash="abc123",
            state=CollectionState.AVAILABLE,
            load_state=LoadState.LOADED,
            created_at_is_synthetic=False,
        )
        mock_connection_manager.check_server_status.return_value = True
        mock_connection_manager.execute_operation_async.return_value = [1]

        manager = DataManager(mock_connection_manager, mock_collection_manager)
        # Use documents matching basic_collection_schema (entity_id and vector only)
        documents = [{"entity_id": 1, "vector": [0.1] * 128}]
        result = await manager.upsert("test_collection", documents)

        assert result.status == OperationStatus.SUCCESS
        assert result.successful_count == 1
        assert result.failed_count == 0

    async def test_upsert_success_batch(
        self,
        mock_connection_manager,
        mock_collection_manager,
        basic_collection_schema,
    ):
        """
        Test upsert with successful batch.

        Coverage: upsert() successfully upserts multiple documents.
        """
        # Setup mocks
        mock_collection_manager.has_collection.return_value = True
        mock_collection_manager.describe_collection.return_value = CollectionDescription(
            name="test_collection",
            collection_schema=basic_collection_schema,
            id="test_collection_id",
            created_at=datetime.now(timezone.utc),
            schema_hash="abc123",
            state=CollectionState.AVAILABLE,
            load_state=LoadState.LOADED,
            created_at_is_synthetic=False,
        )
        mock_connection_manager.check_server_status.return_value = True
        mock_connection_manager.execute_operation_async.return_value = [1, 2, 3]

        manager = DataManager(mock_connection_manager, mock_collection_manager)
        # Use documents matching basic_collection_schema (entity_id and vector only)
        documents = [
            {"entity_id": 1, "vector": [0.1] * 128},
            {"entity_id": 2, "vector": [0.2] * 128},
            {"entity_id": 3, "vector": [0.3] * 128},
        ]
        result = await manager.upsert("test_collection", documents)

        assert result.status == OperationStatus.SUCCESS
        assert result.successful_count == 3

    async def test_upsert_empty_documents(self, mock_connection_manager, mock_collection_manager):
        """
        Test upsert with empty document list.

        Coverage: upsert() handles empty document list gracefully.
        """
        manager = DataManager(mock_connection_manager, mock_collection_manager)
        result = await manager.upsert("test_collection", [])

        assert result.status == OperationStatus.SUCCESS
        assert result.successful_count == 0

    async def test_upsert_with_validation_error(
        self,
        mock_connection_manager,
        mock_collection_manager,
        basic_collection_schema,
    ):
        """
        Test upsert with validation error.

        Coverage: upsert() raises SchemaValidationError when validation fails.
        """
        # Setup mocks
        mock_collection_manager.has_collection.return_value = True
        mock_collection_manager.describe_collection.return_value = CollectionDescription(
            name="test_collection",
            collection_schema=basic_collection_schema,
            id="test_collection_id",
            created_at=datetime.now(timezone.utc),
            schema_hash="abc123",
            state=CollectionState.AVAILABLE,
            load_state=LoadState.LOADED,
            created_at_is_synthetic=False,
        )
        mock_connection_manager.check_server_status.return_value = True

        manager = DataManager(mock_connection_manager, mock_collection_manager)
        invalid_docs = [{"entity_id": 1, "vector": [0.1] * 64}]  # Wrong dimension

        with pytest.raises(SchemaValidationError):
            await manager.upsert("test_collection", invalid_docs, validate=True)

    async def test_upsert_collection_not_found(
        self, mock_connection_manager, mock_collection_manager
    ):
        """
        Test upsert with collection not found.

        Coverage: upsert() raises CollectionOperationError when collection doesn't exist.
        """
        mock_collection_manager.has_collection.return_value = False
        mock_connection_manager.check_server_status.return_value = True

        manager = DataManager(mock_connection_manager, mock_collection_manager)

        with pytest.raises(CollectionOperationError, match="does not exist"):
            await manager.upsert(
                "nonexistent_collection", [{"entity_id": 1, "vector": [0.1] * 128}]
            )


# ============================================================================
# Test DataManager delete
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestDataManagerDelete:
    """
    Test DataManager.delete method.

    Coverage: Delete operations with various scenarios.
    """

    async def test_delete_success(self, mock_connection_manager, mock_collection_manager):
        """
        Test delete with successful deletion.

        Coverage: delete() successfully deletes documents matching expression.
        """
        # Setup mocks
        mock_collection_manager.has_collection.return_value = True
        mock_connection_manager.check_server_status.return_value = True
        mock_connection_manager.execute_operation_async.return_value = 5  # Deleted count

        manager = DataManager(mock_connection_manager, mock_collection_manager)
        result = await manager.delete("test_collection", "id in [1, 2, 3, 4, 5]")

        assert result.status == OperationStatus.SUCCESS
        assert result.deleted_count == 5

    async def test_delete_collection_not_found(
        self, mock_connection_manager, mock_collection_manager
    ):
        """
        Test delete with collection not found.

        Coverage: delete() raises CollectionOperationError when collection doesn't exist.
        """
        mock_collection_manager.has_collection.return_value = False
        mock_connection_manager.check_server_status.return_value = True

        manager = DataManager(mock_connection_manager, mock_collection_manager)

        with pytest.raises(CollectionOperationError, match="does not exist"):
            await manager.delete("nonexistent_collection", "id in [1, 2, 3]")

    async def test_delete_with_partition_key(
        self,
        mock_connection_manager,
        mock_collection_manager,
    ):
        """
        Test delete with partition key.

        Coverage: delete() handles partition_key parameter.
        """
        # Setup mocks
        mock_collection_manager.has_collection.return_value = True
        mock_connection_manager.check_server_status.return_value = True
        mock_connection_manager.execute_operation_async.return_value = 3

        manager = DataManager(mock_connection_manager, mock_collection_manager)
        result = await manager.delete(
            "test_collection", "id in [1, 2, 3]", partition_key="partition1"
        )

        assert result.status == OperationStatus.SUCCESS
        assert result.deleted_count == 3

    async def test_delete_connection_health_failure(
        self, mock_connection_manager, mock_collection_manager
    ):
        """
        Test delete with connection health check failure.

        Coverage: delete() raises ConnectionError when connection is unhealthy.
        """
        mock_connection_manager.check_server_status.return_value = False

        manager = DataManager(mock_connection_manager, mock_collection_manager)

        with pytest.raises(ConnectionError, match="Milvus server is not responding"):
            await manager.delete("test_collection", "id in [1, 2, 3]")

    async def test_delete_milvus_exception(
        self,
        mock_connection_manager,
        mock_collection_manager,
    ):
        """
        Test delete with Milvus exception.

        Coverage: delete() raises DeleteOperationError when Milvus exception occurs.
        """
        # Setup mocks
        mock_collection_manager.has_collection.return_value = True
        mock_connection_manager.check_server_status.return_value = True
        mock_connection_manager.execute_operation_async.side_effect = DeleteOperationError(
            "Delete failed"
        )

        manager = DataManager(mock_connection_manager, mock_collection_manager)

        with pytest.raises(DeleteOperationError, match="Delete failed"):
            await manager.delete("test_collection", "id in [1, 2, 3]")


# ============================================================================
# Test DataManager Timing Methods
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestDataManagerTimingMethods:
    """
    Test DataManager timing methods.

    Coverage: get_timing_history, get_operation_stats, get_performance_summary,
    clear_timing_history.
    """

    async def test_get_timing_history_empty(self, mock_connection_manager, mock_collection_manager):
        """
        Test get_timing_history with empty history.

        Coverage: get_timing_history() returns empty list when no operations.
        """
        manager = DataManager(mock_connection_manager, mock_collection_manager)
        history = manager.get_timing_history()
        assert isinstance(history, list)
        assert len(history) == 0

    async def test_get_timing_history_after_operations(
        self,
        mock_connection_manager,
        mock_collection_manager,
        basic_collection_schema,
    ):
        """
        Test get_timing_history after operations.

        Coverage: get_timing_history() returns timing results after operations.
        """
        # Setup mocks
        mock_collection_manager.has_collection.return_value = True
        mock_collection_manager.describe_collection.return_value = CollectionDescription(
            name="test_collection",
            collection_schema=basic_collection_schema,
            id="test_collection_id",
            created_at=datetime.now(timezone.utc),
            schema_hash="abc123",
            state=CollectionState.AVAILABLE,
            load_state=LoadState.LOADED,
            created_at_is_synthetic=False,
        )
        mock_connection_manager.check_server_status.return_value = True
        mock_connection_manager.execute_operation_async.return_value = [1, 2, 3]

        manager = DataManager(mock_connection_manager, mock_collection_manager)
        # Use documents matching basic_collection_schema (entity_id and vector only)
        documents = [
            {"entity_id": 1, "vector": [0.1] * 128},
            {"entity_id": 2, "vector": [0.2] * 128},
            {"entity_id": 3, "vector": [0.3] * 128},
        ]
        await manager.insert("test_collection", documents)

        history = manager.get_timing_history()
        assert len(history) > 0
        assert history[0].operation_name == "insert_documents"

    async def test_get_operation_stats_no_operations(
        self,
        mock_connection_manager,
        mock_collection_manager,
    ):
        """
        Test get_operation_stats with no operations.

        Coverage: get_operation_stats() returns None when no operations found.
        """
        manager = DataManager(mock_connection_manager, mock_collection_manager)
        stats = manager.get_operation_stats("insert_documents")
        assert stats is None

    async def test_get_operation_stats_with_operations(
        self,
        mock_connection_manager,
        mock_collection_manager,
        basic_collection_schema,
    ):
        """
        Test get_operation_stats with operations.

        Coverage: get_operation_stats() returns statistics for operation type.
        """
        # Setup mocks
        mock_collection_manager.has_collection.return_value = True
        mock_collection_manager.describe_collection.return_value = CollectionDescription(
            name="test_collection",
            collection_schema=basic_collection_schema,
            id="test_collection_id",
            created_at=datetime.now(timezone.utc),
            schema_hash="abc123",
            state=CollectionState.AVAILABLE,
            load_state=LoadState.LOADED,
            created_at_is_synthetic=False,
        )
        mock_connection_manager.check_server_status.return_value = True
        mock_connection_manager.execute_operation_async.return_value = [1, 2, 3]

        manager = DataManager(mock_connection_manager, mock_collection_manager)
        # Use documents matching basic_collection_schema (entity_id and vector only)
        documents = [
            {"entity_id": 1, "vector": [0.1] * 128},
            {"entity_id": 2, "vector": [0.2] * 128},
            {"entity_id": 3, "vector": [0.3] * 128},
        ]
        await manager.insert("test_collection", documents)

        stats = manager.get_operation_stats("insert_documents")
        assert stats is not None
        assert stats.operation_name == "insert_documents"
        assert stats.total_operations == 1

    async def test_get_performance_summary(
        self,
        mock_connection_manager,
        mock_collection_manager,
        basic_collection_schema,
    ):
        """
        Test get_performance_summary.

        Coverage: get_performance_summary() returns summary of all operations.
        """
        # Setup mocks
        mock_collection_manager.has_collection.return_value = True
        mock_collection_manager.describe_collection.return_value = CollectionDescription(
            name="test_collection",
            collection_schema=basic_collection_schema,
            id="test_collection_id",
            created_at=datetime.now(timezone.utc),
            schema_hash="abc123",
            state=CollectionState.AVAILABLE,
            load_state=LoadState.LOADED,
            created_at_is_synthetic=False,
        )
        mock_connection_manager.check_server_status.return_value = True
        mock_connection_manager.execute_operation_async.return_value = [1, 2, 3]

        manager = DataManager(mock_connection_manager, mock_collection_manager)
        # Use documents matching basic_collection_schema (entity_id and vector only)
        documents = [
            {"entity_id": 1, "vector": [0.1] * 128},
            {"entity_id": 2, "vector": [0.2] * 128},
            {"entity_id": 3, "vector": [0.3] * 128},
        ]
        await manager.insert("test_collection", documents)

        summary = manager.get_performance_summary()
        assert isinstance(summary, dict)
        assert "insert_documents" in summary

    async def test_clear_timing_history(
        self,
        mock_connection_manager,
        mock_collection_manager,
        basic_collection_schema,
    ):
        """
        Test clear_timing_history.

        Coverage: clear_timing_history() clears all timing history.
        """
        # Setup mocks
        mock_collection_manager.has_collection.return_value = True
        mock_collection_manager.describe_collection.return_value = CollectionDescription(
            name="test_collection",
            collection_schema=basic_collection_schema,
            id="test_collection_id",
            created_at=datetime.now(timezone.utc),
            schema_hash="abc123",
            state=CollectionState.AVAILABLE,
            load_state=LoadState.LOADED,
            created_at_is_synthetic=False,
        )
        mock_connection_manager.check_server_status.return_value = True
        mock_connection_manager.execute_operation_async.return_value = [1, 2, 3]

        manager = DataManager(mock_connection_manager, mock_collection_manager)
        # Use documents matching basic_collection_schema (entity_id and vector only)
        documents = [
            {"entity_id": 1, "vector": [0.1] * 128},
            {"entity_id": 2, "vector": [0.2] * 128},
            {"entity_id": 3, "vector": [0.3] * 128},
        ]
        await manager.insert("test_collection", documents)

        assert len(manager.get_timing_history()) > 0

        manager.clear_timing_history()

        assert len(manager.get_timing_history()) == 0


# ============================================================================
# Test DataManager Helper Methods
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestDataManagerHelperMethods:
    """
    Test DataManager helper methods.

    Coverage: _verify_connection_health, _verify_collection_exists_or_create,
    _acquire_collection_lock.
    """

    async def test_verify_connection_health_success(
        self, mock_connection_manager, mock_collection_manager
    ):
        """
        Test _verify_connection_health with healthy connection.

        Coverage: _verify_connection_health() passes when connection is healthy.
        """
        mock_connection_manager.check_server_status.return_value = True

        manager = DataManager(mock_connection_manager, mock_collection_manager)
        await manager._verify_connection_health()

        mock_connection_manager.check_server_status.assert_called_once()

    async def test_verify_connection_health_failure(
        self, mock_connection_manager, mock_collection_manager
    ):
        """
        Test _verify_connection_health with unhealthy connection.

        Coverage: _verify_connection_health() raises ConnectionError when connection is unhealthy.
        """
        mock_connection_manager.check_server_status.return_value = False

        manager = DataManager(mock_connection_manager, mock_collection_manager)

        with pytest.raises(ConnectionError, match="Milvus server is not responding"):
            await manager._verify_connection_health()

    async def test_verify_collection_exists_or_create_collection_exists(
        self,
        mock_connection_manager,
        mock_collection_manager,
    ):
        """
        Test _verify_collection_exists_or_create when collection exists.

        Coverage: _verify_collection_exists_or_create() passes when collection exists.
        """
        mock_collection_manager.has_collection.return_value = True

        manager = DataManager(mock_connection_manager, mock_collection_manager)
        await manager._verify_collection_exists_or_create("test_collection")

        mock_collection_manager.has_collection.assert_called_once()

    async def test_verify_collection_exists_or_create_collection_not_exists_no_auto_create(
        self,
        mock_connection_manager,
        mock_collection_manager,
    ):
        """
        Test _verify_collection_exists_or_create when collection doesn't exist and
        auto_create=False.

        Coverage: _verify_collection_exists_or_create() raises CollectionOperationError
        when collection doesn't exist.
        """
        mock_collection_manager.has_collection.return_value = False

        manager = DataManager(mock_connection_manager, mock_collection_manager)

        with pytest.raises(CollectionOperationError, match="does not exist"):
            await manager._verify_collection_exists_or_create("test_collection", auto_create=False)

    async def test_acquire_collection_lock(self, mock_connection_manager, mock_collection_manager):
        """
        Test _acquire_collection_lock.

        Coverage: _acquire_collection_lock() returns a lock for the collection.
        """
        manager = DataManager(mock_connection_manager, mock_collection_manager)
        lock = await manager._acquire_collection_lock("test_collection")

        assert lock is not None
        assert isinstance(lock, asyncio.Lock)
        assert "test_collection" in manager._locks

    async def test_acquire_collection_lock_multiple_collections(
        self,
        mock_connection_manager,
        mock_collection_manager,
    ):
        """
        Test _acquire_collection_lock for multiple collections.

        Coverage: _acquire_collection_lock() creates separate locks for different collections.
        """
        manager = DataManager(mock_connection_manager, mock_collection_manager)
        lock1 = await manager._acquire_collection_lock("collection1")
        lock2 = await manager._acquire_collection_lock("collection2")

        assert lock1 is not lock2
        assert "collection1" in manager._locks
        assert "collection2" in manager._locks

    async def test_acquire_collection_lock_same_collection(
        self,
        mock_connection_manager,
        mock_collection_manager,
    ):
        """
        Test _acquire_collection_lock for same collection multiple times.

        Coverage: _acquire_collection_lock() returns the same lock for the same collection.
        """
        manager = DataManager(mock_connection_manager, mock_collection_manager)
        lock1 = await manager._acquire_collection_lock("test_collection")
        lock2 = await manager._acquire_collection_lock("test_collection")

        assert lock1 is lock2
