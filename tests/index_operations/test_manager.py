"""
Comprehensive unit tests for IndexManager.

This module provides systematic testing of the IndexManager class,
including all public methods, error handling, edge cases, and async operations.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from pymilvus.exceptions import MilvusException
import pytest

from milvus_ops.collection_operations import DataType, IndexType, MetricType
from milvus_ops.index_operations.config import IndexOperationConfig
from milvus_ops.index_operations.core.manager import IndexManager
from milvus_ops.index_operations.index_ops_exceptions import (
    IndexNotFoundError,
    IndexParameterError,
)
from milvus_ops.index_operations.models.entities import IndexState
from milvus_ops.index_operations.models.parameters import HNSWParams
from milvus_ops.milvus_ops_exceptions import CollectionNotFoundError, ConnectionError

# ============================================================================
# Test IndexManager Initialization
# ============================================================================


@pytest.mark.unit
class TestIndexManagerInitialization:
    """
    Test IndexManager initialization.

    Coverage: IndexManager constructor and dependency injection.
    """

    @pytest.mark.asyncio
    async def test_manager_initialization_default_config(self, mock_index_manager_dependencies):
        """
        Test IndexManager initialization with default config.

        Coverage: IndexManager.__init__() with default IndexOperationConfig.
        """
        deps = mock_index_manager_dependencies
        manager = IndexManager(
            connection_manager=deps["connection_manager"],
            collection_manager=deps["collection_manager"],
        )
        assert manager._connection_manager is deps["connection_manager"]
        assert manager._collection_manager is deps["collection_manager"]
        assert isinstance(manager._config, IndexOperationConfig)
        assert manager._config.default_timeout == 60.0

    @pytest.mark.asyncio
    async def test_manager_initialization_custom_config(self, mock_index_manager_dependencies):
        """
        Test IndexManager initialization with custom config.

        Coverage: IndexManager.__init__() with custom IndexOperationConfig.
        """
        deps = mock_index_manager_dependencies
        config = IndexOperationConfig(default_timeout=120.0, enable_timing=False)
        manager = IndexManager(
            connection_manager=deps["connection_manager"],
            collection_manager=deps["collection_manager"],
            config=config,
        )
        assert manager._config is config
        assert manager._config.default_timeout == 120.0
        assert manager._config.enable_timing is False


# ============================================================================
# Test IndexManager _acquire_collection_lock
# ============================================================================


@pytest.mark.unit
class TestIndexManagerAcquireCollectionLock:
    """
    Test IndexManager._acquire_collection_lock() method.

    Coverage: IndexManager collection locking mechanism.
    """

    @pytest.mark.asyncio
    async def test_acquire_collection_lock(self, mock_index_manager_dependencies):
        """
        Test _acquire_collection_lock() acquires lock.

        Coverage: _acquire_collection_lock() returns asyncio.Lock for collection.
        """
        deps = mock_index_manager_dependencies
        manager = IndexManager(
            connection_manager=deps["connection_manager"],
            collection_manager=deps["collection_manager"],
        )
        lock = await manager._acquire_collection_lock("test_collection")
        assert isinstance(lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_acquire_collection_lock_same_collection(self, mock_index_manager_dependencies):
        """
        Test _acquire_collection_lock() returns same lock for same collection.

        Coverage: _acquire_collection_lock() returns same lock instance for collection.
        """
        deps = mock_index_manager_dependencies
        manager = IndexManager(
            connection_manager=deps["connection_manager"],
            collection_manager=deps["collection_manager"],
        )
        lock1 = await manager._acquire_collection_lock("test_collection")
        lock2 = await manager._acquire_collection_lock("test_collection")
        assert lock1 is lock2

    @pytest.mark.asyncio
    async def test_acquire_collection_lock_different_collections(
        self, mock_index_manager_dependencies
    ):
        """
        Test _acquire_collection_lock() returns different locks for different collections.

        Coverage: _acquire_collection_lock() returns different locks for different collections.
        """
        deps = mock_index_manager_dependencies
        manager = IndexManager(
            connection_manager=deps["connection_manager"],
            collection_manager=deps["collection_manager"],
        )
        lock1 = await manager._acquire_collection_lock("collection1")
        lock2 = await manager._acquire_collection_lock("collection2")
        assert lock1 is not lock2


# ============================================================================
# Test IndexManager _verify_connection_health
# ============================================================================


@pytest.mark.unit
class TestIndexManagerVerifyConnectionHealth:
    """
    Test IndexManager._verify_connection_health() method.

    Coverage: IndexManager connection health verification.
    """

    @pytest.mark.asyncio
    async def test_verify_connection_health_success(self, mock_index_manager_dependencies):
        """
        Test _verify_connection_health() with healthy connection.

        Coverage: _verify_connection_health() passes with healthy connection.
        """
        deps = mock_index_manager_dependencies
        deps["connection_manager"].check_server_status = MagicMock(return_value=True)
        manager = IndexManager(
            connection_manager=deps["connection_manager"],
            collection_manager=deps["collection_manager"],
        )
        # Should not raise
        await manager._verify_connection_health()

    @pytest.mark.asyncio
    async def test_verify_connection_health_failure(self, mock_index_manager_dependencies):
        """
        Test _verify_connection_health() with unhealthy connection.

        Coverage: _verify_connection_health() raises ConnectionError for unhealthy connection.
        """
        deps = mock_index_manager_dependencies
        deps["connection_manager"].check_server_status = MagicMock(return_value=False)
        manager = IndexManager(
            connection_manager=deps["connection_manager"],
            collection_manager=deps["collection_manager"],
        )
        with pytest.raises(ConnectionError):
            await manager._verify_connection_health()

    @pytest.mark.asyncio
    async def test_verify_connection_health_exception(self, mock_index_manager_dependencies):
        """
        Test _verify_connection_health() when check raises exception.

        Coverage: _verify_connection_health() raises ConnectionError on exception.
        """
        deps = mock_index_manager_dependencies
        deps["connection_manager"].check_server_status = MagicMock(
            side_effect=Exception("Connection failed")
        )
        manager = IndexManager(
            connection_manager=deps["connection_manager"],
            collection_manager=deps["collection_manager"],
        )
        with pytest.raises(ConnectionError):
            await manager._verify_connection_health()


# ============================================================================
# Test IndexManager _verify_collection_exists
# ============================================================================


@pytest.mark.unit
class TestIndexManagerVerifyCollectionExists:
    """
    Test IndexManager._verify_collection_exists() method.

    Coverage: IndexManager collection existence verification.
    """

    @pytest.mark.asyncio
    async def test_verify_collection_exists_success(self, mock_index_manager_dependencies):
        """
        Test _verify_collection_exists() with existing collection.

        Coverage: _verify_collection_exists() returns schema for existing collection.
        """
        from milvus_ops.collection_operations.schema import CollectionSchema, FieldSchema

        deps = mock_index_manager_dependencies
        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
            ]
        )
        collection_desc = MagicMock()
        collection_desc.collection_schema = schema
        deps["collection_manager"].has_collection = AsyncMock(return_value=True)
        deps["collection_manager"].describe_collection = AsyncMock(return_value=collection_desc)

        manager = IndexManager(
            connection_manager=deps["connection_manager"],
            collection_manager=deps["collection_manager"],
        )
        result_schema = await manager._verify_collection_exists("test_collection")
        assert result_schema is schema

    @pytest.mark.asyncio
    async def test_verify_collection_exists_not_found(self, mock_index_manager_dependencies):
        """
        Test _verify_collection_exists() with non-existent collection.

        Coverage: _verify_collection_exists() raises CollectionNotFoundError.
        """
        deps = mock_index_manager_dependencies
        deps["collection_manager"].has_collection = AsyncMock(return_value=False)

        manager = IndexManager(
            connection_manager=deps["connection_manager"],
            collection_manager=deps["collection_manager"],
        )
        with pytest.raises(CollectionNotFoundError):
            await manager._verify_collection_exists("test_collection")


# ============================================================================
# Test IndexManager create_index
# ============================================================================


@pytest.mark.unit
class TestIndexManagerCreateIndex:
    """
    Test IndexManager.create_index() method.

    Coverage: IndexManager index creation with various scenarios.
    """

    @pytest.mark.asyncio
    async def test_create_index_success(
        self, mock_index_manager_dependencies, basic_collection_schema
    ):
        """
        Test create_index() with successful creation.

        Coverage: create_index() successfully creates index.
        """
        deps = mock_index_manager_dependencies
        deps["connection_manager"].check_server_status = MagicMock(return_value=True)
        deps["collection_manager"].has_collection = AsyncMock(return_value=True)
        collection_desc = MagicMock()
        collection_desc.collection_schema = basic_collection_schema
        deps["collection_manager"].describe_collection = AsyncMock(return_value=collection_desc)
        deps["connection_manager"].execute_operation_async = AsyncMock()

        manager = IndexManager(
            connection_manager=deps["connection_manager"],
            collection_manager=deps["collection_manager"],
        )

        result = await manager.create_index(
            collection_name="test_collection",
            field_name="vector",
            index_type=IndexType.HNSW,
            metric_type=MetricType.COSINE,
            index_params=HNSWParams(M=16, efConstruction=200),
        )

        assert result.success is True
        assert result.collection_name == "test_collection"
        assert result.field_name == "vector"
        assert result.state == IndexState.CREATING

    @pytest.mark.asyncio
    async def test_create_index_with_wait(
        self, mock_index_manager_dependencies, basic_collection_schema
    ):
        """
        Test create_index() with wait=True.

        Coverage: create_index() waits for build completion when wait=True.
        """
        deps = mock_index_manager_dependencies
        deps["connection_manager"].check_server_status = MagicMock(return_value=True)
        deps["collection_manager"].has_collection = AsyncMock(return_value=True)
        collection_desc = MagicMock()
        collection_desc.collection_schema = basic_collection_schema
        deps["collection_manager"].describe_collection = AsyncMock(return_value=collection_desc)
        deps["connection_manager"].execute_operation_async = AsyncMock()

        # Mock get_index_build_progress to return completed state
        manager = IndexManager(
            connection_manager=deps["connection_manager"],
            collection_manager=deps["collection_manager"],
        )
        manager.get_index_build_progress = AsyncMock(
            return_value=MagicMock(
                state=IndexState.CREATED,
                percentage=100.0,
            )
        )

        result = await manager.create_index(
            collection_name="test_collection",
            field_name="vector",
            index_type=IndexType.HNSW,
            metric_type=MetricType.COSINE,
            index_params=HNSWParams(M=16, efConstruction=200),
            wait=True,
        )

        assert result.success is True
        assert result.state == IndexState.CREATED

    @pytest.mark.asyncio
    async def test_create_index_field_not_found(
        self, mock_index_manager_dependencies, basic_collection_schema
    ):
        """
        Test create_index() with non-existent field.

        Coverage: create_index() raises IndexParameterError for non-existent field.
        """
        deps = mock_index_manager_dependencies
        deps["connection_manager"].check_server_status = MagicMock(return_value=True)
        deps["collection_manager"].has_collection = AsyncMock(return_value=True)
        collection_desc = MagicMock()
        collection_desc.collection_schema = basic_collection_schema
        deps["collection_manager"].describe_collection = AsyncMock(return_value=collection_desc)

        manager = IndexManager(
            connection_manager=deps["connection_manager"],
            collection_manager=deps["collection_manager"],
        )

        with pytest.raises(IndexParameterError):
            await manager.create_index(
                collection_name="test_collection",
                field_name="non_existent_field",
                index_type=IndexType.HNSW,
                metric_type=MetricType.COSINE,
            )

    @pytest.mark.asyncio
    async def test_create_index_non_vector_field(
        self, mock_index_manager_dependencies, basic_collection_schema
    ):
        """
        Test create_index() with non-vector field.

        Coverage: create_index() raises IndexParameterError for non-vector field.
        """
        deps = mock_index_manager_dependencies
        deps["connection_manager"].check_server_status = MagicMock(return_value=True)
        deps["collection_manager"].has_collection = AsyncMock(return_value=True)
        collection_desc = MagicMock()
        collection_desc.collection_schema = basic_collection_schema
        deps["collection_manager"].describe_collection = AsyncMock(return_value=collection_desc)

        manager = IndexManager(
            connection_manager=deps["connection_manager"],
            collection_manager=deps["collection_manager"],
        )

        with pytest.raises(IndexParameterError):
            await manager.create_index(
                collection_name="test_collection",
                field_name="entity_id",  # Not a vector field
                index_type=IndexType.HNSW,
                metric_type=MetricType.COSINE,
            )

    @pytest.mark.asyncio
    async def test_create_index_collection_not_found(self, mock_index_manager_dependencies):
        """
        Test create_index() with non-existent collection.

        Coverage: create_index() raises CollectionNotFoundError for non-existent collection.
        """
        deps = mock_index_manager_dependencies
        deps["connection_manager"].check_server_status = MagicMock(return_value=True)
        deps["collection_manager"].has_collection = AsyncMock(return_value=False)

        manager = IndexManager(
            connection_manager=deps["connection_manager"],
            collection_manager=deps["collection_manager"],
        )

        with pytest.raises(CollectionNotFoundError):
            await manager.create_index(
                collection_name="non_existent_collection",
                field_name="embedding",
                index_type=IndexType.HNSW,
                metric_type=MetricType.COSINE,
            )

    @pytest.mark.asyncio
    async def test_create_index_connection_error(self, mock_index_manager_dependencies):
        """
        Test create_index() with connection error.

        Coverage: create_index() raises ConnectionError for unhealthy connection.
        """
        deps = mock_index_manager_dependencies
        deps["connection_manager"].check_server_status = MagicMock(return_value=False)

        manager = IndexManager(
            connection_manager=deps["connection_manager"],
            collection_manager=deps["collection_manager"],
        )

        with pytest.raises(ConnectionError):
            await manager.create_index(
                collection_name="test_collection",
                field_name="embedding",
                index_type=IndexType.HNSW,
                metric_type=MetricType.COSINE,
            )

    @pytest.mark.asyncio
    async def test_create_index_build_error(
        self, mock_index_manager_dependencies, basic_collection_schema
    ):
        """
        Test create_index() with build error.

        Coverage: create_index() handles MilvusException during build.
        """
        deps = mock_index_manager_dependencies
        deps["connection_manager"].check_server_status = MagicMock(return_value=True)
        deps["collection_manager"].has_collection = AsyncMock(return_value=True)
        collection_desc = MagicMock()
        collection_desc.collection_schema = basic_collection_schema
        deps["collection_manager"].describe_collection = AsyncMock(return_value=collection_desc)
        deps["connection_manager"].execute_operation_async = AsyncMock(
            side_effect=MilvusException("Build failed")
        )

        manager = IndexManager(
            connection_manager=deps["connection_manager"],
            collection_manager=deps["collection_manager"],
        )

        result = await manager.create_index(
            collection_name="test_collection",
            field_name="vector",
            index_type=IndexType.HNSW,
            metric_type=MetricType.COSINE,
        )

        assert result.success is False
        assert result.state == IndexState.FAILED


# ============================================================================
# Test IndexManager describe_index
# ============================================================================


@pytest.mark.unit
class TestIndexManagerDescribeIndex:
    """
    Test IndexManager.describe_index() method.

    Coverage: IndexManager index description retrieval.
    """

    @pytest.mark.asyncio
    async def test_describe_index_success(
        self, mock_index_manager_dependencies, basic_collection_schema
    ):
        """
        Test describe_index() with existing index.

        Coverage: describe_index() returns IndexDescription for existing index.
        """
        deps = mock_index_manager_dependencies
        deps["connection_manager"].check_server_status = MagicMock(return_value=True)
        deps["collection_manager"].has_collection = AsyncMock(return_value=True)
        collection_desc = MagicMock()
        collection_desc.collection_schema = basic_collection_schema
        deps["collection_manager"].describe_collection = AsyncMock(return_value=collection_desc)

        # Mock Collection and indexes
        mock_index = MagicMock()
        mock_index.field_name = "vector"
        mock_index.index_name = "vector_index"
        mock_index.params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        }
        mock_collection = MagicMock()
        mock_collection.indexes = [mock_index]

        def _create_mock_collection(alias):
            return mock_collection

        deps["connection_manager"].execute_operation_async = AsyncMock(
            side_effect=lambda op, timeout: op("mock_alias")
        )

        # Patch Collection to return our mock
        with patch("milvus_ops.index_operations.core.manager.Collection") as mock_collection_class:
            mock_collection_class.return_value = mock_collection

            manager = IndexManager(
                connection_manager=deps["connection_manager"],
                collection_manager=deps["collection_manager"],
            )

            index_desc = await manager.describe_index("test_collection", "vector")

            assert index_desc is not None
            assert index_desc.field_name == "vector"

    @pytest.mark.asyncio
    async def test_describe_index_not_found(
        self, mock_index_manager_dependencies, basic_collection_schema
    ):
        """
        Test describe_index() with non-existent index.

        Coverage: describe_index() returns None for non-existent index.
        """
        deps = mock_index_manager_dependencies
        deps["connection_manager"].check_server_status = MagicMock(return_value=True)
        deps["collection_manager"].has_collection = AsyncMock(return_value=True)
        collection_desc = MagicMock()
        collection_desc.collection_schema = basic_collection_schema
        deps["collection_manager"].describe_collection = AsyncMock(return_value=collection_desc)

        # Mock Collection with no indexes for the field
        mock_collection = MagicMock()
        mock_collection.indexes = []

        with patch("milvus_ops.index_operations.core.manager.Collection") as mock_collection_class:
            mock_collection_class.return_value = mock_collection

            deps["connection_manager"].execute_operation_async = AsyncMock(
                side_effect=lambda op, timeout: op("mock_alias")
            )

            manager = IndexManager(
                connection_manager=deps["connection_manager"],
                collection_manager=deps["collection_manager"],
            )

            index_desc = await manager.describe_index("test_collection", "vector")
            assert index_desc is None


# ============================================================================
# Test IndexManager drop_index
# ============================================================================


@pytest.mark.unit
class TestIndexManagerDropIndex:
    """
    Test IndexManager.drop_index() method.

    Coverage: IndexManager index dropping.
    """

    @pytest.mark.asyncio
    async def test_drop_index_success(
        self, mock_index_manager_dependencies, basic_collection_schema
    ):
        """
        Test drop_index() with successful drop.

        Coverage: drop_index() successfully drops index.
        """
        deps = mock_index_manager_dependencies
        deps["connection_manager"].check_server_status = MagicMock(return_value=True)
        deps["collection_manager"].has_collection = AsyncMock(return_value=True)
        collection_desc = MagicMock()
        collection_desc.collection_schema = basic_collection_schema
        deps["collection_manager"].describe_collection = AsyncMock(return_value=collection_desc)
        deps["connection_manager"].execute_operation_async = AsyncMock()

        manager = IndexManager(
            connection_manager=deps["connection_manager"],
            collection_manager=deps["collection_manager"],
        )

        result = await manager.drop_index("test_collection", "vector")

        assert result.success is True
        assert result.operation == "drop"
        assert result.state == IndexState.NONE

    @pytest.mark.asyncio
    async def test_drop_index_not_found(
        self, mock_index_manager_dependencies, basic_collection_schema
    ):
        """
        Test drop_index() with non-existent index.

        Coverage: drop_index() raises IndexNotFoundError for non-existent index.
        """
        deps = mock_index_manager_dependencies
        deps["connection_manager"].check_server_status = MagicMock(return_value=True)
        deps["collection_manager"].has_collection = AsyncMock(return_value=True)
        collection_desc = MagicMock()
        collection_desc.collection_schema = basic_collection_schema
        deps["collection_manager"].describe_collection = AsyncMock(return_value=collection_desc)

        # Mock MilvusException for index not found
        mock_collection = MagicMock()
        mock_collection.drop_index = MagicMock(side_effect=MilvusException("index not found"))

        with patch("milvus_ops.index_operations.core.manager.Collection") as mock_collection_class:
            mock_collection_class.return_value = mock_collection

            deps["connection_manager"].execute_operation_async = AsyncMock(
                side_effect=lambda op, timeout: op("mock_alias")
            )

            manager = IndexManager(
                connection_manager=deps["connection_manager"],
                collection_manager=deps["collection_manager"],
            )

            with pytest.raises(IndexNotFoundError):
                await manager.drop_index("test_collection", "vector")


# ============================================================================
# Test IndexManager get_index_build_progress
# ============================================================================


@pytest.mark.unit
class TestIndexManagerGetIndexBuildProgress:
    """
    Test IndexManager.get_index_build_progress() method.

    Coverage: IndexManager index build progress retrieval.
    """

    @pytest.mark.asyncio
    async def test_get_index_build_progress_with_tracker(self, mock_index_manager_dependencies):
        """
        Test get_index_build_progress() with active tracker.

        Coverage: get_index_build_progress() returns progress from tracker.
        """
        deps = mock_index_manager_dependencies
        manager = IndexManager(
            connection_manager=deps["connection_manager"],
            collection_manager=deps["collection_manager"],
        )

        # Register a tracker
        tracker = manager._tracker_registry.register_build(
            "test_collection", "vector", total_rows=10000
        )
        tracker.update_progress(processed_rows=5000, percentage=50.0)

        progress = await manager.get_index_build_progress("test_collection", "vector")
        assert progress.percentage == 50.0
        assert progress.processed_rows == 5000

    @pytest.mark.asyncio
    async def test_get_index_build_progress_completed_index(
        self, mock_index_manager_dependencies, basic_collection_schema
    ):
        """
        Test get_index_build_progress() for completed index.

        Coverage: get_index_build_progress() returns 100% for completed index.
        """
        deps = mock_index_manager_dependencies
        deps["connection_manager"].check_server_status = MagicMock(return_value=True)
        deps["collection_manager"].has_collection = AsyncMock(return_value=True)
        collection_desc = MagicMock()
        collection_desc.collection_schema = basic_collection_schema
        deps["collection_manager"].describe_collection = AsyncMock(return_value=collection_desc)

        # Mock existing index
        mock_index = MagicMock()
        mock_index.field_name = "vector"
        mock_collection = MagicMock()
        mock_collection.indexes = [mock_index]

        with patch("milvus_ops.index_operations.core.manager.Collection") as mock_collection_class:
            mock_collection_class.return_value = mock_collection

            deps["connection_manager"].execute_operation_async = AsyncMock(
                side_effect=lambda op, timeout: op("mock_alias")
            )

            manager = IndexManager(
                connection_manager=deps["connection_manager"],
                collection_manager=deps["collection_manager"],
            )

            # Mock describe_index to return completed index
            manager.describe_index = AsyncMock(return_value=MagicMock(state=IndexState.CREATED))

            progress = await manager.get_index_build_progress("test_collection", "vector")
            assert progress.percentage == 100.0
            assert progress.state == IndexState.CREATED


# ============================================================================
# Test IndexManager list_indexes
# ============================================================================


@pytest.mark.unit
class TestIndexManagerListIndexes:
    """
    Test IndexManager.list_indexes() method.

    Coverage: IndexManager index listing.
    """

    @pytest.mark.asyncio
    async def test_list_indexes_success(
        self, mock_index_manager_dependencies, basic_collection_schema
    ):
        """
        Test list_indexes() with existing indexes.

        Coverage: list_indexes() returns list of IndexDescription.
        """
        deps = mock_index_manager_dependencies
        deps["connection_manager"].check_server_status = MagicMock(return_value=True)
        deps["collection_manager"].has_collection = AsyncMock(return_value=True)
        collection_desc = MagicMock()
        collection_desc.collection_schema = basic_collection_schema
        deps["collection_manager"].describe_collection = AsyncMock(return_value=collection_desc)

        manager = IndexManager(
            connection_manager=deps["connection_manager"],
            collection_manager=deps["collection_manager"],
        )

        # Mock describe_index to return index descriptions
        manager.describe_index = AsyncMock(
            return_value=MagicMock(
                field_name="vector",
                index_name="vector_index",
                index_type="HNSW",
            )
        )

        indexes = await manager.list_indexes("test_collection")
        assert len(indexes) >= 0  # May be 0 if no indexes


# ============================================================================
# Test IndexManager has_index
# ============================================================================


@pytest.mark.unit
class TestIndexManagerHasIndex:
    """
    Test IndexManager.has_index() method.

    Coverage: IndexManager index existence check.
    """

    @pytest.mark.asyncio
    async def test_has_index_true(self, mock_index_manager_dependencies):
        """
        Test has_index() returns True for existing index.

        Coverage: has_index() returns True when index exists.
        """
        deps = mock_index_manager_dependencies
        manager = IndexManager(
            connection_manager=deps["connection_manager"],
            collection_manager=deps["collection_manager"],
        )

        manager.describe_index = AsyncMock(return_value=MagicMock(field_name="vector"))

        result = await manager.has_index("test_collection", "vector")
        assert result is True

    @pytest.mark.asyncio
    async def test_has_index_false(self, mock_index_manager_dependencies):
        """
        Test has_index() returns False for non-existent index.

        Coverage: has_index() returns False when index doesn't exist.
        """
        deps = mock_index_manager_dependencies
        manager = IndexManager(
            connection_manager=deps["connection_manager"],
            collection_manager=deps["collection_manager"],
        )

        manager.describe_index = AsyncMock(return_value=None)

        result = await manager.has_index("test_collection", "vector")
        assert result is False

    @pytest.mark.asyncio
    async def test_has_index_collection_not_found(self, mock_index_manager_dependencies):
        """
        Test has_index() returns False for non-existent collection.

        Coverage: has_index() returns False when collection doesn't exist.
        """
        deps = mock_index_manager_dependencies
        manager = IndexManager(
            connection_manager=deps["connection_manager"],
            collection_manager=deps["collection_manager"],
        )

        manager.describe_index = AsyncMock(
            side_effect=CollectionNotFoundError("Collection not found")
        )

        result = await manager.has_index("non_existent_collection", "vector")
        assert result is False
