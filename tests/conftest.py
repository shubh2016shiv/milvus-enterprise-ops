"""
Pytest configuration and shared fixtures for connection management and collection operations tests.

This module provides comprehensive fixtures for mocking Milvus connections,
configurations, schemas, and test utilities to support thorough testing of the
connection management and collection_operations modules.
"""

import asyncio
from collections.abc import Callable, Generator
from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from config import MilvusSettings
from config.settings import ConnectionSettings
import pytest

from milvus_ops.collection_operations.entities import (
    CollectionDescription,
    CollectionState,
    CollectionStats,
    LoadProgress,
    LoadState,
    PartitionInfo,
    SegmentInfo,
)
from milvus_ops.collection_operations.schema import (
    CollectionSchema,
    DataType,
    FieldSchema,
)

# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def mock_connection_settings() -> MagicMock:
    """
    Create a mock connection settings object for testing.

    Coverage: Tests connection settings initialization and attribute access.
    """
    settings = MagicMock(spec=ConnectionSettings)
    # Set actual ConnectionSettings attributes
    settings.host = "localhost"
    settings.port = "19530"
    settings.user = "test_user"
    settings.password = "test_password"
    settings.secure = False
    settings.timeout = 10
    settings.connection_pool_size = 3  # Small pool for testing
    settings.retry_count = 2
    settings.retry_interval = 0.1  # Fast retries for testing
    settings.max_requests_per_second = 100
    settings.rate_limiter_burst_multiplier = 2.0
    settings.enable_retry_budget = True
    settings.retry_budget_min_success_rate = 0.8
    settings.retry_budget_window_seconds = 10
    # Add circuit breaker attributes (accessed via getattr)
    settings.circuit_breaker_failure_threshold = 3
    settings.circuit_breaker_recovery_timeout = 5.0
    settings.circuit_breaker_success_threshold = 2
    settings.circuit_breaker_max_half_open = 1
    return settings


@pytest.fixture
def mock_milvus_settings(mock_connection_settings: ConnectionSettings) -> MilvusSettings:
    """
    Create a mock MilvusSettings object for testing.

    Coverage: Tests MilvusSettings initialization with nested connection settings.
    """
    return MilvusSettings(connection=mock_connection_settings)


@pytest.fixture
def minimal_milvus_settings() -> MilvusSettings:
    """
    Create minimal MilvusSettings with defaults for testing edge cases.

    Coverage: Tests behavior with minimal/default configuration.
    """
    return MilvusSettings()


@pytest.fixture
def invalid_milvus_settings() -> dict[str, Any]:
    """
    Create invalid settings dictionary for testing validation.

    Coverage: Tests configuration validation and error handling.
    """
    return {
        "connection": {
            "host": "localhost",
            "port": "invalid_port",  # Invalid port
            "connection_pool_size": -1,  # Invalid pool size
        }
    }


# ============================================================================
# Mock Milvus Connection Fixtures
# ============================================================================


@pytest.fixture
def mock_pymilvus_connections() -> Generator[MagicMock, None, None]:
    """
    Mock the pymilvus.connections module.

    Coverage: Tests interaction with pymilvus connection API.
    Provides comprehensive mocking of connection operations.
    """
    with patch("milvus_ops.connection_management.connection_pool.connections") as mock_conn:
        # Mock connect method
        mock_conn.connect = MagicMock(return_value=None)

        # Mock disconnect method
        mock_conn.disconnect = MagicMock(return_value=None)

        # Mock has_connection to return True by default
        mock_conn.has_connection = MagicMock(return_value=True)

        # Mock get_connection method
        mock_connection_obj = MagicMock()
        mock_conn.get_connection = MagicMock(return_value=mock_connection_obj)

        yield mock_conn


@pytest.fixture
def mock_healthy_connection() -> MagicMock:
    """
    Create a mock healthy Milvus connection object.

    Coverage: Tests successful connection operations.
    """
    mock_conn = MagicMock()
    mock_conn.list_collections = MagicMock(return_value=["collection1", "collection2"])
    mock_conn.has_connection = MagicMock(return_value=True)
    return mock_conn


@pytest.fixture
def mock_unhealthy_connection() -> MagicMock:
    """
    Create a mock unhealthy Milvus connection object.

    Coverage: Tests handling of stale/broken connections.
    """
    mock_conn = MagicMock()
    mock_conn.list_collections = MagicMock(side_effect=Exception("Connection lost"))
    mock_conn.has_connection = MagicMock(return_value=False)
    return mock_conn


@pytest.fixture
def mock_connection_factory() -> Callable[[str, bool], MagicMock]:
    """
    Factory fixture to create mock connections with configurable behavior.

    Args:
        alias: Connection alias
        healthy: Whether connection should be healthy

    Returns:
        Mock connection object

    Coverage: Tests dynamic connection creation with various states.
    """

    def _create_mock_connection(alias: str = "conn_0", healthy: bool = True) -> MagicMock:
        mock_conn = MagicMock()
        mock_conn.alias = alias

        if healthy:
            mock_conn.has_connection = MagicMock(return_value=True)
            mock_conn.list_collections = MagicMock(return_value=[])
            mock_conn.search = MagicMock(return_value=[])
        else:
            mock_conn.has_connection = MagicMock(return_value=False)
            mock_conn.list_collections = MagicMock(side_effect=Exception("Connection failed"))

        return mock_conn

    return _create_mock_connection


# ============================================================================
# Mock load_settings Fixture
# ============================================================================


@pytest.fixture
def mock_load_settings(mock_milvus_settings: MilvusSettings) -> Generator[MagicMock, None, None]:
    """
    Mock the load_settings function to return test configuration.

    Coverage: Tests configuration loading in components that use load_settings.
    """
    with patch("config.load_settings") as mock_load:
        mock_load.return_value = mock_milvus_settings
        yield mock_load


# ============================================================================
# Async Test Fixtures
# ============================================================================


@pytest.fixture
def event_loop():
    """
    Create an event loop for async tests.

    Coverage: Ensures proper async test execution and cleanup.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def async_mock_operation() -> Callable[[Any], AsyncMock]:
    """
    Factory for creating async mock operations.

    Coverage: Tests async operation execution paths.
    """

    def _create_async_mock(
        result: Any = None, delay: float = 0.0, error: Exception | None = None
    ) -> AsyncMock:
        async def _async_operation(*args, **kwargs):
            if delay > 0:
                await asyncio.sleep(delay)
            if error:
                raise error
            return result

        return AsyncMock(side_effect=_async_operation)

    return _create_async_mock


# ============================================================================
# Collection Operations Schema Fixtures
# ============================================================================


@pytest.fixture
def basic_collection_schema():
    """
    Create a basic collection schema for testing.

    Coverage: Common schema structure for tests.
    """
    fields = [
        FieldSchema(
            name="entity_id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
            description="Primary key field",
        ),
        FieldSchema(
            name="vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=128,
            description="Vector embedding field",
        ),
    ]

    return CollectionSchema(fields=fields, description="Basic test collection", shard_num=2)


@pytest.fixture
def complex_collection_schema():
    """
    Create a complex collection schema for testing.

    Coverage: Complex schema with multiple field types.
    """
    fields = [
        FieldSchema(
            name="entity_id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
            description="Primary key field",
        ),
        FieldSchema(
            name="text", dtype=DataType.VARCHAR, max_length=500, description="Text content field"
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=512,
            description="Text embedding vector",
        ),
        FieldSchema(name="metadata", dtype=DataType.JSON, description="Metadata JSON field"),
        FieldSchema(
            name="tags",
            dtype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            description="Tags array field",
        ),
        FieldSchema(
            name="tenant_id",
            dtype=DataType.INT64,
            is_partition_key=True,
            description="Tenant partition key",
        ),
    ]

    return CollectionSchema(
        fields=fields, description="Complex test collection", enable_dynamic_field=True, shard_num=5
    )


@pytest.fixture
def minimal_collection_schema():
    """
    Create a minimal collection schema for testing edge cases.

    Coverage: Minimal schema structure.
    """
    fields = [
        FieldSchema(
            name="entity_id",
            dtype=DataType.INT64,
            is_primary=True,
            description="Minimal primary key",
        ),
    ]

    return CollectionSchema(fields=fields, description="Minimal test collection", shard_num=1)


# ============================================================================
# Collection Operations Entity Fixtures
# ============================================================================


@pytest.fixture
def sample_collection_description(basic_collection_schema):
    """
    Create a sample CollectionDescription for testing.

    Coverage: CollectionDescription creation.
    """
    return CollectionDescription(
        name="test_collection",
        collection_schema=basic_collection_schema,
        id="test_collection_id",
        created_at=datetime.now(timezone.utc),
        schema_hash="abc123def456",
        state=CollectionState.AVAILABLE,
        load_state=LoadState.LOADED,
        created_at_is_synthetic=False,
    )


@pytest.fixture
def sample_collection_stats():
    """
    Create a sample CollectionStats for testing.

    Coverage: CollectionStats creation.
    """
    segments = [
        SegmentInfo(
            segment_id="seg1",
            collection_id="test_collection_id",
            partition_id="part1",
            num_rows=1000,
            state="Sealed",
            created_at=datetime.now(timezone.utc),
            memory_size=1280000,
            disk_size=2560000,
            index_name="vector_idx",
        ),
        SegmentInfo(
            segment_id="seg2",
            collection_id="test_collection_id",
            partition_id="part1",
            num_rows=500,
            state="Sealed",
            created_at=datetime.now(timezone.utc),
            memory_size=640000,
            disk_size=1280000,
            index_name="vector_idx",
        ),
    ]

    partitions = [
        PartitionInfo(
            partition_id="part1",
            name="_default",
            collection_id="test_collection_id",
            created_at=datetime.now(timezone.utc),
            num_segments=2,
            num_rows=1500,
        ),
    ]

    return CollectionStats(
        name="test_collection",
        id="test_collection_id",
        created_at=datetime.now(timezone.utc),
        row_count=1500,
        memory_size=1920000,
        disk_size=3840000,
        index_size=512000,
        partitions=partitions,
        segments=segments,
    )


@pytest.fixture
def sample_load_progress():
    """
    Create a sample LoadProgress for testing.

    Coverage: LoadProgress creation.
    """
    return LoadProgress(
        collection_name="test_collection",
        state=LoadState.LOADING,
        progress=0.75,
        loaded_segments=3,
        total_segments=4,
        error_message=None,
    )


# ============================================================================
# Collection Operations Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_connection_manager():
    """
    Create a mock ConnectionManager for testing.

    Coverage: ConnectionManager mocking.

    This fixture creates a mock that respects both return_value and side_effect,
    allowing tests to control behavior in different ways.
    """
    mock_manager = MagicMock()

    # Create a smart wrapper that respects both return_value and side_effect
    from collections import namedtuple

    Call = namedtuple("Call", ["args", "kwargs"])

    class ExecuteOperationMock:
        def __init__(self):
            self._return_value = None
            self._use_return = False
            self._side_effect = None
            self._call_args = None

        async def __call__(self, operation, timeout=None):
            # Store call arguments for inspection in MagicMock-style format
            self._call_args = Call(args=(operation,), kwargs={"timeout": timeout})

            # Check side_effect first (standard AsyncMock behavior)
            if self._side_effect is not None:
                if callable(self._side_effect):
                    return self._side_effect(operation, timeout)
                else:
                    raise self._side_effect

            # Then check return_value
            if self._use_return:
                return self._return_value

            # Finally, execute the operation
            return operation("mock_alias")

        @property
        def call_args(self):
            """Return the arguments from the last call in MagicMock format."""
            return self._call_args

        def set_return_value(self, value):
            self._use_return = True
            self._return_value = value

        @property
        def return_value(self):
            return self._return_value

        @return_value.setter
        def return_value(self, value):
            self.set_return_value(value)

        @property
        def side_effect(self):
            return self._side_effect

        @side_effect.setter
        def side_effect(self, value):
            self._side_effect = value

    execute_mock = ExecuteOperationMock()
    mock_manager.execute_operation_async = execute_mock

    return mock_manager


@pytest.fixture
def mock_pymilvus_utility() -> Generator[MagicMock, None, None]:
    """
    Mock the pymilvus.utility module for testing.

    Coverage: Tests interaction with pymilvus utility functions.
    """
    with patch("pymilvus.utility") as mock_utility:
        # Mock utility functions
        mock_utility.has_collection = MagicMock(return_value=False)
        mock_utility.list_collections = MagicMock(return_value=[])
        mock_utility.describe_collection = MagicMock(return_value={})
        mock_utility.load_state = MagicMock(return_value="NotLoaded")
        mock_utility.loading_progress = MagicMock(return_value="0%")
        mock_utility.get_collection_stats = MagicMock(return_value={})

        yield mock_utility


@pytest.fixture
def mock_pymilvus_collection() -> MagicMock:
    """
    Create a mock pymilvus.Collection object for testing.

    Coverage: Tests interaction with pymilvus Collection instances.
    """
    mock_collection = MagicMock()
    mock_collection.name = "test_collection"
    mock_collection.schema = MagicMock()
    mock_collection.description = "Test collection"
    mock_collection.is_loaded = False
    mock_collection.loading_progress = "0%"
    mock_collection.num_entities = 0
    mock_collection.create = MagicMock()
    mock_collection.drop = MagicMock()
    mock_collection.load = MagicMock()
    mock_collection.release = MagicMock()
    mock_collection.insert = MagicMock(return_value=MagicMock(ids=[1, 2, 3]))

    return mock_collection


@pytest.fixture
def mock_pymilvus_datatype() -> Generator[MagicMock, None, None]:
    """
    Mock the pymilvus DataType module for testing.

    Coverage: Tests interaction with pymilvus DataType enums.
    """
    with patch("pymilvus.DataType") as mock_datatype:
        # Mock common data types
        mock_datatype.INT64 = 1
        mock_datatype.FLOAT_VECTOR = 101
        mock_datatype.VARCHAR = 21
        mock_datatype.JSON = 23
        mock_datatype.ARRAY = 22

        yield mock_datatype


@pytest.fixture
def valid_test_data():
    """
    Create valid test data for insertion tests.

    Coverage: Valid data structures for testing data insertion.
    """
    return [
        {"entity_id": 1, "vector": [0.1] * 128, "text": "test text"},
        {"entity_id": 2, "vector": [0.2] * 128, "text": "another test"},
    ]


@pytest.fixture
def invalid_test_data():
    """
    Create invalid test data for insertion tests.

    Coverage: Invalid data structures for testing error handling.
    """
    return [
        {
            "entity_id": "invalid_id",  # Should be int
            "vector": [0.1] * 128,
        },
        {
            "entity_id": 3,
            "vector": "not_a_vector",  # Should be list of floats
        },
    ]


# ============================================================================
# Collection Operations Parameterized Test Data
# ============================================================================


@pytest.fixture
def vector_dimensions():
    """
    Provide various vector dimensions for parameterized tests.

    Coverage: Vector dimension testing.
    """
    return [
        1,  # Minimum
        8,  # Minimum binary vector
        64,  # Common dimension
        128,  # Common dimension
        256,  # Common binary vector
        512,  # Common dimension
        1024,  # Large dimension
        32768,  # Maximum float vector
    ]


@pytest.fixture
def varchar_lengths():
    """
    Provide various VARCHAR lengths for parameterized tests.

    Coverage: VARCHAR length testing.
    """
    return [
        1,  # Minimum
        255,  # Common length
        1000,  # Medium length
        65535,  # Maximum
    ]


@pytest.fixture
def shard_numbers():
    """
    Provide various shard numbers for parameterized tests.

    Coverage: Shard number testing.
    """
    return [
        1,  # Minimum
        2,  # Default
        5,  # Common
        10,  # Large
        100,  # Very large
    ]


@pytest.fixture
def reserved_field_names():
    """
    Provide reserved field names for testing.

    Coverage: Reserved name validation.
    """
    return [
        "id",
        "collection_name",
        "timestamp",
        "distance",
        "count",
        "score",
    ]


# ============================================================================
# Collection Operations Large Dataset Fixtures
# ============================================================================


@pytest.fixture
def large_segment_list():
    """
    Create a large list of segments for testing.

    Coverage: Large dataset testing.
    """
    segments = []
    for i in range(1000):
        segment = SegmentInfo(
            segment_id=f"seg_{i:04d}",
            collection_id="large_collection",
            partition_id="part1",
            num_rows=1000,
            state="Sealed",
            created_at=datetime.now(timezone.utc),
            memory_size=1280000,
            disk_size=2560000,
        )
        segments.append(segment)

    return segments


@pytest.fixture
def mock_milvus_stats_response():
    """
    Create a mock Milvus stats response.

    Coverage: Milvus API response simulation.
    """
    return {
        "row_count": 1500,
        "collection_id": "test_collection_id",
        "created_utc": 1234567890.0,
        "segments": [
            {
                "segment_id": "seg1",
                "partition_id": "part1",
                "num_rows": 1000,
                "state": "Sealed",
                "created_utc": 1234567890.0,
                "memory_size": 1280000,
                "disk_size": 2560000,
                "index_size": 256000,
            },
            {
                "segment_id": "seg2",
                "partition_id": "part1",
                "num_rows": 500,
                "state": "Sealed",
                "created_utc": 1234567890.0,
                "memory_size": 640000,
                "disk_size": 1280000,
                "index_size": 128000,
            },
        ],
        "partitions": [
            {
                "partition_id": "part1",
                "name": "_default",
                "created_utc": 1234567890.0,
            },
        ],
    }


@pytest.fixture
def mock_milvus_load_progress_response():
    """
    Create a mock Milvus load progress response.

    Coverage: Milvus load progress API response simulation.
    """
    return {
        "loading_progress": "75%",
        "loaded_segments": 3,
        "total_segments": 4,
        "error": None,
    }


# ============================================================================
# Circuit Breaker Test Fixtures
# ============================================================================


@pytest.fixture
def mock_circuit_breaker_config() -> dict[str, Any]:
    """
    Create circuit breaker configuration for testing.

    Coverage: Tests circuit breaker with various configurations.
    """
    return {
        "failure_threshold": 3,
        "recovery_timeout": 2.0,  # Short timeout for testing
        "half_open_success_threshold": 2,
        "max_half_open_requests": 1,
    }


# ============================================================================
# Time and Performance Test Fixtures
# ============================================================================


@pytest.fixture
def mock_time() -> Generator[MagicMock, None, None]:
    """
    Mock time module for testing time-dependent behavior.

    Coverage: Tests timeout handling, recovery delays, backoff calculations.
    """
    with (
        patch("time.time") as mock_time_func,
        patch("time.monotonic") as mock_monotonic,
        patch("time.sleep") as mock_sleep,
    ):
        current_time = 1000.0

        def _time_side_effect() -> float:
            return current_time

        def _monotonic_side_effect() -> float:
            return current_time

        def _sleep_side_effect(duration: float) -> None:
            nonlocal current_time
            current_time += duration

        mock_time_func.side_effect = _time_side_effect
        mock_monotonic.side_effect = _monotonic_side_effect
        mock_sleep.side_effect = _sleep_side_effect

        yield mock_time_func


@pytest.fixture
def performance_timer() -> Callable[[], float]:
    """
    Timer utility for performance benchmarks.

    Coverage: Performance test measurement accuracy.
    """

    def _timer() -> float:
        return time.perf_counter()

    return _timer


# ============================================================================
# Error Simulation Fixtures
# ============================================================================


@pytest.fixture
def error_scenarios() -> dict[str, Exception]:
    """
    Common error scenarios for testing error handling paths.

    Coverage: Tests all exception handling branches.
    """
    from milvus_ops.connection_management.connection_exceptions import (
        ConnectionError,
        ConnectionTimeoutError,
        ServerUnavailableError,
    )

    return {
        "connection_error": ConnectionError("Connection failed"),
        "timeout_error": ConnectionTimeoutError("Operation timed out"),
        "server_unavailable": ServerUnavailableError("Server unavailable"),
        "generic_error": Exception("Generic error"),
        "network_error": OSError("Network unreachable"),
    }


# ============================================================================
# Threading Test Fixtures
# ============================================================================


@pytest.fixture
def thread_count() -> int:
    """
    Default thread count for threading safety tests.

    Coverage: Multi-threading test configuration.
    """
    return 10


@pytest.fixture
def concurrent_operations() -> Callable[[int, Callable], list[Any]]:
    """
    Utility for running operations concurrently.

    Coverage: Tests concurrent access patterns.
    """
    import threading

    def _run_concurrent(num_threads: int, operation: Callable) -> list[Any]:
        results = []
        errors = []
        lock = threading.Lock()

        def _worker(thread_id: int) -> None:
            try:
                result = operation(thread_id)
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=_worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        if errors:
            raise Exception(f"Errors in concurrent operations: {errors}")

        return results

    return _run_concurrent


# ============================================================================
# Test Data Fixtures
# ============================================================================


@dataclass
class TestConnection:
    """Test connection data structure."""

    alias: str
    healthy: bool
    last_used: float


@pytest.fixture
def connection_aliases() -> list[str]:
    """
    Generate connection aliases for testing.

    Coverage: Tests with multiple connection identifiers.
    """
    return [f"conn_{i}" for i in range(5)]


@pytest.fixture
def test_operation() -> Callable[[str], str]:
    """
    Simple test operation function.

    Coverage: Tests operation execution through connection manager.
    """

    def _operation(conn_alias: str) -> str:
        return f"Operation result for {conn_alias}"

    return _operation


@pytest.fixture
def failing_operation() -> Callable[[str], None]:
    """
    Test operation that always fails.

    Coverage: Tests error handling and retry logic.
    """
    from milvus_ops.connection_management.connection_exceptions import (
        ConnectionError,
    )

    def _operation(conn_alias: str) -> None:
        raise ConnectionError(f"Operation failed for {conn_alias}")

    return _operation


@pytest.fixture
def slow_operation() -> Callable[[str, float], str]:
    """
    Test operation that simulates slow execution.

    Coverage: Tests timeout handling.
    """

    def _operation(conn_alias: str, delay: float = 1.0) -> str:
        time.sleep(delay)
        return f"Slow operation result for {conn_alias}"

    return _operation


# ============================================================================
# Data Management Operations Fixtures
# ============================================================================


@pytest.fixture
def mock_collection_manager():
    """
    Create a mock CollectionManager for testing data management operations.

    Coverage: CollectionManager mocking for data operations tests.
    """
    mock_manager = MagicMock()

    # Mock async methods with AsyncMock
    mock_manager.has_collection = AsyncMock(return_value=True)
    mock_manager.describe_collection = AsyncMock()
    mock_manager.create_collection = AsyncMock()
    mock_manager.load_collection = AsyncMock()

    return mock_manager


@pytest.fixture
def sample_documents():
    """
    Create sample Document instances for testing.

    Coverage: Document creation and validation tests.
    """
    from milvus_ops.data_management_operations.models.entities import Document

    return [
        Document(id=1, vector=[0.1] * 128, text="First document"),
        Document(id=2, vector=[0.2] * 128, text="Second document"),
        Document(id=3, vector=[0.3] * 128, text="Third document"),
    ]


@pytest.fixture
def sample_document_dicts():
    """
    Create sample plain dictionary documents for testing.

    Coverage: Dictionary document handling tests.
    """
    return [
        {"id": 1, "vector": [0.1] * 128, "text": "First document"},
        {"id": 2, "vector": [0.2] * 128, "text": "Second document"},
        {"id": 3, "vector": [0.3] * 128, "text": "Third document"},
    ]


@pytest.fixture
def invalid_documents():
    """
    Create documents with validation errors for testing error handling.

    Coverage: Validation error handling tests.
    """
    return [
        {"id": "invalid_id", "vector": [0.1] * 128},  # String ID when int expected
        {"id": 2, "vector": "not_a_vector"},  # Invalid vector type
        {"id": 3, "vector": [0.1] * 64},  # Wrong vector dimension
        {"missing_id": True, "vector": [0.1] * 128},  # Missing required field
        {"id": 4, "extra_field": "not_in_schema"},  # Extraneous field (if schema doesn't allow)
    ]


@pytest.fixture
def batch_operation_result_samples():
    """
    Create sample BatchOperationResult instances for different scenarios.

    Coverage: BatchOperationResult property tests.
    """
    from milvus_ops.data_management_operations.models.entities import (
        BatchOperationResult,
        OperationStatus,
    )

    return {
        "success": BatchOperationResult(
            status=OperationStatus.SUCCESS,
            successful_count=10,
            failed_count=0,
            inserted_ids=list(range(1, 11)),
        ),
        "partial": BatchOperationResult(
            status=OperationStatus.PARTIAL,
            successful_count=8,
            failed_count=2,
            inserted_ids=list(range(1, 9)),
            error_messages={"9": "Validation error", "10": "Type mismatch"},
        ),
        "failed": BatchOperationResult(
            status=OperationStatus.FAILED,
            successful_count=0,
            failed_count=5,
            error_messages={str(i): f"Error {i}" for i in range(1, 6)},
        ),
    }


@pytest.fixture
def mock_data_manager(mock_connection_manager, mock_collection_manager):
    """
    Factory fixture to create DataManager instances with mocked dependencies.

    Coverage: DataManager initialization tests with various configurations.
    """
    from milvus_ops.data_management_operations import (
        DataManager,
        DataOperationConfig,
    )

    def _create_data_manager(config: DataOperationConfig | None = None):
        return DataManager(
            connection_manager=mock_connection_manager,
            collection_manager=mock_collection_manager,
            config=config,
        )

    return _create_data_manager


@pytest.fixture
def mock_pymilvus_collection_insert():
    """
    Create a mock PyMilvus Collection.insert() return value.

    Coverage: Insert operation mocking.
    """
    mock_result = MagicMock()
    mock_result.primary_keys = [1, 2, 3]
    return mock_result


@pytest.fixture
def mock_pymilvus_collection_upsert():
    """
    Create a mock PyMilvus Collection.upsert() return value.

    Coverage: Upsert operation mocking.
    """
    mock_result = MagicMock()
    mock_result.primary_keys = [1, 2, 3]
    return mock_result


@pytest.fixture
def mock_pymilvus_collection_delete():
    """
    Create a mock PyMilvus Collection.delete() return value.

    Coverage: Delete operation mocking.
    """
    mock_result = MagicMock()
    mock_result.delete_count = 5
    return mock_result


@pytest.fixture
def data_ops_config_variations():
    """
    Provide various DataOperationConfig setups for parameterized tests.

    Coverage: DataOperationConfig with different parameter combinations.
    """
    from milvus_ops.data_management_operations import DataOperationConfig

    return {
        "default": DataOperationConfig(),
        "custom_batch": DataOperationConfig(default_batch_size=500, max_batch_size=5000),
        "no_timeout": DataOperationConfig(default_operation_timeout=None),
        "no_retry": DataOperationConfig(retry_transient_errors=False),
        "timing_disabled": DataOperationConfig(enable_timing=False),
        "strict_validation": DataOperationConfig(strict_validation=True),
        "relaxed_validation": DataOperationConfig(strict_validation=False),
    }


@pytest.fixture
def sample_collection_description_for_data_ops(basic_collection_schema):
    """
    Create a sample CollectionDescription for data operations tests.

    Coverage: CollectionDescription creation for data operation tests.
    """
    from milvus_ops.collection_operations.entities import (
        CollectionDescription,
        CollectionState,
        LoadState,
    )

    return CollectionDescription(
        name="test_collection",
        collection_schema=basic_collection_schema,
        id="test_collection_id",
        created_at=datetime.now(timezone.utc),
        schema_hash="abc123def456",
        state=CollectionState.AVAILABLE,
        load_state=LoadState.LOADED,
        created_at_is_synthetic=False,
    )


@pytest.fixture
def complex_documents_with_metadata():
    """
    Create complex documents with metadata for testing.

    Coverage: Complex document structure tests.
    """
    from milvus_ops.data_management_operations.models.entities import Document

    return [
        Document(
            id=1,
            vector=[0.1] * 512,
            text="Complex document one",
            metadata={"category": "test", "priority": 1},
        ),
        Document(
            id=2,
            vector=[0.2] * 512,
            text="Complex document two",
            metadata={"category": "production", "priority": 2},
        ),
    ]


# ============================================================================
# Cleanup Utilities
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_singletons():
    """
    Cleanup singleton instances between tests.

    Coverage: Ensures test isolation by resetting singletons.
    """
    yield

    # Reset connection pool singleton
    from milvus_ops.connection_management.connection_pool import MilvusConnectionPool

    with MilvusConnectionPool._lock:
        MilvusConnectionPool._instance = None
        MilvusConnectionPool._lock = type(MilvusConnectionPool._lock)()


# ============================================================================
# Marker Registration
# ============================================================================


def pytest_configure(config):
    """
    Register custom pytest markers.

    Coverage: Test organization and selective test execution.
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "performance: marks tests as performance benchmarks")
    config.addinivalue_line("markers", "threading: marks tests that require threading")
    config.addinivalue_line("markers", "async_test: marks tests that use async functionality")
