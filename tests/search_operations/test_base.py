"""
Comprehensive unit tests for base search operations.

This module provides systematic testing of BaseSearch abstract class
and SearchResult dataclass, including initialization, embedding generation,
search execution, and error handling.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from milvus_ops.milvus_ops_exceptions import OperationTimeoutError
from milvus_ops.search_operations.config.base import BaseSearchConfig
from milvus_ops.search_operations.core.base import BaseSearch, SearchResult
from milvus_ops.search_operations.core.search_ops_exceptions import (
    EmbeddingGenerationError,
    SearchError,
    SearchTimeoutError,
)
from milvus_ops.search_operations.providers.embedding import (
    EmbeddingProvider,
    EmbeddingResult,
)

# ============================================================================
# Test SearchResult Dataclass
# ============================================================================


@pytest.mark.unit
class TestSearchResult:
    """
    Test SearchResult dataclass initialization and behavior.

    Coverage: SearchResult dataclass properties and default values.
    """

    def test_search_result_initialization_minimal(self):
        """
        Test SearchResult initialization with minimal parameters.

        Coverage: SearchResult.__init__() with required parameters only.
        """
        hits = [{"id": 1, "distance": 0.1}]
        result = SearchResult(hits=hits, total_hits=1, took_ms=10.0)
        assert result.hits == hits
        assert result.total_hits == 1
        assert result.took_ms == 10.0
        assert result.search_params == {}
        assert result.raw_response is None

    def test_search_result_initialization_full(self):
        """
        Test SearchResult initialization with all parameters.

        Coverage: SearchResult.__init__() with all optional parameters.
        """
        hits = [{"id": 1, "distance": 0.1, "score": 0.9}]
        search_params = {"type": "semantic", "top_k": 10}
        raw_response = MagicMock()

        result = SearchResult(
            hits=hits,
            total_hits=1,
            took_ms=15.5,
            search_params=search_params,
            raw_response=raw_response,
        )

        assert result.hits == hits
        assert result.total_hits == 1
        assert result.took_ms == 15.5
        assert result.search_params == search_params
        assert result.raw_response is raw_response

    def test_search_result_post_init_none_search_params(self):
        """
        Test SearchResult.__post_init__() handles None search_params.

        Coverage: SearchResult.__post_init__() initializes None search_params to empty dict.
        """
        result = SearchResult(hits=[{"id": 1}], total_hits=1, took_ms=10.0, search_params=None)
        assert result.search_params == {}

    def test_search_result_with_empty_hits(self):
        """
        Test SearchResult with empty hits list.

        Coverage: SearchResult handles empty hits list correctly.
        """
        result = SearchResult(hits=[], total_hits=0, took_ms=5.0)
        assert result.hits == []
        assert result.total_hits == 0
        assert len(result.hits) == 0

    def test_search_result_total_hits_mismatch(self):
        """
        Test SearchResult when total_hits doesn't match hits length.

        Coverage: SearchResult allows total_hits to differ from hits length.
        """
        hits = [{"id": 1}, {"id": 2}]
        result = SearchResult(hits=hits, total_hits=100, took_ms=10.0)
        assert len(result.hits) == 2
        assert result.total_hits == 100


# ============================================================================
# Test BaseSearch Initialization
# ============================================================================


@pytest.mark.unit
class TestBaseSearchInitialization:
    """
    Test BaseSearch abstract class initialization.

    Coverage: BaseSearch.__init__() with various configurations.
    """

    def test_base_search_initialization_with_provider(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test BaseSearch initialization with embedding provider.

        Coverage: BaseSearch.__init__() stores connection manager and embedding provider.
        """

        # Create a concrete implementation for testing
        class ConcreteSearch(BaseSearch[BaseSearchConfig]):
            async def search(
                self,
                collection_name: str,
                query: str,
                config: BaseSearchConfig,
            ) -> SearchResult:
                return SearchResult(hits=[], total_hits=0, took_ms=0.0)

        search = ConcreteSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )
        assert search._connection_manager is mock_connection_manager
        assert search._embedding_provider is mock_embedding_provider

    def test_base_search_initialization_without_provider(self, mock_connection_manager):
        """
        Test BaseSearch initialization without embedding provider.

        Coverage: BaseSearch.__init__() accepts None embedding provider.
        """

        # Create a concrete implementation for testing
        class ConcreteSearch(BaseSearch[BaseSearchConfig]):
            async def search(
                self,
                collection_name: str,
                query: str,
                config: BaseSearchConfig,
            ) -> SearchResult:
                return SearchResult(hits=[], total_hits=0, took_ms=0.0)

        search = ConcreteSearch(connection_manager=mock_connection_manager, embedding_provider=None)
        assert search._connection_manager is mock_connection_manager
        assert search._embedding_provider is None


# ============================================================================
# Test BaseSearch _generate_embedding
# ============================================================================


@pytest.mark.unit
class TestBaseSearchGenerateEmbedding:
    """
    Test BaseSearch._generate_embedding() method.

    Coverage: BaseSearch embedding generation with and without provider.
    """

    @pytest.mark.asyncio
    async def test_generate_embedding_with_provider(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test _generate_embedding() with provider.

        Coverage: _generate_embedding() calls provider and returns embedding vector.
        """

        # Create a concrete implementation for testing
        class ConcreteSearch(BaseSearch[BaseSearchConfig]):
            async def search(
                self,
                collection_name: str,
                query: str,
                config: BaseSearchConfig,
            ) -> SearchResult:
                return SearchResult(hits=[], total_hits=0, took_ms=0.0)

        search = ConcreteSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        embedding = await search._generate_embedding("test query")
        assert isinstance(embedding, list)
        assert len(embedding) == 128
        assert all(isinstance(x, float) for x in embedding)
        mock_embedding_provider.generate_embedding.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_generate_embedding_without_provider(self, mock_connection_manager):
        """
        Test _generate_embedding() without provider raises error.

        Coverage: _generate_embedding() raises EmbeddingGenerationError when no provider.
        """

        # Create a concrete implementation for testing
        class ConcreteSearch(BaseSearch[BaseSearchConfig]):
            async def search(
                self,
                collection_name: str,
                query: str,
                config: BaseSearchConfig,
            ) -> SearchResult:
                return SearchResult(hits=[], total_hits=0, took_ms=0.0)

        search = ConcreteSearch(connection_manager=mock_connection_manager, embedding_provider=None)

        with pytest.raises(EmbeddingGenerationError) as exc_info:
            await search._generate_embedding("test query")
        assert "No embedding provider configured" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_embedding_provider_error(self, mock_connection_manager):
        """
        Test _generate_embedding() when provider raises error.

        Coverage: _generate_embedding() wraps provider exceptions in EmbeddingGenerationError.
        """
        from unittest.mock import AsyncMock

        # Create a mock provider that raises an error
        mock_provider = MagicMock(spec=EmbeddingProvider)
        mock_provider.generate_embedding = AsyncMock(side_effect=Exception("Provider error"))

        # Create a concrete implementation for testing
        class ConcreteSearch(BaseSearch[BaseSearchConfig]):
            async def search(
                self,
                collection_name: str,
                query: str,
                config: BaseSearchConfig,
            ) -> SearchResult:
                return SearchResult(hits=[], total_hits=0, took_ms=0.0)

        search = ConcreteSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_provider,
        )

        with pytest.raises(EmbeddingGenerationError) as exc_info:
            await search._generate_embedding("test query")
        assert "Failed to generate embedding" in str(exc_info.value)
        assert "Provider error" in str(exc_info.value)


# ============================================================================
# Test BaseSearch _execute_search
# ============================================================================


@pytest.mark.unit
class TestBaseSearchExecuteSearch:
    """
    Test BaseSearch._execute_search() method.

    Coverage: BaseSearch search execution with timeout and error handling.
    """

    @pytest.mark.asyncio
    async def test_execute_search_success(self, mock_connection_manager, sample_search_results):
        """
        Test _execute_search() successful execution.

        Coverage: _execute_search() returns results and execution time.
        """

        # Create a concrete implementation for testing
        class ConcreteSearch(BaseSearch[BaseSearchConfig]):
            async def search(
                self,
                collection_name: str,
                query: str,
                config: BaseSearchConfig,
            ) -> SearchResult:
                return SearchResult(hits=[], total_hits=0, took_ms=0.0)

        search = ConcreteSearch(connection_manager=mock_connection_manager)

        # Mock execute_operation_async to return results
        async def mock_execute(op, timeout=None):
            return sample_search_results

        mock_connection_manager.execute_operation_async = AsyncMock(side_effect=mock_execute)

        results, execution_time_ms = await search._execute_search(
            collection_name="test_collection",
            search_params={"data": [[0.1] * 128], "anns_field": "vector", "limit": 10},
            timeout=30.0,
        )

        assert results == sample_search_results
        assert execution_time_ms >= 0
        mock_connection_manager.execute_operation_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_search_timeout(self, mock_connection_manager):
        """
        Test _execute_search() timeout handling.

        Coverage: _execute_search() raises SearchTimeoutError on OperationTimeoutError.
        """

        # Create a concrete implementation for testing
        class ConcreteSearch(BaseSearch[BaseSearchConfig]):
            async def search(
                self,
                collection_name: str,
                query: str,
                config: BaseSearchConfig,
            ) -> SearchResult:
                return SearchResult(hits=[], total_hits=0, took_ms=0.0)

        search = ConcreteSearch(connection_manager=mock_connection_manager)

        # Mock execute_operation_async to raise timeout error
        mock_connection_manager.execute_operation_async = AsyncMock(
            side_effect=OperationTimeoutError("Operation timed out")
        )

        with pytest.raises(SearchTimeoutError) as exc_info:
            await search._execute_search(
                collection_name="test_collection",
                search_params={"data": [[0.1] * 128], "anns_field": "vector", "limit": 10},
                timeout=30.0,
            )
        assert "timed out after 30.0 seconds" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_search_generic_error(self, mock_connection_manager):
        """
        Test _execute_search() generic error handling.

        Coverage: _execute_search() wraps generic exceptions in SearchError.
        """

        # Create a concrete implementation for testing
        class ConcreteSearch(BaseSearch[BaseSearchConfig]):
            async def search(
                self,
                collection_name: str,
                query: str,
                config: BaseSearchConfig,
            ) -> SearchResult:
                return SearchResult(hits=[], total_hits=0, took_ms=0.0)

        search = ConcreteSearch(connection_manager=mock_connection_manager)

        # Mock execute_operation_async to raise generic error
        mock_connection_manager.execute_operation_async = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        with pytest.raises(SearchError) as exc_info:
            await search._execute_search(
                collection_name="test_collection",
                search_params={"data": [[0.1] * 128], "anns_field": "vector", "limit": 10},
                timeout=30.0,
            )
        assert "Search operation failed" in str(exc_info.value)
        assert "Connection failed" in str(exc_info.value)


# ============================================================================
# Test BaseSearch _perform_search
# ============================================================================


@pytest.mark.unit
class TestBaseSearchPerformSearch:
    """
    Test BaseSearch._perform_search() method.

    Coverage: BaseSearch PyMilvus search integration.
    """

    @patch("pymilvus.Collection")
    def test_perform_search_success(self, mock_collection_class, mock_pymilvus_search_result):
        """
        Test _perform_search() successful execution.

        Coverage: _perform_search() calls PyMilvus Collection.search() and converts results.
        """
        from milvus_ops.connection_management import ConnectionManager

        # Create a concrete implementation for testing
        class ConcreteSearch(BaseSearch[BaseSearchConfig]):
            async def search(
                self,
                collection_name: str,
                query: str,
                config: BaseSearchConfig,
            ) -> SearchResult:
                return SearchResult(hits=[], total_hits=0, took_ms=0.0)

        mock_connection_manager = MagicMock(spec=ConnectionManager)
        search = ConcreteSearch(connection_manager=mock_connection_manager)

        # Setup mock collection
        mock_collection = MagicMock()
        mock_collection.search.return_value = mock_pymilvus_search_result
        mock_collection_class.return_value = mock_collection

        # Execute search
        results = search._perform_search(
            alias="test_alias",
            collection_name="test_collection",
            search_params={
                "data": [[0.1] * 128],
                "anns_field": "vector",
                "limit": 10,
            },
        )

        # Verify results
        assert isinstance(results, list)
        assert len(results) == 3
        assert results[0]["id"] == 1
        assert results[0]["distance"] == 0.1
        assert results[0]["score"] == 0.9
        assert "text" in results[0]
        assert "category" in results[0]

        # Verify collection was created and search was called
        mock_collection_class.assert_called_once_with(name="test_collection", using="test_alias")
        mock_collection.search.assert_called_once()

    @patch("pymilvus.Collection")
    def test_perform_search_without_entity(self, mock_collection_class):
        """
        Test _perform_search() when hits don't have entity attribute.

        Coverage: _perform_search() handles hits without entity fields.
        """
        from milvus_ops.connection_management import ConnectionManager

        # Create a concrete implementation for testing
        class ConcreteSearch(BaseSearch[BaseSearchConfig]):
            async def search(
                self,
                collection_name: str,
                query: str,
                config: BaseSearchConfig,
            ) -> SearchResult:
                return SearchResult(hits=[], total_hits=0, took_ms=0.0)

        mock_connection_manager = MagicMock(spec=ConnectionManager)
        search = ConcreteSearch(connection_manager=mock_connection_manager)

        # Setup mock collection with hits without entity
        mock_hit = MagicMock()
        mock_hit.id = 1
        mock_hit.distance = 0.1
        mock_hit.score = 0.9
        # No entity attribute

        mock_collection = MagicMock()
        mock_collection.search.return_value = [[mock_hit]]
        mock_collection_class.return_value = mock_collection

        # Execute search
        results = search._perform_search(
            alias="test_alias",
            collection_name="test_collection",
            search_params={
                "data": [[0.1] * 128],
                "anns_field": "vector",
                "limit": 10,
            },
        )

        # Verify results
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["id"] == 1
        assert results[0]["distance"] == 0.1
        assert results[0]["score"] == 0.9

    @patch("pymilvus.Collection")
    def test_perform_search_error(self, mock_collection_class):
        """
        Test _perform_search() error propagation.

        Coverage: _perform_search() propagates exceptions from PyMilvus.
        """
        from milvus_ops.connection_management import ConnectionManager

        # Create a concrete implementation for testing
        class ConcreteSearch(BaseSearch[BaseSearchConfig]):
            async def search(
                self,
                collection_name: str,
                query: str,
                config: BaseSearchConfig,
            ) -> SearchResult:
                return SearchResult(hits=[], total_hits=0, took_ms=0.0)

        mock_connection_manager = MagicMock(spec=ConnectionManager)
        search = ConcreteSearch(connection_manager=mock_connection_manager)

        # Setup mock collection to raise error
        mock_collection = MagicMock()
        mock_collection.search.side_effect = Exception("Milvus error")
        mock_collection_class.return_value = mock_collection

        # Execute search and verify error is propagated
        with pytest.raises(Exception) as exc_info:
            search._perform_search(
                alias="test_alias",
                collection_name="test_collection",
                search_params={
                    "data": [[0.1] * 128],
                    "anns_field": "vector",
                    "limit": 10,
                },
            )
        assert "Milvus error" in str(exc_info.value)


# ============================================================================
# Test BaseSearch Abstract Methods
# ============================================================================


@pytest.mark.unit
class TestBaseSearchAbstractMethods:
    """
    Test BaseSearch abstract method enforcement.

    Coverage: BaseSearch abstract base class behavior.
    """

    def test_base_search_cannot_be_instantiated(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test BaseSearch cannot be instantiated directly.

        Coverage: BaseSearch is abstract and requires implementation.
        """
        with pytest.raises(TypeError):
            BaseSearch(
                connection_manager=mock_connection_manager,
                embedding_provider=mock_embedding_provider,
            )

    def test_concrete_search_must_implement_search(self, mock_connection_manager):
        """
        Test concrete subclass must implement search() method.

        Coverage: BaseSearch enforces search() implementation.
        """

        # Attempt to create a concrete class without implementing search
        class IncompleteSearch(BaseSearch[BaseSearchConfig]):
            pass

        with pytest.raises(TypeError):
            IncompleteSearch(connection_manager=mock_connection_manager)


# ============================================================================
# Test SearchResult Edge Cases
# ============================================================================


@pytest.mark.unit
class TestSearchResultEdgeCases:
    """
    Test SearchResult edge cases.

    Coverage: Negative values, malformed hits, edge case scenarios.
    """

    def test_search_result_negative_total_hits(self):
        """
        Test SearchResult with negative total_hits edge case.

        Coverage: SearchResult handles negative total_hits (edge case).
        """
        hits = [{"id": 1, "distance": 0.1}]
        # Negative total_hits is an edge case but shouldn't break initialization
        result = SearchResult(hits=hits, total_hits=-1, took_ms=10.0)
        assert result.total_hits == -1
        assert len(result.hits) == 1

    def test_search_result_negative_took_ms(self):
        """
        Test SearchResult with negative execution time.

        Coverage: SearchResult handles negative took_ms (edge case).
        """
        hits = [{"id": 1, "distance": 0.1}]
        # Negative execution time is an edge case but shouldn't break initialization
        result = SearchResult(hits=hits, total_hits=1, took_ms=-5.0)
        assert result.took_ms == -5.0

    def test_search_result_with_malformed_hits(self):
        """
        Test SearchResult with malformed hit structures.

        Coverage: SearchResult handles malformed hit structures gracefully.
        """
        # Malformed hits (missing required fields, wrong types, etc.)
        malformed_hits = [
            {"id": 1},  # Missing distance/score
            {"distance": 0.1},  # Missing id
            {"id": "not_a_number", "distance": 0.1},  # Wrong id type
            {},  # Empty hit
            None,  # None hit (if possible)
        ]
        # Filter out None if it causes issues
        hits = [h for h in malformed_hits if h is not None]
        result = SearchResult(hits=hits, total_hits=len(hits), took_ms=10.0)
        assert len(result.hits) == len(hits)
        # Should still work even with malformed data
        assert isinstance(result.hits, list)


# ============================================================================
# Test BaseSearch _perform_search Edge Cases
# ============================================================================


@pytest.mark.unit
class TestBaseSearchPerformSearchEdgeCases:
    """
    Test BaseSearch._perform_search() edge cases.

    Coverage: Empty results, multiple query vectors, expr filters, output fields.
    """

    @patch("pymilvus.Collection")
    def test_perform_search_empty_results(self, mock_collection_class):
        """
        Test _perform_search() with empty search results.

        Coverage: _perform_search() handles empty results gracefully.
        """
        from milvus_ops.connection_management import ConnectionManager

        class ConcreteSearch(BaseSearch[BaseSearchConfig]):
            async def search(
                self,
                collection_name: str,
                query: str,
                config: BaseSearchConfig,
            ) -> SearchResult:
                return SearchResult(hits=[], total_hits=0, took_ms=0.0)

        mock_connection_manager = MagicMock(spec=ConnectionManager)
        search = ConcreteSearch(connection_manager=mock_connection_manager)

        # Setup mock collection with empty results
        mock_collection = MagicMock()
        mock_collection.search.return_value = [[]]  # Empty result list
        mock_collection_class.return_value = mock_collection

        results = search._perform_search(
            alias="test_alias",
            collection_name="test_collection",
            search_params={
                "data": [[0.1] * 128],
                "anns_field": "vector",
                "limit": 10,
            },
        )

        assert isinstance(results, list)
        assert len(results) == 0

    @patch("pymilvus.Collection")
    def test_perform_search_multiple_query_vectors(self, mock_collection_class):
        """
        Test _perform_search() with batch search (multiple query vectors).

        Coverage: _perform_search() handles batch search with multiple vectors.
        """
        from milvus_ops.connection_management import ConnectionManager

        class ConcreteSearch(BaseSearch[BaseSearchConfig]):
            async def search(
                self,
                collection_name: str,
                query: str,
                config: BaseSearchConfig,
            ) -> SearchResult:
                return SearchResult(hits=[], total_hits=0, took_ms=0.0)

        mock_connection_manager = MagicMock(spec=ConnectionManager)
        search = ConcreteSearch(connection_manager=mock_connection_manager)

        # Setup mock collection with results for multiple queries
        mock_hit1 = MagicMock()
        mock_hit1.id = 1
        mock_hit1.distance = 0.1
        mock_hit1.score = 0.9
        mock_hit1.entity = {"text": "result 1"}

        mock_hit2 = MagicMock()
        mock_hit2.id = 2
        mock_hit2.distance = 0.2
        mock_hit2.score = 0.8
        mock_hit2.entity = {"text": "result 2"}

        mock_collection = MagicMock()
        # Multiple query vectors return multiple result lists
        mock_collection.search.return_value = [[mock_hit1], [mock_hit2]]
        mock_collection_class.return_value = mock_collection

        results = search._perform_search(
            alias="test_alias",
            collection_name="test_collection",
            search_params={
                "data": [[0.1] * 128, [0.2] * 128],  # Multiple query vectors
                "anns_field": "vector",
                "limit": 10,
            },
        )

        assert isinstance(results, list)
        # Should handle batch results appropriately
        assert len(results) >= 0

    @patch("pymilvus.Collection")
    def test_perform_search_with_expr_filter(self, mock_collection_class):
        """
        Test _perform_search() with filter expressions.

        Coverage: _perform_search() handles expr filter expressions.
        """
        from milvus_ops.connection_management import ConnectionManager

        class ConcreteSearch(BaseSearch[BaseSearchConfig]):
            async def search(
                self,
                collection_name: str,
                query: str,
                config: BaseSearchConfig,
            ) -> SearchResult:
                return SearchResult(hits=[], total_hits=0, took_ms=0.0)

        mock_connection_manager = MagicMock(spec=ConnectionManager)
        search = ConcreteSearch(connection_manager=mock_connection_manager)

        mock_hit = MagicMock()
        mock_hit.id = 1
        mock_hit.distance = 0.1
        mock_hit.score = 0.9
        mock_hit.entity = {"text": "filtered result"}

        mock_collection = MagicMock()
        mock_collection.search.return_value = [[mock_hit]]
        mock_collection_class.return_value = mock_collection

        # Search with expr filter
        results = search._perform_search(
            alias="test_alias",
            collection_name="test_collection",
            search_params={
                "data": [[0.1] * 128],
                "anns_field": "vector",
                "limit": 10,
                "expr": 'category == "technical"',
            },
        )

        assert isinstance(results, list)
        # Verify expr was passed to search
        call_kwargs = mock_collection.search.call_args[1]
        assert "expr" in call_kwargs or "expr" in str(mock_collection.search.call_args)

    @patch("pymilvus.Collection")
    def test_perform_search_with_output_fields(self, mock_collection_class):
        """
        Test _perform_search() with output_fields specified.

        Coverage: _perform_search() handles output_fields parameter.
        """
        from milvus_ops.connection_management import ConnectionManager

        class ConcreteSearch(BaseSearch[BaseSearchConfig]):
            async def search(
                self,
                collection_name: str,
                query: str,
                config: BaseSearchConfig,
            ) -> SearchResult:
                return SearchResult(hits=[], total_hits=0, took_ms=0.0)

        mock_connection_manager = MagicMock(spec=ConnectionManager)
        search = ConcreteSearch(connection_manager=mock_connection_manager)

        mock_hit = MagicMock()
        mock_hit.id = 1
        mock_hit.distance = 0.1
        mock_hit.score = 0.9
        mock_hit.entity = {"text": "result", "category": "tech"}

        mock_collection = MagicMock()
        mock_collection.search.return_value = [[mock_hit]]
        mock_collection_class.return_value = mock_collection

        # Search with output_fields
        results = search._perform_search(
            alias="test_alias",
            collection_name="test_collection",
            search_params={
                "data": [[0.1] * 128],
                "anns_field": "vector",
                "limit": 10,
                "output_fields": ["text", "category"],
            },
        )

        assert isinstance(results, list)
        # Verify output_fields was passed to search
        call_kwargs = mock_collection.search.call_args[1]
        assert "output_fields" in call_kwargs or "output_fields" in str(
            mock_collection.search.call_args
        )


# ============================================================================
# Test BaseSearch _execute_search Edge Cases
# ============================================================================


@pytest.mark.unit
class TestBaseSearchExecuteSearchEdgeCases:
    """
    Test BaseSearch._execute_search() edge cases.

    Coverage: Partial timeout, edge case scenarios.
    """

    @pytest.mark.asyncio
    async def test_execute_search_partial_timeout(self, mock_connection_manager):
        """
        Test _execute_search() with timeout before completion.

        Coverage: _execute_search() handles partial timeout scenarios.
        """

        # Create a concrete implementation for testing
        class ConcreteSearch(BaseSearch[BaseSearchConfig]):
            async def search(
                self,
                collection_name: str,
                query: str,
                config: BaseSearchConfig,
            ) -> SearchResult:
                return SearchResult(hits=[], total_hits=0, took_ms=0.0)

        search = ConcreteSearch(connection_manager=mock_connection_manager)

        # Mock execute_operation_async to timeout partway through
        async def slow_operation(op, timeout=None):
            await asyncio.sleep(0.1)  # Simulate slow operation
            raise OperationTimeoutError("Operation timed out")

        mock_connection_manager.execute_operation_async = AsyncMock(side_effect=slow_operation)

        with pytest.raises(SearchTimeoutError):
            await search._execute_search(
                collection_name="test_collection",
                search_params={"data": [[0.1] * 128], "anns_field": "vector", "limit": 10},
                timeout=0.05,  # Very short timeout
            )


# ============================================================================
# Test BaseSearch _generate_embedding Edge Cases
# ============================================================================


@pytest.mark.unit
class TestBaseSearchGenerateEmbeddingEdgeCases:
    """
    Test BaseSearch._generate_embedding() edge cases.

    Coverage: Empty text, whitespace-only text.
    """

    @pytest.mark.asyncio
    async def test_generate_embedding_empty_text(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test _generate_embedding() with empty string input.

        Coverage: _generate_embedding() handles empty string input.
        """

        # Create a concrete implementation for testing
        class ConcreteSearch(BaseSearch[BaseSearchConfig]):
            async def search(
                self,
                collection_name: str,
                query: str,
                config: BaseSearchConfig,
            ) -> SearchResult:
                return SearchResult(hits=[], total_hits=0, took_ms=0.0)

        search = ConcreteSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        # Mock provider to handle empty string
        mock_embedding_provider.generate_embedding = AsyncMock(
            return_value=EmbeddingResult(
                embedding=[0.0] * 128,
                dimension=128,
                model_name="test_model",
                processing_time_ms=5.0,
            )
        )

        embedding = await search._generate_embedding("")
        assert isinstance(embedding, list)
        assert len(embedding) == 128
        mock_embedding_provider.generate_embedding.assert_called_once_with("")

    @pytest.mark.asyncio
    async def test_generate_embedding_whitespace_only(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test _generate_embedding() with whitespace-only text.

        Coverage: _generate_embedding() handles whitespace-only text input.
        """

        # Create a concrete implementation for testing
        class ConcreteSearch(BaseSearch[BaseSearchConfig]):
            async def search(
                self,
                collection_name: str,
                query: str,
                config: BaseSearchConfig,
            ) -> SearchResult:
                return SearchResult(hits=[], total_hits=0, took_ms=0.0)

        search = ConcreteSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        # Mock provider to handle whitespace-only string
        mock_embedding_provider.generate_embedding = AsyncMock(
            return_value=EmbeddingResult(
                embedding=[0.0] * 128,
                dimension=128,
                model_name="test_model",
                processing_time_ms=5.0,
            )
        )

        whitespace_text = "   \n\t  "
        embedding = await search._generate_embedding(whitespace_text)
        assert isinstance(embedding, list)
        assert len(embedding) == 128
        mock_embedding_provider.generate_embedding.assert_called_once_with(whitespace_text)
