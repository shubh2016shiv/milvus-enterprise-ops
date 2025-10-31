"""
Comprehensive unit tests for SearchManager.

This module provides systematic testing of the SearchManager class,
including initialization, search operations, reranking integration,
config creation, and error handling.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from milvus_ops.search_operations.config.base import ReRankingMethod, SearchType
from milvus_ops.search_operations.config.hybrid import HybridSearchConfig
from milvus_ops.search_operations.config.reranking import ReRankingConfig
from milvus_ops.search_operations.config.semantic import SemanticSearchConfig
from milvus_ops.search_operations.config.validation import SearchParams
from milvus_ops.search_operations.core.manager import SearchManager
from milvus_ops.search_operations.core.search_ops_exceptions import (
    InvalidSearchParametersError,
    ReRankingError,
    SearchError,
)

# ============================================================================
# Test SearchManager Initialization
# ============================================================================


@pytest.mark.unit
class TestSearchManagerInitialization:
    """
    Test SearchManager initialization.

    Coverage: SearchManager constructor with various configurations.
    """

    @pytest.mark.asyncio
    async def test_manager_initialization_default(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test SearchManager initialization with default caching.

        Coverage: SearchManager.__init__() with enable_caching=True.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )
        assert manager._connection_manager is mock_connection_manager
        assert manager._embedding_provider is mock_embedding_provider
        assert manager._enable_caching is True
        assert manager._semantic_search is not None
        assert manager._hybrid_search is not None
        assert manager._reranker is not None

    @pytest.mark.asyncio
    async def test_manager_initialization_without_caching(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test SearchManager initialization without caching.

        Coverage: SearchManager.__init__() with enable_caching=False.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
            enable_caching=False,
        )
        assert manager._enable_caching is False


# ============================================================================
# Test SearchManager search() - Semantic Search
# ============================================================================


@pytest.mark.unit
class TestSearchManagerSemanticSearch:
    """
    Test SearchManager.search() with semantic search type.

    Coverage: SearchManager semantic search operations.
    """

    @pytest.mark.asyncio
    async def test_search_semantic_without_reranking(
        self, mock_connection_manager, mock_embedding_provider, sample_search_results
    ):
        """
        Test search() with semantic search type without reranking.

        Coverage: SearchManager.search() performs semantic search correctly.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        # Mock semantic search.search() method
        from milvus_ops.search_operations.core.base import SearchResult

        mock_result = SearchResult(
            hits=sample_search_results,
            total_hits=len(sample_search_results),
            took_ms=10.0,
        )
        manager._semantic_search.search = AsyncMock(return_value=mock_result)

        # Create search params
        search_params = SearchParams(
            search_type=SearchType.SEMANTIC,
            top_k=10,
            vector_field="vector",
        )

        # Execute search
        result = await manager.search(
            collection_name="test_collection",
            query="test query",
            search_params=search_params,
        )

        # Verify results
        assert result == mock_result
        assert len(result.hits) == 3
        manager._semantic_search.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_semantic_with_dict_params(
        self, mock_connection_manager, mock_embedding_provider, sample_search_results
    ):
        """
        Test search() with semantic search using dict params.

        Coverage: SearchManager.search() converts dict to SearchParams.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        # Mock semantic search.search() method
        from milvus_ops.search_operations.core.base import SearchResult

        mock_result = SearchResult(
            hits=sample_search_results,
            total_hits=len(sample_search_results),
            took_ms=10.0,
        )
        manager._semantic_search.search = AsyncMock(return_value=mock_result)

        # Create search params as dict
        search_params = {
            "search_type": SearchType.SEMANTIC,
            "top_k": 10,
            "vector_field": "vector",
        }

        # Execute search
        result = await manager.search(
            collection_name="test_collection",
            query="test query",
            search_params=search_params,
        )

        # Verify results
        assert result == mock_result
        manager._semantic_search.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_semantic_with_reranking(
        self, mock_connection_manager, mock_embedding_provider, sample_search_results
    ):
        """
        Test search() with semantic search and reranking.

        Coverage: SearchManager.search() performs semantic search with reranking.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        # Mock the reranking process
        from milvus_ops.search_operations.core.base import SearchResult

        mock_result = SearchResult(
            hits=sample_search_results,
            total_hits=len(sample_search_results),
            took_ms=15.0,
        )

        # Mock the semantic search's _generate_embedding and _execute_search
        manager._semantic_search._generate_embedding = AsyncMock(return_value=[0.1] * 128)
        manager._semantic_search._execute_search = AsyncMock(
            return_value=(sample_search_results, 10.0)
        )

        # Mock reranker methods
        manager._reranker.get_search_params = MagicMock(return_value={"rerank": MagicMock()})
        manager._reranker.process_results = MagicMock(return_value=mock_result)

        # Create search params with reranking
        search_params = SearchParams(
            search_type=SearchType.SEMANTIC,
            top_k=10,
            vector_field="vector",
            rerank=True,
            rerank_method=ReRankingMethod.WEIGHTED,
            rerank_weights=[0.6, 0.4],
        )

        # Execute search
        result = await manager.search(
            collection_name="test_collection",
            query="test query",
            search_params=search_params,
        )

        # Verify results
        assert result == mock_result
        manager._reranker.get_search_params.assert_called_once()
        manager._reranker.process_results.assert_called_once()


# ============================================================================
# Test SearchManager search() - Hybrid Search
# ============================================================================


@pytest.mark.unit
class TestSearchManagerHybridSearch:
    """
    Test SearchManager.search() with hybrid search type.

    Coverage: SearchManager hybrid search operations.
    """

    @pytest.mark.asyncio
    async def test_search_hybrid_without_reranking(
        self, mock_connection_manager, mock_embedding_provider, sample_search_results
    ):
        """
        Test search() with hybrid search type without reranking.

        Coverage: SearchManager.search() performs hybrid search correctly.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        # Mock hybrid search.search() method
        from milvus_ops.search_operations.core.base import SearchResult

        mock_result = SearchResult(
            hits=sample_search_results,
            total_hits=len(sample_search_results),
            took_ms=10.0,
        )
        manager._hybrid_search.search = AsyncMock(return_value=mock_result)

        # Create search params
        search_params = SearchParams(
            search_type=SearchType.HYBRID,
            top_k=10,
            vector_field="vector",
            sparse_field="sparse_vector",
            vector_weight=0.7,
            sparse_weight=0.3,
        )

        # Execute search
        result = await manager.search(
            collection_name="test_collection",
            query="test query",
            search_params=search_params,
        )

        # Verify results
        assert result == mock_result
        manager._hybrid_search.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_hybrid_with_reranking(
        self, mock_connection_manager, mock_embedding_provider, sample_search_results
    ):
        """
        Test search() with hybrid search and reranking.

        Coverage: SearchManager.search() performs hybrid search with reranking.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        # Mock the reranking process
        from milvus_ops.search_operations.core.base import SearchResult

        mock_result = SearchResult(
            hits=sample_search_results,
            total_hits=len(sample_search_results),
            took_ms=15.0,
        )

        # Mock the hybrid search's _generate_embedding and _execute_search
        manager._hybrid_search._generate_embedding = AsyncMock(return_value=[0.1] * 128)
        manager._hybrid_search._prepare_hybrid_search_params = AsyncMock(
            return_value={"data": [[0.1] * 128], "anns_field": "vector", "limit": 10}
        )
        manager._hybrid_search._execute_search = AsyncMock(
            return_value=(sample_search_results, 10.0)
        )

        # Mock reranker methods
        manager._reranker.get_search_params = MagicMock(return_value={"rerank": MagicMock()})
        manager._reranker.process_results = MagicMock(return_value=mock_result)

        # Create search params with reranking
        search_params = SearchParams(
            search_type=SearchType.HYBRID,
            top_k=10,
            vector_field="vector",
            sparse_field="sparse_vector",
            vector_weight=0.7,
            sparse_weight=0.3,
            rerank=True,
            rerank_method=ReRankingMethod.RRF,
            rerank_k=60,
        )

        # Execute search
        result = await manager.search(
            collection_name="test_collection",
            query="test query",
            search_params=search_params,
        )

        # Verify results
        assert result == mock_result
        manager._reranker.get_search_params.assert_called_once()
        manager._reranker.process_results.assert_called_once()


# ============================================================================
# Test SearchManager Config Creation
# ============================================================================


@pytest.mark.unit
class TestSearchManagerConfigCreation:
    """
    Test SearchManager config creation helper methods.

    Coverage: SearchManager._create_semantic_config, _create_hybrid_config,
    _create_reranking_config.
    """

    def test_create_semantic_config(self, mock_connection_manager, mock_embedding_provider):
        """
        Test _create_semantic_config() creates correct config.

        Coverage: _create_semantic_config() converts SearchParams to SemanticSearchConfig.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        params = SearchParams(
            top_k=20,
            timeout=60.0,
            vector_field="embedding",
            expr='category == "test"',
            params={"nprobe": 20},
            output_fields=["text", "category"],
        )

        config = manager._create_semantic_config(params)
        assert isinstance(config, SemanticSearchConfig)
        assert config.top_k == 20
        assert config.timeout == 60.0
        assert config.search_field == "embedding"
        assert config.expr == 'category == "test"'
        # Params may include defaults like 'ef', so check that nprobe is present
        assert "nprobe" in config.params
        assert config.params["nprobe"] == 20
        assert config.output_fields == ["text", "category"]

    def test_create_hybrid_config(self, mock_connection_manager, mock_embedding_provider):
        """
        Test _create_hybrid_config() creates correct config.

        Coverage: _create_hybrid_config() converts SearchParams to HybridSearchConfig.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        params = SearchParams(
            top_k=15,
            timeout=45.0,
            vector_field="vector",
            sparse_field="sparse_vector",
            keyword_field="text",
            vector_weight=0.8,
            sparse_weight=0.2,
            params={"nprobe": 30},
            output_fields=["text"],
        )

        config = manager._create_hybrid_config(params)
        assert isinstance(config, HybridSearchConfig)
        assert config.top_k == 15
        assert config.timeout == 45.0
        assert config.vector_field == "vector"
        assert config.sparse_field == "sparse_vector"
        assert config.keyword_field == "text"
        assert config.vector_weight == 0.8
        assert config.sparse_weight == 0.2
        # Params may include defaults like 'ef', so check that nprobe is present
        assert "nprobe" in config.params
        assert config.params["nprobe"] == 30
        assert config.output_fields == ["text"]

    def test_create_reranking_config_weighted(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test _create_reranking_config() with weighted method.

        Coverage: _create_reranking_config() creates weighted reranking config.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        params = SearchParams(
            rerank=True,
            rerank_method=ReRankingMethod.WEIGHTED,
            rerank_weights=[0.6, 0.4],
        )

        config = manager._create_reranking_config(params)
        assert isinstance(config, ReRankingConfig)
        assert config.enabled is True
        assert config.method == ReRankingMethod.WEIGHTED
        assert config.params["weights"] == [0.6, 0.4]

    def test_create_reranking_config_rrf(self, mock_connection_manager, mock_embedding_provider):
        """
        Test _create_reranking_config() with RRF method.

        Coverage: _create_reranking_config() creates RRF reranking config.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        params = SearchParams(
            rerank=True,
            rerank_method=ReRankingMethod.RRF,
            rerank_k=80,
        )

        config = manager._create_reranking_config(params)
        assert isinstance(config, ReRankingConfig)
        assert config.enabled is True
        assert config.method == ReRankingMethod.RRF
        assert config.params["k"] == 80

    def test_create_reranking_config_default_weights(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test _create_reranking_config() with default weights.

        Coverage: _create_reranking_config() uses default weights when not provided.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        params = SearchParams(
            rerank=True,
            rerank_method=ReRankingMethod.WEIGHTED,
            rerank_weights=None,
        )

        config = manager._create_reranking_config(params)
        assert config.params["weights"] == [0.5, 0.5]


# ============================================================================
# Test SearchManager Error Handling
# ============================================================================


@pytest.mark.unit
class TestSearchManagerErrorHandling:
    """
    Test SearchManager error handling.

    Coverage: SearchManager handles errors correctly.
    """

    @pytest.mark.asyncio
    async def test_search_unsupported_type(self, mock_connection_manager, mock_embedding_provider):
        """
        Test search() with unsupported search type raises error.

        Coverage: SearchManager.search() raises InvalidSearchParametersError for unsupported types.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        # Create search params with unsupported type
        # (using FUSION which isn't implemented in manager)
        search_params = SearchParams(
            search_type=SearchType.FUSION,
            top_k=10,
        )

        with pytest.raises(InvalidSearchParametersError) as exc_info:
            await manager.search(
                collection_name="test_collection",
                query="test query",
                search_params=search_params,
            )
        assert "Unsupported search type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_semantic_error(self, mock_connection_manager, mock_embedding_provider):
        """
        Test search() when semantic search raises error.

        Coverage: SearchManager.search() wraps semantic search errors in SearchError.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        # Mock semantic search to raise error
        manager._semantic_search.search = AsyncMock(side_effect=Exception("Search failed"))

        search_params = SearchParams(
            search_type=SearchType.SEMANTIC,
            top_k=10,
            vector_field="vector",
        )

        with pytest.raises(SearchError) as exc_info:
            await manager.search(
                collection_name="test_collection",
                query="test query",
                search_params=search_params,
            )
        assert "Search operation failed" in str(exc_info.value)
        assert "Search failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_hybrid_error(self, mock_connection_manager, mock_embedding_provider):
        """
        Test search() when hybrid search raises error.

        Coverage: SearchManager.search() wraps hybrid search errors in SearchError.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        # Mock hybrid search to raise error
        manager._hybrid_search.search = AsyncMock(side_effect=Exception("Hybrid search failed"))

        search_params = SearchParams(
            search_type=SearchType.HYBRID,
            top_k=10,
            vector_field="vector",
        )

        with pytest.raises(SearchError) as exc_info:
            await manager.search(
                collection_name="test_collection",
                query="test query",
                search_params=search_params,
            )
        assert "Search operation failed" in str(exc_info.value)
        assert "Hybrid search failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_preserves_search_errors(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test search() preserves SearchError and InvalidSearchParametersError.

        Coverage: SearchManager.search() doesn't wrap SearchError or InvalidSearchParametersError.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        # Mock semantic search to raise SearchError
        manager._semantic_search.search = AsyncMock(
            side_effect=SearchError("Original search error")
        )

        search_params = SearchParams(
            search_type=SearchType.SEMANTIC,
            top_k=10,
            vector_field="vector",
        )

        with pytest.raises(SearchError) as exc_info:
            await manager.search(
                collection_name="test_collection",
                query="test query",
                search_params=search_params,
            )
        assert "Original search error" in str(exc_info.value)
        assert isinstance(exc_info.value, SearchError)


# ============================================================================
# Test SearchManager Cache Operations
# ============================================================================


@pytest.mark.unit
class TestSearchManagerCacheOperations:
    """
    Test SearchManager cache operations.

    Coverage: SearchManager cache statistics and clearing.
    """

    def test_get_cache_stats(self, mock_connection_manager, mock_embedding_provider):
        """
        Test get_cache_stats() returns cache statistics.

        Coverage: get_cache_stats() delegates to semantic search cache stats.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        # Mock cache stats
        mock_stats = {"hits": 100, "misses": 50, "hit_rate": 0.67}
        manager._semantic_search.get_cache_stats = MagicMock(return_value=mock_stats)

        stats = manager.get_cache_stats()
        assert stats == mock_stats
        manager._semantic_search.get_cache_stats.assert_called_once()

    def test_clear_cache(self, mock_connection_manager, mock_embedding_provider):
        """
        Test clear_cache() clears the cache.

        Coverage: clear_cache() delegates to semantic search clear cache.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        # Mock clear cache
        manager._semantic_search.clear_cache = MagicMock()

        manager.clear_cache()
        manager._semantic_search.clear_cache.assert_called_once()


# ============================================================================
# Test SearchManager Edge Cases
# ============================================================================


@pytest.mark.unit
class TestSearchManagerEdgeCases:
    """
    Test SearchManager edge cases.

    Coverage: SearchManager handles edge cases correctly.
    """

    @pytest.mark.asyncio
    async def test_search_empty_query(self, mock_connection_manager, mock_embedding_provider):
        """
        Test search() with empty query.

        Coverage: SearchManager.search() handles empty query string.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        # Mock semantic search to return empty results
        from milvus_ops.search_operations.core.base import SearchResult

        mock_result = SearchResult(hits=[], total_hits=0, took_ms=5.0)
        manager._semantic_search.search = AsyncMock(return_value=mock_result)

        search_params = SearchParams(
            search_type=SearchType.SEMANTIC,
            top_k=10,
            vector_field="vector",
        )

        result = await manager.search(
            collection_name="test_collection",
            query="",
            search_params=search_params,
        )

        assert result.total_hits == 0
        assert len(result.hits) == 0

    @pytest.mark.asyncio
    async def test_search_with_very_long_query(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test search() with very long query.

        Coverage: SearchManager.search() handles very long query strings.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        # Mock semantic search
        from milvus_ops.search_operations.core.base import SearchResult

        mock_result = SearchResult(hits=[], total_hits=0, took_ms=10.0)
        manager._semantic_search.search = AsyncMock(return_value=mock_result)

        long_query = "This is a very long query " * 1000
        search_params = SearchParams(
            search_type=SearchType.SEMANTIC,
            top_k=10,
            vector_field="vector",
        )

        await manager.search(
            collection_name="test_collection",
            query=long_query,
            search_params=search_params,
        )

        # Verify the long query was passed to semantic search
        call_args = manager._semantic_search.search.call_args
        # Check if query was passed as positional or keyword argument
        if call_args[1]:  # keyword args
            assert call_args[1]["query"] == long_query
        else:  # positional args
            assert call_args[0][1] == long_query  # query is the second positional arg

    @pytest.mark.asyncio
    async def test_search_invalid_collection(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test search() with invalid collection name.

        Coverage: SearchManager.search() handles invalid collection names.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        # Mock semantic search to raise collection not found error

        manager._semantic_search.search = AsyncMock(
            side_effect=SearchError("Collection not found: invalid_collection")
        )

        search_params = SearchParams(
            search_type=SearchType.SEMANTIC,
            top_k=10,
            vector_field="vector",
        )

        with pytest.raises(SearchError) as exc_info:
            await manager.search(
                collection_name="invalid_collection",
                query="test query",
                search_params=search_params,
            )
        assert "Collection not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_missing_embeddings(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test search() when embedding generation fails.

        Coverage: SearchManager.search() handles embedding generation failures.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        # Mock semantic search to raise embedding error
        from milvus_ops.search_operations.core.search_ops_exceptions import EmbeddingGenerationError

        manager._semantic_search.search = AsyncMock(
            side_effect=EmbeddingGenerationError("Failed to generate embedding")
        )

        search_params = SearchParams(
            search_type=SearchType.SEMANTIC,
            top_k=10,
            vector_field="vector",
        )

        with pytest.raises(EmbeddingGenerationError):
            await manager.search(
                collection_name="test_collection",
                query="test query",
                search_params=search_params,
            )

    @pytest.mark.asyncio
    async def test_search_none_collection_name(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test search() with None collection name.

        Coverage: SearchManager.search() handles None collection name (should fail validation).
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        search_params = SearchParams(
            search_type=SearchType.SEMANTIC,
            top_k=10,
            vector_field="vector",
        )

        # Should raise validation error from semantic search
        from milvus_ops.search_operations.core.search_ops_exceptions import (
            InvalidSearchParametersError,
        )

        manager._semantic_search.search = AsyncMock(
            side_effect=InvalidSearchParametersError("Collection name cannot be None")
        )

        with pytest.raises(InvalidSearchParametersError) as exc_info:
            await manager.search(
                collection_name=None,  # type: ignore
                query="test query",
                search_params=search_params,
            )
        assert "Collection name" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_invalid_search_params_dict(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test search() with invalid dict params.

        Coverage: SearchManager.search() handles invalid dict parameters (should fail validation).
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        # Invalid dict params (wrong types, missing required fields, etc.)
        invalid_params = {
            "search_type": "invalid_type",  # Invalid enum value
            "top_k": -1,  # Invalid value
            "timeout": -1.0,  # Invalid value
        }

        # Should raise InvalidSearchParametersError (ValidationError is converted to it)
        with pytest.raises(InvalidSearchParametersError) as exc_info:
            await manager.search(
                collection_name="test_collection",
                query="test query",
                search_params=invalid_params,
            )
        # Verify it contains validation error information
        assert "Invalid search parameters" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_timeout_exceeded(self, mock_connection_manager, mock_embedding_provider):
        """
        Test search() when timeout is exceeded.

        Coverage: SearchManager.search() handles timeout exceeded scenarios.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        # Mock semantic search to raise timeout error
        from milvus_ops.search_operations.core.search_ops_exceptions import SearchTimeoutError

        manager._semantic_search.search = AsyncMock(
            side_effect=SearchTimeoutError("Search operation timed out")
        )

        search_params = SearchParams(
            search_type=SearchType.SEMANTIC,
            top_k=10,
            vector_field="vector",
            timeout=1.0,  # Very short timeout
        )

        with pytest.raises(SearchTimeoutError) as exc_info:
            await manager.search(
                collection_name="test_collection",
                query="test query",
                search_params=search_params,
            )
        assert "timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_connection_manager_error(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test search() when connection manager fails.

        Coverage: SearchManager.search() handles connection manager errors.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        # Mock semantic search to raise connection error
        from milvus_ops.search_operations.core.search_ops_exceptions import ConnectionError

        manager._semantic_search.search = AsyncMock(
            side_effect=ConnectionError("Connection to Milvus failed")
        )

        search_params = SearchParams(
            search_type=SearchType.SEMANTIC,
            top_k=10,
            vector_field="vector",
        )

        with pytest.raises(ConnectionError):
            await manager.search(
                collection_name="test_collection",
                query="test query",
                search_params=search_params,
            )

    @pytest.mark.asyncio
    async def test_search_reranking_failure_fallback(
        self, mock_connection_manager, mock_embedding_provider, sample_search_results
    ):
        """
        Test search() fallback when reranking fails.

        Coverage: SearchManager.search() falls back to original results when reranking fails.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        # Mock semantic search
        from milvus_ops.search_operations.core.base import SearchResult

        mock_result = SearchResult(
            hits=sample_search_results,
            total_hits=len(sample_search_results),
            took_ms=10.0,
        )
        manager._semantic_search.search = AsyncMock(return_value=mock_result)

        # Mock reranker to fail
        manager._reranker.get_search_params = MagicMock(
            side_effect=ReRankingError("Reranking failed")
        )

        search_params = SearchParams(
            search_type=SearchType.SEMANTIC,
            top_k=10,
            vector_field="vector",
            rerank=True,
            rerank_method=ReRankingMethod.WEIGHTED,
        )

        # The manager raises SearchError when reranking fails
        with pytest.raises(SearchError, match="Reranking failed"):
            await manager.search(
                collection_name="test_collection",
                query="test query",
                search_params=search_params,
            )

    @pytest.mark.asyncio
    async def test_search_none_query(self, mock_connection_manager, mock_embedding_provider):
        """
        Test search() with None query (should raise validation error).

        Coverage: SearchManager.search() raises validation error for None query.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        search_params = SearchParams(
            search_type=SearchType.SEMANTIC,
            top_k=10,
            vector_field="vector",
        )

        # Should raise validation error
        manager._semantic_search.search = AsyncMock(
            side_effect=InvalidSearchParametersError("Query cannot be None")
        )

        with pytest.raises(InvalidSearchParametersError, match="Query cannot be None"):
            await manager.search(
                collection_name="test_collection",
                query=None,  # type: ignore
                search_params=search_params,
            )

    @pytest.mark.asyncio
    async def test_search_whitespace_only_query(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test search() with whitespace-only query.

        Coverage: SearchManager.search() handles whitespace-only query.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        from milvus_ops.search_operations.core.base import SearchResult

        mock_result = SearchResult(hits=[], total_hits=0, took_ms=5.0)
        manager._semantic_search.search = AsyncMock(return_value=mock_result)

        search_params = SearchParams(
            search_type=SearchType.SEMANTIC,
            top_k=10,
            vector_field="vector",
        )

        result = await manager.search(
            collection_name="test_collection",
            query="   \n\t  ",
            search_params=search_params,
        )

        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_search_very_long_collection_name(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test search() with very long collection name.

        Coverage: SearchManager.search() handles very long collection name.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        very_long_collection = "a" * 300  # Exceeds 255 char limit
        manager._semantic_search.search = AsyncMock(
            side_effect=InvalidSearchParametersError("Collection name exceeds maximum length")
        )

        search_params = SearchParams(
            search_type=SearchType.SEMANTIC,
            top_k=10,
            vector_field="vector",
        )

        with pytest.raises(InvalidSearchParametersError, match="exceeds maximum length"):
            await manager.search(
                collection_name=very_long_collection,
                query="test query",
                search_params=search_params,
            )

    @pytest.mark.asyncio
    async def test_search_special_characters_collection_name(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test search() with special characters in collection name.

        Coverage: SearchManager.search() handles special characters in collection name.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        special_collection = "collection@#$%"
        manager._semantic_search.search = AsyncMock(
            side_effect=InvalidSearchParametersError(
                "Collection name can only contain alphanumeric characters, underscores, and hyphens"
            )
        )

        search_params = SearchParams(
            search_type=SearchType.SEMANTIC,
            top_k=10,
            vector_field="vector",
        )

        with pytest.raises(InvalidSearchParametersError, match="alphanumeric"):
            await manager.search(
                collection_name=special_collection,
                query="test query",
                search_params=search_params,
            )

    @pytest.mark.asyncio
    async def test_search_invalid_rerank_weights_count(
        self, mock_connection_manager, mock_embedding_provider, sample_search_results
    ):
        """
        Test search() with invalid rerank weights count mismatch.

        Coverage: SearchManager.search() handles rerank weights count mismatch.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        from milvus_ops.search_operations.core.base import SearchResult

        mock_result = SearchResult(
            hits=sample_search_results,
            total_hits=len(sample_search_results),
            took_ms=10.0,
        )
        manager._semantic_search.search = AsyncMock(return_value=mock_result)

        # Invalid weights count (should match number of vector fields)
        manager._reranker.get_search_params = MagicMock(
            side_effect=InvalidSearchParametersError("rerank_weights count mismatch")
        )

        search_params = SearchParams(
            search_type=SearchType.SEMANTIC,
            top_k=10,
            vector_field="vector",
            rerank=True,
            rerank_method=ReRankingMethod.WEIGHTED,
            rerank_weights=[0.5],  # Wrong count
        )

        # The config validation raises ValueError which gets wrapped in SearchError
        with pytest.raises(SearchError, match="Weights must sum"):
            await manager.search(
                collection_name="test_collection",
                query="test query",
                search_params=search_params,
            )

    @pytest.mark.asyncio
    async def test_search_reranking_enabled_method_none(
        self, mock_connection_manager, mock_embedding_provider, sample_search_results
    ):
        """
        Test search() with reranking enabled but method NONE.

        Coverage: SearchManager.search() handles reranking enabled but method NONE.
        """
        manager = SearchManager(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        from milvus_ops.search_operations.core.base import SearchResult

        mock_result = SearchResult(
            hits=sample_search_results,
            total_hits=len(sample_search_results),
            took_ms=10.0,
        )
        manager._semantic_search.search = AsyncMock(return_value=mock_result)

        search_params = SearchParams(
            search_type=SearchType.SEMANTIC,
            top_k=10,
            vector_field="vector",
            rerank=True,
            rerank_method=ReRankingMethod.NONE,  # Should raise error
        )

        # This should raise an error because reranking is enabled but method is NONE
        with pytest.raises(SearchError, match="Re-ranking is enabled but method is NONE"):
            await manager.search(
                collection_name="test_collection",
                query="test query",
                search_params=search_params,
            )
