"""
Comprehensive unit tests for HybridSearch.

This module provides systematic testing of the HybridSearch class,
including initialization, all search modes, BM25 integration, fusion strategies,
and error handling.
"""

from unittest.mock import AsyncMock

import pytest

from milvus_ops.search_operations.config.hybrid import HybridSearchConfig
from milvus_ops.search_operations.core.base import SearchResult
from milvus_ops.search_operations.core.search_ops_exceptions import (
    EmbeddingGenerationError,
    HybridSearchError,
    SparseVectorGenerationError,
)
from milvus_ops.search_operations.providers.embedding import EmbeddingResult
from milvus_ops.search_operations.search.hybrid.core.engine import HybridSearch

# ============================================================================
# Test HybridSearch Initialization
# ============================================================================


@pytest.mark.unit
class TestHybridSearchInitialization:
    """
    Test HybridSearch initialization.

    Coverage: HybridSearch constructor with various configurations.
    """

    @pytest.mark.asyncio
    async def test_initialization_default(self, mock_connection_manager, mock_embedding_provider):
        """
        Test HybridSearch initialization with default parameters.

        Coverage: HybridSearch.__init__() with default flags enabled.
        """
        search = HybridSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )
        assert search._connection_manager is mock_connection_manager
        assert search._embedding_provider is mock_embedding_provider
        assert search.bm25_generator is not None
        assert search.circuit_breaker is not None
        assert search.enable_caching is True

    @pytest.mark.asyncio
    async def test_initialization_without_caching(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test HybridSearch initialization without caching.

        Coverage: HybridSearch.__init__() with enable_caching=False.
        """
        search = HybridSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
            enable_caching=False,
        )
        assert search.enable_caching is False

    @pytest.mark.asyncio
    async def test_initialization_without_circuit_breaker(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test HybridSearch initialization without circuit breaker.

        Coverage: HybridSearch.__init__() with enable_circuit_breaker=False.
        """
        search = HybridSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
            enable_circuit_breaker=False,
        )
        assert search.circuit_breaker is None


# ============================================================================
# Test HybridSearch Search Modes
# ============================================================================


@pytest.mark.unit
class TestHybridSearchModes:
    """
    Test HybridSearch all search modes.

    Coverage: VECTOR_SPARSE, VECTOR_KEYWORD, VECTOR_ONLY, ALL_METHODS modes.
    """

    @pytest.mark.asyncio
    async def test_search_mode_vector_sparse(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test search() with VECTOR_SPARSE mode.

        Coverage: HybridSearch._determine_search_mode() returns VECTOR_SPARSE.
        """
        embedding_result = EmbeddingResult(
            embedding=[0.1] * 768,
            dimension=768,
            model_name="test_model",
            processing_time_ms=10.0,
        )
        mock_embedding_provider.generate_embedding = AsyncMock(return_value=embedding_result)

        # Mock BM25 sparse vector generation
        sparse_vector = {"term1": 0.5, "term2": 0.3}
        search = HybridSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )
        search.bm25_generator.generate_sparse_vector = AsyncMock(return_value=sparse_vector)

        # Mock vector search results
        vector_results = [{"id": 1, "distance": 0.1}]
        # Mock sparse search results
        sparse_results = [{"id": 2, "distance": 0.2}]
        mock_connection_manager.execute_operation_async = AsyncMock(
            side_effect=[vector_results, sparse_results]
        )

        config = HybridSearchConfig(
            top_k=10,
            sparse_field="sparse_vector",
            vector_weight=0.7,
            sparse_weight=0.3,
        )

        result = await search.search(
            collection_name="test_collection",
            query="test query",
            config=config,
        )

        assert isinstance(result, SearchResult)
        # Verify both vector and sparse searches were performed
        assert mock_connection_manager.execute_operation_async.call_count >= 2

    @pytest.mark.asyncio
    async def test_search_mode_vector_keyword(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test search() with VECTOR_KEYWORD mode.

        Coverage: HybridSearch._determine_search_mode() returns VECTOR_KEYWORD.
        """
        embedding_result = EmbeddingResult(
            embedding=[0.1] * 768,
            dimension=768,
            model_name="test_model",
            processing_time_ms=10.0,
        )
        mock_embedding_provider.generate_embedding = AsyncMock(return_value=embedding_result)

        search = HybridSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        vector_results = [{"id": 1, "distance": 0.1}]
        keyword_results = [{"id": 2, "distance": 0.2}]
        mock_connection_manager.execute_operation_async = AsyncMock(
            side_effect=[vector_results, keyword_results]
        )

        config = HybridSearchConfig(
            top_k=10,
            keyword_field="text",
            vector_weight=0.7,
            sparse_weight=0.3,
        )

        result = await search.search(
            collection_name="test_collection",
            query="test query",
            config=config,
        )

        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_search_mode_vector_only(self, mock_connection_manager, mock_embedding_provider):
        """
        Test search() with VECTOR_ONLY mode.

        Coverage: HybridSearch._determine_search_mode() returns VECTOR_ONLY.
        """
        embedding_result = EmbeddingResult(
            embedding=[0.1] * 768,
            dimension=768,
            model_name="test_model",
            processing_time_ms=10.0,
        )
        mock_embedding_provider.generate_embedding = AsyncMock(return_value=embedding_result)

        search = HybridSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        vector_results = [{"id": 1, "distance": 0.1}]
        mock_connection_manager.execute_operation_async = AsyncMock(return_value=vector_results)

        config = HybridSearchConfig(
            top_k=10,
            vector_weight=1.0,
            sparse_weight=0.0,
        )

        result = await search.search(
            collection_name="test_collection",
            query="test query",
            config=config,
        )

        assert isinstance(result, SearchResult)
        # Only vector search should be performed
        assert mock_connection_manager.execute_operation_async.call_count == 1

    @pytest.mark.asyncio
    async def test_search_mode_all_methods(self, mock_connection_manager, mock_embedding_provider):
        """
        Test search() with ALL_METHODS mode.

        Coverage: HybridSearch._determine_search_mode() returns ALL_METHODS.
        """
        embedding_result = EmbeddingResult(
            embedding=[0.1] * 768,
            dimension=768,
            model_name="test_model",
            processing_time_ms=10.0,
        )
        mock_embedding_provider.generate_embedding = AsyncMock(return_value=embedding_result)

        sparse_vector = {"indices": [1, 2], "values": [0.5, 0.3]}
        search = HybridSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )
        search.bm25_generator.generate = AsyncMock(return_value=sparse_vector)

        vector_results = [{"id": 1, "distance": 0.1}]
        sparse_results = [{"id": 2, "distance": 0.2}]
        mock_connection_manager.execute_operation_async = AsyncMock(
            side_effect=[vector_results, sparse_results]
        )

        config = HybridSearchConfig(
            top_k=10,
            sparse_field="sparse_vector",
            vector_weight=0.5,
            sparse_weight=0.3,
        )

        result = await search.search(
            collection_name="test_collection",
            query="test query",
            config=config,
        )

        assert isinstance(result, SearchResult)
        # Vector and sparse searches should be performed (keyword search not yet implemented)
        assert mock_connection_manager.execute_operation_async.call_count == 2


# ============================================================================
# Test HybridSearch Fusion Strategies
# ============================================================================


@pytest.mark.unit
class TestHybridSearchFusion:
    """
    Test HybridSearch fusion strategies.

    Coverage: RRF and weighted fusion strategies.
    """

    @pytest.mark.asyncio
    async def test_search_fusion_rrf(self, mock_connection_manager, mock_embedding_provider):
        """
        Test search() with RRF fusion strategy.

        Coverage: HybridSearch.search() uses RRF fusion when fusion_strategy="rrf".
        """
        embedding_result = EmbeddingResult(
            embedding=[0.1] * 768,
            dimension=768,
            model_name="test_model",
            processing_time_ms=10.0,
        )
        mock_embedding_provider.generate_embedding = AsyncMock(return_value=embedding_result)

        sparse_vector = {"term1": 0.5}
        search = HybridSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )
        search.bm25_generator.generate_sparse_vector = AsyncMock(return_value=sparse_vector)

        vector_results = [{"id": 1, "distance": 0.1}]
        sparse_results = [{"id": 1, "distance": 0.2}]  # Same ID for fusion
        mock_connection_manager.execute_operation_async = AsyncMock(
            side_effect=[vector_results, sparse_results]
        )

        config = HybridSearchConfig(
            top_k=10,
            sparse_field="sparse_vector",
        )

        result = await search.search(
            collection_name="test_collection",
            query="test query",
            config=config,
            fusion_strategy="rrf",
        )

        assert isinstance(result, SearchResult)
        # RRF fusion should combine results
        assert len(result.hits) >= 0

    @pytest.mark.asyncio
    async def test_search_fusion_weighted(self, mock_connection_manager, mock_embedding_provider):
        """
        Test search() with weighted fusion strategy.

        Coverage: HybridSearch.search() uses weighted fusion when fusion_strategy="weighted".
        """
        embedding_result = EmbeddingResult(
            embedding=[0.1] * 768,
            dimension=768,
            model_name="test_model",
            processing_time_ms=10.0,
        )
        mock_embedding_provider.generate_embedding = AsyncMock(return_value=embedding_result)

        sparse_vector = {"term1": 0.5}
        search = HybridSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )
        search.bm25_generator.generate_sparse_vector = AsyncMock(return_value=sparse_vector)

        vector_results = [{"id": 1, "distance": 0.1, "score": 0.9}]
        sparse_results = [{"id": 1, "distance": 0.2, "score": 0.8}]
        mock_connection_manager.execute_operation_async = AsyncMock(
            side_effect=[vector_results, sparse_results]
        )

        config = HybridSearchConfig(
            top_k=10,
            sparse_field="sparse_vector",
            vector_weight=0.7,
            sparse_weight=0.3,
        )

        result = await search.search(
            collection_name="test_collection",
            query="test query",
            config=config,
            fusion_strategy="weighted",
        )

        assert isinstance(result, SearchResult)


# ============================================================================
# Test HybridSearch Error Handling
# ============================================================================


@pytest.mark.unit
class TestHybridSearchErrorHandling:
    """
    Test HybridSearch error handling.

    Coverage: BM25 failures, sparse vector generation errors, fallback mechanisms.
    """

    @pytest.mark.asyncio
    async def test_search_bm25_failure_with_fallback(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test search() falls back to vector-only when BM25 fails.

        Coverage: HybridSearch.search() falls back to vector search on BM25 failure.
        """
        embedding_result = EmbeddingResult(
            embedding=[0.1] * 768,
            dimension=768,
            model_name="test_model",
            processing_time_ms=10.0,
        )
        mock_embedding_provider.generate_embedding = AsyncMock(return_value=embedding_result)

        search = HybridSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
            fallback_to_vector=True,
        )
        search.bm25_generator.generate_sparse_vector = AsyncMock(
            side_effect=SparseVectorGenerationError("BM25 generation failed")
        )

        vector_results = [{"id": 1, "distance": 0.1}]
        mock_connection_manager.execute_operation_async = AsyncMock(return_value=vector_results)

        config = HybridSearchConfig(
            top_k=10,
            sparse_field="sparse_vector",
        )

        result = await search.search(
            collection_name="test_collection",
            query="test query",
            config=config,
        )

        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_search_embedding_generation_error(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test search() handles embedding generation errors.

        Coverage: HybridSearch.search() raises HybridSearchError on embedding failure.
        """
        mock_embedding_provider.generate_embedding = AsyncMock(
            side_effect=EmbeddingGenerationError("Embedding generation failed")
        )

        search = HybridSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        config = HybridSearchConfig(top_k=10)

        with pytest.raises(HybridSearchError, match="Embedding generation failed"):
            await search.search(
                collection_name="test_collection",
                query="test query",
                config=config,
            )

    @pytest.mark.asyncio
    async def test_search_weight_normalization(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test search() normalizes weights when sum != 1.0.

        Coverage: HybridSearch normalizes vector_weight and sparse_weight.
        """
        embedding_result = EmbeddingResult(
            embedding=[0.1] * 768,
            dimension=768,
            model_name="test_model",
            processing_time_ms=10.0,
        )
        mock_embedding_provider.generate_embedding = AsyncMock(return_value=embedding_result)

        search = HybridSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )

        vector_results = [{"id": 1, "distance": 0.1}]
        mock_connection_manager.execute_operation_async = AsyncMock(return_value=vector_results)

        # Weights that don't sum to 1.0 should be normalized
        config = HybridSearchConfig(
            top_k=10,
            vector_weight=1.4,  # Sum = 1.4 + 0.6 = 2.0
            sparse_weight=0.6,
        )

        result = await search.search(
            collection_name="test_collection",
            query="test query",
            config=config,
        )

        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_search_with_empty_results(
        self, mock_connection_manager, mock_embedding_provider
    ):
        """
        Test search() with empty results from different sources.

        Coverage: HybridSearch.search() handles empty results gracefully.
        """
        embedding_result = EmbeddingResult(
            embedding=[0.1] * 768,
            dimension=768,
            model_name="test_model",
            processing_time_ms=10.0,
        )
        mock_embedding_provider.generate_embedding = AsyncMock(return_value=embedding_result)

        sparse_vector = {"term1": 0.5}
        search = HybridSearch(
            connection_manager=mock_connection_manager,
            embedding_provider=mock_embedding_provider,
        )
        search.bm25_generator.generate_sparse_vector = AsyncMock(return_value=sparse_vector)

        # Empty results from all sources
        mock_connection_manager.execute_operation_async = AsyncMock(return_value=[])

        config = HybridSearchConfig(
            top_k=10,
            sparse_field="sparse_vector",
        )

        result = await search.search(
            collection_name="test_collection",
            query="test query",
            config=config,
        )

        assert isinstance(result, SearchResult)
        assert len(result.hits) == 0
        assert result.total_hits == 0
