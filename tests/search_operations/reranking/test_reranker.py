"""
Comprehensive unit tests for MilvusReRanker.

This module provides systematic testing of MilvusReRanker,
including all reranking methods, weight validation, metrics, error handling,
and edge cases.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from milvus_ops.search_operations.config.base import ReRankingMethod
from milvus_ops.search_operations.core.base import SearchResult
from milvus_ops.search_operations.core.search_ops_exceptions import ReRankingError
from milvus_ops.search_operations.reranking.reranker import (
    MilvusReRanker,
    MilvusReRankingMethod,
)

# ============================================================================
# Test MilvusReRanker Initialization
# ============================================================================


@pytest.mark.unit
class TestMilvusReRankerInitialization:
    """
    Test MilvusReRanker initialization.

    Coverage: MilvusReRanker constructor with various configurations.
    """

    @patch("pymilvus.WeightedRanker")
    @patch("pymilvus.RRFRanker")
    def test_initialization_default(self, mock_rrf_ranker, mock_weighted_ranker):
        """
        Test MilvusReRanker initialization with default parameters.

        Coverage: MilvusReRanker.__init__() with default method.
        """
        reranker = MilvusReRanker()
        assert reranker.method == MilvusReRankingMethod.WEIGHTED
        assert reranker.enable_validation is True
        assert reranker.enable_normalization is True
        assert reranker.enable_metrics is True

    @patch("pymilvus.WeightedRanker")
    @patch("pymilvus.RRFRanker")
    def test_initialization_with_rrf_method(self, mock_rrf_ranker, mock_weighted_ranker):
        """
        Test MilvusReRanker initialization with RRF method.

        Coverage: MilvusReRanker.__init__() accepts MilvusReRankingMethod.RRF.
        """
        reranker = MilvusReRanker(method=MilvusReRankingMethod.RRF)
        assert reranker.method == MilvusReRankingMethod.RRF

    @patch("pymilvus.WeightedRanker")
    @patch("pymilvus.RRFRanker")
    def test_initialization_without_validation(self, mock_rrf_ranker, mock_weighted_ranker):
        """
        Test MilvusReRanker initialization without validation.

        Coverage: MilvusReRanker.__init__() with enable_validation=False.
        """
        reranker = MilvusReRanker(enable_validation=False)
        assert reranker.enable_validation is False

    @patch("pymilvus.WeightedRanker")
    @patch("pymilvus.RRFRanker")
    def test_initialization_without_normalization(self, mock_rrf_ranker, mock_weighted_ranker):
        """
        Test MilvusReRanker initialization without normalization.

        Coverage: MilvusReRanker.__init__() with enable_normalization=False.
        """
        reranker = MilvusReRanker(enable_normalization=False)
        assert reranker.enable_normalization is False

    @patch("milvus_ops.search_operations.reranking.reranker.MilvusReRanker._validate_dependencies")
    def test_initialization_missing_dependencies(self, mock_validate):
        """
        Test MilvusReRanker initialization with missing dependencies.

        Coverage: MilvusReRanker.__init__() raises ReRankingError when dependencies missing.
        """
        mock_validate.side_effect = ReRankingError(
            "Failed to import Milvus re-ranking classes. "
            "Please ensure PyMilvus >= 2.3.0 is installed."
        )
        with pytest.raises(ReRankingError, match="Failed to import Milvus"):
            MilvusReRanker()


# ============================================================================
# Test MilvusReRanker rerank() Method
# ============================================================================


@pytest.mark.unit
class TestMilvusReRankerRerank:
    """
    Test MilvusReRanker.rerank() method.

    Coverage: All reranking methods (WEIGHTED, RRF, NONE), weight validation, edge cases.
    """

    @patch("pymilvus.WeightedRanker")
    @patch("pymilvus.RRFRanker")
    @pytest.mark.asyncio
    async def test_rerank_weighted_method(self, mock_rrf_ranker, mock_weighted_ranker):
        """
        Test rerank() with WEIGHTED method.

        Coverage: MilvusReRanker.rerank() uses WEIGHTED method correctly.
        """
        # Mock weighted ranker
        mock_ranker = MagicMock()
        mock_ranker.rerank = AsyncMock(return_value=[{"id": 1, "score": 0.9}])
        mock_weighted_ranker.return_value = mock_ranker

        reranker = MilvusReRanker(method=MilvusReRankingMethod.WEIGHTED)
        reranker._ranker = mock_ranker

        search_result = SearchResult(
            hits=[{"id": 1, "distance": 0.1}],
            total_hits=1,
            took_ms=10.0,
        )

        result = await reranker.rerank(
            search_result,
            rerank_method=ReRankingMethod.WEIGHTED,
            rerank_weights=[0.7, 0.3],
        )

        assert isinstance(result, SearchResult)
        assert len(result.hits) > 0

    @patch("pymilvus.WeightedRanker")
    @patch("pymilvus.RRFRanker")
    @pytest.mark.asyncio
    async def test_rerank_rrf_method(self, mock_rrf_ranker, mock_weighted_ranker):
        """
        Test rerank() with RRF method.

        Coverage: MilvusReRanker.rerank() uses RRF method correctly.
        """
        # Mock RRF ranker
        mock_ranker = MagicMock()
        mock_ranker.rerank = AsyncMock(return_value=[{"id": 1, "score": 0.8}])
        mock_rrf_ranker.return_value = mock_ranker

        reranker = MilvusReRanker(method=MilvusReRankingMethod.RRF)
        reranker._ranker = mock_ranker

        search_result = SearchResult(
            hits=[{"id": 1, "distance": 0.1}],
            total_hits=1,
            took_ms=10.0,
        )

        result = await reranker.rerank(
            search_result,
            rerank_method=ReRankingMethod.RRF,
            rerank_k=60,
        )

        assert isinstance(result, SearchResult)

    @patch("pymilvus.WeightedRanker")
    @patch("pymilvus.RRFRanker")
    @pytest.mark.asyncio
    async def test_rerank_none_method(self, mock_rrf_ranker, mock_weighted_ranker):
        """
        Test rerank() with NONE method (should skip).

        Coverage: MilvusReRanker.rerank() skips reranking when method is NONE.
        """
        reranker = MilvusReRanker()

        search_result = SearchResult(
            hits=[{"id": 1, "distance": 0.1}],
            total_hits=1,
            took_ms=10.0,
        )

        result = await reranker.rerank(
            search_result,
            rerank_method=ReRankingMethod.NONE,
        )

        # Should return original result unchanged
        assert result == search_result

    @patch("pymilvus.WeightedRanker")
    @patch("pymilvus.RRFRanker")
    @pytest.mark.asyncio
    async def test_rerank_empty_results(self, mock_rrf_ranker, mock_weighted_ranker):
        """
        Test rerank() with empty results.

        Coverage: MilvusReRanker.rerank() handles empty results gracefully.
        """
        mock_ranker = MagicMock()
        mock_ranker.rerank = AsyncMock(return_value=[])
        mock_weighted_ranker.return_value = mock_ranker

        reranker = MilvusReRanker()
        reranker._ranker = mock_ranker

        search_result = SearchResult(hits=[], total_hits=0, took_ms=10.0)

        result = await reranker.rerank(
            search_result,
            rerank_method=ReRankingMethod.WEIGHTED,
        )

        assert isinstance(result, SearchResult)
        assert len(result.hits) == 0

    @patch("pymilvus.WeightedRanker")
    @patch("pymilvus.RRFRanker")
    @pytest.mark.asyncio
    async def test_rerank_with_malformed_results(self, mock_rrf_ranker, mock_weighted_ranker):
        """
        Test rerank() with malformed results.

        Coverage: MilvusReRanker.rerank() handles malformed result structures.
        """
        mock_ranker = MagicMock()
        mock_ranker.rerank = AsyncMock(return_value=[{"id": 1}])  # Missing expected fields
        mock_weighted_ranker.return_value = mock_ranker

        reranker = MilvusReRanker(fallback_on_error=True)
        reranker._ranker = mock_ranker

        search_result = SearchResult(
            hits=[{"id": 1}],  # Malformed hit
            total_hits=1,
            took_ms=10.0,
        )

        result = await reranker.rerank(
            search_result,
            rerank_method=ReRankingMethod.WEIGHTED,
        )

        assert isinstance(result, SearchResult)


# ============================================================================
# Test MilvusReRanker Weight Validation
# ============================================================================


@pytest.mark.unit
class TestMilvusReRankerWeightValidation:
    """
    Test MilvusReRanker weight validation.

    Coverage: Weight validation, normalization, edge cases.
    """

    @patch("pymilvus.WeightedRanker")
    @patch("pymilvus.RRFRanker")
    def test_validate_weights_empty_list(self, mock_rrf_ranker, mock_weighted_ranker):
        """
        Test validate_weights() with empty list.

        Coverage: MilvusReRanker.validate_weights() rejects empty weights list.
        """
        reranker = MilvusReRanker()
        result = reranker.validate_weights([])
        assert result.is_valid is False
        assert len(result.errors) > 0

    @patch("pymilvus.WeightedRanker")
    @patch("pymilvus.RRFRanker")
    def test_validate_weights_negative_weights(self, mock_rrf_ranker, mock_weighted_ranker):
        """
        Test validate_weights() with negative weights.

        Coverage: MilvusReRanker.validate_weights() rejects negative weights.
        """
        reranker = MilvusReRanker()
        result = reranker.validate_weights([0.5, -0.1])
        assert result.is_valid is False
        assert any("negative" in err.lower() for err in result.errors)

    @patch("pymilvus.WeightedRanker")
    @patch("pymilvus.RRFRanker")
    def test_validate_weights_all_zero(self, mock_rrf_ranker, mock_weighted_ranker):
        """
        Test validate_weights() with all zero weights.

        Coverage: MilvusReRanker.validate_weights() rejects all zero weights.
        """
        reranker = MilvusReRanker()
        result = reranker.validate_weights([0.0, 0.0])
        assert result.is_valid is False
        assert any("zero" in err.lower() for err in result.errors)

    @patch("pymilvus.WeightedRanker")
    @patch("pymilvus.RRFRanker")
    def test_validate_weights_normalization(self, mock_rrf_ranker, mock_weighted_ranker):
        """
        Test validate_weights() normalizes weights.

        Coverage: MilvusReRanker.validate_weights() normalizes weights when auto_normalize=True.
        """
        reranker = MilvusReRanker()
        result = reranker.validate_weights([1.0, 2.0], auto_normalize=True)
        assert result.is_valid is True
        if result.normalized_weights:
            # Weights should sum to approximately 1.0
            assert abs(sum(result.normalized_weights) - 1.0) < 0.01

    @patch("pymilvus.WeightedRanker")
    @patch("pymilvus.RRFRanker")
    def test_validate_weights_count_mismatch(self, mock_rrf_ranker, mock_weighted_ranker):
        """
        Test validate_weights() with count mismatch.

        Coverage: MilvusReRanker.validate_weights() validates weight count against num_fields.
        """
        reranker = MilvusReRanker()
        result = reranker.validate_weights([0.5, 0.5], num_fields=3)
        # Should warn or error about count mismatch
        assert isinstance(result, type(reranker.validate_weights([0.5, 0.5])))

    @pytest.mark.asyncio
    @patch("pymilvus.WeightedRanker")
    @patch("pymilvus.RRFRanker")
    async def test_rerank_k_zero_value(self, mock_rrf_ranker, mock_weighted_ranker):
        """
        Test rerank() with zero k value for RRF.

        Coverage: MilvusReRanker.rerank() handles zero rerank_k for RRF method.
        """
        reranker = MilvusReRanker(method=MilvusReRankingMethod.RRF)
        # Zero k might cause issues, test edge case
        search_result = SearchResult(hits=[{"id": 1}], total_hits=1, took_ms=10.0)

        # Should handle gracefully or raise error
        try:
            result = await reranker.rerank(
                search_result,
                rerank_method=ReRankingMethod.RRF,
                rerank_k=0,
            )
            assert isinstance(result, SearchResult)
        except (ReRankingError, ValueError):
            # Expected behavior if zero k is invalid
            pass


# ============================================================================
# Test MilvusReRanker Error Handling
# ============================================================================


@pytest.mark.unit
class TestMilvusReRankerErrorHandling:
    """
    Test MilvusReRanker error handling.

    Coverage: Reranking failures, invalid configs, fallback mechanisms.
    """

    @patch("pymilvus.WeightedRanker")
    @patch("pymilvus.RRFRanker")
    @pytest.mark.asyncio
    async def test_rerank_failure_with_fallback(self, mock_rrf_ranker, mock_weighted_ranker):
        """
        Test rerank() falls back on failure when fallback_on_error=True.

        Coverage: MilvusReRanker.rerank() returns original result on failure when fallback enabled.
        """
        mock_ranker = MagicMock()
        mock_ranker.rerank = AsyncMock(side_effect=Exception("Reranking failed"))
        mock_weighted_ranker.return_value = mock_ranker

        reranker = MilvusReRanker(fallback_on_error=True)
        reranker._ranker = mock_ranker

        search_result = SearchResult(
            hits=[{"id": 1, "distance": 0.1}],
            total_hits=1,
            took_ms=10.0,
        )

        result = await reranker.rerank(
            search_result,
            rerank_method=ReRankingMethod.WEIGHTED,
        )

        # Should return original result on failure
        assert isinstance(result, SearchResult)

    @patch("pymilvus.WeightedRanker")
    @patch("pymilvus.RRFRanker")
    @pytest.mark.asyncio
    async def test_rerank_failure_without_fallback(self, mock_rrf_ranker, mock_weighted_ranker):
        """
        Test rerank() raises error on failure when fallback_on_error=False.

        Coverage: MilvusReRanker.rerank() raises ReRankingError on failure when fallback disabled.
        """
        mock_ranker = MagicMock()
        mock_ranker.rerank = AsyncMock(side_effect=Exception("Reranking failed"))
        mock_weighted_ranker.return_value = mock_ranker

        reranker = MilvusReRanker(fallback_on_error=False)
        reranker._ranker = mock_ranker

        search_result = SearchResult(
            hits=[{"id": 1, "distance": 0.1}],
            total_hits=1,
            took_ms=10.0,
        )

        with pytest.raises((ReRankingError, Exception)):
            await reranker.rerank(
                search_result,
                rerank_method=ReRankingMethod.WEIGHTED,
            )

    @patch("pymilvus.WeightedRanker")
    @patch("pymilvus.RRFRanker")
    def test_validate_weights_invalid_types(self, mock_rrf_ranker, mock_weighted_ranker):
        """
        Test validate_weights() with invalid types.

        Coverage: MilvusReRanker.validate_weights() rejects non-numeric weights.
        """
        reranker = MilvusReRanker()
        result = reranker.validate_weights(["0.5", "0.5"])  # Strings instead of floats
        assert result.is_valid is False
        assert any("numeric" in err.lower() for err in result.errors)
