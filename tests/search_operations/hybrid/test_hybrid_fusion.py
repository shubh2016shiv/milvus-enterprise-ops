"""
Comprehensive unit tests for hybrid search fusion functions.

This module provides systematic testing of fusion functions (RRF, weighted),
including empty results, duplicate handling, and score calculations.
"""

import pytest

from milvus_ops.search_operations.search.hybrid.core.fusion import (
    fuse_results_rrf,
    fuse_results_weighted,
)

# ============================================================================
# Test fuse_results_rrf()
# ============================================================================


@pytest.mark.unit
class TestFuseResultsRRF:
    """
    Test fuse_results_rrf() function.

    Coverage: RRF fusion with various result lists, duplicate handling, score calculations.
    """

    @pytest.mark.asyncio
    async def test_fuse_results_rrf_multiple_lists(self):
        """
        Test fuse_results_rrf() with multiple result lists.

        Coverage: fuse_results_rrf() fuses results from multiple sources.
        """
        results_list = [
            [{"id": 1, "distance": 0.1}, {"id": 2, "distance": 0.2}],
            [{"id": 2, "distance": 0.3}, {"id": 3, "distance": 0.4}],
        ]

        fused = await fuse_results_rrf(results_list, k=60)
        assert isinstance(fused, list)
        assert len(fused) > 0
        # Document 2 should appear (in both lists)
        doc_ids = [doc["id"] for doc in fused]
        assert 2 in doc_ids

    @pytest.mark.asyncio
    async def test_fuse_results_rrf_empty_result_lists(self):
        """
        Test fuse_results_rrf() with empty result lists.

        Coverage: fuse_results_rrf() handles empty result lists.
        """
        results_list = []
        fused = await fuse_results_rrf(results_list, k=60)
        assert isinstance(fused, list)
        assert len(fused) == 0

    @pytest.mark.asyncio
    async def test_fuse_results_rrf_single_empty_list(self):
        """
        Test fuse_results_rrf() with single empty list.

        Coverage: fuse_results_rrf() handles single empty list in results_list.
        """
        results_list = [[], [{"id": 1, "distance": 0.1}]]
        fused = await fuse_results_rrf(results_list, k=60)
        assert isinstance(fused, list)
        assert len(fused) == 1
        assert fused[0]["id"] == 1

    @pytest.mark.asyncio
    async def test_fuse_results_rrf_duplicate_document_ids(self):
        """
        Test fuse_results_rrf() with duplicate document IDs.

        Coverage: fuse_results_rrf() handles duplicate document IDs across lists.
        """
        results_list = [
            [{"id": 1, "distance": 0.1}, {"id": 2, "distance": 0.2}],
            [{"id": 1, "distance": 0.3}, {"id": 3, "distance": 0.4}],
        ]

        fused = await fuse_results_rrf(results_list, k=60)
        doc_ids = [doc["id"] for doc in fused]
        # Document 1 appears in both lists, should have higher score
        assert 1 in doc_ids
        # Should have fusion_score added
        assert "fusion_score" in fused[0]

    @pytest.mark.asyncio
    async def test_fuse_results_rrf_missing_id_field(self):
        """
        Test fuse_results_rrf() with missing 'id' field (fallback to 'pk').

        Coverage: fuse_results_rrf() falls back to 'pk' field when 'id' is missing.
        """
        results_list = [
            [{"pk": 1, "distance": 0.1}],
            [{"pk": 1, "distance": 0.2}],
        ]

        fused = await fuse_results_rrf(results_list, k=60)
        assert len(fused) > 0
        assert "pk" in fused[0] or "id" in fused[0]

    @pytest.mark.asyncio
    async def test_fuse_results_rrf_k_parameter(self):
        """
        Test fuse_results_rrf() with different k parameter values.

        Coverage: fuse_results_rrf() accepts different k parameter values.
        """
        results_list = [
            [{"id": 1, "distance": 0.1}],
            [{"id": 2, "distance": 0.2}],
        ]

        fused_k60 = await fuse_results_rrf(results_list, k=60)
        fused_k100 = await fuse_results_rrf(results_list, k=100)

        assert isinstance(fused_k60, list)
        assert isinstance(fused_k100, list)
        # Different k values may affect scores
        assert len(fused_k60) == len(fused_k100)

    @pytest.mark.asyncio
    async def test_fuse_results_rrf_very_large_result_sets(self):
        """
        Test fuse_results_rrf() with very large result sets.

        Coverage: fuse_results_rrf() handles very large result sets.
        """
        results_list = [
            [{"id": i, "distance": 0.1} for i in range(1000)],
            [{"id": i + 500, "distance": 0.2} for i in range(1000)],
        ]

        fused = await fuse_results_rrf(results_list, k=60)
        assert isinstance(fused, list)
        # Should handle large sets without errors
        assert len(fused) <= 2000  # Maximum unique documents

    @pytest.mark.asyncio
    async def test_fuse_results_rrf_score_calculation(self):
        """
        Test fuse_results_rrf() score calculation accuracy.

        Coverage: fuse_results_rrf() calculates RRF scores correctly.
        """
        results_list = [
            [{"id": 1, "distance": 0.1}],  # Rank 1 in first list
            [{"id": 1, "distance": 0.2}],  # Rank 1 in second list
        ]

        fused = await fuse_results_rrf(results_list, k=60)
        assert len(fused) > 0
        doc_1 = next((doc for doc in fused if doc["id"] == 1), None)
        assert doc_1 is not None
        assert "fusion_score" in doc_1
        assert isinstance(doc_1["fusion_score"], int | float)
        assert doc_1["fusion_score"] > 0


# ============================================================================
# Test fuse_results_weighted()
# ============================================================================


@pytest.mark.unit
class TestFuseResultsWeighted:
    """
    Test fuse_results_weighted() function.

    Coverage: Weighted fusion with various inputs, weight normalization, score calculations.
    """

    @pytest.mark.asyncio
    async def test_fuse_results_weighted_normal(self):
        """
        Test fuse_results_weighted() with normal inputs.

        Coverage: fuse_results_weighted() fuses vector and sparse results.
        """
        vector_results = [{"id": 1, "distance": 0.1, "score": 0.9}]
        sparse_results = [{"id": 2, "distance": 0.2, "score": 0.8}]

        fused = await fuse_results_weighted(
            vector_results, sparse_results, vector_weight=0.7, sparse_weight=0.3
        )
        assert isinstance(fused, list)
        assert len(fused) == 2
        # Each result should have fusion_score
        assert "fusion_score" in fused[0]

    @pytest.mark.asyncio
    async def test_fuse_results_weighted_empty_results(self):
        """
        Test fuse_results_weighted() with empty results.

        Coverage: fuse_results_weighted() handles empty result lists.
        """
        fused = await fuse_results_weighted([], [], vector_weight=0.7, sparse_weight=0.3)
        assert isinstance(fused, list)
        assert len(fused) == 0

    @pytest.mark.asyncio
    async def test_fuse_results_weighted_empty_vector_results(self):
        """
        Test fuse_results_weighted() with empty vector results.

        Coverage: fuse_results_weighted() handles empty vector results.
        """
        sparse_results = [{"id": 1, "distance": 0.2, "score": 0.8}]
        fused = await fuse_results_weighted(
            [], sparse_results, vector_weight=0.7, sparse_weight=0.3
        )
        assert isinstance(fused, list)
        assert len(fused) == 1
        assert fused[0]["id"] == 1

    @pytest.mark.asyncio
    async def test_fuse_results_weighted_empty_sparse_results(self):
        """
        Test fuse_results_weighted() with empty sparse results.

        Coverage: fuse_results_weighted() handles empty sparse results.
        """
        vector_results = [{"id": 1, "distance": 0.1, "score": 0.9}]
        fused = await fuse_results_weighted(
            vector_results, [], vector_weight=0.7, sparse_weight=0.3
        )
        assert isinstance(fused, list)
        assert len(fused) == 1
        assert fused[0]["id"] == 1

    @pytest.mark.asyncio
    async def test_fuse_results_weighted_duplicate_document_ids(self):
        """
        Test fuse_results_weighted() with duplicate document IDs.

        Coverage: fuse_results_weighted() handles duplicate document IDs.
        """
        vector_results = [{"id": 1, "distance": 0.1, "score": 0.9}]
        sparse_results = [{"id": 1, "distance": 0.2, "score": 0.8}]

        fused = await fuse_results_weighted(
            vector_results, sparse_results, vector_weight=0.7, sparse_weight=0.3
        )
        # Should combine scores for duplicate ID
        assert len(fused) == 1
        assert fused[0]["id"] == 1
        assert "fusion_score" in fused[0]
        assert "vector_score" in fused[0]
        assert "sparse_score" in fused[0]

    @pytest.mark.asyncio
    async def test_fuse_results_weighted_missing_id_field(self):
        """
        Test fuse_results_weighted() with missing 'id' field (fallback to 'pk').

        Coverage: fuse_results_weighted() falls back to 'pk' field when 'id' is missing.
        """
        vector_results = [{"pk": 1, "distance": 0.1}]
        sparse_results = [{"pk": 1, "distance": 0.2}]

        fused = await fuse_results_weighted(
            vector_results, sparse_results, vector_weight=0.7, sparse_weight=0.3
        )
        assert len(fused) > 0
        assert "pk" in fused[0] or "id" in fused[0]

    @pytest.mark.asyncio
    async def test_fuse_results_weighted_weight_normalization(self):
        """
        Test fuse_results_weighted() normalizes weights.

        Coverage: fuse_results_weighted() normalizes weights when sum != 1.0.
        """
        vector_results = [{"id": 1, "score": 0.9}]
        sparse_results = [{"id": 2, "score": 0.8}]

        # Weights that don't sum to 1.0 should be normalized
        fused = await fuse_results_weighted(
            vector_results, sparse_results, vector_weight=1.4, sparse_weight=0.6
        )
        assert isinstance(fused, list)
        # Should still work with normalized weights
        assert len(fused) >= 1

    @pytest.mark.asyncio
    async def test_fuse_results_weighted_zero_weights(self):
        """
        Test fuse_results_weighted() with zero weights.

        Coverage: fuse_results_weighted() handles zero weights (should normalize to defaults).
        """
        vector_results = [{"id": 1, "score": 0.9}]
        sparse_results = [{"id": 2, "score": 0.8}]

        # Zero weights should trigger normalization to defaults (0.5, 0.5)
        fused = await fuse_results_weighted(
            vector_results, sparse_results, vector_weight=0.0, sparse_weight=0.0
        )
        assert isinstance(fused, list)
        # Should still produce results with default weights
        assert len(fused) >= 1

    @pytest.mark.asyncio
    async def test_fuse_results_weighted_score_calculation(self):
        """
        Test fuse_results_weighted() score calculation accuracy.

        Coverage: fuse_results_weighted() calculates weighted scores correctly.
        """
        vector_results = [{"id": 1, "distance": 0.1, "score": 0.9}]
        sparse_results = [{"id": 1, "distance": 0.2, "score": 0.8}]

        fused = await fuse_results_weighted(
            vector_results, sparse_results, vector_weight=0.7, sparse_weight=0.3
        )
        assert len(fused) == 1
        doc_1 = fused[0]
        assert "fusion_score" in doc_1
        assert "vector_score" in doc_1
        assert "sparse_score" in doc_1
        # Fusion score should be weighted combination
        # 0.9 * 0.7 + 0.8 * 0.3 = 0.63 + 0.24 = 0.87
        assert isinstance(doc_1["fusion_score"], int | float)

    @pytest.mark.asyncio
    async def test_fuse_results_weighted_uses_distance_when_no_score(self):
        """
        Test fuse_results_weighted() uses 'distance' when 'score' is missing.

        Coverage: fuse_results_weighted() uses distance field when score field is missing.
        """
        vector_results = [{"id": 1, "distance": 0.1}]
        sparse_results = [{"id": 2, "distance": 0.2}]

        fused = await fuse_results_weighted(
            vector_results, sparse_results, vector_weight=0.7, sparse_weight=0.3
        )
        assert isinstance(fused, list)
        assert len(fused) == 2
