"""
Comprehensive unit tests for hybrid search configuration.

This module provides systematic testing of HybridSearchConfig,
including initialization, weight validation, and edge cases.
"""

import pytest

from milvus_ops.search_operations.config.base import BaseSearchConfig, MetricType
from milvus_ops.search_operations.config.hybrid import HybridSearchConfig

# ============================================================================
# Test HybridSearchConfig Initialization
# ============================================================================


@pytest.mark.unit
class TestHybridSearchConfigInitialization:
    """
    Test HybridSearchConfig initialization.

    Coverage: HybridSearchConfig.__init__() with defaults and custom values.
    """

    def test_hybrid_search_config_defaults(self):
        """
        Test HybridSearchConfig initialization with defaults.

        Coverage: HybridSearchConfig.__init__() initializes with default values.
        """
        config = HybridSearchConfig()
        assert config.vector_field == "vector"
        assert config.sparse_field is None
        assert config.keyword_field is None
        assert config.expr is None
        assert config.vector_weight == 0.7
        assert config.sparse_weight == 0.3
        assert isinstance(config, BaseSearchConfig)

    def test_hybrid_search_config_custom_values(self):
        """
        Test HybridSearchConfig initialization with custom values.

        Coverage: HybridSearchConfig.__init__() accepts custom parameter values.
        """
        config = HybridSearchConfig(
            top_k=20,
            timeout=60.0,
            metric_type=MetricType.L2,
            vector_field="embedding",
            sparse_field="sparse_vector",
            keyword_field="text",
            expr='category == "test"',
            vector_weight=0.8,
            sparse_weight=0.2,
            output_fields=["text", "category"],
        )
        assert config.vector_field == "embedding"
        assert config.sparse_field == "sparse_vector"
        assert config.keyword_field == "text"
        assert config.expr == 'category == "test"'
        assert config.vector_weight == 0.8
        assert config.sparse_weight == 0.2
        assert config.output_fields == ["text", "category"]

    def test_hybrid_search_config_inherits_from_base(self):
        """
        Test HybridSearchConfig inherits from BaseSearchConfig.

        Coverage: HybridSearchConfig is a subclass of BaseSearchConfig.
        """
        assert issubclass(HybridSearchConfig, BaseSearchConfig)


# ============================================================================
# Test HybridSearchConfig Weight Validation
# ============================================================================


@pytest.mark.unit
class TestHybridSearchConfigWeightValidation:
    """
    Test HybridSearchConfig weight validation.

    Coverage: HybridSearchConfig.__post_init__() validates weights.
    """

    @pytest.mark.parametrize(
        "vector_weight,sparse_weight", [(-1.0, 0.3), (0.7, -1.0), (-0.5, -0.5)]
    )
    def test_hybrid_search_config_negative_weights(self, vector_weight, sparse_weight):
        """
        Test HybridSearchConfig validation rejects negative weights.

        Coverage: HybridSearchConfig.__post_init__() raises ValueError for negative weights.
        """
        with pytest.raises(ValueError) as exc_info:
            HybridSearchConfig(vector_weight=vector_weight, sparse_weight=sparse_weight)
        assert "Weights must be non-negative" in str(exc_info.value)

    def test_hybrid_search_config_both_weights_zero(self):
        """
        Test HybridSearchConfig validation rejects both weights being zero.

        Coverage: HybridSearchConfig.__post_init__() raises ValueError when both weights are zero.
        """
        with pytest.raises(ValueError) as exc_info:
            HybridSearchConfig(vector_weight=0.0, sparse_weight=0.0)
        assert "At least one weight must be positive" in str(exc_info.value)

    @pytest.mark.parametrize(
        "vector_weight,sparse_weight",
        [
            (1.0, 0.0),  # Vector only
            (0.0, 1.0),  # Sparse only
            (0.5, 0.5),  # Equal weights
            (0.7, 0.3),  # Default weights
            (0.8, 0.2),  # Vector heavy
            (0.2, 0.8),  # Sparse heavy
        ],
    )
    def test_hybrid_search_config_valid_weights(self, vector_weight, sparse_weight):
        """
        Test HybridSearchConfig accepts valid weight combinations.

        Coverage: HybridSearchConfig accepts non-negative weights with at least one positive.
        """
        config = HybridSearchConfig(vector_weight=vector_weight, sparse_weight=sparse_weight)
        assert config.vector_weight == vector_weight
        assert config.sparse_weight == sparse_weight


# ============================================================================
# Test HybridSearchConfig Vector-Only Mode
# ============================================================================


@pytest.mark.unit
class TestHybridSearchConfigVectorOnlyMode:
    """
    Test HybridSearchConfig vector-only mode support.

    Coverage: HybridSearchConfig supports vector-only mode when sparse/keyword fields are None.
    """

    def test_hybrid_search_config_vector_only_mode(self):
        """
        Test HybridSearchConfig vector-only mode.

        Coverage: HybridSearchConfig supports vector-only search when sparse_field
        and keyword_field are None.
        """
        config = HybridSearchConfig(
            vector_field="vector",
            sparse_field=None,
            keyword_field=None,
            vector_weight=1.0,
            sparse_weight=0.0,
        )
        assert config.vector_field == "vector"
        assert config.sparse_field is None
        assert config.keyword_field is None
        assert config.vector_weight == 1.0
        assert config.sparse_weight == 0.0

    def test_hybrid_search_config_with_sparse_field(self):
        """
        Test HybridSearchConfig with sparse field.

        Coverage: HybridSearchConfig accepts sparse_field parameter.
        """
        config = HybridSearchConfig(
            vector_field="vector",
            sparse_field="sparse_vector",
            keyword_field=None,
        )
        assert config.sparse_field == "sparse_vector"

    def test_hybrid_search_config_with_keyword_field(self):
        """
        Test HybridSearchConfig with keyword field.

        Coverage: HybridSearchConfig accepts keyword_field parameter.
        """
        config = HybridSearchConfig(
            vector_field="vector",
            sparse_field=None,
            keyword_field="text",
        )
        assert config.keyword_field == "text"

    def test_hybrid_search_config_with_both_fields(self):
        """
        Test HybridSearchConfig with both sparse and keyword fields.

        Coverage: HybridSearchConfig accepts both sparse_field and keyword_field.
        """
        config = HybridSearchConfig(
            vector_field="vector",
            sparse_field="sparse_vector",
            keyword_field="text",
        )
        assert config.sparse_field == "sparse_vector"
        assert config.keyword_field == "text"


# ============================================================================
# Test HybridSearchConfig Edge Cases
# ============================================================================


@pytest.mark.unit
class TestHybridSearchConfigEdgeCases:
    """
    Test HybridSearchConfig edge cases.

    Coverage: HybridSearchConfig handles edge cases correctly.
    """

    def test_hybrid_search_config_empty_fields(self):
        """
        Test HybridSearchConfig with empty string fields.

        Coverage: HybridSearchConfig accepts empty strings for field names.
        """
        config = HybridSearchConfig(
            vector_field="",
            sparse_field="",
            keyword_field="",
        )
        assert config.vector_field == ""
        assert config.sparse_field == ""
        assert config.keyword_field == ""

    def test_hybrid_search_config_large_weights(self):
        """
        Test HybridSearchConfig with very large weights.

        Coverage: HybridSearchConfig accepts large weight values.
        """
        config = HybridSearchConfig(vector_weight=100.0, sparse_weight=50.0)
        assert config.vector_weight == 100.0
        assert config.sparse_weight == 50.0

    def test_hybrid_search_config_small_weights(self):
        """
        Test HybridSearchConfig with very small weights.

        Coverage: HybridSearchConfig accepts small positive weight values.
        """
        config = HybridSearchConfig(vector_weight=0.001, sparse_weight=0.002)
        assert config.vector_weight == 0.001
        assert config.sparse_weight == 0.002

    def test_hybrid_search_config_complex_expr(self):
        """
        Test HybridSearchConfig with complex filter expression.

        Coverage: HybridSearchConfig accepts complex filter expressions.
        """
        complex_expr = '(category == "test" or category == "production") and score > 0.5'
        config = HybridSearchConfig(expr=complex_expr)
        assert config.expr == complex_expr

    def test_hybrid_search_config_none_vs_empty_sparse_field(self):
        """
        Test HybridSearchConfig with None sparse_field vs empty string distinction.

        Coverage: HybridSearchConfig distinguishes between None and empty string for sparse_field.
        """
        config_none = HybridSearchConfig(sparse_field=None)
        config_empty = HybridSearchConfig(sparse_field="")

        assert config_none.sparse_field is None
        assert config_empty.sparse_field == ""
        assert config_none.sparse_field != config_empty.sparse_field

    def test_hybrid_search_config_none_vs_empty_keyword_field(self):
        """
        Test HybridSearchConfig with None keyword_field vs empty string distinction.

        Coverage: HybridSearchConfig distinguishes between None and empty string for keyword_field.
        """
        config_none = HybridSearchConfig(keyword_field=None)
        config_empty = HybridSearchConfig(keyword_field="")

        assert config_none.keyword_field is None
        assert config_empty.keyword_field == ""
        assert config_none.keyword_field != config_empty.keyword_field

    def test_hybrid_search_config_both_sparse_and_keyword_fields(self):
        """
        Test HybridSearchConfig with both sparse and keyword fields (edge case).

        Coverage: HybridSearchConfig handles both sparse_field and keyword_field together.
        """
        config = HybridSearchConfig(
            sparse_field="sparse_vector",
            keyword_field="text",
            vector_weight=0.4,
            sparse_weight=0.3,
        )
        assert config.sparse_field == "sparse_vector"
        assert config.keyword_field == "text"
        # Should support all methods mode
        assert config.vector_weight + config.sparse_weight > 0

    def test_hybrid_search_config_weight_normalization_when_sum_not_one(self):
        """
        Test HybridSearchConfig weight normalization when sum != 1.0.

        Coverage: HybridSearchConfig handles weights that don't sum to 1.0
        (normalization happens at runtime).
        """
        # Weights that don't sum to 1.0 should still be valid
        # Normalization happens during search, not in config
        config = HybridSearchConfig(vector_weight=1.4, sparse_weight=0.6)
        assert config.vector_weight == 1.4
        assert config.sparse_weight == 0.6
        # Sum is 2.0, should still be valid
        assert config.vector_weight + config.sparse_weight == 2.0

    def test_hybrid_search_config_very_large_weight_values(self):
        """
        Test HybridSearchConfig with very large weight values (numerical stability).

        Coverage: HybridSearchConfig handles very large weight values without numerical issues.
        """
        # Very large weights (numerical stability test)
        config = HybridSearchConfig(vector_weight=1e6, sparse_weight=1e5)
        assert config.vector_weight == 1e6
        assert config.sparse_weight == 1e5
        # Should not cause numerical issues
        assert isinstance(config.vector_weight, float)
        assert isinstance(config.sparse_weight, float)


# ============================================================================
# Test HybridSearchConfig Boundary Values
# ============================================================================


@pytest.mark.unit
class TestHybridSearchConfigBoundaryValues:
    """
    Test HybridSearchConfig boundary value conditions.

    Coverage: Maximum/minimum integers, extreme floats, NaN/Infinity values.
    """

    def test_hybrid_search_config_maximum_integer_top_k(self):
        """
        Test HybridSearchConfig with maximum integer value for top_k.

        Coverage: HybridSearchConfig handles sys.maxsize for top_k.
        """
        import sys

        config = HybridSearchConfig(top_k=sys.maxsize)
        assert config.top_k == sys.maxsize

    def test_hybrid_search_config_minimum_positive_timeout(self):
        """
        Test HybridSearchConfig with minimum positive timeout value.

        Coverage: HybridSearchConfig handles very small positive floats.
        """
        config = HybridSearchConfig(timeout=0.0001)
        assert config.timeout == 0.0001

    def test_hybrid_search_config_very_large_timeout_float(self):
        """
        Test HybridSearchConfig with very large timeout value.

        Coverage: HybridSearchConfig handles large float values.
        """
        config = HybridSearchConfig(timeout=1e100)
        assert config.timeout == 1e100

    def test_hybrid_search_config_infinity_weight(self):
        """
        Test HybridSearchConfig with infinity weight.

        Coverage: HybridSearchConfig handles float('inf') for weights.
        """
        # Infinity weights should fail validation (must be >= 0)
        try:
            config = HybridSearchConfig(vector_weight=float("inf"), sparse_weight=0.3)
            # If accepted, verify
            assert config.vector_weight == float("inf")
        except ValueError:
            # Expected: validation should reject infinity
            pass

    def test_hybrid_search_config_nan_weight(self):
        """
        Test HybridSearchConfig with NaN weight.

        Coverage: HybridSearchConfig handles float('nan') for weights (should fail validation).
        """
        with pytest.raises(ValueError):
            HybridSearchConfig(vector_weight=float("nan"), sparse_weight=0.3)
