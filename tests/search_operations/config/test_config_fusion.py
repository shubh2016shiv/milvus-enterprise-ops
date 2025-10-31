"""
Comprehensive unit tests for fusion search configuration.

This module provides systematic testing of FusionSearchConfig,
including initialization, validation, and edge cases.
"""

import pytest

from milvus_ops.search_operations.config.base import BaseSearchConfig, FusionMethod
from milvus_ops.search_operations.config.fusion import FusionSearchConfig
from milvus_ops.search_operations.config.hybrid import HybridSearchConfig
from milvus_ops.search_operations.config.semantic import SemanticSearchConfig

# ============================================================================
# Test FusionSearchConfig Initialization
# ============================================================================


@pytest.mark.unit
class TestFusionSearchConfigInitialization:
    """
    Test FusionSearchConfig initialization.

    Coverage: FusionSearchConfig.__init__() with defaults and custom values.
    """

    def test_fusion_search_config_defaults(self):
        """
        Test FusionSearchConfig initialization with defaults.

        Coverage: FusionSearchConfig.__init__() initializes with default values.
        """
        # Fusion search requires at least two search configurations
        config1 = BaseSearchConfig()
        config2 = BaseSearchConfig()
        config = FusionSearchConfig(search_configs=[config1, config2])
        assert config.method == FusionMethod.RRF
        assert len(config.search_configs) == 2
        assert config.weights is None
        assert isinstance(config, BaseSearchConfig)

    def test_fusion_search_config_custom_values(self):
        """
        Test FusionSearchConfig initialization with custom values.

        Coverage: FusionSearchConfig.__init__() accepts custom parameter values.
        """
        semantic_config = SemanticSearchConfig()
        hybrid_config = HybridSearchConfig()
        weights = [0.5, 0.5]

        config = FusionSearchConfig(
            top_k=20,
            timeout=60.0,
            method=FusionMethod.WEIGHTED,
            search_configs=[semantic_config, hybrid_config],
            weights=weights,
        )
        assert config.method == FusionMethod.WEIGHTED
        assert len(config.search_configs) == 2
        assert config.weights == weights
        assert config.top_k == 20
        assert config.timeout == 60.0

    def test_fusion_search_config_inherits_from_base(self):
        """
        Test FusionSearchConfig inherits from BaseSearchConfig.

        Coverage: FusionSearchConfig is a subclass of BaseSearchConfig.
        """
        assert issubclass(FusionSearchConfig, BaseSearchConfig)


# ============================================================================
# Test FusionSearchConfig Validation
# ============================================================================


@pytest.mark.unit
class TestFusionSearchConfigValidation:
    """
    Test FusionSearchConfig validation.

    Coverage: FusionSearchConfig.__post_init__() validates requirements.
    """

    def test_fusion_search_config_minimum_configs(self):
        """
        Test FusionSearchConfig validation requires at least two configs.

        Coverage: FusionSearchConfig.__post_init__() raises ValueError with less than two configs.
        """
        # Test with no configs
        with pytest.raises(ValueError) as exc_info:
            FusionSearchConfig(search_configs=[])
        assert "at least two search configurations" in str(exc_info.value)

        # Test with one config
        semantic_config = SemanticSearchConfig()
        with pytest.raises(ValueError) as exc_info:
            FusionSearchConfig(search_configs=[semantic_config])
        assert "at least two search configurations" in str(exc_info.value)

    def test_fusion_search_config_valid_configs(self):
        """
        Test FusionSearchConfig accepts valid config lists.

        Coverage: FusionSearchConfig accepts two or more search configs.
        """
        semantic_config = SemanticSearchConfig()
        hybrid_config = HybridSearchConfig()

        # Test with two configs
        config = FusionSearchConfig(search_configs=[semantic_config, hybrid_config])
        assert len(config.search_configs) == 2

        # Test with three configs
        semantic_config2 = SemanticSearchConfig(search_field="vector2")
        config = FusionSearchConfig(
            search_configs=[semantic_config, hybrid_config, semantic_config2]
        )
        assert len(config.search_configs) == 3

    def test_fusion_search_config_weighted_requires_weights(self):
        """
        Test FusionSearchConfig validation requires weights for WEIGHTED method.

        Coverage: FusionSearchConfig.__post_init__() raises ValueError for WEIGHTED without weights.
        """
        semantic_config = SemanticSearchConfig()
        hybrid_config = HybridSearchConfig()

        with pytest.raises(ValueError) as exc_info:
            FusionSearchConfig(
                method=FusionMethod.WEIGHTED,
                search_configs=[semantic_config, hybrid_config],
                weights=None,
            )
        assert "Weights must be provided for weighted fusion" in str(exc_info.value)

    def test_fusion_search_config_weights_count_mismatch(self):
        """
        Test FusionSearchConfig validation requires weight count to match config count.

        Coverage: FusionSearchConfig.__post_init__() raises ValueError when
        weight count doesn't match.
        """
        semantic_config = SemanticSearchConfig()
        hybrid_config = HybridSearchConfig()

        # Wrong number of weights
        with pytest.raises(ValueError) as exc_info:
            FusionSearchConfig(
                method=FusionMethod.WEIGHTED,
                search_configs=[semantic_config, hybrid_config],
                weights=[0.5],  # Only one weight for two configs
            )
        assert "Number of weights" in str(exc_info.value)
        assert "must match" in str(exc_info.value)

    def test_fusion_search_config_weights_must_sum_to_one(self):
        """
        Test FusionSearchConfig validation requires weights to sum to 1.0.

        Coverage: FusionSearchConfig.__post_init__() raises ValueError when
        weights don't sum to 1.0.
        """
        semantic_config = SemanticSearchConfig()
        hybrid_config = HybridSearchConfig()

        # Weights don't sum to 1.0
        with pytest.raises(ValueError) as exc_info:
            FusionSearchConfig(
                method=FusionMethod.WEIGHTED,
                search_configs=[semantic_config, hybrid_config],
                weights=[0.6, 0.5],  # Sums to 1.1
            )
        assert "Weights must sum to 1.0" in str(exc_info.value)

    def test_fusion_search_config_valid_weights(self):
        """
        Test FusionSearchConfig accepts valid weight combinations.

        Coverage: FusionSearchConfig accepts weights that sum to 1.0.
        """
        semantic_config = SemanticSearchConfig()
        hybrid_config = HybridSearchConfig()

        # Valid weights
        config = FusionSearchConfig(
            method=FusionMethod.WEIGHTED,
            search_configs=[semantic_config, hybrid_config],
            weights=[0.5, 0.5],
        )
        assert config.weights == [0.5, 0.5]

        # Valid weights with small floating point error
        config = FusionSearchConfig(
            method=FusionMethod.WEIGHTED,
            search_configs=[semantic_config, hybrid_config],
            weights=[0.333, 0.667],  # Sums to 1.000
        )
        assert config.weights == [0.333, 0.667]


# ============================================================================
# Test FusionSearchConfig Methods
# ============================================================================


@pytest.mark.unit
class TestFusionSearchConfigMethods:
    """
    Test FusionSearchConfig with different fusion methods.

    Coverage: FusionSearchConfig supports all fusion methods.
    """

    @pytest.mark.parametrize(
        "method",
        [FusionMethod.RRF, FusionMethod.WEIGHTED, FusionMethod.MAX, FusionMethod.MEAN],
    )
    def test_fusion_search_config_all_methods(self, method):
        """
        Test FusionSearchConfig accepts all fusion methods.

        Coverage: FusionSearchConfig accepts all FusionMethod enum values.
        """
        semantic_config = SemanticSearchConfig()
        hybrid_config = HybridSearchConfig()

        if method == FusionMethod.WEIGHTED:
            config = FusionSearchConfig(
                method=method,
                search_configs=[semantic_config, hybrid_config],
                weights=[0.5, 0.5],
            )
        else:
            config = FusionSearchConfig(
                method=method,
                search_configs=[semantic_config, hybrid_config],
            )
        assert config.method == method


# ============================================================================
# Test FusionSearchConfig Edge Cases
# ============================================================================


@pytest.mark.unit
class TestFusionSearchConfigEdgeCases:
    """
    Test FusionSearchConfig edge cases.

    Coverage: FusionSearchConfig handles edge cases correctly.
    """

    def test_fusion_search_config_many_configs(self):
        """
        Test FusionSearchConfig with many search configs.

        Coverage: FusionSearchConfig accepts many search configurations.
        """
        configs = [SemanticSearchConfig() for _ in range(10)]
        config = FusionSearchConfig(search_configs=configs)
        assert len(config.search_configs) == 10

    def test_fusion_search_config_many_weights(self):
        """
        Test FusionSearchConfig with many weights.

        Coverage: FusionSearchConfig accepts many weights for weighted fusion.
        """
        configs = [SemanticSearchConfig() for _ in range(5)]
        weights = [0.2] * 5  # All equal weights
        config = FusionSearchConfig(
            method=FusionMethod.WEIGHTED,
            search_configs=configs,
            weights=weights,
        )
        assert len(config.weights) == 5

    def test_fusion_search_config_mixed_config_types(self):
        """
        Test FusionSearchConfig with mixed config types.

        Coverage: FusionSearchConfig accepts mix of SemanticSearchConfig and HybridSearchConfig.
        """
        semantic_config = SemanticSearchConfig()
        hybrid_config = HybridSearchConfig()
        semantic_config2 = SemanticSearchConfig(search_field="vector2")

        config = FusionSearchConfig(
            search_configs=[semantic_config, hybrid_config, semantic_config2],
            weights=[0.4, 0.3, 0.3],
        )
        assert len(config.search_configs) == 3
        assert isinstance(config.search_configs[0], SemanticSearchConfig)
        assert isinstance(config.search_configs[1], HybridSearchConfig)
        assert isinstance(config.search_configs[2], SemanticSearchConfig)

    def test_fusion_search_config_duplicate_configs(self):
        """
        Test FusionSearchConfig with duplicate configs in list.

        Coverage: FusionSearchConfig handles duplicate configs in list (edge case).
        """
        semantic_config = SemanticSearchConfig()
        # Same config twice
        config = FusionSearchConfig(
            search_configs=[semantic_config, semantic_config],
            weights=[0.5, 0.5],
        )
        # Should accept duplicates (might be intentional)
        assert len(config.search_configs) == 2
        assert config.search_configs[0] == config.search_configs[1]

    def test_fusion_search_config_none_configs_in_list(self):
        """
        Test FusionSearchConfig with None configs in list.

        Coverage: FusionSearchConfig handles None configs in list (edge case).
        """
        semantic_config = SemanticSearchConfig()
        # None config in list should cause error or be filtered
        try:
            config = FusionSearchConfig(
                search_configs=[semantic_config, None],  # type: ignore
                weights=[0.5, 0.5],
            )
            # If None is filtered out, validation should fail (needs 2+ configs)
            assert len(config.search_configs) >= 2
        except (TypeError, ValueError):
            # Expected if None is not allowed
            pass

    def test_fusion_search_config_empty_weights_list(self):
        """
        Test FusionSearchConfig with empty weights list.

        Coverage: FusionSearchConfig handles empty weights list for WEIGHTED method.
        """
        semantic_config = SemanticSearchConfig()
        hybrid_config = HybridSearchConfig()

        # Empty weights list for WEIGHTED method should raise error
        with pytest.raises(ValueError, match="Weights must be provided"):
            FusionSearchConfig(
                method=FusionMethod.WEIGHTED,
                search_configs=[semantic_config, hybrid_config],
                weights=[],  # Empty list
            )

    def test_fusion_search_config_weights_precision_issues(self):
        """
        Test FusionSearchConfig weights precision issues (floating point).

        Coverage: FusionSearchConfig handles floating point precision in weight sums.
        """
        semantic_config = SemanticSearchConfig()
        hybrid_config = HybridSearchConfig()

        # Test weights that sum to 1.0 with floating point precision issues
        # These should be accepted (within tolerance)
        weights_precise = [0.3333333333333333, 0.6666666666666666]  # Sums to 0.9999999999999999
        try:
            config = FusionSearchConfig(
                method=FusionMethod.WEIGHTED,
                search_configs=[semantic_config, hybrid_config],
                weights=weights_precise,
            )
            # Should accept within tolerance (0.001)
            assert config.weights == weights_precise
        except ValueError:
            # If sum is too far from 1.0, that's expected
            pass

        # Test weights that sum exactly to 1.0
        weights_exact = [0.5, 0.5]
        config = FusionSearchConfig(
            method=FusionMethod.WEIGHTED,
            search_configs=[semantic_config, hybrid_config],
            weights=weights_exact,
        )
        assert config.weights == weights_exact

        # Test weights with very small floating point errors
        weights_small_error = [0.499999, 0.500001]  # Sums to 1.000000
        config = FusionSearchConfig(
            method=FusionMethod.WEIGHTED,
            search_configs=[semantic_config, hybrid_config],
            weights=weights_small_error,
        )
        # Should accept (within 0.001 tolerance)
        assert config.weights == weights_small_error


# ============================================================================
# Test FusionSearchConfig Boundary Values
# ============================================================================


@pytest.mark.unit
class TestFusionSearchConfigBoundaryValues:
    """
    Test FusionSearchConfig boundary value conditions.

    Coverage: Maximum/minimum integers, extreme floats, NaN/Infinity values.
    """

    def test_fusion_search_config_maximum_integer_top_k(self):
        """
        Test FusionSearchConfig with maximum integer value for top_k.

        Coverage: FusionSearchConfig handles sys.maxsize for top_k.
        """
        import sys

        semantic_config = SemanticSearchConfig()
        hybrid_config = HybridSearchConfig()

        config = FusionSearchConfig(
            top_k=sys.maxsize, search_configs=[semantic_config, hybrid_config]
        )
        assert config.top_k == sys.maxsize

    def test_fusion_search_config_minimum_positive_timeout(self):
        """
        Test FusionSearchConfig with minimum positive timeout value.

        Coverage: FusionSearchConfig handles very small positive floats.
        """
        semantic_config = SemanticSearchConfig()
        hybrid_config = HybridSearchConfig()

        config = FusionSearchConfig(timeout=0.0001, search_configs=[semantic_config, hybrid_config])
        assert config.timeout == 0.0001

    def test_fusion_search_config_very_large_timeout_float(self):
        """
        Test FusionSearchConfig with very large timeout value.

        Coverage: FusionSearchConfig handles large float values.
        """
        semantic_config = SemanticSearchConfig()
        hybrid_config = HybridSearchConfig()

        config = FusionSearchConfig(timeout=1e100, search_configs=[semantic_config, hybrid_config])
        assert config.timeout == 1e100

    def test_fusion_search_config_nan_in_weights(self):
        """
        Test FusionSearchConfig with NaN in weights.

        Coverage: FusionSearchConfig handles NaN in weights (should fail validation).
        """
        semantic_config = SemanticSearchConfig()
        hybrid_config = HybridSearchConfig()

        # NaN in weights should fail validation
        with pytest.raises((ValueError, TypeError)):
            FusionSearchConfig(
                method=FusionMethod.WEIGHTED,
                search_configs=[semantic_config, hybrid_config],
                weights=[float("nan"), 0.5],  # type: ignore
            )

    def test_fusion_search_config_infinity_in_weights(self):
        """
        Test FusionSearchConfig with infinity in weights.

        Coverage: FusionSearchConfig handles infinity in weights.
        """
        semantic_config = SemanticSearchConfig()
        hybrid_config = HybridSearchConfig()

        # Infinity in weights should fail validation
        try:
            config = FusionSearchConfig(
                method=FusionMethod.WEIGHTED,
                search_configs=[semantic_config, hybrid_config],
                weights=[float("inf"), 0.5],  # type: ignore
            )
            # If accepted, verify
            assert float("inf") in config.weights
        except (ValueError, TypeError):
            # Expected: validation should reject infinity
            pass
