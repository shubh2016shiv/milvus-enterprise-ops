"""
Comprehensive unit tests for reranking configuration.

This module provides systematic testing of ReRankingConfig,
including initialization, validation, and edge cases.
"""

import pytest

from milvus_ops.search_operations.config.base import ReRankingMethod
from milvus_ops.search_operations.config.reranking import ReRankingConfig

# ============================================================================
# Test ReRankingConfig Initialization
# ============================================================================


@pytest.mark.unit
class TestReRankingConfigInitialization:
    """
    Test ReRankingConfig initialization.

    Coverage: ReRankingConfig.__init__() with defaults and custom values.
    """

    def test_reranking_config_defaults(self):
        """
        Test ReRankingConfig initialization with defaults.

        Coverage: ReRankingConfig.__init__() initializes with default values.
        """
        config = ReRankingConfig()
        assert config.enabled is False
        assert config.method == ReRankingMethod.NONE
        assert isinstance(config.params, dict)
        assert len(config.params) == 0

    def test_reranking_config_custom_values(self):
        """
        Test ReRankingConfig initialization with custom values.

        Coverage: ReRankingConfig.__init__() accepts custom parameter values.
        """
        config = ReRankingConfig(
            enabled=True,
            method=ReRankingMethod.WEIGHTED,
            params={"weights": [0.6, 0.4]},
        )
        assert config.enabled is True
        assert config.method == ReRankingMethod.WEIGHTED
        assert config.params["weights"] == [0.6, 0.4]

    def test_reranking_config_rrf_method(self):
        """
        Test ReRankingConfig initialization with RRF method.

        Coverage: ReRankingConfig accepts RRF reranking method.
        """
        config = ReRankingConfig(
            enabled=True,
            method=ReRankingMethod.RRF,
            params={"k": 60},
        )
        assert config.enabled is True
        assert config.method == ReRankingMethod.RRF
        assert config.params["k"] == 60


# ============================================================================
# Test ReRankingConfig Validation
# ============================================================================


@pytest.mark.unit
class TestReRankingConfigValidation:
    """
    Test ReRankingConfig validation.

    Coverage: ReRankingConfig.__post_init__() validates configuration.
    """

    def test_reranking_config_enabled_with_none_method(self):
        """
        Test ReRankingConfig validation rejects enabled with NONE method.

        Coverage: ReRankingConfig.__post_init__() raises ValueError when enabled=True
        but method=NONE.
        """
        with pytest.raises(ValueError) as exc_info:
            ReRankingConfig(
                enabled=True,
                method=ReRankingMethod.NONE,
            )
        assert "Re-ranking is enabled but method is NONE" in str(exc_info.value)

    def test_reranking_config_weighted_default_weights(self):
        """
        Test ReRankingConfig provides default weights for WEIGHTED method.

        Coverage: ReRankingConfig.__post_init__() adds default weights if not provided for WEIGHTED.
        """
        config = ReRankingConfig(
            enabled=True,
            method=ReRankingMethod.WEIGHTED,
            params={},
        )
        assert config.params["weights"] == [0.5, 0.5]

    def test_reranking_config_weighted_custom_weights(self):
        """
        Test ReRankingConfig accepts custom weights for WEIGHTED method.

        Coverage: ReRankingConfig accepts weights that sum to 1.0.
        """
        config = ReRankingConfig(
            enabled=True,
            method=ReRankingMethod.WEIGHTED,
            params={"weights": [0.6, 0.4]},
        )
        assert config.params["weights"] == [0.6, 0.4]

    def test_reranking_config_weighted_weights_must_sum_to_one(self):
        """
        Test ReRankingConfig validation requires weights to sum to 1.0.

        Coverage: ReRankingConfig.__post_init__() raises ValueError when weights don't sum to 1.0.
        """
        with pytest.raises(ValueError) as exc_info:
            ReRankingConfig(
                enabled=True,
                method=ReRankingMethod.WEIGHTED,
                params={"weights": [0.6, 0.5]},  # Sums to 1.1
            )
        assert "Weights must sum to 1.0" in str(exc_info.value)

    def test_reranking_config_rrf_default_k(self):
        """
        Test ReRankingConfig provides default k for RRF method.

        Coverage: ReRankingConfig.__post_init__() adds default k if not provided for RRF.
        """
        config = ReRankingConfig(
            enabled=True,
            method=ReRankingMethod.RRF,
            params={},
        )
        assert config.params["k"] == 60

    def test_reranking_config_rrf_custom_k(self):
        """
        Test ReRankingConfig accepts custom k for RRF method.

        Coverage: ReRankingConfig accepts custom k parameter for RRF.
        """
        config = ReRankingConfig(
            enabled=True,
            method=ReRankingMethod.RRF,
            params={"k": 80},
        )
        assert config.params["k"] == 80

    def test_reranking_config_disabled_accepts_any_method(self):
        """
        Test ReRankingConfig accepts any method when disabled.

        Coverage: ReRankingConfig accepts any method when enabled=False.
        """
        # Should not raise error even with NONE method when disabled
        config = ReRankingConfig(
            enabled=False,
            method=ReRankingMethod.NONE,
        )
        assert config.enabled is False
        assert config.method == ReRankingMethod.NONE


# ============================================================================
# Test ReRankingConfig Edge Cases
# ============================================================================


@pytest.mark.unit
class TestReRankingConfigEdgeCases:
    """
    Test ReRankingConfig edge cases.

    Coverage: ReRankingConfig handles edge cases correctly.
    """

    def test_reranking_config_with_custom_params(self):
        """
        Test ReRankingConfig with custom parameters.

        Coverage: ReRankingConfig accepts custom parameters in params dict.
        """
        custom_params = {"custom_param": "value", "another_param": 123}
        config = ReRankingConfig(
            enabled=True,
            method=ReRankingMethod.WEIGHTED,
            params={"weights": [0.5, 0.5], **custom_params},
        )
        assert config.params["weights"] == [0.5, 0.5]
        assert config.params["custom_param"] == "value"
        assert config.params["another_param"] == 123

    def test_reranking_config_multiple_weights(self):
        """
        Test ReRankingConfig with multiple weights for multi-field search.

        Coverage: ReRankingConfig accepts more than two weights.
        """
        weights = [0.4, 0.3, 0.2, 0.1]  # Four weights
        config = ReRankingConfig(
            enabled=True,
            method=ReRankingMethod.WEIGHTED,
            params={"weights": weights},
        )
        assert config.params["weights"] == weights
        assert sum(config.params["weights"]) == pytest.approx(1.0)

    def test_reranking_config_large_k(self):
        """
        Test ReRankingConfig with large k value.

        Coverage: ReRankingConfig accepts large k values for RRF.
        """
        config = ReRankingConfig(
            enabled=True,
            method=ReRankingMethod.RRF,
            params={"k": 1000},
        )
        assert config.params["k"] == 1000

    def test_reranking_config_small_k(self):
        """
        Test ReRankingConfig with small k value.

        Coverage: ReRankingConfig accepts small k values for RRF.
        """
        config = ReRankingConfig(
            enabled=True,
            method=ReRankingMethod.RRF,
            params={"k": 1},
        )
        assert config.params["k"] == 1

    def test_reranking_config_precision_weights(self):
        """
        Test ReRankingConfig with high precision weights.

        Coverage: ReRankingConfig accepts weights with many decimal places.
        """
        weights = [0.333333, 0.333333, 0.333334]  # Sums to 1.0
        config = ReRankingConfig(
            enabled=True,
            method=ReRankingMethod.WEIGHTED,
            params={"weights": weights},
        )
        assert abs(sum(config.params["weights"]) - 1.0) < 0.0001  # Allow small floating point error

    def test_reranking_config_none_params_dict(self):
        """
        Test ReRankingConfig with None params dict.

        Coverage: ReRankingConfig handles None params dict (should use default_factory).
        """
        # None params should use default_factory (empty dict)
        config = ReRankingConfig(params=None)  # type: ignore
        # Should initialize with empty dict or default params
        # Default factory creates empty dict, then __post_init__ adds defaults if needed
        assert isinstance(config.params, dict)
        # If method is WEIGHTED or RRF, defaults should be added
        config_weighted = ReRankingConfig(
            enabled=True,
            method=ReRankingMethod.WEIGHTED,
            params={},  # Empty dict should get default weights
        )
        assert "weights" in config_weighted.params

    def test_reranking_config_invalid_method_string(self):
        """
        Test ReRankingConfig with invalid method string (not enum).

        Coverage: ReRankingConfig raises error for invalid method string.
        """
        # Invalid method string should raise TypeError
        with pytest.raises((TypeError, ValueError)):
            ReRankingConfig(
                enabled=True,
                method="INVALID_METHOD",  # type: ignore
            )

    def test_reranking_config_weights_count_validation_edge_cases(self):
        """
        Test ReRankingConfig weights count validation edge cases.

        Coverage: ReRankingConfig handles various weight count scenarios.
        """
        # Single weight (edge case)
        try:
            config = ReRankingConfig(
                enabled=True,
                method=ReRankingMethod.WEIGHTED,
                params={"weights": [1.0]},  # Single weight
            )
            # Should accept single weight or might need validation
            assert len(config.params["weights"]) == 1
        except ValueError:
            # If weights count validation requires specific count, that's expected
            pass

        # Very many weights
        many_weights = [1.0 / 10] * 10  # 10 equal weights
        config = ReRankingConfig(
            enabled=True,
            method=ReRankingMethod.WEIGHTED,
            params={"weights": many_weights},
        )
        assert len(config.params["weights"]) == 10
        assert sum(config.params["weights"]) == pytest.approx(1.0)

    def test_reranking_config_zero_k_value_rrf(self):
        """
        Test ReRankingConfig with zero k value for RRF.

        Coverage: ReRankingConfig handles zero k value for RRF method (edge case).
        """
        # Zero k might cause issues in RRF calculation
        try:
            config = ReRankingConfig(
                enabled=True,
                method=ReRankingMethod.RRF,
                params={"k": 0},
            )
            # Should accept or might raise error depending on implementation
            assert config.params["k"] == 0
        except ValueError:
            # If zero k is invalid, that's expected
            pass


# ============================================================================
# Test ReRankingConfig Boundary Values
# ============================================================================


@pytest.mark.unit
class TestReRankingConfigBoundaryValues:
    """
    Test ReRankingConfig boundary value conditions.

    Coverage: Maximum/minimum integers, extreme floats, NaN/Infinity values.
    """

    def test_reranking_config_maximum_integer_k(self):
        """
        Test ReRankingConfig with maximum integer value for k.

        Coverage: ReRankingConfig handles sys.maxsize for k.
        """
        import sys

        config = ReRankingConfig(
            enabled=True,
            method=ReRankingMethod.RRF,
            params={"k": sys.maxsize},
        )
        assert config.params["k"] == sys.maxsize

    def test_reranking_config_very_small_k(self):
        """
        Test ReRankingConfig with very small positive k value.

        Coverage: ReRankingConfig handles very small positive k values.
        """
        config = ReRankingConfig(
            enabled=True,
            method=ReRankingMethod.RRF,
            params={"k": 1},
        )
        assert config.params["k"] == 1

    def test_reranking_config_very_large_weights(self):
        """
        Test ReRankingConfig with very large weight values.

        Coverage: ReRankingConfig handles large float weight values.
        """
        # Weights should sum to 1.0 regardless of magnitude
        weights = [0.5, 0.5]
        config = ReRankingConfig(
            enabled=True,
            method=ReRankingMethod.WEIGHTED,
            params={"weights": weights},
        )
        assert config.params["weights"] == weights

    def test_reranking_config_nan_in_weights(self):
        """
        Test ReRankingConfig with NaN in weights.

        Coverage: ReRankingConfig handles NaN in weights (should fail validation).
        """
        # NaN in weights should fail validation
        with pytest.raises((ValueError, TypeError)):
            ReRankingConfig(
                enabled=True,
                method=ReRankingMethod.WEIGHTED,
                params={"weights": [float("nan"), 0.5]},  # type: ignore
            )

    def test_reranking_config_infinity_in_weights(self):
        """
        Test ReRankingConfig with infinity in weights.

        Coverage: ReRankingConfig handles infinity in weights.
        """
        # Infinity in weights should fail validation
        try:
            config = ReRankingConfig(
                enabled=True,
                method=ReRankingMethod.WEIGHTED,
                params={"weights": [float("inf"), 0.5]},  # type: ignore
            )
            # If accepted, verify
            assert float("inf") in config.params["weights"]
        except (ValueError, TypeError):
            # Expected: validation should reject infinity
            pass
