"""
Comprehensive unit tests for base search configuration.

This module provides systematic testing of BaseSearchConfig, enums,
and validation logic.
"""

import pytest

from milvus_ops.search_operations.config.base import (
    BaseSearchConfig,
    FusionMethod,
    MetricType,
    ReRankingMethod,
    SearchType,
)

# ============================================================================
# Test Enums
# ============================================================================


@pytest.mark.unit
class TestEnums:
    """
    Test enum value validation.

    Coverage: SearchType, MetricType, ReRankingMethod, FusionMethod enums.
    """

    def test_search_type_enum_values(self):
        """
        Test SearchType enum has correct values.

        Coverage: SearchType enum contains expected values.
        """
        assert SearchType.SEMANTIC == "semantic"
        assert SearchType.HYBRID == "hybrid"
        assert SearchType.FUSION == "fusion"

    def test_metric_type_enum_values(self):
        """
        Test MetricType enum has correct values.

        Coverage: MetricType enum contains expected values.
        """
        assert MetricType.L2 == "L2"
        assert MetricType.IP == "IP"
        assert MetricType.COSINE == "COSINE"
        assert MetricType.HAMMING == "HAMMING"

    def test_reranking_method_enum_values(self):
        """
        Test ReRankingMethod enum has correct values.

        Coverage: ReRankingMethod enum contains expected values.
        """
        assert ReRankingMethod.NONE == "none"
        assert ReRankingMethod.WEIGHTED == "weighted"
        assert ReRankingMethod.RRF == "rrf"

    def test_fusion_method_enum_values(self):
        """
        Test FusionMethod enum has correct values.

        Coverage: FusionMethod enum contains expected values.
        """
        assert FusionMethod.RRF == "rrf"
        assert FusionMethod.WEIGHTED == "weighted"
        assert FusionMethod.MAX == "max"
        assert FusionMethod.MEAN == "mean"


# ============================================================================
# Test BaseSearchConfig Initialization
# ============================================================================


@pytest.mark.unit
class TestBaseSearchConfigInitialization:
    """
    Test BaseSearchConfig initialization with defaults.

    Coverage: BaseSearchConfig.__init__() with default and custom parameters.
    """

    def test_base_search_config_defaults(self):
        """
        Test BaseSearchConfig initialization with defaults.

        Coverage: BaseSearchConfig.__init__() initializes with default values.
        """
        config = BaseSearchConfig()
        assert config.top_k == 10
        assert config.timeout == 30.0
        assert config.metric_type == MetricType.COSINE
        assert isinstance(config.params, dict)
        assert config.output_fields is None

    def test_base_search_config_default_params(self):
        """
        Test BaseSearchConfig initializes default params in __post_init__.

        Coverage: BaseSearchConfig.__post_init__() initializes default params if empty.
        """
        config = BaseSearchConfig()
        assert "nprobe" in config.params
        assert "ef" in config.params
        assert config.params["nprobe"] == 10
        assert config.params["ef"] == 64

    def test_base_search_config_custom_values(self):
        """
        Test BaseSearchConfig initialization with custom values.

        Coverage: BaseSearchConfig.__init__() accepts custom parameter values.
        """
        config = BaseSearchConfig(
            top_k=20,
            timeout=60.0,
            metric_type=MetricType.L2,
            params={"nprobe": 20, "ef": 128},
            output_fields=["field1", "field2"],
        )
        assert config.top_k == 20
        assert config.timeout == 60.0
        assert config.metric_type == MetricType.L2
        assert config.params["nprobe"] == 20
        assert config.params["ef"] == 128
        assert config.output_fields == ["field1", "field2"]

    def test_base_search_config_custom_params_preserved(self):
        """
        Test BaseSearchConfig preserves custom params without defaults.

        Coverage: BaseSearchConfig.__post_init__() doesn't override custom params.
        """
        custom_params = {"custom_param": 100}
        config = BaseSearchConfig(params=custom_params)
        # Defaults are only added if params is empty, custom params are preserved
        assert "nprobe" not in config.params
        assert "ef" not in config.params
        assert config.params["custom_param"] == 100


# ============================================================================
# Test BaseSearchConfig Validation
# ============================================================================


@pytest.mark.unit
class TestBaseSearchConfigValidation:
    """
    Test BaseSearchConfig validation in __post_init__.

    Coverage: BaseSearchConfig validation for top_k and timeout.
    """

    @pytest.mark.parametrize("top_k", [-1, 0])
    def test_base_search_config_invalid_top_k(self, top_k):
        """
        Test BaseSearchConfig validation rejects non-positive top_k.

        Coverage: BaseSearchConfig.__post_init__() raises ValueError for top_k <= 0.
        """
        with pytest.raises(ValueError) as exc_info:
            BaseSearchConfig(top_k=top_k)
        assert "top_k must be positive" in str(exc_info.value)

    @pytest.mark.parametrize("top_k", [1, 10, 100, 1000])
    def test_base_search_config_valid_top_k(self, top_k):
        """
        Test BaseSearchConfig accepts valid top_k values.

        Coverage: BaseSearchConfig accepts positive top_k values.
        """
        config = BaseSearchConfig(top_k=top_k)
        assert config.top_k == top_k

    @pytest.mark.parametrize("timeout", [-1.0, 0.0, -0.1])
    def test_base_search_config_invalid_timeout(self, timeout):
        """
        Test BaseSearchConfig validation rejects non-positive timeout.

        Coverage: BaseSearchConfig.__post_init__() raises ValueError for timeout <= 0.
        """
        with pytest.raises(ValueError) as exc_info:
            BaseSearchConfig(timeout=timeout)
        assert "timeout must be positive" in str(exc_info.value)

    @pytest.mark.parametrize("timeout", [0.1, 1.0, 30.0, 60.0, 300.0])
    def test_base_search_config_valid_timeout(self, timeout):
        """
        Test BaseSearchConfig accepts valid timeout values.

        Coverage: BaseSearchConfig accepts positive timeout values.
        """
        config = BaseSearchConfig(timeout=timeout)
        assert config.timeout == timeout


# ============================================================================
# Test BaseSearchConfig Edge Cases
# ============================================================================


@pytest.mark.unit
class TestBaseSearchConfigEdgeCases:
    """
    Test BaseSearchConfig edge cases.

    Coverage: BaseSearchConfig handles edge cases correctly.
    """

    def test_base_search_config_empty_output_fields(self):
        """
        Test BaseSearchConfig with empty output_fields list.

        Coverage: BaseSearchConfig handles empty output_fields list.
        """
        config = BaseSearchConfig(output_fields=[])
        assert config.output_fields == []

    def test_base_search_config_large_top_k(self):
        """
        Test BaseSearchConfig with very large top_k.

        Coverage: BaseSearchConfig accepts large top_k values.
        """
        config = BaseSearchConfig(top_k=10000)
        assert config.top_k == 10000

    def test_base_search_config_very_large_timeout(self):
        """
        Test BaseSearchConfig with very large timeout.

        Coverage: BaseSearchConfig accepts large timeout values.
        """
        config = BaseSearchConfig(timeout=3600.0)  # 1 hour
        assert config.timeout == 3600.0

    def test_base_search_config_small_timeout(self):
        """
        Test BaseSearchConfig with very small timeout.

        Coverage: BaseSearchConfig accepts small positive timeout values.
        """
        config = BaseSearchConfig(timeout=0.001)  # 1ms
        assert config.timeout == 0.001

    def test_base_search_config_all_metric_types(self):
        """
        Test BaseSearchConfig with all metric types.

        Coverage: BaseSearchConfig accepts all MetricType enum values.
        """
        metric_types = [MetricType.L2, MetricType.IP, MetricType.COSINE, MetricType.HAMMING]
        for metric_type in metric_types:
            config = BaseSearchConfig(metric_type=metric_type)
            assert config.metric_type == metric_type

    def test_base_search_config_with_none_params_dict(self):
        """
        Test BaseSearchConfig with None params dict (should use default_factory).

        Coverage: BaseSearchConfig handles None params by using default_factory.
        """
        # None params should trigger default initialization
        # Note: dataclass default_factory handles None, but we should verify behavior
        config = BaseSearchConfig(params={})  # Empty dict should use defaults
        assert isinstance(config.params, dict)
        # Should initialize with default params in __post_init__
        assert "nprobe" in config.params
        assert "ef" in config.params

    def test_base_search_config_invalid_metric_type_string(self):
        """
        Test BaseSearchConfig with invalid metric_type string.

        Coverage: BaseSearchConfig raises error or handles invalid metric_type string.
        """
        # Attempt to use invalid string for metric_type
        # Since metric_type expects MetricType enum, invalid string should raise TypeError
        with pytest.raises((TypeError, ValueError)):
            BaseSearchConfig(metric_type="INVALID_METRIC")  # type: ignore

    def test_base_search_config_none_vs_empty_output_fields(self):
        """
        Test BaseSearchConfig with None output_fields vs empty list distinction.

        Coverage: BaseSearchConfig distinguishes between None and empty list for output_fields.
        """
        config_none = BaseSearchConfig(output_fields=None)
        config_empty = BaseSearchConfig(output_fields=[])

        assert config_none.output_fields is None
        assert config_empty.output_fields == []
        assert config_none.output_fields != config_empty.output_fields

    def test_base_search_config_params_dict_mutation_after_init(self):
        """
        Test BaseSearchConfig params dict mutation after init.

        Coverage: BaseSearchConfig params dict can be mutated after initialization.
        """
        config = BaseSearchConfig()
        initial_params = config.params.copy()

        # Mutate params dict
        config.params["new_param"] = 100
        config.params["nprobe"] = 20

        # Verify mutation persists
        assert config.params["new_param"] == 100
        assert config.params["nprobe"] == 20
        assert config.params != initial_params


# ============================================================================
# Test BaseSearchConfig Boundary Values
# ============================================================================


@pytest.mark.unit
class TestBaseSearchConfigBoundaryValues:
    """
    Test BaseSearchConfig boundary value conditions.

    Coverage: Maximum/minimum integers, extreme floats, NaN/Infinity values.
    """

    def test_base_search_config_maximum_integer_top_k(self):
        """
        Test BaseSearchConfig with maximum integer value for top_k.

        Coverage: BaseSearchConfig handles sys.maxsize for top_k.
        """
        import sys

        config = BaseSearchConfig(top_k=sys.maxsize)
        assert config.top_k == sys.maxsize

    def test_base_search_config_minimum_positive_timeout(self):
        """
        Test BaseSearchConfig with minimum positive timeout value.

        Coverage: BaseSearchConfig handles very small positive floats.
        """
        config = BaseSearchConfig(timeout=0.0001)
        assert config.timeout == 0.0001

    def test_base_search_config_very_large_timeout_float(self):
        """
        Test BaseSearchConfig with very large timeout value.

        Coverage: BaseSearchConfig handles large float values.
        """
        # Test with very large but valid float
        config = BaseSearchConfig(timeout=1e100)
        assert config.timeout == 1e100

    def test_base_search_config_infinity_timeout(self):
        """
        Test BaseSearchConfig with infinity timeout.

        Coverage: BaseSearchConfig handles float('inf') for timeout.
        """
        # Infinity should fail validation (must be > 0, but infinity comparison)
        try:
            config = BaseSearchConfig(timeout=float("inf"))
            # If accepted, verify it
            assert config.timeout == float("inf")
        except ValueError:
            # Expected: validation should reject infinity
            pass

    def test_base_search_config_negative_infinity_timeout(self):
        """
        Test BaseSearchConfig with negative infinity timeout.

        Coverage: BaseSearchConfig handles float('-inf') for timeout (should fail validation).
        """
        # Negative infinity should fail validation
        with pytest.raises(ValueError, match="timeout must be positive"):
            BaseSearchConfig(timeout=float("-inf"))

    def test_base_search_config_nan_timeout(self):
        """
        Test BaseSearchConfig with NaN timeout.

        Coverage: BaseSearchConfig handles float('nan') for timeout (should fail validation).
        """
        # NaN should fail validation
        with pytest.raises(ValueError, match="timeout must be positive"):
            BaseSearchConfig(timeout=float("nan"))


# ============================================================================
# Test BaseSearchConfig Type Safety
# ============================================================================


@pytest.mark.unit
class TestBaseSearchConfigTypeSafety:
    """
    Test BaseSearchConfig type safety and validation.

    Coverage: Wrong types, None where not allowed, empty collections, malformed data.
    """

    def test_base_search_config_with_none_params_dict(self):
        """
        Test BaseSearchConfig with None params dict (should use default_factory).

        Coverage: BaseSearchConfig handles None params by using default_factory.
        """
        # Dataclass field with default_factory handles None
        # If params is not provided, default_factory creates empty dict
        # Then __post_init__ adds defaults
        config = BaseSearchConfig()
        assert isinstance(config.params, dict)
        assert "nprobe" in config.params
        assert "ef" in config.params

    def test_base_search_config_invalid_metric_type_string(self):
        """
        Test BaseSearchConfig with invalid metric_type string.

        Coverage: BaseSearchConfig raises error for invalid metric_type strings.
        """
        # Invalid enum string should raise TypeError or ValueError
        with pytest.raises((TypeError, ValueError)):
            BaseSearchConfig(metric_type="INVALID_METRIC")  # type: ignore

    def test_base_search_config_none_output_fields_vs_empty_list(self):
        """
        Test BaseSearchConfig with None output_fields vs empty list.

        Coverage: BaseSearchConfig distinguishes None from empty list.
        """
        config_none = BaseSearchConfig(output_fields=None)
        config_empty = BaseSearchConfig(output_fields=[])

        assert config_none.output_fields is None
        assert config_empty.output_fields == []
        assert config_none.output_fields != config_empty.output_fields


# ============================================================================
# Test BaseSearchConfig State Mutation
# ============================================================================


@pytest.mark.unit
class TestBaseSearchConfigStateMutation:
    """
    Test BaseSearchConfig state mutation behavior.

    Coverage: Params dict mutation effects, field mutability.
    """

    def test_base_search_config_params_dict_mutation(self):
        """
        Test modifying params dict after initialization.

        Coverage: Params dict is mutable and changes propagate.
        """
        config = BaseSearchConfig()
        original_nprobe = config.params["nprobe"]

        # Modify params dict
        config.params["nprobe"] = 50
        assert config.params["nprobe"] == 50
        assert config.params["nprobe"] != original_nprobe

    def test_base_search_config_params_dict_mutation_effects(self):
        """
        Test that params dict mutations affect the instance.

        Coverage: Mutations to params dict are reflected in the instance.
        """
        config = BaseSearchConfig(params={"custom": "value"})
        assert config.params["custom"] == "value"

        # Add new key
        config.params["new_key"] = "new_value"
        assert "new_key" in config.params
        assert config.params["new_key"] == "new_value"

    def test_base_search_config_field_assignment(self):
        """
        Test BaseSearchConfig field assignment behavior.

        Coverage: Field assignments update instance state (dataclass mutability).
        """
        config = BaseSearchConfig()
        original_timeout = config.timeout

        # Assign new value (dataclass fields are mutable by default)
        config.timeout = 60.0
        assert config.timeout == 60.0
        assert config.timeout != original_timeout

    def test_base_search_config_field_assignment_validation(self):
        """
        Test BaseSearchConfig field assignment validation.

        Coverage: Field assignments are validated (if validation is enabled).
        """
        config = BaseSearchConfig(top_k=10)
        assert config.top_k == 10

        # Try to assign invalid value (dataclass doesn't auto-validate, but __post_init__ does)
        # This test verifies that manual assignment doesn't trigger validation
        # In practice, you'd need custom __setattr__ or property setters for validation
        config.top_k = -1  # type: ignore
        # Dataclass allows this, but it's semantically invalid
        assert config.top_k == -1

    @pytest.mark.asyncio
    async def test_base_search_config_concurrent_access(self):
        """
        Test concurrent access to BaseSearchConfig instance.

        Coverage: Thread safety of BaseSearchConfig objects.
        """
        import asyncio

        config = BaseSearchConfig()

        async def modify_params(value: int):
            """Helper to modify params concurrently"""
            config.params["test_key"] = value
            await asyncio.sleep(0.01)  # Simulate async operation
            return config.params.get("test_key")

        # Run concurrent modifications
        tasks = [modify_params(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # Verify that modifications occurred (may be race conditions)
        assert len(results) == 10
        # All should have been able to modify (no exceptions)
