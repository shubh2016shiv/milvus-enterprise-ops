"""
Comprehensive unit tests for IndexOperationConfig.

This module provides systematic testing of the IndexOperationConfig class,
including initialization, validation, serialization, and boundary conditions.
"""

import pytest

from milvus_ops.index_operations.config import IndexOperationConfig

# ============================================================================
# Test IndexOperationConfig Initialization
# ============================================================================


@pytest.mark.unit
class TestIndexOperationConfigInitialization:
    """
    Test IndexOperationConfig initialization.

    Coverage: IndexOperationConfig constructor and default values.
    """

    def test_init_default_values(self):
        """
        Test IndexOperationConfig with default values.

        Coverage: IndexOperationConfig.__init__() with no arguments.
        """
        config = IndexOperationConfig()
        assert config.default_timeout == 60.0
        assert config.build_progress_poll_interval == 2.0
        assert config.max_concurrent_builds == 0
        assert config.enable_timing is True
        assert config.auto_optimize_params is False
        assert config.resource_monitoring is False
        assert config.retry_transient_errors is True
        assert config.max_transient_retries == 3
        assert config.transient_retry_delay == 0.5

    def test_init_custom_values(self):
        """
        Test IndexOperationConfig with custom values.

        Coverage: IndexOperationConfig.__init__() with custom parameters.
        """
        config = IndexOperationConfig(
            default_timeout=120.0,
            build_progress_poll_interval=5.0,
            max_concurrent_builds=5,
            enable_timing=False,
            auto_optimize_params=True,
            resource_monitoring=True,
            retry_transient_errors=False,
            max_transient_retries=5,
            transient_retry_delay=1.0,
        )
        assert config.default_timeout == 120.0
        assert config.build_progress_poll_interval == 5.0
        assert config.max_concurrent_builds == 5
        assert config.enable_timing is False
        assert config.auto_optimize_params is True
        assert config.resource_monitoring is True
        assert config.retry_transient_errors is False
        assert config.max_transient_retries == 5
        assert config.transient_retry_delay == 1.0

    def test_init_none_timeout(self):
        """
        Test IndexOperationConfig with None timeout.

        Coverage: IndexOperationConfig with default_timeout=None.
        """
        config = IndexOperationConfig(default_timeout=None)
        assert config.default_timeout is None


# ============================================================================
# Test IndexOperationConfig __post_init__
# ============================================================================


@pytest.mark.unit
class TestIndexOperationConfigPostInit:
    """
    Test IndexOperationConfig __post_init__ validation.

    Coverage: IndexOperationConfig validation and auto-correction.
    """

    def test_post_init_poll_interval_zero(self):
        """
        Test __post_init__ corrects poll_interval to 2.0 when zero.

        Coverage: __post_init__() auto-corrects build_progress_poll_interval <= 0.
        """
        config = IndexOperationConfig(build_progress_poll_interval=0.0)
        assert config.build_progress_poll_interval == 2.0

    def test_post_init_poll_interval_negative(self):
        """
        Test __post_init__ corrects poll_interval to 2.0 when negative.

        Coverage: __post_init__() auto-corrects build_progress_poll_interval < 0.
        """
        config = IndexOperationConfig(build_progress_poll_interval=-1.0)
        assert config.build_progress_poll_interval == 2.0

    def test_post_init_max_concurrent_builds_negative(self):
        """
        Test __post_init__ corrects max_concurrent_builds to 0 when negative.

        Coverage: __post_init__() auto-corrects max_concurrent_builds < 0.
        """
        config = IndexOperationConfig(max_concurrent_builds=-1)
        assert config.max_concurrent_builds == 0

    def test_post_init_max_transient_retries_negative(self):
        """
        Test __post_init__ corrects max_transient_retries to 0 when negative.

        Coverage: __post_init__() auto-corrects max_transient_retries < 0.
        """
        config = IndexOperationConfig(max_transient_retries=-1)
        assert config.max_transient_retries == 0

    def test_post_init_transient_retry_delay_negative(self):
        """
        Test __post_init__ corrects transient_retry_delay to 0.5 when negative.

        Coverage: __post_init__() auto-corrects transient_retry_delay < 0.
        """
        config = IndexOperationConfig(transient_retry_delay=-1.0)
        assert config.transient_retry_delay == 0.5


# ============================================================================
# Test IndexOperationConfig from_dict
# ============================================================================


@pytest.mark.unit
class TestIndexOperationConfigFromDict:
    """
    Test IndexOperationConfig.from_dict() method.

    Coverage: IndexOperationConfig dictionary deserialization.
    """

    def test_from_dict_valid(self):
        """
        Test from_dict() with valid dictionary.

        Coverage: IndexOperationConfig.from_dict() with valid parameters.
        """
        config_dict = {
            "default_timeout": 120.0,
            "build_progress_poll_interval": 5.0,
            "max_concurrent_builds": 5,
            "enable_timing": False,
        }
        config = IndexOperationConfig.from_dict(config_dict)
        assert config.default_timeout == 120.0
        assert config.build_progress_poll_interval == 5.0
        assert config.max_concurrent_builds == 5
        assert config.enable_timing is False

    def test_from_dict_partial(self):
        """
        Test from_dict() with partial dictionary.

        Coverage: IndexOperationConfig.from_dict() with only some parameters.
        """
        config_dict = {"default_timeout": 120.0}
        config = IndexOperationConfig.from_dict(config_dict)
        assert config.default_timeout == 120.0
        # Other fields should use defaults
        assert config.build_progress_poll_interval == 2.0
        assert config.enable_timing is True

    def test_from_dict_invalid_keys(self):
        """
        Test from_dict() filters invalid keys.

        Coverage: IndexOperationConfig.from_dict() ignores invalid keys.
        """
        config_dict = {
            "default_timeout": 120.0,
            "invalid_key": "should_be_ignored",
            "another_invalid": 123,
        }
        config = IndexOperationConfig.from_dict(config_dict)
        assert config.default_timeout == 120.0
        assert not hasattr(config, "invalid_key")
        assert not hasattr(config, "another_invalid")

    def test_from_dict_empty(self):
        """
        Test from_dict() with empty dictionary.

        Coverage: IndexOperationConfig.from_dict() with empty dict uses defaults.
        """
        config = IndexOperationConfig.from_dict({})
        assert config.default_timeout == 60.0
        assert config.build_progress_poll_interval == 2.0


# ============================================================================
# Test IndexOperationConfig to_dict
# ============================================================================


@pytest.mark.unit
class TestIndexOperationConfigToDict:
    """
    Test IndexOperationConfig.to_dict() method.

    Coverage: IndexOperationConfig dictionary serialization.
    """

    def test_to_dict_default(self):
        """
        Test to_dict() with default values.

        Coverage: IndexOperationConfig.to_dict() with default configuration.
        """
        config = IndexOperationConfig()
        config_dict = config.to_dict()
        assert config_dict["default_timeout"] == 60.0
        assert config_dict["build_progress_poll_interval"] == 2.0
        assert config_dict["max_concurrent_builds"] == 0
        assert config_dict["enable_timing"] is True
        assert config_dict["auto_optimize_params"] is False
        assert config_dict["resource_monitoring"] is False
        assert config_dict["retry_transient_errors"] is True
        assert config_dict["max_transient_retries"] == 3
        assert config_dict["transient_retry_delay"] == 0.5

    def test_to_dict_custom(self):
        """
        Test to_dict() with custom values.

        Coverage: IndexOperationConfig.to_dict() with custom configuration.
        """
        config = IndexOperationConfig(
            default_timeout=120.0,
            build_progress_poll_interval=5.0,
            enable_timing=False,
        )
        config_dict = config.to_dict()
        assert config_dict["default_timeout"] == 120.0
        assert config_dict["build_progress_poll_interval"] == 5.0
        assert config_dict["enable_timing"] is False

    def test_to_dict_roundtrip(self):
        """
        Test to_dict() and from_dict() roundtrip.

        Coverage: IndexOperationConfig serialization roundtrip consistency.
        """
        original = IndexOperationConfig(
            default_timeout=120.0,
            build_progress_poll_interval=5.0,
            max_concurrent_builds=5,
            enable_timing=False,
        )
        config_dict = original.to_dict()
        restored = IndexOperationConfig.from_dict(config_dict)
        assert restored.default_timeout == original.default_timeout
        assert restored.build_progress_poll_interval == original.build_progress_poll_interval
        assert restored.max_concurrent_builds == original.max_concurrent_builds
        assert restored.enable_timing == original.enable_timing


# ============================================================================
# Test IndexOperationConfig Boundary Conditions
# ============================================================================


@pytest.mark.unit
class TestIndexOperationConfigBoundaries:
    """
    Test IndexOperationConfig boundary conditions.

    Coverage: IndexOperationConfig edge cases and boundary values.
    """

    def test_boundary_timeout_zero(self):
        """
        Test IndexOperationConfig with timeout=0.0.

        Coverage: IndexOperationConfig with zero timeout.
        """
        config = IndexOperationConfig(default_timeout=0.0)
        assert config.default_timeout == 0.0

    def test_boundary_timeout_very_large(self):
        """
        Test IndexOperationConfig with very large timeout.

        Coverage: IndexOperationConfig with large timeout value.
        """
        config = IndexOperationConfig(default_timeout=999999.0)
        assert config.default_timeout == 999999.0

    def test_boundary_poll_interval_small(self):
        """
        Test IndexOperationConfig with very small poll_interval.

        Coverage: IndexOperationConfig with minimal poll_interval.
        """
        config = IndexOperationConfig(build_progress_poll_interval=0.1)
        assert config.build_progress_poll_interval == 0.1

    def test_boundary_poll_interval_large(self):
        """
        Test IndexOperationConfig with large poll_interval.

        Coverage: IndexOperationConfig with large poll_interval.
        """
        config = IndexOperationConfig(build_progress_poll_interval=3600.0)
        assert config.build_progress_poll_interval == 3600.0

    def test_boundary_max_concurrent_builds_zero(self):
        """
        Test IndexOperationConfig with max_concurrent_builds=0.

        Coverage: IndexOperationConfig with zero max_concurrent_builds (no limit).
        """
        config = IndexOperationConfig(max_concurrent_builds=0)
        assert config.max_concurrent_builds == 0

    def test_boundary_max_concurrent_builds_large(self):
        """
        Test IndexOperationConfig with large max_concurrent_builds.

        Coverage: IndexOperationConfig with large max_concurrent_builds.
        """
        config = IndexOperationConfig(max_concurrent_builds=1000)
        assert config.max_concurrent_builds == 1000

    def test_boundary_max_transient_retries_zero(self):
        """
        Test IndexOperationConfig with max_transient_retries=0.

        Coverage: IndexOperationConfig with zero retries.
        """
        config = IndexOperationConfig(max_transient_retries=0)
        assert config.max_transient_retries == 0

    def test_boundary_transient_retry_delay_zero(self):
        """
        Test IndexOperationConfig with transient_retry_delay=0.0.

        Coverage: IndexOperationConfig with zero retry delay.
        """
        config = IndexOperationConfig(transient_retry_delay=0.0)
        assert config.transient_retry_delay == 0.0


# ============================================================================
# Test IndexOperationConfig Parameter Combinations
# ============================================================================


@pytest.mark.unit
class TestIndexOperationConfigCombinations:
    """
    Test IndexOperationConfig with various parameter combinations.

    Coverage: IndexOperationConfig with different feature combinations.
    """

    def test_combination_no_retry(self):
        """
        Test IndexOperationConfig with retries disabled.

        Coverage: IndexOperationConfig with retry_transient_errors=False.
        """
        config = IndexOperationConfig(retry_transient_errors=False, max_transient_retries=0)
        assert config.retry_transient_errors is False
        assert config.max_transient_retries == 0

    def test_combination_timing_disabled(self):
        """
        Test IndexOperationConfig with timing disabled.

        Coverage: IndexOperationConfig with enable_timing=False.
        """
        config = IndexOperationConfig(enable_timing=False)
        assert config.enable_timing is False

    def test_combination_optimization_enabled(self):
        """
        Test IndexOperationConfig with auto-optimization enabled.

        Coverage: IndexOperationConfig with auto_optimize_params=True.
        """
        config = IndexOperationConfig(auto_optimize_params=True, resource_monitoring=True)
        assert config.auto_optimize_params is True
        assert config.resource_monitoring is True
