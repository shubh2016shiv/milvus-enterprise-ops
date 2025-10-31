"""
Comprehensive unit tests for DataOperationConfig.

This module provides systematic testing of the DataOperationConfig class,
including initialization, validation, serialization, and batch size handling.
"""

import pytest

from milvus_ops.data_management_operations.data_ops_config import DataOperationConfig

# ============================================================================
# Test DataOperationConfig Initialization
# ============================================================================


@pytest.mark.unit
class TestDataOperationConfigInitialization:
    """
    Test DataOperationConfig initialization.

    Coverage: DataOperationConfig constructor and default values.
    """

    def test_init_default_values(self):
        """
        Test DataOperationConfig with default values.

        Coverage: DataOperationConfig.__init__() with no arguments.
        """
        config = DataOperationConfig()
        assert config.default_batch_size == 1000
        assert config.max_batch_size == 10000
        assert config.min_batch_size == 100
        assert config.default_operation_timeout == 30.0
        assert config.health_check_timeout == 5.0
        assert config.retry_transient_errors is True
        assert config.max_transient_retries == 3
        assert config.transient_retry_delay == 0.5
        assert config.enable_timing is True
        assert config.strict_validation is True

    def test_init_custom_values(self):
        """
        Test DataOperationConfig with custom values.

        Coverage: DataOperationConfig.__init__() with custom parameters.
        """
        config = DataOperationConfig(
            default_batch_size=500,
            max_batch_size=5000,
            min_batch_size=50,
            default_operation_timeout=60.0,
            health_check_timeout=10.0,
            retry_transient_errors=False,
            max_transient_retries=5,
            transient_retry_delay=1.0,
            enable_timing=False,
            strict_validation=False,
        )
        assert config.default_batch_size == 500
        assert config.max_batch_size == 5000
        assert config.min_batch_size == 50
        assert config.default_operation_timeout == 60.0
        assert config.health_check_timeout == 10.0
        assert config.retry_transient_errors is False
        assert config.max_transient_retries == 5
        assert config.transient_retry_delay == 1.0
        assert config.enable_timing is False
        assert config.strict_validation is False

    def test_init_none_timeout(self):
        """
        Test DataOperationConfig with None timeout.

        Coverage: DataOperationConfig with default_operation_timeout=None.
        """
        config = DataOperationConfig(default_operation_timeout=None)
        assert config.default_operation_timeout is None


# ============================================================================
# Test DataOperationConfig __post_init__
# ============================================================================


@pytest.mark.unit
class TestDataOperationConfigPostInit:
    """
    Test DataOperationConfig __post_init__ validation.

    Coverage: DataOperationConfig batch size normalization and validation.
    """

    def test_post_init_batch_size_below_min(self):
        """
        Test __post_init__ corrects batch size below minimum.

        Coverage: __post_init__() auto-corrects default_batch_size below min_batch_size.
        """
        config = DataOperationConfig(default_batch_size=50, min_batch_size=100)
        assert config.default_batch_size == 100  # Auto-corrected to min

    def test_post_init_batch_size_above_max(self):
        """
        Test __post_init__ corrects batch size above maximum.

        Coverage: __post_init__() auto-corrects default_batch_size above max_batch_size.
        """
        config = DataOperationConfig(default_batch_size=20000, max_batch_size=10000)
        assert config.default_batch_size == 10000  # Auto-corrected to max

    def test_post_init_batch_size_valid(self):
        """
        Test __post_init__ with valid batch size.

        Coverage: __post_init__() leaves valid batch size unchanged.
        """
        config = DataOperationConfig(
            default_batch_size=500, min_batch_size=100, max_batch_size=1000
        )
        assert config.default_batch_size == 500  # Unchanged

    def test_post_init_negative_retries_raises_error(self):
        """
        Test __post_init__ raises error for negative max_transient_retries.

        Coverage: __post_init__() validates max_transient_retries >= 0.
        """
        with pytest.raises(ValueError, match="max_transient_retries must be non-negative"):
            DataOperationConfig(max_transient_retries=-1)

    def test_post_init_negative_retry_delay_raises_error(self):
        """
        Test __post_init__ raises error for negative transient_retry_delay.

        Coverage: __post_init__() validates transient_retry_delay >= 0.
        """
        with pytest.raises(ValueError, match="transient_retry_delay must be non-negative"):
            DataOperationConfig(transient_retry_delay=-0.1)

    @pytest.mark.parametrize(
        "default_batch_size,expected",
        [
            (50, 100),  # Below min -> corrected to min
            (500, 500),  # Within range -> unchanged
            (15000, 10000),  # Above max -> corrected to max
            (100, 100),  # At min -> unchanged
            (10000, 10000),  # At max -> unchanged
        ],
    )
    def test_post_init_batch_size_boundary_values(self, default_batch_size, expected):
        """
        Test __post_init__ with boundary batch size values.

        Coverage: __post_init__() handles boundary batch sizes correctly.
        """
        config = DataOperationConfig(
            default_batch_size=default_batch_size, min_batch_size=100, max_batch_size=10000
        )
        assert config.default_batch_size == expected


# ============================================================================
# Test DataOperationConfig from_dict
# ============================================================================


@pytest.mark.unit
class TestDataOperationConfigFromDict:
    """
    Test DataOperationConfig.from_dict method.

    Coverage: DataOperationConfig dictionary deserialization.
    """

    def test_from_dict_valid_keys(self):
        """
        Test from_dict with valid keys.

        Coverage: from_dict() with all valid configuration keys.
        """
        config_dict = {
            "default_batch_size": 500,
            "max_batch_size": 5000,
            "min_batch_size": 100,
            "default_operation_timeout": 60.0,
            "health_check_timeout": 10.0,
            "retry_transient_errors": False,
            "max_transient_retries": 5,
            "transient_retry_delay": 1.0,
            "enable_timing": False,
            "strict_validation": False,
        }
        config = DataOperationConfig.from_dict(config_dict)
        assert config.default_batch_size == 500
        assert config.max_batch_size == 5000
        assert config.retry_transient_errors is False

    def test_from_dict_invalid_keys_filtered(self):
        """
        Test from_dict filters invalid keys.

        Coverage: from_dict() ignores keys not in dataclass fields.
        """
        config_dict = {
            "default_batch_size": 500,
            "invalid_key": "should_be_ignored",
            "another_invalid": 123,
        }
        config = DataOperationConfig.from_dict(config_dict)
        assert config.default_batch_size == 500
        assert not hasattr(config, "invalid_key")
        assert not hasattr(config, "another_invalid")

    def test_from_dict_partial_keys(self):
        """
        Test from_dict with partial keys.

        Coverage: from_dict() with only some keys specified.
        """
        config_dict = {
            "default_batch_size": 750,
            "enable_timing": False,
        }
        config = DataOperationConfig.from_dict(config_dict)
        assert config.default_batch_size == 750
        assert config.enable_timing is False
        # Other values should use defaults
        assert config.max_batch_size == 10000  # Default
        assert config.retry_transient_errors is True  # Default

    def test_from_dict_empty_dict(self):
        """
        Test from_dict with empty dictionary.

        Coverage: from_dict() with empty dict uses all defaults.
        """
        config = DataOperationConfig.from_dict({})
        assert config.default_batch_size == 1000  # Default
        assert config.enable_timing is True  # Default


# ============================================================================
# Test DataOperationConfig to_dict
# ============================================================================


@pytest.mark.unit
class TestDataOperationConfigToDict:
    """
    Test DataOperationConfig.to_dict method.

    Coverage: DataOperationConfig dictionary serialization.
    """

    def test_to_dict_all_fields(self):
        """
        Test to_dict includes all fields.

        Coverage: to_dict() serializes all configuration fields.
        """
        config = DataOperationConfig(
            default_batch_size=500,
            max_batch_size=5000,
            min_batch_size=100,
            default_operation_timeout=60.0,
            enable_timing=False,
        )
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["default_batch_size"] == 500
        assert config_dict["max_batch_size"] == 5000
        assert config_dict["min_batch_size"] == 100
        assert config_dict["default_operation_timeout"] == 60.0
        assert config_dict["enable_timing"] is False

    def test_to_dict_defaults(self):
        """
        Test to_dict with default values.

        Coverage: to_dict() includes default values.
        """
        config = DataOperationConfig()
        config_dict = config.to_dict()
        assert config_dict["default_batch_size"] == 1000
        assert config_dict["max_batch_size"] == 10000
        assert config_dict["retry_transient_errors"] is True

    def test_to_dict_none_timeout(self):
        """
        Test to_dict with None timeout.

        Coverage: to_dict() preserves None values.
        """
        config = DataOperationConfig(default_operation_timeout=None)
        config_dict = config.to_dict()
        assert config_dict["default_operation_timeout"] is None

    def test_to_dict_round_trip(self):
        """
        Test to_dict and from_dict round trip.

        Coverage: to_dict() and from_dict() are inverse operations.
        """
        original = DataOperationConfig(
            default_batch_size=750,
            max_batch_size=5000,
            enable_timing=False,
            retry_transient_errors=False,
        )
        config_dict = original.to_dict()
        restored = DataOperationConfig.from_dict(config_dict)
        assert restored.default_batch_size == original.default_batch_size
        assert restored.max_batch_size == original.max_batch_size
        assert restored.enable_timing == original.enable_timing
        assert restored.retry_transient_errors == original.retry_transient_errors


# ============================================================================
# Test DataOperationConfig validate_batch_size
# ============================================================================


@pytest.mark.unit
class TestDataOperationConfigValidateBatchSize:
    """
    Test DataOperationConfig.validate_batch_size method.

    Coverage: DataOperationConfig batch size validation.
    """

    def test_validate_batch_size_none(self):
        """
        Test validate_batch_size with None returns default.

        Coverage: validate_batch_size() returns default_batch_size when None.
        """
        config = DataOperationConfig(default_batch_size=500)
        assert config.validate_batch_size(None) == 500

    @pytest.mark.parametrize(
        "batch_size",
        [100, 500, 1000, 5000, 10000],
    )
    def test_validate_batch_size_valid(self, batch_size):
        """
        Test validate_batch_size with valid values.

        Coverage: validate_batch_size() accepts values within min/max bounds.
        """
        config = DataOperationConfig(min_batch_size=100, max_batch_size=10000)
        assert config.validate_batch_size(batch_size) == batch_size

    def test_validate_batch_size_below_min_raises_error(self):
        """
        Test validate_batch_size raises error below minimum.

        Coverage: validate_batch_size() raises ValueError below min_batch_size.
        """
        config = DataOperationConfig(min_batch_size=100)
        with pytest.raises(ValueError, match="Batch size .* is below minimum"):
            config.validate_batch_size(50)

    def test_validate_batch_size_above_max_raises_error(self):
        """
        Test validate_batch_size raises error above maximum.

        Coverage: validate_batch_size() raises ValueError above max_batch_size.
        """
        config = DataOperationConfig(max_batch_size=10000)
        with pytest.raises(ValueError, match="Batch size .* exceeds maximum"):
            config.validate_batch_size(15000)

    @pytest.mark.parametrize(
        "batch_size,should_raise",
        [
            (99, True),  # Below min
            (100, False),  # At min
            (500, False),  # Valid
            (10000, False),  # At max
            (10001, True),  # Above max
        ],
    )
    def test_validate_batch_size_boundaries(self, batch_size, should_raise):
        """
        Test validate_batch_size with boundary values.

        Coverage: validate_batch_size() correctly handles boundary conditions.
        """
        config = DataOperationConfig(min_batch_size=100, max_batch_size=10000)
        if should_raise:
            with pytest.raises(ValueError):
                config.validate_batch_size(batch_size)
        else:
            assert config.validate_batch_size(batch_size) == batch_size

    def test_validate_batch_size_zero(self):
        """
        Test validate_batch_size with zero.

        Coverage: validate_batch_size() rejects zero batch size.
        """
        config = DataOperationConfig(min_batch_size=100)
        with pytest.raises(ValueError):
            config.validate_batch_size(0)

    def test_validate_batch_size_negative(self):
        """
        Test validate_batch_size with negative value.

        Coverage: validate_batch_size() rejects negative batch size.
        """
        config = DataOperationConfig(min_batch_size=100)
        with pytest.raises(ValueError):
            config.validate_batch_size(-10)
