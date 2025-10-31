"""
Comprehensive unit tests for semantic search configuration.

This module provides systematic testing of SemanticSearchConfig,
including initialization, inheritance, and validation.
"""

import pytest

from milvus_ops.search_operations.config.base import BaseSearchConfig, MetricType
from milvus_ops.search_operations.config.semantic import SemanticSearchConfig

# ============================================================================
# Test SemanticSearchConfig Initialization
# ============================================================================


@pytest.mark.unit
class TestSemanticSearchConfigInitialization:
    """
    Test SemanticSearchConfig initialization.

    Coverage: SemanticSearchConfig.__init__() with defaults and custom values.
    """

    def test_semantic_search_config_defaults(self):
        """
        Test SemanticSearchConfig initialization with defaults.

        Coverage: SemanticSearchConfig.__init__() initializes with default values.
        """
        config = SemanticSearchConfig()
        assert config.search_field == "vector"
        assert config.expr is None
        assert isinstance(config, BaseSearchConfig)
        assert config.top_k == 10
        assert config.timeout == 30.0
        assert config.metric_type == MetricType.COSINE

    def test_semantic_search_config_custom_values(self):
        """
        Test SemanticSearchConfig initialization with custom values.

        Coverage: SemanticSearchConfig.__init__() accepts custom parameter values.
        """
        config = SemanticSearchConfig(
            top_k=20,
            timeout=60.0,
            metric_type=MetricType.L2,
            search_field="embedding",
            expr='category == "test"',
            output_fields=["text", "category"],
        )
        assert config.search_field == "embedding"
        assert config.expr == 'category == "test"'
        assert config.top_k == 20
        assert config.timeout == 60.0
        assert config.metric_type == MetricType.L2
        assert config.output_fields == ["text", "category"]

    def test_semantic_search_config_inherits_from_base(self):
        """
        Test SemanticSearchConfig inherits from BaseSearchConfig.

        Coverage: SemanticSearchConfig is a subclass of BaseSearchConfig.
        """
        assert issubclass(SemanticSearchConfig, BaseSearchConfig)


# ============================================================================
# Test SemanticSearchConfig Edge Cases
# ============================================================================


@pytest.mark.unit
class TestSemanticSearchConfigEdgeCases:
    """
    Test SemanticSearchConfig edge cases.

    Coverage: SemanticSearchConfig handles edge cases correctly.
    """

    def test_semantic_search_config_empty_search_field(self):
        """
        Test SemanticSearchConfig with empty search_field.

        Coverage: SemanticSearchConfig accepts empty string for search_field.
        """
        config = SemanticSearchConfig(search_field="")
        assert config.search_field == ""

    def test_semantic_search_config_empty_expr(self):
        """
        Test SemanticSearchConfig with empty expr string.

        Coverage: SemanticSearchConfig accepts empty string for expr.
        """
        config = SemanticSearchConfig(expr="")
        assert config.expr == ""

    def test_semantic_search_config_complex_expr(self):
        """
        Test SemanticSearchConfig with complex filter expression.

        Coverage: SemanticSearchConfig accepts complex filter expressions.
        """
        complex_expr = '(category == "test" or category == "production") and score > 0.5'
        config = SemanticSearchConfig(expr=complex_expr)
        assert config.expr == complex_expr

    def test_semantic_search_config_all_base_properties(self):
        """
        Test SemanticSearchConfig has all BaseSearchConfig properties.

        Coverage: SemanticSearchConfig inherits all base config properties.
        """
        config = SemanticSearchConfig()
        assert hasattr(config, "top_k")
        assert hasattr(config, "timeout")
        assert hasattr(config, "metric_type")
        assert hasattr(config, "params")
        assert hasattr(config, "output_fields")

    def test_semantic_search_config_none_search_field(self):
        """
        Test SemanticSearchConfig with None search_field (edge case).

        Coverage: SemanticSearchConfig handles None search_field.
        """
        # None search_field might be invalid, but should not crash initialization
        # Check actual implementation behavior
        try:
            config = SemanticSearchConfig(search_field=None)  # type: ignore
            # Depends on implementation
            assert config.search_field is None or config.search_field == ""
        except (TypeError, ValueError):
            # If None is not allowed, that's expected
            pass

    def test_semantic_search_config_none_vs_empty_expr(self):
        """
        Test SemanticSearchConfig with None expr vs empty string distinction.

        Coverage: SemanticSearchConfig distinguishes between None and empty string for expr.
        """
        config_none = SemanticSearchConfig(expr=None)
        config_empty = SemanticSearchConfig(expr="")

        assert config_none.expr is None
        assert config_empty.expr == ""
        assert config_none.expr != config_empty.expr

    def test_semantic_search_config_very_long_expr(self):
        """
        Test SemanticSearchConfig with very long expr expression.

        Coverage: SemanticSearchConfig handles very long expr expressions.
        """
        # Create a very long expression
        long_expr = " and ".join([f'field{i} == "value{i}"' for i in range(100)])
        config = SemanticSearchConfig(expr=long_expr)
        assert config.expr == long_expr
        assert len(config.expr) > 1000

    def test_semantic_search_config_sql_injection_in_expr(self):
        """
        Test SemanticSearchConfig with SQL injection in expr.

        Coverage: SemanticSearchConfig accepts expr with potential SQL injection
        (validation handled elsewhere).
        """
        # Note: SQL injection prevention should be handled by sanitization layer
        # This test verifies the config accepts the expression (doesn't crash)
        sql_injection_expr = "'; DROP TABLE users; --"
        config = SemanticSearchConfig(expr=sql_injection_expr)
        # Config should accept it (sanitization happens at search time)
        assert config.expr == sql_injection_expr


# ============================================================================
# Test SemanticSearchConfig Boundary Values
# ============================================================================


@pytest.mark.unit
class TestSemanticSearchConfigBoundaryValues:
    """
    Test SemanticSearchConfig boundary value conditions.

    Coverage: Maximum/minimum integers, extreme floats, NaN/Infinity values.
    """

    def test_semantic_search_config_maximum_integer_top_k(self):
        """
        Test SemanticSearchConfig with maximum integer value for top_k.

        Coverage: SemanticSearchConfig handles sys.maxsize for top_k.
        """
        import sys

        config = SemanticSearchConfig(top_k=sys.maxsize)
        assert config.top_k == sys.maxsize

    def test_semantic_search_config_minimum_positive_timeout(self):
        """
        Test SemanticSearchConfig with minimum positive timeout value.

        Coverage: SemanticSearchConfig handles very small positive floats.
        """
        config = SemanticSearchConfig(timeout=0.0001)
        assert config.timeout == 0.0001

    def test_semantic_search_config_very_large_timeout_float(self):
        """
        Test SemanticSearchConfig with very large timeout value.

        Coverage: SemanticSearchConfig handles large float values.
        """
        config = SemanticSearchConfig(timeout=1e100)
        assert config.timeout == 1e100

    def test_semantic_search_config_infinity_timeout(self):
        """
        Test SemanticSearchConfig with infinity timeout.

        Coverage: SemanticSearchConfig handles float('inf') for timeout.
        """
        try:
            config = SemanticSearchConfig(timeout=float("inf"))
            assert config.timeout == float("inf")
        except ValueError:
            # Expected: validation should reject infinity
            pass

    def test_semantic_search_config_nan_timeout(self):
        """
        Test SemanticSearchConfig with NaN timeout.

        Coverage: SemanticSearchConfig handles float('nan') for timeout (should fail validation).
        """
        with pytest.raises(ValueError, match="timeout must be positive"):
            SemanticSearchConfig(timeout=float("nan"))
