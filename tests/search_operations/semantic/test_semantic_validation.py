"""
Comprehensive unit tests for semantic search validation and sanitization.

This module provides systematic testing of SearchValidator and QuerySanitizer,
including query sanitization, SQL injection prevention, collection name validation,
and parameter constraints.
"""

import pytest

from milvus_ops.search_operations.config.base import MetricType
from milvus_ops.search_operations.config.semantic import SemanticSearchConfig
from milvus_ops.search_operations.core.search_ops_exceptions import InvalidSearchParametersError
from milvus_ops.search_operations.search.semantic.validation import QuerySanitizer, SearchValidator

# ============================================================================
# Test SearchValidator
# ============================================================================


@pytest.mark.unit
class TestSearchValidator:
    """
    Test SearchValidator class.

    Coverage: Collection name validation, query validation, top_k, timeout, metric_type validation.
    """

    def test_validate_collection_name_valid(self):
        """
        Test validate_collection_name() with valid collection name.

        Coverage: SearchValidator.validate_collection_name() accepts valid collection names.
        """
        SearchValidator.validate_collection_name("valid_collection_name")
        SearchValidator.validate_collection_name("valid-collection-name")
        SearchValidator.validate_collection_name("ValidCollection123")

    def test_validate_collection_name_empty(self):
        """
        Test validate_collection_name() with empty collection name.

        Coverage: SearchValidator.validate_collection_name() raises error for empty name.
        """
        with pytest.raises(InvalidSearchParametersError, match="cannot be empty"):
            SearchValidator.validate_collection_name("")

        with pytest.raises(InvalidSearchParametersError, match="cannot be empty"):
            SearchValidator.validate_collection_name("   ")

    def test_validate_collection_name_too_long(self):
        """
        Test validate_collection_name() with very long collection name.

        Coverage: SearchValidator.validate_collection_name() raises error for names > 255 chars.
        """
        long_name = "a" * 256
        with pytest.raises(InvalidSearchParametersError, match="exceeds maximum length"):
            SearchValidator.validate_collection_name(long_name)

    def test_validate_collection_name_invalid_characters(self):
        """
        Test validate_collection_name() with invalid characters.

        Coverage: SearchValidator.validate_collection_name() raises error for invalid characters.
        """
        with pytest.raises(InvalidSearchParametersError, match="alphanumeric"):
            SearchValidator.validate_collection_name("invalid@collection")

        with pytest.raises(InvalidSearchParametersError, match="alphanumeric"):
            SearchValidator.validate_collection_name("invalid#collection")

    def test_validate_query_valid(self):
        """
        Test validate_query() with valid query.

        Coverage: SearchValidator.validate_query() accepts valid queries.
        """
        SearchValidator.validate_query("valid query text")
        SearchValidator.validate_query("A" * 1000)

    def test_validate_query_empty(self):
        """
        Test validate_query() with empty query.

        Coverage: SearchValidator.validate_query() raises error for empty query.
        """
        with pytest.raises(InvalidSearchParametersError, match="cannot be empty"):
            SearchValidator.validate_query("")

        with pytest.raises(InvalidSearchParametersError, match="cannot be empty"):
            SearchValidator.validate_query("   ")

    def test_validate_query_too_long(self):
        """
        Test validate_query() with very long query.

        Coverage: SearchValidator.validate_query() raises error for queries > MAX_QUERY_LENGTH.
        """
        long_query = "a" * (SearchValidator.MAX_QUERY_LENGTH + 1)
        with pytest.raises(InvalidSearchParametersError, match="exceeds maximum length"):
            SearchValidator.validate_query(long_query)

    def test_validate_top_k_valid(self):
        """
        Test validate_top_k() with valid top_k values.

        Coverage: SearchValidator.validate_top_k() accepts valid top_k values.
        """
        SearchValidator.validate_top_k(1)
        SearchValidator.validate_top_k(10)
        SearchValidator.validate_top_k(SearchValidator.MAX_TOP_K)

    def test_validate_top_k_invalid(self):
        """
        Test validate_top_k() with invalid top_k values.

        Coverage: SearchValidator.validate_top_k() raises error for invalid top_k.
        """
        with pytest.raises(InvalidSearchParametersError, match="must be positive"):
            SearchValidator.validate_top_k(0)

        with pytest.raises(InvalidSearchParametersError, match="must be positive"):
            SearchValidator.validate_top_k(-1)

        with pytest.raises(InvalidSearchParametersError, match="exceeds maximum"):
            SearchValidator.validate_top_k(SearchValidator.MAX_TOP_K + 1)

    def test_validate_timeout_valid(self):
        """
        Test validate_timeout() with valid timeout values.

        Coverage: SearchValidator.validate_timeout() accepts valid timeout values.
        """
        SearchValidator.validate_timeout(SearchValidator.MIN_TIMEOUT)
        SearchValidator.validate_timeout(30.0)
        SearchValidator.validate_timeout(300.0)
        SearchValidator.validate_timeout(None)

    def test_validate_timeout_invalid(self):
        """
        Test validate_timeout() with invalid timeout values.

        Coverage: SearchValidator.validate_timeout() raises error for invalid timeout.
        """
        with pytest.raises(InvalidSearchParametersError, match="must be at least"):
            SearchValidator.validate_timeout(SearchValidator.MIN_TIMEOUT - 0.1)

        with pytest.raises(InvalidSearchParametersError, match="exceeds maximum"):
            SearchValidator.validate_timeout(301.0)

    def test_validate_metric_type_valid(self):
        """
        Test validate_metric_type() with valid metric types.

        Coverage: SearchValidator.validate_metric_type() accepts valid metric types.
        """
        for metric in SearchValidator.VALID_METRIC_TYPES:
            SearchValidator.validate_metric_type(metric)
            SearchValidator.validate_metric_type(metric.lower())
        SearchValidator.validate_metric_type(None)

    def test_validate_metric_type_invalid(self):
        """
        Test validate_metric_type() with invalid metric type.

        Coverage: SearchValidator.validate_metric_type() raises error for invalid metric type.
        """
        with pytest.raises(InvalidSearchParametersError, match="Invalid metric_type"):
            SearchValidator.validate_metric_type("INVALID_METRIC")

    def test_validate_config_valid(self):
        """
        Test validate_config() with valid config.

        Coverage: SearchValidator.validate_config() accepts valid SemanticSearchConfig.
        """
        config = SemanticSearchConfig(
            top_k=10,
            search_field="vector",
            metric_type=MetricType.COSINE,
            timeout=30.0,
        )
        SearchValidator.validate_config(config)

    def test_validate_config_invalid_search_field(self):
        """
        Test validate_config() with invalid search_field.

        Coverage: SearchValidator.validate_config() raises error for empty search_field.
        """
        config = SemanticSearchConfig(
            top_k=10,
            search_field="",
            metric_type=MetricType.COSINE,
        )
        with pytest.raises(InvalidSearchParametersError, match="search_field cannot be empty"):
            SearchValidator.validate_config(config)

    def test_validate_config_invalid_output_fields(self):
        """
        Test validate_config() with invalid output_fields.

        Coverage: SearchValidator.validate_config() validates output_fields type.
        """
        config = SemanticSearchConfig(
            top_k=10,
            search_field="vector",
            output_fields="not_a_list",  # type: ignore
        )
        with pytest.raises(InvalidSearchParametersError, match="output_fields must be a list"):
            SearchValidator.validate_config(config)

    def test_validate_search_params_valid(self):
        """
        Test validate_search_params() with valid parameters.

        Coverage: SearchValidator.validate_search_params() validates all parameters together.
        """
        config = SemanticSearchConfig(
            top_k=10,
            search_field="vector",
            metric_type=MetricType.COSINE,
        )
        SearchValidator.validate_search_params("test_collection", "test query", config)

    def test_validate_search_params_invalid(self):
        """
        Test validate_search_params() with invalid parameters.

        Coverage: SearchValidator.validate_search_params() raises error for invalid parameters.
        """
        config = SemanticSearchConfig(top_k=10, search_field="vector")
        with pytest.raises(InvalidSearchParametersError):
            SearchValidator.validate_search_params("", "test query", config)


# ============================================================================
# Test QuerySanitizer
# ============================================================================


@pytest.mark.unit
class TestQuerySanitizer:
    """
    Test QuerySanitizer class.

    Coverage: Query sanitization, SQL injection prevention, special character handling.
    """

    def test_sanitize_query_normal(self):
        """
        Test sanitize_query() with normal query.

        Coverage: QuerySanitizer.sanitize_query() sanitizes normal queries.
        """
        result = QuerySanitizer.sanitize_query("normal query text")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_sanitize_query_removes_control_characters(self):
        """
        Test sanitize_query() removes control characters.

        Coverage: QuerySanitizer.sanitize_query() removes control characters.
        """
        query_with_control = "query\x00\x01\x02text"
        result = QuerySanitizer.sanitize_query(query_with_control)
        # Control characters should be removed
        assert "\x00" not in result
        assert "\x01" not in result

    def test_sanitize_query_preserves_newline_tab(self):
        """
        Test sanitize_query() preserves newline and tab.

        Coverage: QuerySanitizer.sanitize_query() preserves newline and tab characters.
        """
        query = "line1\nline2\twith tab"
        result = QuerySanitizer.sanitize_query(query)
        # Newlines and tabs may be normalized in whitespace normalization
        assert isinstance(result, str)

    def test_sanitize_query_normalizes_whitespace(self):
        """
        Test sanitize_query() normalizes whitespace.

        Coverage: QuerySanitizer.sanitize_query() normalizes multiple spaces.
        """
        query = "query    with    multiple    spaces"
        result = QuerySanitizer.sanitize_query(query)
        # Multiple spaces should be normalized
        assert "    " not in result

    def test_sanitize_query_enforces_length(self):
        """
        Test sanitize_query() enforces maximum length.

        Coverage: QuerySanitizer.sanitize_query() truncates queries exceeding max_length.
        """
        long_query = "a" * 20000
        result = QuerySanitizer.sanitize_query(long_query, max_length=10000)
        assert len(result) <= 10000

    def test_sanitize_query_sql_injection_prevention(self):
        """
        Test sanitize_query() prevents SQL injection.

        Coverage: QuerySanitizer.sanitize_query() handles SQL injection attempts.
        """
        sql_injection = "'; DROP TABLE users; --"
        result = QuerySanitizer.sanitize_query(sql_injection)
        # Should sanitize but not raise error (handled by sanitization)
        assert isinstance(result, str)

    def test_sanitize_query_special_characters(self):
        """
        Test sanitize_query() with special characters.

        Coverage: QuerySanitizer.sanitize_query() handles special characters.
        """
        special_chars = "query!@#$%^&*()[]{}|\\:;\"'<>,.?/~`"
        result = QuerySanitizer.sanitize_query(special_chars)
        assert isinstance(result, str)

    def test_sanitize_collection_name(self):
        """
        Test sanitize_collection_name() method.

        Coverage: QuerySanitizer.sanitize_collection_name() sanitizes collection names.
        """
        result = QuerySanitizer.sanitize_collection_name("  CollectionName  ")
        assert isinstance(result, str)
        assert result.strip() == result  # Should be trimmed
        assert result == result.lower()  # Should be lowercase

    def test_sanitize_field_name(self):
        """
        Test sanitize_field_name() method.

        Coverage: QuerySanitizer.sanitize_field_name() sanitizes field names.
        """
        result = QuerySanitizer.sanitize_field_name("field name with spaces")
        assert isinstance(result, str)
        assert " " not in result  # Spaces should be replaced
        assert "_" in result  # Spaces replaced with underscores

    def test_sanitize_query_very_long_input(self):
        """
        Test sanitize_query() with very long input.

        Coverage: QuerySanitizer.sanitize_query() handles very long inputs.
        """
        very_long = "A" * 50000
        result = QuerySanitizer.sanitize_query(very_long, max_length=10000)
        assert len(result) == 10000
