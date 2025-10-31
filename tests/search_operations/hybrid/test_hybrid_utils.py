"""
Comprehensive unit tests for hybrid search utility functions.

This module provides systematic testing of utility functions in hybrid/utils,
including validation, config, metrics utilities, and helper functions.
"""

import pytest

from milvus_ops.search_operations.config.hybrid import HybridSearchConfig
from milvus_ops.search_operations.core.search_ops_exceptions import InvalidSearchParametersError
from milvus_ops.search_operations.search.hybrid.utils.config import (
    BM25Config,
    HybridSearchMode,
    RetryConfig,
)
from milvus_ops.search_operations.search.hybrid.utils.validation import (
    sanitize_query,
    validate_batch_size,
    validate_fusion_weights,
    validate_search_params,
)

# ============================================================================
# Test HybridSearchMode Enum
# ============================================================================


@pytest.mark.unit
class TestHybridSearchMode:
    """
    Test HybridSearchMode enum.

    Coverage: HybridSearchMode enum values.
    """

    def test_hybrid_search_mode_enum_values(self):
        """
        Test HybridSearchMode enum has correct values.

        Coverage: HybridSearchMode enum contains expected values.
        """
        assert HybridSearchMode.VECTOR_SPARSE.value == "vector_sparse"
        assert HybridSearchMode.VECTOR_KEYWORD.value == "vector_keyword"
        assert HybridSearchMode.VECTOR_ONLY.value == "vector_only"
        assert HybridSearchMode.ALL_METHODS.value == "all_methods"


# ============================================================================
# Test BM25Config
# ============================================================================


@pytest.mark.unit
class TestBM25Config:
    """
    Test BM25Config validation.

    Coverage: BM25Config parameter validation and edge cases.
    """

    def test_bm25_config_default(self):
        """
        Test BM25Config initialization with defaults.

        Coverage: BM25Config.__init__() with default values.
        """
        config = BM25Config()
        assert config.k1 == 1.5
        assert config.b == 0.75
        assert config.delta == 1.0
        assert config.min_term_length == 2
        assert config.max_term_length == 50
        assert config.max_dimensions == 10000
        assert config.enable_stopwords is True

    def test_bm25_config_invalid_k1(self):
        """
        Test BM25Config raises error for invalid k1.

        Coverage: BM25Config.__post_init__() validates k1 > 0.
        """
        with pytest.raises(ValueError, match="k1 must be positive"):
            BM25Config(k1=0)

    def test_bm25_config_invalid_b(self):
        """
        Test BM25Config raises error for invalid b.

        Coverage: BM25Config.__post_init__() validates 0 <= b <= 1.
        """
        with pytest.raises(ValueError, match="b must be between 0 and 1"):
            BM25Config(b=1.5)


# ============================================================================
# Test RetryConfig
# ============================================================================


@pytest.mark.unit
class TestRetryConfigValidation:
    """
    Test RetryConfig validation.

    Coverage: RetryConfig parameter validation.
    """

    def test_retry_config_default(self):
        """
        Test RetryConfig initialization with defaults.

        Coverage: RetryConfig.__init__() with default values.
        """
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 2.0

    def test_retry_config_invalid_max_retries(self):
        """
        Test RetryConfig raises error for negative max_retries.

        Coverage: RetryConfig.__post_init__() validates max_retries >= 0.
        """
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            RetryConfig(max_retries=-1)

    def test_retry_config_invalid_initial_delay(self):
        """
        Test RetryConfig raises error for invalid initial_delay.

        Coverage: RetryConfig.__post_init__() validates initial_delay > 0.
        """
        with pytest.raises(ValueError, match="initial_delay must be positive"):
            RetryConfig(initial_delay=0)

    def test_retry_config_max_delay_less_than_initial(self):
        """
        Test RetryConfig raises error when max_delay < initial_delay.

        Coverage: RetryConfig.__post_init__() validates max_delay >= initial_delay.
        """
        with pytest.raises(ValueError, match="max_delay.*must be >=.*initial_delay"):
            RetryConfig(initial_delay=10.0, max_delay=5.0)

    def test_retry_config_invalid_exponential_base(self):
        """
        Test RetryConfig raises error for invalid exponential_base.

        Coverage: RetryConfig.__post_init__() validates exponential_base > 1.
        """
        with pytest.raises(ValueError, match="exponential_base must be > 1"):
            RetryConfig(exponential_base=1.0)


# ============================================================================
# Test validate_search_params()
# ============================================================================


@pytest.mark.unit
class TestValidateSearchParams:
    """
    Test validate_search_params() function.

    Coverage: Parameter validation for hybrid search.
    """

    def test_validate_search_params_valid(self):
        """
        Test validate_search_params() with valid parameters.

        Coverage: validate_search_params() accepts valid parameters.
        """
        config = HybridSearchConfig(
            top_k=10,
            vector_weight=0.7,
            sparse_weight=0.3,
        )
        validate_search_params("test_collection", "test query", config)

    def test_validate_search_params_empty_collection(self):
        """
        Test validate_search_params() with empty collection name.

        Coverage: validate_search_params() raises error for empty collection name.
        """
        config = HybridSearchConfig(top_k=10)
        with pytest.raises(InvalidSearchParametersError, match="cannot be empty"):
            validate_search_params("", "test query", config)

    def test_validate_search_params_empty_query(self):
        """
        Test validate_search_params() with empty query.

        Coverage: validate_search_params() raises error for empty query.
        """
        config = HybridSearchConfig(top_k=10)
        with pytest.raises(InvalidSearchParametersError, match="cannot be empty"):
            validate_search_params("test_collection", "", config)

    def test_validate_search_params_invalid_top_k(self):
        """
        Test validate_search_params() with invalid top_k.

        Coverage: validate_search_params() raises error for invalid top_k.
        """
        # HybridSearchConfig now validates top_k during initialization
        with pytest.raises(ValueError, match="top_k must be positive"):
            HybridSearchConfig(top_k=0)

    def test_validate_search_params_invalid_weights(self):
        """
        Test validate_search_params() with invalid weights.

        Coverage: validate_search_params() raises error for negative weights.
        """
        # HybridSearchConfig now validates weights during initialization
        with pytest.raises(ValueError, match="Weights must be non-negative"):
            HybridSearchConfig(top_k=10, vector_weight=-0.1)


# ============================================================================
# Test sanitize_query()
# ============================================================================


@pytest.mark.unit
class TestSanitizeQuery:
    """
    Test sanitize_query() function.

    Coverage: Query sanitization for hybrid search.
    """

    def test_sanitize_query_normal(self):
        """
        Test sanitize_query() with normal query.

        Coverage: sanitize_query() sanitizes normal queries.
        """
        result = sanitize_query("normal query text")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_sanitize_query_empty(self):
        """
        Test sanitize_query() with empty query.

        Coverage: sanitize_query() raises error for empty query.
        """
        with pytest.raises(InvalidSearchParametersError, match="cannot be empty"):
            sanitize_query("")

    def test_sanitize_query_removes_control_characters(self):
        """
        Test sanitize_query() removes control characters.

        Coverage: sanitize_query() removes control characters.
        """
        query_with_control = "query\x00\x01text"
        result = sanitize_query(query_with_control)
        assert "\x00" not in result
        assert "\x01" not in result

    def test_sanitize_query_enforces_length(self):
        """
        Test sanitize_query() enforces maximum length.

        Coverage: sanitize_query() truncates queries exceeding max_length.
        """
        long_query = "a" * 20000
        result = sanitize_query(long_query, max_length=10000)
        assert len(result) <= 10000

    def test_sanitize_query_normalizes_whitespace(self):
        """
        Test sanitize_query() normalizes whitespace.

        Coverage: sanitize_query() normalizes multiple spaces.
        """
        query = "query    with    spaces"
        result = sanitize_query(query)
        assert "    " not in result


# ============================================================================
# Test validate_fusion_weights()
# ============================================================================


@pytest.mark.unit
class TestValidateFusionWeights:
    """
    Test validate_fusion_weights() function.

    Coverage: Fusion weight validation.
    """

    def test_validate_fusion_weights_valid(self):
        """
        Test validate_fusion_weights() with valid weights.

        Coverage: validate_fusion_weights() accepts valid weights.
        """
        validate_fusion_weights(0.7, 0.3)

    def test_validate_fusion_weights_negative(self):
        """
        Test validate_fusion_weights() with negative weights.

        Coverage: validate_fusion_weights() raises error for negative weights.
        """
        with pytest.raises(InvalidSearchParametersError, match="must be non-negative"):
            validate_fusion_weights(-0.1, 0.5)

    def test_validate_fusion_weights_all_zero(self):
        """
        Test validate_fusion_weights() with all zero weights.

        Coverage: validate_fusion_weights() raises error when all weights are zero.
        """
        with pytest.raises(
            InvalidSearchParametersError, match="At least one weight must be positive"
        ):
            validate_fusion_weights(0.0, 0.0)


# ============================================================================
# Test validate_batch_size()
# ============================================================================


@pytest.mark.unit
class TestValidateBatchSize:
    """
    Test validate_batch_size() function.

    Coverage: Batch size validation.
    """

    def test_validate_batch_size_valid(self):
        """
        Test validate_batch_size() with valid batch size.

        Coverage: validate_batch_size() accepts valid batch sizes.
        """
        validate_batch_size(10)
        validate_batch_size(50)
        validate_batch_size(100)

    def test_validate_batch_size_invalid(self):
        """
        Test validate_batch_size() with invalid batch size.

        Coverage: validate_batch_size() raises error for invalid batch sizes.
        """
        with pytest.raises(InvalidSearchParametersError, match="must be positive"):
            validate_batch_size(0)

        with pytest.raises(InvalidSearchParametersError, match="must be positive"):
            validate_batch_size(-1)

    def test_validate_batch_size_exceeds_maximum(self):
        """
        Test validate_batch_size() with batch size exceeding maximum.

        Coverage: validate_batch_size() raises error when batch size exceeds maximum.
        """
        with pytest.raises(InvalidSearchParametersError, match="exceeds maximum"):
            validate_batch_size(101, max_batch_size=100)
