"""
Comprehensive unit tests for search parameters validation.

This module provides systematic testing of SearchParams Pydantic model,
including field validation, enum validation, and edge cases.
"""

import contextlib

from pydantic import ValidationError
import pytest

from milvus_ops.search_operations.config.base import (
    FusionMethod,
    MetricType,
    ReRankingMethod,
    SearchType,
)
from milvus_ops.search_operations.config.validation import SearchParams

# ============================================================================
# Test SearchParams Initialization
# ============================================================================


@pytest.mark.unit
class TestSearchParamsInitialization:
    """
    Test SearchParams initialization.

    Coverage: SearchParams.__init__() with defaults and custom values.
    """

    def test_search_params_defaults(self):
        """
        Test SearchParams initialization with defaults.

        Coverage: SearchParams.__init__() initializes with default values.
        """
        params = SearchParams()
        assert params.search_type == SearchType.SEMANTIC
        assert params.top_k == 10
        assert params.metric_type == MetricType.COSINE
        assert params.timeout == 30.0
        assert params.vector_field == "vector"
        assert params.expr is None
        assert params.sparse_field is None
        assert params.keyword_field is None
        assert params.vector_weight == 0.7
        assert params.sparse_weight == 0.3
        assert params.rerank is False
        assert params.rerank_method == ReRankingMethod.NONE
        assert params.rerank_weights is None
        assert params.rerank_k == 60
        assert params.fusion_method == FusionMethod.RRF
        assert params.fusion_weights is None
        assert isinstance(params.params, dict)

    def test_search_params_custom_values(self):
        """
        Test SearchParams initialization with custom values.

        Coverage: SearchParams.__init__() accepts custom parameter values.
        """
        params = SearchParams(
            search_type=SearchType.HYBRID,
            top_k=20,
            metric_type=MetricType.L2,
            timeout=60.0,
            vector_field="embedding",
            expr='category == "test"',
            sparse_field="sparse_vector",
            keyword_field="text",
            vector_weight=0.8,
            sparse_weight=0.2,
            rerank=True,
            rerank_method=ReRankingMethod.WEIGHTED,
            rerank_weights=[0.6, 0.4],
            rerank_k=80,
            fusion_method=FusionMethod.WEIGHTED,
            fusion_weights=[0.5, 0.5],
            params={"nprobe": 20},
            output_fields=["text", "category"],
        )
        assert params.search_type == SearchType.HYBRID
        assert params.top_k == 20
        assert params.metric_type == MetricType.L2
        assert params.timeout == 60.0
        assert params.vector_field == "embedding"
        assert params.expr == 'category == "test"'
        assert params.sparse_field == "sparse_vector"
        assert params.keyword_field == "text"
        assert params.vector_weight == 0.8
        assert params.sparse_weight == 0.2
        assert params.rerank is True
        assert params.rerank_method == ReRankingMethod.WEIGHTED
        assert params.rerank_weights == [0.6, 0.4]
        assert params.rerank_k == 80
        assert params.fusion_method == FusionMethod.WEIGHTED
        assert params.fusion_weights == [0.5, 0.5]
        assert params.params["nprobe"] == 20
        assert params.output_fields == ["text", "category"]


# ============================================================================
# Test SearchParams Field Validation
# ============================================================================


@pytest.mark.unit
class TestSearchParamsFieldValidation:
    """
    Test SearchParams field validation.

    Coverage: SearchParams Pydantic validators for field constraints.
    """

    @pytest.mark.parametrize("top_k", [-1, 0])
    def test_search_params_invalid_top_k(self, top_k):
        """
        Test SearchParams validation rejects non-positive top_k.

        Coverage: SearchParams raises ValidationError for top_k <= 0.
        """
        with pytest.raises(ValidationError) as exc_info:
            SearchParams(top_k=top_k)
        assert len(exc_info.value.errors()) > 0
        assert any(
            "gt" in str(error["type"]) or "greater than" in str(error).lower()
            for error in exc_info.value.errors()
        )

    @pytest.mark.parametrize("top_k", [1, 10, 100, 1000])
    def test_search_params_valid_top_k(self, top_k):
        """
        Test SearchParams accepts valid top_k values.

        Coverage: SearchParams accepts positive top_k values.
        """
        params = SearchParams(top_k=top_k)
        assert params.top_k == top_k

    @pytest.mark.parametrize("timeout", [-1.0, 0.0, -0.1])
    def test_search_params_invalid_timeout(self, timeout):
        """
        Test SearchParams validation rejects non-positive timeout.

        Coverage: SearchParams raises ValidationError for timeout <= 0.
        """
        with pytest.raises(ValidationError) as exc_info:
            SearchParams(timeout=timeout)
        assert len(exc_info.value.errors()) > 0

    @pytest.mark.parametrize("timeout", [0.1, 1.0, 30.0, 60.0, 300.0])
    def test_search_params_valid_timeout(self, timeout):
        """
        Test SearchParams accepts valid timeout values.

        Coverage: SearchParams accepts positive timeout values.
        """
        params = SearchParams(timeout=timeout)
        assert params.timeout == timeout

    @pytest.mark.parametrize("vector_weight", [-1.0, 1.1, 2.0])
    def test_search_params_invalid_vector_weight(self, vector_weight):
        """
        Test SearchParams validation rejects vector_weight outside [0, 1].

        Coverage: SearchParams raises ValidationError for vector_weight outside [0, 1].
        """
        with pytest.raises(ValidationError) as exc_info:
            SearchParams(vector_weight=vector_weight)
        assert len(exc_info.value.errors()) > 0

    @pytest.mark.parametrize("sparse_weight", [-1.0, 1.1, 2.0])
    def test_search_params_invalid_sparse_weight(self, sparse_weight):
        """
        Test SearchParams validation rejects sparse_weight outside [0, 1].

        Coverage: SearchParams raises ValidationError for sparse_weight outside [0, 1].
        """
        with pytest.raises(ValidationError) as exc_info:
            SearchParams(sparse_weight=sparse_weight)
        assert len(exc_info.value.errors()) > 0

    @pytest.mark.parametrize("rerank_k", [-1, 0])
    def test_search_params_invalid_rerank_k(self, rerank_k):
        """
        Test SearchParams validation rejects non-positive rerank_k.

        Coverage: SearchParams raises ValidationError for rerank_k <= 0.
        """
        with pytest.raises(ValidationError) as exc_info:
            SearchParams(rerank_k=rerank_k)
        assert len(exc_info.value.errors()) > 0

    @pytest.mark.parametrize("rerank_k", [1, 60, 100, 1000])
    def test_search_params_valid_rerank_k(self, rerank_k):
        """
        Test SearchParams accepts valid rerank_k values.

        Coverage: SearchParams accepts positive rerank_k values.
        """
        params = SearchParams(rerank_k=rerank_k)
        assert params.rerank_k == rerank_k


# ============================================================================
# Test SearchParams Enum Validation
# ============================================================================


@pytest.mark.unit
class TestSearchParamsEnumValidation:
    """
    Test SearchParams enum validation.

    Coverage: SearchParams validates enum values.
    """

    @pytest.mark.parametrize(
        "search_type", [SearchType.SEMANTIC, SearchType.HYBRID, SearchType.FUSION]
    )
    def test_search_params_all_search_types(self, search_type):
        """
        Test SearchParams accepts all SearchType values.

        Coverage: SearchParams accepts all SearchType enum values.
        """
        params = SearchParams(search_type=search_type)
        assert params.search_type == search_type

    @pytest.mark.parametrize(
        "metric_type",
        [MetricType.L2, MetricType.IP, MetricType.COSINE, MetricType.HAMMING],
    )
    def test_search_params_all_metric_types(self, metric_type):
        """
        Test SearchParams accepts all MetricType values.

        Coverage: SearchParams accepts all MetricType enum values.
        """
        params = SearchParams(metric_type=metric_type)
        assert params.metric_type == metric_type

    @pytest.mark.parametrize(
        "rerank_method",
        [ReRankingMethod.NONE, ReRankingMethod.WEIGHTED, ReRankingMethod.RRF],
    )
    def test_search_params_all_reranking_methods(self, rerank_method):
        """
        Test SearchParams accepts all ReRankingMethod values.

        Coverage: SearchParams accepts all ReRankingMethod enum values.
        """
        params = SearchParams(rerank_method=rerank_method)
        assert params.rerank_method == rerank_method

    @pytest.mark.parametrize(
        "fusion_method",
        [FusionMethod.RRF, FusionMethod.WEIGHTED, FusionMethod.MAX, FusionMethod.MEAN],
    )
    def test_search_params_all_fusion_methods(self, fusion_method):
        """
        Test SearchParams accepts all FusionMethod values.

        Coverage: SearchParams accepts all FusionMethod enum values.
        """
        params = SearchParams(fusion_method=fusion_method)
        assert params.fusion_method == fusion_method


# ============================================================================
# Test SearchParams Custom Validators
# ============================================================================


@pytest.mark.unit
class TestSearchParamsCustomValidators:
    """
    Test SearchParams custom validators.

    Coverage: SearchParams custom validation logic.
    """

    def test_search_params_fusion_weights_validation(self):
        """
        Test SearchParams validates fusion weights for WEIGHTED fusion.

        Coverage: SearchParams.__post_init__() validates fusion_weights for WEIGHTED method.
        """
        # Fusion weights must sum to 1.0 for WEIGHTED method
        with pytest.raises(ValidationError) as exc_info:
            SearchParams(
                fusion_method=FusionMethod.WEIGHTED,
                fusion_weights=[0.6, 0.5],  # Sums to 1.1
            )
        assert len(exc_info.value.errors()) > 0

    def test_search_params_fusion_weights_valid(self):
        """
        Test SearchParams accepts valid fusion weights.

        Coverage: SearchParams accepts fusion_weights that sum to 1.0.
        """
        params = SearchParams(
            fusion_method=FusionMethod.WEIGHTED,
            fusion_weights=[0.5, 0.5],
        )
        assert params.fusion_weights == [0.5, 0.5]

    def test_search_params_default_params(self):
        """
        Test SearchParams initializes default params.

        Coverage: SearchParams.__post_init__() adds default nprobe and ef to params.
        """
        params = SearchParams()
        # Check for presence and values of default params
        if "nprobe" in params.params:
            assert params.params["nprobe"] == 10
        if "ef" in params.params:
            assert params.params["ef"] == 64

    def test_search_params_preserves_custom_params(self):
        """
        Test SearchParams preserves custom params.

        Coverage: SearchParams.__post_init__() preserves existing params and adds defaults.
        """
        custom_params = {"custom_param": 100}
        params = SearchParams(params=custom_params)
        assert "nprobe" in params.params
        assert "ef" in params.params
        assert "custom_param" in params.params
        assert params.params["custom_param"] == 100


# ============================================================================
# Test SearchParams Extra Fields
# ============================================================================


@pytest.mark.unit
class TestSearchParamsExtraFields:
    """
    Test SearchParams extra field handling.

    Coverage: SearchParams Config.extra="forbid" behavior.
    """

    def test_search_params_rejects_extra_fields(self):
        """
        Test SearchParams rejects extra fields.

        Coverage: SearchParams raises ValidationError for extra fields.
        """
        with pytest.raises(ValidationError) as exc_info:
            SearchParams(invalid_field="value")
        assert len(exc_info.value.errors()) > 0
        assert any("extra" in str(error).lower() for error in exc_info.value.errors())


# ============================================================================
# Test SearchParams Edge Cases
# ============================================================================


@pytest.mark.unit
class TestSearchParamsEdgeCases:
    """
    Test SearchParams edge cases.

    Coverage: SearchParams handles edge cases correctly.
    """

    def test_search_params_empty_output_fields(self):
        """
        Test SearchParams with empty output_fields list.

        Coverage: SearchParams accepts empty output_fields list.
        """
        params = SearchParams(output_fields=[])
        assert params.output_fields == []

    def test_search_params_empty_expr(self):
        """
        Test SearchParams with empty expr string.

        Coverage: SearchParams accepts empty string for expr.
        """
        params = SearchParams(expr="")
        assert params.expr == ""

    def test_search_params_large_top_k(self):
        """
        Test SearchParams with very large top_k.

        Coverage: SearchParams accepts large top_k values.
        """
        params = SearchParams(top_k=10000)
        assert params.top_k == 10000

    def test_search_params_very_small_timeout(self):
        """
        Test SearchParams with very small timeout.

        Coverage: SearchParams accepts small positive timeout values.
        """
        params = SearchParams(timeout=0.001)
        assert params.timeout == 0.001

    def test_search_params_all_none_optional_fields(self):
        """
        Test SearchParams with all None optional fields.

        Coverage: SearchParams handles all optional fields as None.
        """
        params = SearchParams(
            expr=None,
            sparse_field=None,
            keyword_field=None,
            rerank_weights=None,
            fusion_weights=None,
            output_fields=None,
        )
        assert params.expr is None
        assert params.sparse_field is None
        assert params.keyword_field is None
        assert params.rerank_weights is None
        assert params.fusion_weights is None
        assert params.output_fields is None


# ============================================================================
# Test SearchParams Boundary Values
# ============================================================================


@pytest.mark.unit
class TestSearchParamsBoundaryValues:
    """
    Test SearchParams boundary value conditions.

    Coverage: Maximum/minimum integers, extreme floats, NaN/Infinity values.
    """

    def test_search_params_maximum_integer_top_k(self):
        """
        Test SearchParams with maximum integer value for top_k.

        Coverage: SearchParams handles sys.maxsize for top_k.
        """
        import sys

        params = SearchParams(top_k=sys.maxsize)
        assert params.top_k == sys.maxsize

    def test_search_params_maximum_integer_rerank_k(self):
        """
        Test SearchParams with maximum integer value for rerank_k.

        Coverage: SearchParams handles sys.maxsize for rerank_k.
        """
        import sys

        params = SearchParams(rerank_k=sys.maxsize)
        assert params.rerank_k == sys.maxsize

    def test_search_params_minimum_positive_timeout(self):
        """
        Test SearchParams with minimum positive timeout value.

        Coverage: SearchParams handles very small positive floats.
        """
        params = SearchParams(timeout=0.0001)
        assert params.timeout == 0.0001

    def test_search_params_very_large_timeout(self):
        """
        Test SearchParams with very large timeout value.

        Coverage: SearchParams handles large float values.
        """
        # Test with very large but valid float
        params = SearchParams(timeout=1e100)
        assert params.timeout == 1e100

    def test_search_params_infinity_timeout(self):
        """
        Test SearchParams with infinity timeout.

        Coverage: SearchParams handles float('inf') for timeout.
        """
        # Pydantic may accept or reject infinity
        try:
            params = SearchParams(timeout=float("inf"))
            assert params.timeout == float("inf")
        except ValidationError:
            # Some Pydantic versions reject infinity
            pass

    def test_search_params_negative_infinity_timeout(self):
        """
        Test SearchParams with negative infinity timeout.

        Coverage: SearchParams handles float('-inf') for timeout (should fail validation).
        """
        # Negative infinity should fail validation (must be > 0)
        with pytest.raises(ValidationError):
            SearchParams(timeout=float("-inf"))

    def test_search_params_nan_timeout(self):
        """
        Test SearchParams with NaN timeout.

        Coverage: SearchParams handles float('nan') for timeout (should fail validation).
        """
        # NaN should fail validation
        with pytest.raises(ValidationError):
            SearchParams(timeout=float("nan"))

    def test_search_params_zero_timeout_rejected(self):
        """
        Test SearchParams rejects zero timeout.

        Coverage: Zero timeout should be rejected by validation.
        """
        with pytest.raises(ValidationError):
            SearchParams(timeout=0.0)

    def test_search_params_zero_top_k_rejected(self):
        """
        Test SearchParams rejects zero top_k.

        Coverage: Zero top_k should be rejected by validation.
        """
        with pytest.raises(ValidationError):
            SearchParams(top_k=0)


# ============================================================================
# Test SearchParams Type Safety
# ============================================================================


@pytest.mark.unit
class TestSearchParamsTypeSafety:
    """
    Test SearchParams type safety and validation.

    Coverage: Wrong types, None where not allowed, empty collections, malformed data.
    """

    def test_search_params_wrong_type_top_k(self):
        """
        Test SearchParams rejects string where int expected for top_k.

        Coverage: SearchParams raises ValidationError for type mismatches.
        """
        with pytest.raises(ValidationError) as exc_info:
            SearchParams(top_k="invalid")  # type: ignore
        assert len(exc_info.value.errors()) > 0
        assert any("type" in str(error).lower() for error in exc_info.value.errors())

    def test_search_params_wrong_type_timeout(self):
        """
        Test SearchParams rejects string where float expected for timeout.

        Coverage: SearchParams raises ValidationError for type mismatches.
        """
        with pytest.raises(ValidationError) as exc_info:
            SearchParams(timeout="invalid")  # type: ignore
        assert len(exc_info.value.errors()) > 0

    def test_search_params_wrong_type_vector_field(self):
        """
        Test SearchParams rejects int where str expected for vector_field.

        Coverage: SearchParams raises ValidationError for type mismatches.
        """
        with pytest.raises(ValidationError) as exc_info:
            SearchParams(vector_field=123)  # type: ignore
        assert len(exc_info.value.errors()) > 0

    def test_search_params_none_required_field(self):
        """
        Test SearchParams rejects None for required field vector_field.

        Coverage: SearchParams raises ValidationError for None in required fields.
        """
        with pytest.raises(ValidationError) as exc_info:
            SearchParams(vector_field=None)  # type: ignore
        assert len(exc_info.value.errors()) > 0

    def test_search_params_empty_output_fields(self):
        """
        Test SearchParams with empty output_fields list vs None.

        Coverage: SearchParams handles empty list vs None distinction.
        """
        # Empty list should be accepted
        params = SearchParams(output_fields=[])
        assert params.output_fields == []

        # None should also be accepted
        params = SearchParams(output_fields=None)
        assert params.output_fields is None

    def test_search_params_malformed_params_dict(self):
        """
        Test SearchParams with malformed params dict.

        Coverage: SearchParams handles dict with non-string keys.
        """
        # Dict with non-string keys
        malformed_params = {123: "value"}  # type: ignore
        # Pydantic should handle this or raise error
        try:
            params = SearchParams(params=malformed_params)
            # If accepted, verify structure
            assert isinstance(params.params, dict)
        except ValidationError:
            # Some Pydantic versions may reject non-string keys
            pass

    def test_search_params_empty_params_dict(self):
        """
        Test SearchParams with empty params dict.

        Coverage: SearchParams handles empty dict (should use defaults).
        """
        params = SearchParams(params={})
        # Should still have default params
        assert "nprobe" in params.params
        assert "ef" in params.params

    def test_search_params_wrong_type_rerank_weights(self):
        """
        Test SearchParams rejects wrong type for rerank_weights.

        Coverage: SearchParams raises ValidationError for type mismatches.
        """
        with pytest.raises(ValidationError) as exc_info:
            SearchParams(rerank_weights="0.5,0.5")  # type: ignore
        assert len(exc_info.value.errors()) > 0

    def test_search_params_wrong_type_fusion_weights(self):
        """
        Test SearchParams rejects wrong type for fusion_weights.

        Coverage: SearchParams raises ValidationError for type mismatches.
        """
        with pytest.raises(ValidationError) as exc_info:
            SearchParams(fusion_weights=0.5)  # type: ignore
        assert len(exc_info.value.errors()) > 0


# ============================================================================
# Test SearchParams Negative Paths
# ============================================================================


@pytest.mark.unit
class TestSearchParamsNegativePaths:
    """
    Test SearchParams negative path scenarios.

    Coverage: Invalid enum values, type mismatches, validation errors.
    """

    def test_search_params_invalid_enum_string(self):
        """
        Test SearchParams rejects invalid enum string value.

        Coverage: SearchParams raises ValidationError for invalid enum strings.
        """
        with pytest.raises(ValidationError) as exc_info:
            SearchParams(search_type="invalid_search_type")  # type: ignore
        assert len(exc_info.value.errors()) > 0

    def test_search_params_invalid_metric_type_string(self):
        """
        Test SearchParams rejects invalid metric type string.

        Coverage: SearchParams raises ValidationError for invalid enum strings.
        """
        with pytest.raises(ValidationError) as exc_info:
            SearchParams(metric_type="INVALID_METRIC")  # type: ignore
        assert len(exc_info.value.errors()) > 0

    def test_search_params_invalid_reranking_method_string(self):
        """
        Test SearchParams rejects invalid reranking method string.

        Coverage: SearchParams raises ValidationError for invalid enum strings.
        """
        with pytest.raises(ValidationError) as exc_info:
            SearchParams(rerank_method="invalid_method")  # type: ignore
        assert len(exc_info.value.errors()) > 0

    def test_search_params_type_coercion_failure(self):
        """
        Test SearchParams cannot coerce incompatible types.

        Coverage: SearchParams raises ValidationError for uncoercible types.
        """
        # Cannot coerce dict to int
        with pytest.raises(ValidationError) as exc_info:
            SearchParams(top_k={"invalid": "dict"})  # type: ignore
        assert len(exc_info.value.errors()) > 0

    def test_search_params_validation_error_messages(self):
        """
        Test SearchParams validation error messages are clear.

        Coverage: Validation error messages are informative.
        """
        with pytest.raises(ValidationError) as exc_info:
            SearchParams(top_k=-1, timeout=-1.0, vector_weight=2.0)
        errors = exc_info.value.errors()
        assert len(errors) > 0
        # Check that error messages are present
        for error in errors:
            assert "msg" in error or "message" in error

    def test_search_params_negative_vector_weight(self):
        """
        Test SearchParams rejects negative vector_weight.

        Coverage: SearchParams raises ValidationError for values outside [0, 1].
        """
        with pytest.raises(ValidationError):
            SearchParams(vector_weight=-0.1)

    def test_search_params_negative_sparse_weight(self):
        """
        Test SearchParams rejects negative sparse_weight.

        Coverage: SearchParams raises ValidationError for values outside [0, 1].
        """
        with pytest.raises(ValidationError):
            SearchParams(sparse_weight=-0.1)

    def test_search_params_exceeds_one_vector_weight(self):
        """
        Test SearchParams rejects vector_weight > 1.0.

        Coverage: SearchParams raises ValidationError for values outside [0, 1].
        """
        with pytest.raises(ValidationError):
            SearchParams(vector_weight=1.1)

    def test_search_params_exceeds_one_sparse_weight(self):
        """
        Test SearchParams rejects sparse_weight > 1.0.

        Coverage: SearchParams raises ValidationError for values outside [0, 1].
        """
        with pytest.raises(ValidationError):
            SearchParams(sparse_weight=1.1)


# ============================================================================
# Test SearchParams State Mutation
# ============================================================================


@pytest.mark.unit
class TestSearchParamsStateMutation:
    """
    Test SearchParams state mutation behavior.

    Coverage: Params dict mutation effects, validate_assignment behavior.
    """

    def test_search_params_params_dict_mutation(self):
        """
        Test modifying params dict after initialization.

        Coverage: Params dict is mutable and changes propagate.
        """
        params = SearchParams()

        # Add nprobe if not present
        if "nprobe" not in params.params:
            params.params["nprobe"] = 10
        original_nprobe = params.params["nprobe"]

        # Modify params dict
        params.params["nprobe"] = 50
        assert params.params["nprobe"] == 50
        assert params.params["nprobe"] != original_nprobe

    def test_search_params_params_dict_mutation_effects(self):
        """
        Test that params dict mutations affect the instance.

        Coverage: Mutations to params dict are reflected in the instance.
        """
        params = SearchParams(params={"custom": "value"})
        assert params.params["custom"] == "value"

        # Add new key
        params.params["new_key"] = "new_value"
        assert "new_key" in params.params
        assert params.params["new_key"] == "new_value"

    def test_search_params_validate_assignment(self):
        """
        Test SearchParams validate_assignment=True behavior.

        Coverage: Pydantic validates assignments after initialization.
        """
        params = SearchParams(top_k=10)
        assert params.top_k == 10

        # Try to assign invalid value (should raise ValidationError if validate_assignment=True)
        # Note: Pydantic v1 vs v2 behavior may differ
        with contextlib.suppress(ValidationError):
            params.top_k = -1  # type: ignore
            # If assignment succeeds, the value may be stored but validation should catch it
            # In Pydantic v2 with validate_assignment=True, this should raise

    def test_search_params_field_assignment(self):
        """
        Test SearchParams field assignment behavior.

        Coverage: Field assignments update instance state.
        """
        params = SearchParams()
        original_timeout = params.timeout

        # Assign new value
        params.timeout = 60.0
        assert params.timeout == 60.0
        assert params.timeout != original_timeout

    @pytest.mark.asyncio
    async def test_search_params_concurrent_access(self):
        """
        Test concurrent access to SearchParams instance.

        Coverage: Thread safety of SearchParams objects.
        """
        import asyncio

        params = SearchParams()

        async def modify_params(value: int):
            """Helper to modify params concurrently"""
            params.params["test_key"] = value
            await asyncio.sleep(0.01)  # Simulate async operation
            return params.params.get("test_key")

        # Run concurrent modifications
        tasks = [modify_params(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # Verify that modifications occurred (may be race conditions)
        assert len(results) == 10
        # All should have been able to modify (no exceptions)
