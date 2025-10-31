"""
Comprehensive unit tests for IndexValidator.

This module provides systematic testing of the IndexValidator class,
including validation, optimization, and memory estimation methods.
"""

import pytest

from milvus_ops.collection_operations import DataType, IndexType, MetricType
from milvus_ops.index_operations.core.validator import IndexValidator
from milvus_ops.index_operations.index_ops_exceptions import (
    IndexParameterError,
    IndexTypeError,
)
from milvus_ops.index_operations.models.parameters import (
    ANNOYParams,
    HNSWParams,
    IvfFlatParams,
    IvfPQParams,
    IvfSQ8Params,
)

# ============================================================================
# Test IndexValidator Initialization
# ============================================================================


@pytest.mark.unit
class TestIndexValidatorInitialization:
    """
    Test IndexValidator initialization.

    Coverage: IndexValidator constructor.
    """

    def test_validator_initialization(self):
        """
        Test IndexValidator initialization.

        Coverage: IndexValidator can be instantiated.
        """
        validator = IndexValidator()
        assert isinstance(validator, IndexValidator)


# ============================================================================
# Test IndexValidator validate_index_params
# ============================================================================


@pytest.mark.unit
class TestIndexValidatorValidateIndexParams:
    """
    Test IndexValidator.validate_index_params() method.

    Coverage: IndexValidator parameter validation.
    """

    def test_validate_index_params_ivf_flat(self):
        """
        Test validate_index_params() for IVF_FLAT.

        Coverage: validate_index_params() validates IVF_FLAT parameters.
        """
        validator = IndexValidator()
        params = validator.validate_index_params(
            index_type=IndexType.IVF_FLAT,
            metric_type=MetricType.L2,
            dimension=128,
            params={"nlist": 1024},
        )
        assert isinstance(params, IvfFlatParams)
        assert params.nlist == 1024

    def test_validate_index_params_hnsw(self):
        """
        Test validate_index_params() for HNSW.

        Coverage: validate_index_params() validates HNSW parameters.
        """
        validator = IndexValidator()
        params = validator.validate_index_params(
            index_type=IndexType.HNSW,
            metric_type=MetricType.COSINE,
            dimension=128,
            params={"M": 16, "efConstruction": 200},
        )
        assert isinstance(params, HNSWParams)
        assert params.M == 16
        assert params.efConstruction == 200

    def test_validate_index_params_string_types(self):
        """
        Test validate_index_params() with string types.

        Coverage: validate_index_params() converts string types to enums.
        """
        validator = IndexValidator()
        params = validator.validate_index_params(
            index_type="IVF_FLAT",
            metric_type="L2",
            dimension=128,
            params={"nlist": 1024},
        )
        assert isinstance(params, IvfFlatParams)

    def test_validate_index_params_invalid_index_type(self):
        """
        Test validate_index_params() with invalid index type.

        Coverage: validate_index_params() raises IndexTypeError for invalid type.
        """
        validator = IndexValidator()
        with pytest.raises(IndexTypeError):
            validator.validate_index_params(
                index_type="INVALID_TYPE",
                metric_type=MetricType.L2,
                dimension=128,
            )

    def test_validate_index_params_invalid_metric_type(self):
        """
        Test validate_index_params() with invalid metric type.

        Coverage: validate_index_params() raises IndexParameterError for invalid metric.
        """
        validator = IndexValidator()
        with pytest.raises(IndexParameterError):
            validator.validate_index_params(
                index_type=IndexType.IVF_FLAT,
                metric_type="INVALID_METRIC",
                dimension=128,
            )

    def test_validate_index_params_dict_params(self):
        """
        Test validate_index_params() with dict params.

        Coverage: validate_index_params() accepts dict parameters.
        """
        validator = IndexValidator()
        params = validator.validate_index_params(
            index_type=IndexType.IVF_FLAT,
            metric_type=MetricType.L2,
            dimension=128,
            params={"nlist": 1024},
        )
        assert isinstance(params, IvfFlatParams)

    def test_validate_index_params_none_params(self):
        """
        Test validate_index_params() with None params.

        Coverage: validate_index_params() uses defaults when params is None.
        """
        validator = IndexValidator()
        params = validator.validate_index_params(
            index_type=IndexType.IVF_FLAT,
            metric_type=MetricType.L2,
            dimension=128,
            params=None,
        )
        assert isinstance(params, IvfFlatParams)
        # Should use default nlist
        assert params.nlist == 1024

    def test_validate_index_params_incompatible_metric(self):
        """
        Test validate_index_params() with incompatible metric type.

        Coverage: validate_index_params() raises error for incompatible metric.
        """
        validator = IndexValidator()
        with pytest.raises(IndexParameterError):
            validator.validate_index_params(
                index_type=IndexType.IVF_FLAT,
                metric_type=MetricType.HAMMING,  # Incompatible with IVF_FLAT
                dimension=128,
            )


# ============================================================================
# Test IndexValidator validate_metric_compatibility
# ============================================================================


@pytest.mark.unit
class TestIndexValidatorValidateMetricCompatibility:
    """
    Test IndexValidator.validate_metric_compatibility() method.

    Coverage: IndexValidator metric compatibility validation.
    """

    def test_validate_metric_compatibility_valid(self):
        """
        Test validate_metric_compatibility() with valid combinations.

        Coverage: validate_metric_compatibility() accepts valid index/metric combinations.
        """
        validator = IndexValidator()
        # Should not raise
        validator.validate_metric_compatibility(IndexType.IVF_FLAT, MetricType.L2)
        validator.validate_metric_compatibility(IndexType.HNSW, MetricType.COSINE)
        validator.validate_metric_compatibility(IndexType.IVF_FLAT, MetricType.IP)

    def test_validate_metric_compatibility_invalid(self):
        """
        Test validate_metric_compatibility() with invalid combinations.

        Coverage: validate_metric_compatibility() raises error for incompatible combinations.
        """
        validator = IndexValidator()
        with pytest.raises(IndexParameterError):
            validator.validate_metric_compatibility(IndexType.IVF_FLAT, MetricType.HAMMING)

    @pytest.mark.parametrize(
        "index_type,metric_type",
        [
            (IndexType.IVF_FLAT, MetricType.L2),
            (IndexType.IVF_FLAT, MetricType.IP),
            (IndexType.IVF_FLAT, MetricType.COSINE),
            (IndexType.HNSW, MetricType.L2),
            (IndexType.HNSW, MetricType.IP),
            (IndexType.HNSW, MetricType.COSINE),
            (IndexType.IVF_SQ8, MetricType.L2),
            (IndexType.IVF_SQ8, MetricType.IP),
            (IndexType.IVF_SQ8, MetricType.COSINE),
        ],
    )
    def test_validate_metric_compatibility_parametrized(self, index_type, metric_type):
        """
        Test validate_metric_compatibility() with parameterized valid combinations.

        Coverage: validate_metric_compatibility() accepts multiple valid combinations.
        """
        validator = IndexValidator()
        # Should not raise
        validator.validate_metric_compatibility(index_type, metric_type)


# ============================================================================
# Test IndexValidator validate_field_type_compatibility
# ============================================================================


@pytest.mark.unit
class TestIndexValidatorValidateFieldTypeCompatibility:
    """
    Test IndexValidator.validate_field_type_compatibility() method.

    Coverage: IndexValidator field type compatibility validation.
    """

    def test_validate_field_type_compatibility_valid(self):
        """
        Test validate_field_type_compatibility() with valid combinations.

        Coverage: validate_field_type_compatibility() accepts valid index/field combinations.
        """
        validator = IndexValidator()
        # Should not raise
        validator.validate_field_type_compatibility(IndexType.IVF_FLAT, DataType.FLOAT_VECTOR)
        validator.validate_field_type_compatibility(IndexType.HNSW, DataType.FLOAT_VECTOR)
        validator.validate_field_type_compatibility(IndexType.BIN_FLAT, DataType.BINARY_VECTOR)

    def test_validate_field_type_compatibility_invalid(self):
        """
        Test validate_field_type_compatibility() with invalid combinations.

        Coverage: validate_field_type_compatibility() raises error for incompatible combinations.
        """
        validator = IndexValidator()
        with pytest.raises(IndexParameterError):
            validator.validate_field_type_compatibility(IndexType.IVF_FLAT, DataType.BINARY_VECTOR)

    def test_validate_field_type_compatibility_binary(self):
        """
        Test validate_field_type_compatibility() with binary index and vector.

        Coverage: validate_field_type_compatibility() accepts binary index with binary vector.
        """
        validator = IndexValidator()
        # Should not raise
        validator.validate_field_type_compatibility(IndexType.BIN_FLAT, DataType.BINARY_VECTOR)


# ============================================================================
# Test IndexValidator validate_dimension_compatibility
# ============================================================================


@pytest.mark.unit
class TestIndexValidatorValidateDimensionCompatibility:
    """
    Test IndexValidator.validate_dimension_compatibility() method.

    Coverage: IndexValidator dimension compatibility validation.
    """

    def test_validate_dimension_compatibility_no_constraints(self):
        """
        Test validate_dimension_compatibility() for index types without constraints.

        Coverage: validate_dimension_compatibility() accepts any dimension for unconstrained types.
        """
        validator = IndexValidator()
        # Should not raise for index types without dimension constraints
        validator.validate_dimension_compatibility(IndexType.IVF_FLAT, 128)
        validator.validate_dimension_compatibility(IndexType.HNSW, 512)
        validator.validate_dimension_compatibility(IndexType.IVF_FLAT, 1)

    def test_validate_dimension_compatibility_bin_flat_divisible_by_8(self):
        """
        Test validate_dimension_compatibility() for BIN_FLAT divisible by 8 requirement.

        Coverage: validate_dimension_compatibility() validates BIN_FLAT dimension divisible by 8.
        """
        validator = IndexValidator()
        # Valid: divisible by 8
        validator.validate_dimension_compatibility(IndexType.BIN_FLAT, 8)
        validator.validate_dimension_compatibility(IndexType.BIN_FLAT, 16)
        validator.validate_dimension_compatibility(IndexType.BIN_FLAT, 256)

    def test_validate_dimension_compatibility_bin_flat_not_divisible(self):
        """
        Test validate_dimension_compatibility() for BIN_FLAT with dimension not divisible by 8.

        Coverage: validate_dimension_compatibility() raises error for non-divisible dimension.
        """
        validator = IndexValidator()
        with pytest.raises(IndexParameterError):
            validator.validate_dimension_compatibility(IndexType.BIN_FLAT, 7)
        with pytest.raises(IndexParameterError):
            validator.validate_dimension_compatibility(IndexType.BIN_FLAT, 9)

    def test_validate_dimension_compatibility_bin_flat_minimum(self):
        """
        Test validate_dimension_compatibility() for BIN_FLAT minimum dimension.

        Coverage: validate_dimension_compatibility() validates minimum dimension for BIN_FLAT.
        """
        validator = IndexValidator()
        # Minimum is 8
        validator.validate_dimension_compatibility(IndexType.BIN_FLAT, 8)
        with pytest.raises(IndexParameterError):
            validator.validate_dimension_compatibility(IndexType.BIN_FLAT, 7)


# ============================================================================
# Test IndexValidator optimize_parameters
# ============================================================================


@pytest.mark.unit
class TestIndexValidatorOptimizeParameters:
    """
    Test IndexValidator.optimize_parameters() method.

    Coverage: IndexValidator parameter optimization.
    """

    def test_optimize_parameters_ivf_flat(self):
        """
        Test optimize_parameters() for IVF_FLAT.

        Coverage: optimize_parameters() suggests optimal IVF_FLAT parameters.
        """
        validator = IndexValidator()
        params = validator.optimize_parameters(IndexType.IVF_FLAT, dimension=128, row_count=100000)
        assert isinstance(params, IvfFlatParams)
        assert params.nlist >= 1

    def test_optimize_parameters_ivf_flat_with_row_count(self):
        """
        Test optimize_parameters() for IVF_FLAT with row_count.

        Coverage: optimize_parameters() scales nlist based on row_count.
        """
        validator = IndexValidator()
        params_small = validator.optimize_parameters(
            IndexType.IVF_FLAT, dimension=128, row_count=1000
        )
        params_large = validator.optimize_parameters(
            IndexType.IVF_FLAT, dimension=128, row_count=10000000
        )
        # Larger dataset should have larger nlist (up to max)
        assert params_large.nlist >= params_small.nlist

    def test_optimize_parameters_ivf_sq8(self):
        """
        Test optimize_parameters() for IVF_SQ8.

        Coverage: optimize_parameters() suggests optimal IVF_SQ8 parameters.
        """
        validator = IndexValidator()
        params = validator.optimize_parameters(IndexType.IVF_SQ8, dimension=128, row_count=100000)
        assert isinstance(params, IvfSQ8Params)

    def test_optimize_parameters_ivf_pq(self):
        """
        Test optimize_parameters() for IVF_PQ.

        Coverage: optimize_parameters() suggests optimal IVF_PQ parameters.
        """
        validator = IndexValidator()
        params = validator.optimize_parameters(IndexType.IVF_PQ, dimension=128, row_count=100000)
        assert isinstance(params, IvfPQParams)
        assert params.m > 0
        # m should divide dimension
        assert 128 % params.m == 0

    def test_optimize_parameters_hnsw(self):
        """
        Test optimize_parameters() for HNSW.

        Coverage: optimize_parameters() suggests optimal HNSW parameters.
        """
        validator = IndexValidator()
        params = validator.optimize_parameters(IndexType.HNSW, dimension=128, row_count=100000)
        assert isinstance(params, HNSWParams)
        assert params.M >= 4
        assert params.efConstruction >= 8

    def test_optimize_parameters_hnsw_large_dimension(self):
        """
        Test optimize_parameters() for HNSW with large dimension.

        Coverage: optimize_parameters() adjusts M for large dimensions.
        """
        validator = IndexValidator()
        validator.optimize_parameters(IndexType.HNSW, dimension=128, row_count=100000)
        params_large = validator.optimize_parameters(
            IndexType.HNSW, dimension=2000, row_count=100000
        )
        # For very large dimensions, M might be reduced
        assert params_large.M >= 4

    def test_optimize_parameters_hnsw_large_dataset(self):
        """
        Test optimize_parameters() for HNSW with large dataset.

        Coverage: optimize_parameters() increases efConstruction for large datasets.
        """
        validator = IndexValidator()
        params_small = validator.optimize_parameters(IndexType.HNSW, dimension=128, row_count=1000)
        params_large = validator.optimize_parameters(
            IndexType.HNSW, dimension=128, row_count=2000000
        )
        # Large dataset should have higher efConstruction
        assert params_large.efConstruction >= params_small.efConstruction

    def test_optimize_parameters_annoy(self):
        """
        Test optimize_parameters() for ANNOY.

        Coverage: optimize_parameters() suggests optimal ANNOY parameters.
        """
        validator = IndexValidator()
        params = validator.optimize_parameters(IndexType.ANNOY, dimension=128, row_count=100000)
        assert isinstance(params, ANNOYParams)
        assert params.n_trees >= 1

    def test_optimize_parameters_without_row_count(self):
        """
        Test optimize_parameters() without row_count.

        Coverage: optimize_parameters() uses defaults when row_count is None.
        """
        validator = IndexValidator()
        params = validator.optimize_parameters(IndexType.IVF_FLAT, dimension=128, row_count=None)
        assert isinstance(params, IvfFlatParams)


# ============================================================================
# Test IndexValidator estimate_memory_usage
# ============================================================================


@pytest.mark.unit
class TestIndexValidatorEstimateMemoryUsage:
    """
    Test IndexValidator.estimate_memory_usage() method.

    Coverage: IndexValidator memory estimation.
    """

    def test_estimate_memory_usage_flat(self):
        """
        Test estimate_memory_usage() for FLAT index.

        Coverage: estimate_memory_usage() estimates FLAT index memory.
        """
        validator = IndexValidator()
        memory = validator.estimate_memory_usage(IndexType.FLAT, dimension=128, row_count=10000)
        # FLAT stores vectors as-is (float32 = 4 bytes per dimension)
        expected = 10000 * 128 * 4
        assert memory == expected

    def test_estimate_memory_usage_ivf_flat(self):
        """
        Test estimate_memory_usage() for IVF_FLAT index.

        Coverage: estimate_memory_usage() estimates IVF_FLAT index memory.
        """
        validator = IndexValidator()
        memory = validator.estimate_memory_usage(IndexType.IVF_FLAT, dimension=128, row_count=10000)
        # IVF_FLAT adds ~5% overhead
        base_memory = 10000 * 128 * 4
        assert memory > base_memory
        assert memory <= int(base_memory * 1.1)

    def test_estimate_memory_usage_ivf_sq8(self):
        """
        Test estimate_memory_usage() for IVF_SQ8 index.

        Coverage: estimate_memory_usage() estimates IVF_SQ8 index memory.
        """
        validator = IndexValidator()
        memory = validator.estimate_memory_usage(IndexType.IVF_SQ8, dimension=128, row_count=10000)
        # IVF_SQ8 compresses to 1 byte per dimension
        compressed = 10000 * 128
        base_memory = 10000 * 128 * 4
        overhead = int(base_memory * 0.05)
        expected = compressed + overhead
        assert memory == expected

    def test_estimate_memory_usage_ivf_pq(self):
        """
        Test estimate_memory_usage() for IVF_PQ index.

        Coverage: estimate_memory_usage() estimates IVF_PQ index memory.
        """
        validator = IndexValidator()
        params = IvfPQParams(nlist=1024, m=8, nbits=8)
        memory = validator.estimate_memory_usage(
            IndexType.IVF_PQ, dimension=128, row_count=10000, params=params
        )
        # IVF_PQ has significant compression
        base_memory = 10000 * 128 * 4
        assert memory < base_memory
        assert memory > 0

    def test_estimate_memory_usage_hnsw(self):
        """
        Test estimate_memory_usage() for HNSW index.

        Coverage: estimate_memory_usage() estimates HNSW index memory.
        """
        validator = IndexValidator()
        params = HNSWParams(M=16, efConstruction=200)
        memory = validator.estimate_memory_usage(
            IndexType.HNSW, dimension=128, row_count=10000, params=params
        )
        # HNSW has overhead for graph connections
        base_memory = 10000 * 128 * 4
        assert memory > base_memory
        # Should account for M connections
        assert memory >= base_memory + (10000 * 16 * 8)

    def test_estimate_memory_usage_without_params(self):
        """
        Test estimate_memory_usage() without params.

        Coverage: estimate_memory_usage() uses rough estimates when params are None.
        """
        validator = IndexValidator()
        memory = validator.estimate_memory_usage(
            IndexType.HNSW, dimension=128, row_count=10000, params=None
        )
        # Should still provide estimate
        assert memory > 0

    def test_estimate_memory_usage_zero_rows(self):
        """
        Test estimate_memory_usage() with zero rows.

        Coverage: estimate_memory_usage() handles zero rows.
        """
        validator = IndexValidator()
        memory = validator.estimate_memory_usage(IndexType.FLAT, dimension=128, row_count=0)
        assert memory == 0

    def test_estimate_memory_usage_large_dimension(self):
        """
        Test estimate_memory_usage() with large dimension.

        Coverage: estimate_memory_usage() handles large dimensions.
        """
        validator = IndexValidator()
        memory = validator.estimate_memory_usage(IndexType.FLAT, dimension=2048, row_count=10000)
        expected = 10000 * 2048 * 4
        assert memory == expected
