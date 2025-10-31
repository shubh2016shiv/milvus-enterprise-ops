"""
Comprehensive unit tests for index parameters.

This module provides systematic testing of all parameter classes in the
index_operations.models.parameters module, including IvfFlatParams,
IvfSQ8Params, IvfPQParams, HNSWParams, ANNOYParams, and factory functions.
"""

import pytest

from milvus_ops.collection_operations import IndexType
from milvus_ops.index_operations.models.parameters import (
    ANNOYParams,
    HNSWParams,
    IvfFlatParams,
    IvfPQParams,
    IvfSQ8Params,
    create_index_params,
    get_default_params,
)

# ============================================================================
# Test IndexParams Base Class
# ============================================================================


@pytest.mark.unit
class TestIndexParamsBase:
    """
    Test IndexParams base class.

    Coverage: IndexParams base class methods and properties.
    """

    def test_index_params_to_dict(self):
        """
        Test IndexParams to_dict() method.

        Coverage: IndexParams.to_dict() converts to dictionary.
        """
        params = IvfFlatParams(nlist=1024)
        params_dict = params.to_dict()
        assert isinstance(params_dict, dict)
        assert params_dict["nlist"] == 1024

    def test_index_params_from_dict(self):
        """
        Test IndexParams from_dict() method.

        Coverage: IndexParams.from_dict() creates from dictionary.
        """
        params_dict = {"nlist": 1024}
        params = IvfFlatParams.from_dict(params_dict)
        assert isinstance(params, IvfFlatParams)
        assert params.nlist == 1024


# ============================================================================
# Test IvfFlatParams
# ============================================================================


@pytest.mark.unit
class TestIvfFlatParams:
    """
    Test IvfFlatParams class.

    Coverage: IvfFlatParams initialization and validation.
    """

    def test_ivf_flat_params_initialization(self):
        """
        Test IvfFlatParams initialization.

        Coverage: IvfFlatParams can be created with nlist.
        """
        params = IvfFlatParams(nlist=1024)
        assert params.nlist == 1024
        assert params.index_type == IndexType.IVF_FLAT

    def test_ivf_flat_params_default(self):
        """
        Test IvfFlatParams default value.

        Coverage: IvfFlatParams uses default nlist=1024.
        """
        params = IvfFlatParams()
        assert params.nlist == 1024

    def test_ivf_flat_params_validation_minimum(self):
        """
        Test IvfFlatParams validation for minimum nlist.

        Coverage: IvfFlatParams validates nlist >= 1.
        """
        params = IvfFlatParams(nlist=1)
        assert params.nlist == 1

    def test_ivf_flat_params_validation_large(self):
        """
        Test IvfFlatParams validation for large nlist.

        Coverage: IvfFlatParams accepts large nlist values.
        """
        params = IvfFlatParams(nlist=16384)
        assert params.nlist == 16384


# ============================================================================
# Test IvfSQ8Params
# ============================================================================


@pytest.mark.unit
class TestIvfSQ8Params:
    """
    Test IvfSQ8Params class.

    Coverage: IvfSQ8Params initialization and validation.
    """

    def test_ivf_sq8_params_initialization(self):
        """
        Test IvfSQ8Params initialization.

        Coverage: IvfSQ8Params can be created with nlist.
        """
        params = IvfSQ8Params(nlist=1024)
        assert params.nlist == 1024
        assert params.index_type == IndexType.IVF_SQ8

    def test_ivf_sq8_params_default(self):
        """
        Test IvfSQ8Params default value.

        Coverage: IvfSQ8Params uses default nlist=1024.
        """
        params = IvfSQ8Params()
        assert params.nlist == 1024


# ============================================================================
# Test IvfPQParams
# ============================================================================


@pytest.mark.unit
class TestIvfPQParams:
    """
    Test IvfPQParams class.

    Coverage: IvfPQParams initialization and validation.
    """

    def test_ivf_pq_params_initialization(self):
        """
        Test IvfPQParams initialization.

        Coverage: IvfPQParams can be created with nlist, m, nbits.
        """
        params = IvfPQParams(nlist=1024, m=8, nbits=8)
        assert params.nlist == 1024
        assert params.m == 8
        assert params.nbits == 8
        assert params.index_type == IndexType.IVF_PQ

    def test_ivf_pq_params_default(self):
        """
        Test IvfPQParams default values.

        Coverage: IvfPQParams uses default values.
        """
        params = IvfPQParams()
        assert params.nlist == 1024
        assert params.m == 8
        assert params.nbits == 8

    def test_ivf_pq_params_nbits_validation(self):
        """
        Test IvfPQParams nbits validation.

        Coverage: IvfPQParams validates nbits <= 8.
        """
        params = IvfPQParams(nbits=8)
        assert params.nbits == 8

    def test_ivf_pq_params_nbits_too_large(self):
        """
        Test IvfPQParams nbits validation for values > 8.

        Coverage: IvfPQParams raises error for nbits > 8.
        """
        with pytest.raises(ValueError):
            IvfPQParams(nbits=9)


# ============================================================================
# Test HNSWParams
# ============================================================================


@pytest.mark.unit
class TestHNSWParams:
    """
    Test HNSWParams class.

    Coverage: HNSWParams initialization and validation.
    """

    def test_hnsw_params_initialization(self):
        """
        Test HNSWParams initialization.

        Coverage: HNSWParams can be created with M and efConstruction.
        """
        params = HNSWParams(M=16, efConstruction=200)
        assert params.M == 16
        assert params.efConstruction == 200
        assert params.index_type == IndexType.HNSW

    def test_hnsw_params_default(self):
        """
        Test HNSWParams default values.

        Coverage: HNSWParams uses default values.
        """
        params = HNSWParams()
        assert params.M == 16
        assert params.efConstruction == 200

    def test_hnsw_params_m_validation_minimum(self):
        """
        Test HNSWParams M validation for minimum value.

        Coverage: HNSWParams validates M >= 4.
        """
        params = HNSWParams(M=4)
        assert params.M == 4

    def test_hnsw_params_m_validation_maximum(self):
        """
        Test HNSWParams M validation for maximum value.

        Coverage: HNSWParams validates M <= 64.
        """
        params = HNSWParams(M=64)
        assert params.M == 64

    def test_hnsw_params_m_validation_too_small(self):
        """
        Test HNSWParams M validation for values < 4.

        Coverage: HNSWParams raises error for M < 4.
        """
        with pytest.raises(ValueError):
            HNSWParams(M=3)

    def test_hnsw_params_ef_construction_validation_minimum(self):
        """
        Test HNSWParams efConstruction validation for minimum value.

        Coverage: HNSWParams validates efConstruction >= 8.
        """
        params = HNSWParams(efConstruction=8)
        assert params.efConstruction == 8

    def test_hnsw_params_ef_construction_validation_warning(self):
        """
        Test HNSWParams efConstruction validation for values < 40.

        Coverage: HNSWParams warns for efConstruction < 40.
        """
        # Should warn but still work
        with pytest.warns(UserWarning):
            params = HNSWParams(efConstruction=20)
            assert params.efConstruction == 20


# ============================================================================
# Test ANNOYParams
# ============================================================================


@pytest.mark.unit
class TestANNOYParams:
    """
    Test ANNOYParams class.

    Coverage: ANNOYParams initialization and validation.
    """

    def test_annoy_params_initialization(self):
        """
        Test ANNOYParams initialization.

        Coverage: ANNOYParams can be created with n_trees.
        """
        params = ANNOYParams(n_trees=8)
        assert params.n_trees == 8
        assert params.index_type == IndexType.ANNOY

    def test_annoy_params_default(self):
        """
        Test ANNOYParams default value.

        Coverage: ANNOYParams uses default n_trees=8.
        """
        params = ANNOYParams()
        assert params.n_trees == 8

    def test_annoy_params_validation_minimum(self):
        """
        Test ANNOYParams validation for minimum n_trees.

        Coverage: ANNOYParams validates n_trees >= 1.
        """
        params = ANNOYParams(n_trees=1)
        assert params.n_trees == 1


# ============================================================================
# Test create_index_params Function
# ============================================================================


@pytest.mark.unit
class TestCreateIndexParams:
    """
    Test create_index_params() function.

    Coverage: create_index_params() factory function.
    """

    def test_create_index_params_ivf_flat(self):
        """
        Test create_index_params() for IVF_FLAT.

        Coverage: create_index_params() creates IvfFlatParams for IVF_FLAT.
        """
        params = create_index_params(IndexType.IVF_FLAT, nlist=1024)
        assert isinstance(params, IvfFlatParams)
        assert params.nlist == 1024

    def test_create_index_params_ivf_sq8(self):
        """
        Test create_index_params() for IVF_SQ8.

        Coverage: create_index_params() creates IvfSQ8Params for IVF_SQ8.
        """
        params = create_index_params(IndexType.IVF_SQ8, nlist=1024)
        assert isinstance(params, IvfSQ8Params)
        assert params.nlist == 1024

    def test_create_index_params_ivf_pq(self):
        """
        Test create_index_params() for IVF_PQ.

        Coverage: create_index_params() creates IvfPQParams for IVF_PQ.
        """
        params = create_index_params(IndexType.IVF_PQ, nlist=1024, m=8, nbits=8)
        assert isinstance(params, IvfPQParams)
        assert params.nlist == 1024
        assert params.m == 8
        assert params.nbits == 8

    def test_create_index_params_hnsw(self):
        """
        Test create_index_params() for HNSW.

        Coverage: create_index_params() creates HNSWParams for HNSW.
        """
        params = create_index_params(IndexType.HNSW, M=16, efConstruction=200)
        assert isinstance(params, HNSWParams)
        assert params.M == 16
        assert params.efConstruction == 200

    def test_create_index_params_annoy(self):
        """
        Test create_index_params() for ANNOY.

        Coverage: create_index_params() creates ANNOYParams for ANNOY.
        """
        params = create_index_params(IndexType.ANNOY, n_trees=8)
        assert isinstance(params, ANNOYParams)
        assert params.n_trees == 8

    def test_create_index_params_string_type(self):
        """
        Test create_index_params() with string index type.

        Coverage: create_index_params() converts string to IndexType enum.
        """
        params = create_index_params("IVF_FLAT", nlist=1024)
        assert isinstance(params, IvfFlatParams)
        assert params.nlist == 1024

    def test_create_index_params_string_type_lowercase(self):
        """
        Test create_index_params() with lowercase string type.

        Coverage: create_index_params() handles lowercase string types.
        """
        params = create_index_params("ivf_flat", nlist=1024)
        assert isinstance(params, IvfFlatParams)

    def test_create_index_params_invalid_type(self):
        """
        Test create_index_params() with invalid index type.

        Coverage: create_index_params() raises ValueError for invalid type.
        """
        with pytest.raises(ValueError):
            create_index_params("INVALID_TYPE")

    def test_create_index_params_unsupported_type(self):
        """
        Test create_index_params() with unsupported index type.

        Coverage: create_index_params() raises ValueError for unsupported type.
        """
        with pytest.raises(ValueError):
            create_index_params(IndexType.FLAT)  # FLAT doesn't have params class


# ============================================================================
# Test get_default_params Function
# ============================================================================


@pytest.mark.unit
class TestGetDefaultParams:
    """
    Test get_default_params() function.

    Coverage: get_default_params() factory function.
    """

    def test_get_default_params_ivf_flat(self):
        """
        Test get_default_params() for IVF_FLAT.

        Coverage: get_default_params() returns default IvfFlatParams.
        """
        params = get_default_params(IndexType.IVF_FLAT)
        assert isinstance(params, IvfFlatParams)
        assert params.nlist == 1024

    def test_get_default_params_ivf_sq8(self):
        """
        Test get_default_params() for IVF_SQ8.

        Coverage: get_default_params() returns default IvfSQ8Params.
        """
        params = get_default_params(IndexType.IVF_SQ8)
        assert isinstance(params, IvfSQ8Params)
        assert params.nlist == 1024

    def test_get_default_params_ivf_pq(self):
        """
        Test get_default_params() for IVF_PQ.

        Coverage: get_default_params() returns default IvfPQParams.
        """
        params = get_default_params(IndexType.IVF_PQ)
        assert isinstance(params, IvfPQParams)
        assert params.nlist == 1024
        assert params.m == 8
        assert params.nbits == 8

    def test_get_default_params_hnsw(self):
        """
        Test get_default_params() for HNSW.

        Coverage: get_default_params() returns default HNSWParams.
        """
        params = get_default_params(IndexType.HNSW)
        assert isinstance(params, HNSWParams)
        assert params.M == 16
        assert params.efConstruction == 200

    def test_get_default_params_annoy(self):
        """
        Test get_default_params() for ANNOY.

        Coverage: get_default_params() returns default ANNOYParams.
        """
        params = get_default_params(IndexType.ANNOY)
        assert isinstance(params, ANNOYParams)
        assert params.n_trees == 8

    def test_get_default_params_string_type(self):
        """
        Test get_default_params() with string index type.

        Coverage: get_default_params() converts string to IndexType enum.
        """
        params = get_default_params("IVF_FLAT")
        assert isinstance(params, IvfFlatParams)

    def test_get_default_params_invalid_type(self):
        """
        Test get_default_params() with invalid index type.

        Coverage: get_default_params() raises ValueError for invalid type.
        """
        with pytest.raises(ValueError):
            get_default_params("INVALID_TYPE")


# ============================================================================
# Test Parameter Edge Cases
# ============================================================================


@pytest.mark.unit
class TestParameterEdgeCases:
    """
    Test parameter edge cases and boundary conditions.

    Coverage: Parameter validation edge cases.
    """

    def test_ivf_flat_params_boundary_values(self):
        """
        Test IvfFlatParams with boundary nlist values.

        Coverage: IvfFlatParams handles boundary nlist values.
        """
        # Minimum
        params_min = IvfFlatParams(nlist=1)
        assert params_min.nlist == 1

        # Large value
        params_large = IvfFlatParams(nlist=65536)
        assert params_large.nlist == 65536

    def test_hnsw_params_boundary_values(self):
        """
        Test HNSWParams with boundary M and efConstruction values.

        Coverage: HNSWParams handles boundary values.
        """
        # Minimum M
        params_min_m = HNSWParams(M=4, efConstruction=200)
        assert params_min_m.M == 4

        # Maximum M
        params_max_m = HNSWParams(M=64, efConstruction=200)
        assert params_max_m.M == 64

        # Minimum efConstruction
        params_min_ef = HNSWParams(M=16, efConstruction=8)
        assert params_min_ef.efConstruction == 8

    def test_ivf_pq_params_boundary_values(self):
        """
        Test IvfPQParams with boundary values.

        Coverage: IvfPQParams handles boundary values.
        """
        # Minimum nbits
        params_min = IvfPQParams(nlist=1024, m=8, nbits=1)
        assert params_min.nbits == 1

        # Maximum nbits
        params_max = IvfPQParams(nlist=1024, m=8, nbits=8)
        assert params_max.nbits == 8
