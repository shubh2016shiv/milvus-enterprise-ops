"""
Comprehensive unit tests for data management operations models.

This module provides systematic testing of all model classes in the
data_management_operations.models module, including Document, DocumentBase,
BatchOperationResult, DeleteResult, DataValidationResult, and OperationStatus.
"""

from uuid import UUID

import pytest

from milvus_ops.data_management_operations.models.entities import (
    BatchOperationResult,
    DataValidationResult,
    DeleteResult,
    Document,
    DocumentBase,
    OperationStatus,
)
from milvus_ops.data_management_operations.utils.timing import TimingResult

# ============================================================================
# Test OperationStatus Enum
# ============================================================================


@pytest.mark.unit
class TestOperationStatus:
    """
    Test OperationStatus enumeration.

    Coverage: OperationStatus enum values and behavior.
    """

    def test_operation_status_success(self):
        """
        Test OperationStatus.SUCCESS value.

        Coverage: SUCCESS enum value.
        """
        assert OperationStatus.SUCCESS == "success"
        assert OperationStatus.SUCCESS.value == "success"

    def test_operation_status_failed(self):
        """
        Test OperationStatus.FAILED value.

        Coverage: FAILED enum value.
        """
        assert OperationStatus.FAILED == "failed"
        assert OperationStatus.FAILED.value == "failed"

    def test_operation_status_partial(self):
        """
        Test OperationStatus.PARTIAL value.

        Coverage: PARTIAL enum value.
        """
        assert OperationStatus.PARTIAL == "partial"
        assert OperationStatus.PARTIAL.value == "partial"

    def test_operation_status_all_values(self):
        """
        Test all OperationStatus values are defined.

        Coverage: Complete enum value set.
        """
        all_statuses = {OperationStatus.SUCCESS, OperationStatus.FAILED, OperationStatus.PARTIAL}
        assert len(all_statuses) == 3


# ============================================================================
# Test DocumentBase
# ============================================================================


@pytest.mark.unit
class TestDocumentBase:
    """
    Test DocumentBase model.

    Coverage: DocumentBase ID field handling with various types.
    """

    @pytest.mark.parametrize(
        "id_value",
        [
            1,
            1000,
            "string_id",
            "uuid_string",
            None,
            UUID("12345678-1234-5678-1234-567812345678"),
        ],
    )
    def test_document_base_id_types(self, id_value):
        """
        Test DocumentBase with various ID types.

        Coverage: DocumentBase ID field accepts int, str, UUID, and None.
        """
        doc = DocumentBase(id=id_value)
        assert doc.id == id_value

    def test_document_base_no_id(self):
        """
        Test DocumentBase without ID.

        Coverage: DocumentBase with None ID (auto-id scenario).
        """
        doc = DocumentBase()
        assert doc.id is None

    def test_document_base_extra_fields(self):
        """
        Test DocumentBase allows extra fields.

        Coverage: DocumentBase extra field handling.
        """
        doc = DocumentBase(id=1, extra_field="value")
        assert doc.id == 1
        assert doc.extra_field == "value"


# ============================================================================
# Test Document
# ============================================================================


@pytest.mark.unit
class TestDocument:
    """
    Test Document model.

    Coverage: Document vector validation and field handling.
    """

    def test_document_with_list_vector(self):
        """
        Test Document with list vector.

        Coverage: Document accepts list of floats for vector.
        """
        doc = Document(id=1, vector=[0.1, 0.2, 0.3, 0.4])
        assert doc.id == 1
        assert doc.vector == [0.1, 0.2, 0.3, 0.4]

    def test_document_with_dict_vector(self):
        """
        Test Document with dictionary of named vectors.

        Coverage: Document accepts dict of named vectors.
        """
        doc = Document(id=1, vector={"embedding": [0.1, 0.2], "feature": [0.3, 0.4]})
        assert doc.id == 1
        assert isinstance(doc.vector, dict)
        assert "embedding" in doc.vector
        assert "feature" in doc.vector

    def test_document_with_none_vector(self):
        """
        Test Document with None vector.

        Coverage: Document allows None vector.
        """
        doc = Document(id=1, vector=None)
        assert doc.id == 1
        assert doc.vector is None

    def test_document_invalid_vector_type(self):
        """
        Test Document with invalid vector type.

        Coverage: Document raises ValidationError for invalid vector types.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Document(id=1, vector="not_a_vector")

    def test_document_invalid_list_vector_content(self):
        """
        Test Document with invalid list vector content.

        Coverage: Document raises ValidationError for non-numeric list elements.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Document(id=1, vector=[0.1, "not_a_number", 0.3])

    def test_document_invalid_dict_vector_content(self):
        """
        Test Document with invalid dict vector content.

        Coverage: Document raises ValidationError for invalid dict vector values.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Document(id=1, vector={"embedding": "not_a_list"})

    def test_document_inherits_from_document_base(self):
        """
        Test Document inherits from DocumentBase.

        Coverage: Document inheritance hierarchy.
        """
        doc = Document(id=1, vector=[0.1, 0.2])
        assert isinstance(doc, DocumentBase)

    @pytest.mark.parametrize(
        "vector",
        [
            [0.1, 0.2, 0.3],
            [1, 2, 3],  # Integers should work too
            {"v1": [0.1, 0.2], "v2": [0.3, 0.4]},
            None,
        ],
    )
    def test_document_valid_vector_formats(self, vector):
        """
        Test Document with various valid vector formats.

        Coverage: Document accepts multiple valid vector formats.
        """
        doc = Document(id=1, vector=vector)
        assert doc.vector == vector


# ============================================================================
# Test BatchOperationResult
# ============================================================================


@pytest.mark.unit
class TestBatchOperationResult:
    """
    Test BatchOperationResult model.

    Coverage: BatchOperationResult properties and calculations.
    """

    def test_batch_operation_result_success(self):
        """
        Test BatchOperationResult with successful operation.

        Coverage: BatchOperationResult with SUCCESS status.
        """
        result = BatchOperationResult(
            status=OperationStatus.SUCCESS,
            successful_count=10,
            failed_count=0,
            inserted_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )
        assert result.status == OperationStatus.SUCCESS
        assert result.successful_count == 10
        assert result.failed_count == 0
        assert result.total_count == 10
        assert result.success_rate == 100.0
        assert len(result.inserted_ids) == 10
        assert len(result.error_messages) == 0

    def test_batch_operation_result_partial(self):
        """
        Test BatchOperationResult with partial success.

        Coverage: BatchOperationResult with PARTIAL status.
        """
        result = BatchOperationResult(
            status=OperationStatus.PARTIAL,
            successful_count=8,
            failed_count=2,
            inserted_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            error_messages={"9": "Error 1", "10": "Error 2"},
        )
        assert result.status == OperationStatus.PARTIAL
        assert result.successful_count == 8
        assert result.failed_count == 2
        assert result.total_count == 10
        assert result.success_rate == 80.0
        assert len(result.inserted_ids) == 8
        assert len(result.error_messages) == 2

    def test_batch_operation_result_failed(self):
        """
        Test BatchOperationResult with complete failure.

        Coverage: BatchOperationResult with FAILED status.
        """
        result = BatchOperationResult(
            status=OperationStatus.FAILED,
            successful_count=0,
            failed_count=5,
            error_messages={
                "1": "Error 1",
                "2": "Error 2",
                "3": "Error 3",
                "4": "Error 4",
                "5": "Error 5",
            },
        )
        assert result.status == OperationStatus.FAILED
        assert result.successful_count == 0
        assert result.failed_count == 5
        assert result.total_count == 5
        assert result.success_rate == 0.0
        assert len(result.inserted_ids) == 0
        assert len(result.error_messages) == 5

    def test_batch_operation_result_success_rate_calculation(self):
        """
        Test BatchOperationResult success_rate calculation.

        Coverage: success_rate property calculation.
        """
        # 50% success rate
        result = BatchOperationResult(
            status=OperationStatus.PARTIAL,
            successful_count=5,
            failed_count=5,
        )
        assert result.success_rate == 50.0

        # 0% success rate
        result = BatchOperationResult(
            status=OperationStatus.FAILED,
            successful_count=0,
            failed_count=10,
        )
        assert result.success_rate == 0.0

        # 100% success rate
        result = BatchOperationResult(
            status=OperationStatus.SUCCESS,
            successful_count=10,
            failed_count=0,
        )
        assert result.success_rate == 100.0

    def test_batch_operation_result_empty(self):
        """
        Test BatchOperationResult with zero counts.

        Coverage: BatchOperationResult with empty counts.
        """
        result = BatchOperationResult(
            status=OperationStatus.SUCCESS,
            successful_count=0,
            failed_count=0,
        )
        assert result.total_count == 0
        assert result.success_rate == 0.0

    def test_batch_operation_result_default_error_messages(self):
        """
        Test BatchOperationResult defaults error_messages to empty dict.

        Coverage: BatchOperationResult default error_messages.
        """
        result = BatchOperationResult(
            status=OperationStatus.SUCCESS,
            successful_count=10,
            failed_count=0,
        )
        assert isinstance(result.error_messages, dict)
        assert len(result.error_messages) == 0

    def test_batch_operation_result_default_inserted_ids(self):
        """
        Test BatchOperationResult defaults inserted_ids to empty list.

        Coverage: BatchOperationResult default inserted_ids.
        """
        result = BatchOperationResult(
            status=OperationStatus.SUCCESS,
            successful_count=0,
            failed_count=0,
        )
        assert isinstance(result.inserted_ids, list)
        assert len(result.inserted_ids) == 0


# ============================================================================
# Test DeleteResult
# ============================================================================


@pytest.mark.unit
class TestDeleteResult:
    """
    Test DeleteResult model.

    Coverage: DeleteResult properties and field handling.
    """

    def test_delete_result_success(self):
        """
        Test DeleteResult with successful deletion.

        Coverage: DeleteResult with SUCCESS status.
        """
        result = DeleteResult(status=OperationStatus.SUCCESS, deleted_count=5)
        assert result.status == OperationStatus.SUCCESS
        assert result.deleted_count == 5
        assert result.error_message is None
        assert result.timing is None

    def test_delete_result_failed(self):
        """
        Test DeleteResult with failed deletion.

        Coverage: DeleteResult with FAILED status and error message.
        """
        result = DeleteResult(
            status=OperationStatus.FAILED,
            deleted_count=0,
            error_message="Collection not found",
        )
        assert result.status == OperationStatus.FAILED
        assert result.deleted_count == 0
        assert result.error_message == "Collection not found"

    def test_delete_result_with_timing(self):
        """
        Test DeleteResult with timing information.

        Coverage: DeleteResult with TimingResult.
        """
        timing = TimingResult(operation_name="delete", execution_time=0.5, success=True)
        result = DeleteResult(status=OperationStatus.SUCCESS, deleted_count=3, timing=timing)
        assert result.timing == timing
        assert result.timing.execution_time == 0.5

    def test_delete_result_default_values(self):
        """
        Test DeleteResult default values.

        Coverage: DeleteResult default field values.
        """
        result = DeleteResult(status=OperationStatus.SUCCESS)
        assert result.deleted_count == 0
        assert result.error_message is None
        assert result.timing is None


# ============================================================================
# Test DataValidationResult
# ============================================================================


@pytest.mark.unit
class TestDataValidationResult:
    """
    Test DataValidationResult model.

    Coverage: DataValidationResult properties and error handling.
    """

    def test_data_validation_result_valid(self):
        """
        Test DataValidationResult with valid documents.

        Coverage: DataValidationResult with is_valid=True.
        """
        result = DataValidationResult(is_valid=True)
        assert result.is_valid is True
        assert isinstance(result.errors, dict)
        assert len(result.errors) == 0

    def test_data_validation_result_invalid(self):
        """
        Test DataValidationResult with invalid documents.

        Coverage: DataValidationResult with is_valid=False and errors.
        """
        errors = {
            1: ["Missing required field 'vector'"],
            2: ["Invalid type for field 'id'", "Vector dimension mismatch"],
        }
        result = DataValidationResult(is_valid=False, errors=errors)
        assert result.is_valid is False
        assert result.errors == errors
        assert len(result.errors) == 2
        assert 1 in result.errors
        assert 2 in result.errors

    def test_data_validation_result_default_errors(self):
        """
        Test DataValidationResult defaults errors to empty dict.

        Coverage: DataValidationResult default errors.
        """
        result = DataValidationResult(is_valid=True)
        assert isinstance(result.errors, dict)
        assert len(result.errors) == 0

    def test_data_validation_result_string_keys(self):
        """
        Test DataValidationResult with string document IDs.

        Coverage: DataValidationResult errors dict with string keys.
        """
        errors = {
            "doc_1": ["Error 1"],
            "doc_2": ["Error 2", "Error 3"],
        }
        result = DataValidationResult(is_valid=False, errors=errors)
        assert "doc_1" in result.errors
        assert "doc_2" in result.errors
        assert len(result.errors["doc_2"]) == 2

    def test_data_validation_result_multiple_errors_per_document(self):
        """
        Test DataValidationResult with multiple errors per document.

        Coverage: DataValidationResult error list per document.
        """
        errors = {
            1: ["Error 1", "Error 2", "Error 3"],
        }
        result = DataValidationResult(is_valid=False, errors=errors)
        assert len(result.errors[1]) == 3
        assert "Error 1" in result.errors[1]
        assert "Error 2" in result.errors[1]
        assert "Error 3" in result.errors[1]
