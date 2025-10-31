"""
Comprehensive unit tests for data management operations exceptions.

This module provides systematic testing of all exception classes in the
data_management_operations module, including their instantiation, attributes,
and inheritance hierarchy.
"""

import pytest

from milvus_ops.data_management_operations.data_ops_exceptions import (
    BatchPartialFailureError,
    CollectionOperationError,
    DataOperationError,
    DeleteOperationError,
    DocumentPreparationError,
    InsertionError,
    MilvusOpsError,
    SchemaValidationError,
    TransientOperationError,
)

# ============================================================================
# Test Exception Instantiation
# ============================================================================


@pytest.mark.unit
class TestExceptionInstantiation:
    """
    Test exception class instantiation.

    Coverage: All exception types can be instantiated with messages.
    """

    def test_data_operation_error(self):
        """
        Test DataOperationError instantiation.

        Coverage: DataOperationError can be created with a message.
        """
        error = DataOperationError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_batch_partial_failure_error(self):
        """
        Test BatchPartialFailureError instantiation.

        Coverage: BatchPartialFailureError can be created with required attributes.
        """
        error = BatchPartialFailureError(
            message="Partial failure occurred",
            successful_count=8,
            failed_count=2,
            failed_ids=[9, 10],
            error_details={9: "Error 1", 10: "Error 2"},
        )
        assert str(error) == "Partial failure occurred"
        assert error.successful_count == 8
        assert error.failed_count == 2
        assert error.failed_ids == [9, 10]
        assert error.error_details == {9: "Error 1", 10: "Error 2"}

    def test_schema_validation_error(self):
        """
        Test SchemaValidationError instantiation.

        Coverage: SchemaValidationError can be created with validation errors.
        """
        validation_errors = {1: ["Error 1"], 2: ["Error 2", "Error 3"]}
        error = SchemaValidationError(
            "Schema validation failed", validation_errors=validation_errors
        )
        assert str(error) == "Schema validation failed"
        assert error.validation_errors == validation_errors

    def test_collection_operation_error(self):
        """
        Test CollectionOperationError instantiation.

        Coverage: CollectionOperationError can be created with a message.
        """
        error = CollectionOperationError("Collection operation failed")
        assert str(error) == "Collection operation failed"

    def test_delete_operation_error(self):
        """
        Test DeleteOperationError instantiation.

        Coverage: DeleteOperationError can be created with a message.
        """
        error = DeleteOperationError("Delete operation failed")
        assert str(error) == "Delete operation failed"

    def test_document_preparation_error(self):
        """
        Test DocumentPreparationError instantiation.

        Coverage: DocumentPreparationError can be created with a message.
        """
        error = DocumentPreparationError("Document preparation failed")
        assert str(error) == "Document preparation failed"

    def test_insertion_error(self):
        """
        Test InsertionError instantiation.

        Coverage: InsertionError can be created with a message.
        """
        error = InsertionError("Insertion failed")
        assert str(error) == "Insertion failed"

    def test_transient_operation_error(self):
        """
        Test TransientOperationError instantiation.

        Coverage: TransientOperationError can be created with a message.
        """
        error = TransientOperationError("Transient operation error")
        assert str(error) == "Transient operation error"


# ============================================================================
# Test BatchPartialFailureError Special Attributes
# ============================================================================


@pytest.mark.unit
class TestBatchPartialFailureErrorAttributes:
    """
    Test BatchPartialFailureError special attributes.

    Coverage: BatchPartialFailureError properties and calculations.
    """

    def test_batch_partial_failure_error_total_count(self):
        """
        Test BatchPartialFailureError total_count property.

        Coverage: total_count property calculation.
        """
        error = BatchPartialFailureError(
            message="Partial failure",
            successful_count=8,
            failed_count=2,
            failed_ids=[9, 10],
        )
        assert error.total_count == 10
        assert error.total_count == error.successful_count + error.failed_count

    def test_batch_partial_failure_error_success_rate(self):
        """
        Test BatchPartialFailureError success_rate property.

        Coverage: success_rate property calculation.
        """
        # 80% success rate
        error = BatchPartialFailureError(
            message="Partial failure",
            successful_count=8,
            failed_count=2,
            failed_ids=[9, 10],
        )
        assert error.success_rate == 80.0

        # 50% success rate
        error = BatchPartialFailureError(
            message="Partial failure",
            successful_count=5,
            failed_count=5,
            failed_ids=[6, 7, 8, 9, 10],
        )
        assert error.success_rate == 50.0

        # 0% success rate
        error = BatchPartialFailureError(
            message="Partial failure",
            successful_count=0,
            failed_count=10,
            failed_ids=list(range(1, 11)),
        )
        assert error.success_rate == 0.0

    def test_batch_partial_failure_error_success_rate_zero_total(self):
        """
        Test BatchPartialFailureError success_rate with zero total count.

        Coverage: success_rate returns 0.0 when total_count is 0.
        """
        error = BatchPartialFailureError(
            message="Partial failure",
            successful_count=0,
            failed_count=0,
            failed_ids=[],
        )
        assert error.total_count == 0
        assert error.success_rate == 0.0

    def test_batch_partial_failure_error_default_error_details(self):
        """
        Test BatchPartialFailureError defaults error_details to empty dict.

        Coverage: error_details defaults to empty dict if not provided.
        """
        error = BatchPartialFailureError(
            message="Partial failure",
            successful_count=8,
            failed_count=2,
            failed_ids=[9, 10],
        )
        assert isinstance(error.error_details, dict)

    def test_batch_partial_failure_error_none_error_details(self):
        """
        Test BatchPartialFailureError with None error_details.

        Coverage: error_details defaults to empty dict when None is provided.
        """
        error = BatchPartialFailureError(
            message="Partial failure",
            successful_count=8,
            failed_count=2,
            failed_ids=[9, 10],
            error_details=None,
        )
        assert isinstance(error.error_details, dict)
        assert len(error.error_details) == 0


# ============================================================================
# Test SchemaValidationError Attributes
# ============================================================================


@pytest.mark.unit
class TestSchemaValidationErrorAttributes:
    """
    Test SchemaValidationError attributes.

    Coverage: SchemaValidationError validation_errors attribute.
    """

    def test_schema_validation_error_default_validation_errors(self):
        """
        Test SchemaValidationError defaults validation_errors to empty dict.

        Coverage: validation_errors defaults to empty dict if not provided.
        """
        error = SchemaValidationError("Schema validation failed")
        assert isinstance(error.validation_errors, dict)
        assert len(error.validation_errors) == 0

    def test_schema_validation_error_none_validation_errors(self):
        """
        Test SchemaValidationError with None validation_errors.

        Coverage: validation_errors defaults to empty dict when None is provided.
        """
        error = SchemaValidationError("Schema validation failed", validation_errors=None)
        assert isinstance(error.validation_errors, dict)
        assert len(error.validation_errors) == 0

    def test_schema_validation_error_with_validation_errors(self):
        """
        Test SchemaValidationError with validation errors.

        Coverage: validation_errors stores error dictionary correctly.
        """
        validation_errors = {
            1: ["Missing field 'vector'"],
            2: ["Invalid type for 'id'", "Vector dimension mismatch"],
            3: ["Extraneous field 'extra'"],
        }
        error = SchemaValidationError(
            "Schema validation failed", validation_errors=validation_errors
        )
        assert error.validation_errors == validation_errors
        assert len(error.validation_errors) == 3
        assert len(error.validation_errors[2]) == 2


# ============================================================================
# Test Exception Hierarchy
# ============================================================================


@pytest.mark.unit
class TestExceptionHierarchy:
    """
    Test exception inheritance hierarchy.

    Coverage: Exception inheritance relationships.
    """

    def test_data_operation_error_inherits_from_milvus_ops_error(self):
        """
        Test DataOperationError inheritance.

        Coverage: DataOperationError inherits from MilvusOpsError.
        """
        error = DataOperationError("Test")
        assert isinstance(error, MilvusOpsError)
        assert isinstance(error, Exception)

    def test_batch_partial_failure_error_inherits_from_data_operation_error(self):
        """
        Test BatchPartialFailureError inheritance.

        Coverage: BatchPartialFailureError inherits from DataOperationError.
        """
        error = BatchPartialFailureError(
            message="Test",
            successful_count=1,
            failed_count=1,
            failed_ids=[1],
        )
        assert isinstance(error, DataOperationError)
        assert isinstance(error, MilvusOpsError)
        assert isinstance(error, Exception)

    def test_transient_operation_error_inherits_from_data_operation_error(self):
        """
        Test TransientOperationError inheritance.

        Coverage: TransientOperationError inherits from DataOperationError.
        """
        error = TransientOperationError("Test")
        assert isinstance(error, DataOperationError)
        assert isinstance(error, MilvusOpsError)
        assert isinstance(error, Exception)

    def test_schema_validation_error_inherits_from_data_operation_error(self):
        """
        Test SchemaValidationError inheritance.

        Coverage: SchemaValidationError inherits from DataOperationError.
        """
        error = SchemaValidationError("Test")
        assert isinstance(error, DataOperationError)
        assert isinstance(error, MilvusOpsError)
        assert isinstance(error, Exception)

    def test_collection_operation_error_inherits_from_data_operation_error(self):
        """
        Test CollectionOperationError inheritance.

        Coverage: CollectionOperationError inherits from DataOperationError.
        """
        error = CollectionOperationError("Test")
        assert isinstance(error, DataOperationError)
        assert isinstance(error, MilvusOpsError)
        assert isinstance(error, Exception)

    def test_delete_operation_error_inherits_from_data_operation_error(self):
        """
        Test DeleteOperationError inheritance.

        Coverage: DeleteOperationError inherits from DataOperationError.
        """
        error = DeleteOperationError("Test")
        assert isinstance(error, DataOperationError)
        assert isinstance(error, MilvusOpsError)
        assert isinstance(error, Exception)

    def test_document_preparation_error_inherits_from_data_operation_error(self):
        """
        Test DocumentPreparationError inheritance.

        Coverage: DocumentPreparationError inherits from DataOperationError.
        """
        error = DocumentPreparationError("Test")
        assert isinstance(error, DataOperationError)
        assert isinstance(error, MilvusOpsError)
        assert isinstance(error, Exception)

    def test_insertion_error_inherits_from_milvus_ops_error(self):
        """
        Test InsertionError inheritance.

        Coverage: InsertionError inherits from MilvusOpsError.
        """
        error = InsertionError("Test")
        assert isinstance(error, MilvusOpsError)
        assert isinstance(error, Exception)


# ============================================================================
# Test Exception Chaining
# ============================================================================


@pytest.mark.unit
class TestExceptionChaining:
    """
    Test exception chaining.

    Coverage: Exception __cause__ and __context__ attributes.
    """

    def test_exception_chaining_with_from(self):
        """
        Test exception chaining using from.

        Coverage: Exception chaining preserves original exception.
        """
        original_error = ValueError("Original error")
        try:
            raise DataOperationError("Wrapper error") from original_error
        except DataOperationError as error:
            assert error.__cause__ == original_error
            assert isinstance(error.__cause__, ValueError)

    def test_exception_chaining_with_raise(self):
        """
        Test exception chaining in raise statement.

        Coverage: Exception context is preserved during raise.
        """
        try:
            try:
                raise ValueError("Inner error")
            except ValueError as e:
                raise DataOperationError("Outer error") from e
        except DataOperationError as error:
            assert error.__cause__ is not None
            assert isinstance(error.__cause__, ValueError)

    def test_batch_partial_failure_error_chaining(self):
        """
        Test BatchPartialFailureError with exception chaining.

        Coverage: BatchPartialFailureError supports exception chaining.
        """
        original = InsertionError("Insert failed")
        try:
            raise BatchPartialFailureError(
                message="Partial failure",
                successful_count=5,
                failed_count=5,
                failed_ids=[6, 7, 8, 9, 10],
            ) from original
        except BatchPartialFailureError as error:
            assert error.__cause__ == original
