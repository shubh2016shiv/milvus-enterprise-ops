"""
Comprehensive unit tests for index operations exceptions.

This module provides systematic testing of all exception classes in the
index_operations module, including their instantiation, attributes,
and inheritance hierarchy.
"""

import pytest

from milvus_ops.index_operations.index_ops_exceptions import (
    IndexBuildError,
    IndexBuildInProgressError,
    IndexNotFoundError,
    IndexOperationError,
    IndexParameterError,
    IndexResourceError,
    IndexTimeoutError,
    IndexTypeError,
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

    def test_index_operation_error(self):
        """
        Test IndexOperationError instantiation.

        Coverage: IndexOperationError can be created with a message.
        """
        error = IndexOperationError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_index_build_error(self):
        """
        Test IndexBuildError instantiation.

        Coverage: IndexBuildError can be created with required attributes.
        """
        error = IndexBuildError(
            "Index build failed",
            collection_name="test_collection",
            field_name="embedding",
            index_type="HNSW",
        )
        assert str(error) == "Index build failed"
        assert error.collection_name == "test_collection"
        assert error.field_name == "embedding"
        assert error.index_type == "HNSW"

    def test_index_build_error_minimal(self):
        """
        Test IndexBuildError instantiation with minimal arguments.

        Coverage: IndexBuildError can be created with only message.
        """
        error = IndexBuildError("Index build failed")
        assert str(error) == "Index build failed"
        assert error.collection_name is None
        assert error.field_name is None
        assert error.index_type is None

    def test_index_not_found_error(self):
        """
        Test IndexNotFoundError instantiation.

        Coverage: IndexNotFoundError can be created with collection and field names.
        """
        error = IndexNotFoundError(
            "Index not found",
            collection_name="test_collection",
            field_name="embedding",
        )
        assert str(error) == "Index not found"
        assert error.collection_name == "test_collection"
        assert error.field_name == "embedding"

    def test_index_not_found_error_minimal(self):
        """
        Test IndexNotFoundError instantiation with minimal arguments.

        Coverage: IndexNotFoundError can be created with only message.
        """
        error = IndexNotFoundError("Index not found")
        assert str(error) == "Index not found"
        assert error.collection_name is None
        assert error.field_name is None

    def test_index_parameter_error(self):
        """
        Test IndexParameterError instantiation.

        Coverage: IndexParameterError can be created with parameter errors.
        """
        parameter_errors = {"dimension": "Too large", "metric_type": "Invalid"}
        error = IndexParameterError(
            "Invalid parameters",
            collection_name="test_collection",
            field_name="embedding",
            index_type="HNSW",
            parameter_errors=parameter_errors,
        )
        assert str(error) == "Invalid parameters"
        assert error.collection_name == "test_collection"
        assert error.field_name == "embedding"
        assert error.index_type == "HNSW"
        assert error.parameter_errors == parameter_errors

    def test_index_parameter_error_minimal(self):
        """
        Test IndexParameterError instantiation with minimal arguments.

        Coverage: IndexParameterError can be created with only message.
        """
        error = IndexParameterError("Invalid parameters")
        assert str(error) == "Invalid parameters"
        assert error.parameter_errors == {}

    def test_index_type_error(self):
        """
        Test IndexTypeError instantiation.

        Coverage: IndexTypeError can be created with supported types.
        """
        supported_types = ["HNSW", "IVF_FLAT", "IVF_SQ8"]
        error = IndexTypeError(
            "Unsupported index type",
            index_type="INVALID",
            supported_types=supported_types,
        )
        assert str(error) == "Unsupported index type"
        assert error.index_type == "INVALID"
        assert error.supported_types == supported_types

    def test_index_type_error_minimal(self):
        """
        Test IndexTypeError instantiation with minimal arguments.

        Coverage: IndexTypeError can be created with only message.
        """
        error = IndexTypeError("Unsupported index type")
        assert str(error) == "Unsupported index type"
        assert error.index_type is None
        assert error.supported_types == []

    def test_index_build_in_progress_error(self):
        """
        Test IndexBuildInProgressError instantiation.

        Coverage: IndexBuildInProgressError can be created with progress.
        """
        error = IndexBuildInProgressError(
            "Index build in progress",
            collection_name="test_collection",
            field_name="embedding",
            progress=45.5,
        )
        assert str(error) == "Index build in progress"
        assert error.collection_name == "test_collection"
        assert error.field_name == "embedding"
        assert error.progress == 45.5

    def test_index_build_in_progress_error_minimal(self):
        """
        Test IndexBuildInProgressError instantiation with minimal arguments.

        Coverage: IndexBuildInProgressError can be created with only message.
        """
        error = IndexBuildInProgressError("Index build in progress")
        assert str(error) == "Index build in progress"
        assert error.progress is None

    def test_index_resource_error(self):
        """
        Test IndexResourceError instantiation.

        Coverage: IndexResourceError can be created with resource information.
        """
        error = IndexResourceError(
            "Insufficient resources",
            resource_type="memory",
            required=1024 * 1024 * 100,  # 100 MB
            available=1024 * 1024 * 50,  # 50 MB
        )
        assert str(error) == "Insufficient resources"
        assert error.resource_type == "memory"
        assert error.required == 1024 * 1024 * 100
        assert error.available == 1024 * 1024 * 50

    def test_index_resource_error_minimal(self):
        """
        Test IndexResourceError instantiation with minimal arguments.

        Coverage: IndexResourceError can be created with only message.
        """
        error = IndexResourceError("Insufficient resources")
        assert str(error) == "Insufficient resources"
        assert error.resource_type is None
        assert error.required is None
        assert error.available is None

    def test_index_timeout_error(self):
        """
        Test IndexTimeoutError instantiation.

        Coverage: IndexTimeoutError can be created with operation and timeout.
        """
        error = IndexTimeoutError(
            "Operation timed out",
            operation="create_index",
            timeout_seconds=60.0,
        )
        assert str(error) == "Operation timed out"
        assert error.operation == "create_index"
        assert error.timeout_seconds == 60.0

    def test_index_timeout_error_minimal(self):
        """
        Test IndexTimeoutError instantiation with minimal arguments.

        Coverage: IndexTimeoutError can be created with only message.
        """
        error = IndexTimeoutError("Operation timed out")
        assert str(error) == "Operation timed out"
        assert error.operation is None
        assert error.timeout_seconds is None


# ============================================================================
# Test Exception Inheritance Hierarchy
# ============================================================================


@pytest.mark.unit
class TestExceptionInheritance:
    """
    Test exception inheritance hierarchy.

    Coverage: Exception inheritance chains and base class relationships.
    """

    def test_index_operation_error_base(self):
        """
        Test IndexOperationError is base class.

        Coverage: IndexOperationError is base for all index operation exceptions.
        """
        error = IndexOperationError("Base error")
        assert isinstance(error, Exception)

    def test_index_build_error_inheritance(self):
        """
        Test IndexBuildError inheritance.

        Coverage: IndexBuildError inherits from IndexOperationError.
        """
        error = IndexBuildError("Build error")
        assert isinstance(error, IndexOperationError)
        assert isinstance(error, Exception)

    def test_index_not_found_error_inheritance(self):
        """
        Test IndexNotFoundError inheritance.

        Coverage: IndexNotFoundError inherits from IndexOperationError.
        """
        error = IndexNotFoundError("Not found error")
        assert isinstance(error, IndexOperationError)
        assert isinstance(error, Exception)

    def test_index_parameter_error_inheritance(self):
        """
        Test IndexParameterError inheritance.

        Coverage: IndexParameterError inherits from IndexOperationError.
        """
        error = IndexParameterError("Parameter error")
        assert isinstance(error, IndexOperationError)
        assert isinstance(error, Exception)

    def test_index_type_error_inheritance(self):
        """
        Test IndexTypeError inheritance.

        Coverage: IndexTypeError inherits from IndexOperationError.
        """
        error = IndexTypeError("Type error")
        assert isinstance(error, IndexOperationError)
        assert isinstance(error, Exception)

    def test_index_build_in_progress_error_inheritance(self):
        """
        Test IndexBuildInProgressError inheritance.

        Coverage: IndexBuildInProgressError inherits from IndexOperationError.
        """
        error = IndexBuildInProgressError("In progress error")
        assert isinstance(error, IndexOperationError)
        assert isinstance(error, Exception)

    def test_index_resource_error_inheritance(self):
        """
        Test IndexResourceError inheritance.

        Coverage: IndexResourceError inherits from IndexOperationError.
        """
        error = IndexResourceError("Resource error")
        assert isinstance(error, IndexOperationError)
        assert isinstance(error, Exception)

    def test_index_timeout_error_inheritance(self):
        """
        Test IndexTimeoutError inheritance.

        Coverage: IndexTimeoutError inherits from IndexOperationError.
        """
        error = IndexTimeoutError("Timeout error")
        assert isinstance(error, IndexOperationError)
        assert isinstance(error, Exception)


# ============================================================================
# Test Exception Attribute Access
# ============================================================================


@pytest.mark.unit
class TestExceptionAttributes:
    """
    Test exception attribute access.

    Coverage: Exception attributes can be accessed and modified.
    """

    def test_index_build_error_attributes(self):
        """
        Test IndexBuildError attribute access.

        Coverage: IndexBuildError attributes can be accessed after creation.
        """
        error = IndexBuildError(
            "Build error",
            collection_name="test_collection",
            field_name="embedding",
            index_type="HNSW",
        )
        assert error.collection_name == "test_collection"
        assert error.field_name == "embedding"
        assert error.index_type == "HNSW"

    def test_index_parameter_error_dict(self):
        """
        Test IndexParameterError parameter_errors dictionary.

        Coverage: IndexParameterError parameter_errors is accessible dictionary.
        """
        parameter_errors = {"dimension": "Too large", "metric": "Invalid"}
        error = IndexParameterError("Parameter error", parameter_errors=parameter_errors)
        assert isinstance(error.parameter_errors, dict)
        assert len(error.parameter_errors) == 2
        assert error.parameter_errors["dimension"] == "Too large"
        assert error.parameter_errors["metric"] == "Invalid"

    def test_index_parameter_error_empty_dict(self):
        """
        Test IndexParameterError with empty parameter_errors.

        Coverage: IndexParameterError parameter_errors defaults to empty dict.
        """
        error = IndexParameterError("Parameter error")
        assert isinstance(error.parameter_errors, dict)
        assert len(error.parameter_errors) == 0

    def test_index_type_error_supported_types_list(self):
        """
        Test IndexTypeError supported_types list.

        Coverage: IndexTypeError supported_types is accessible list.
        """
        supported_types = ["HNSW", "IVF_FLAT", "IVF_SQ8"]
        error = IndexTypeError("Type error", supported_types=supported_types)
        assert isinstance(error.supported_types, list)
        assert len(error.supported_types) == 3
        assert "HNSW" in error.supported_types

    def test_index_type_error_empty_list(self):
        """
        Test IndexTypeError with empty supported_types.

        Coverage: IndexTypeError supported_types defaults to empty list.
        """
        error = IndexTypeError("Type error")
        assert isinstance(error.supported_types, list)
        assert len(error.supported_types) == 0


# ============================================================================
# Test Exception Error Messages
# ============================================================================


@pytest.mark.unit
class TestExceptionMessages:
    """
    Test exception error message formatting.

    Coverage: Exception error messages are formatted correctly.
    """

    def test_error_message_with_details(self):
        """
        Test error message with collection and field details.

        Coverage: Error messages can include collection and field context.
        """
        error = IndexBuildError(
            "Build failed",
            collection_name="test_collection",
            field_name="embedding",
        )
        message = str(error)
        assert "Build failed" in message
        assert message == "Build failed"

    def test_error_message_parameter_details(self):
        """
        Test error message with parameter details.

        Coverage: ParameterError messages include parameter context.
        """
        parameter_errors = {"dimension": "Invalid", "metric": "Wrong type"}
        error = IndexParameterError("Invalid parameters", parameter_errors=parameter_errors)
        message = str(error)
        assert "Invalid parameters" in message
        assert message == "Invalid parameters"

    def test_error_message_resource_details(self):
        """
        Test error message with resource details.

        Coverage: ResourceError messages include resource context.
        """
        error = IndexResourceError(
            "Insufficient memory",
            resource_type="memory",
            required=1000000,
            available=500000,
        )
        message = str(error)
        assert "Insufficient memory" in message
        assert message == "Insufficient memory"

    def test_error_message_timeout_details(self):
        """
        Test error message with timeout details.

        Coverage: TimeoutError messages include timeout context.
        """
        error = IndexTimeoutError(
            "Operation timed out",
            operation="create_index",
            timeout_seconds=60.0,
        )
        message = str(error)
        assert "Operation timed out" in message
        assert message == "Operation timed out"
