"""
Index Operations Exceptions

This module defines a granular exception hierarchy for index operations,
allowing for precise error handling and reporting. Each exception type
corresponds to a specific failure mode in index operations.

Typical usage:
    from Milvus_Ops.index_operations import IndexParameterError

    try:
        result = await index_manager.create_index(...)
    except IndexParameterError as e:
        print(f"Invalid index parameters: {e}")
        # Handle parameter error specifically
    except IndexBuildError as e:
        print(f"Index build failed: {e}")
        # Handle build failure
"""

from milvus_ops.milvus_ops_exceptions import IndexError as BaseIndexError


class IndexOperationError(BaseIndexError):
    """
    Base exception for all index operation errors.

    This serves as the parent class for all index-specific exceptions,
    allowing for catch-all error handling while still providing
    granular exception types for specific error conditions.
    """


class IndexBuildError(IndexOperationError):
    """
    Raised when an index build operation fails.

    This exception indicates that the index creation process started
    but encountered an error during the build phase.

    Attributes:
        collection_name: Name of the collection
        field_name: Name of the field
        index_type: Type of index that failed to build
    """

    def __init__(
        self,
        message: str,
        collection_name: str | None = None,
        field_name: str | None = None,
        index_type: str | None = None,
    ):
        super().__init__(message)
        self.collection_name = collection_name
        self.field_name = field_name
        self.index_type = index_type


class IndexNotFoundError(IndexOperationError):
    """
    Raised when an index does not exist.

    This exception is raised when trying to perform operations on
    an index that doesn't exist, such as getting build progress
    or dropping an index.

    Attributes:
        collection_name: Name of the collection
        field_name: Name of the field
    """

    def __init__(
        self,
        message: str,
        collection_name: str | None = None,
        field_name: str | None = None,
    ):
        super().__init__(message)
        self.collection_name = collection_name
        self.field_name = field_name


class IndexParameterError(IndexOperationError):
    """
    Raised when index parameters are invalid.

    This exception indicates that the parameters provided for
    index creation are invalid or incompatible with the index type.

    Attributes:
        collection_name: Name of the collection
        field_name: Name of the field
        index_type: Type of index
        parameter_errors: Dictionary of parameter errors
    """

    def __init__(
        self,
        message: str,
        collection_name: str | None = None,
        field_name: str | None = None,
        index_type: str | None = None,
        parameter_errors: dict[str, str] | None = None,
    ):
        super().__init__(message)
        self.collection_name = collection_name
        self.field_name = field_name
        self.index_type = index_type
        self.parameter_errors = parameter_errors or {}


class IndexTypeError(IndexOperationError):
    """
    Raised when an unsupported index type is specified.

    This exception is raised when trying to create an index with
    an unsupported or invalid index type.

    Attributes:
        index_type: The invalid index type
        supported_types: List of supported index types
    """

    def __init__(
        self,
        message: str,
        index_type: str | None = None,
        supported_types: list[str] | None = None,
    ):
        super().__init__(message)
        self.index_type = index_type
        self.supported_types = supported_types or []


class IndexBuildInProgressError(IndexOperationError):
    """
    Raised when trying to build an index that is already being built.

    This exception indicates that an index build operation is already
    in progress for the specified field.

    Attributes:
        collection_name: Name of the collection
        field_name: Name of the field
        progress: Current build progress percentage
    """

    def __init__(
        self,
        message: str,
        collection_name: str | None = None,
        field_name: str | None = None,
        progress: float | None = None,
    ):
        super().__init__(message)
        self.collection_name = collection_name
        self.field_name = field_name
        self.progress = progress


class IndexResourceError(IndexOperationError):
    """
    Raised when there are insufficient resources for index operations.

    This exception indicates that there are insufficient resources
    (memory, disk space, etc.) to complete the index operation.

    Attributes:
        resource_type: Type of resource that's insufficient
        required: Required amount of resource
        available: Available amount of resource
    """

    def __init__(
        self,
        message: str,
        resource_type: str | None = None,
        required: float | None = None,
        available: float | None = None,
    ):
        super().__init__(message)
        self.resource_type = resource_type
        self.required = required
        self.available = available


class IndexTimeoutError(IndexOperationError):
    """
    Raised when an index operation times out.

    This exception indicates that an index operation took longer
    than the specified timeout.

    Attributes:
        operation: The operation that timed out
        timeout_seconds: The timeout in seconds
    """

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        timeout_seconds: float | None = None,
    ):
        super().__init__(message)
        self.operation = operation
        self.timeout_seconds = timeout_seconds
