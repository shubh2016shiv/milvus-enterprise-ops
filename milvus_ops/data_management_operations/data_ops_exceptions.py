"""
Data Management Operations Exceptions

Granular exception hierarchy for data management operations,
providing clear error reporting and enabling precise error handling
in external projects.
"""

from typing import List, Any, Optional, Dict

# Define base exceptions locally to avoid circular imports
# This ensures data_management_operations can be imported independently
class MilvusOpsError(Exception):
    """Base exception for all Milvus_Ops errors"""
    pass


class InsertionError(MilvusOpsError):
    """Raised when data insertion fails"""
    pass


class DataOperationError(MilvusOpsError):
    """
    Base exception for all data management operation errors.
    
    This serves as the parent class for all data operation-specific
    exceptions, allowing external projects to catch all data operation
    errors with a single except clause if desired.
    """
    pass


class BatchPartialFailureError(DataOperationError):
    """
    Raised when a batch operation partially succeeds.
    
    This exception is raised when some documents in a batch operation
    succeed while others fail. It provides detailed information about
    which documents succeeded and which failed, enabling external
    projects to handle partial failures appropriately.
    
    Attributes:
        message: Human-readable error message
        successful_count: Number of documents that succeeded
        failed_count: Number of documents that failed
        failed_ids: List of IDs for documents that failed
        error_details: Dictionary mapping failed IDs to error messages
    
    Example:
        ```python
        try:
            result = await manager.insert_documents(...)
        except BatchPartialFailureError as e:
            logger.error(
                f"Partial failure: {e.successful_count} succeeded, "
                f"{e.failed_count} failed"
            )
            for doc_id, error in e.error_details.items():
                logger.error(f"Document {doc_id} failed: {error}")
        ```
    """
    
    def __init__(
        self,
        message: str,
        successful_count: int,
        failed_count: int,
        failed_ids: List[Any],
        error_details: Optional[Dict[Any, str]] = None
    ):
        """
        Initialize partial failure exception.
        
        Args:
            message: Human-readable error message
            successful_count: Number of documents that succeeded
            failed_count: Number of documents that failed
            failed_ids: List of IDs for failed documents
            error_details: Optional mapping of failed IDs to error messages
        """
        super().__init__(message)
        self.successful_count = successful_count
        self.failed_count = failed_count
        self.failed_ids = failed_ids
        self.error_details = error_details or {}
    
    @property
    def total_count(self) -> int:
        """Total number of documents in the batch."""
        return self.successful_count + self.failed_count
    
    @property
    def success_rate(self) -> float:
        """Success rate as a percentage (0-100)."""
        if self.total_count == 0:
            return 0.0
        return (self.successful_count / self.total_count) * 100.0


class TransientOperationError(DataOperationError):
    """
    Raised for transient errors that can be retried.
    
    This exception indicates that an operation failed due to a transient
    condition (e.g., temporary schema unavailability, collection temporarily
    locked) that may succeed if retried. The DataManager will automatically
    retry operations that raise this exception if retry_transient_errors
    is enabled in the configuration.
    
    Note: This is distinct from transient errors handled by ConnectionManager
    (network issues, server unavailability). This exception is for application-
    level transient conditions.
    
    Example:
        ```python
        # This is typically raised internally, but can be caught:
        try:
            result = await manager.insert_documents(...)
        except TransientOperationError as e:
            # Operation was retried but still failed
            logger.error(f"Transient error persisted: {e}")
        ```
    """
    pass


class SchemaValidationError(DataOperationError):
    """
    Raised when data doesn't match the collection schema.
    
    This exception is raised during pre-operation validation when
    documents don't conform to the collection's schema requirements
    (missing fields, wrong types, incorrect vector dimensions, etc.).
    
    Attributes:
        message: Human-readable error message
        validation_errors: Dictionary mapping document IDs/indices to error lists
    
    Example:
        ```python
        try:
            result = await manager.insert_documents(...)
        except SchemaValidationError as e:
            logger.error(f"Schema validation failed: {e}")
            for doc_id, errors in e.validation_errors.items():
                logger.error(f"Document {doc_id} errors: {errors}")
        ```
    """
    
    def __init__(self, message: str, validation_errors: Optional[Dict[Any, List[str]]] = None):
        """
        Initialize schema validation exception.
        
        Args:
            message: Human-readable error message
            validation_errors: Dictionary mapping document IDs to error lists
        """
        super().__init__(message)
        self.validation_errors = validation_errors or {}


class DocumentPreparationError(DataOperationError):
    """
    Raised when document preparation for insertion fails.
    
    This exception is raised when documents cannot be properly prepared
    for insertion into Milvus (e.g., serialization failures, type conversion
    errors, or internal processing errors).
    
    Example:
        ```python
        try:
            result = await manager.insert_documents(...)
        except DocumentPreparationError as e:
            logger.error(f"Failed to prepare documents: {e}")
        ```
    """
    pass


class CollectionOperationError(DataOperationError):
    """
    Raised when a collection-level operation fails.
    
    This exception is raised when operations involving collection
    metadata or state fail (e.g., collection doesn't exist and
    auto_create is disabled, collection is not loaded).
    
    Example:
        ```python
        try:
            result = await manager.insert_documents(...)
        except CollectionOperationError as e:
            logger.error(f"Collection operation failed: {e}")
        ```
    """
    pass


class DeleteOperationError(DataOperationError):
    """
    Raised when a delete operation fails.
    
    This exception is raised when documents cannot be deleted from
    a collection due to expression errors, permission issues, or
    other delete-specific failures.
    
    Example:
        ```python
        try:
            result = await manager.delete_documents(...)
        except DeleteOperationError as e:
            logger.error(f"Delete operation failed: {e}")
        ```
    """
    pass


# InsertionError is already defined above as a local class