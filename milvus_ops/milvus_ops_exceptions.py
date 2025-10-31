"""
Milvus Operations Exceptions

This module defines custom exceptions for the Milvus_Ops package
to provide clear error handling and reporting.
"""


class MilvusOpsError(Exception):
    """Base exception for all Milvus_Ops errors"""


class ConnectionError(MilvusOpsError):
    """Raised when connection to Milvus server fails"""


class ConfigurationError(MilvusOpsError):
    """Raised when configuration is invalid or missing"""


class CollectionError(MilvusOpsError):
    """Base exception for collection-related errors"""


class CollectionNotFoundError(CollectionError):
    """Raised when a collection does not exist"""


class SchemaError(MilvusOpsError):
    """Raised when there's an issue with collection schema"""


class IndexError(MilvusOpsError):
    """Raised when there's an issue with index operations"""


class InsertionError(MilvusOpsError):
    """Raised when data insertion fails"""


class QueryError(MilvusOpsError):
    """Raised when a query operation fails"""


class DataValidationError(MilvusOpsError):
    """Raised when data validation fails"""


class PartitionError(MilvusOpsError):
    """Raised when a partition operation fails"""


class BackupError(MilvusOpsError):
    """Raised when a backup or recovery operation fails"""


class MonitoringError(MilvusOpsError):
    """Raised when a monitoring operation fails"""


class OperationTimeoutError(MilvusOpsError):
    """Raised when an operation times out"""
