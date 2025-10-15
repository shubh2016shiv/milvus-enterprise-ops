"""
Milvus Operations Exceptions

This module defines custom exceptions for the Milvus_Ops package
to provide clear error handling and reporting.
"""

class MilvusOpsError(Exception):
    """Base exception for all Milvus_Ops errors"""
    pass


class ConnectionError(MilvusOpsError):
    """Raised when connection to Milvus server fails"""
    pass


class ConfigurationError(MilvusOpsError):
    """Raised when configuration is invalid or missing"""
    pass


class CollectionError(MilvusOpsError):
    """Base exception for collection-related errors"""
    pass


class CollectionNotFoundError(CollectionError):
    """Raised when a collection does not exist"""
    pass


class SchemaError(MilvusOpsError):
    """Raised when there's an issue with collection schema"""
    pass


class IndexError(MilvusOpsError):
    """Raised when there's an issue with index operations"""
    pass


class InsertionError(MilvusOpsError):
    """Raised when data insertion fails"""
    pass


class QueryError(MilvusOpsError):
    """Raised when a query operation fails"""
    pass


class DataValidationError(MilvusOpsError):
    """Raised when data validation fails"""
    pass


class PartitionError(MilvusOpsError):
    """Raised when a partition operation fails"""
    pass


class BackupError(MilvusOpsError):
    """Raised when a backup or recovery operation fails"""
    pass


class MonitoringError(MilvusOpsError):
    """Raised when a monitoring operation fails"""
    pass


class OperationTimeoutError(MilvusOpsError):
    """Raised when an operation times out"""
    pass
