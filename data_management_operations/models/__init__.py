"""
Data Models

Contains Pydantic models for documents and operation results.
"""

from .entities import (
    Document,
    DocumentBase,
    BatchOperationResult,
    DeleteResult,
    DataValidationResult,
    OperationStatus
)

__all__ = [
    'Document',
    'DocumentBase',
    'BatchOperationResult',
    'DeleteResult',
    'DataValidationResult',
    'OperationStatus'
]

