"""
Data Entities

Defines Pydantic models for documents and operation results in data management.
Provides structured representations of data that can be inserted, updated,
or deleted in Milvus collections, along with detailed operation results.

Typical usage from external projects:

    from data_management_operations import Document, BatchOperationResult
    
    # Create documents
    doc = Document(id=1, vector=[0.1, 0.2, 0.3], metadata="example")
    
    # Access operation results
    result = await manager.insert_documents(...)
    print(f"Inserted {result.successful_count} documents")
    print(f"Success rate: {result.success_rate:.2f}%")
"""

from typing import Dict, List, Any, Optional, Union, TypeVar, Generic, TYPE_CHECKING
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, validator, root_validator
from uuid import UUID

from data_management_operations.utils.timing import TimingResult

if TYPE_CHECKING:
    from data_management_operations.utils.timing import TimingResult

# Type variable for generic document type
T = TypeVar('T')


class OperationStatus(str, Enum):
    """
    Enumeration of possible statuses for data operations.
    
    This provides a standardized way to track and report the status of
    data operations across the system.
    """
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"  # For batch operations where some succeeded and some failed


class DocumentBase(BaseModel):
    """
    Base model for documents to be stored in Milvus.
    
    This provides a common structure for all documents, with required
    fields that ensure proper identification and tracking.
    
    Note: This model uses Pydantic v1 style model definitions.
    If upgrading to Pydantic v2, Config classes would need to be replaced with
    model_config dictionaries and other v2 compatible patterns.
    """
    id: Optional[Union[int, str, UUID]] = Field(
        None, 
        description="Primary key. If None and auto_id=True in collection, Milvus will generate one."
    )
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields for flexibility
        validate_assignment = True


class Document(DocumentBase):
    """
    Standard document model for Milvus data operations.
    
    This extends the base document with vector field(s) and additional metadata.
    The model is flexible enough to handle different vector types and dimensions.
    """
    vector: Optional[Union[List[float], Dict[str, List[float]]]] = Field(
        None,
        description="Vector data. Can be a single vector or a dictionary of named vectors."
    )
    
    @validator('vector')
    def validate_vector(cls, v):
        """Validate vector format."""
        if v is None:
            return v
            
        if isinstance(v, list):
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("Vector must contain only numeric values")
        elif isinstance(v, dict):
            for key, vec in v.items():
                if not isinstance(vec, list) or not all(isinstance(x, (int, float)) for x in vec):
                    raise ValueError(f"Vector '{key}' must be a list of numeric values")
        else:
            raise ValueError("Vector must be either a list of numbers or a dictionary of named vectors")
        return v


class BatchOperationResult(BaseModel):
    """
    Result of a batch operation on multiple documents.
    
    This provides detailed information about the success or failure of
    each document in a batch operation, allowing for precise error handling
    and reporting.
    """
    status: OperationStatus
    successful_count: int = 0
    failed_count: int = 0
    error_messages: Dict[Union[int, str], str] = Field(
        default_factory=dict,
        description="Map of document IDs to error messages for failed operations"
    )
    inserted_ids: List[Any] = Field(
        default_factory=list,
        description="List of IDs for successfully inserted documents"
    )
    
    @property
    def total_count(self) -> int:
        """Total number of documents processed in this batch."""
        return self.successful_count + self.failed_count
    
    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.total_count == 0:
            return 0.0
        return (self.successful_count / self.total_count) * 100


class DeleteResult(BaseModel):
    """
    Result of a delete operation.
    
    This provides information about the success or failure of a delete
    operation, including the number of documents deleted and any timing data.
    """
    status: OperationStatus
    deleted_count: int = 0
    error_message: Optional[str] = None
    timing: Optional[TimingResult] = Field(None, description="Performance timing result for the operation.")


class DataValidationResult(BaseModel):
    """
    Result of data validation before insertion.
    
    This provides detailed information about validation errors,
    allowing for precise error reporting and correction.
    """
    is_valid: bool
    errors: Dict[Union[int, str], List[str]] = Field(
        default_factory=dict,
        description="Map of document indices or IDs to lists of validation errors"
    )

# Forward reference resolution is handled automatically in Pydantic v2
