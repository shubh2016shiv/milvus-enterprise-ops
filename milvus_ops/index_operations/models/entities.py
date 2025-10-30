"""
Index Operation Entities

This module defines Pydantic models for representing Milvus index operations,
including index states, descriptions, build progress, and operation results.
These models provide structured representations of index metadata and operation
outcomes with strong type safety.

Typical usage:
    from Milvus_Ops.index_operations import IndexDescription, IndexState
    
    # Get index information
    index_info = await index_manager.describe_index(
        collection_name="documents",
        field_name="embedding"
    )
    
    # Check index state
    if index_info.state == IndexState.CREATED:
        print(f"Index is ready: {index_info.index_name}")
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Union, TypeVar, Generic
from datetime import datetime
from pydantic import BaseModel, Field, validator
import math


class IndexState(str, Enum):
    """
    Enumeration of possible index states.
    
    Represents the lifecycle states of an index, from non-existent
    through creation to completion or failure.
    
    Attributes:
        NONE: No index exists on the field
        CREATING: Index is currently being built
        CREATED: Index has been successfully created and is ready for use
        FAILED: Index creation failed
    """
    NONE = "none"
    CREATING = "creating"
    CREATED = "created"
    FAILED = "failed"


class IndexDescription(BaseModel):
    """
    Comprehensive description of a Milvus index.
    
    Contains all metadata about an index, including its configuration,
    parameters, and current state. This model is returned by the
    describe_index operation.
    
    Attributes:
        collection_name: Name of the collection containing the index
        field_name: Name of the field the index is built on
        index_name: Name of the index (auto-generated if not specified)
        index_type: Type of the index (e.g., HNSW, IVF_FLAT)
        metric_type: Distance metric used by the index
        params: Index-specific parameters
        state: Current state of the index
        created_at: When the index was created (None if not yet created)
        indexed_rows: Number of rows indexed (None if not available)
        index_size_bytes: Size of the index in bytes (None if not available)
        failed_reason: Reason for failure if state is FAILED
    """
    collection_name: str
    field_name: str
    index_name: str
    index_type: str
    metric_type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    state: IndexState = IndexState.NONE
    created_at: Optional[datetime] = None
    indexed_rows: Optional[int] = None
    index_size_bytes: Optional[int] = None
    failed_reason: Optional[str] = None
    
    @property
    def index_size_mb(self) -> Optional[float]:
        """Size of the index in megabytes, or None if size is unknown."""
        if self.index_size_bytes is None:
            return None
        return self.index_size_bytes / (1024 * 1024)
    
    @property
    def is_available(self) -> bool:
        """Whether the index is available for use."""
        return self.state == IndexState.CREATED


class IndexBuildProgress(BaseModel):
    """
    Progress information for an ongoing index build operation.
    
    Provides detailed information about the progress of an index build,
    including completion percentage and estimated time remaining.
    
    Attributes:
        collection_name: Name of the collection
        field_name: Name of the field being indexed
        state: Current state of the index build
        percentage: Completion percentage (0-100)
        start_time: When the build started
        current_time: Current time when progress was checked
        estimated_completion_time: Estimated time of completion
        estimated_remaining_time_seconds: Estimated seconds remaining
        total_rows: Total number of rows to index
        processed_rows: Number of rows processed so far
        failed_reason: Reason for failure if state is FAILED
    """
    collection_name: str
    field_name: str
    state: IndexState
    percentage: float = 0.0
    start_time: Optional[datetime] = None
    current_time: Optional[datetime] = None
    estimated_completion_time: Optional[datetime] = None
    estimated_remaining_time_seconds: Optional[float] = None
    total_rows: Optional[int] = None
    processed_rows: Optional[int] = None
    failed_reason: Optional[str] = None
    
    @validator('percentage')
    def validate_percentage(cls, v):
        """Ensure percentage is between 0 and 100."""
        return max(0.0, min(100.0, v))
    
    @property
    def is_complete(self) -> bool:
        """Whether the index build is complete."""
        return self.state == IndexState.CREATED
    
    @property
    def has_failed(self) -> bool:
        """Whether the index build has failed."""
        return self.state == IndexState.FAILED
    
    @property
    def formatted_eta(self) -> Optional[str]:
        """Human-readable estimated time remaining."""
        if self.estimated_remaining_time_seconds is None:
            return None
        
        seconds = self.estimated_remaining_time_seconds
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{int(hours)} hours, {int(minutes)} minutes"


class IndexStats(BaseModel):
    """
    Performance statistics for an index.
    
    Contains metrics about index performance, memory usage,
    and other operational characteristics.
    
    Attributes:
        collection_name: Name of the collection
        field_name: Name of the indexed field
        index_type: Type of the index
        query_latency_ms: Average query latency in milliseconds
        memory_usage_bytes: Memory usage of the index
        disk_usage_bytes: Disk usage of the index
        indexed_vectors: Number of vectors in the index
        last_updated: When the stats were last updated
    """
    collection_name: str
    field_name: str
    index_type: str
    query_latency_ms: Optional[float] = None
    memory_usage_bytes: Optional[int] = None
    disk_usage_bytes: Optional[int] = None
    indexed_vectors: Optional[int] = None
    last_updated: Optional[datetime] = None
    
    @property
    def memory_usage_mb(self) -> Optional[float]:
        """Memory usage in megabytes."""
        if self.memory_usage_bytes is None:
            return None
        return self.memory_usage_bytes / (1024 * 1024)
    
    @property
    def disk_usage_mb(self) -> Optional[float]:
        """Disk usage in megabytes."""
        if self.disk_usage_bytes is None:
            return None
        return self.disk_usage_bytes / (1024 * 1024)


class IndexResult(BaseModel):
    """
    Result of an index operation.
    
    Provides information about the outcome of an index operation,
    such as creation, dropping, or rebuilding.
    
    Attributes:
        success: Whether the operation was successful
        collection_name: Name of the collection
        field_name: Name of the field
        index_name: Name of the index
        operation: Type of operation performed
        state: Current state of the index
        error_message: Error message if operation failed
        execution_time_ms: Time taken to execute the operation
    """
    success: bool
    collection_name: str
    field_name: str
    index_name: Optional[str] = None
    operation: str  # 'create', 'drop', 'rebuild'
    state: IndexState
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None
    
    @property
    def is_complete(self) -> bool:
        """Whether the operation is complete (success or failure)."""
        return self.state in (IndexState.CREATED, IndexState.FAILED)
    
    @property
    def is_in_progress(self) -> bool:
        """Whether the operation is still in progress."""
        return self.state == IndexState.CREATING
