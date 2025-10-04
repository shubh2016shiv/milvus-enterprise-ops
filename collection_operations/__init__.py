"""
Collection Operations Module

This module provides comprehensive functionality for managing Milvus collections:
- Collection creation with optimized schemas and validation
- Collection loading and unloading with progress tracking
- Collection statistics and metadata retrieval
- Collection dropping with safety checks
- Schema definition and management with strong typing

Future enhancements may include:
- Collection cloning (currently not implemented)
- Collection renaming (currently not implemented)
- Schema evolution (currently not implemented)

Implements best practices for collection management in production environments
with proper error handling and validation.
"""

from .manager import CollectionManager
from .schema import CollectionSchema, FieldSchema, DataType, IndexType, MetricType
from .entities import (
    CollectionDescription, 
    CollectionStats, 
    LoadProgress, 
    LoadState, 
    CollectionState
)

__all__ = [
    'CollectionManager',
    'CollectionSchema',
    'FieldSchema',
    'DataType',
    'IndexType',
    'MetricType',
    'CollectionDescription',
    'CollectionStats',
    'LoadProgress',
    'LoadState',
    'CollectionState',
]