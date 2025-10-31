"""
Core package for partition operations.

This package contains the main functionality for partition management
in the Milvus enterprise operations system.
"""

from .manager import PartitionManager
from .validator import PartitionValidator

__all__ = ["PartitionManager", "PartitionValidator"]
