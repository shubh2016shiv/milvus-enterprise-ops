"""
Providers Module

This module provides interfaces and implementations for external providers
such as embedding generation services.
"""

from .embedding import EmbeddingProvider, EmbeddingResult
from .gemini_embedding import DimensionMismatchError, GeminiEmbeddingProvider, TaskType

__all__ = [
    # Base interfaces
    "EmbeddingProvider",
    "EmbeddingResult",
    # Gemini implementation
    "GeminiEmbeddingProvider",
    "DimensionMismatchError",
    "TaskType",
]
