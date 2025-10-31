"""
Semantic Search Configuration

This module defines configuration for semantic (dense vector) search operations.
"""

from dataclasses import dataclass

from .base import BaseSearchConfig


@dataclass
class SemanticSearchConfig(BaseSearchConfig):
    """
    Configuration for semantic (dense vector) search.

    This configuration is used for pure vector similarity search
    using dense embeddings.
    """

    search_field: str = "vector"  # Default vector field name
    expr: str | None = None  # Optional filtering expression
