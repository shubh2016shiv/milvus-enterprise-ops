"""
Search Operations Exceptions

This module defines custom exceptions for search operations in Milvus,
providing clear error handling and reporting for search-related issues.
"""

from milvus_ops.milvus_ops_exceptions import QueryError


class SearchError(QueryError):
    """Base exception for all search-related errors"""


class InvalidSearchParametersError(SearchError):
    """Raised when search parameters are invalid"""


class EmbeddingGenerationError(SearchError):
    """Raised when embedding generation fails"""


class SearchTimeoutError(SearchError):
    """Raised when a search operation times out"""


class ReRankingError(SearchError):
    """Raised when re-ranking fails"""


class HybridSearchError(SearchError):
    """Raised when hybrid search fails"""


class FusionError(SearchError):
    """Raised when result fusion fails"""


class EmptyResultError(SearchError):
    """Raised when search returns empty results unexpectedly"""


class SparseVectorGenerationError(SearchError):
    """Raised when sparse vector (BM25) generation fails"""


class ConnectionError(SearchError):
    """Raised when connection to Milvus fails"""


class TimeoutError(SearchError):
    """Raised when operation times out"""
