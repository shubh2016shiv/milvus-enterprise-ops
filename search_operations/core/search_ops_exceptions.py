"""
Search Operations Exceptions

This module defines custom exceptions for search operations in Milvus,
providing clear error handling and reporting for search-related issues.
"""

from milvus_ops_exceptions import QueryError


class SearchError(QueryError):
    """Base exception for all search-related errors"""
    pass


class InvalidSearchParametersError(SearchError):
    """Raised when search parameters are invalid"""
    pass


class EmbeddingGenerationError(SearchError):
    """Raised when embedding generation fails"""
    pass


class SearchTimeoutError(SearchError):
    """Raised when a search operation times out"""
    pass


class ReRankingError(SearchError):
    """Raised when re-ranking fails"""
    pass


class HybridSearchError(SearchError):
    """Raised when hybrid search fails"""
    pass


class FusionError(SearchError):
    """Raised when result fusion fails"""
    pass


class EmptyResultError(SearchError):
    """Raised when search returns empty results unexpectedly"""
    pass


class SparseVectorGenerationError(SearchError):
    """Raised when sparse vector (BM25) generation fails"""
    pass


class ConnectionError(SearchError):
    """Raised when connection to Milvus fails"""
    pass


class TimeoutError(SearchError):
    """Raised when operation times out"""
    pass

