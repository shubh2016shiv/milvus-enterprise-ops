"""
Base Search Operations

This module provides the abstract base class for all search operations,
defining the common interface and functionality for different search types.
"""

import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, TypeVar, Generic
from dataclasses import dataclass, field

from connection_management import ConnectionManager
from exceptions import OperationTimeoutError
from .exceptions import (
    SearchError, 
    EmbeddingGenerationError,
    SearchTimeoutError
)
from ..config.base import BaseSearchConfig
from ..providers.embedding import EmbeddingProvider

# Type variable for search configuration
T = TypeVar('T', bound=BaseSearchConfig)


@dataclass
class SearchResult:
    """
    Result of a search operation.
    
    This class encapsulates the search results and metadata
    about the search operation.
    """
    hits: List[Dict[str, Any]]
    total_hits: int
    took_ms: float
    search_params: Dict[str, Any] = field(default_factory=dict)
    raw_response: Optional[Any] = None
    
    def __post_init__(self):
        """Initialize default values if not provided."""
        if self.search_params is None:
            self.search_params = {}


class BaseSearch(Generic[T], ABC):
    """
    Abstract base class for all search operations.
    
    This class defines the common interface and functionality
    for different search types, providing a consistent API
    for search operations.
    """
    
    def __init__(
        self, 
        connection_manager: ConnectionManager,
        embedding_provider: Optional[EmbeddingProvider] = None
    ):
        """
        Initialize the search operation.
        
        Args:
            connection_manager: ConnectionManager for Milvus operations
            embedding_provider: Provider for generating embeddings
        """
        self._connection_manager = connection_manager
        self._embedding_provider = embedding_provider
    
    @abstractmethod
    async def search(
        self, 
        collection_name: str,
        query: str,
        config: T
    ) -> SearchResult:
        """
        Perform search operation.
        
        Args:
            collection_name: Name of the collection to search
            query: Query text or vector
            config: Search configuration
            
        Returns:
            SearchResult with hits and metadata
            
        Raises:
            SearchError: If search operation fails
        """
        pass
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector
            
        Raises:
            EmbeddingGenerationError: If embedding generation fails
        """
        if not self._embedding_provider:
            raise EmbeddingGenerationError("No embedding provider configured")
        
        try:
            result = await self._embedding_provider.generate_embedding(text)
            return result.embedding  # type: ignore
            
        except Exception as e:
            raise EmbeddingGenerationError(f"Failed to generate embedding: {str(e)}") from e
    
    async def _execute_search(
        self,
        collection_name: str,
        search_params: Dict[str, Any],
        timeout: float
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Execute search operation via connection manager.
        
        This method handles the actual search execution through
        the connection manager, with proper error handling.
        
        Args:
            collection_name: Name of the collection to search
            search_params: Parameters for the search operation
            timeout: Search timeout in seconds
            
        Returns:
            Tuple of (search results, execution time in ms)
            
        Raises:
            SearchError: If search operation fails
        """
        start_time = time.time()
        
        try:
            # Execute search operation
            results = await self._connection_manager.execute_operation_async(
                lambda alias: self._perform_search(
                    alias=alias,
                    collection_name=collection_name,
                    search_params=search_params
                ),
                timeout=timeout
            )
            
            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000
            
            return results, execution_time_ms
            
        except OperationTimeoutError as e:
            raise SearchTimeoutError(f"Search operation timed out after {timeout} seconds") from e
            
        except Exception as e:
            raise SearchError(f"Search operation failed: {str(e)}") from e
    
    def _perform_search(
        self,
        alias: str,
        collection_name: str,
        search_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Perform search using PyMilvus client.
        
        This method is executed via the connection manager
        to perform the actual search operation.
        
        Args:
            alias: Connection alias
            collection_name: Name of the collection to search
            search_params: Parameters for the search operation
            
        Returns:
            List of search results
            
        Raises:
            Exception: If search operation fails
        """
        from pymilvus import Collection
        
        try:
            # Get collection
            collection = Collection(name=collection_name, using=alias)
            
            # Perform search
            search_results = collection.search(**search_params)
            
            # Convert to list of dictionaries
            results = []
            for hits in search_results:
                for hit in hits:
                    result = {
                        "id": hit.id,
                        "distance": hit.distance,
                        "score": hit.score,
                    }
                    
                    # Add entity fields if available
                    if hasattr(hit, "entity"):
                        for field_name, field_value in hit.entity.items():
                            result[field_name] = field_value
                    
                    results.append(result)
            
            return results
            
        except Exception as e:
            raise
    

