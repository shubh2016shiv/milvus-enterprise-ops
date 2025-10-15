"""
Search Operations Manager

This module provides a unified interface for all search operations,
allowing for easy switching between different search types and configurations.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Union

from connection_management import ConnectionManager
from .search_ops_exceptions import SearchError, InvalidSearchParametersError
from ..config.base import SearchType, ReRankingMethod
from ..config.semantic import SemanticSearchConfig
from ..config.hybrid import HybridSearchConfig
from ..config.reranking import ReRankingConfig
from ..config.validation import SearchParams
from ..providers.embedding import EmbeddingProvider
from .base import SearchResult
from ..search.semantic.engine import SemanticSearch
from ..search.hybrid.core.engine import HybridSearch
from ..reranking.reranker import MilvusReRanker

logger = logging.getLogger(__name__)


class SearchManager:
    """
    Manager for search operations.
    
    This class provides a unified interface for all search operations,
    handling the creation and management of different search types.
    """
    
    def __init__(
        self,
        connection_manager: ConnectionManager,
        embedding_provider: EmbeddingProvider,
        enable_caching: bool = True
    ):
        """
        Initialize search manager.
        
        Args:
            connection_manager: ConnectionManager for Milvus operations
            embedding_provider: Provider for generating embeddings
            enable_caching: Whether to enable embedding caching
        """
        self._connection_manager = connection_manager
        self._embedding_provider = embedding_provider
        self._enable_caching = enable_caching
        
        # Initialize search instances
        self._semantic_search = SemanticSearch(
            connection_manager=connection_manager,
            embedding_provider=embedding_provider,
            enable_caching=enable_caching
        )
        
        self._hybrid_search = HybridSearch(
            connection_manager=connection_manager,
            embedding_provider=embedding_provider,
            enable_caching=enable_caching
        )
        
        # Initialize reranker
        self._reranker = MilvusReRanker()
    
    async def search(
        self,
        collection_name: str,
        query: str,
        search_params: Union[Dict[str, Any], SearchParams]
    ) -> SearchResult:
        """
        Perform search operation based on parameters.
        
        This method provides a unified interface for all search types,
        automatically selecting the appropriate search implementation
        based on the provided parameters.
        
        Args:
            collection_name: Name of the collection to search
            query: Query text
            search_params: Search parameters as dict or SearchParams
            
        Returns:
            SearchResult with hits and metadata
            
        Raises:
            InvalidSearchParametersError: If parameters are invalid
            SearchError: If search operation fails
        """
        try:
            # Convert dict to SearchParams if needed
            if isinstance(search_params, dict):
                search_params = SearchParams(**search_params)
            
            # Extract search type
            search_type = search_params.search_type
            
            # Create appropriate configurations
            if search_type == SearchType.SEMANTIC:
                # Create semantic search config
                search_config = self._create_semantic_config(search_params)
                
                # Apply reranking if enabled
                if search_params.rerank:
                    rerank_config = self._create_reranking_config(search_params)
                    
                    # Perform search with native Milvus reranking
                    return await self._perform_semantic_search_with_reranking(
                        collection_name=collection_name,
                        query=query,
                        search_config=search_config,
                        rerank_config=rerank_config
                    )
                else:
                    # Perform regular semantic search
                    return await self._semantic_search.search(
                        collection_name=collection_name,
                        query=query,
                        config=search_config
                    )
            elif search_type == SearchType.HYBRID:
                # Create hybrid search config
                search_config = self._create_hybrid_config(search_params)
                
                # Apply reranking if enabled
                if search_params.rerank:
                    rerank_config = self._create_reranking_config(search_params)
                    
                    # Perform search with native Milvus reranking
                    return await self._perform_hybrid_search_with_reranking(
                        collection_name=collection_name,
                        query=query,
                        search_config=search_config,
                        rerank_config=rerank_config
                    )
                else:
                    # Perform regular hybrid search
                    return await self._hybrid_search.search(
                        collection_name=collection_name,
                        query=query,
                        config=search_config
                    )
            else:
                raise InvalidSearchParametersError(f"Unsupported search type: {search_type}")
                
        except Exception as e:
            if isinstance(e, (InvalidSearchParametersError, SearchError)):
                raise
            
            error_msg = f"Search operation failed: {str(e)}"
            logger.error(error_msg)
            raise SearchError(error_msg) from e
    
    def _create_semantic_config(self, params: SearchParams) -> SemanticSearchConfig:
        """
        Create semantic search configuration from parameters.
        
        Args:
            params: Search parameters
            
        Returns:
            SemanticSearchConfig instance
        """
        return SemanticSearchConfig(
            top_k=params.top_k,
            timeout=params.timeout,
            metric_type=params.metric_type,
            search_field=params.vector_field,
            expr=params.expr,
            params=params.params,
            output_fields=params.output_fields
        )
    
    def _create_hybrid_config(self, params: SearchParams) -> HybridSearchConfig:
        """
        Create hybrid search configuration from parameters.
        
        Args:
            params: Search parameters
            
        Returns:
            HybridSearchConfig instance
        """
        return HybridSearchConfig(
            top_k=params.top_k,
            timeout=params.timeout,
            metric_type=params.metric_type,
            vector_field=params.vector_field,
            sparse_field=params.sparse_field,
            keyword_field=params.keyword_field,
            vector_weight=params.vector_weight,
            sparse_weight=params.sparse_weight,
            params=params.params,
            output_fields=params.output_fields
        )
    
    def _create_reranking_config(self, params: SearchParams) -> ReRankingConfig:
        """
        Create re-ranking configuration from parameters.
        
        Args:
            params: Search parameters
            
        Returns:
            ReRankingConfig instance
        """
        rerank_params = {}
        
        # Add method-specific parameters
        if params.rerank_method == ReRankingMethod.WEIGHTED:
            rerank_params["weights"] = params.rerank_weights or [0.5, 0.5]
        elif params.rerank_method == ReRankingMethod.RRF:
            rerank_params["k"] = params.rerank_k
        
        return ReRankingConfig(
            enabled=params.rerank,
            method=params.rerank_method,
            params=rerank_params
        )
    
    async def _perform_semantic_search_with_reranking(
        self,
        collection_name: str,
        query: str,
        search_config: SemanticSearchConfig,
        rerank_config: ReRankingConfig
    ) -> SearchResult:
        """
        Perform semantic search with Milvus native reranking.
        
        This method adds reranking parameters to the search config
        and performs the search in a single operation, using Milvus's
        native reranking capabilities.
        
        Args:
            collection_name: Name of the collection to search
            query: Query text
            search_config: Semantic search configuration
            rerank_config: Reranking configuration
            
        Returns:
            Search results
            
        Raises:
            SearchError: If search fails
            ReRankingError: If reranking fails
        """
        try:
            # Generate query embedding
            query_vector = await self._semantic_search._generate_embedding(query)
            
            # Prepare search parameters
            search_params = {
                "data": [query_vector],
                "anns_field": search_config.search_field,
                "param": search_config.params,
                "limit": search_config.top_k,
                "expr": search_config.expr,
                "output_fields": search_config.output_fields or []
            }
            
            # Add reranking parameters if enabled
            if rerank_config.enabled:
                search_params = self._reranker.get_search_params(
                    config=rerank_config,
                    search_params=search_params
                )
            
            # Execute search operation
            results, execution_time_ms = await self._semantic_search._execute_search(
                collection_name=collection_name,
                search_params=search_params,
                timeout=search_config.timeout
            )
            
            # Create search result
            search_result = SearchResult(
                hits=results,
                total_hits=len(results),
                took_ms=execution_time_ms,
                search_params={
                    "type": "semantic",
                    "field": search_config.search_field,
                    "top_k": search_config.top_k,
                    "metric_type": search_config.metric_type,
                    "params": search_config.params,
                    "reranking": {
                        "enabled": rerank_config.enabled,
                        "method": rerank_config.method
                    }
                }
            )
            
            # Process results
            if rerank_config.enabled:
                search_result = self._reranker.process_results(
                    results=search_result,
                    config=rerank_config
                )
            
            logger.debug(
                f"Semantic search with reranking completed in {execution_time_ms:.2f}ms, "
                f"found {len(results)} results"
            )
            
            return search_result
            
        except Exception as e:
            error_msg = f"Semantic search with reranking failed: {str(e)}"
            logger.error(error_msg)
            raise SearchError(error_msg) from e
    
    async def _perform_hybrid_search_with_reranking(
        self,
        collection_name: str,
        query: str,
        search_config: HybridSearchConfig,
        rerank_config: ReRankingConfig
    ) -> SearchResult:
        """
        Perform hybrid search with Milvus native reranking.
        
        This method adds reranking parameters to the search config
        and performs the search in a single operation, using Milvus's
        native reranking capabilities.
        
        Args:
            collection_name: Name of the collection to search
            query: Query text
            search_config: Hybrid search configuration
            rerank_config: Reranking configuration
            
        Returns:
            Search results
            
        Raises:
            SearchError: If search fails
            ReRankingError: If reranking fails
        """
        try:
            # Generate query embedding
            query_vector = await self._hybrid_search._generate_embedding(query)
            
            # Prepare hybrid search parameters
            search_params = await self._hybrid_search._prepare_hybrid_search_params(
                query=query,
                query_vector=query_vector,
                config=search_config
            )
            
            # Add reranking parameters if enabled
            if rerank_config.enabled:
                search_params = self._reranker.get_search_params(
                    config=rerank_config,
                    search_params=search_params
                )
            
            # Execute search operation
            results, execution_time_ms = await self._hybrid_search._execute_search(
                collection_name=collection_name,
                search_params=search_params,
                timeout=search_config.timeout
            )
            
            # Create search result
            search_result = SearchResult(
                hits=results,
                total_hits=len(results),
                took_ms=execution_time_ms,
                search_params={
                    "type": "hybrid",
                    "vector_field": search_config.vector_field,
                    "sparse_field": search_config.sparse_field,
                    "keyword_field": search_config.keyword_field,
                    "top_k": search_config.top_k,
                    "metric_type": search_config.metric_type,
                    "params": search_config.params,
                    "vector_weight": search_config.vector_weight,
                    "sparse_weight": search_config.sparse_weight,
                    "reranking": {
                        "enabled": rerank_config.enabled,
                        "method": rerank_config.method
                    }
                }
            )
            
            # Process results
            if rerank_config.enabled:
                search_result = self._reranker.process_results(
                    results=search_result,
                    config=rerank_config
                )
            
            logger.debug(
                f"Hybrid search with reranking completed in {execution_time_ms:.2f}ms, "
                f"found {len(results)} results"
            )
            
            return search_result
            
        except Exception as e:
            error_msg = f"Hybrid search with reranking failed: {str(e)}"
            logger.error(error_msg)
            raise SearchError(error_msg) from e
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get embedding cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return self._semantic_search.get_cache_stats()
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._semantic_search.clear_cache()

