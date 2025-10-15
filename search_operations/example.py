"""
Example usage of search operations module.

This module demonstrates how to use the search operations module
for different search types and configurations.
"""

import asyncio
import logging
from typing import List, Dict, Any

from connection_management import MilvusConnector
from search_operations import (
    SearchManager,
    SearchType,
    MetricType,
    ReRankingMethod,
    SearchParams,
    EmbeddingProvider,
    EmbeddingResult
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Example embedding provider implementation
class SimpleEmbeddingProvider(EmbeddingProvider):
    """Simple embedding provider for demonstration purposes."""
    
    def __init__(self, dimension: int = 128):
        """Initialize with specified dimension."""
        self.dimension = dimension
        self.model_name = "example-model"
    
    async def generate_embedding(self, text: str) -> EmbeddingResult:
        """Generate a simple embedding for demonstration."""
        import random
        import time
        
        # Simulate processing time
        start_time = time.time()
        
        # Generate random embedding vector
        embedding = [random.random() for _ in range(self.dimension)]
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        return EmbeddingResult(
            embedding=embedding,
            dimension=self.dimension,
            model_name=self.model_name,
            processing_time_ms=processing_time_ms
        )
    
    async def generate_embeddings(self, texts: List[str]) -> EmbeddingResult:
        """Generate embeddings for multiple texts."""
        import random
        import time
        
        # Simulate processing time
        start_time = time.time()
        
        # Generate random embedding vectors
        embeddings = [
            [random.random() for _ in range(self.dimension)]
            for _ in texts
        ]
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        return EmbeddingResult(
            embedding=embeddings,
            dimension=self.dimension,
            model_name=self.model_name,
            processing_time_ms=processing_time_ms,
            is_batch=True
        )
    
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        return self.dimension
    
    def get_model_name(self) -> str:
        """Return the model name."""
        return self.model_name


async def run_example():
    """Run example search operations."""
    try:
        # Initialize connection
        connector = MilvusConnector()
        connection_manager = await connector.establish_connection()
        
        # Initialize embedding provider
        embedding_provider = SimpleEmbeddingProvider(dimension=128)
        
        # Initialize search manager
        search_manager = SearchManager(
            connection_manager=connection_manager,
            embedding_provider=embedding_provider,
            enable_caching=True
        )
        
        # Collection to search
        collection_name = "example_collection"
        
        # Example query
        query = "What is vector search?"
        
        # Example 1: Simple semantic search
        logger.info("Example 1: Simple semantic search")
        semantic_params = SearchParams(
            search_type=SearchType.SEMANTIC,
            top_k=10,
            metric_type=MetricType.COSINE,
            vector_field="vector",
            params={"nprobe": 10, "ef": 64}
        )
        
        try:
            semantic_results = await search_manager.search(
                collection_name=collection_name,
                query=query,
                search_params=semantic_params
            )
            
            logger.info(f"Found {len(semantic_results.hits)} results in {semantic_results.took_ms:.2f}ms")
            
            # Display top 3 results
            for i, hit in enumerate(semantic_results.hits[:3]):
                logger.info(f"Result {i+1}: ID={hit['id']}, Score={hit['score']:.4f}")
                
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
        
        # Example 2: Semantic search with weighted re-ranking
        logger.info("\nExample 2: Semantic search with weighted re-ranking")
        reranking_params = SearchParams(
            search_type=SearchType.SEMANTIC,
            top_k=20,
            metric_type=MetricType.COSINE,
            vector_field="vector",
            params={"nprobe": 10, "ef": 64},
            rerank=True,
            rerank_method=ReRankingMethod.WEIGHTED,
            rerank_weights=[0.7, 0.3]  # Weights for different vector fields
        )
        
        try:
            reranked_results = await search_manager.search(
                collection_name=collection_name,
                query=query,
                search_params=reranking_params
            )
            
            logger.info(f"Found {len(reranked_results.hits)} results in {reranked_results.took_ms:.2f}ms")
            
            # Display top 3 results
            for i, hit in enumerate(reranked_results.hits[:3]):
                logger.info(
                    f"Result {i+1}: ID={hit['id']}, "
                    f"Score={hit['score']:.4f}"
                )
                
        except Exception as e:
            logger.error(f"Semantic search with re-ranking failed: {e}")
        
        # Example 3: Hybrid search with RRF reranking
        logger.info("\nExample 3: Hybrid search with RRF reranking")
        hybrid_params = SearchParams(
            search_type=SearchType.HYBRID,
            top_k=10,
            metric_type=MetricType.COSINE,
            vector_field="vector",
            sparse_field="sparse_vector",
            vector_weight=0.7,
            sparse_weight=0.3,
            params={"nprobe": 10, "ef": 64},
            rerank=True,
            rerank_method=ReRankingMethod.RRF,
            rerank_k=60  # RRF constant
        )
        
        try:
            hybrid_results = await search_manager.search(
                collection_name=collection_name,
                query=query,
                search_params=hybrid_params
            )
            
            logger.info(f"Found {len(hybrid_results.hits)} results in {hybrid_results.took_ms:.2f}ms")
            
            # Display top 3 results
            for i, hit in enumerate(hybrid_results.hits[:3]):
                logger.info(f"Result {i+1}: ID={hit['id']}, Score={hit['score']:.4f}")
                
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
        
        # Get cache statistics
        cache_stats = search_manager.get_cache_stats()
        logger.info(f"\nEmbedding cache stats: {cache_stats}")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
    finally:
        # Close connection
        if 'connection_manager' in locals():
            await connection_manager.close()


if __name__ == "__main__":
    asyncio.run(run_example())
