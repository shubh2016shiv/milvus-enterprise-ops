# Milvus Search Operations

This module provides advanced, high-performance search capabilities for Milvus, offering semantic search, hybrid search, and intelligent result reranking with built-in optimization for enterprise-scale vector databases.

## Key Features

- **Unified search interface**: Single `SearchManager` class handles all search types seamlessly
- **Semantic search**: Dense vector similarity search with configurable metrics (L2, IP, COSINE, HAMMING)
- **Hybrid search**: Combines dense vectors with sparse retrieval (BM25-style) for superior results
- **Native reranking**: Built-in Milvus reranking using WEIGHTED and RRF methods
- **Embedding caching**: Reduces latency and API calls for repeated queries
- **Production-ready configurations**: Optimized defaults for enterprise workloads
- **Multiple embedding providers**: Support for Gemini and custom embedding providers
- **Search parameter optimization**: Fine-tuned parameters (nprobe, ef) for speed/accuracy tradeoffs
- **Comprehensive search metrics**: Detailed performance and result statistics
- **Error handling**: Specialized exception hierarchy for precise error management
- **Async/await support**: Full asynchronous operation support for high-concurrency environments
- **Flexible configuration**: SearchParams class for easy parameter management

## Usage Examples

### Basic Semantic Search

```python
from connection_management import ConnectionManager
from milvus_ops.search_operations import (
    SearchManager,
    SearchType,
    MetricType,
    SearchParams,
    GeminiEmbeddingProvider
)

# Initialize connection and search components
conn_manager = ConnectionManager()
embedding_provider = GeminiEmbeddingProvider(
    api_key="your-gemini-key",
    dimension=768
)

# Create search manager with caching enabled
search_manager = SearchManager(
    connection_manager=conn_manager,
    embedding_provider=embedding_provider,
    enable_caching=True
)

# Configure semantic search parameters
search_params = SearchParams(
    search_type=SearchType.SEMANTIC,
    top_k=10,
    metric_type=MetricType.COSINE,
    vector_field="vector",
    params={"nprobe": 10, "ef": 64}
)

# Perform semantic search
results = await search_manager.search(
    collection_name="my_documents",
    query="machine learning algorithms",
    search_params=search_params
)

print(f"Found {results.total_hits} results in {results.took_ms:.2f}ms")
for hit in results.hits[:3]:
    print(f"ID: {hit['id']}, Score: {hit['score']:.4f}")

# Clean up
search_manager.clear_cache()
conn_manager.close()
```

### Hybrid Search with Fusion

```python
# Configure hybrid search combining dense vectors and keyword matching
hybrid_params = SearchParams(
    search_type=SearchType.HYBRID,
    top_k=20,
    metric_type=MetricType.COSINE,
    vector_field="vector",
    sparse_field="sparse_vector",
    keyword_field="text",
    vector_weight=0.7,    # 70% weight for semantic similarity
    sparse_weight=0.3,    # 30% weight for keyword matching
    params={"nprobe": 15, "ef": 128}
)

# Execute hybrid search
results = await search_manager.search(
    collection_name="my_documents",
    query="deep learning frameworks comparison",
    search_params=hybrid_params
)

print(f"Hybrid search results: {results.total_hits} matches")
for hit in results.hits[:5]:
    print(f"Document {hit['id']}: {hit.get('text', '')[:100]}...")
```

### Semantic Search with Reranking

```python
# Enable Milvus native reranking for improved results
reranking_params = SearchParams(
    search_type=SearchType.SEMANTIC,
    top_k=50,
    metric_type=MetricType.COSINE,
    vector_field="vector",
    params={"nprobe": 20, "ef": 256},
    rerank=True,
    rerank_method=ReRankingMethod.WEIGHTED,
    rerank_weights=[0.7, 0.3]  # Weights for multi-vector scenarios
)

# Search with reranking
results = await search_manager.search(
    collection_name="my_documents",
    query="neural network architectures",
    search_params=reranking_params
)

# Results are automatically reranked using Milvus native capabilities
print(f"Reranked results: {results.total_hits} matches")
print(f"Search parameters: {results.search_params}")
```

### Advanced Hybrid Search with RRF Fusion

```python
# Hybrid search with Reciprocal Rank Fusion (RRF)
advanced_hybrid_params = SearchParams(
    search_type=SearchType.HYBRID,
    top_k=30,
    metric_type=MetricType.COSINE,
    vector_field="vector",
    sparse_field="sparse_vector",
    vector_weight=0.6,
    sparse_weight=0.4,
    params={"nprobe": 25, "ef": 512},
    rerank=True,
    rerank_method=ReRankingMethod.RRF,
    rerank_k=60  # RRF constant parameter
)

results = await search_manager.search(
    collection_name="my_documents",
    query="distributed computing patterns",
    search_params=advanced_hybrid_params
)

print(f"Advanced hybrid results: {results.total_hits} matches")
```

### Custom Embedding Provider

```python
from milvus_ops.search_operations import EmbeddingProvider, EmbeddingResult


class CustomEmbeddingProvider(EmbeddingProvider):
    """Custom embedding provider implementation."""

    def __init__(self, model_name: str = "custom-model", dimension: int = 512):
        self.model_name = model_name
        self.dimension = dimension

    async def generate_embedding(self, text: str) -> EmbeddingResult:
        # Your embedding generation logic here
        embedding_vector = await your_embedding_service.embed(text)

        return EmbeddingResult(
            embedding=embedding_vector,
            dimension=self.dimension,
            model_name=self.model_name,
            processing_time_ms=processing_time
        )


# Use custom provider
custom_provider = CustomEmbeddingProvider(dimension=512)
search_manager = SearchManager(
    connection_manager=conn_manager,
    embedding_provider=custom_provider,
    enable_caching=True
)
```

### Search Performance Monitoring

```python
# Get cache statistics for performance monitoring
cache_stats = search_manager.get_cache_stats()
print(f"Embedding cache hit rate: {cache_stats['hit_rate']:.2%}")
print(f"Cache size: {cache_stats['size']} entries")
print(f"Total embedding time saved: {cache_stats['time_saved_ms']:.2f}ms")

# Monitor search performance
results = await search_manager.search(
    collection_name="my_documents",
    query="performance optimization techniques",
    search_params=semantic_params
)

print(f"Search took {results.took_ms:.2f}ms")
print(f"Vector generation: {results.search_params.get('embedding_time_ms', 'N/A')}ms")
print(f"Search execution: {results.search_params.get('search_time_ms', 'N/A')}ms")
```

### Filtered Search with Metadata

```python
# Search with field filters
filtered_params = SearchParams(
    search_type=SearchType.SEMANTIC,
    top_k=10,
    metric_type=MetricType.COSINE,
    vector_field="vector",
    expr='category == "technical" && published_date >= "2023-01-01"',
    params={"nprobe": 10}
)

results = await search_manager.search(
    collection_name="my_documents",
    query="scalable system design",
    search_params=filtered_params
)

# Filtered results only include documents matching the criteria
print(f"Filtered results: {results.total_hits} matches")
```

## Error Handling

```python
from milvus_ops.search_operations import (
    SearchError,
    InvalidSearchParametersError,
    EmbeddingGenerationError,
    SearchTimeoutError,
    ReRankingError,
    EmptyResultError
)

try:
    results = await search_manager.search(
        collection_name="my_documents",
        query="machine learning",
        search_params=semantic_params
    )
except InvalidSearchParametersError as e:
    print(f"Invalid search configuration: {e}")
except EmbeddingGenerationError as e:
    print(f"Failed to generate embeddings: {e}")
except SearchTimeoutError as e:
    print(f"Search operation timed out: {e}")
except EmptyResultError:
    print("No results found for the query")
except SearchError as e:
    print(f"General search error: {e}")
```

## Configuration Reference

### SearchType
- **SEMANTIC**: Dense vector similarity search
- **HYBRID**: Combined dense + sparse search
- **FUSION**: Multiple search result fusion

### MetricType
- **L2**: Euclidean distance (default for numeric vectors)
- **IP**: Inner product (good for recommendation systems)
- **COSINE**: Cosine similarity (best for text embeddings)
- **HAMMING**: Hamming distance (for binary vectors)

### ReRankingMethod
- **NONE**: No additional reranking
- **WEIGHTED**: Weighted reranking for multi-vector scenarios
- **RRF**: Reciprocal Rank Fusion for result combination

### Optimization Parameters
- **nprobe**: Number of clusters to search (10-100, higher = more accurate, slower)
- **ef**: HNSW search parameter (higher = more accurate, slower, 32-1024)
- **top_k**: Number of results to return (1-10000)
- **timeout**: Search timeout in seconds (default: 30.0)

## Performance Considerations

This module is optimized for high-performance search scenarios:

- **Embedding caching** reduces API calls and latency for repeated queries
- **Connection pooling** ensures efficient database access
- **Async operations** enable high-concurrency workloads
- **Parameter tuning** allows balancing speed vs accuracy

For production environments with millions of queries:

1. **Enable embedding caching** to reduce external API calls
2. **Tune search parameters** (nprobe, ef) based on your accuracy requirements
3. **Monitor cache performance** using `get_cache_stats()`
4. **Use connection pooling** from connection_management for efficiency
5. **Implement proper error handling** for graceful degradation
6. **Consider hybrid search** for improved relevance over semantic-only search
7. **Use Milvus native reranking** for better results without additional latency

### Performance Tuning Guidelines

- **Speed Priority**: Use `nprobe=10, ef=64` for faster searches
- **Accuracy Priority**: Use `nprobe=50, ef=512` for better results
- **Balanced**: Default parameters provide good speed/accuracy tradeoff
- **High-Volume**: Enable caching and monitor cache hit rates
- **Real-time**: Consider hybrid search with appropriate weights (0.7 semantic, 0.3 sparse)

### Monitoring Best Practices

```python
# Monitor search performance over time
import time

start_time = time.time()
results = await search_manager.search(collection_name, query, params)
total_time = time.time() - start_time

print(f"End-to-end time: {total_time:.3f}s")
print(f"Search time: {results.took_ms:.2f}ms")

# Track cache performance
cache_stats = search_manager.get_cache_stats()
if cache_stats['hit_rate'] < 0.8:
    print("Warning: Low cache hit rate, consider increasing cache size")
```

This search operations module provides enterprise-grade search capabilities with comprehensive configuration options, intelligent caching, and production-ready performance optimizations for your Milvus vector database.