#!/usr/bin/env python3
"""
Gemini Embeddings Example

This example demonstrates how to use the GeminiEmbeddingProvider
for generating text embeddings using Google's Gemini API.

Requirements:
- GEMINI_API_KEY environment variable set
- google-generativeai library installed
- numpy library installed
"""

import asyncio
import os
import sys
import logging
from typing import List

# Suppress Google library warnings
logging.getLogger('google.generativeai').setLevel(logging.ERROR)
logging.getLogger('google').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# Add the parent directory to the path to import search_operations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_operations.providers.gemini_embedding import (
    GeminiEmbeddingProvider, 
    TaskType,
    DimensionMismatchError
)
from search_operations.core.exceptions import EmbeddingGenerationError


async def test_single_embedding():
    """Test generating a single text embedding."""
    print("Testing single embedding generation...")
    
    provider = GeminiEmbeddingProvider(
        output_dimensionality=768,
        task_type=TaskType.RETRIEVAL_DOCUMENT
    )
    
    text = "This is a sample document for embedding generation."
    result = await provider.generate_embedding(text)
    
    print(f"Single embedding generated successfully:")
    print(f"  - Dimension: {result.dimension}")
    print(f"  - Model: {result.model_name}")
    print(f"  - Processing time: {result.processing_time_ms:.2f}ms")
    print(f"  - Embedding vector length: {len(result.embedding)}")
    print(f"  - First 5 values: {result.embedding[:5]}")
    print()


async def test_batch_embeddings():
    """Test generating embeddings for multiple texts."""
    print("Testing batch embedding generation...")
    
    provider = GeminiEmbeddingProvider(
        output_dimensionality=1536,
        task_type=TaskType.SEMANTIC_SIMILARITY
    )
    
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand text.",
        "Computer vision enables machines to interpret visual information."
    ]
    
    result = await provider.generate_embeddings(texts)
    
    print(f"Batch embeddings generated successfully:")
    print(f"  - Number of texts: {len(texts)}")
    print(f"  - Dimension: {result.dimension}")
    print(f"  - Model: {result.model_name}")
    print(f"  - Processing time: {result.processing_time_ms:.2f}ms")
    print(f"  - Is batch: {result.is_batch}")
    print(f"  - Embedding vectors count: {len(result.embedding)}")
    print()


async def test_different_task_types():
    """Test different task types for embedding optimization."""
    print("Testing different task types...")
    
    text = "What is the capital of France?"
    
    task_types = [
        (TaskType.RETRIEVAL_QUERY, "Retrieval Query"),
        (TaskType.RETRIEVAL_DOCUMENT, "Retrieval Document"),
        (TaskType.SEMANTIC_SIMILARITY, "Semantic Similarity"),
        (TaskType.CLASSIFICATION, "Classification")
    ]
    
    for task_type, description in task_types:
        provider = GeminiEmbeddingProvider(
            output_dimensionality=512,
            task_type=task_type
        )
        
        result = await provider.generate_embedding(text)
        print(f"  - {description}: {result.processing_time_ms:.2f}ms")
    
    print()


async def test_different_dimensions():
    """Test different embedding dimensions."""
    print("Testing different embedding dimensions...")
    
    text = "Testing various embedding dimensions for performance comparison."
    dimensions = [128, 256, 512, 768, 1536, 3072]
    
    for dim in dimensions:
        try:
            provider = GeminiEmbeddingProvider(
                output_dimensionality=dim,
                task_type=TaskType.RETRIEVAL_DOCUMENT
            )
            
            result = await provider.generate_embedding(text)
            print(f"  - Dimension {dim}: {result.processing_time_ms:.2f}ms")
            
        except Exception as e:
            print(f"  - Dimension {dim}: Failed - {str(e)}")
    
    print()


async def test_error_handling():
    """Test error handling scenarios."""
    print("Testing error handling...")
    
    # Test with invalid API key
    try:
        provider = GeminiEmbeddingProvider(
            api_key="invalid_key",
            output_dimensionality=768
        )
        await provider.generate_embedding("Test text")
        print("  - Invalid API key: Unexpected success")
    except Exception as e:
        print(f"  - Invalid API key: Correctly caught error - {type(e).__name__}")
    
    # Test without output dimensionality
    try:
        provider = GeminiEmbeddingProvider(
            task_type=TaskType.RETRIEVAL_DOCUMENT
        )
        # This should fail when trying to get dimension
        dimension = provider.get_dimension()
        print("  - No output dimensionality: Unexpected success")
    except ValueError as e:
        print(f"  - No output dimensionality: Correctly caught error - {str(e)}")
    
    print()


def check_environment():
    """Check if the required environment is set up correctly."""
    print("Checking environment setup...")
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not found")
        print("Please set your Gemini API key in the environment or .env file")
        return False
    else:
        print(f"API key found: {api_key[:10]}...")
    
    # Check required libraries
    try:
        import google.generativeai
        print("google-generativeai library: Available")
    except ImportError:
        print("ERROR: google-generativeai library not installed")
        return False
    
    try:
        import numpy
        print("numpy library: Available")
    except ImportError:
        print("ERROR: numpy library not installed")
        return False
    
    print("Environment setup: OK")
    print()
    return True


async def main():
    """Main function to run all examples."""
    print("Gemini Embeddings Example")
    print("=" * 50)
    print()
    
    # Check environment first
    if not check_environment():
        sys.exit(1)
    
    try:
        # Run all test scenarios
        await test_single_embedding()
        await test_batch_embeddings()
        await test_different_task_types()
        await test_different_dimensions()
        await test_error_handling()
        
        print("All examples completed successfully!")
        
    except EmbeddingGenerationError as e:
        print(f"Embedding generation error: {e}")
        sys.exit(1)
    except DimensionMismatchError as e:
        print(f"Dimension mismatch error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
