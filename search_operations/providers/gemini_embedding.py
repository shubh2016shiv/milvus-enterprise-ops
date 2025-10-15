"""
Gemini Embedding Provider

This module provides a concrete implementation of the EmbeddingProvider
interface for Google's Gemini models.
"""

import os
import time
from enum import Enum
from typing import List, Union, Optional

import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

from .embedding import EmbeddingProvider, EmbeddingResult
from ..core.search_ops_exceptions import EmbeddingGenerationError

# Load environment variables from a .env file
load_dotenv()

class TaskType(str, Enum):
    """Enumeration of supported task types for Gemini embeddings."""
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    CLASSIFICATION = "CLASSIFICATION"
    CLUSTERING = "CLUSTERING"

class DimensionMismatchError(ValueError):
    """Custom exception for embedding dimension mismatches."""
    pass

class GeminiEmbeddingProvider(EmbeddingProvider):
    """
    An implementation of EmbeddingProvider that uses the Gemini API.
    
    This provider handles the generation of embeddings using Google's
    Gemini models, including configuration for different task types
    and output dimensions.
    """

    def __init__(
        self,
        model_name: str = "gemini-embedding-001",
        task_type: TaskType = TaskType.RETRIEVAL_DOCUMENT,
        output_dimensionality: Optional[int] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the Gemini embedding provider.

        Args:
            model_name: The name of the Gemini embedding model to use.
            task_type: The intended task for the embeddings.
            output_dimensionality: The desired dimension of the output embeddings.
            api_key: The Gemini API key. If not provided, it will be
                     loaded from the GEMINI_API_KEY environment variable.
        """
        self._model_name = model_name
        self._task_type = task_type.value
        self._output_dimensionality = output_dimensionality

        gemini_api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables or provided directly.")
        
        genai.configure(api_key=gemini_api_key)

    async def generate_embedding(self, text: str) -> EmbeddingResult:
        """
        Generate an embedding for a single piece of text.

        Args:
            text: The text to generate an embedding for.

        Returns:
            An EmbeddingResult containing the generated embedding.
        
        Raises:
            EmbeddingGenerationError: If the embedding generation fails.
        """
        return await self.generate_embeddings([text])

    async def generate_embeddings(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: A list of texts to generate embeddings for.

        Returns:
            An EmbeddingResult containing the batch of generated embeddings.
            
        Raises:
            EmbeddingGenerationError: If the embedding generation fails.
        """
        start_time = time.time()
        try:
            result = genai.embed_content(
                model=self._model_name,
                content=texts,
                task_type=self._task_type,
                output_dimensionality=self._output_dimensionality
            )
            
            embeddings = result['embedding']
            if not isinstance(embeddings[0], list): # Handle single text case
                 embeddings = [embeddings]

            # Normalize embeddings if dimensionality is not 3072
            if self._output_dimensionality and self._output_dimensionality != 3072:
                embeddings = self._normalize_embeddings(embeddings)

            processing_time_ms = (time.time() - start_time) * 1000
            
            # Check if all embeddings have the correct dimension
            if self._output_dimensionality:
                for emb in embeddings:
                    if len(emb) != self._output_dimensionality:
                        raise DimensionMismatchError(
                            f"Expected dimension {self._output_dimensionality}, but got {len(emb)}"
                        )

            return EmbeddingResult(
                embedding=embeddings if len(texts) > 1 else embeddings[0],
                dimension=self.get_dimension(),
                model_name=self.get_model_name(),
                processing_time_ms=processing_time_ms,
                is_batch=len(texts) > 1
            )

        except Exception as e:
            raise EmbeddingGenerationError(f"Gemini embedding generation failed: {e}") from e

    def get_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns the configured output dimensionality, or raises an error if it's not set.
        """
        if self._output_dimensionality is None:
            # You might want to query the model for its default dimension if not set
            raise ValueError("Output dimensionality must be configured for this provider.")
        return self._output_dimensionality

    def get_model_name(self) -> str:
        """Get the name of the embedding model."""
        return self._model_name
        
    def _normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Normalize embeddings to unit length."""
        normed_embeddings = []
        for emb in embeddings:
            np_emb = np.array(emb)
            norm = np.linalg.norm(np_emb)
            if norm == 0:
                normed_embeddings.append(emb)
            else:
                normed_embeddings.append((np_emb / norm).tolist())
        return normed_embeddings
    
if __name__ == "__main__":
    import asyncio
    
    async def test_provider():
        provider = GeminiEmbeddingProvider(
            output_dimensionality=768,
            task_type=TaskType.RETRIEVAL_DOCUMENT
        )
        result = await provider.generate_embedding("Hello, world!")
        print(f"Embedding dimension: {result.dimension}")
        print(f"Model: {result.model_name}")
        print(f"Processing time: {result.processing_time_ms:.2f}ms")
        print(f"Embedding length: {len(result.embedding)}")
    
    asyncio.run(test_provider())
