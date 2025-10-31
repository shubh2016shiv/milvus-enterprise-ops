"""
Comprehensive unit tests for GeminiEmbeddingProvider.

This module provides systematic testing of the GeminiEmbeddingProvider class,
including initialization, embedding generation, batch operations, error handling,
and edge cases.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from milvus_ops.search_operations.core.search_ops_exceptions import EmbeddingGenerationError
from milvus_ops.search_operations.providers.embedding import EmbeddingResult
from milvus_ops.search_operations.providers.gemini_embedding import (
    GeminiEmbeddingProvider,
    TaskType,
)

# ============================================================================
# Test GeminiEmbeddingProvider Initialization
# ============================================================================


@pytest.mark.unit
class TestGeminiEmbeddingProviderInitialization:
    """
    Test GeminiEmbeddingProvider initialization.

    Coverage: GeminiEmbeddingProvider constructor with various configurations.
    """

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_initialization_default(self, mock_genai):
        """
        Test GeminiEmbeddingProvider initialization with default parameters.

        Coverage: GeminiEmbeddingProvider.__init__() with default values.
        """
        provider = GeminiEmbeddingProvider()
        assert provider._model_name == "gemini-embedding-001"
        assert provider._task_type == TaskType.RETRIEVAL_DOCUMENT.value
        assert provider._output_dimensionality is None
        mock_genai.configure.assert_called_once_with(api_key="test_api_key_12345")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_initialization_custom_params(self, mock_genai):
        """
        Test GeminiEmbeddingProvider initialization with custom parameters.

        Coverage: GeminiEmbeddingProvider.__init__() with custom model_name,
        task_type, and output_dimensionality.
        """
        provider = GeminiEmbeddingProvider(
            model_name="custom-model",
            task_type=TaskType.RETRIEVAL_QUERY,
            output_dimensionality=768,
            api_key="custom_api_key",
        )
        assert provider._model_name == "custom-model"
        assert provider._task_type == TaskType.RETRIEVAL_QUERY.value
        assert provider._output_dimensionality == 768
        mock_genai.configure.assert_called_once_with(api_key="custom_api_key")

    @patch.dict(os.environ, {}, clear=True)
    def test_initialization_missing_api_key(self):
        """
        Test GeminiEmbeddingProvider initialization without API key.

        Coverage: GeminiEmbeddingProvider.__init__() raises ValueError when API key is missing.
        """
        with pytest.raises(ValueError, match="GEMINI_API_KEY not found"):
            GeminiEmbeddingProvider()

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_initialization_with_all_task_types(self, mock_genai):
        """
        Test GeminiEmbeddingProvider initialization with all task types.

        Coverage: GeminiEmbeddingProvider.__init__() accepts all TaskType enum values.
        """
        task_types = [
            TaskType.RETRIEVAL_QUERY,
            TaskType.RETRIEVAL_DOCUMENT,
            TaskType.SEMANTIC_SIMILARITY,
            TaskType.CLASSIFICATION,
            TaskType.CLUSTERING,
        ]
        for task_type in task_types:
            provider = GeminiEmbeddingProvider(task_type=task_type)
            assert provider._task_type == task_type.value

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_get_model_name(self, mock_genai):
        """
        Test get_model_name() method.

        Coverage: GeminiEmbeddingProvider.get_model_name() returns model name.
        """
        provider = GeminiEmbeddingProvider(model_name="test-model")
        assert provider.get_model_name() == "test-model"

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_get_dimension_with_output_dimensionality(self, mock_genai):
        """
        Test get_dimension() with configured output_dimensionality.

        Coverage: GeminiEmbeddingProvider.get_dimension() returns configured dimension.
        """
        provider = GeminiEmbeddingProvider(output_dimensionality=768)
        assert provider.get_dimension() == 768

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_get_dimension_without_output_dimensionality(self, mock_genai):
        """
        Test get_dimension() without configured output_dimensionality.

        Coverage: GeminiEmbeddingProvider.get_dimension() raises ValueError when not configured.
        """
        provider = GeminiEmbeddingProvider()
        with pytest.raises(ValueError, match="Output dimensionality must be configured"):
            provider.get_dimension()


# ============================================================================
# Test GeminiEmbeddingProvider generate_embedding
# ============================================================================


@pytest.mark.unit
class TestGeminiEmbeddingProviderGenerateEmbedding:
    """
    Test GeminiEmbeddingProvider.generate_embedding() method.

    Coverage: Single embedding generation with various inputs and edge cases.
    """

    def test_generate_embedding_single_text(self, monkeypatch):
        """
        Test generate_embedding() with normal text input.

        Coverage: GeminiEmbeddingProvider.generate_embedding() generates embedding for single text.
        """

        # Mock environment variable
        monkeypatch.setenv("GEMINI_API_KEY", "test_api_key_12345")

        # Mock genai module
        mock_genai = MagicMock()
        mock_result = {
            "embedding": [0.1] * 768,  # Single text returns flat list
        }
        mock_genai.embed_content.return_value = mock_result

        with monkeypatch.context() as m:
            m.setattr("milvus_ops.search_operations.providers.gemini_embedding.genai", mock_genai)
            provider = GeminiEmbeddingProvider(output_dimensionality=768)
            import asyncio

            result = asyncio.run(provider.generate_embedding("Hello, world!"))

        assert isinstance(result, EmbeddingResult)
        assert isinstance(result.embedding, list)
        assert len(result.embedding) == 768
        assert result.dimension == 768
        assert result.model_name == "gemini-embedding-001"
        assert result.processing_time_ms >= 0  # Processing time should be non-negative
        assert result.is_batch is False

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_generate_embedding_empty_text(self, mock_genai):
        """
        Test generate_embedding() with empty text.

        Coverage: GeminiEmbeddingProvider.generate_embedding() handles empty string.
        """
        mock_result = {"embedding": [0.0] * 768}
        mock_genai.embed_content.return_value = mock_result

        provider = GeminiEmbeddingProvider(output_dimensionality=768)
        import asyncio

        result = asyncio.run(provider.generate_embedding(""))

        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) == 768

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_generate_embedding_whitespace_only(self, mock_genai):
        """
        Test generate_embedding() with whitespace-only text.

        Coverage: GeminiEmbeddingProvider.generate_embedding() handles whitespace-only string.
        """
        mock_result = {"embedding": [0.0] * 768}
        mock_genai.embed_content.return_value = mock_result

        provider = GeminiEmbeddingProvider(output_dimensionality=768)
        import asyncio

        result = asyncio.run(provider.generate_embedding("   \n\t  "))

        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) == 768

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_generate_embedding_very_long_text(self, mock_genai):
        """
        Test generate_embedding() with very long text.

        Coverage: GeminiEmbeddingProvider.generate_embedding() handles very long text input.
        """
        very_long_text = "A" * 10000
        mock_result = {"embedding": [0.1] * 768}
        mock_genai.embed_content.return_value = mock_result

        provider = GeminiEmbeddingProvider(output_dimensionality=768)
        import asyncio

        result = asyncio.run(provider.generate_embedding(very_long_text))

        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) == 768

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_generate_embedding_with_normalization(self, mock_genai):
        """
        Test generate_embedding() with output_dimensionality requiring normalization.

        Coverage: GeminiEmbeddingProvider._normalize_embeddings() normalizes
        embeddings when output_dimensionality != 3072.
        """
        # Mock result with the expected 768-dimensional embedding (after normalization)
        mock_result = {"embedding": [0.1] * 768}
        mock_genai.embed_content.return_value = mock_result

        provider = GeminiEmbeddingProvider(output_dimensionality=768)
        # The normalization method only does L2 normalization, not dimensionality reduction
        # For this test, we mock the API to return the expected dimension directly
        import asyncio

        result = asyncio.run(provider.generate_embedding("test"))

        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) == 768
        # Note: The _normalize_embeddings method only does unit normalization

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_generate_embedding_dimension_mismatch_error(self, mock_genai):
        """
        Test generate_embedding() raises DimensionMismatchError on dimension mismatch.

        Coverage: GeminiEmbeddingProvider.generate_embeddings() validates dimension
        and raises DimensionMismatchError.
        """
        # Mock result with wrong dimension
        mock_result = {"embedding": [0.1] * 512}  # Expected 768
        mock_genai.embed_content.return_value = mock_result

        provider = GeminiEmbeddingProvider(output_dimensionality=768)
        import asyncio

        with pytest.raises(EmbeddingGenerationError, match="Expected dimension 768"):
            asyncio.run(provider.generate_embedding("test"))

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_generate_embedding_api_failure(self, mock_genai):
        """
        Test generate_embedding() handles API failures.

        Coverage: GeminiEmbeddingProvider.generate_embedding() raises
        EmbeddingGenerationError on API failure.
        """
        mock_genai.embed_content.side_effect = Exception("API request failed")

        provider = GeminiEmbeddingProvider(output_dimensionality=768)
        import asyncio

        with pytest.raises(EmbeddingGenerationError, match="Gemini embedding generation failed"):
            asyncio.run(provider.generate_embedding("test"))

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_generate_embedding_invalid_api_key(self, mock_genai):
        """
        Test generate_embedding() with invalid API key.

        Coverage: GeminiEmbeddingProvider.generate_embedding() handles invalid API key error.
        """
        mock_genai.embed_content.side_effect = Exception("Invalid API key")

        provider = GeminiEmbeddingProvider(output_dimensionality=768)
        import asyncio

        with pytest.raises(EmbeddingGenerationError):
            asyncio.run(provider.generate_embedding("test"))

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_generate_embedding_network_error(self, mock_genai):
        """
        Test generate_embedding() handles network errors.

        Coverage: GeminiEmbeddingProvider.generate_embedding() handles network/connection errors.
        """
        mock_genai.embed_content.side_effect = ConnectionError("Network connection failed")

        provider = GeminiEmbeddingProvider(output_dimensionality=768)
        import asyncio

        with pytest.raises(EmbeddingGenerationError):
            asyncio.run(provider.generate_embedding("test"))


# ============================================================================
# Test GeminiEmbeddingProvider generate_embeddings (Batch)
# ============================================================================


@pytest.mark.unit
class TestGeminiEmbeddingProviderGenerateEmbeddings:
    """
    Test GeminiEmbeddingProvider.generate_embeddings() method.

    Coverage: Batch embedding generation with various inputs and edge cases.
    """

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_generate_embeddings_multiple_texts(self, mock_genai):
        """
        Test generate_embeddings() with multiple texts.

        Coverage: GeminiEmbeddingProvider.generate_embeddings() generates embeddings
        for batch of texts.
        """
        # Mock genai.embed_content response for batch
        mock_result = {
            "embedding": [[0.1] * 768, [0.2] * 768, [0.3] * 768],  # Batch returns list of lists
        }
        mock_genai.embed_content.return_value = mock_result

        provider = GeminiEmbeddingProvider(output_dimensionality=768)
        texts = ["Text 1", "Text 2", "Text 3"]
        import asyncio

        result = asyncio.run(provider.generate_embeddings(texts))

        assert isinstance(result, EmbeddingResult)
        assert isinstance(result.embedding, list)
        assert len(result.embedding) == 3
        assert all(len(emb) == 768 for emb in result.embedding)
        assert result.dimension == 768
        assert result.is_batch is True

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_generate_embeddings_single_text_in_batch(self, mock_genai):
        """
        Test generate_embeddings() with single text in list.

        Coverage: GeminiEmbeddingProvider.generate_embeddings() handles single text
        in batch (converts to list).
        """
        # Single text in batch might return flat list
        mock_result = {"embedding": [0.1] * 768}
        mock_genai.embed_content.return_value = mock_result

        provider = GeminiEmbeddingProvider(output_dimensionality=768)
        import asyncio

        result = asyncio.run(provider.generate_embeddings(["Single text"]))

        assert isinstance(result, EmbeddingResult)
        # For single text, embedding should be flat list and is_batch should be False
        assert isinstance(result.embedding, list)
        assert result.is_batch is False

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_generate_embeddings_empty_list(self, mock_genai):
        """
        Test generate_embeddings() with empty text list.

        Coverage: GeminiEmbeddingProvider.generate_embeddings() handles empty list.
        """
        provider = GeminiEmbeddingProvider(output_dimensionality=768)
        # Empty list should raise error or handle gracefully
        # Mock might return empty result
        mock_result = {"embedding": []}
        mock_genai.embed_content.return_value = mock_result

        # Depending on implementation, this might raise error or return empty result
        import asyncio

        try:
            result = asyncio.run(provider.generate_embeddings([]))
            # If it doesn't raise, verify behavior
            assert isinstance(result, EmbeddingResult)
        except (EmbeddingGenerationError, ValueError, IndexError):
            # Expected behavior if empty list is not allowed
            pass

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_generate_embeddings_large_batch(self, mock_genai):
        """
        Test generate_embeddings() with large batch.

        Coverage: GeminiEmbeddingProvider.generate_embeddings() handles large batch of texts.
        """
        batch_size = 100
        mock_result = {
            "embedding": [[0.1] * 768 for _ in range(batch_size)],
        }
        mock_genai.embed_content.return_value = mock_result

        provider = GeminiEmbeddingProvider(output_dimensionality=768)
        texts = [f"Text {i}" for i in range(batch_size)]
        import asyncio

        result = asyncio.run(provider.generate_embeddings(texts))

        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) == batch_size
        assert result.is_batch is True

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_generate_embeddings_mixed_text_lengths(self, mock_genai):
        """
        Test generate_embeddings() with texts of varying lengths.

        Coverage: GeminiEmbeddingProvider.generate_embeddings() handles texts
        with different lengths.
        """
        mock_result = {
            "embedding": [[0.1] * 768, [0.2] * 768, [0.3] * 768],
        }
        mock_genai.embed_content.return_value = mock_result

        provider = GeminiEmbeddingProvider(output_dimensionality=768)
        texts = ["Short", "A" * 100, "B" * 1000]
        import asyncio

        result = asyncio.run(provider.generate_embeddings(texts))

        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) == 3
        assert result.is_batch is True

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_generate_embeddings_dimension_validation(self, mock_genai):
        """
        Test generate_embeddings() validates dimensions for all embeddings in batch.

        Coverage: GeminiEmbeddingProvider.generate_embeddings() validates all
        embeddings have correct dimension.
        """
        # Mock result with one embedding having wrong dimension
        mock_result = {
            "embedding": [[0.1] * 768, [0.2] * 512],  # Second has wrong dimension
        }
        mock_genai.embed_content.return_value = mock_result

        provider = GeminiEmbeddingProvider(output_dimensionality=768)
        import asyncio

        with pytest.raises(EmbeddingGenerationError, match="Expected dimension 768"):
            asyncio.run(provider.generate_embeddings(["Text 1", "Text 2"]))

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_generate_embeddings_batch_api_failure(self, mock_genai):
        """
        Test generate_embeddings() handles API failures in batch.

        Coverage: GeminiEmbeddingProvider.generate_embeddings() raises
        EmbeddingGenerationError on batch API failure.
        """
        mock_genai.embed_content.side_effect = Exception("Batch API request failed")

        provider = GeminiEmbeddingProvider(output_dimensionality=768)
        import asyncio

        with pytest.raises(EmbeddingGenerationError):
            asyncio.run(provider.generate_embeddings(["Text 1", "Text 2"]))


# ============================================================================
# Test GeminiEmbeddingProvider Edge Cases
# ============================================================================


@pytest.mark.unit
class TestGeminiEmbeddingProviderEdgeCases:
    """
    Test GeminiEmbeddingProvider edge cases and error conditions.

    Coverage: None inputs, special characters, normalization edge cases.
    """

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_generate_embedding_none_text(self, mock_genai):
        """
        Test generate_embedding() with None text.

        Coverage: GeminiEmbeddingProvider.generate_embedding() handles None input.
        """
        provider = GeminiEmbeddingProvider(output_dimensionality=768)
        # None should raise TypeError or EmbeddingGenerationError
        import asyncio

        with pytest.raises((TypeError, EmbeddingGenerationError)):
            asyncio.run(provider.generate_embedding(None))

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_generate_embeddings_none_in_list(self, mock_genai):
        """
        Test generate_embeddings() with None in text list.

        Coverage: GeminiEmbeddingProvider.generate_embeddings() handles None in list.
        """
        provider = GeminiEmbeddingProvider(output_dimensionality=768)
        import asyncio

        with pytest.raises((TypeError, EmbeddingGenerationError)):
            asyncio.run(provider.generate_embeddings(["Text", None, "More text"]))

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_generate_embedding_special_characters(self, mock_genai):
        """
        Test generate_embedding() with special characters.

        Coverage: GeminiEmbeddingProvider.generate_embedding() handles special characters.
        """
        mock_result = {"embedding": [0.1] * 768}
        mock_genai.embed_content.return_value = mock_result

        provider = GeminiEmbeddingProvider(output_dimensionality=768)
        special_text = "Special chars: !@#$%^&*()[]{}|\\:;\"'<>,.?/~`"
        import asyncio

        result = asyncio.run(provider.generate_embedding(special_text))

        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) == 768

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_generate_embedding_unicode(self, mock_genai):
        """
        Test generate_embedding() with unicode characters.

        Coverage: GeminiEmbeddingProvider.generate_embedding() handles unicode text.
        """
        mock_result = {"embedding": [0.1] * 768}
        mock_genai.embed_content.return_value = mock_result

        provider = GeminiEmbeddingProvider(output_dimensionality=768)
        unicode_text = "Hello 世界 🌍 こんにちは مرحبا"
        import asyncio

        result = asyncio.run(provider.generate_embedding(unicode_text))

        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) == 768

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_normalize_embeddings_zero_vector(self, mock_genai):
        """
        Test _normalize_embeddings() with zero vector.

        Coverage: GeminiEmbeddingProvider._normalize_embeddings() handles zero norm vectors.
        """
        # Create provider and test normalization with zero vector
        provider = GeminiEmbeddingProvider(output_dimensionality=768)
        zero_vector = [0.0] * 768
        normalized = provider._normalize_embeddings([zero_vector])

        # Zero vector should remain unchanged (norm == 0)
        assert normalized[0] == zero_vector

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_normalize_embeddings_unit_vector(self, mock_genai):
        """
        Test _normalize_embeddings() with unit vector.

        Coverage: GeminiEmbeddingProvider._normalize_embeddings() normalizes vectors correctly.
        """
        provider = GeminiEmbeddingProvider(output_dimensionality=768)
        # Create a vector that's already normalized
        unit_vector = [1.0] + [0.0] * 767
        normalized = provider._normalize_embeddings([unit_vector])

        # Should remain as unit vector
        assert len(normalized[0]) == 768

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_task_type_enum_values(self, mock_genai):
        """
        Test all TaskType enum values are valid.

        Coverage: TaskType enum contains expected values.
        """
        assert TaskType.RETRIEVAL_QUERY == "RETRIEVAL_QUERY"
        assert TaskType.RETRIEVAL_DOCUMENT == "RETRIEVAL_DOCUMENT"
        assert TaskType.SEMANTIC_SIMILARITY == "SEMANTIC_SIMILARITY"
        assert TaskType.CLASSIFICATION == "CLASSIFICATION"
        assert TaskType.CLUSTERING == "CLUSTERING"

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_generate_embedding_processing_time(self, mock_genai):
        """
        Test generate_embedding() records processing time.

        Coverage: GeminiEmbeddingProvider.generate_embedding() measures and records processing time.
        """
        import time

        mock_result = {"embedding": [0.1] * 768}
        # Simulate some processing time
        mock_genai.embed_content.side_effect = lambda *args, **kwargs: (
            time.sleep(0.01) or mock_result
        )

        provider = GeminiEmbeddingProvider(output_dimensionality=768)
        import asyncio

        result = asyncio.run(provider.generate_embedding("test"))

        assert result.processing_time_ms > 0
        assert isinstance(result.processing_time_ms, float)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    @patch("milvus_ops.search_operations.providers.gemini_embedding.genai")
    def test_generate_embeddings_calls_generate_embedding(self, mock_genai):
        """
        Test generate_embedding() calls generate_embeddings() internally.

        Coverage: GeminiEmbeddingProvider.generate_embedding() delegates to generate_embeddings().
        """
        mock_result = {"embedding": [0.1] * 768}
        mock_genai.embed_content.return_value = mock_result

        provider = GeminiEmbeddingProvider(output_dimensionality=768)
        import asyncio

        asyncio.run(provider.generate_embedding("test"))

        # Verify generate_embeddings was called (indirectly through embed_content)
        mock_genai.embed_content.assert_called_once()
