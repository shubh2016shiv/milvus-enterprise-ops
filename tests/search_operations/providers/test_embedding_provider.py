"""
Comprehensive unit tests for embedding provider abstract class.

This module provides systematic testing of EmbeddingProvider abstract class
and EmbeddingResult dataclass, including property access and abstract method enforcement.
"""

import pytest

from milvus_ops.search_operations.providers.embedding import (
    EmbeddingProvider,
    EmbeddingResult,
)

# ============================================================================
# Test EmbeddingResult Dataclass
# ============================================================================


@pytest.mark.unit
class TestEmbeddingResult:
    """
    Test EmbeddingResult dataclass initialization and properties.

    Coverage: EmbeddingResult dataclass properties and default values.
    """

    def test_embedding_result_initialization_single(self):
        """
        Test EmbeddingResult initialization with single embedding.

        Coverage: EmbeddingResult.__init__() with single embedding vector.
        """
        embedding = [0.1] * 128
        result = EmbeddingResult(
            embedding=embedding,
            dimension=128,
            model_name="test_model",
            processing_time_ms=10.0,
            is_batch=False,
        )
        assert result.embedding == embedding
        assert result.dimension == 128
        assert result.model_name == "test_model"
        assert result.processing_time_ms == 10.0
        assert result.is_batch is False
        assert isinstance(result.metadata, dict)

    def test_embedding_result_initialization_batch(self):
        """
        Test EmbeddingResult initialization with batch embeddings.

        Coverage: EmbeddingResult.__init__() with batch embeddings.
        """
        embeddings = [[0.1] * 128 for _ in range(3)]
        result = EmbeddingResult(
            embedding=embeddings,
            dimension=128,
            model_name="test_model",
            processing_time_ms=30.0,
            is_batch=True,
        )
        assert result.embedding == embeddings
        assert result.dimension == 128
        assert result.is_batch is True
        assert len(result.embedding) == 3

    def test_embedding_result_with_metadata(self):
        """
        Test EmbeddingResult initialization with custom metadata.

        Coverage: EmbeddingResult accepts custom metadata dictionary.
        """
        metadata = {"task_type": "retrieval", "normalized": True}
        result = EmbeddingResult(
            embedding=[0.1] * 128,
            dimension=128,
            model_name="test_model",
            processing_time_ms=10.0,
            metadata=metadata,
        )
        assert result.metadata == metadata

    def test_embedding_result_default_metadata(self):
        """
        Test EmbeddingResult initializes empty metadata by default.

        Coverage: EmbeddingResult.__post_init__() initializes empty metadata if not provided.
        """
        result = EmbeddingResult(
            embedding=[0.1] * 128,
            dimension=128,
            model_name="test_model",
            processing_time_ms=10.0,
        )
        assert isinstance(result.metadata, dict)
        assert len(result.metadata) == 0


# ============================================================================
# Test EmbeddingProvider Abstract Class
# ============================================================================


@pytest.mark.unit
class TestEmbeddingProvider:
    """
    Test EmbeddingProvider abstract class behavior.

    Coverage: EmbeddingProvider abstract method enforcement.
    """

    def test_embedding_provider_cannot_be_instantiated(self):
        """
        Test EmbeddingProvider cannot be instantiated directly.

        Coverage: EmbeddingProvider is abstract and requires implementation.
        """
        with pytest.raises(TypeError):
            EmbeddingProvider()

    def test_concrete_provider_must_implement_methods(self):
        """
        Test concrete subclass must implement all abstract methods.

        Coverage: EmbeddingProvider enforces implementation of all abstract methods.
        """

        # Attempt to create a concrete class without implementing all methods
        class IncompleteProvider(EmbeddingProvider):
            async def generate_embedding(self, text: str):
                pass

            async def generate_embeddings(self, texts: list[str]):
                pass

            def get_dimension(self):
                pass

            # Missing get_model_name

        with pytest.raises(TypeError):
            IncompleteProvider()

    def test_concrete_provider_must_implement_all_methods(self):
        """
        Test concrete provider implements all required methods.

        Coverage: EmbeddingProvider accepts fully implemented subclasses.
        """

        # Create a fully implemented concrete class
        class ConcreteProvider(EmbeddingProvider):
            async def generate_embedding(self, text: str):
                from milvus_ops.search_operations.providers.embedding import EmbeddingResult

                return EmbeddingResult(
                    embedding=[0.1] * 128,
                    dimension=128,
                    model_name="concrete_model",
                    processing_time_ms=10.0,
                )

            async def generate_embeddings(self, texts: list[str]):
                from milvus_ops.search_operations.providers.embedding import EmbeddingResult

                return EmbeddingResult(
                    embedding=[[0.1] * 128 for _ in texts],
                    dimension=128,
                    model_name="concrete_model",
                    processing_time_ms=20.0,
                    is_batch=True,
                )

            def get_dimension(self):
                return 128

            def get_model_name(self):
                return "concrete_model"

        # Should not raise error
        provider = ConcreteProvider()
        assert isinstance(provider, EmbeddingProvider)


# ============================================================================
# Test EmbeddingResult Edge Cases
# ============================================================================


@pytest.mark.unit
class TestEmbeddingResultEdgeCases:
    """
    Test EmbeddingResult edge cases and boundary conditions.

    Coverage: EmbeddingResult validation with None embeddings, dimension
    mismatches, and negative values.
    """

    def test_embedding_result_none_embedding(self):
        """
        Test EmbeddingResult with None embedding.

        Coverage: EmbeddingResult should raise error when embedding is None.
        """
        # EmbeddingResult now validates that embedding cannot be None
        with pytest.raises(ValueError, match="Embedding cannot be None"):
            EmbeddingResult(
                embedding=None,  # type: ignore
                dimension=128,
                model_name="test_model",
                processing_time_ms=10.0,
            )

    def test_embedding_result_dimension_mismatch(self):
        """
        Test EmbeddingResult with batch embeddings having inconsistent dimensions.

        Coverage: Batch embeddings should have consistent dimensions.
        """
        # Create batch embeddings with mismatched dimensions
        embeddings = [[0.1] * 128, [0.1] * 64, [0.1] * 256]  # Inconsistent dimensions

        # EmbeddingResult doesn't validate dimension consistency automatically,
        # but this is an edge case that should be tested
        result = EmbeddingResult(
            embedding=embeddings,
            dimension=128,  # Declared dimension doesn't match all embeddings
            model_name="test_model",
            processing_time_ms=30.0,
            is_batch=True,
        )

        # Verify the embedding is stored as-is (no automatic validation)
        assert len(result.embedding) == 3
        assert len(result.embedding[0]) == 128
        assert len(result.embedding[1]) == 64  # Mismatch
        assert len(result.embedding[2]) == 256  # Mismatch

    def test_embedding_result_negative_dimension(self):
        """
        Test EmbeddingResult with negative dimension value.

        Coverage: Negative dimension values should be caught by validation.
        """
        # EmbeddingResult now validates that dimension must be positive
        with pytest.raises(ValueError, match="Dimension must be positive"):
            EmbeddingResult(
                embedding=[0.1] * 128,
                dimension=-128,  # Negative dimension
                model_name="test_model",
                processing_time_ms=10.0,
            )

    def test_embedding_result_negative_processing_time(self):
        """
        Test EmbeddingResult with negative processing_time_ms.

        Coverage: Negative processing time values should be caught by validation.
        """
        # EmbeddingResult now validates that processing time cannot be negative
        with pytest.raises(ValueError, match="Processing time cannot be negative"):
            EmbeddingResult(
                embedding=[0.1] * 128,
                dimension=128,
                model_name="test_model",
                processing_time_ms=-10.0,  # Negative processing time
            )

    def test_embedding_result_empty_embedding(self):
        """
        Test EmbeddingResult with empty embedding list.

        Coverage: Empty embedding list edge case.
        """
        # EmbeddingResult validates that dimension must be positive, so dimension=0 is invalid
        with pytest.raises(ValueError, match="Dimension must be positive"):
            EmbeddingResult(
                embedding=[],
                dimension=0,
                model_name="test_model",
                processing_time_ms=0.0,
            )

    def test_embedding_result_empty_batch_embedding(self):
        """
        Test EmbeddingResult with empty batch embedding list.

        Coverage: Empty batch embedding list edge case.
        """
        result = EmbeddingResult(
            embedding=[],
            dimension=128,
            model_name="test_model",
            processing_time_ms=0.0,
            is_batch=True,
        )
        assert result.embedding == []
        assert result.is_batch is True


# ============================================================================
# Test EmbeddingProvider Edge Cases
# ============================================================================


@pytest.mark.unit
class TestEmbeddingProviderEdgeCases:
    """
    Test EmbeddingProvider edge cases and error handling.

    Coverage: EmbeddingProvider async error handling and None return values.
    """

    @pytest.mark.asyncio
    async def test_embedding_provider_none_get_dimension(self):
        """
        Test concrete provider returning None from get_dimension().

        Coverage: get_dimension() should not return None.
        """

        # Create a provider that returns None from get_dimension
        class BadProvider(EmbeddingProvider):
            async def generate_embedding(self, text: str):
                from milvus_ops.search_operations.providers.embedding import EmbeddingResult

                return EmbeddingResult(
                    embedding=[0.1] * 128,
                    dimension=128,
                    model_name="bad_model",
                    processing_time_ms=10.0,
                )

            async def generate_embeddings(self, texts: list[str]):
                from milvus_ops.search_operations.providers.embedding import EmbeddingResult

                return EmbeddingResult(
                    embedding=[[0.1] * 128 for _ in texts],
                    dimension=128,
                    model_name="bad_model",
                    processing_time_ms=20.0,
                    is_batch=True,
                )

            def get_dimension(self):
                return None  # type: ignore

            def get_model_name(self):
                return "bad_model"

        provider = BadProvider()
        # The method can return None, but it's semantically invalid
        assert provider.get_dimension() is None

    @pytest.mark.asyncio
    async def test_embedding_provider_async_error_handling(self):
        """
        Test async methods raising exceptions during execution.

        Coverage: Async error propagation in embedding provider methods.
        """

        # Create a provider that raises exceptions
        class ErrorProvider(EmbeddingProvider):
            async def generate_embedding(self, text: str):
                raise ValueError("Failed to generate embedding")

            async def generate_embeddings(self, texts: list[str]):
                raise RuntimeError("Failed to generate embeddings")

            def get_dimension(self):
                return 128

            def get_model_name(self):
                return "error_model"

        provider = ErrorProvider()

        # Test that errors propagate correctly
        with pytest.raises(ValueError, match="Failed to generate embedding"):
            await provider.generate_embedding("test")

        with pytest.raises(RuntimeError, match="Failed to generate embeddings"):
            await provider.generate_embeddings(["test1", "test2"])

    @pytest.mark.asyncio
    async def test_embedding_provider_partial_batch_failure(self):
        """
        Test batch operation with partial failures.

        Coverage: Some embeddings succeed, others fail in batch operations.
        """

        # This would typically be handled by the provider implementation
        # Testing that errors are properly raised
        class PartialFailureProvider(EmbeddingProvider):
            async def generate_embedding(self, text: str):
                from milvus_ops.search_operations.providers.embedding import EmbeddingResult

                if text == "fail":
                    raise ValueError("Failed to generate embedding")
                return EmbeddingResult(
                    embedding=[0.1] * 128,
                    dimension=128,
                    model_name="partial_model",
                    processing_time_ms=10.0,
                )

            async def generate_embeddings(self, texts: list[str]):
                # Simulate partial failure scenario
                results = []
                for text in texts:
                    if text == "fail":
                        raise ValueError("Failed to generate embedding")
                    results.append([0.1] * 128)

                from milvus_ops.search_operations.providers.embedding import EmbeddingResult

                return EmbeddingResult(
                    embedding=results,
                    dimension=128,
                    model_name="partial_model",
                    processing_time_ms=20.0,
                    is_batch=True,
                )

            def get_dimension(self):
                return 128

            def get_model_name(self):
                return "partial_model"

        provider = PartialFailureProvider()

        # Single embedding failure
        with pytest.raises(ValueError, match="Failed to generate embedding"):
            await provider.generate_embedding("fail")

        # Batch embedding failure
        with pytest.raises(ValueError, match="Failed to generate embedding"):
            await provider.generate_embeddings(["test1", "fail", "test2"])

        # Successful batch
        result = await provider.generate_embeddings(["test1", "test2"])
        assert result.is_batch is True
        assert len(result.embedding) == 2

    def test_embedding_provider_zero_dimension(self):
        """
        Test provider returning zero dimension.

        Coverage: Zero dimension edge case.
        """

        class ZeroDimensionProvider(EmbeddingProvider):
            async def generate_embedding(self, text: str):
                from milvus_ops.search_operations.providers.embedding import EmbeddingResult

                return EmbeddingResult(
                    embedding=[],
                    dimension=0,
                    model_name="zero_model",
                    processing_time_ms=0.0,
                )

            async def generate_embeddings(self, texts: list[str]):
                from milvus_ops.search_operations.providers.embedding import EmbeddingResult

                return EmbeddingResult(
                    embedding=[[] for _ in texts],
                    dimension=0,
                    model_name="zero_model",
                    processing_time_ms=0.0,
                    is_batch=True,
                )

            def get_dimension(self):
                return 0

            def get_model_name(self):
                return "zero_model"

        provider = ZeroDimensionProvider()
        assert provider.get_dimension() == 0
