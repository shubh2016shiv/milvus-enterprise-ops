"""
Comprehensive unit tests for BM25SparseVectorGenerator.

This module provides systematic testing of BM25SparseVectorGenerator,
including initialization, fitting, sparse vector generation, normalization,
caching, and edge cases.
"""

import pytest

from milvus_ops.search_operations.core.search_ops_exceptions import SparseVectorGenerationError
from milvus_ops.search_operations.search.hybrid.core.bm25 import BM25SparseVectorGenerator
from milvus_ops.search_operations.search.hybrid.utils.config import BM25Config

# ============================================================================
# Test BM25SparseVectorGenerator Initialization
# ============================================================================


@pytest.mark.unit
class TestBM25SparseVectorGeneratorInitialization:
    """
    Test BM25SparseVectorGenerator initialization.

    Coverage: BM25SparseVectorGenerator constructor with various configurations.
    """

    def test_initialization_default(self):
        """
        Test BM25SparseVectorGenerator initialization with default config.

        Coverage: BM25SparseVectorGenerator.__init__() with default BM25Config.
        """
        generator = BM25SparseVectorGenerator()
        assert generator.config is not None
        assert generator.enable_caching is True
        assert generator.config.k1 == 1.5
        assert generator.config.b == 0.75

    def test_initialization_custom_config(self):
        """
        Test BM25SparseVectorGenerator initialization with custom config.

        Coverage: BM25SparseVectorGenerator.__init__() accepts custom BM25Config.
        """
        config = BM25Config(k1=2.0, b=0.8, enable_stopwords=False)
        generator = BM25SparseVectorGenerator(config=config, enable_caching=False)
        assert generator.config.k1 == 2.0
        assert generator.config.b == 0.8
        assert generator.config.enable_stopwords is False
        assert generator.enable_caching is False

    def test_initialization_with_custom_stopwords(self):
        """
        Test BM25SparseVectorGenerator initialization with custom stopwords.

        Coverage: BM25SparseVectorGenerator.__init__() uses custom stopwords.
        """
        custom_stopwords = {"custom1", "custom2"}
        config = BM25Config(custom_stopwords=custom_stopwords)
        generator = BM25SparseVectorGenerator(config=config)
        assert generator.stopwords == custom_stopwords


# ============================================================================
# Test BM25SparseVectorGenerator fit() Method
# ============================================================================


@pytest.mark.unit
class TestBM25SparseVectorGeneratorFit:
    """
    Test BM25SparseVectorGenerator.generate() method.

    Coverage: Fitting with document corpus, IDF calculation, statistics tracking.
    """

    @pytest.mark.asyncio
    async def test_fit_with_documents(self):
        """
        Test BM25 generator processes documents dynamically.

        Coverage: BM25SparseVectorGenerator processes documents and calculates
        statistics dynamically.
        """
        generator = BM25SparseVectorGenerator()
        documents = [
            "The quick brown fox jumps over the lazy dog",
            "A brown dog is lazy",
            "The fox is quick",
        ]

        # Generate vectors for each document to build statistics
        for doc in documents:
            await generator.generate(doc)
        assert generator._doc_count == 3
        assert generator._avg_doc_length > 0

    @pytest.mark.asyncio
    async def test_fit_with_empty_documents(self):
        """
        Test BM25 generator with no documents.

        Coverage: BM25SparseVectorGenerator handles case with no document statistics.
        """
        generator = BM25SparseVectorGenerator()
        # With no documents processed, statistics are minimal
        assert generator._doc_count == 0
        assert generator._doc_count == 0

    @pytest.mark.asyncio
    async def test_fit_updates_term_doc_freq(self):
        """
        Test BM25 generator updates term document frequency.

        Coverage: BM25SparseVectorGenerator tracks term document frequencies dynamically.
        """
        generator = BM25SparseVectorGenerator()
        documents = ["fox jumps", "fox runs", "dog jumps"]

        # Process documents to build statistics
        for doc in documents:
            await generator.generate(doc)
        assert "fox" in generator._term_doc_freq
        assert generator._term_doc_freq["fox"] == 2


# ============================================================================
# Test BM25SparseVectorGenerator generate
# ============================================================================


@pytest.mark.unit
class TestBM25SparseVectorGeneratorGenerate:
    """
    Test BM25SparseVectorGenerator.generate() method.

    Coverage: Sparse vector generation with various inputs and edge cases.
    """

    @pytest.mark.asyncio
    async def test_generate_normal(self):
        """
        Test generate() with normal text.

        Coverage: BM25SparseVectorGenerator.generate() generates sparse vector.
        """
        generator = BM25SparseVectorGenerator()
        documents = ["The quick brown fox", "A lazy dog", "The fox is quick"]

        # Build statistics by processing documents
        for doc in documents:
            await generator.generate(doc)

        result = await generator.generate("quick fox")
        assert isinstance(result, dict)
        assert "indices" in result
        assert "values" in result
        assert isinstance(result["indices"], list)
        assert isinstance(result["values"], list)
        assert len(result["indices"]) == len(result["values"])
        assert all(isinstance(idx, int) for idx in result["indices"])
        assert all(isinstance(val, int | float) for val in result["values"])

    @pytest.mark.asyncio
    async def test_generate_empty_query(self):
        """
        Test generate() with empty query.

        Coverage: BM25SparseVectorGenerator.generate() handles empty query.
        """
        generator = BM25SparseVectorGenerator()
        documents = ["The quick brown fox"]

        # Build statistics
        await generator.generate(documents[0])

        # Empty query should raise SparseVectorGenerationError
        with pytest.raises(SparseVectorGenerationError, match="Input text cannot be empty"):
            await generator.generate("")

    @pytest.mark.asyncio
    async def test_generate_whitespace_only(self):
        """
        Test generate() with whitespace-only query.

        Coverage: BM25SparseVectorGenerator.generate() handles whitespace-only query.
        """
        generator = BM25SparseVectorGenerator()
        documents = ["The quick brown fox"]

        # Build statistics
        await generator.generate(documents[0])

        # Whitespace-only query should raise SparseVectorGenerationError
        with pytest.raises(SparseVectorGenerationError, match="Input text cannot be empty"):
            await generator.generate("   \n\t  ")

    @pytest.mark.asyncio
    async def test_generate_very_long_text(self):
        """
        Test generate() with very long text.

        Coverage: BM25SparseVectorGenerator.generate() handles very long text.
        """
        generator = BM25SparseVectorGenerator()
        documents = ["The quick brown fox"] * 10

        # Build statistics by processing documents
        for doc in documents:
            await generator.generate(doc)

        very_long_query = "word " * 1000
        result = await generator.generate(very_long_query)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_generate_special_characters(self):
        """
        Test generate() with special characters.

        Coverage: BM25SparseVectorGenerator.generate() handles special characters.
        """
        generator = BM25SparseVectorGenerator()
        documents = ["The quick brown fox"]

        # Build statistics
        await generator.generate(documents[0])

        special_query = "test!@#$%^&*()query"
        result = await generator.generate(special_query)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_generate_unicode(self):
        """
        Test generate() with unicode characters.

        Coverage: BM25SparseVectorGenerator.generate() handles unicode text.
        """
        generator = BM25SparseVectorGenerator()
        documents = ["The quick brown fox"]

        # Build statistics
        await generator.generate(documents[0])

        unicode_query = "test 世界 query"
        result = await generator.generate(unicode_query)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_generate_without_fit(self):
        """
        Test generate() without building document statistics first.

        Coverage: BM25SparseVectorGenerator.generate() handles case where
        no documents processed yet.
        """
        generator = BM25SparseVectorGenerator()
        # Should still work but with limited statistics
        result = await generator.generate("test query")
        assert isinstance(result, dict)


# ============================================================================
# Test BM25SparseVectorGenerator Tokenization and Normalization
# ============================================================================


@pytest.mark.unit
class TestBM25SparseVectorGeneratorTokenization:
    """
    Test BM25SparseVectorGenerator tokenization and normalization.

    Coverage: Stopword removal, normalization, term frequency, IDF calculation.
    """

    def test_tokenize_removes_stopwords(self):
        """
        Test _tokenize() removes stopwords.

        Coverage: BM25SparseVectorGenerator._tokenize() filters stopwords.
        """
        generator = BM25SparseVectorGenerator()
        text = "the quick brown fox and the lazy dog"
        tokens = generator._tokenize(text)
        # "the" and "and" should be filtered if stopwords enabled
        if generator.config.enable_stopwords:
            assert "the" not in tokens
            assert "and" not in tokens

    def test_tokenize_without_stopwords(self):
        """
        Test _tokenize() without stopword removal.

        Coverage: BM25SparseVectorGenerator._tokenize() preserves stopwords when disabled.
        """
        config = BM25Config(enable_stopwords=False)
        generator = BM25SparseVectorGenerator(config=config)
        text = "the quick brown"
        tokens = generator._tokenize(text)
        assert "the" in tokens

    def test_tokenize_filters_by_length(self):
        """
        Test _tokenize() filters by term length.

        Coverage: BM25SparseVectorGenerator._tokenize() filters tokens by
        min_term_length and max_term_length.
        """
        config = BM25Config(min_term_length=3, max_term_length=5)
        generator = BM25SparseVectorGenerator(config=config)
        text = "a ab abc abcd abcde abcdef"
        tokens = generator._tokenize(text)
        # "a" and "ab" should be filtered (too short)
        assert "a" not in tokens
        assert "ab" not in tokens
        # "abcdef" should be filtered (too long)
        assert "abcdef" not in tokens

    def test_calculate_idf_smoothing(self):
        """
        Test _calculate_idf() with smoothing.

        Coverage: BM25SparseVectorGenerator._calculate_idf() calculates smoothed IDF.
        """
        generator = BM25SparseVectorGenerator()
        generator._doc_count = 10
        generator._term_doc_freq["test"] = 2

        idf = generator._calculate_idf("test", 10)
        assert isinstance(idf, float)
        assert idf >= 0

    def test_calculate_bm25_score(self):
        """
        Test _calculate_bm25_score() calculation.

        Coverage: BM25SparseVectorGenerator._calculate_bm25_score() calculates BM25 score.
        """
        generator = BM25SparseVectorGenerator()
        generator._avg_doc_length = 10.0

        score = generator._calculate_bm25_score(term_freq=2, doc_length=10, idf=1.5)
        assert isinstance(score, float)
        assert score >= 0


# ============================================================================
# Test BM25SparseVectorGenerator Caching
# ============================================================================


@pytest.mark.unit
class TestBM25SparseVectorGeneratorCaching:
    """
    Test BM25SparseVectorGenerator caching behavior.

    Coverage: Caching functionality and cache hits/misses.
    """

    @pytest.mark.asyncio
    async def test_generate_caching(self):
        """
        Test generate() uses cache when enabled.

        Coverage: BM25SparseVectorGenerator.generate() caches results.
        """
        generator = BM25SparseVectorGenerator(enable_caching=True)
        documents = ["The quick brown fox"]

        # Build statistics
        await generator.generate(documents[0])

        query = "quick fox"
        result1 = await generator.generate(query)
        result2 = await generator.generate(query)

        # Results should be identical (cached)
        assert result1 == result2
        assert generator._cache_hits >= 0

    @pytest.mark.asyncio
    async def test_generate_without_caching(self):
        """
        Test generate() without caching.

        Coverage: BM25SparseVectorGenerator.generate() skips cache when disabled.
        """
        generator = BM25SparseVectorGenerator(enable_caching=False)
        documents = ["The quick brown fox"]

        # Build statistics
        await generator.generate(documents[0])

        query = "quick fox"
        result1 = await generator.generate(query)
        result2 = await generator.generate(query)

        # Results may differ due to dynamic statistics building
        # but both should be valid sparse vectors
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)
        assert "indices" in result1 and "values" in result1
        assert "indices" in result2 and "values" in result2
        assert len(generator._cache) == 0


# ============================================================================
# Test BM25Config Validation
# ============================================================================


@pytest.mark.unit
class TestBM25ConfigValidation:
    """
    Test BM25Config parameter validation.

    Coverage: BM25Config validation and edge cases.
    """

    def test_bm25_config_invalid_k1(self):
        """
        Test BM25Config raises error for invalid k1.

        Coverage: BM25Config.__post_init__() validates k1 > 0.
        """
        with pytest.raises(ValueError, match="k1 must be positive"):
            BM25Config(k1=0)

    def test_bm25_config_invalid_b(self):
        """
        Test BM25Config raises error for invalid b.

        Coverage: BM25Config.__post_init__() validates 0 <= b <= 1.
        """
        with pytest.raises(ValueError, match="b must be between 0 and 1"):
            BM25Config(b=1.5)

        with pytest.raises(ValueError, match="b must be between 0 and 1"):
            BM25Config(b=-0.1)

    def test_bm25_config_invalid_delta(self):
        """
        Test BM25Config raises error for negative delta.

        Coverage: BM25Config.__post_init__() validates delta >= 0.
        """
        with pytest.raises(ValueError, match="delta must be non-negative"):
            BM25Config(delta=-1.0)

    def test_bm25_config_invalid_term_length(self):
        """
        Test BM25Config raises error for invalid term length parameters.

        Coverage: BM25Config.__post_init__() validates min_term_length and max_term_length.
        """
        with pytest.raises(ValueError, match="min_term_length must be at least 1"):
            BM25Config(min_term_length=0)

        with pytest.raises(ValueError, match="max_term_length.*must be >=.*min_term_length"):
            BM25Config(min_term_length=5, max_term_length=3)

    def test_bm25_config_invalid_max_dimensions(self):
        """
        Test BM25Config raises error for invalid max_dimensions.

        Coverage: BM25Config.__post_init__() validates max_dimensions >= 1.
        """
        with pytest.raises(ValueError, match="max_dimensions must be at least 1"):
            BM25Config(max_dimensions=0)
