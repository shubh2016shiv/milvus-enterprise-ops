"""
Comprehensive unit tests for search operations exceptions.

This module provides systematic testing of custom exception classes,
including exception hierarchy validation, message formatting, and exception chaining.
"""

import pytest

from milvus_ops.milvus_ops_exceptions import QueryError
from milvus_ops.search_operations.core.search_ops_exceptions import (
    ConnectionError,
    EmbeddingGenerationError,
    EmptyResultError,
    FusionError,
    HybridSearchError,
    InvalidSearchParametersError,
    ReRankingError,
    SearchError,
    SearchTimeoutError,
    SparseVectorGenerationError,
    TimeoutError,
)

# ============================================================================
# Test Exception Hierarchy
# ============================================================================


@pytest.mark.unit
class TestExceptionHierarchy:
    """
    Test exception class hierarchy and inheritance.

    Coverage: Exception inheritance relationships.
    """

    def test_search_error_inherits_from_query_error(self):
        """
        Test SearchError inherits from QueryError.

        Coverage: SearchError is a subclass of QueryError.
        """
        assert issubclass(SearchError, QueryError)

    def test_all_exceptions_inherit_from_search_error(self):
        """
        Test all search exceptions inherit from SearchError.

        Coverage: All custom exceptions are subclasses of SearchError.
        """
        exceptions = [
            InvalidSearchParametersError,
            EmbeddingGenerationError,
            SearchTimeoutError,
            ReRankingError,
            HybridSearchError,
            FusionError,
            EmptyResultError,
            SparseVectorGenerationError,
            ConnectionError,
            TimeoutError,
        ]

        for exc_class in exceptions:
            assert issubclass(
                exc_class, SearchError
            ), f"{exc_class.__name__} should inherit from SearchError"


# ============================================================================
# Test Exception Instantiation
# ============================================================================


@pytest.mark.unit
class TestExceptionInstantiation:
    """
    Test exception instantiation with various messages.

    Coverage: Exception instantiation and message handling.
    """

    def test_search_error_instantiation(self):
        """
        Test SearchError can be instantiated with message.

        Coverage: SearchError.__init__() accepts message parameter.
        """
        error = SearchError("Test error message")
        assert isinstance(error, SearchError)
        assert str(error) == "Test error message"

    def test_invalid_search_parameters_error_instantiation(self):
        """
        Test InvalidSearchParametersError can be instantiated.

        Coverage: InvalidSearchParametersError.__init__() works correctly.
        """
        error = InvalidSearchParametersError("Invalid parameters")
        assert isinstance(error, InvalidSearchParametersError)
        assert isinstance(error, SearchError)
        assert str(error) == "Invalid parameters"

    def test_embedding_generation_error_instantiation(self):
        """
        Test EmbeddingGenerationError can be instantiated.

        Coverage: EmbeddingGenerationError.__init__() works correctly.
        """
        error = EmbeddingGenerationError("Embedding failed")
        assert isinstance(error, EmbeddingGenerationError)
        assert isinstance(error, SearchError)
        assert str(error) == "Embedding failed"

    def test_search_timeout_error_instantiation(self):
        """
        Test SearchTimeoutError can be instantiated.

        Coverage: SearchTimeoutError.__init__() works correctly.
        """
        error = SearchTimeoutError("Search timed out")
        assert isinstance(error, SearchTimeoutError)
        assert isinstance(error, SearchError)
        assert str(error) == "Search timed out"

    def test_reranking_error_instantiation(self):
        """
        Test ReRankingError can be instantiated.

        Coverage: ReRankingError.__init__() works correctly.
        """
        error = ReRankingError("Reranking failed")
        assert isinstance(error, ReRankingError)
        assert isinstance(error, SearchError)
        assert str(error) == "Reranking failed"

    def test_hybrid_search_error_instantiation(self):
        """
        Test HybridSearchError can be instantiated.

        Coverage: HybridSearchError.__init__() works correctly.
        """
        error = HybridSearchError("Hybrid search failed")
        assert isinstance(error, HybridSearchError)
        assert isinstance(error, SearchError)
        assert str(error) == "Hybrid search failed"

    def test_fusion_error_instantiation(self):
        """
        Test FusionError can be instantiated.

        Coverage: FusionError.__init__() works correctly.
        """
        error = FusionError("Fusion failed")
        assert isinstance(error, FusionError)
        assert isinstance(error, SearchError)
        assert str(error) == "Fusion failed"

    def test_empty_result_error_instantiation(self):
        """
        Test EmptyResultError can be instantiated.

        Coverage: EmptyResultError.__init__() works correctly.
        """
        error = EmptyResultError("No results found")
        assert isinstance(error, EmptyResultError)
        assert isinstance(error, SearchError)
        assert str(error) == "No results found"

    def test_sparse_vector_generation_error_instantiation(self):
        """
        Test SparseVectorGenerationError can be instantiated.

        Coverage: SparseVectorGenerationError.__init__() works correctly.
        """
        error = SparseVectorGenerationError("BM25 generation failed")
        assert isinstance(error, SparseVectorGenerationError)
        assert isinstance(error, SearchError)
        assert str(error) == "BM25 generation failed"

    def test_connection_error_instantiation(self):
        """
        Test ConnectionError can be instantiated.

        Coverage: ConnectionError.__init__() works correctly.
        """
        error = ConnectionError("Connection failed")
        assert isinstance(error, ConnectionError)
        assert isinstance(error, SearchError)
        assert str(error) == "Connection failed"

    def test_timeout_error_instantiation(self):
        """
        Test TimeoutError can be instantiated.

        Coverage: TimeoutError.__init__() works correctly.
        """
        error = TimeoutError("Operation timed out")
        assert isinstance(error, TimeoutError)
        assert isinstance(error, SearchError)
        assert str(error) == "Operation timed out"


# ============================================================================
# Test Exception Message Formatting
# ============================================================================


@pytest.mark.unit
class TestExceptionMessageFormatting:
    """
    Test exception message formatting and edge cases.

    Coverage: Exception message handling with various input types.
    """

    def test_search_error_with_empty_message(self):
        """
        Test SearchError with empty message string.

        Coverage: SearchError handles empty string messages.
        """
        error = SearchError("")
        assert str(error) == ""

    def test_search_error_with_multiline_message(self):
        """
        Test SearchError with multiline message.

        Coverage: SearchError handles multiline messages.
        """
        message = "Error line 1\nError line 2\nError line 3"
        error = SearchError(message)
        assert str(error) == message

    def test_search_error_with_special_characters(self):
        """
        Test SearchError with special characters in message.

        Coverage: SearchError handles special characters in messages.
        """
        message = "Error: @#$%^&*()!~`"
        error = SearchError(message)
        assert str(error) == message

    def test_search_error_with_unicode_message(self):
        """
        Test SearchError with unicode characters in message.

        Coverage: SearchError handles unicode characters.
        """
        message = "错误消息 🚀"
        error = SearchError(message)
        assert str(error) == message


# ============================================================================
# Test Exception Chaining
# ============================================================================


@pytest.mark.unit
class TestExceptionChaining:
    """
    Test exception chaining functionality.

    Coverage: Exception chaining with cause and context.
    """

    def test_search_error_with_cause(self):
        """
        Test SearchError can be chained with cause.

        Coverage: SearchError supports exception chaining via from clause.
        """
        original_error = ValueError("Original error")
        try:
            raise original_error
        except ValueError:
            try:
                raise SearchError("Wrapped error") from original_error
            except SearchError as chained_error:
                assert isinstance(chained_error, SearchError)
                assert chained_error.__cause__ is original_error

    def test_invalid_search_parameters_error_with_cause(self):
        """
        Test InvalidSearchParametersError can be chained.

        Coverage: InvalidSearchParametersError supports exception chaining.
        """
        original_error = TypeError("Type mismatch")
        try:
            raise original_error
        except TypeError:
            try:
                raise InvalidSearchParametersError("Invalid params") from original_error
            except InvalidSearchParametersError as chained_error:
                assert isinstance(chained_error, InvalidSearchParametersError)
                assert chained_error.__cause__ is original_error

    def test_embedding_generation_error_with_cause(self):
        """
        Test EmbeddingGenerationError can be chained.

        Coverage: EmbeddingGenerationError supports exception chaining.
        """
        original_error = ConnectionError("Connection failed")
        try:
            raise original_error
        except ConnectionError:
            try:
                raise EmbeddingGenerationError("Embedding failed") from original_error
            except EmbeddingGenerationError as chained_error:
                assert isinstance(chained_error, EmbeddingGenerationError)
                assert chained_error.__cause__ is original_error


# ============================================================================
# Test Exception Raised Correctly
# ============================================================================


@pytest.mark.unit
class TestExceptionRaising:
    """
    Test that exceptions can be raised and caught correctly.

    Coverage: Exception raising and catching behavior.
    """

    def test_search_error_can_be_raised(self):
        """
        Test SearchError can be raised and caught.

        Coverage: SearchError can be raised as an exception.
        """
        with pytest.raises(SearchError) as exc_info:
            raise SearchError("Test error")
        assert str(exc_info.value) == "Test error"

    def test_invalid_search_parameters_error_can_be_raised(self):
        """
        Test InvalidSearchParametersError can be raised and caught.

        Coverage: InvalidSearchParametersError can be raised as an exception.
        """
        with pytest.raises(InvalidSearchParametersError) as exc_info:
            raise InvalidSearchParametersError("Invalid parameters")
        assert str(exc_info.value) == "Invalid parameters"
        assert isinstance(exc_info.value, SearchError)

    def test_embedding_generation_error_can_be_raised(self):
        """
        Test EmbeddingGenerationError can be raised and caught.

        Coverage: EmbeddingGenerationError can be raised as an exception.
        """
        with pytest.raises(EmbeddingGenerationError) as exc_info:
            raise EmbeddingGenerationError("Embedding failed")
        assert str(exc_info.value) == "Embedding failed"
        assert isinstance(exc_info.value, SearchError)

    def test_search_timeout_error_can_be_raised(self):
        """
        Test SearchTimeoutError can be raised and caught.

        Coverage: SearchTimeoutError can be raised as an exception.
        """
        with pytest.raises(SearchTimeoutError) as exc_info:
            raise SearchTimeoutError("Timeout")
        assert str(exc_info.value) == "Timeout"
        assert isinstance(exc_info.value, SearchError)


# ============================================================================
# Test Exception Edge Cases
# ============================================================================


@pytest.mark.unit
class TestExceptionEdgeCases:
    """
    Test exception edge cases and boundary conditions.

    Coverage: Exception handling with None messages, very long messages,
    unicode, and nested chaining.
    """

    def test_exception_with_none_message(self):
        """
        Test exception instantiation with None message.

        Coverage: Exceptions handle None message gracefully (should use default or handle error).
        """
        # Python exceptions typically accept None but may not display it well
        # Test that it doesn't crash
        exceptions = [
            SearchError,
            InvalidSearchParametersError,
            EmbeddingGenerationError,
            SearchTimeoutError,
            ReRankingError,
            HybridSearchError,
            FusionError,
            EmptyResultError,
            SparseVectorGenerationError,
            ConnectionError,
            TimeoutError,
        ]

        for exc_class in exceptions:
            # Exception.__init__ accepts None, but str(None) returns 'None'
            error = exc_class(None)  # type: ignore
            assert isinstance(error, exc_class)
            # str() of None is the string 'None'
            assert str(error) == "None" or str(error) == ""

    def test_exception_with_very_long_message(self):
        """
        Test exception with extremely long message.

        Coverage: Exceptions handle very long messages for memory/stability testing.
        """
        # Create a very long message (10MB+)
        very_long_message = "A" * (10 * 1024 * 1024)  # 10MB string

        error = SearchError(very_long_message)
        assert isinstance(error, SearchError)
        assert len(str(error)) == len(very_long_message)
        assert str(error) == very_long_message

    def test_exception_str_vs_repr(self):
        """
        Test exception __str__() vs __repr__() behavior with unicode.

        Coverage: Exception string representation with unicode characters.
        """
        unicode_message = "错误消息"
        error = SearchError(unicode_message)

        # __str__ should return the message
        assert str(error) == unicode_message

        # __repr__ should include class name and message
        repr_str = repr(error)
        assert "SearchError" in repr_str
        # repr may escape unicode differently, so just check it contains the class name

    def test_exception_unicode_in_message(self):
        """
        Test exception with unicode characters in message.

        Coverage: Unicode characters are preserved correctly in exception messages.
        """
        unicode_messages = [
            "错误消息",
            "日本語のエラー",
            "Русская ошибка",
            "العربية خطأ",
            "Combined: 错误 Русская",
        ]

        exceptions = [
            SearchError,
            InvalidSearchParametersError,
            EmbeddingGenerationError,
        ]

        for exc_class in exceptions:
            for message in unicode_messages:
                error = exc_class(message)
                assert isinstance(error, exc_class)
                assert str(error) == message

    def test_exception_nested_chaining(self):
        """
        Test multiple levels of exception chaining.

        Coverage: Exception chaining with __cause__ and __context__ relationships.
        """
        # Create a chain of exceptions
        original_error = ValueError("Original value error")
        try:
            raise original_error
        except ValueError:
            try:
                raise TypeError("Intermediate type error") from original_error
            except TypeError as intermediate_error:
                try:
                    raise SearchError("Final search error") from intermediate_error
                except SearchError as final_error:
                    # Verify the chain
                    assert isinstance(final_error, SearchError)
                    assert final_error.__cause__ is intermediate_error
                    assert intermediate_error.__cause__ is original_error

        # Test with context (implicit chaining)
        try:
            try:
                raise ValueError("Inner error")
            except ValueError as e:
                raise TypeError("Outer error") from e
        except TypeError as e:
            # Context should be set
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)

        # Test nested search exceptions
        base_error = ConnectionError("Connection failed")
        try:
            raise base_error
        except ConnectionError:
            try:
                raise EmbeddingGenerationError("Embedding failed") from base_error
            except EmbeddingGenerationError as search_error:
                try:
                    raise HybridSearchError("Hybrid search failed") from search_error
                except HybridSearchError as hybrid_error:
                    assert hybrid_error.__cause__ is search_error
                    assert search_error.__cause__ is base_error

    @pytest.mark.parametrize(
        "exc_class",
        [
            SearchError,
            InvalidSearchParametersError,
            EmbeddingGenerationError,
            SearchTimeoutError,
            ReRankingError,
            HybridSearchError,
            FusionError,
            EmptyResultError,
            SparseVectorGenerationError,
            ConnectionError,
            TimeoutError,
        ],
    )
    def test_all_exceptions_with_unicode(self, exc_class):
        """
        Test all exception classes with unicode characters.

        Coverage: All exception classes handle unicode correctly.
        """
        unicode_message = "错误消息 🚀"
        error = exc_class(unicode_message)
        assert isinstance(error, exc_class)
        assert str(error) == unicode_message

    @pytest.mark.parametrize(
        "exc_class",
        [
            SearchError,
            InvalidSearchParametersError,
            EmbeddingGenerationError,
            SearchTimeoutError,
            ReRankingError,
            HybridSearchError,
            FusionError,
            EmptyResultError,
            SparseVectorGenerationError,
            ConnectionError,
            TimeoutError,
        ],
    )
    def test_all_exceptions_with_none_message(self, exc_class):
        """
        Test all exception classes with None message.

        Coverage: All exception classes handle None message.
        """
        error = exc_class(None)  # type: ignore
        assert isinstance(error, exc_class)
        # str() of None is the string 'None'
        assert str(error) in ("None", "")

    @pytest.mark.parametrize(
        "exc_class",
        [
            SearchError,
            InvalidSearchParametersError,
            EmbeddingGenerationError,
            SearchTimeoutError,
        ],
    )
    def test_all_exceptions_with_very_long_message(self, exc_class):
        """
        Test all exception classes with very long message.

        Coverage: All exception classes handle very long messages.
        """
        long_message = "A" * (1024 * 1024)  # 1MB string
        error = exc_class(long_message)
        assert isinstance(error, exc_class)
        assert len(str(error)) == len(long_message)
