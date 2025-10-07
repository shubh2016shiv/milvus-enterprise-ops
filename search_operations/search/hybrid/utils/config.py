"""
Configuration Module

This module provides configuration classes for hybrid search operations,
including BM25 parameters, retry configuration, and search mode enumerations.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Set, Tuple, Type


class HybridSearchMode(Enum):
    """
    Hybrid search combination modes.
    
    Defines the different ways dense and sparse/keyword searches
    can be combined in hybrid search operations.
    """
    VECTOR_SPARSE = "vector_sparse"
    VECTOR_KEYWORD = "vector_keyword"
    VECTOR_ONLY = "vector_only"
    ALL_METHODS = "all_methods"


@dataclass
class BM25Config:
    """
    Configuration for BM25 sparse vector generation.
    
    BM25 is a probabilistic ranking function used to estimate the relevance
    of documents to a given search query. This configuration controls its behavior.
    
    Attributes:
        k1: Term frequency saturation parameter (typical range: 1.2-2.0)
        b: Length normalization parameter (0 = no normalization, 1 = full normalization)
        delta: Lower bound for term frequency normalization
        min_term_length: Minimum length of tokens to consider
        max_term_length: Maximum length of tokens to consider
        max_dimensions: Maximum number of dimensions in sparse vector
        enable_stemming: Whether to apply stemming (not yet implemented)
        enable_stopwords: Whether to filter stopwords
        custom_stopwords: Optional custom stopword set
        idf_smoothing: Whether to use smoothed IDF calculation
    """
    k1: float = 1.5
    b: float = 0.75
    delta: float = 1.0
    min_term_length: int = 2
    max_term_length: int = 50
    max_dimensions: int = 10000
    enable_stemming: bool = False
    enable_stopwords: bool = True
    custom_stopwords: Optional[Set[str]] = None
    idf_smoothing: bool = True
    
    def __post_init__(self):
        """Validate BM25 configuration parameters."""
        if self.k1 <= 0:
            raise ValueError(f"k1 must be positive, got {self.k1}")
        if not 0 <= self.b <= 1:
            raise ValueError(f"b must be between 0 and 1, got {self.b}")
        if self.delta < 0:
            raise ValueError(f"delta must be non-negative, got {self.delta}")
        if self.min_term_length < 1:
            raise ValueError(f"min_term_length must be at least 1, got {self.min_term_length}")
        if self.max_term_length < self.min_term_length:
            raise ValueError(
                f"max_term_length ({self.max_term_length}) must be >= "
                f"min_term_length ({self.min_term_length})"
            )
        if self.max_dimensions < 1:
            raise ValueError(f"max_dimensions must be at least 1, got {self.max_dimensions}")


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior with exponential backoff.
    
    This configuration controls how failed operations are retried,
    including timing, jitter, and which exceptions should trigger retries.
    
    Attributes:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay in seconds between retries
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delays
        retriable_exceptions: Tuple of exception types that should trigger retries
    """
    max_retries: int = 3
    initial_delay: float = 0.5
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    retriable_exceptions: Tuple[Type[Exception], ...] = field(default_factory=tuple)
    
    def __post_init__(self):
        """Validate retry configuration parameters."""
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {self.max_retries}")
        if self.initial_delay <= 0:
            raise ValueError(f"initial_delay must be positive, got {self.initial_delay}")
        if self.max_delay < self.initial_delay:
            raise ValueError(
                f"max_delay ({self.max_delay}) must be >= "
                f"initial_delay ({self.initial_delay})"
            )
        if self.exponential_base <= 1:
            raise ValueError(f"exponential_base must be > 1, got {self.exponential_base}")
        
        # Set default retriable exceptions if none provided
        if not self.retriable_exceptions:
            # Import here to avoid circular dependencies
            from ....core.exceptions import SearchTimeoutError
            from exceptions import ConnectionError as MilvusConnectionError
            
            self.retriable_exceptions = (
                SearchTimeoutError,
                MilvusConnectionError,
                TimeoutError,
            )

