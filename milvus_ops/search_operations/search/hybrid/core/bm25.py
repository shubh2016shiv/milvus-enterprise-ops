"""
BM25 Sparse Vector Generator

This module provides a production-grade implementation of BM25 (Best Matching 25)
algorithm for generating sparse vector representations of text, suitable for
hybrid search operations combining dense and sparse retrieval.
"""

import math
import re
import asyncio
import logging
from typing import Dict, Any, List, Set
from collections import Counter, defaultdict

from ....core.search_ops_exceptions import SparseVectorGenerationError
from ..utils.config import BM25Config

logger = logging.getLogger(__name__)


class BM25SparseVectorGenerator:
    """
    Production-grade BM25 sparse vector generator.
    
    This class implements the BM25 algorithm for generating sparse vector representations
    with proper normalization, stopword removal, document statistics tracking, and caching.
    
    BM25 is a probabilistic retrieval function that ranks documents based on query terms,
    considering term frequency, document length, and inverse document frequency.
    """
    
    # Common English stopwords
    DEFAULT_STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
        'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how'
    }
    
    def __init__(
        self,
        config: BM25Config = None,
        enable_caching: bool = True
    ):
        """
        Initialize BM25 sparse vector generator.
        
        Args:
            config: BM25 configuration (uses defaults if not provided)
            enable_caching: Enable caching of generated vectors
        """
        self.config = config or BM25Config()
        self.enable_caching = enable_caching
        
        # Initialize stopwords
        self.stopwords = self.config.custom_stopwords or self.DEFAULT_STOPWORDS
        
        # Cache for generated vectors
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Document statistics for IDF calculation
        self._doc_count = 0
        self._term_doc_freq: Dict[str, int] = defaultdict(int)
        self._avg_doc_length = 0.0
        self._lock = asyncio.Lock()
        
        logger.info(
            f"BM25SparseVectorGenerator initialized - "
            f"k1: {self.config.k1}, b: {self.config.b}, "
            f"caching: {enable_caching}"
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms.
        
        This method converts text to lowercase, extracts alphanumeric tokens,
        filters by length, and optionally removes stopwords.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and split into alphanumeric tokens
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        
        # Filter by length
        tokens = [
            t for t in tokens
            if self.config.min_term_length <= len(t) <= self.config.max_term_length
        ]
        
        # Remove stopwords if enabled
        if self.config.enable_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        
        return tokens
    
    def _calculate_idf(self, term: str, total_docs: int) -> float:
        """
        Calculate IDF (Inverse Document Frequency) for a term.
        
        IDF measures how important a term is across all documents.
        Rare terms get higher scores, common terms get lower scores.
        
        Args:
            term: The term to calculate IDF for
            total_docs: Total number of documents processed
            
        Returns:
            IDF score for the term
        """
        doc_freq = self._term_doc_freq.get(term, 0)
        
        if self.config.idf_smoothing:
            # Smoothed IDF to avoid division by zero and negative values
            idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        else:
            # Standard IDF
            if doc_freq == 0:
                idf = math.log(total_docs + 1)
            else:
                idf = math.log(total_docs / doc_freq)
        
        return max(idf, 0.0)  # Ensure non-negative
    
    def _calculate_bm25_score(
        self,
        term_freq: int,
        doc_length: int,
        idf: float
    ) -> float:
        """
        Calculate BM25 score for a term.
        
        The BM25 formula combines term frequency, document length normalization,
        and IDF to produce a relevance score.
        
        Args:
            term_freq: Term frequency in document
            doc_length: Length of the document (in tokens)
            idf: IDF score of the term
            
        Returns:
            BM25 score for the term
        """
        # Normalize by document length
        if self._avg_doc_length > 0:
            normalized_length = doc_length / self._avg_doc_length
        else:
            normalized_length = 1.0
        
        # BM25 formula with saturation
        numerator = term_freq * (self.config.k1 + 1)
        denominator = (
            term_freq +
            self.config.k1 * (1 - self.config.b + self.config.b * normalized_length)
        )
        
        score = idf * (numerator / (denominator + self.config.delta))
        
        return score
    
    async def generate(
        self,
        text: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Generate BM25 sparse vector for text.
        
        This method tokenizes the input text, calculates BM25 scores for each term,
        and returns a sparse vector representation as indices and values.
        
        Args:
            text: Input text to generate sparse vector for
            use_cache: Whether to use cached results
            
        Returns:
            Sparse vector as dict with 'indices' and 'values' lists
            
        Raises:
            SparseVectorGenerationError: If generation fails
        """
        if not text or not text.strip():
            raise SparseVectorGenerationError("Input text cannot be empty")
        
        # Check cache
        cache_key = hash(text)
        if use_cache and self.enable_caching and cache_key in self._cache:
            self._cache_hits += 1
            logger.debug("BM25 cache hit")
            return self._cache[cache_key].copy()
        
        self._cache_misses += 1
        
        try:
            # Tokenize text
            tokens = self._tokenize(text)
            
            if not tokens:
                logger.warning("No valid tokens after tokenization")
                return {"indices": [], "values": []}
            
            # Count term frequencies
            term_freq = Counter(tokens)
            doc_length = len(tokens)
            
            # Update document statistics
            async with self._lock:
                self._doc_count += 1
                for term in term_freq.keys():
                    self._term_doc_freq[term] += 1
                
                # Update average document length (running average)
                if self._doc_count == 1:
                    self._avg_doc_length = doc_length
                else:
                    self._avg_doc_length = (
                        (self._avg_doc_length * (self._doc_count - 1) + doc_length) /
                        self._doc_count
                    )
            
            # Calculate BM25 scores
            indices = []
            values = []
            
            for term, freq in term_freq.items():
                # Calculate IDF
                idf = self._calculate_idf(term, self._doc_count)
                
                # Calculate BM25 score
                score = self._calculate_bm25_score(freq, doc_length, idf)
                
                # Use term hash as index (modulo max_dimensions to stay within bounds)
                index = hash(term) % self.config.max_dimensions
                
                indices.append(index)
                values.append(score)
            
            # Sort by indices for consistency
            sorted_pairs = sorted(zip(indices, values))
            indices, values = zip(*sorted_pairs) if sorted_pairs else ([], [])
            
            result = {
                "indices": list(indices),
                "values": list(values)
            }
            
            # Cache result
            if self.enable_caching:
                self._cache[cache_key] = result.copy()
                # Limit cache size to prevent memory issues
                if len(self._cache) > 10000:
                    # Remove oldest 20% of entries
                    remove_count = len(self._cache) // 5
                    for key in list(self._cache.keys())[:remove_count]:
                        del self._cache[key]
            
            logger.debug(
                f"Generated BM25 sparse vector - "
                f"tokens: {len(tokens)}, dimensions: {len(indices)}"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"BM25 sparse vector generation failed: {str(e)}"
            logger.error(error_msg)
            raise SparseVectorGenerationError(error_msg) from e
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the BM25 generator.
        
        Returns:
            Dictionary containing cache statistics and document statistics
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (
            self._cache_hits / total_requests if total_requests > 0 else 0.0
        )
        
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": round(hit_rate, 3),
            "cache_size": len(self._cache),
            "doc_count": self._doc_count,
            "unique_terms": len(self._term_doc_freq),
            "avg_doc_length": round(self._avg_doc_length, 2)
        }
    
    async def clear_cache(self) -> None:
        """
        Clear the cache and reset cache statistics.
        
        This method clears all cached sparse vectors and resets hit/miss counters,
        but preserves document statistics for IDF calculations.
        """
        async with self._lock:
            self._cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
        logger.info("BM25 cache cleared")
    
    async def reset_statistics(self) -> None:
        """
        Reset all document statistics.
        
        This method clears document count, term frequencies, and average document
        length. Use this when starting a new indexing session.
        """
        async with self._lock:
            self._doc_count = 0
            self._term_doc_freq.clear()
            self._avg_doc_length = 0.0
        logger.info("BM25 document statistics reset")

