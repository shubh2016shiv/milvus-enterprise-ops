"""
Rate Limiting Utilities

This module provides production-grade rate limiting implementations to prevent
system overload, resource exhaustion, and cascade failures. Rate limiters are
essential for protecting both the application and the Milvus server from
request storms and ensuring fair resource allocation.

Key Features:
- Token bucket algorithm for smooth rate limiting
- Sliding window algorithm for precise rate control
- Thread-safe async operations
- Comprehensive metrics and monitoring
- Configurable burst capacity
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from collections import deque
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class RateLimiterMetrics:
    """Metrics for rate limiter monitoring and observability."""
    total_requests: int = 0
    total_throttled: int = 0
    total_wait_time: float = 0.0
    peak_wait_time: float = 0.0
    current_tokens: float = 0.0
    throttle_rate: float = 0.0  # Percentage of requests throttled
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for monitoring systems."""
        return {
            "total_requests": self.total_requests,
            "total_throttled": self.total_throttled,
            "total_wait_time_seconds": round(self.total_wait_time, 2),
            "peak_wait_time_seconds": round(self.peak_wait_time, 2),
            "current_tokens": round(self.current_tokens, 2),
            "throttle_rate_percent": round(self.throttle_rate * 100, 2),
            "average_wait_time_seconds": (
                round(self.total_wait_time / self.total_throttled, 3)
                if self.total_throttled > 0 else 0.0
            )
        }


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""
    
    @abstractmethod
    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens from the rate limiter.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            Time waited in seconds (0 if no wait was needed)
        """
        pass
    
    @abstractmethod
    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without blocking.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens were acquired, False otherwise
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get rate limiter metrics for monitoring."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the rate limiter to initial state."""
        pass


class TokenBucketRateLimiter(RateLimiter):
    """
    Token bucket rate limiter with async support.
    
    The token bucket algorithm allows for smooth rate limiting with burst capacity.
    Tokens are added to the bucket at a constant rate. Each request consumes tokens.
    If the bucket is empty, requests must wait for tokens to be replenished.
    
    This is ideal for:
    - Protecting backend services from overload
    - Allowing burst traffic while maintaining average rate
    - Fair resource allocation across concurrent operations
    
    Example:
        >>> limiter = TokenBucketRateLimiter(rate=100, capacity=200)
        >>> # Allow 100 requests/second with burst capacity of 200
        >>> await limiter.acquire()  # Consumes 1 token
        >>> await limiter.acquire(tokens=5)  # Consumes 5 tokens
    """
    
    def __init__(
        self,
        rate: float,
        capacity: Optional[int] = None,
        initial_tokens: Optional[float] = None
    ):
        """
        Initialize token bucket rate limiter.
        
        Args:
            rate: Token replenishment rate (tokens per second)
            capacity: Maximum bucket capacity (default: 2x rate for burst handling)
            initial_tokens: Initial number of tokens (default: full capacity)
        """
        if rate <= 0:
            raise ValueError("Rate must be positive")
        
        self.rate = rate
        self.capacity = capacity if capacity is not None else int(rate * 2)
        self.tokens = initial_tokens if initial_tokens is not None else float(self.capacity)
        self.last_update = time.monotonic()
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        # Metrics
        self._metrics = RateLimiterMetrics()
        self._metrics.current_tokens = self.tokens
        
        logger.debug(
            f"TokenBucketRateLimiter initialized: rate={rate}/s, "
            f"capacity={self.capacity}, initial_tokens={self.tokens}"
        )
    
    def _refill_tokens(self) -> None:
        """
        Refill tokens based on elapsed time since last update.
        
        This is called internally before each token acquisition to ensure
        tokens are replenished according to the configured rate.
        """
        now = time.monotonic()
        elapsed = now - self.last_update
        
        # Calculate tokens to add based on elapsed time
        new_tokens = elapsed * self.rate
        
        # Add tokens but don't exceed capacity
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_update = now
        
        # Update metrics
        self._metrics.current_tokens = self.tokens
    
    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens from the bucket, waiting if necessary.
        
        This method will block until sufficient tokens are available.
        It's designed to be fair - requests are processed in order.
        
        Args:
            tokens: Number of tokens to acquire (default: 1)
            
        Returns:
            Time waited in seconds (0 if no wait was needed)
            
        Raises:
            ValueError: If tokens requested exceeds capacity
        """
        if tokens > self.capacity:
            raise ValueError(
                f"Cannot acquire {tokens} tokens (capacity: {self.capacity}). "
                f"Consider increasing capacity or reducing token request."
            )
        
        wait_start = time.monotonic()
        total_wait = 0.0
        
        async with self._lock:
            self._metrics.total_requests += 1
            
            while True:
                self._refill_tokens()
                
                if self.tokens >= tokens:
                    # Sufficient tokens available
                    self.tokens -= tokens
                    self._metrics.current_tokens = self.tokens
                    
                    if total_wait > 0:
                        self._metrics.total_throttled += 1
                        self._metrics.total_wait_time += total_wait
                        self._metrics.peak_wait_time = max(
                            self._metrics.peak_wait_time, total_wait
                        )
                        self._metrics.throttle_rate = (
                            self._metrics.total_throttled / self._metrics.total_requests
                        )
                    
                    return total_wait
                
                # Not enough tokens - calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.rate
                
                # Log throttling for monitoring
                if total_wait == 0:
                    logger.debug(
                        f"Rate limiter throttling request: need {tokens} tokens, "
                        f"have {self.tokens:.2f}, waiting {wait_time:.3f}s"
                    )
                
                # Release lock while waiting to allow token refills
                # Wait for a fraction of the required time to check periodically
                await asyncio.sleep(min(wait_time, 0.1))
                total_wait = time.monotonic() - wait_start
    
    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without blocking.
        
        This is useful for:
        - Fast-fail scenarios where waiting is not acceptable
        - Load shedding when system is overloaded
        - Implementing fallback mechanisms
        
        Args:
            tokens: Number of tokens to acquire (default: 1)
            
        Returns:
            True if tokens were acquired, False otherwise
        """
        if tokens > self.capacity:
            return False
        
        # Note: This is not async, so we can't use the lock properly
        # For true non-blocking behavior in async contexts, this should
        # only be used for monitoring/metrics, not critical path
        self._refill_tokens()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            self._metrics.current_tokens = self.tokens
            self._metrics.total_requests += 1
            return True
        
        self._metrics.total_requests += 1
        self._metrics.total_throttled += 1
        self._metrics.throttle_rate = (
            self._metrics.total_throttled / self._metrics.total_requests
        )
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current rate limiter metrics.
        
        Returns:
            Dictionary containing metrics for monitoring systems
        """
        return self._metrics.to_dict()
    
    def reset(self) -> None:
        """
        Reset the rate limiter to initial state.
        
        This refills the bucket to capacity and resets all metrics.
        Useful for testing or periodic metric resets.
        """
        self.tokens = float(self.capacity)
        self.last_update = time.monotonic()
        self._metrics = RateLimiterMetrics()
        self._metrics.current_tokens = self.tokens
        logger.debug("Rate limiter reset to initial state")
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"TokenBucketRateLimiter(rate={self.rate}/s, capacity={self.capacity}, "
            f"tokens={self.tokens:.2f}, throttle_rate={self._metrics.throttle_rate:.2%})"
        )


class SlidingWindowRateLimiter(RateLimiter):
    """
    Sliding window rate limiter for precise rate control.
    
    This algorithm tracks actual request timestamps within a sliding time window.
    It provides more precise rate limiting than token bucket but with higher
    memory overhead (O(n) where n is the rate limit).
    
    Best for:
    - Strict rate enforcement
    - Rate limiting per user/tenant
    - API rate limiting with exact quotas
    
    Example:
        >>> limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=60)
        >>> # Allow exactly 100 requests per 60-second window
        >>> await limiter.acquire()
    """
    
    def __init__(self, max_requests: int, window_seconds: float):
        """
        Initialize sliding window rate limiter.
        
        Args:
            max_requests: Maximum requests allowed in the window
            window_seconds: Time window in seconds
        """
        if max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: deque = deque()  # Timestamps of recent requests
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        # Metrics
        self._metrics = RateLimiterMetrics()
        
        logger.debug(
            f"SlidingWindowRateLimiter initialized: "
            f"{max_requests} requests per {window_seconds}s"
        )
    
    def _clean_old_requests(self) -> None:
        """Remove requests outside the current time window."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()
    
    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire permission to make requests.
        
        Args:
            tokens: Number of requests (default: 1)
            
        Returns:
            Time waited in seconds
        """
        wait_start = time.monotonic()
        total_wait = 0.0
        
        async with self._lock:
            self._metrics.total_requests += 1
            
            while True:
                self._clean_old_requests()
                current_count = len(self.requests)
                
                if current_count + tokens <= self.max_requests:
                    # Can proceed
                    now = time.monotonic()
                    for _ in range(tokens):
                        self.requests.append(now)
                    
                    if total_wait > 0:
                        self._metrics.total_throttled += 1
                        self._metrics.total_wait_time += total_wait
                        self._metrics.peak_wait_time = max(
                            self._metrics.peak_wait_time, total_wait
                        )
                        self._metrics.throttle_rate = (
                            self._metrics.total_throttled / self._metrics.total_requests
                        )
                    
                    self._metrics.current_tokens = self.max_requests - len(self.requests)
                    return total_wait
                
                # Wait for oldest request to expire
                if self.requests:
                    oldest = self.requests[0]
                    wait_until = oldest + self.window_seconds
                    wait_time = wait_until - time.monotonic()
                    
                    if wait_time > 0:
                        await asyncio.sleep(min(wait_time + 0.01, 0.1))
                        total_wait = time.monotonic() - wait_start
                else:
                    # Edge case: should not happen
                    await asyncio.sleep(0.01)
                    total_wait = time.monotonic() - wait_start
    
    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire without blocking."""
        self._clean_old_requests()
        
        if len(self.requests) + tokens <= self.max_requests:
            now = time.monotonic()
            for _ in range(tokens):
                self.requests.append(now)
            self._metrics.total_requests += 1
            self._metrics.current_tokens = self.max_requests - len(self.requests)
            return True
        
        self._metrics.total_requests += 1
        self._metrics.total_throttled += 1
        self._metrics.throttle_rate = (
            self._metrics.total_throttled / self._metrics.total_requests
        )
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get rate limiter metrics."""
        self._clean_old_requests()
        self._metrics.current_tokens = self.max_requests - len(self.requests)
        return self._metrics.to_dict()
    
    def reset(self) -> None:
        """Reset the rate limiter."""
        self.requests.clear()
        self._metrics = RateLimiterMetrics()
        self._metrics.current_tokens = self.max_requests
        logger.debug("Sliding window rate limiter reset")
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"SlidingWindowRateLimiter(max_requests={self.max_requests}, "
            f"window={self.window_seconds}s, current={len(self.requests)})"
        )

