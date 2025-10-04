"""
Retry Budget Pattern

Implementation of Netflix's retry budget pattern to prevent retry storms.
A retry budget limits the total number of retries allowed based on the
recent success rate, preventing cascading failures when the system is
already struggling.

Key Concepts:
- Only allow retries if the success rate is above a threshold
- Prevent retry storms during outages
- Maintain system stability under load
"""

import time
import logging
from typing import Dict, Any, Optional
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetryBudgetMetrics:
    """Metrics for retry budget monitoring."""
    total_attempts: int = 0
    total_successes: int = 0
    total_failures: int = 0
    retries_allowed: int = 0
    retries_denied: int = 0
    current_success_rate: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_attempts": self.total_attempts,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "retries_allowed": self.retries_allowed,
            "retries_denied": self.retries_denied,
            "current_success_rate": round(self.current_success_rate, 4),
            "retry_budget_exhausted": self.current_success_rate < 0.8
        }


class RetryBudget:
    """
    Implements Netflix's retry budget pattern.
    
    This pattern prevents retry storms by only allowing retries when the
    system's success rate is above a threshold. If too many requests are
    failing, retries are denied to prevent overwhelming the system.
    
    Example:
        >>> budget = RetryBudget(window_seconds=10, min_success_rate=0.8)
        >>> 
        >>> # Record attempt and check if retry is allowed
        >>> if budget.record_attempt(success=False):
        >>>     # Retry allowed - success rate is healthy
        >>>     await retry_operation()
        >>> else:
        >>>     # Retry denied - success rate too low
        >>>     raise MaxRetriesExceededError()
    """
    
    def __init__(
        self,
        window_seconds: int = 10,
        min_success_rate: float = 0.8,
        min_attempts: int = 10
    ):
        """
        Initialize retry budget.
        
        Args:
            window_seconds: Time window for tracking success rate
            min_success_rate: Minimum success rate to allow retries (0.0-1.0)
            min_attempts: Minimum attempts before enforcing budget
        """
        if not 0.0 <= min_success_rate <= 1.0:
            raise ValueError("min_success_rate must be between 0.0 and 1.0")
        
        self.window_seconds = window_seconds
        self.min_success_rate = min_success_rate
        self.min_attempts = min_attempts
        
        # Track attempts: (timestamp, success)
        self.attempts: deque = deque()
        self.successes: deque = deque()
        
        # Metrics
        self._metrics = RetryBudgetMetrics()
        
        logger.debug(
            f"RetryBudget initialized: window={window_seconds}s, "
            f"min_success_rate={min_success_rate}, min_attempts={min_attempts}"
        )
    
    def _clean_old_entries(self) -> None:
        """Remove entries outside the time window."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        
        while self.attempts and self.attempts[0] < cutoff:
            self.attempts.popleft()
        
        while self.successes and self.successes[0] < cutoff:
            self.successes.popleft()
    
    def can_retry(self) -> bool:
        """
        Check if a retry is allowed based on current success rate.
        
        Returns:
            True if retry is allowed, False if budget is exhausted
        """
        self._clean_old_entries()
        
        # Always allow retries if we don't have enough data
        if len(self.attempts) < self.min_attempts:
            return True
        
        # Calculate success rate
        success_rate = len(self.successes) / len(self.attempts)
        self._metrics.current_success_rate = success_rate
        
        # Allow retry if success rate is above threshold
        allowed = success_rate >= self.min_success_rate
        
        if allowed:
            self._metrics.retries_allowed += 1
        else:
            self._metrics.retries_denied += 1
            logger.warning(
                f"Retry budget exhausted: success_rate={success_rate:.2%} "
                f"< {self.min_success_rate:.2%} (attempts={len(self.attempts)})"
            )
        
        return allowed
    
    def record_attempt(self, success: bool) -> bool:
        """
        Record an attempt result and check if retry is allowed.
        
        Args:
            success: Whether the attempt succeeded
            
        Returns:
            True if retry is allowed for failed attempts, always True for successes
        """
        now = time.monotonic()
        
        # Record the attempt
        self.attempts.append(now)
        self._metrics.total_attempts += 1
        
        if success:
            self.successes.append(now)
            self._metrics.total_successes += 1
            return True  # Success - no retry needed
        else:
            self._metrics.total_failures += 1
            # Check if retry is allowed for this failure
            return self.can_retry()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get retry budget metrics."""
        self._clean_old_entries()
        
        if len(self.attempts) > 0:
            self._metrics.current_success_rate = (
                len(self.successes) / len(self.attempts)
            )
        
        return self._metrics.to_dict()
    
    def reset(self) -> None:
        """Reset the retry budget."""
        self.attempts.clear()
        self.successes.clear()
        self._metrics = RetryBudgetMetrics()
        self._metrics.current_success_rate = 1.0
        logger.debug("Retry budget reset")
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        self._clean_old_entries()
        success_rate = (
            len(self.successes) / len(self.attempts)
            if len(self.attempts) > 0 else 1.0
        )
        return (
            f"RetryBudget(window={self.window_seconds}s, "
            f"success_rate={success_rate:.2%}, "
            f"attempts={len(self.attempts)}/{self.min_attempts})"
        )

