import asyncio
import time
from functools import wraps
from typing import Dict, Callable, Any
from collections import defaultdict, deque
import logging

from .exceptions import RateLimitExceeded

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Implements a token bucket algorithm to limit the rate of API calls
    with support for burst requests and gradual refill.
    """

    def __init__(self, calls: int, period: int, burst_factor: float = 1.5):
        """
        Initialize rate limiter.

        Args:
            calls: Number of calls allowed per period
            period: Time period in seconds
            burst_factor: Multiplier for burst capacity (default 1.5)
        """
        self.calls = calls
        self.period = period
        self.max_tokens = int(calls * burst_factor)
        self.tokens = self.max_tokens
        self.last_refill = time.time()
        self.lock = asyncio.Lock()

        # Refill rate: tokens per second
        self.refill_rate = calls / period

        logger.debug(
            f"Rate limiter initialized: {calls} calls per {period}s, max tokens: {self.max_tokens}"
        )

    async def acquire(self, tokens_needed: int = 1) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens_needed: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False if rate limit exceeded

        Raises:
            RateLimitExceeded: If no tokens available and wait would be too long
        """
        async with self.lock:
            await self._refill_tokens()

            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                logger.debug(
                    f"Acquired {tokens_needed} tokens, {self.tokens} remaining"
                )
                return True
            else:
                # Calculate wait time for next available token
                wait_time = (tokens_needed - self.tokens) / self.refill_rate

                if wait_time > 60:  # Don't wait more than 1 minute
                    raise RateLimitExceeded(
                        f"Rate limit exceeded. Would need to wait {wait_time:.1f}s for {tokens_needed} tokens"
                    )

                logger.warning(
                    f"Rate limit reached, waiting {wait_time:.1f}s for {tokens_needed} tokens"
                )
                await asyncio.sleep(wait_time)

                # Try again after waiting
                await self._refill_tokens()
                if self.tokens >= tokens_needed:
                    self.tokens -= tokens_needed
                    return True
                else:
                    raise RateLimitExceeded("Rate limit exceeded after waiting")

    async def _refill_tokens(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill

        if elapsed > 0:
            new_tokens = elapsed * self.refill_rate
            self.tokens = min(self.max_tokens, self.tokens + new_tokens)
            self.last_refill = now

            if new_tokens > 0:
                logger.debug(
                    f"Refilled {new_tokens:.2f} tokens, total: {self.tokens:.2f}"
                )

    @property
    def tokens_available(self) -> int:
        """Get current number of available tokens."""
        return int(self.tokens)

    async def wait_for_tokens(self, tokens_needed: int = 1) -> float:
        """
        Calculate wait time for specified number of tokens.

        Args:
            tokens_needed: Number of tokens needed

        Returns:
            Wait time in seconds
        """
        async with self.lock:
            await self._refill_tokens()

            if self.tokens >= tokens_needed:
                return 0.0
            else:
                return (tokens_needed - self.tokens) / self.refill_rate


class GlobalRateLimiter:
    """
    Global rate limiter that manages multiple rate limiters by endpoint/client.
    """

    def __init__(self):
        self._limiters: Dict[str, RateLimiter] = {}
        self._call_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

    def get_limiter(
        self, key: str, calls: int = 60, period: int = 60, burst_factor: float = 1.5
    ) -> RateLimiter:
        """
        Get or create a rate limiter for a specific key.

        Args:
            key: Unique identifier for the rate limiter
            calls: Number of calls allowed per period
            period: Time period in seconds
            burst_factor: Multiplier for burst capacity

        Returns:
            RateLimiter instance
        """
        if key not in self._limiters:
            self._limiters[key] = RateLimiter(calls, period, burst_factor)
            logger.debug(f"Created new rate limiter for key: {key}")

        return self._limiters[key]

    async def acquire(
        self, key: str, tokens: int = 1, calls: int = 60, period: int = 60
    ) -> bool:
        """
        Acquire tokens from the rate limiter for a specific key.

        Args:
            key: Unique identifier for the rate limiter
            tokens: Number of tokens to acquire
            calls: Number of calls allowed per period
            period: Time period in seconds

        Returns:
            True if tokens were acquired
        """
        limiter = self.get_limiter(key, calls, period)
        success = await limiter.acquire(tokens)

        if success:
            # Record the call for statistics
            self._call_history[key].append(time.time())

        return success

    def get_stats(self, key: str) -> Dict[str, Any]:
        """
        Get statistics for a specific rate limiter.

        Args:
            key: Rate limiter key

        Returns:
            Statistics dictionary
        """
        limiter = self._limiters.get(key)
        history = self._call_history.get(key, deque())

        if not limiter:
            return {"error": "Rate limiter not found"}

        now = time.time()
        recent_calls = [t for t in history if now - t < 3600]  # Last hour

        return {
            "tokens_available": limiter.tokens_available,
            "max_tokens": limiter.max_tokens,
            "refill_rate": limiter.refill_rate,
            "calls_last_hour": len(recent_calls),
            "total_calls": len(history),
        }


# Global instance
_global_limiter = GlobalRateLimiter()


def rate_limit(
    calls: int = 60,
    period: int = 60,
    key_func: Callable = None,
    burst_factor: float = 1.5,
):
    """
    Decorator for rate limiting function calls.

    Args:
        calls: Number of calls allowed per period
        period: Time period in seconds
        key_func: Function to generate rate limiter key (default: use function name)
        burst_factor: Multiplier for burst capacity

    Example:
        @rate_limit(calls=30, period=60)
        async def api_call():
            pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate key for rate limiter
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                # Use function name and class name if available
                if args and hasattr(args[0], "__class__"):
                    key = f"{args[0].__class__.__name__}.{func.__name__}"
                else:
                    key = func.__name__

            # Acquire token before executing function
            await _global_limiter.acquire(key, 1, calls, period)

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.debug(f"Error in rate-limited function {key}: {e}")
                raise

        # Add utility methods to the wrapper
        wrapper.get_rate_limit_stats = lambda: _global_limiter.get_stats(
            key_func() if key_func else func.__name__
        )

        return wrapper

    return decorator


def sliding_window_rate_limit(
    calls: int = 60, window: int = 60, key_func: Callable = None
):
    """
    Decorator for sliding window rate limiting.

    Args:
        calls: Number of calls allowed in the window
        window: Time window in seconds
        key_func: Function to generate rate limiter key

    Example:
        @sliding_window_rate_limit(calls=100, window=300)  # 100 calls per 5 minutes
        async def api_call():
            pass
    """
    call_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=calls * 2))
    locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate key for rate limiter
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                if args and hasattr(args[0], "__class__"):
                    key = f"{args[0].__class__.__name__}.{func.__name__}"
                else:
                    key = func.__name__

            async with locks[key]:
                now = time.time()
                times = call_times[key]

                # Remove old entries outside the window
                while times and (float(now) - float(times[0])) > window:
                    times.popleft()

                # Check if we can make the call
                if len(times) >= calls:
                    oldest_call = times[0]
                    wait_time = window - (float(now) - float(oldest_call))

                    if wait_time > 0:
                        raise RateLimitExceeded(
                            f"Sliding window rate limit exceeded. "
                            f"Wait {wait_time:.1f}s before next call."
                        )

                # Record the call
                times.append(float(now))

            return await func(*args, **kwargs)

        return wrapper

    return decorator


# Utility functions


async def wait_for_rate_limit(key: str, tokens: int = 1) -> float:
    """
    Wait for rate limit tokens to be available.

    Args:
        key: Rate limiter key
        tokens: Number of tokens needed

    Returns:
        Actual wait time in seconds
    """
    start_time = time.time()
    limiter = _global_limiter.get_limiter(key)
    await limiter.wait_for_tokens(tokens)
    return time.time() - start_time


def get_rate_limit_stats(key: str) -> Dict[str, Any]:
    """
    Get rate limit statistics for a key.

    Args:
        key: Rate limiter key

    Returns:
        Statistics dictionary
    """
    return _global_limiter.get_stats(key)


def reset_rate_limiter(key: str):
    """
    Reset a specific rate limiter.

    Args:
        key: Rate limiter key to reset
    """
    if key in _global_limiter._limiters:
        limiter = _global_limiter._limiters[key]
        limiter.tokens = limiter.max_tokens
        limiter.last_refill = time.time()
        logger.info(f"Reset rate limiter for key: {key}")


def clear_all_rate_limiters():
    """Clear all rate limiters (for testing purposes)."""
    _global_limiter._limiters.clear()
    _global_limiter._call_history.clear()
    logger.info("Cleared all rate limiters")
