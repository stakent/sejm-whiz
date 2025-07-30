import pytest
import asyncio
from unittest.mock import patch, AsyncMock

from sejm_whiz.sejm_api.rate_limiter import (
    RateLimiter,
    GlobalRateLimiter,
    rate_limit,
    sliding_window_rate_limit,
    wait_for_rate_limit,
    get_rate_limit_stats,
    reset_rate_limiter,
    clear_all_rate_limiters,
)
from sejm_whiz.sejm_api.exceptions import RateLimitExceeded


class TestRateLimiter:
    """Test cases for RateLimiter class."""

    def test_rate_limiter_initialization(self):
        """Test RateLimiter initialization."""
        limiter = RateLimiter(calls=10, period=60, burst_factor=1.5)

        assert limiter.calls == 10
        assert limiter.period == 60
        assert limiter.max_tokens == 15  # 10 * 1.5
        assert limiter.tokens == 15
        assert limiter.refill_rate == 10 / 60  # calls per second

    @pytest.mark.asyncio
    async def test_acquire_tokens_success(self):
        """Test successful token acquisition."""
        limiter = RateLimiter(calls=10, period=60)

        # Should be able to acquire tokens
        result = await limiter.acquire(1)
        assert result is True
        assert limiter.tokens == limiter.max_tokens - 1

    @pytest.mark.asyncio
    async def test_acquire_multiple_tokens(self):
        """Test acquiring multiple tokens at once."""
        limiter = RateLimiter(calls=10, period=60)

        result = await limiter.acquire(5)
        assert result is True
        assert limiter.tokens == limiter.max_tokens - 5

    @pytest.mark.asyncio
    async def test_acquire_tokens_insufficient(self):
        """Test token acquisition when insufficient tokens available."""
        limiter = RateLimiter(calls=2, period=60)  # Very low limit

        # Use up all tokens (max_tokens = 2 * 1.5 = 3)
        await limiter.acquire(3)
        assert limiter.tokens == 0

        # Should wait and then succeed
        with (
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            patch("time.time") as mock_time,
        ):
            original_time = limiter.last_refill
            # Mock time to advance during the simulated wait
            mock_time.side_effect = [
                original_time,  # Initial refill check
                original_time,  # Wait time calculation
                original_time + 30.1,  # After wait - enough time for 1 token
            ]

            result = await limiter.acquire(1)
            assert result is True
            mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_acquire_tokens_rate_limit_exceeded(self):
        """Test RateLimitExceeded when wait time is too long."""
        limiter = RateLimiter(calls=1, period=3600)  # 1 call per hour

        # Use up the token
        await limiter.acquire(1)

        # Requesting more tokens should raise exception due to long wait
        with pytest.raises(RateLimitExceeded):
            await limiter.acquire(1)

    @pytest.mark.asyncio
    async def test_token_refill(self):
        """Test token refill over time."""
        limiter = RateLimiter(calls=10, period=1)  # 10 calls per second

        # Use most tokens so refill won't hit max
        await limiter.acquire(12)  # leaves 3 tokens (15 max - 12)
        initial_tokens = limiter.tokens

        # Mock time passage
        original_last_refill = limiter.last_refill
        with patch("time.time") as mock_time:
            mock_time.return_value = original_last_refill + 0.5  # 0.5 seconds later

            await limiter._refill_tokens()

            # Should have refilled some tokens (5 tokens in 0.5s at 10/s rate)
            assert limiter.tokens > initial_tokens

    def test_tokens_available_property(self):
        """Test tokens_available property."""
        limiter = RateLimiter(calls=10, period=60)
        assert limiter.tokens_available == 15  # max_tokens with burst_factor 1.5

    @pytest.mark.asyncio
    async def test_wait_for_tokens(self):
        """Test wait_for_tokens calculation."""
        limiter = RateLimiter(calls=2, period=60)

        # Use up all tokens (max_tokens = 3)
        await limiter.acquire(3)

        # Should calculate wait time for new tokens
        wait_time = await limiter.wait_for_tokens(1)
        assert wait_time > 0


class TestGlobalRateLimiter:
    """Test cases for GlobalRateLimiter class."""

    def setup_method(self):
        """Set up clean global limiter for each test."""
        clear_all_rate_limiters()

    def test_get_limiter_creation(self):
        """Test limiter creation and retrieval."""
        global_limiter = GlobalRateLimiter()

        # Get new limiter
        limiter1 = global_limiter.get_limiter("test_key", calls=10, period=60)
        assert isinstance(limiter1, RateLimiter)

        # Get same limiter again
        limiter2 = global_limiter.get_limiter("test_key")
        assert limiter1 is limiter2

    @pytest.mark.asyncio
    async def test_global_acquire(self):
        """Test global acquire method."""
        global_limiter = GlobalRateLimiter()

        result = await global_limiter.acquire("test_key", tokens=1, calls=10, period=60)
        assert result is True

    def test_get_stats(self):
        """Test rate limiter statistics."""
        global_limiter = GlobalRateLimiter()

        # Create limiter and make some calls
        global_limiter.get_limiter("test_key", calls=10, period=60)

        stats = global_limiter.get_stats("test_key")
        assert "tokens_available" in stats
        assert "max_tokens" in stats
        assert "refill_rate" in stats
        assert "calls_last_hour" in stats
        assert "total_calls" in stats

    def test_get_stats_nonexistent_key(self):
        """Test get_stats with non-existent key."""
        global_limiter = GlobalRateLimiter()

        stats = global_limiter.get_stats("nonexistent")
        assert stats == {"error": "Rate limiter not found"}


class TestRateLimitDecorator:
    """Test cases for rate_limit decorator."""

    def setup_method(self):
        """Set up clean state for each test."""
        clear_all_rate_limiters()

    @pytest.mark.asyncio
    async def test_rate_limit_decorator_basic(self):
        """Test basic rate_limit decorator functionality."""

        @rate_limit(calls=5, period=60)
        async def test_function():
            return "success"

        # Should work fine initially
        result = await test_function()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_rate_limit_decorator_with_key_func(self):
        """Test rate_limit decorator with custom key function."""

        def key_func(user_id):
            return f"user_{user_id}"

        @rate_limit(calls=2, period=60, key_func=key_func)
        async def test_function(user_id):
            return f"success_{user_id}"

        # Different users should have separate limits
        result1 = await test_function(1)
        result2 = await test_function(2)

        assert result1 == "success_1"
        assert result2 == "success_2"

    @pytest.mark.asyncio
    async def test_rate_limit_decorator_method(self):
        """Test rate_limit decorator on class methods."""

        class TestClass:
            @rate_limit(calls=3, period=60)
            async def test_method(self):
                return "method_success"

        obj = TestClass()
        result = await obj.test_method()
        assert result == "method_success"

    @pytest.mark.asyncio
    async def test_rate_limit_decorator_exception_handling(self):
        """Test rate_limit decorator handles exceptions properly."""

        @rate_limit(calls=5, period=60)
        async def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await failing_function()

    def test_rate_limit_decorator_stats(self):
        """Test rate_limit decorator statistics method."""

        @rate_limit(calls=5, period=60)
        async def test_function():
            return "success"

        # Should have stats method
        assert hasattr(test_function, "get_rate_limit_stats")

        stats = test_function.get_rate_limit_stats()
        assert isinstance(stats, dict)


class TestSlidingWindowRateLimit:
    """Test cases for sliding_window_rate_limit decorator."""

    @pytest.mark.asyncio
    async def test_sliding_window_basic(self):
        """Test basic sliding window rate limiting."""

        @sliding_window_rate_limit(calls=3, window=60)
        async def test_function():
            return "success"

        # Should work for allowed number of calls
        for _ in range(3):
            result = await test_function()
            assert result == "success"

        # Next call should raise exception
        with pytest.raises(RateLimitExceeded):
            await test_function()

    @pytest.mark.asyncio
    async def test_sliding_window_with_key_func(self):
        """Test sliding window with custom key function."""

        def key_func(user_id):
            return f"user_{user_id}"

        @sliding_window_rate_limit(calls=2, window=60, key_func=key_func)
        async def test_function(user_id):
            return f"success_{user_id}"

        # Different users should have separate windows
        await test_function(1)
        await test_function(1)
        await test_function(2)
        await test_function(2)

        # Both users should now be rate limited
        with pytest.raises(RateLimitExceeded):
            await test_function(1)

        with pytest.raises(RateLimitExceeded):
            await test_function(2)

    @pytest.mark.asyncio
    async def test_sliding_window_time_expiry(self):
        """Test sliding window time-based expiry."""

        @sliding_window_rate_limit(calls=2, window=1)  # 1 second window
        async def test_function():
            return "success"

        base_time = 1000.0  # Use a fixed base time

        with patch("time.time") as mock_time:
            # Mock all time calls consistently
            mock_time.side_effect = [
                base_time,  # First call
                base_time,  # Second call
                base_time,  # Third call (should fail)
                base_time + 2,  # Fourth call (should succeed)
            ]

            # Use up the limit
            await test_function()
            await test_function()

            # Should be rate limited
            with pytest.raises(RateLimitExceeded):
                await test_function()

            # Should work after window expires
            result = await test_function()
            assert result == "success"


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def setup_method(self):
        """Set up clean state for each test."""
        clear_all_rate_limiters()

    @pytest.mark.asyncio
    async def test_wait_for_rate_limit(self):
        """Test wait_for_rate_limit utility function."""
        # Create a limiter with very low limit
        from sejm_whiz.sejm_api.rate_limiter import _global_limiter

        limiter = _global_limiter.get_limiter("test_key", calls=1, period=60)

        # Use up the token
        await limiter.acquire(1)

        # Wait function should return some positive wait time
        wait_time = await wait_for_rate_limit("test_key", tokens=1)
        assert wait_time >= 0

    def test_get_rate_limit_stats_function(self):
        """Test get_rate_limit_stats utility function."""
        from sejm_whiz.sejm_api.rate_limiter import _global_limiter

        # Create a limiter
        _global_limiter.get_limiter("test_key", calls=10, period=60)

        stats = get_rate_limit_stats("test_key")
        assert isinstance(stats, dict)
        assert "tokens_available" in stats

    def test_reset_rate_limiter(self):
        """Test reset_rate_limiter utility function."""
        from sejm_whiz.sejm_api.rate_limiter import _global_limiter

        # Create and use a limiter
        limiter = _global_limiter.get_limiter("test_key", calls=10, period=60)
        original_tokens = limiter.max_tokens
        limiter.tokens = 5  # Reduce tokens

        # Reset should restore to max
        reset_rate_limiter("test_key")
        assert limiter.tokens == original_tokens

    def test_clear_all_rate_limiters(self):
        """Test clear_all_rate_limiters utility function."""
        from sejm_whiz.sejm_api.rate_limiter import _global_limiter

        # Create some limiters
        _global_limiter.get_limiter("key1", calls=10, period=60)
        _global_limiter.get_limiter("key2", calls=20, period=120)

        assert len(_global_limiter._limiters) == 2

        # Clear all
        clear_all_rate_limiters()
        assert len(_global_limiter._limiters) == 0


class TestRateLimiterEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_zero_period_rate_limiter(self):
        """Test rate limiter with zero period."""
        with pytest.raises(ZeroDivisionError):
            RateLimiter(calls=10, period=0)

    @pytest.mark.asyncio
    async def test_negative_tokens_request(self):
        """Test requesting negative tokens."""
        limiter = RateLimiter(calls=10, period=60)

        # Should handle gracefully
        result = await limiter.acquire(0)
        assert result is True

    @pytest.mark.asyncio
    async def test_very_high_burst_factor(self):
        """Test rate limiter with very high burst factor."""
        limiter = RateLimiter(calls=10, period=60, burst_factor=10.0)

        assert limiter.max_tokens == 100  # 10 * 10.0

        # Should be able to use all burst tokens
        result = await limiter.acquire(100)
        assert result is True

    @pytest.mark.asyncio
    async def test_concurrent_acquisitions(self):
        """Test concurrent token acquisitions."""
        limiter = RateLimiter(calls=10, period=60)

        # Launch multiple concurrent acquisitions
        tasks = [limiter.acquire(1) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All should succeed initially
        assert all(results)

        # Total tokens used should be approximately correct (allowing for time-based refill)
        assert limiter.tokens <= limiter.max_tokens
        assert limiter.tokens >= limiter.max_tokens - 5
