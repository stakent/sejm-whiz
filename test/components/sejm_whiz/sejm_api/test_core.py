"""
Core test module for sejm_api component.

This module imports and runs all tests for the sejm_api component,
providing a comprehensive test suite for the Polish Sejm API integration.
"""

# Import all test modules to ensure they're discovered by pytest
from .test_client import TestSejmApiClient
from .test_models import TestSejmApiModels
from .test_rate_limiter import (
    TestRateLimiter,
    TestGlobalRateLimiter,
    TestRateLimitDecorator,
    TestSlidingWindowRateLimit,
    TestUtilityFunctions,
    TestRateLimiterEdgeCases,
)
from .test_exceptions import (
    TestSejmApiError,
    TestRateLimitExceeded,
    TestSpecificExceptions,
    TestErrorFactoryFunctions,
    TestErrorUtilities,
    TestExceptionInheritance,
)


# Test basic component functionality
def test_component_imports():
    """Test that the sejm_api component can be imported successfully."""
    from sejm_whiz.sejm_api import SejmApiClient, Session, VotingResult

    assert SejmApiClient is not None
    assert Session is not None
    assert VotingResult is not None


def test_component_functionality():
    """Test basic component functionality."""
    from sejm_whiz.sejm_api import SejmApiClient

    # Should be able to create client instance
    client = SejmApiClient()
    assert client is not None
    assert hasattr(client, "get_sessions")
    assert hasattr(client, "get_deputies")
    assert hasattr(client, "health_check")


__all__ = [
    # Client tests
    "TestSejmApiClient",
    # Model tests
    "TestSejmApiModels",
    # Rate limiter tests
    "TestRateLimiter",
    "TestGlobalRateLimiter",
    "TestRateLimitDecorator",
    "TestSlidingWindowRateLimit",
    "TestUtilityFunctions",
    "TestRateLimiterEdgeCases",
    # Exception tests
    "TestSejmApiError",
    "TestRateLimitExceeded",
    "TestSpecificExceptions",
    "TestErrorFactoryFunctions",
    "TestErrorUtilities",
    "TestExceptionInheritance",
]
