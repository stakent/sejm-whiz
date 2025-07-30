import pytest

from sejm_whiz.sejm_api.exceptions import (
    SejmApiError,
    RateLimitExceeded,
    AuthenticationError,
    AuthorizationError,
    ResourceNotFoundError,
    ValidationError,
    NetworkError,
    ServerError,
    DataParsingError,
    DeprecatedEndpointError,
    ConfigurationError,
    create_http_error,
    handle_api_exception,
    is_retryable_error,
    get_retry_delay,
)


class TestSejmApiError:
    """Test cases for base SejmApiError class."""

    def test_basic_error_creation(self):
        """Test basic error creation with message only."""
        error = SejmApiError("Test error message")

        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.status_code is None
        assert error.response_data is None
        assert error.endpoint is None

    def test_error_with_all_parameters(self):
        """Test error creation with all parameters."""
        response_data = {"error": "Not found"}
        error = SejmApiError(
            message="Resource not found",
            status_code=404,
            response_data=response_data,
            endpoint="/api/deputies/123",
        )

        assert error.message == "Resource not found"
        assert error.status_code == 404
        assert error.response_data == response_data
        assert error.endpoint == "/api/deputies/123"

    def test_error_string_formatting(self):
        """Test error string formatting with various fields."""
        error = SejmApiError(
            message="Test error", status_code=500, endpoint="/api/test"
        )

        error_str = str(error)
        assert "Test error" in error_str
        assert "Status: 500" in error_str
        assert "Endpoint: /api/test" in error_str

    def test_error_to_dict(self):
        """Test error conversion to dictionary."""
        response_data = {"detail": "Error details"}
        error = SejmApiError(
            message="Test error",
            status_code=400,
            endpoint="/api/test",
            response_data=response_data,
        )

        error_dict = error.to_dict()

        expected = {
            "error_type": "SejmApiError",
            "message": "Test error",
            "status_code": 400,
            "endpoint": "/api/test",
            "response_data": response_data,
        }

        assert error_dict == expected


class TestRateLimitExceeded:
    """Test cases for RateLimitExceeded exception."""

    def test_basic_rate_limit_error(self):
        """Test basic rate limit error creation."""
        error = RateLimitExceeded()

        assert error.status_code == 429
        assert "API rate limit exceeded" in str(error)
        assert error.retry_after is None

    def test_rate_limit_error_with_retry_after(self):
        """Test rate limit error with retry_after parameter."""
        error = RateLimitExceeded(message="Rate limit exceeded", retry_after=60.5)

        assert error.retry_after == 60.5
        error_str = str(error)
        assert "Retry after: 60.5s" in error_str

    def test_rate_limit_error_custom_message(self):
        """Test rate limit error with custom message."""
        error = RateLimitExceeded("Custom rate limit message")

        assert "Custom rate limit message" in str(error)
        assert error.status_code == 429


class TestSpecificExceptions:
    """Test cases for specific exception types."""

    def test_authentication_error(self):
        """Test AuthenticationError creation."""
        error = AuthenticationError("Invalid API key")

        assert error.status_code == 401
        assert error.message == "Invalid API key"

    def test_authentication_error_default_message(self):
        """Test AuthenticationError with default message."""
        error = AuthenticationError()

        assert error.message == "Authentication failed"
        assert error.status_code == 401

    def test_authorization_error(self):
        """Test AuthorizationError creation."""
        error = AuthorizationError("Access denied")

        assert error.status_code == 403
        assert error.message == "Access denied"

    def test_resource_not_found_error(self):
        """Test ResourceNotFoundError creation."""
        error = ResourceNotFoundError(
            message="Deputy not found", resource_type="deputy", resource_id="123"
        )

        assert error.status_code == 404
        assert error.resource_type == "deputy"
        assert error.resource_id == "123"

        error_str = str(error)
        assert "deputy: 123" in error_str

    def test_resource_not_found_error_with_type_only(self):
        """Test ResourceNotFoundError with resource type only."""
        error = ResourceNotFoundError(
            message="Resource not found", resource_type="session"
        )

        error_str = str(error)
        assert "Resource type: session" in error_str

    def test_validation_error(self):
        """Test ValidationError creation."""
        error = ValidationError(
            message="Invalid parameter", field="term", value="invalid"
        )

        assert error.status_code == 400
        assert error.field == "term"
        assert error.value == "invalid"

        error_str = str(error)
        assert "Field: term" in error_str
        assert "value: invalid" in error_str

    def test_network_error(self):
        """Test NetworkError creation."""
        error = NetworkError(message="Connection timeout", timeout=30.0)

        assert error.timeout == 30.0
        error_str = str(error)
        assert "Timeout: 30.0s" in error_str

    def test_server_error(self):
        """Test ServerError creation."""
        error = ServerError(message="Internal server error", is_temporary=True)

        assert error.is_temporary is True
        error_str = str(error)
        assert "Temporary error - retry recommended" in error_str

    def test_server_error_persistent(self):
        """Test ServerError with persistent error."""
        error = ServerError(message="Database corrupted", is_temporary=False)

        error_str = str(error)
        assert "Persistent server error" in error_str

    def test_data_parsing_error(self):
        """Test DataParsingError creation."""
        error = DataParsingError(
            message="Invalid JSON",
            raw_data='{"invalid": json}',
            expected_format="valid JSON",
        )

        assert error.raw_data == '{"invalid": json}'
        assert error.expected_format == "valid JSON"

        error_str = str(error)
        assert "Expected: valid JSON" in error_str

    def test_deprecated_endpoint_error(self):
        """Test DeprecatedEndpointError creation."""
        error = DeprecatedEndpointError(
            message="Endpoint deprecated",
            deprecated_endpoint="/api/v1/deputies",
            replacement_endpoint="/api/v2/deputies",
        )

        assert error.deprecated_endpoint == "/api/v1/deputies"
        assert error.replacement_endpoint == "/api/v2/deputies"

        error_str = str(error)
        assert "Deprecated: /api/v1/deputies" in error_str
        assert "Use instead: /api/v2/deputies" in error_str

    def test_configuration_error(self):
        """Test ConfigurationError creation."""
        error = ConfigurationError(
            message="Invalid timeout", config_key="timeout", config_value=-1
        )

        assert error.config_key == "timeout"
        assert error.config_value == -1

        error_str = str(error)
        assert "Config: timeout = -1" in error_str


class TestErrorFactoryFunctions:
    """Test cases for error factory functions."""

    def test_create_http_error_400(self):
        """Test create_http_error for 400 status."""
        error = create_http_error(
            status_code=400, message="Bad request", endpoint="/api/test"
        )

        assert isinstance(error, ValidationError)
        assert error.status_code == 400
        assert error.endpoint == "/api/test"

    def test_create_http_error_401(self):
        """Test create_http_error for 401 status."""
        error = create_http_error(401, "Unauthorized")

        assert isinstance(error, AuthenticationError)
        assert error.status_code == 401

    def test_create_http_error_403(self):
        """Test create_http_error for 403 status."""
        error = create_http_error(403, "Forbidden")

        assert isinstance(error, AuthorizationError)
        assert error.status_code == 403

    def test_create_http_error_404(self):
        """Test create_http_error for 404 status."""
        error = create_http_error(404, "Not found")

        assert isinstance(error, ResourceNotFoundError)
        assert error.status_code == 404

    def test_create_http_error_429(self):
        """Test create_http_error for 429 status."""
        error = create_http_error(429, "Rate limited")

        assert isinstance(error, RateLimitExceeded)
        assert error.status_code == 429

    def test_create_http_error_500(self):
        """Test create_http_error for 500 status."""
        error = create_http_error(500, "Internal server error")

        assert isinstance(error, ServerError)
        assert error.status_code == 500

    def test_create_http_error_unknown_status(self):
        """Test create_http_error for unknown status code."""
        error = create_http_error(418, "I'm a teapot")

        assert isinstance(error, SejmApiError)
        assert error.status_code == 418

    def test_handle_api_exception_decorator(self):
        """Test handle_api_exception decorator."""

        @handle_api_exception
        def test_function():
            raise ValueError("Test error")

        with pytest.raises(SejmApiError) as exc_info:
            test_function()

        assert "Unexpected error: Test error" in str(exc_info.value)

    def test_handle_api_exception_with_sejm_error(self):
        """Test handle_api_exception decorator with SejmApiError."""

        @handle_api_exception
        def test_function():
            raise ValidationError("Validation failed")

        # Should pass through SejmApiError unchanged
        with pytest.raises(ValidationError):
            test_function()


class TestErrorUtilities:
    """Test cases for error utility functions."""

    def test_is_retryable_error_rate_limit(self):
        """Test is_retryable_error with rate limit error."""
        error = RateLimitExceeded()
        assert is_retryable_error(error) is True

    def test_is_retryable_error_server_error_temporary(self):
        """Test is_retryable_error with temporary server error."""
        error = ServerError("Server error", is_temporary=True)
        assert is_retryable_error(error) is True

    def test_is_retryable_error_server_error_persistent(self):
        """Test is_retryable_error with persistent server error."""
        error = ServerError("Server error", is_temporary=False)
        assert is_retryable_error(error) is False

    def test_is_retryable_error_network_error(self):
        """Test is_retryable_error with network error."""
        error = NetworkError("Connection timeout")
        assert is_retryable_error(error) is True

    def test_is_retryable_error_5xx_status(self):
        """Test is_retryable_error with 5xx status code."""
        error = SejmApiError("Server error", status_code=503)
        assert is_retryable_error(error) is True

    def test_is_retryable_error_4xx_status(self):
        """Test is_retryable_error with 4xx status code."""
        error = SejmApiError("Client error", status_code=400)
        assert is_retryable_error(error) is False

    def test_is_retryable_error_non_sejm_error(self):
        """Test is_retryable_error with non-SejmApiError."""
        error = ValueError("Some error")
        assert is_retryable_error(error) is False

    def test_get_retry_delay_rate_limit(self):
        """Test get_retry_delay for rate limit error."""
        error = RateLimitExceeded(retry_after=30.0)
        delay = get_retry_delay(error, attempt=1)
        assert delay == 30.0

    def test_get_retry_delay_rate_limit_no_retry_after(self):
        """Test get_retry_delay for rate limit error without retry_after."""
        error = RateLimitExceeded()
        delay = get_retry_delay(error, attempt=2)
        assert delay == 120.0  # 60 * attempt

    def test_get_retry_delay_server_error(self):
        """Test get_retry_delay for server error."""
        error = ServerError("Server error")

        # Exponential backoff
        delay1 = get_retry_delay(error, attempt=1)
        delay2 = get_retry_delay(error, attempt=2)

        assert delay1 == 2.0  # 2^1
        assert delay2 == 4.0  # 2^2

    def test_get_retry_delay_server_error_max(self):
        """Test get_retry_delay for server error with max cap."""
        error = ServerError("Server error")
        delay = get_retry_delay(error, attempt=10)
        assert delay == 60.0  # Capped at 60 seconds

    def test_get_retry_delay_network_error(self):
        """Test get_retry_delay for network error."""
        error = NetworkError("Connection timeout")

        # Linear backoff
        delay1 = get_retry_delay(error, attempt=1)
        delay2 = get_retry_delay(error, attempt=2)

        assert delay1 == 5.0  # 5 * 1
        assert delay2 == 10.0  # 5 * 2

    def test_get_retry_delay_network_error_max(self):
        """Test get_retry_delay for network error with max cap."""
        error = NetworkError("Connection timeout")
        delay = get_retry_delay(error, attempt=10)
        assert delay == 30.0  # Capped at 30 seconds

    def test_get_retry_delay_generic_error(self):
        """Test get_retry_delay for generic error."""
        error = ValidationError("Invalid input")

        # Default exponential backoff
        delay1 = get_retry_delay(error, attempt=1)
        delay2 = get_retry_delay(error, attempt=2)

        assert delay1 == 2.0  # 2^1
        assert delay2 == 4.0  # 2^2

    def test_get_retry_delay_generic_error_max(self):
        """Test get_retry_delay for generic error with max cap."""
        error = ValidationError("Invalid input")
        delay = get_retry_delay(error, attempt=10)
        assert delay == 60.0  # Capped at 60 seconds


class TestExceptionInheritance:
    """Test exception inheritance and isinstance checks."""

    def test_specific_exceptions_inherit_from_base(self):
        """Test that specific exceptions inherit from SejmApiError."""
        exceptions = [
            RateLimitExceeded(),
            AuthenticationError(),
            AuthorizationError(),
            ResourceNotFoundError(),
            ValidationError(),
            NetworkError(),
            ServerError(),
            DataParsingError(),
            DeprecatedEndpointError(),
            ConfigurationError(),
        ]

        for exc in exceptions:
            assert isinstance(exc, SejmApiError)
            assert isinstance(exc, Exception)

    def test_exception_type_identification(self):
        """Test proper exception type identification."""
        rate_limit_error = RateLimitExceeded()
        auth_error = AuthenticationError()
        not_found_error = ResourceNotFoundError()

        assert isinstance(rate_limit_error, RateLimitExceeded)
        assert not isinstance(rate_limit_error, AuthenticationError)

        assert isinstance(auth_error, AuthenticationError)
        assert not isinstance(auth_error, ResourceNotFoundError)

        assert isinstance(not_found_error, ResourceNotFoundError)
        assert not isinstance(not_found_error, RateLimitExceeded)
