"""
Custom exceptions for the Sejm API client.

This module defines specific exceptions for different error conditions
that can occur when interacting with the Polish Sejm API.
"""

from typing import Optional, Dict, Any


class SejmApiError(Exception):
    """
    Base exception for all Sejm API related errors.

    Attributes:
        message: Human-readable error message
        status_code: HTTP status code if applicable
        response_data: Raw response data from the API
        endpoint: API endpoint that caused the error
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        endpoint: Optional[str] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.response_data = response_data
        self.endpoint = endpoint
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return a formatted error message."""
        parts = [self.message]

        if self.status_code:
            parts.append(f"Status: {self.status_code}")

        if self.endpoint:
            parts.append(f"Endpoint: {self.endpoint}")

        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "status_code": self.status_code,
            "endpoint": self.endpoint,
            "response_data": self.response_data,
        }


class RateLimitExceeded(SejmApiError):
    """
    Exception raised when API rate limits are exceeded.

    This exception is raised when the client has made too many requests
    in a given time period and needs to wait before making more requests.
    """

    def __init__(
        self,
        message: str = "API rate limit exceeded",
        retry_after: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(message, status_code=429, **kwargs)
        self.retry_after = retry_after

    def __str__(self) -> str:
        """Return a formatted error message with retry information."""
        base_msg = super().__str__()
        if self.retry_after:
            return f"{base_msg} | Retry after: {self.retry_after:.1f}s"
        return base_msg


class AuthenticationError(SejmApiError):
    """
    Exception raised for authentication-related errors.

    This includes invalid API keys, expired tokens, or insufficient permissions.
    """

    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, status_code=401, **kwargs)


class AuthorizationError(SejmApiError):
    """
    Exception raised when access to a resource is forbidden.

    This occurs when the client is authenticated but doesn't have
    permission to access the requested resource.
    """

    def __init__(self, message: str = "Access forbidden", **kwargs):
        super().__init__(message, status_code=403, **kwargs)


class ResourceNotFoundError(SejmApiError):
    """
    Exception raised when a requested resource is not found.

    This includes non-existent deputies, votings, sessions, etc.
    """

    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, status_code=404, **kwargs)
        self.resource_type = resource_type
        self.resource_id = resource_id

    def __str__(self) -> str:
        """Return a formatted error message with resource information."""
        base_msg = super().__str__()
        if self.resource_type and self.resource_id:
            return f"{base_msg} | {self.resource_type}: {self.resource_id}"
        elif self.resource_type:
            return f"{base_msg} | Resource type: {self.resource_type}"
        return base_msg


class ValidationError(SejmApiError):
    """
    Exception raised for data validation errors.

    This includes invalid parameters, malformed data, or constraint violations.
    """

    def __init__(
        self,
        message: str = "Validation error",
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(message, status_code=400, **kwargs)
        self.field = field
        self.value = value

    def __str__(self) -> str:
        """Return a formatted error message with validation details."""
        base_msg = super().__str__()
        if self.field:
            field_info = f"Field: {self.field}"
            if self.value is not None:
                field_info += f" (value: {self.value})"
            return f"{base_msg} | {field_info}"
        return base_msg


class NetworkError(SejmApiError):
    """
    Exception raised for network-related errors.

    This includes connection timeouts, DNS resolution failures,
    and other network connectivity issues.
    """

    def __init__(
        self,
        message: str = "Network error occurred",
        timeout: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.timeout = timeout

    def __str__(self) -> str:
        """Return a formatted error message with timeout information."""
        base_msg = super().__str__()
        if self.timeout:
            return f"{base_msg} | Timeout: {self.timeout}s"
        return base_msg


class ServerError(SejmApiError):
    """
    Exception raised for server-side errors (5xx status codes).

    This indicates issues on the Sejm API server side that are
    typically temporary and may resolve with retry.
    """

    def __init__(
        self,
        message: str = "Server error occurred",
        is_temporary: bool = True,
        **kwargs,
    ):
        # Ensure status_code is set for server errors if not provided
        if "status_code" not in kwargs:
            kwargs["status_code"] = 500
        super().__init__(message, **kwargs)
        self.is_temporary = is_temporary

    def __str__(self) -> str:
        """Return a formatted error message with server error info."""
        base_msg = super().__str__()
        if self.is_temporary:
            return f"{base_msg} | Temporary error - retry recommended"
        return f"{base_msg} | Persistent server error"


class DataParsingError(SejmApiError):
    """
    Exception raised when API response data cannot be parsed.

    This occurs when the API returns malformed JSON or data that
    doesn't match the expected schema.
    """

    def __init__(
        self,
        message: str = "Failed to parse API response",
        raw_data: Optional[str] = None,
        expected_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.raw_data = raw_data
        self.expected_format = expected_format

    def __str__(self) -> str:
        """Return a formatted error message with parsing details."""
        base_msg = super().__str__()
        if self.expected_format:
            return f"{base_msg} | Expected: {self.expected_format}"
        return base_msg


class DeprecatedEndpointError(SejmApiError):
    """
    Exception raised when attempting to use a deprecated API endpoint.

    This helps maintain compatibility and guides users to updated endpoints.
    """

    def __init__(
        self,
        message: str = "API endpoint is deprecated",
        deprecated_endpoint: Optional[str] = None,
        replacement_endpoint: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.deprecated_endpoint = deprecated_endpoint
        self.replacement_endpoint = replacement_endpoint

    def __str__(self) -> str:
        """Return a formatted error message with deprecation info."""
        base_msg = super().__str__()
        parts = [base_msg]

        if self.deprecated_endpoint:
            parts.append(f"Deprecated: {self.deprecated_endpoint}")

        if self.replacement_endpoint:
            parts.append(f"Use instead: {self.replacement_endpoint}")

        return " | ".join(parts)


class ConfigurationError(SejmApiError):
    """
    Exception raised for client configuration errors.

    This includes invalid settings, missing required configuration,
    or incompatible option combinations.
    """

    def __init__(
        self,
        message: str = "Client configuration error",
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.config_key = config_key
        self.config_value = config_value

    def __str__(self) -> str:
        """Return a formatted error message with configuration details."""
        base_msg = super().__str__()
        if self.config_key:
            config_info = f"Config: {self.config_key}"
            if self.config_value is not None:
                config_info += f" = {self.config_value}"
            return f"{base_msg} | {config_info}"
        return base_msg


# Exception factory functions


def create_http_error(
    status_code: int,
    message: str,
    endpoint: Optional[str] = None,
    response_data: Optional[Dict[str, Any]] = None,
) -> SejmApiError:
    """
    Create appropriate exception based on HTTP status code.

    Args:
        status_code: HTTP status code
        message: Error message
        endpoint: API endpoint that caused the error
        response_data: Raw response data

    Returns:
        Appropriate exception instance
    """
    kwargs = {"endpoint": endpoint, "response_data": response_data}

    if status_code == 400:
        return ValidationError(message, **kwargs)
    elif status_code == 401:
        return AuthenticationError(message, **kwargs)
    elif status_code == 403:
        return AuthorizationError(message, **kwargs)
    elif status_code == 404:
        return ResourceNotFoundError(message, **kwargs)
    elif status_code == 429:
        return RateLimitExceeded(message, **kwargs)
    elif 500 <= status_code < 600:
        return ServerError(message, **kwargs)
    else:
        return SejmApiError(message, status_code=status_code, **kwargs)


def handle_api_exception(func):
    """
    Decorator to handle and convert HTTP exceptions to Sejm API exceptions.

    This decorator wraps functions that make HTTP requests and converts
    generic HTTP exceptions to more specific Sejm API exceptions.
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Convert generic exceptions to SejmApiError if not already
            if not isinstance(e, SejmApiError):
                raise SejmApiError(f"Unexpected error: {str(e)}") from e
            raise

    return wrapper


# Exception utilities


def is_retryable_error(exception: Exception) -> bool:
    """
    Check if an exception represents a retryable error.

    Args:
        exception: Exception to check

    Returns:
        True if the error is likely temporary and retryable
    """
    if isinstance(exception, RateLimitExceeded):
        return True
    elif isinstance(exception, ServerError):
        return exception.is_temporary
    elif isinstance(exception, NetworkError):
        return True
    elif isinstance(exception, SejmApiError):
        # Retry on 5xx status codes
        return exception.status_code and exception.status_code >= 500
    else:
        return False


def get_retry_delay(exception: Exception, attempt: int) -> float:
    """
    Calculate appropriate retry delay for an exception.

    Args:
        exception: Exception that occurred
        attempt: Current attempt number (1-based)

    Returns:
        Delay in seconds before retry
    """
    if isinstance(exception, RateLimitExceeded):
        return exception.retry_after or (60.0 * attempt)
    elif isinstance(exception, ServerError):
        # Exponential backoff for server errors
        return min(2.0**attempt, 60.0)
    elif isinstance(exception, NetworkError):
        # Linear backoff for network errors
        return min(5.0 * attempt, 30.0)
    else:
        # Default exponential backoff
        return min(2.0**attempt, 60.0)
