"""
Tests for input validation and security features in SejmApiClient.
"""

import pytest
from datetime import datetime, date, timedelta
from sejm_whiz.sejm_api.client import SejmApiClient
from sejm_whiz.sejm_api.exceptions import ValidationError


class TestSejmApiValidation:
    """Test validation methods in SejmApiClient."""

    def setup_method(self):
        """Set up test client."""
        self.client = SejmApiClient()

    def test_validate_endpoint_valid(self):
        """Test endpoint validation with valid endpoints."""
        valid_endpoints = [
            "sessions",
            "deputies/123",
            "votings/10/1/1/1",
            "committees",
            "current-term",
        ]

        for endpoint in valid_endpoints:
            result = self.client._validate_endpoint(endpoint)
            assert isinstance(result, str)
            assert result == endpoint.strip("/")

    def test_validate_endpoint_invalid(self):
        """Test endpoint validation with invalid endpoints."""
        invalid_endpoints = [
            "",
            None,
            "../sessions",
            "sessions//committees",
            "http://evil.com/sessions",
            "sessions:8080",
            "sessions@evil.com",
            "sessions with spaces",
            "sessions?param=value",
            "sessions#fragment",
        ]

        for endpoint in invalid_endpoints:
            with pytest.raises(ValidationError):
                self.client._validate_endpoint(endpoint)

    def test_sanitize_error_message(self):
        """Test error message sanitization."""
        test_cases = [
            ("Simple error", "Simple error"),
            ("", "No error details available"),
            ("Error with token: abc123def", "Error with [REDACTED]"),
            ("API key: xyz789", "API [REDACTED]"),
            ("Password: secret123", "[REDACTED]"),
            ("IP: 192.168.1.1", "IP: [IP_REDACTED]"),
            ("Email: user@example.com", "Email: [EMAIL_REDACTED]"),
            ("A" * 300, "A" * 200),  # Test length limit
        ]

        for input_msg, expected_pattern in test_cases:
            result = self.client._sanitize_error_message(input_msg)
            if expected_pattern == input_msg:
                assert result == expected_pattern
            else:
                assert (
                    expected_pattern.replace("[REDACTED]", "") in result
                    or "[REDACTED]" in result
                    or "[IP_REDACTED]" in result
                    or "[EMAIL_REDACTED]" in result
                )

    def test_validate_pagination_params_valid(self):
        """Test valid pagination parameters."""
        valid_cases = [(None, None), (1, 0), (100, 50), (1000, 999)]

        for limit, offset in valid_cases:
            # Should not raise an exception
            self.client._validate_pagination_params(limit, offset)

    def test_validate_pagination_params_invalid(self):
        """Test invalid pagination parameters."""
        invalid_cases = [
            (0, None),  # limit too small
            (1001, None),  # limit too large
            (-1, None),  # negative limit
            (None, -1),  # negative offset
            ("10", None),  # wrong type
            (None, "5"),  # wrong type
        ]

        for limit, offset in invalid_cases:
            with pytest.raises(ValidationError):
                self.client._validate_pagination_params(limit, offset)

    def test_validate_date_param_valid(self):
        """Test valid date parameters."""
        valid_dates = [
            datetime(2020, 1, 1),
            date(2022, 6, 15),
            datetime.now(),
            datetime(2000, 1, 1),  # minimum date
        ]

        for test_date in valid_dates:
            result = self.client._validate_date_param(test_date, "test_date")
            assert isinstance(result, str)
            assert len(result) == 10  # YYYY-MM-DD format

    def test_validate_date_param_invalid(self):
        """Test invalid date parameters."""
        invalid_dates = [
            "2020-01-01",  # string instead of date
            1577836800,  # timestamp
            datetime(1999, 12, 31),  # before minimum date
            datetime.now() + timedelta(days=400),  # too far in future
        ]

        for test_date in invalid_dates:
            with pytest.raises(ValidationError):
                self.client._validate_date_param(test_date, "test_date")

    def test_validate_term_param_valid(self):
        """Test valid term parameters."""
        valid_terms = [None, 1, 10, 20]

        for term in valid_terms:
            # Should not raise an exception
            self.client._validate_term_param(term)

    def test_validate_term_param_invalid(self):
        """Test invalid term parameters."""
        invalid_terms = [0, -1, 21, "10", 10.5]

        for term in invalid_terms:
            with pytest.raises(ValidationError):
                self.client._validate_term_param(term)

    def test_validate_session_param_valid(self):
        """Test valid session parameters."""
        valid_sessions = [None, 1, 100, 1000]

        for session in valid_sessions:
            # Should not raise an exception
            self.client._validate_session_param(session)

    def test_validate_session_param_invalid(self):
        """Test invalid session parameters."""
        invalid_sessions = [0, -1, 1001, "100", 100.5]

        for session in invalid_sessions:
            with pytest.raises(ValidationError):
                self.client._validate_session_param(session)

    def test_validate_id_param_valid(self):
        """Test valid ID parameters."""
        valid_ids = [None, 1, 1000, 999999]

        for id_val in valid_ids:
            # Should not raise an exception
            self.client._validate_id_param(id_val, "test_id")

    def test_validate_id_param_invalid(self):
        """Test invalid ID parameters."""
        invalid_ids = [0, -1, "123", 123.5]

        for id_val in invalid_ids:
            with pytest.raises(ValidationError):
                self.client._validate_id_param(id_val, "test_id")


@pytest.mark.asyncio
class TestSejmApiSecurityIntegration:
    """Integration tests for security features."""

    def setup_method(self):
        """Set up test client."""
        self.client = SejmApiClient()

    async def test_endpoint_validation_in_request(self):
        """Test that endpoint validation is applied in _make_request."""
        with pytest.raises(ValidationError, match="contains invalid characters"):
            await self.client._make_request("../malicious")

    async def test_parameter_validation_in_methods(self):
        """Test parameter validation in API methods."""
        # Test invalid term parameter
        with pytest.raises(ValidationError, match="Term must be"):
            await self.client.get_sessions(term=0)

        # Test invalid limit parameter
        with pytest.raises(ValidationError, match="Limit must be"):
            await self.client.get_sessions(limit=1001)

        # Test invalid offset parameter
        with pytest.raises(ValidationError, match="Offset must be"):
            await self.client.get_sessions(offset=-1)

    async def test_date_validation_in_interpellations(self):
        """Test date validation in get_interpellations method."""
        with pytest.raises(ValidationError, match="date_from must be"):
            await self.client.get_interpellations(date_from="2020-01-01")

        with pytest.raises(ValidationError, match="date_to must be"):
            await self.client.get_interpellations(date_to=datetime(1999, 1, 1))
