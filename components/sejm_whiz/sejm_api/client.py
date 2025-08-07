import asyncio
from typing import Dict, List, Optional, Any
import httpx
from datetime import datetime, date, timedelta
import logging
import re

from .models import (
    Session,
    CommitteeSitting,
    Proceeding,
    Voting,
    Deputy,
    Committee,
    Interpellation,
)
from .rate_limiter import rate_limit
from .exceptions import SejmApiError, RateLimitExceeded, ValidationError

logger = logging.getLogger(__name__)


class SejmApiClient:
    """
    Asynchronous client for the Polish Sejm API.

    Provides methods to fetch parliamentary proceedings data including:
    - Sessions and sittings
    - Votings and their results
    - Deputy information
    - Committee data
    - Interpellations and questions
    """

    BASE_URL = "https://api.sejm.gov.pl"

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        rate_limit_requests: int = 60,
        rate_limit_window: int = 60,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_window = rate_limit_window

        self._client: Optional[httpx.AsyncClient] = None
        self._last_url: str = ""  # Store for error logging

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _ensure_client(self):
        """Ensure the HTTP client is initialized."""
        if not self._client:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={"User-Agent": "sejm-whiz/1.0.0", "Accept": "application/json"},
            )

    def _validate_endpoint(self, endpoint: str) -> str:
        """
        Validate and sanitize API endpoint to prevent URL manipulation.

        Args:
            endpoint: API endpoint to validate

        Returns:
            Validated endpoint string

        Raises:
            ValidationError: If endpoint is invalid or potentially malicious
        """
        if not endpoint or not isinstance(endpoint, str):
            raise ValidationError("Invalid endpoint parameter")

        # Remove leading/trailing slashes and validate format
        clean_endpoint = endpoint.strip("/")

        # Ensure only allowed characters (alphanumeric, hyphens, underscores, slashes)
        if not re.match(r"^[a-zA-Z0-9\-_/]+$", clean_endpoint):
            raise ValidationError("Endpoint contains invalid characters")

        # Prevent path traversal attempts
        if ".." in clean_endpoint or "//" in clean_endpoint:
            raise ValidationError("Invalid endpoint format")

        # Check for suspicious patterns
        if any(
            pattern in clean_endpoint.lower() for pattern in ["http", "ftp", ":", "@"]
        ):
            raise ValidationError("Endpoint contains suspicious patterns")

        return clean_endpoint

    def _sanitize_error_message(self, error_text: str) -> str:
        """
        Sanitize error messages to prevent information disclosure.

        Args:
            error_text: Raw error message

        Returns:
            Sanitized error message
        """
        if not error_text:
            return "No error details available"

        # Limit message length
        sanitized = error_text[:200]

        # Remove potentially sensitive patterns
        patterns_to_redact = [
            (r"token[s]?[:\s=]+[\w\-\.]+", "[REDACTED]"),
            (r"password[s]?[:\s=]+[\w\-\.]+", "[REDACTED]"),
            (r"secret[s]?[:\s=]+[\w\-\.]+", "[REDACTED]"),
            (r"api[_\s]?key[s]?[:\s=]+[\w\-\.]+", "[REDACTED]"),
            (r"auth[a-z]*[:\s=]+[\w\-\.]+", "[REDACTED]"),
            (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP_REDACTED]"),
            (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[EMAIL_REDACTED]"),
        ]

        for pattern, replacement in patterns_to_redact:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        return sanitized

    def _validate_pagination_params(
        self, limit: Optional[int], offset: Optional[int]
    ) -> None:
        """
        Validate pagination parameters.

        Args:
            limit: Number of items to return
            offset: Number of items to skip

        Raises:
            ValidationError: If parameters are invalid
        """
        if limit is not None:
            if not isinstance(limit, int) or limit < 1 or limit > 1000:
                raise ValidationError("Limit must be an integer between 1 and 1000")

        if offset is not None:
            if not isinstance(offset, int) or offset < 0:
                raise ValidationError("Offset must be a non-negative integer")

    def _validate_date_param(self, date_param: datetime, param_name: str) -> str:
        """
        Validate and format date parameter.

        Args:
            date_param: Date to validate
            param_name: Parameter name for error messages

        Returns:
            Formatted date string

        Raises:
            ValidationError: If date is invalid
        """
        if not isinstance(date_param, (datetime, date)):
            raise ValidationError(f"{param_name} must be a datetime or date object")

        # Ensure reasonable date range (year 2000 to one year in the future)
        min_date = datetime(2000, 1, 1).date()
        max_date = datetime.now().date() + timedelta(days=365)

        check_date = (
            date_param.date() if isinstance(date_param, datetime) else date_param
        )

        if check_date < min_date or check_date > max_date:
            raise ValidationError(
                f"{param_name} must be between {min_date} and {max_date}"
            )

        return check_date.strftime("%Y-%m-%d")

    def _validate_term_param(self, term: Optional[int]) -> None:
        """
        Validate term parameter.

        Args:
            term: Term number to validate

        Raises:
            ValidationError: If term is invalid
        """
        if term is not None:
            if not isinstance(term, int) or term < 1 or term > 20:
                raise ValidationError("Term must be an integer between 1 and 20")

    def _validate_session_param(self, session: Optional[int]) -> None:
        """
        Validate session parameter.

        Args:
            session: Session number to validate

        Raises:
            ValidationError: If session is invalid
        """
        if session is not None:
            if not isinstance(session, int) or session < 1 or session > 1000:
                raise ValidationError("Session must be an integer between 1 and 1000")

    def _validate_id_param(self, id_param: Optional[int], param_name: str) -> None:
        """
        Validate ID parameter.

        Args:
            id_param: ID to validate
            param_name: Parameter name for error messages

        Raises:
            ValidationError: If ID is invalid
        """
        if id_param is not None:
            if not isinstance(id_param, int) or id_param < 1:
                raise ValidationError(f"{param_name} must be a positive integer")

    @rate_limit(calls=60, period=60)
    async def _make_request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the Sejm API with rate limiting and error handling.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response data

        Raises:
            SejmApiError: On API errors
            RateLimitExceeded: When rate limit is exceeded
        """
        await self._ensure_client()

        # Validate endpoint to prevent URL manipulation
        validated_endpoint = self._validate_endpoint(endpoint)
        url = f"{self.BASE_URL}/{validated_endpoint}"
        self._last_url = url  # Store for error logging

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Making request to {url} (attempt {attempt + 1})")

                response = await self._client.get(url, params=params)
                response.raise_for_status()

                data = response.json()
                logger.debug(f"Received response with {len(str(data))} characters")

                return data

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    raise RateLimitExceeded("API rate limit exceeded") from e
                elif e.response.status_code >= 500 and attempt < self.max_retries - 1:
                    # Retry on server errors
                    wait_time = 2**attempt
                    logger.warning(
                        f"Server error {e.response.status_code}, retrying in {wait_time}s"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Sanitize error message to prevent information disclosure
                    sanitized_message = self._sanitize_error_message(e.response.text)
                    raise SejmApiError(
                        f"HTTP {e.response.status_code}: {sanitized_message} url: {url}"
                    ) from e

            except httpx.RequestError as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(f"Request error {e}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise SejmApiError(f"Request failed: {e} url: {url}") from e

        raise SejmApiError(f"Max retries exceeded url: {url}")

    # Session and Sitting methods

    async def get_sessions(
        self,
        term: Optional[int] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Session]:
        """
        Get parliamentary sessions.

        Args:
            term: Parliamentary term number (e.g., 10 for X kadencja)
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of Session objects
        """
        # Validate parameters
        self._validate_term_param(term)
        self._validate_pagination_params(limit, offset)

        params = {}
        if term:
            params["term"] = term
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset

        data = await self._make_request("sessions", params)
        return [Session.model_validate(item) for item in data.get("sessions", [])]

    async def get_proceeding_sittings(
        self,
        term: Optional[int] = None,
        session: Optional[int] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Proceeding]:
        """
        Get parliamentary proceedings (full Sejm assembly sessions).

        Note: This returns proceedings, each of which contains multiple sitting dates.

        Args:
            term: Parliamentary term number
            session: Session number (not used by API endpoint)
            limit: Maximum number of results (not used by API endpoint)
            offset: Offset for pagination (not used by API endpoint)

        Returns:
            List of Proceeding objects
        """
        # Validate parameters
        self._validate_term_param(term)
        self._validate_session_param(session)
        self._validate_pagination_params(limit, offset)

        # Term is required for proceedings endpoint, get current term if not provided
        if not term:
            term = await self.get_current_term()

        # Note: The proceedings endpoint doesn't support session, limit, offset parameters
        # It returns all proceedings for the term
        endpoint = f"sejm/term{term}/proceedings"
        data = await self._make_request(endpoint, {})
        # API returns a list directly, not wrapped in a dict
        if isinstance(data, list):
            return [Proceeding.model_validate(item) for item in data]
        else:
            return []

    # Voting methods

    async def get_votings(
        self,
        term: Optional[int] = None,
        session: Optional[int] = None,
        proceeding_sitting: Optional[int] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Voting]:
        """
        Get voting records.

        Args:
            term: Parliamentary term number
            session: Session number
            proceeding_sitting: Proceeding sitting number
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of Voting objects
        """
        # Validate parameters
        self._validate_term_param(term)
        self._validate_session_param(session)
        self._validate_id_param(proceeding_sitting, "proceeding_sitting")
        self._validate_pagination_params(limit, offset)

        params = {}
        if term:
            params["term"] = term
        if session:
            params["session"] = session
        if proceeding_sitting:
            params["sitting"] = proceeding_sitting
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset

        data = await self._make_request("votings", params)
        return [Voting.model_validate(item) for item in data.get("votings", [])]

    async def get_voting_results(
        self, term: int, session: int, proceeding_sitting: int, voting: int
    ) -> Dict[str, Any]:
        """
        Get detailed voting results for a specific voting.

        Args:
            term: Parliamentary term number
            session: Session number
            proceeding_sitting: Proceeding sitting number
            voting: Voting number

        Returns:
            Detailed voting results
        """
        # Validate parameters
        self._validate_term_param(term)
        self._validate_session_param(session)
        self._validate_id_param(proceeding_sitting, "proceeding_sitting")
        self._validate_id_param(voting, "voting")

        endpoint = f"votings/{term}/{session}/{proceeding_sitting}/{voting}"
        return await self._make_request(endpoint)

    # Deputy methods

    async def get_deputies(
        self,
        term: Optional[int] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Deputy]:
        """
        Get deputy information.

        Args:
            term: Parliamentary term number
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of Deputy objects
        """
        # Validate parameters
        self._validate_term_param(term)
        self._validate_pagination_params(limit, offset)

        params = {}
        if term:
            params["term"] = term
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset

        data = await self._make_request("deputies", params)
        return [Deputy.model_validate(item) for item in data.get("deputies", [])]

    async def get_deputy(self, deputy_id: int, term: Optional[int] = None) -> Deputy:
        """
        Get detailed information about a specific deputy.

        Args:
            deputy_id: Deputy ID
            term: Parliamentary term number

        Returns:
            Deputy object
        """
        # Validate parameters
        self._validate_id_param(deputy_id, "deputy_id")
        self._validate_term_param(term)

        params = {}
        if term:
            params["term"] = term

        endpoint = f"deputies/{deputy_id}"
        data = await self._make_request(endpoint, params)
        return Deputy.model_validate(data)

    # Committee methods

    async def get_committees(
        self,
        term: Optional[int] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Committee]:
        """
        Get parliamentary committees.

        Args:
            term: Parliamentary term number
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of Committee objects
        """
        # Validate parameters
        self._validate_term_param(term)
        self._validate_pagination_params(limit, offset)

        params = {}
        if term:
            params["term"] = term
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset

        data = await self._make_request("committees", params)
        return [Committee.model_validate(item) for item in data.get("committees", [])]

    # Interpellation methods

    async def get_interpellations(
        self,
        term: Optional[int] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> List[Interpellation]:
        """
        Get interpellations and parliamentary questions.

        Args:
            term: Parliamentary term number
            limit: Maximum number of results
            offset: Offset for pagination
            date_from: Filter from date
            date_to: Filter to date

        Returns:
            List of Interpellation objects
        """
        # Validate parameters
        self._validate_term_param(term)
        self._validate_pagination_params(limit, offset)

        params = {}
        if term:
            params["term"] = term
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        if date_from:
            params["dateFrom"] = self._validate_date_param(date_from, "date_from")
        if date_to:
            params["dateTo"] = self._validate_date_param(date_to, "date_to")

        data = await self._make_request("interpellations", params)
        return [
            Interpellation.model_validate(item)
            for item in data.get("interpellations", [])
        ]

    # Processing information methods

    async def get_proceedings(
        self,
        term: Optional[int] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Proceeding]:
        """
        Get parliamentary proceedings (full Sejm assembly sessions).

        Args:
            term: Parliamentary term number
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of Proceeding objects
        """
        # Validate parameters
        self._validate_term_param(term)
        self._validate_pagination_params(limit, offset)

        params = {}
        if term:
            params["term"] = term
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset

        data = await self._make_request("proceedings", params)
        return [Proceeding.model_validate(item) for item in data.get("proceedings", [])]

    # Committee sitting methods

    async def get_committee_sittings_by_date(
        self,
        term: int,
        date: datetime,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[CommitteeSitting]:
        """
        Get committee sittings for a specific date.

        Args:
            term: Parliamentary term number
            date: Date to get sittings for
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of CommitteeSitting objects
        """
        # Validate parameters
        self._validate_term_param(term)
        date_str = self._validate_date_param(date, "date")
        self._validate_pagination_params(limit, offset)

        params = {}
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset

        endpoint = f"sejm/term{term}/committees/sittings/{date_str}"
        data = await self._make_request(endpoint, params)
        return [
            CommitteeSitting.model_validate(item) for item in data.get("sittings", [])
        ]

    async def get_committee_sittings(
        self,
        term: int,
        committee_code: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[CommitteeSitting]:
        """
        Get sittings for a specific committee.

        Args:
            term: Parliamentary term number
            committee_code: Committee code
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of CommitteeSitting objects
        """
        # Validate parameters
        self._validate_term_param(term)
        self._validate_pagination_params(limit, offset)

        if not committee_code or not isinstance(committee_code, str):
            raise ValidationError("Committee code must be a non-empty string")

        params = {}
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset

        endpoint = f"sejm/term{term}/committees/{committee_code}/sittings"
        data = await self._make_request(endpoint, params)
        return [
            CommitteeSitting.model_validate(item) for item in data.get("sittings", [])
        ]

    async def get_committee_sitting_details(
        self, term: int, committee_code: str, sitting_num: int
    ) -> CommitteeSitting:
        """
        Get detailed information about a specific committee sitting.

        Args:
            term: Parliamentary term number
            committee_code: Committee code
            sitting_num: Sitting number

        Returns:
            CommitteeSitting object with detailed information
        """
        # Validate parameters
        self._validate_term_param(term)
        self._validate_id_param(sitting_num, "sitting_num")

        if not committee_code or not isinstance(committee_code, str):
            raise ValidationError("Committee code must be a non-empty string")

        endpoint = f"sejm/term{term}/committees/{committee_code}/sittings/{sitting_num}"
        data = await self._make_request(endpoint)
        return CommitteeSitting.model_validate(data)

    async def get_committee_sitting_transcript(
        self, term: int, committee_code: str, sitting_num: int, format: str = "html"
    ) -> str:
        """
        Get committee sitting transcript in specified format.

        Args:
            term: Parliamentary term number
            committee_code: Committee code
            sitting_num: Sitting number
            format: Transcript format ("html" or "pdf")

        Returns:
            Transcript content as string
        """
        # Validate parameters
        self._validate_term_param(term)
        self._validate_id_param(sitting_num, "sitting_num")

        if not committee_code or not isinstance(committee_code, str):
            raise ValidationError("Committee code must be a non-empty string")

        if format not in ["html", "pdf"]:
            raise ValidationError("Format must be 'html' or 'pdf'")

        endpoint = f"sejm/term{term}/committees/{committee_code}/sittings/{sitting_num}/{format}"
        response = await self._make_request(endpoint)

        # Return the content directly for transcripts
        if isinstance(response, dict) and "content" in response:
            return response["content"]
        return str(response)

    # Utility methods

    async def get_current_term(self) -> int:
        """
        Get the current parliamentary term number.

        Returns:
            Current term number
        """
        data = await self._make_request("current-term")
        return data.get("term", 10)  # Default to term 10 if not found

    async def health_check(self) -> bool:
        """
        Check if the API is accessible.

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            await self._make_request("current-term")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e} url: {getattr(self, '_last_url', 'unknown')}")
            return False
