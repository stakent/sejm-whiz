import asyncio
from typing import Dict, List, Optional, Any
import httpx
from datetime import datetime, date, timedelta
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
from sejm_whiz.logging import get_enhanced_logger, add_context_to_message

logger = get_enhanced_logger(__name__)


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

    def _get_status_message(self, status_code: int) -> str:
        """
        Get clean status message based on HTTP status code.

        Args:
            status_code: HTTP status code

        Returns:
            Clean, descriptive status message
        """
        status_messages = {
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "Not Found",
            405: "Method Not Allowed",
            429: "Rate Limited",
            500: "Internal Server Error",
            502: "Bad Gateway",
            503: "Service Unavailable",
            504: "Gateway Timeout",
        }

        return status_messages.get(status_code, f"HTTP Error {status_code}")

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
                    # Use HTTP status code for clean error messages (log once at debug level)
                    status_message = self._get_status_message(e.response.status_code)
                    context_msg = add_context_to_message(
                        logger,
                        "DEBUG",
                        f"HTTP {e.response.status_code}: {status_message}",
                        status_code=e.response.status_code,
                        api_url=url,
                        attempt=f"{attempt + 1}/{self.max_retries}",
                    )
                    logger.debug(context_msg)
                    raise SejmApiError(
                        f"HTTP {e.response.status_code}: {status_message} url: {url}"
                    ) from e

            except httpx.RequestError as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt
                    context_msg = add_context_to_message(
                        logger,
                        "WARNING",
                        f"Request error {e}, retrying in {wait_time}s",
                        api_url=url,
                        attempt=f"{attempt + 1}/{self.max_retries}",
                        retry_delay=f"{wait_time}s",
                    )
                    logger.warning(context_msg)
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    context_msg = add_context_to_message(
                        logger,
                        "ERROR",
                        f"Request failed: {e}",
                        api_url=url,
                        max_retries=self.max_retries,
                    )
                    logger.error(context_msg)
                    raise SejmApiError(f"Request failed: {e} url: {url}") from e

        context_msg = add_context_to_message(
            logger,
            "ERROR",
            "Max retries exceeded",
            api_url=url,
            max_retries=self.max_retries,
        )
        logger.error(context_msg)
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

        Since Sejm API doesn't have a current-term endpoint,
        we default to term 10 (current as of 2025).

        Returns:
            Current term number
        """
        # Sejm API uses term-specific endpoints like /sejm/term10/
        # No current-term endpoint exists, so use current term (10)
        return 10

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
            # Enhanced error logging with health check context
            context_msg = add_context_to_message(
                logger,
                "ERROR",
                f"Health check failed: {e}",
                api_endpoint="current-term",
                api_url=getattr(self, "_last_url", "unknown"),
            )
            logger.error(context_msg)
            return False

    # Legal Act Methods for Multi-API Integration

    async def get_act_with_full_text(self, term: int, number: int) -> Dict[str, Any]:
        """Get legal act with full text content from Sejm API.

        NOTE: This method only accesses document content endpoints.
        Videos endpoint (/sejm/term{term}/videos) is excluded as it contains
        video content, not text documents suitable for legal analysis.

        Args:
            term: Parliamentary term number (e.g., 10)
            number: Document number within the term (e.g., 1, 2, 3...)

        Returns:
            Dictionary with act data including full text
        """
        logger.info(
            f"Attempting to fetch document {number} from term {term} via Sejm API"
        )

        try:
            # Try to get print document (legislative document with PDF attachments)
            # Endpoint: /sejm/term{term}/prints/{number}
            endpoint = f"sejm/term{term}/prints/{number}"
            print_data = await self._make_request(endpoint, {})

            # Extract document text from PDF attachments if available
            document_text = ""
            pdf_attachments = print_data.get("attachments", [])

            if pdf_attachments:
                # For now, create a summary from metadata until PDF processing is implemented
                document_text = f"Legislative Document #{number}\n"
                document_text += f"Title: {print_data.get('title', 'No title')}\n"
                document_text += (
                    f"Document Date: {print_data.get('documentDate', 'Unknown')}\n"
                )
                document_text += (
                    f"Delivery Date: {print_data.get('deliveryDate', 'Unknown')}\n"
                )
                document_text += f"PDF Attachments: {', '.join(pdf_attachments)}\n"

                # Include additional prints information
                additional_prints = print_data.get("additionalPrints", [])
                if additional_prints:
                    document_text += "\nAdditional Documents:\n"
                    for i, additional in enumerate(additional_prints, 1):
                        document_text += f"  {i}. {additional.get('title', 'No title')} ({additional.get('number', 'No number')})\n"
                        if additional.get("attachments"):
                            document_text += f"     Attachments: {', '.join(additional.get('attachments', []))}\n"

                logger.info(
                    f"Retrieved print document {number} from term {term} with {len(pdf_attachments)} PDF attachments"
                )
            else:
                # Fallback if no attachments
                document_text = f"Legislative Document #{number}: {print_data.get('title', 'No title')}\n"
                document_text += "Note: No PDF attachments found for this document."
                logger.warning(
                    f"Print document {number} from term {term} has no PDF attachments"
                )

            act_data = {
                "text": document_text,
                "title": print_data.get("title", f"Print Document {number}"),
                "term": term,
                "number": number,
                "document_type": "legislative_print",
                "source": "sejm_api_print",
                "document_date": print_data.get("documentDate"),
                "delivery_date": print_data.get("deliveryDate"),
                "change_date": print_data.get("changeDate"),
                "pdf_attachments": pdf_attachments,
                "additional_prints": print_data.get("additionalPrints", []),
                "raw_metadata": print_data,  # Store complete API response
            }

            return act_data

        except Exception as e:
            # Don't duplicate log - pipeline will log business-level error
            raise SejmApiError(
                f"Failed to fetch print document {number} from term {term}: {e}"
            )

    async def extract_act_metadata(self, act_data: Dict) -> Dict[str, Any]:
        """Extract standardized metadata from Sejm API response.

        NOTE: Metadata structure varies by Sejm API endpoint. Store as received from API.
        Sejm Term 10 example from /sejm/term10/prints:
        {
            "attachments": ["2.pdf"],
            "changeDate": "2023-11-29T12:27:56",
            "deliveryDate": "2023-11-13",
            "documentDate": "2023-11-13",
            "number": "2",
            "processPrint": ["2"],
            "term": 10,
            "title": "Poselski projekt uchwały..."
        }
        Unique ID: term + number (e.g., "10_2")

        Args:
            act_data: Raw act data from Sejm API

        Returns:
            Standardized metadata dictionary with original API fields preserved
        """
        try:
            # Generate unique document ID from term + number
            term = act_data.get("term")
            number = act_data.get("number")
            document_id = (
                f"{term}_{number}"
                if term and number
                else act_data.get("sejm_id", "unknown")
            )

            # Base metadata with original API fields preserved
            metadata = {
                "source_api": "sejm_api",
                "extraction_timestamp": datetime.now().isoformat(),
                "document_id": document_id,
                "unique_id": document_id,  # term_number format
                # Original API fields - store as received
                **act_data,  # Preserve all original fields
                # Standardized fields for compatibility
                "title": act_data.get("title", "Untitled"),
                "content_length": len(act_data.get("text", "")),
            }

            # Add additional metadata based on source type
            if act_data.get("source") == "sejm_api_voting":
                metadata["document_type"] = "voting_record"
                metadata["parliamentary_process"] = "voting"
            elif act_data.get("source") == "sejm_api_proceeding":
                metadata["document_type"] = "proceeding_record"
                metadata["parliamentary_process"] = "proceeding"
            else:
                metadata["document_type"] = "unknown"
                metadata["parliamentary_process"] = "unknown"

            logger.debug(
                f"Extracted metadata for {metadata['document_id']}: {metadata['document_type']}"
            )
            return metadata

        except Exception as e:
            logger.error(f"Failed to extract metadata from act data: {e}")
            # Return minimal metadata on error
            return {
                "source_api": "sejm_api",
                "extraction_timestamp": datetime.now().isoformat(),
                "document_id": act_data.get("sejm_id", "unknown"),
                "title": "Extraction Failed",
                "error": str(e),
            }

    def is_sejm_content_complete(self, act_data: Dict) -> bool:
        """Check if Sejm API returned complete act text.

        Args:
            act_data: Act data dictionary to validate

        Returns:
            True if content appears complete and usable
        """
        try:
            # Check if we have basic required fields
            required_fields = ["text", "title", "sejm_id"]
            if not all(field in act_data for field in required_fields):
                logger.debug("Missing required fields in act data")
                return False

            # Check text content length and quality
            text_content = act_data.get("text", "")
            if not isinstance(text_content, str):
                logger.debug("Text content is not a string")
                return False

            text_length = len(text_content.strip())

            # Minimum length threshold for Sejm content (more lenient than ELI)
            min_length = 100
            if text_length < min_length:
                logger.debug(f"Text too short: {text_length} < {min_length}")
                return False

            # Check for placeholder content
            placeholder_indicators = [
                "content not available",
                "placeholder implementation",
                "extraction failed",
                "not implemented",
            ]

            text_lower = text_content.lower()
            if any(indicator in text_lower for indicator in placeholder_indicators):
                logger.debug("Content appears to be placeholder")
                return False

            # Check for reasonable content structure
            # Sejm documents should have some structure (sentences, reasonable words)
            words = text_content.split()
            if len(words) < 10:  # Very basic check
                logger.debug(f"Too few words: {len(words)}")
                return False

            # Content passed all checks
            logger.debug(
                f"Sejm content validation passed: {text_length} chars, {len(words)} words"
            )
            return True

        except Exception as e:
            logger.error(f"Error validating Sejm content: {e}")
            return False
