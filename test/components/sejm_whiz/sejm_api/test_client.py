import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
import httpx

from sejm_whiz.sejm_api.client import SejmApiClient
from sejm_whiz.sejm_api.models import (
    Session,
    ProceedingSitting,
    Proceeding,
    Voting,
    Deputy,
)
from sejm_whiz.sejm_api.exceptions import (
    SejmApiError,
    RateLimitExceeded,
)


class TestSejmApiClient:
    """Test cases for SejmApiClient."""

    @pytest_asyncio.fixture
    async def client(self):
        """Create test client instance."""
        client = SejmApiClient(timeout=5.0, max_retries=2)
        yield client
        if client._client:
            await client._client.aclose()

    @pytest.fixture
    def mock_response(self):
        """Create mock HTTP response."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"sessions": []}
        response.raise_for_status.return_value = None
        return response

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test client initialization with custom parameters."""
        client = SejmApiClient(
            timeout=10.0, max_retries=5, rate_limit_requests=100, rate_limit_window=120
        )

        assert client.timeout == 10.0
        assert client.max_retries == 5
        assert client.rate_limit_requests == 100
        assert client.rate_limit_window == 120
        assert client._client is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager functionality."""
        async with SejmApiClient() as client:
            assert client._client is not None
            assert isinstance(client._client, httpx.AsyncClient)

        # Client should be closed after context exit
        assert client._client is None

    @pytest.mark.asyncio
    async def test_ensure_client(self, client):
        """Test HTTP client initialization."""
        assert client._client is None

        await client._ensure_client()

        assert client._client is not None
        assert isinstance(client._client, httpx.AsyncClient)
        # Check timeout is configured correctly
        assert client._client.timeout is not None

    @pytest.mark.asyncio
    async def test_make_request_success(self, client, mock_response):
        """Test successful API request."""
        with patch.object(client, "_ensure_client", new_callable=AsyncMock):
            client._client = AsyncMock()
            client._client.get.return_value = mock_response

            result = await client._make_request("test-endpoint", {"param": "value"})

            assert result == {"sessions": []}
            client._client.get.assert_called_once_with(
                f"{client.BASE_URL}/test-endpoint", params={"param": "value"}
            )

    @pytest.mark.asyncio
    async def test_make_request_http_error(self, client):
        """Test HTTP error handling."""
        with patch.object(client, "_ensure_client", new_callable=AsyncMock):
            client._client = AsyncMock()

            # Mock HTTP 404 error
            error = httpx.HTTPStatusError(
                "Not found",
                request=MagicMock(),
                response=MagicMock(status_code=404, text="Not found"),
            )
            client._client.get.side_effect = error

            with pytest.raises(SejmApiError) as exc_info:
                await client._make_request("nonexistent-endpoint")

            assert "HTTP 404" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_make_request_rate_limit(self, client):
        """Test rate limit error handling."""
        with patch.object(client, "_ensure_client", new_callable=AsyncMock):
            client._client = AsyncMock()

            # Mock HTTP 429 rate limit error
            error = httpx.HTTPStatusError(
                "Rate limit exceeded",
                request=MagicMock(),
                response=MagicMock(status_code=429, text="Rate limit exceeded"),
            )
            client._client.get.side_effect = error

            with pytest.raises(RateLimitExceeded):
                await client._make_request("test-endpoint")

    @pytest.mark.asyncio
    async def test_make_request_retry_on_server_error(self, client):
        """Test retry logic for server errors."""
        with patch.object(client, "_ensure_client", new_callable=AsyncMock):
            client._client = AsyncMock()

            # First call fails with 500, second succeeds
            error = httpx.HTTPStatusError(
                "Internal server error",
                request=MagicMock(),
                response=MagicMock(status_code=500, text="Internal server error"),
            )

            success_response = MagicMock()
            success_response.status_code = 200
            success_response.json.return_value = {"data": "success"}
            success_response.raise_for_status.return_value = None

            client._client.get.side_effect = [error, success_response]

            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await client._make_request("test-endpoint")

            assert result == {"data": "success"}
            assert client._client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_get_sessions(self, client):
        """Test getting parliamentary sessions."""
        mock_data = {
            "sessions": [
                {
                    "id": 1,
                    "term": 10,
                    "sessionNumber": 1,
                    "startDate": "2023-01-15",
                    "title": "Test Session",
                }
            ]
        }

        with patch.object(
            client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_data

            sessions = await client.get_sessions(term=10, limit=10)

            assert len(sessions) == 1
            assert isinstance(sessions[0], Session)
            assert sessions[0].term == 10
            assert sessions[0].session_number == 1

            mock_request.assert_called_once_with("sessions", {"term": 10, "limit": 10})

    @pytest.mark.asyncio
    async def test_get_proceeding_sittings(self, client):
        """Test getting parliamentary proceedings."""
        # API returns array directly, not wrapped in object
        mock_data = [
            {
                "number": 1,
                "title": "1. Posiedzenie Sejmu RP",
                "agenda": "<div>Agenda content</div>",
                "current": False,
                "dates": ["2023-01-15", "2023-01-16"],
            }
        ]

        with patch.object(
            client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_data

            proceedings = await client.get_proceeding_sittings(term=10, session=1)

            assert len(proceedings) == 1
            assert isinstance(proceedings[0], Proceeding)
            assert proceedings[0].number == 1
            assert proceedings[0].title == "1. Posiedzenie Sejmu RP"
            assert len(proceedings[0].dates) == 2

            mock_request.assert_called_once_with("sejm/term10/proceedings", {})

    @pytest.mark.asyncio
    async def test_get_votings(self, client):
        """Test getting voting records."""
        mock_data = {
            "votings": [
                {
                    "id": 1,
                    "term": 10,
                    "session": 1,
                    "sitting": 1,
                    "votingNumber": 1,
                    "date": "2023-01-15",
                    "title": "Test Voting",
                    "yesVotes": 230,
                    "noVotes": 200,
                    "abstainVotes": 30,
                    "totalVotes": 460,
                }
            ]
        }

        with patch.object(
            client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_data

            votings = await client.get_votings(term=10, session=1, proceeding_sitting=1)

            assert len(votings) == 1
            assert isinstance(votings[0], Voting)
            assert votings[0].yes_votes == 230
            assert votings[0].no_votes == 200

    @pytest.mark.asyncio
    async def test_get_deputies(self, client):
        """Test getting deputy information."""
        mock_data = {
            "deputies": [
                {
                    "id": 1,
                    "firstName": "Jan",
                    "lastName": "Kowalski",
                    "club": "Test Party",
                    "voivodeship": "mazowieckie",
                }
            ]
        }

        with patch.object(
            client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_data

            deputies = await client.get_deputies(term=10)

            assert len(deputies) == 1
            assert isinstance(deputies[0], Deputy)
            assert deputies[0].first_name == "Jan"
            assert deputies[0].last_name == "Kowalski"
            assert deputies[0].full_name == "Jan Kowalski"

    @pytest.mark.asyncio
    async def test_get_deputy_by_id(self, client):
        """Test getting specific deputy information."""
        mock_data = {
            "id": 123,
            "firstName": "Anna",
            "lastName": "Nowak",
            "email": "anna.nowak@sejm.pl",
            "club": "Test Party",
        }

        with patch.object(
            client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_data

            deputy = await client.get_deputy(123, term=10)

            assert isinstance(deputy, Deputy)
            assert deputy.id == 123
            assert deputy.first_name == "Anna"
            assert deputy.last_name == "Nowak"
            assert deputy.email == "anna.nowak@sejm.pl"

            mock_request.assert_called_once_with("deputies/123", {"term": 10})

    @pytest.mark.asyncio
    async def test_get_current_term(self, client):
        """Test getting current parliamentary term."""
        mock_data = {"term": 10}

        with patch.object(
            client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_data

            term = await client.get_current_term()

            assert term == 10
            mock_request.assert_called_once_with("current-term")

    @pytest.mark.asyncio
    async def test_get_current_term_default(self, client):
        """Test getting current term with default fallback."""
        mock_data = {}  # No term in response

        with patch.object(
            client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_data

            term = await client.get_current_term()

            assert term == 10  # Default value

    @pytest.mark.asyncio
    async def test_health_check_success(self, client):
        """Test successful health check."""
        with patch.object(
            client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = {"term": 10}

            is_healthy = await client.health_check()

            assert is_healthy is True
            mock_request.assert_called_once_with("current-term")

    @pytest.mark.asyncio
    async def test_health_check_failure(self, client):
        """Test failed health check."""
        with patch.object(
            client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.side_effect = SejmApiError("Connection failed")

            is_healthy = await client.health_check()

            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_voting_results_endpoint(self, client):
        """Test getting detailed voting results."""
        mock_data = {
            "voting": {
                "id": 1,
                "title": "Test Voting",
                "votes": [
                    {"deputyId": 1, "vote": "za"},
                    {"deputyId": 2, "vote": "przeciw"},
                ],
            }
        }

        with patch.object(
            client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_data

            result = await client.get_voting_results(10, 1, 1, 1)

            assert result == mock_data
            mock_request.assert_called_once_with("votings/10/1/1/1")

    @pytest.mark.asyncio
    async def test_get_interpellations_with_date_filter(self, client):
        """Test getting interpellations with date filters."""
        mock_data = {"interpellations": []}

        with patch.object(
            client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_data

            date_from = datetime(2023, 1, 1).date()
            date_to = datetime(2023, 12, 31).date()

            await client.get_interpellations(
                term=10, date_from=date_from, date_to=date_to
            )

            expected_params = {
                "term": 10,
                "dateFrom": "2023-01-01",
                "dateTo": "2023-12-31",
            }
            mock_request.assert_called_once_with("interpellations", expected_params)

    @pytest.mark.asyncio
    async def test_request_with_empty_params(self, client):
        """Test request with no parameters."""
        mock_data = {"sessions": []}

        with patch.object(
            client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_data

            await client.get_sessions()

            mock_request.assert_called_once_with("sessions", {})

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, client):
        """Test max retries exceeded scenario."""
        with patch.object(client, "_ensure_client", new_callable=AsyncMock):
            client._client = AsyncMock()

            # All requests fail
            error = httpx.RequestError("Connection failed")
            client._client.get.side_effect = error

            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(SejmApiError) as exc_info:
                    await client._make_request("test-endpoint")

                assert "Request failed" in str(exc_info.value)
                assert client._client.get.call_count == client.max_retries
