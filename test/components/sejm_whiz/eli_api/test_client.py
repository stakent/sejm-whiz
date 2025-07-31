"""Tests for ELI API client."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import httpx

from sejm_whiz.eli_api.client import (
    EliApiClient,
    EliApiConfig,
    EliApiError,
    EliNotFoundError,
    EliValidationError,
    RateLimiter,
    get_client,
    close_client,
)
from sejm_whiz.eli_api.models import LegalDocument, DocumentSearchResult, Amendment


class TestRateLimiter:
    """Test rate limiter functionality."""

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests_within_limit(self):
        """Test that requests within rate limit are allowed."""
        limiter = RateLimiter(rate_limit=10)  # 10 requests per second

        # Should allow immediate requests
        await limiter.acquire()
        await limiter.acquire()

        # No exception should be raised
        assert True

    @pytest.mark.asyncio
    async def test_rate_limiter_delays_excessive_requests(self):
        """Test that excessive requests are delayed."""
        limiter = RateLimiter(rate_limit=2)  # 2 requests per second

        start_time = asyncio.get_event_loop().time()

        # First two requests should be immediate
        await limiter.acquire()
        await limiter.acquire()

        # Third request should be delayed
        await limiter.acquire()

        end_time = asyncio.get_event_loop().time()
        elapsed = end_time - start_time

        # Should have taken at least 0.5 seconds (1/2 rate limit)
        assert elapsed >= 0.4  # Small tolerance for timing

    @pytest.mark.asyncio
    async def test_rate_limiter_zero_limit_allows_all(self):
        """Test that zero rate limit allows all requests."""
        limiter = RateLimiter(rate_limit=0)

        # Should allow many requests without delay
        for _ in range(10):
            await limiter.acquire()

        assert True


class TestEliApiConfig:
    """Test ELI API configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EliApiConfig()

        assert config.base_url == "https://api.sejm.gov.pl/eli"
        assert config.rate_limit == 10
        assert config.timeout == 30
        assert config.max_retries == 3
        assert "sejm-whiz" in config.user_agent

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EliApiConfig(
            base_url="https://test.api.gov.pl/eli",
            rate_limit=5,
            timeout=60,
            max_retries=5,
            user_agent="test-agent/1.0",
        )

        assert config.base_url == "https://test.api.gov.pl/eli"
        assert config.rate_limit == 5
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.user_agent == "test-agent/1.0"


class TestEliApiClient:
    """Test ELI API client functionality."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return EliApiConfig(
            base_url="https://test.api.gov.pl/eli",
            rate_limit=0,  # No rate limiting for tests
            timeout=10,
            max_retries=1,
        )

    @pytest.fixture
    def client(self, config):
        """Test client instance."""
        return EliApiClient(config)

    @pytest.mark.asyncio
    async def test_client_context_manager(self, client):
        """Test client as async context manager."""
        async with client as c:
            assert c._client is not None

        # Client should be closed after context
        assert client._client is None

    @pytest.mark.asyncio
    async def test_client_manual_lifecycle(self, client):
        """Test manual client lifecycle management."""
        await client._ensure_client()
        assert client._client is not None

        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_make_request_success(self, client):
        """Test successful API request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"test": "data"}

        with (
            patch.object(client, "_ensure_client"),
            patch.object(client, "_rate_limiter") as mock_limiter,
        ):
            mock_limiter.acquire = AsyncMock()
            client._client = AsyncMock()
            client._client.get.return_value = mock_response

            result = await client._make_request("/test")

            assert result == {"test": "data"}
            mock_limiter.acquire.assert_called_once()
            client._client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_make_request_404_error(self, client):
        """Test 404 error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with (
            patch.object(client, "_ensure_client"),
            patch.object(client, "_rate_limiter") as mock_limiter,
        ):
            mock_limiter.acquire = AsyncMock()
            client._client = AsyncMock()
            client._client.get.return_value = mock_response

            with pytest.raises(EliNotFoundError):
                await client._make_request("/test")

    @pytest.mark.asyncio
    async def test_make_request_rate_limit_error(self):
        """Test rate limit error handling."""
        # Use a config with more retries for this test
        config = EliApiConfig(
            base_url="https://test.api.gov.pl/eli",
            rate_limit=0,
            timeout=10,
            max_retries=3,
        )
        client = EliApiClient(config)

        call_count = 0

        def mock_get_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call returns 429
                mock_response = MagicMock()
                mock_response.status_code = 429
                mock_response.headers = {"Retry-After": "1"}
                mock_response.raise_for_status.return_value = (
                    None  # Don't raise for 429
                )
                return mock_response
            else:
                # Second call returns success
                mock_response_success = MagicMock()
                mock_response_success.status_code = 200
                mock_response_success.json.return_value = {"success": True}
                mock_response_success.raise_for_status.return_value = (
                    None  # Don't raise for 200
                )
                return mock_response_success

        with (
            patch.object(client, "_ensure_client"),
            patch.object(client, "_rate_limiter") as mock_limiter,
            patch("asyncio.sleep") as mock_sleep,
        ):
            mock_limiter.acquire = AsyncMock()
            client._client = AsyncMock()
            client._client.get.side_effect = mock_get_side_effect

            result = await client._make_request("/test")

            assert result == {"success": True}
            mock_sleep.assert_called_once_with(1)
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_make_request_http_error_with_retries(self, client):
        """Test HTTP error with retry logic."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_response
        )

        with (
            patch.object(client, "_ensure_client"),
            patch.object(client, "_rate_limiter") as mock_limiter,
            patch("asyncio.sleep") as mock_sleep,
        ):
            mock_limiter.acquire = AsyncMock()
            client._client = AsyncMock()
            client._client.get.return_value = mock_response

            with pytest.raises(EliApiError) as exc_info:
                await client._make_request("/test")

            assert "HTTP error after" in str(exc_info.value)
            # Should have made max_retries attempts
            assert client._client.get.call_count == client.config.max_retries

    @pytest.mark.asyncio
    async def test_search_documents_success(self, client):
        """Test successful document search."""
        mock_api_response = {
            "documents": [
                {
                    "eli_id": "pl/test/2023/1",
                    "title": "Test Act",
                    "type": "ustawa",
                    "status": "obowiązująca",
                }
            ],
            "total": 1,
        }

        with patch.object(client, "_make_request", return_value=mock_api_response):
            result = await client.search_documents(query="test", limit=10)

            assert isinstance(result, DocumentSearchResult)
            assert len(result.documents) == 1
            assert result.total == 1
            assert result.documents[0].eli_id == "pl/test/2023/1"

    @pytest.mark.asyncio
    async def test_search_documents_validation(self, client):
        """Test search validation."""
        with pytest.raises(EliValidationError):
            await client.search_documents(limit=0)  # Invalid limit

        with pytest.raises(EliValidationError):
            await client.search_documents(limit=2000)  # Limit too high

        with pytest.raises(EliValidationError):
            await client.search_documents(offset=-1)  # Negative offset

    @pytest.mark.asyncio
    async def test_get_document_success(self, client):
        """Test successful document retrieval."""
        mock_api_response = {
            "eli_id": "pl/test/2023/1",
            "title": "Test Act",
            "type": "ustawa",
            "status": "obowiązująca",
        }

        with patch.object(client, "_make_request", return_value=mock_api_response):
            result = await client.get_document("pl/test/2023/1")

            assert isinstance(result, LegalDocument)
            assert result.eli_id == "pl/test/2023/1"
            assert result.title == "Test Act"

    @pytest.mark.asyncio
    async def test_get_document_invalid_eli_id(self, client):
        """Test document retrieval with invalid ELI ID."""
        with pytest.raises(EliValidationError):
            await client.get_document("invalid-eli-id")

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, client):
        """Test document retrieval when document not found."""
        with patch.object(
            client, "_make_request", side_effect=EliNotFoundError("Not found")
        ):
            with pytest.raises(EliNotFoundError):
                await client.get_document("pl/test/2023/1")

    @pytest.mark.asyncio
    async def test_get_document_content_success(self, client):
        """Test successful document content retrieval."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>Document content</body></html>"

        with (
            patch.object(client, "_rate_limiter") as mock_limiter,
            patch.object(client, "_ensure_client"),
        ):
            mock_limiter.acquire = AsyncMock()
            client._client = AsyncMock()
            client._client.get.return_value = mock_response

            result = await client.get_document_content("pl/test/2023/1", "html")

            assert result == "<html><body>Document content</body></html>"

    @pytest.mark.asyncio
    async def test_get_document_content_invalid_format(self, client):
        """Test document content retrieval with invalid format."""
        with pytest.raises(EliValidationError):
            await client.get_document_content("pl/test/2023/1", "invalid_format")

    @pytest.mark.asyncio
    async def test_get_document_amendments_success(self, client):
        """Test successful amendment retrieval."""
        mock_api_response = {
            "amendments": [
                {
                    "eli_id": "pl/amendment/2023/1",
                    "target_eli_id": "pl/test/2023/1",
                    "amendment_type": "nowelizacja",
                    "title": "Amendment to Test Act",
                }
            ]
        }

        with patch.object(client, "_make_request", return_value=mock_api_response):
            result = await client.get_document_amendments("pl/test/2023/1")

            assert len(result) == 1
            assert isinstance(result[0], Amendment)
            assert result[0].eli_id == "pl/amendment/2023/1"

    @pytest.mark.asyncio
    async def test_get_recent_documents_success(self, client):
        """Test successful recent documents retrieval."""
        mock_api_response = {
            "documents": [
                {
                    "eli_id": "pl/recent/2023/1",
                    "title": "Recent Act",
                    "type": "ustawa",
                    "status": "obowiązująca",
                    "published_date": "2023-12-01T00:00:00Z",
                }
            ],
            "total": 1,
        }

        with patch.object(client, "search_documents") as mock_search:
            # Mock returns one document for "ustawa", empty for others
            def mock_search_side_effect(document_type=None, **kwargs):
                if document_type == "ustawa":
                    return DocumentSearchResult(
                        documents=[
                            LegalDocument(
                                eli_id="pl/recent/2023/1",
                                title="Recent Act",
                                document_type="ustawa",
                                status="obowiązująca",
                                published_date=datetime(2023, 12, 1),
                            )
                        ],
                        total=1,
                        offset=0,
                        limit=100,
                    )
                else:
                    return DocumentSearchResult(
                        documents=[], total=0, offset=0, limit=100
                    )

            mock_search.side_effect = mock_search_side_effect

            result = await client.get_recent_documents(days=7)

            assert len(result) == 1
            assert result[0].eli_id == "pl/recent/2023/1"
            # Verify it was called for all 3 document types
            assert mock_search.call_count == 3

    @pytest.mark.asyncio
    async def test_get_recent_documents_validation(self, client):
        """Test recent documents validation."""
        with pytest.raises(EliValidationError):
            await client.get_recent_documents(days=0)  # Invalid days

        with pytest.raises(EliValidationError):
            await client.get_recent_documents(days=400)  # Too many days

    @pytest.mark.asyncio
    async def test_batch_get_documents_success(self, client):
        """Test successful batch document retrieval."""
        eli_ids = ["pl/test/2023/1", "pl/test/2023/2"]

        def mock_get_document(eli_id):
            return LegalDocument(
                eli_id=eli_id,
                title=f"Test Act {eli_id[-1]}",
                document_type="ustawa",
                status="obowiązująca",
            )

        with patch.object(client, "get_document", side_effect=mock_get_document):
            result = await client.batch_get_documents(eli_ids)

            assert len(result) == 2
            assert all(doc is not None for doc in result)
            assert result[0].eli_id == "pl/test/2023/1"
            assert result[1].eli_id == "pl/test/2023/2"

    @pytest.mark.asyncio
    async def test_batch_get_documents_with_failures(self, client):
        """Test batch document retrieval with some failures."""
        eli_ids = ["pl/test/2023/1", "pl/nonexistent/2023/1"]

        def mock_get_document(eli_id):
            if "nonexistent" in eli_id:
                raise EliNotFoundError("Not found")
            return LegalDocument(
                eli_id=eli_id,
                title="Test Act",
                document_type="ustawa",
                status="obowiązująca",
            )

        with patch.object(client, "get_document", side_effect=mock_get_document):
            result = await client.batch_get_documents(eli_ids)

            assert len(result) == 2
            assert result[0] is not None  # First document found
            assert result[1] is None  # Second document not found

    @pytest.mark.asyncio
    async def test_batch_get_documents_size_limit(self, client):
        """Test batch size limit enforcement."""
        # Create a list exceeding the default limit of 50
        eli_ids = [f"pl/test/2023/{i}" for i in range(51)]

        with pytest.raises(
            EliValidationError, match="Batch size 51 exceeds maximum allowed 50"
        ):
            await client.batch_get_documents(eli_ids)

    @pytest.mark.asyncio
    async def test_batch_get_documents_custom_limits(self, client):
        """Test batch processing with custom limits."""
        eli_ids = ["pl/test/2023/1", "pl/test/2023/2", "pl/test/2023/3"]

        def mock_get_document(eli_id):
            return LegalDocument(
                eli_id=eli_id,
                title=f"Test Act {eli_id[-1]}",
                document_type="ustawa",
                status="obowiązująca",
            )

        with patch.object(client, "get_document", side_effect=mock_get_document):
            # Test with custom batch size limit
            result = await client.batch_get_documents(
                eli_ids, max_batch_size=5, max_concurrent=2
            )
            assert len(result) == 3
            assert all(doc is not None for doc in result)

            # Test batch size exceeding custom limit
            with pytest.raises(
                EliValidationError, match="Batch size 3 exceeds maximum allowed 2"
            ):
                await client.batch_get_documents(eli_ids, max_batch_size=2)

    @pytest.mark.asyncio
    async def test_batch_get_documents_input_validation(self, client):
        """Test input validation for batch requests."""

        # Test empty list
        result = await client.batch_get_documents([])
        assert result == []

        # Test non-list input
        with pytest.raises(EliValidationError, match="eli_ids must be a list"):
            await client.batch_get_documents("not_a_list")

        # Test invalid ELI IDs
        with pytest.raises(EliValidationError, match="Invalid ELI IDs found"):
            await client.batch_get_documents(["valid_id", "", None, "  "])

        # Test invalid max_concurrent
        with pytest.raises(
            EliValidationError, match="max_concurrent must be at least 1"
        ):
            await client.batch_get_documents(["pl/test/2023/1"], max_concurrent=0)

    @pytest.mark.asyncio
    async def test_batch_get_documents_duplicate_handling(self, client):
        """Test handling of duplicate ELI IDs."""
        eli_ids = [
            "pl/test/2023/1",
            "pl/test/2023/1",
            "pl/test/2023/2",
        ]  # Has duplicate
        call_count = 0

        def mock_get_document(eli_id):
            nonlocal call_count
            call_count += 1
            return LegalDocument(
                eli_id=eli_id,
                title=f"Test Act {eli_id[-1]}",
                document_type="ustawa",
                status="obowiązująca",
            )

        with patch.object(client, "get_document", side_effect=mock_get_document):
            result = await client.batch_get_documents(eli_ids)

            # Should return 3 results (preserving duplicates in output)
            assert len(result) == 3
            # But only call get_document 2 times (unique IDs)
            assert call_count == 2
            # Results should match original order
            assert result[0].eli_id == "pl/test/2023/1"
            assert result[1].eli_id == "pl/test/2023/1"  # Duplicate
            assert result[2].eli_id == "pl/test/2023/2"

    @pytest.mark.asyncio
    async def test_batch_get_documents_concurrency_control(self, client):
        """Test concurrency control with semaphore."""
        eli_ids = ["pl/test/2023/1", "pl/test/2023/2", "pl/test/2023/3"]
        concurrent_calls = 0
        max_concurrent_seen = 0

        async def mock_get_document(eli_id):
            nonlocal concurrent_calls, max_concurrent_seen
            concurrent_calls += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_calls)

            # Simulate async work
            import asyncio

            await asyncio.sleep(0.01)  # Small delay

            concurrent_calls -= 1
            return LegalDocument(
                eli_id=eli_id,
                title=f"Test Act {eli_id[-1]}",
                document_type="ustawa",
                status="obowiązująca",
            )

        with patch.object(client, "get_document", side_effect=mock_get_document):
            result = await client.batch_get_documents(eli_ids, max_concurrent=2)

            assert len(result) == 3
            assert all(doc is not None for doc in result)
            # Should not exceed max_concurrent limit
            assert max_concurrent_seen <= 2


class TestGlobalClient:
    """Test global client functions."""

    @pytest.mark.asyncio
    async def test_get_global_client(self):
        """Test getting global client instance."""
        # Reset global client
        import sejm_whiz.eli_api.client as client_module

        client_module._client = None

        client1 = await get_client()
        client2 = await get_client()

        # Should return same instance
        assert client1 is client2

        await close_client()

    @pytest.mark.asyncio
    async def test_close_global_client(self):
        """Test closing global client instance."""
        # Reset global client
        import sejm_whiz.eli_api.client as client_module

        client_module._client = None

        client = await get_client()
        assert client is not None

        await close_client()

        # Global client should be reset
        assert client_module._client is None
