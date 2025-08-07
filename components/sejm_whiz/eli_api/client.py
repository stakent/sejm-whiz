"""ELI API client for fetching Polish legal documents."""

import httpx
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from urllib.parse import urljoin, quote
import time
from dataclasses import dataclass

from .models import LegalDocument, Amendment, DocumentSearchResult
from .utils import validate_eli_id, sanitize_query

# Import for compatibility with document ingestion
try:
    from sejm_whiz.document_ingestion.config import DocumentIngestionConfig
except ImportError:
    DocumentIngestionConfig = None

logger = logging.getLogger(__name__)


@dataclass
class EliApiConfig:
    """Configuration for ELI API client."""

    base_url: str = "https://api.sejm.gov.pl"
    rate_limit: int = 10  # requests per second
    timeout: int = 30
    max_retries: int = 3
    user_agent: str = "sejm-whiz/1.0 (legal document analysis)"


class EliApiError(Exception):
    """Base exception for ELI API errors."""

    pass


class EliRateLimitError(EliApiError):
    """Rate limit exceeded error."""

    pass


class EliNotFoundError(EliApiError):
    """Document not found error."""

    pass


class EliValidationError(EliApiError):
    """Input validation error."""

    pass


class RateLimiter:
    """Token bucket rate limiter for API requests."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.tokens = rate_limit
        self.last_update = time.time()
        self.min_interval = 1.0 / rate_limit if rate_limit > 0 else 0

    async def acquire(self):
        """Acquire permission to make a request."""
        if self.rate_limit <= 0:
            return

        current_time = time.time()
        elapsed = current_time - self.last_update

        # Add tokens based on elapsed time
        self.tokens = min(self.rate_limit, self.tokens + elapsed * self.rate_limit)
        self.last_update = current_time

        if self.tokens >= 1:
            self.tokens -= 1
        else:
            sleep_time = (1 - self.tokens) / self.rate_limit
            await asyncio.sleep(sleep_time)
            self.tokens = 0


class EliApiClient:
    """Client for Polish ELI (European Legislation Identifier) API."""

    def __init__(self, config: Optional[Union[EliApiConfig, 'DocumentIngestionConfig']] = None):
        # Handle compatibility with DocumentIngestionConfig
        if config and DocumentIngestionConfig and isinstance(config, DocumentIngestionConfig):
            # Convert DocumentIngestionConfig to EliApiConfig
            self.config = EliApiConfig(
                base_url=getattr(config, 'eli_api_base_url', 'https://api.sejm.gov.pl'),
                rate_limit=getattr(config, 'eli_api_rate_limit', 10),
                timeout=getattr(config, 'eli_api_timeout', 30),
                max_retries=getattr(config, 'eli_api_max_retries', 3),
                user_agent=getattr(config, 'user_agent', 'sejm-whiz/1.0 (legal document analysis)')
            )
        else:
            self.config = config or EliApiConfig()
        
        self._client: Optional[httpx.AsyncClient] = None
        self._rate_limiter = RateLimiter(self.config.rate_limit)
        self._last_url: str = ""

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _ensure_client(self):
        """Ensure HTTP client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout),
                headers={
                    "User-Agent": self.config.user_agent,
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip, deflate",
                },
            )

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _make_request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make rate-limited request to ELI API with error handling."""
        await self._rate_limiter.acquire()
        await self._ensure_client()

        # Validate endpoint
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint

        url = urljoin(self.config.base_url, endpoint)
        self._last_url = url  # Store for error logging

        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.get(url, params=params)

                if response.status_code == 429:  # Rate limited
                    retry_after = int(response.headers.get("Retry-After", "60"))
                    logger.warning(f"Rate limited, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue

                if response.status_code == 404:
                    raise EliNotFoundError(f"Document not found: {endpoint}")

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    raise EliRateLimitError(f"Rate limit exceeded: {e}")
                elif e.response.status_code == 404:
                    raise EliNotFoundError(f"Document not found: {endpoint}")

                if attempt == self.config.max_retries - 1:
                    raise EliApiError(
                        f"HTTP error after {self.config.max_retries} attempts: {e}"
                    )

                logger.warning(f"HTTP error (attempt {attempt + 1}): {e}")
                await asyncio.sleep(2**attempt)  # Exponential backoff

            except httpx.RequestError as e:
                if attempt == self.config.max_retries - 1:
                    raise EliApiError(
                        f"Request error after {self.config.max_retries} attempts: {e}"
                    )

                logger.warning(f"Request error (attempt {attempt + 1}): {e}")
                await asyncio.sleep(2**attempt)

        raise EliApiError("Max retries exceeded")

    async def search_documents(
        self,
        query: Optional[str] = None,
        document_type: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> DocumentSearchResult:
        """Search for legal documents using ELI API."""

        # Validate and sanitize inputs
        if query:
            query = sanitize_query(query)

        if limit < 1 or limit > 1000:
            raise EliValidationError("Limit must be between 1 and 1000")

        if offset < 0:
            raise EliValidationError("Offset must be non-negative")

        params = {"limit": limit, "offset": offset, "format": "json"}

        if query:
            params["q"] = query

        # Temporarily disabled due to ELI API 403 error with type parameter
        # if document_type:
        #     params["type"] = document_type

        if date_from:
            params["date_from"] = date_from.strftime("%Y-%m-%d")

        if date_to:
            params["date_to"] = date_to.strftime("%Y-%m-%d")

        logger.info(f"Searching documents with query: {query}")

        try:
            # Use working publisher/year endpoint instead of problematic search endpoint
            # Fetch from both DU (Dziennik Ustaw) and MP (Monitor Polski) for current year
            current_year = datetime.now().year
            all_documents = []

            # Fetch ALL documents from DU (Dziennik Ustaw - primary official journal)
            try:
                du_result = await self._make_request(
                    f"/eli/acts/DU/{current_year}/", {}
                )
                du_documents = du_result.get("items", [])
                all_documents.extend(du_documents)
                logger.info(
                    f"Successfully fetched {len(du_documents)} documents from DU/{current_year}"
                )
            except Exception as e:
                logger.warning(f"Failed to fetch from DU: {e}")

            # Fetch ALL documents from MP (Monitor Polski - official announcements)
            try:
                mp_result = await self._make_request(
                    f"/eli/acts/MP/{current_year}/", {}
                )
                mp_documents = mp_result.get("items", [])
                all_documents.extend(mp_documents)
                logger.info(
                    f"Successfully fetched {len(mp_documents)} documents from MP/{current_year}"
                )
            except Exception as e:
                logger.warning(f"Failed to fetch from MP: {e}")

            # Apply limit to combined results
            all_documents = all_documents[:limit] if limit else all_documents

            # Create combined result
            result = {
                "items": all_documents,
                "count": len(all_documents),
                "totalCount": len(all_documents),
            }

            # Parse documents
            documents = []
            for doc_data in result.get("items", []):
                try:
                    document = LegalDocument.from_api_response(doc_data)
                    documents.append(document)
                except Exception as e:
                    logger.warning(f"Failed to parse document: {e}")
                    continue

            search_result = DocumentSearchResult(
                documents=documents,
                total=result.get("totalCount", len(documents)),
                offset=offset,
                limit=limit,
            )

            logger.info(f"Found {search_result.total} documents")
            return search_result

        except Exception as e:
            logger.error(f"Document search failed: {e} url: {getattr(self, '_last_url', 'unknown')}")
            raise

    async def get_document(self, eli_id: str) -> LegalDocument:
        """Get specific document by ELI identifier."""

        # Validate ELI ID
        if not validate_eli_id(eli_id):
            raise EliValidationError(f"Invalid ELI ID format: {eli_id}")

        endpoint = f"/document/{quote(eli_id, safe='')}"

        logger.info(f"Fetching document: {eli_id}")

        try:
            result = await self._make_request(endpoint)

            document = LegalDocument.from_api_response(result)
            logger.info(f"Successfully fetched document: {eli_id}")

            return document

        except EliNotFoundError:
            logger.warning(f"Document not found: {eli_id}")
            raise
        except Exception as e:
            logger.error(f"Failed to fetch document {eli_id}: {e}")
            raise

    async def get_document_content(self, eli_id: str, format: str = "html") -> str:
        """Get document content in specified format."""

        # Validate inputs
        if not validate_eli_id(eli_id):
            raise EliValidationError(f"Invalid ELI ID format: {eli_id}")

        valid_formats = ["html", "xml", "txt", "pdf"]
        if format not in valid_formats:
            raise EliValidationError(
                f"Invalid format: {format}. Must be one of {valid_formats}"
            )

        endpoint = f"/document/{quote(eli_id, safe='')}/content"
        params = {"format": format}

        logger.info(f"Fetching document content: {eli_id} (format: {format})")

        try:
            # For content requests, we expect text response
            await self._rate_limiter.acquire()
            await self._ensure_client()

            url = urljoin(self.config.base_url, endpoint)
            response = await self._client.get(url, params=params)

            if response.status_code == 404:
                raise EliNotFoundError(f"Document content not found: {eli_id}")

            response.raise_for_status()

            content = response.text
            logger.info(
                f"Successfully fetched content for {eli_id} ({len(content)} chars)"
            )

            return content

        except EliNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch content for {eli_id}: {e}")
            raise

    async def get_document_amendments(self, eli_id: str) -> List[Amendment]:
        """Get amendments for a specific document."""

        # Validate ELI ID
        if not validate_eli_id(eli_id):
            raise EliValidationError(f"Invalid ELI ID format: {eli_id}")

        endpoint = f"/document/{quote(eli_id, safe='')}/amendments"

        logger.info(f"Fetching amendments for: {eli_id}")

        try:
            result = await self._make_request(endpoint)

            amendments = []
            for amendment_data in result.get("amendments", []):
                try:
                    amendment = Amendment.from_api_response(amendment_data)
                    amendments.append(amendment)
                except Exception as e:
                    logger.warning(f"Failed to parse amendment: {e}")
                    continue

            logger.info(f"Found {len(amendments)} amendments for {eli_id}")
            return amendments

        except EliNotFoundError:
            logger.warning(f"No amendments found for document: {eli_id}")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch amendments for {eli_id}: {e}")
            raise

    async def get_recent_documents(
        self, days: int = 7, document_types: Optional[List[str]] = None
    ) -> List[LegalDocument]:
        """Get recently published documents."""

        if days < 1 or days > 365:
            raise EliValidationError("Days must be between 1 and 365")

        date_from = datetime.now() - timedelta(days=days)
        document_types = document_types or ["ustawa", "rozporządzenie", "kodeks"]

        all_documents = []

        for doc_type in document_types:
            try:
                result = await self.search_documents(
                    document_type=doc_type, date_from=date_from, limit=100
                )

                all_documents.extend(result.documents)
                logger.info(
                    f"Found {len(result.documents)} recent {doc_type} documents"
                )

            except Exception as e:
                logger.error(f"Failed to fetch recent {doc_type} documents: {e}")
                continue

        # Sort by publication date
        all_documents.sort(key=lambda x: x.published_date or datetime.min, reverse=True)

        logger.info(f"Total recent documents found: {len(all_documents)}")
        
        # For compatibility with document ingestion pipeline, convert to dict format
        documents_as_dicts = []
        for doc in all_documents:
            try:
                doc_dict = {
                    "id": doc.id,
                    "identifier": doc.id,
                    "title": doc.title,
                    "url": getattr(doc, 'url', ''),
                    "document_type": doc.document_type,
                    "published_date": doc.published_date,
                    "language": doc.language,
                }
                documents_as_dicts.append(doc_dict)
            except Exception as e:
                logger.warning(f"Failed to convert document to dict format: {e}")
                continue
        
        return documents_as_dicts

    async def batch_get_documents(
        self, eli_ids: List[str], max_batch_size: int = 50, max_concurrent: int = 10
    ) -> List[Union[LegalDocument, None]]:
        """Fetch multiple documents in batch with rate limiting and concurrency controls.

        Args:
            eli_ids: List of ELI IDs to fetch
            max_batch_size: Maximum number of documents to fetch in one batch
            max_concurrent: Maximum number of concurrent requests

        Returns:
            List of documents (None for failed requests)

        Raises:
            EliValidationError: If batch size exceeds limits or input is invalid
        """

        # Input validation
        if not eli_ids:
            return []

        if not isinstance(eli_ids, list):
            raise EliValidationError("eli_ids must be a list")

        if len(eli_ids) > max_batch_size:
            raise EliValidationError(
                f"Batch size {len(eli_ids)} exceeds maximum allowed {max_batch_size}"
            )

        if max_concurrent < 1:
            raise EliValidationError("max_concurrent must be at least 1")

        # Validate individual ELI IDs
        invalid_ids = [
            eli_id
            for eli_id in eli_ids
            if not isinstance(eli_id, str) or not eli_id.strip()
        ]
        if invalid_ids:
            raise EliValidationError(f"Invalid ELI IDs found: {invalid_ids}")

        # Remove duplicates while preserving order
        unique_eli_ids = []
        seen = set()
        for eli_id in eli_ids:
            eli_id_clean = eli_id.strip()
            if eli_id_clean not in seen:
                seen.add(eli_id_clean)
                unique_eli_ids.append(eli_id_clean)

        if len(unique_eli_ids) != len(eli_ids):
            logger.info(
                f"Removed {len(eli_ids) - len(unique_eli_ids)} duplicate ELI IDs from batch"
            )

        logger.info(
            f"Starting batch fetch of {len(unique_eli_ids)} documents with max_concurrent={max_concurrent}"
        )

        # Process documents with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_single_document(eli_id: str) -> Union[LegalDocument, None]:
            """Fetch a single document with semaphore control."""
            async with semaphore:
                try:
                    document = await self.get_document(eli_id)
                    return document

                except EliNotFoundError:
                    logger.warning(f"Document not found in batch: {eli_id}")
                    return None
                except Exception as e:
                    logger.error(f"Failed to fetch document {eli_id} in batch: {e}")
                    return None

        # Execute all requests concurrently with semaphore control
        tasks = [fetch_single_document(eli_id) for eli_id in unique_eli_ids]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Map results back to original order (including duplicates)
        results_map = dict(zip(unique_eli_ids, results))
        final_results = [results_map[eli_id.strip()] for eli_id in eli_ids]

        successful_count = sum(1 for r in final_results if r is not None)
        logger.info(
            f"Batch fetched {successful_count}/{len(eli_ids)} documents successfully"
        )
        return final_results

    async def fetch_document_content(
        self, document_id: str, document_url: str
    ) -> Optional[Tuple[Union[str, bytes], str, Dict[str, Any]]]:
        """Fetch document content with metadata (compatibility method for document ingestion)."""
        try:
            # Try to get document metadata first
            try:
                document = await self.get_document(document_id)
                metadata = {
                    "title": document.title,
                    "document_type": document.document_type,
                    "published_date": document.published_date,
                    "language": document.language,
                }
            except Exception:
                # If metadata fetch fails, use basic metadata
                metadata = {"title": document_id, "document_type": "unknown"}

            # Fetch content in HTML format
            try:
                content = await self.get_document_content(document_id, "html")
                return content, "html", metadata
            except EliNotFoundError:
                # Try other formats if HTML fails
                for format_type in ["xml", "txt"]:
                    try:
                        content = await self.get_document_content(document_id, format_type)
                        return content, format_type, metadata
                    except Exception:
                        continue
                
                return None

        except Exception as e:
            logger.error(f"Failed to fetch document content for {document_id}: {e} url: {getattr(self, '_last_url', 'unknown')}")
            return None


# Global client instance for convenience
_client: Optional[EliApiClient] = None


async def get_client(config: Optional[EliApiConfig] = None) -> EliApiClient:
    """Get global ELI API client instance."""
    global _client

    if _client is None:
        _client = EliApiClient(config)
        await _client._ensure_client()

    return _client


async def close_client():
    """Close global client instance."""
    global _client
    if _client:
        await _client.close()
        _client = None
