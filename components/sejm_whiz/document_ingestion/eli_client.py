"""ELI API client for fetching Polish legal documents."""

import httpx
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from urllib.parse import urljoin, quote
import time

from .config import get_ingestion_config, DocumentIngestionConfig

logger = logging.getLogger(__name__)


class ELIApiError(Exception):
    """ELI API specific error."""
    pass


class ELIRateLimitError(ELIApiError):
    """Rate limit exceeded error."""
    pass


class ELIClient:
    """Client for Polish ELI (European Legislation Identifier) API."""
    
    def __init__(self, config: Optional[DocumentIngestionConfig] = None):
        self.config = config or get_ingestion_config()
        self._client: Optional[httpx.AsyncClient] = None
        self._rate_limiter = RateLimiter(self.config.eli_api_rate_limit)
    
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
                timeout=httpx.Timeout(self.config.eli_api_timeout),
                headers={
                    "User-Agent": "sejm-whiz/1.0 (https://github.com/sejm-whiz/sejm-whiz)",
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip, deflate"
                }
            )
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make rate-limited request to ELI API."""
        await self._rate_limiter.acquire()
        await self._ensure_client()
        
        url = urljoin(self.config.eli_api_base_url, endpoint)
        
        for attempt in range(self.config.eli_api_max_retries):
            try:
                response = await self._client.get(url, params=params)
                
                if response.status_code == 429:  # Rate limited
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue
                
                if response.status_code == 404:
                    logger.info(f"Document not found: {url}")
                    return {}
                
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    raise ELIRateLimitError(f"Rate limit exceeded: {e}")
                
                if attempt == self.config.eli_api_max_retries - 1:
                    raise ELIApiError(f"HTTP error after {self.config.eli_api_max_retries} attempts: {e}")
                
                logger.warning(f"HTTP error (attempt {attempt + 1}): {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except httpx.RequestError as e:
                if attempt == self.config.eli_api_max_retries - 1:
                    raise ELIApiError(f"Request error after {self.config.eli_api_max_retries} attempts: {e}")
                
                logger.warning(f"Request error (attempt {attempt + 1}): {e}")
                await asyncio.sleep(2 ** attempt)
        
        raise ELIApiError("Max retries exceeded")
    
    async def search_documents(self, 
                              query: Optional[str] = None,
                              document_type: Optional[str] = None,
                              date_from: Optional[datetime] = None,
                              date_to: Optional[datetime] = None,
                              limit: int = 100,
                              offset: int = 0) -> Dict[str, Any]:
        """Search for legal documents using ELI API."""
        
        params = {
            "limit": limit,
            "offset": offset,
            "format": "json"
        }
        
        if query:
            params["q"] = query
        
        if document_type:
            params["type"] = document_type
        
        if date_from:
            params["date_from"] = date_from.strftime("%Y-%m-%d")
        
        if date_to:
            params["date_to"] = date_to.strftime("%Y-%m-%d")
        
        logger.info(f"Searching documents with params: {params}")
        
        try:
            result = await self._make_request("/search", params)
            
            logger.info(f"Found {result.get('total', 0)} documents")
            return result
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            raise
    
    async def get_document(self, eli_id: str) -> Dict[str, Any]:
        """Get specific document by ELI identifier."""
        
        endpoint = f"/document/{quote(eli_id, safe='')}"
        
        logger.info(f"Fetching document: {eli_id}")
        
        try:
            result = await self._make_request(endpoint)
            
            if result:
                logger.info(f"Successfully fetched document: {eli_id}")
            else:
                logger.warning(f"Document not found: {eli_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch document {eli_id}: {e}")
            raise
    
    async def get_document_content(self, eli_id: str, format: str = "html") -> str:
        """Get document content in specified format."""
        
        endpoint = f"/document/{quote(eli_id, safe='')}/content"
        params = {"format": format}
        
        logger.info(f"Fetching document content: {eli_id} (format: {format})")
        
        try:
            # For content requests, we expect text response
            await self._rate_limiter.acquire()
            await self._ensure_client()
            
            url = urljoin(self.config.eli_api_base_url, endpoint)
            response = await self._client.get(url, params=params)
            response.raise_for_status()
            
            content = response.text
            logger.info(f"Successfully fetched content for {eli_id} ({len(content)} chars)")
            
            return content
            
        except Exception as e:
            logger.error(f"Failed to fetch content for {eli_id}: {e}")
            raise
    
    async def get_document_amendments(self, eli_id: str) -> List[Dict[str, Any]]:
        """Get amendments for a specific document."""
        
        endpoint = f"/document/{quote(eli_id, safe='')}/amendments"
        
        logger.info(f"Fetching amendments for: {eli_id}")
        
        try:
            result = await self._make_request(endpoint)
            
            amendments = result.get('amendments', []) if result else []
            logger.info(f"Found {len(amendments)} amendments for {eli_id}")
            
            return amendments
            
        except Exception as e:
            logger.error(f"Failed to fetch amendments for {eli_id}: {e}")
            raise
    
    async def get_recent_documents(self, days: int = 7, document_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get recently published documents."""
        
        date_from = datetime.utcnow() - timedelta(days=days)
        document_types = document_types or self.config.legal_document_types
        
        all_documents = []
        
        for doc_type in document_types:
            try:
                result = await self.search_documents(
                    document_type=doc_type,
                    date_from=date_from,
                    limit=100
                )
                
                documents = result.get('documents', [])
                all_documents.extend(documents)
                
                logger.info(f"Found {len(documents)} recent {doc_type} documents")
                
            except Exception as e:
                logger.error(f"Failed to fetch recent {doc_type} documents: {e}")
                continue
        
        # Sort by publication date
        all_documents.sort(key=lambda x: x.get('published_date', ''), reverse=True)
        
        logger.info(f"Total recent documents found: {len(all_documents)}")
        return all_documents
    
    async def batch_get_documents(self, eli_ids: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
        """Fetch multiple documents in batch with rate limiting."""
        
        results = []
        
        for eli_id in eli_ids:
            try:
                document = await self.get_document(eli_id)
                results.append((eli_id, document))
                
            except Exception as e:
                logger.error(f"Failed to fetch document {eli_id} in batch: {e}")
                results.append((eli_id, {}))
        
        logger.info(f"Batch fetched {len(results)} documents")
        return results


class RateLimiter:
    """Simple rate limiter for API requests."""
    
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit  # requests per second
        self.min_interval = 1.0 / rate_limit if rate_limit > 0 else 0
        self.last_request_time = 0.0
    
    async def acquire(self):
        """Acquire permission to make a request."""
        if self.rate_limit <= 0:
            return
        
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()


# Global ELI client instance
_eli_client: Optional[ELIClient] = None


async def get_eli_client(config: Optional[DocumentIngestionConfig] = None) -> ELIClient:
    """Get global ELI client instance."""
    global _eli_client
    
    if _eli_client is None:
        _eli_client = ELIClient(config)
        await _eli_client._ensure_client()
    
    return _eli_client