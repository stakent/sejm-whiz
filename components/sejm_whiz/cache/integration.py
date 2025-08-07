"""Cache integration for API clients and data processing pipelines."""

import logging
from typing import Any, Dict, Optional, Callable, Awaitable
from functools import wraps

from .manager import CacheManager, get_cache_manager

logger = logging.getLogger(__name__)


def cached_api_call(api_type: str, cache_manager: Optional[CacheManager] = None):
    """Decorator for caching API calls."""

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            manager = cache_manager or get_cache_manager()

            # Extract endpoint and params for cache key
            # This assumes the function has endpoint as first arg and params as kwargs
            endpoint = args[0] if args else "unknown"
            params = kwargs.copy()

            # Remove non-cacheable parameters
            cache_params = {
                k: v
                for k, v in params.items()
                if not k.startswith("_") and k not in ["timeout", "retries"]
            }

            # Try to get from cache first
            cached_result = manager.get_cached_api_response(
                api_type, endpoint, cache_params
            )
            if cached_result is not None:
                logger.debug(f"Cache hit for {api_type} API call: {endpoint}")
                return cached_result

            # Call the actual function
            logger.debug(f"Cache miss for {api_type} API call: {endpoint}")
            result = await func(*args, **kwargs)

            # Cache the result
            if result is not None:
                manager.cache_api_response(api_type, endpoint, cache_params, result)

            return result

        return wrapper

    return decorator


def cached_processing(data_type: str, cache_manager: Optional[CacheManager] = None):
    """Decorator for caching expensive processing operations."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            manager = cache_manager or get_cache_manager()

            # Generate identifier from function name and arguments
            # This is a simple approach - for production, consider more sophisticated key generation
            identifier = (
                f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            )

            # Try to get from cache first
            cached_result = manager.get_cached_processed_data(data_type, identifier)
            if cached_result is not None:
                logger.debug(f"Cache hit for processed data: {data_type}/{identifier}")
                return cached_result

            # Process the data
            logger.debug(f"Cache miss for processed data: {data_type}/{identifier}")
            result = func(*args, **kwargs)

            # Cache the result
            if result is not None:
                manager.cache_processed_data(data_type, identifier, result)

            return result

        return wrapper

    return decorator


class CachedSejmApiClient:
    """Wrapper for Sejm API client with caching capabilities."""

    def __init__(self, original_client, cache_manager: Optional[CacheManager] = None):
        self.client = original_client
        self.cache = cache_manager or get_cache_manager()
        logger.info("Sejm API client wrapped with caching")

    async def get_proceedings(self, **params) -> Any:
        """Get proceedings with caching."""
        endpoint = "proceedings"

        # Check cache first
        cached_result = self.cache.get_cached_api_response("sejm", endpoint, params)
        if cached_result is not None:
            return cached_result

        # Fetch from API
        result = await self.client.get_proceedings(**params)

        # Cache the result
        if result is not None:
            self.cache.cache_api_response("sejm", endpoint, params, result)

        return result

    async def get_print_details(self, print_id: str, **params) -> Any:
        """Get print details with caching."""
        endpoint = f"print/{print_id}"
        cache_params = {"print_id": print_id, **params}

        # Check cache first
        cached_result = self.cache.get_cached_api_response(
            "sejm", endpoint, cache_params
        )
        if cached_result is not None:
            return cached_result

        # Fetch from API
        result = await self.client.get_print_details(print_id, **params)

        # Cache the result
        if result is not None:
            self.cache.cache_api_response("sejm", endpoint, cache_params, result)

        return result

    async def get_term_meetings(self, term: int, **params) -> Any:
        """Get term meetings with caching."""
        endpoint = f"term/{term}/meetings"
        cache_params = {"term": term, **params}

        # Check cache first
        cached_result = self.cache.get_cached_api_response(
            "sejm", endpoint, cache_params
        )
        if cached_result is not None:
            return cached_result

        # Fetch from API
        result = await self.client.get_term_meetings(term, **params)

        # Cache the result
        if result is not None:
            self.cache.cache_api_response("sejm", endpoint, cache_params, result)

        return result

    async def get_documents_for_date(self, date: str, **params) -> Any:
        """Get documents for specific date with caching."""
        endpoint = f"documents/{date}"
        cache_params = {"date": date, **params}

        # Check cache first
        cached_result = self.cache.get_cached_api_response(
            "sejm", endpoint, cache_params
        )
        if cached_result is not None:
            return cached_result

        # Fetch from API
        result = await self.client.get_documents_for_date(date, **params)

        # Cache the result
        if result is not None:
            self.cache.cache_api_response("sejm", endpoint, cache_params, result)

        return result


class CachedEliApiClient:
    """Wrapper for ELI API client with caching capabilities."""

    def __init__(self, original_client, cache_manager: Optional[CacheManager] = None):
        self.client = original_client
        self.cache = cache_manager or get_cache_manager()
        logger.info("ELI API client wrapped with caching")

    async def search_documents(self, **params) -> Any:
        """Search documents with caching."""
        endpoint = "search"

        # Check cache first
        cached_result = self.cache.get_cached_api_response("eli", endpoint, params)
        if cached_result is not None:
            return cached_result

        # Fetch from API
        result = await self.client.search_documents(**params)

        # Cache the result
        if result is not None:
            self.cache.cache_api_response("eli", endpoint, params, result)

        return result

    async def get_document(self, document_id: str, **params) -> Any:
        """Get document details with caching."""
        endpoint = f"document/{document_id}"
        cache_params = {"document_id": document_id, **params}

        # Check cache first
        cached_result = self.cache.get_cached_api_response(
            "eli", endpoint, cache_params
        )
        if cached_result is not None:
            return cached_result

        # Fetch from API
        result = await self.client.get_document(document_id, **params)

        # Cache the result
        if result is not None:
            self.cache.cache_api_response("eli", endpoint, cache_params, result)

        return result


class CacheAwareEmbeddingProcessor:
    """Embedding processor with cache integration."""

    def __init__(
        self, original_processor, cache_manager: Optional[CacheManager] = None
    ):
        self.processor = original_processor
        self.cache = cache_manager or get_cache_manager()
        logger.info("Embedding processor wrapped with caching")

    async def process_document(self, document_id: str, text: str) -> list[float]:
        """Process document embeddings with caching."""
        # Check cache first
        cached_embeddings = self.cache.get_cached_embeddings(document_id)
        if cached_embeddings is not None:
            logger.info(f"Using cached embeddings for document {document_id}")
            return cached_embeddings

        # Generate embeddings
        logger.info(f"Generating new embeddings for document {document_id}")
        embeddings = await self.processor.process_document(document_id, text)

        # Cache the embeddings
        if embeddings:
            self.cache.cache_embeddings(document_id, embeddings)

        return embeddings


def create_cache_aware_clients(
    sejm_client=None,
    eli_client=None,
    embedding_processor=None,
    cache_manager: Optional[CacheManager] = None,
) -> Dict[str, Any]:
    """Create cache-aware versions of API clients and processors."""
    cache = cache_manager or get_cache_manager()

    result = {}

    if sejm_client:
        result["sejm_client"] = CachedSejmApiClient(sejm_client, cache)

    if eli_client:
        result["eli_client"] = CachedEliApiClient(eli_client, cache)

    if embedding_processor:
        result["embedding_processor"] = CacheAwareEmbeddingProcessor(
            embedding_processor, cache
        )

    return result
