"""Cache integration for API clients and data processing pipelines."""

import logging
from typing import Any, Dict, Optional, Callable, Awaitable, Union, Tuple
from functools import wraps

from .manager import CacheManager, get_cache_manager
from .document_cache import DocumentContentCache, get_document_cache
from .processed_text_cache import ProcessedTextCache, get_processed_text_cache

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


class CacheAwareDocumentProcessor:
    """Document processor with comprehensive caching for content and processed text."""

    def __init__(
        self,
        original_processor,
        document_cache: Optional[DocumentContentCache] = None,
        text_cache: Optional[ProcessedTextCache] = None,
    ):
        self.processor = original_processor
        self.document_cache = document_cache or get_document_cache()
        self.text_cache = text_cache or get_processed_text_cache()
        logger.info("Document processor wrapped with comprehensive caching")

    def cache_source_document(
        self,
        document_id: str,
        content: Union[str, bytes],
        content_type: str,
        source_url: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Cache the original source document (HTML/PDF)."""
        return self.document_cache.cache_document_content(
            document_id=document_id,
            content=content,
            content_type=content_type,
            source_url=source_url,
        )

    def get_cached_source_document(
        self, document_id: str, content_type: str
    ) -> Optional[Tuple[Union[str, bytes], Dict[str, Any]]]:
        """Get cached source document."""
        return self.document_cache.get_cached_document_content(
            document_id, content_type
        )

    def process_and_cache_text(
        self,
        document_id: str,
        stage: str,
        processing_params: Dict[str, Any],
        force_reprocess: bool = False,
    ) -> str:
        """Process document text and cache the result."""
        # Check if we already have processed text for these parameters
        if not force_reprocess:
            cached_result = self.text_cache.get_processed_text(
                document_id, stage, processing_params
            )
            if cached_result is not None:
                text, metadata = cached_result
                logger.info(f"Using cached processed text for {document_id}/{stage}")
                return text

        # Process the text (delegate to original processor)
        logger.info(f"Processing text for {document_id}/{stage}")
        import time

        start_time = time.time()

        processed_text = self.processor.process_text(
            document_id, stage, processing_params
        )

        processing_time_ms = (time.time() - start_time) * 1000

        # Cache the processed text
        self.text_cache.cache_processed_text(
            document_id=document_id,
            stage=stage,
            processed_text=processed_text,
            processing_params=processing_params,
            processing_time_ms=processing_time_ms,
        )

        return processed_text

    def get_processing_pipeline_status(self, document_id: str) -> Dict[str, Any]:
        """Get comprehensive status of document processing pipeline."""
        status = {
            "document_id": document_id,
            "source_content": {},
            "processed_text": {},
            "cache_info": {},
        }

        # Check source document cache
        doc_info = self.document_cache.get_document_cache_info(document_id)
        if doc_info:
            status["source_content"] = doc_info.get("content_versions", {})

        # Check processed text cache
        text_info = self.text_cache.get_processing_history(document_id)
        if text_info:
            status["processed_text"] = text_info.get("processing_stages", {})

        # Summary statistics
        status["cache_info"] = {
            "source_content_types": len(status["source_content"]),
            "processing_stages": len(status["processed_text"]),
            "total_cached_items": len(status["source_content"])
            + len(status["processed_text"]),
        }

        return status


def create_cache_aware_clients(
    sejm_client=None,
    eli_client=None,
    embedding_processor=None,
    document_processor=None,
    cache_manager: Optional[CacheManager] = None,
    document_cache: Optional[DocumentContentCache] = None,
    text_cache: Optional[ProcessedTextCache] = None,
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

    if document_processor:
        result["document_processor"] = CacheAwareDocumentProcessor(
            document_processor, document_cache, text_cache
        )

    return result


def get_comprehensive_cache_status() -> Dict[str, Any]:
    """Get status of all cache components."""
    try:
        # Get cache managers
        cache_manager = get_cache_manager()
        document_cache = get_document_cache()
        text_cache = get_processed_text_cache()

        status = {
            "timestamp": logging.Formatter().formatTime(
                logging.LogRecord("", 0, "", 0, "", (), None)
            ),
            "cache_components": {
                "api_cache": cache_manager.get_cache_stats(),
                "document_cache": document_cache.get_document_cache_info("summary")
                or {},
                "processed_text_cache": text_cache.get_cache_statistics(),
            },
        }

        return status

    except Exception as e:
        logger.error(f"Failed to get comprehensive cache status: {e}")
        return {
            "error": str(e),
            "timestamp": logging.Formatter().formatTime(
                logging.LogRecord("", 0, "", 0, "", (), None)
            ),
        }
