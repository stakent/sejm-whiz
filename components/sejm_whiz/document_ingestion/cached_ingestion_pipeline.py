"""Cache-enhanced document ingestion pipeline."""

import asyncio
import logging
from typing import Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta, UTC

from sejm_whiz.cache import (
    CacheAwareDocumentProcessor,
    get_cache_manager,
    get_document_cache,
    get_processed_text_cache,
)

from .config import DocumentIngestionConfig
from sejm_whiz.eli_api.client import EliApiClient as ELIClient
from sejm_whiz.sejm_api.client import SejmApiClient
from .ingestion_pipeline import DocumentIngestionPipeline, IngestionPipelineError
from .multi_api_pipeline import MultiApiDocumentProcessor

logger = logging.getLogger(__name__)


class CachedDocumentIngestionPipeline(DocumentIngestionPipeline):
    """Document ingestion pipeline enhanced with filesystem caching."""

    def __init__(self, config: Optional[DocumentIngestionConfig] = None):
        super().__init__(config)

        # Initialize cache managers
        self.cache_manager = get_cache_manager()
        self.document_cache = get_document_cache()
        self.text_cache = get_processed_text_cache()

        # Create cache-aware text processor
        self.cached_processor = CacheAwareDocumentProcessor(
            original_processor=self.text_processor,
            document_cache=self.document_cache,
            text_cache=self.text_cache,
        )

        # Initialize multi-API processor with clients
        self.eli_client = ELIClient(config)
        self.sejm_client = SejmApiClient()
        self.multi_api_processor = MultiApiDocumentProcessor(
            sejm_client=self.sejm_client,
            eli_client=self.eli_client,
            content_validator=self.eli_client.content_validator,
        )

        logger.info(
            "Cached document ingestion pipeline initialized with multi-API support"
        )

    async def _fetch_document_with_cache(
        self, eli_client: ELIClient, document_id: str, document_url: str
    ) -> Optional[Tuple[Union[str, bytes], str, Dict[str, Any]]]:
        """Fetch document content with caching support."""

        # Check if we have cached document content
        for content_type in ["html", "pdf"]:
            cached_result = self.document_cache.get_cached_document_content(
                document_id, content_type
            )
            if cached_result:
                content, metadata = cached_result
                logger.info(f"Using cached {content_type} for document {document_id}")
                return content, content_type, metadata

        # Fetch from ELI API with cache for API response
        cached_api_response = self.cache_manager.get_cached_api_response(
            "eli", "document_fetch", {"document_id": document_id}
        )

        if cached_api_response:
            logger.info(f"Using cached API response for document {document_id}")
            content = cached_api_response.get("content")
            content_type = cached_api_response.get("content_type", "html")
            metadata = cached_api_response.get("metadata", {})
        else:
            # Fetch from API
            logger.info(f"Fetching document {document_id} from ELI API")
            try:
                result = await eli_client.fetch_document_content(
                    document_id, document_url
                )
                if not result:
                    return None

                content, content_type, metadata = result

                # Cache the API response
                api_response = {
                    "content": content,
                    "content_type": content_type,
                    "metadata": metadata,
                }
                self.cache_manager.cache_api_response(
                    "eli", "document_fetch", {"document_id": document_id}, api_response
                )

            except Exception as e:
                logger.error(f"Failed to fetch document {document_id}: {e}")
                return None

        # Cache the document content
        if content:
            try:
                self.document_cache.cache_document_content(
                    document_id=document_id,
                    content=content,
                    content_type=content_type,
                    source_url=document_url,
                    additional_metadata=metadata,
                )
                logger.debug(
                    f"Cached {content_type} content for document {document_id}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to cache document content for {document_id}: {e}"
                )

        return content, content_type, metadata

    async def _process_document_text_with_cache(
        self, document_id: str, content: Union[str, bytes], content_type: str
    ) -> Optional[str]:
        """Process document text with caching using TextProcessor."""

        # Define a single processing stage that uses the TextProcessor
        stage = "processed"
        params = {"content_type": content_type, "processor": "TextProcessor"}

        # Check if we have cached processed text
        cached_result = self.text_cache.get_processed_text(document_id, stage, params)

        if cached_result:
            processed_text, stage_metadata = cached_result
            logger.info(f"Using cached processed text for document {document_id}")
            return processed_text
        else:
            # Process text using TextProcessor
            logger.info(f"Processing text for document {document_id}")

            try:
                import time

                start_time = time.time()

                # Use the existing TextProcessor interface
                processed_text = self._process_content_with_text_processor(
                    content, content_type, document_id
                )

                processing_time_ms = (time.time() - start_time) * 1000

                # Cache the processed text
                if processed_text:
                    self.text_cache.cache_processed_text(
                        document_id=document_id,
                        stage=stage,
                        processed_text=processed_text,
                        processing_params=params,
                        processing_time_ms=processing_time_ms,
                    )

                return processed_text

            except Exception as e:
                logger.error(f"Failed to process text for document {document_id}: {e}")
                return None

    def _process_content_with_text_processor(
        self, content: Union[str, bytes], content_type: str, document_id: str
    ) -> str:
        """Process content using the existing TextProcessor interface."""
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")

        # Use the existing TextProcessor.process_document method
        processed_doc = self.text_processor.process_document(
            raw_content=content, eli_id=document_id, source_url=""
        )

        return processed_doc.content

    async def process_document_multi_api(
        self, document_id: str, preferred_source: str = "auto"
    ) -> Dict[str, Any]:
        """Process a document using multi-API approach with caching.

        Args:
            document_id: Document identifier to process
            preferred_source: "sejm", "eli", or "auto" for automatic selection

        Returns:
            Processing result with content and metadata
        """
        result = {
            "document_id": document_id,
            "status": "processing",
            "source_used": "none",
            "processing_time_ms": 0,
            "cache_hits": {},
        }

        start_time = asyncio.get_event_loop().time()

        try:
            # Check cache for multi-API processed result
            cached_result = self.cache_manager.get_cached_api_response(
                "multi_api",
                "document_process",
                {"document_id": document_id, "source": preferred_source},
            )

            if cached_result:
                logger.info(f"Using cached multi-API result for {document_id}")
                result.update(cached_result)
                result["cache_hits"]["multi_api_result"] = True
                return result

            # Process using multi-API processor
            logger.info(
                f"Processing {document_id} with multi-API processor (preferred: {preferred_source})"
            )

            # Choose processing method based on preferred source
            if preferred_source == "sejm":
                # Try Sejm API first
                multi_result = await self.multi_api_processor._try_sejm_api(document_id)
            elif preferred_source == "eli":
                # Try ELI API first
                multi_result = (
                    await self.multi_api_processor._try_eli_api_with_fallback(
                        document_id
                    )
                )
            else:
                # Use automatic source selection
                multi_result = (
                    await self.multi_api_processor.process_document_from_any_source(
                        document_id
                    )
                )

            if multi_result.success:
                result.update(
                    {
                        "status": "completed",
                        "act_text": multi_result.act_text,
                        "metadata": multi_result.metadata,
                        "source_used": multi_result.source_used,
                        "content_quality_score": multi_result.content_quality_score,
                        "text_length": len(multi_result.act_text),
                        "api_processing_time": multi_result.processing_time,
                    }
                )

                # Cache the successful result
                cache_data = {
                    "status": "completed",
                    "act_text": multi_result.act_text,
                    "metadata": multi_result.metadata,
                    "source_used": multi_result.source_used,
                    "content_quality_score": multi_result.content_quality_score,
                    "text_length": len(multi_result.act_text),
                }
                self.cache_manager.cache_api_response(
                    "multi_api",
                    "document_process",
                    {"document_id": document_id, "source": preferred_source},
                    cache_data,
                )

                logger.info(
                    f"Successfully processed {document_id} via {multi_result.source_used}"
                )

            else:
                result.update(
                    {
                        "status": "failed",
                        "reason": multi_result.error_message
                        or "multi_api_processing_failed",
                        "source_used": multi_result.source_used,
                    }
                )
                logger.warning(
                    f"Multi-API processing failed for {document_id}: {multi_result.error_message}"
                )

        except Exception as e:
            logger.error(f"Multi-API document processing failed for {document_id}: {e}")
            result.update(
                {
                    "status": "failed",
                    "reason": str(e),
                    "error_type": "multi_api_exception",
                }
            )

        finally:
            end_time = asyncio.get_event_loop().time()
            result["processing_time_ms"] = (end_time - start_time) * 1000

        return result

    async def ingest_documents_multi_source(
        self,
        document_ids: list,
        preferred_source: str = "auto",
        enable_fallback: bool = True,
    ) -> Dict[str, Any]:
        """Ingest documents from multiple sources with fallback support.

        Args:
            document_ids: List of document IDs to process
            preferred_source: "sejm", "eli", or "auto" for source preference
            enable_fallback: Whether to enable cross-API fallback

        Returns:
            Ingestion statistics and results
        """
        logger.info(f"Starting multi-source ingestion of {len(document_ids)} documents")
        logger.info(
            f"Preferred source: {preferred_source}, Fallback enabled: {enable_fallback}"
        )

        stats = {
            "start_time": datetime.now(UTC),
            "documents_processed": 0,
            "documents_successful": 0,
            "documents_failed": 0,
            "documents_cached": 0,
            "source_breakdown": {
                "sejm_api": 0,
                "eli_api_html": 0,
                "eli_api_pdf": 0,
                "failed": 0,
            },
            "results": [],
        }

        # Process documents with concurrency control
        semaphore = asyncio.Semaphore(self.config.parallel_workers)

        async def process_with_semaphore(doc_id):
            async with semaphore:
                return await self.process_document_multi_api(doc_id, preferred_source)

        try:
            # Process all documents
            tasks = [process_with_semaphore(doc_id) for doc_id in document_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect statistics
            for i, result in enumerate(results):
                stats["documents_processed"] += 1

                if isinstance(result, Exception):
                    logger.error(
                        f"Document {document_ids[i]} processing exception: {result}"
                    )
                    stats["documents_failed"] += 1
                    stats["source_breakdown"]["failed"] += 1
                    stats["results"].append(
                        {
                            "document_id": document_ids[i],
                            "status": "exception",
                            "error": str(result),
                        }
                    )
                elif isinstance(result, dict):
                    stats["results"].append(result)

                    if result.get("status") == "completed":
                        stats["documents_successful"] += 1

                        # Track source usage
                        source_used = result.get("source_used", "unknown")
                        if "sejm" in source_used:
                            stats["source_breakdown"]["sejm_api"] += 1
                        elif "eli_api_html" in source_used:
                            stats["source_breakdown"]["eli_api_html"] += 1
                        elif "eli_api_pdf" in source_used:
                            stats["source_breakdown"]["eli_api_pdf"] += 1

                        # Track cache hits
                        if result.get("cache_hits", {}).get("multi_api_result"):
                            stats["documents_cached"] += 1

                    elif result.get("status") == "failed":
                        stats["documents_failed"] += 1
                        stats["source_breakdown"]["failed"] += 1

        except Exception as e:
            logger.error(f"Multi-source document ingestion failed: {e}")
            raise IngestionPipelineError(f"Multi-source ingestion failed: {e}")

        finally:
            stats["end_time"] = datetime.now(UTC)
            stats["total_time_seconds"] = (
                stats["end_time"] - stats["start_time"]
            ).total_seconds()

        # Calculate success rate
        if stats["documents_processed"] > 0:
            stats["success_rate"] = (
                stats["documents_successful"] / stats["documents_processed"]
            )
        else:
            stats["success_rate"] = 0.0

        logger.info(
            f"Multi-source ingestion completed: {stats['documents_successful']}/{stats['documents_processed']} successful"
        )
        logger.info(f"Source breakdown: {stats['source_breakdown']}")

        return stats

    async def process_document(
        self, document_metadata: Dict[str, Any], eli_client: ELIClient
    ) -> Dict[str, Any]:
        """Process a single document with comprehensive caching."""

        document_id = (
            document_metadata.get("eli_id")
            or document_metadata.get("id")
            or document_metadata.get("identifier", "")
        )
        if not document_id:
            logger.warning("Document missing ID, skipping")
            return {"status": "skipped", "reason": "missing_id"}

        document_url = document_metadata.get("url", "")

        result = {
            "document_id": document_id,
            "status": "processing",
            "cache_hits": {},
            "processing_time_ms": 0,
        }

        start_time = asyncio.get_event_loop().time()

        try:
            # Step 1: Fetch document content (with caching)
            logger.info(f"Processing document {document_id}")

            content_result = await self._fetch_document_with_cache(
                eli_client, document_id, document_url
            )

            if not content_result:
                result.update({"status": "failed", "reason": "fetch_failed"})
                return result

            content, content_type, content_metadata = content_result
            result["content_type"] = content_type

            # Step 2: Process document text (with stage-wise caching)
            processed_text = await self._process_document_text_with_cache(
                document_id, content, content_type
            )

            if not processed_text:
                result.update({"status": "failed", "reason": "processing_failed"})
                return result

            # Step 3: Store in database (check for duplicates first)
            existing_doc = await self.db_operations.get_document_by_eli_id(document_id)
            if existing_doc and not self.config.backup_processed_documents:
                logger.info(
                    f"Document {document_id} already exists in database, skipping"
                )
                result.update({"status": "skipped", "reason": "already_exists"})
                return result

            # Prepare document data for database
            doc_data = {
                "eli_identifier": document_id,
                "title": document_metadata.get("title", ""),
                "content_type": content_type,
                "source_url": document_url,
                "raw_content": content
                if isinstance(content, str)
                else content.decode("utf-8", errors="ignore"),
                "processed_text": processed_text,
                "metadata": {
                    **document_metadata,
                    **content_metadata,
                    "processing_pipeline": "cached_ingestion",
                    "cache_enabled": True,
                },
            }

            # Store in database
            stored_doc = await self.db_operations.store_legal_document(**doc_data)

            if stored_doc:
                result.update(
                    {
                        "status": "completed",
                        "database_id": stored_doc.get("id"),
                        "text_length": len(processed_text),
                    }
                )
                self.stats["documents_stored"] += 1
            else:
                result.update({"status": "failed", "reason": "database_storage_failed"})
                self.stats["documents_failed"] += 1

        except Exception as e:
            logger.error(f"Failed to process document {document_id}: {e}")
            result.update({"status": "failed", "reason": str(e)})
            self.stats["documents_failed"] += 1

        finally:
            end_time = asyncio.get_event_loop().time()
            result["processing_time_ms"] = (end_time - start_time) * 1000
            self.stats["documents_processed"] += 1

        return result

    async def ingest_recent_documents(
        self, days: int = 7, limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Ingest recent documents with caching."""
        limit_msg = f" (limit: {limit})" if limit else ""
        logger.info(
            f"Starting cached document ingestion for last {days} days{limit_msg}"
        )

        self.stats["start_time"] = datetime.now(UTC)

        # Use cached ELI client
        eli_client = ELIClient()

        try:
            # Search for recent documents (with API caching)
            end_date = datetime.now(UTC)
            start_date = end_date - timedelta(days=days)

            search_params = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "limit": 100,
            }

            # Check cache for search results
            cached_search = self.cache_manager.get_cached_api_response(
                "eli", "document_search", search_params
            )

            if cached_search:
                logger.info(f"Using cached search results for {days} days")
                documents = cached_search.get("documents", [])
            else:
                logger.info(f"Searching ELI API for documents from last {days} days")
                documents = await eli_client.get_recent_documents(days=days)

                # Cache the search results
                search_result = {"documents": documents}
                self.cache_manager.cache_api_response(
                    "eli", "document_search", search_params, search_result
                )

            logger.info(f"Found {len(documents)} documents to process")

            # Apply limit if specified
            if limit and len(documents) > limit:
                documents = documents[:limit]
                logger.info(f"Limited to {limit} documents for processing")

            # Process documents with concurrency control
            semaphore = asyncio.Semaphore(self.config.parallel_workers)

            async def process_with_semaphore(doc_metadata):
                async with semaphore:
                    return await self.process_document(doc_metadata, eli_client)

            # Process all documents
            tasks = [process_with_semaphore(doc) for doc in documents]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect statistics
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Document processing exception: {result}")
                    self.stats["documents_failed"] += 1
                elif isinstance(result, dict):
                    if result.get("status") == "completed":
                        self.stats["documents_stored"] += 1
                    elif result.get("status") == "skipped":
                        self.stats["documents_skipped"] += 1
                    elif result.get("status") == "failed":
                        self.stats["documents_failed"] += 1

        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            raise IngestionPipelineError(f"Ingestion failed: {e}")

        finally:
            self.stats["end_time"] = datetime.now(UTC)

        logger.info(f"Cached document ingestion completed: {self.stats}")
        return self.stats.copy()

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            return {
                "api_cache": self.cache_manager.get_cache_stats(),
                "document_cache": self.document_cache.get_document_cache_info("summary")
                or {"total_documents": 0},
                "text_cache": self.text_cache.get_cache_statistics(),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            logger.error(f"Failed to get cache statistics: {e}")
            return {"error": str(e), "timestamp": datetime.now(UTC).isoformat()}

    async def cleanup_cache(self, dry_run: bool = False) -> Dict[str, Any]:
        """Clean up expired cache entries."""
        from sejm_whiz.cache import get_maintenance_manager

        maintenance_manager = get_maintenance_manager()
        return await maintenance_manager.cleanup_expired_files(dry_run=dry_run)
