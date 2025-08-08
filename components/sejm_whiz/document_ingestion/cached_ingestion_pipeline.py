"""Cache-enhanced document ingestion pipeline."""

import asyncio
import logging
from typing import Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta, UTC
from enum import Enum

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
from .dual_stream_pipeline import DualApiDocumentProcessor

logger = logging.getLogger(__name__)


class StreamType(Enum):
    """Valid document stream types."""

    ELI = "eli"  # Enacted law documents (law in effect)
    SEJM = "sejm"  # Legislative work documents (law in development)


class ProcessingStatus(Enum):
    """Document processing status values."""

    PROCESSING = "processing"
    COMPLETED = "completed"  # Successfully extracted content and metadata, then stored, calculated and stored embeddings
    FAILED = "failed"  # Could not extract usable content or metadata
    EXCEPTION = "exception"  # Processing threw an exception


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
        self.dual_api_processor = DualApiDocumentProcessor(
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

    async def process_document_by_stream(
        self, document_id: str, stream_type: str
    ) -> Dict[str, Any]:
        """Process a document from a specific stream with caching.

        Args:
            document_id: Document identifier to process
            stream_type: "eli" (enacted law) or "sejm" (legislative work)

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
            # Check cache for stream-processed result
            cached_result = self.cache_manager.get_cached_api_response(
                stream_type,
                f"{stream_type}_stream_document_process",
                {"document_id": document_id, "stream": stream_type},
            )

            if cached_result:
                logger.info(
                    f"Using cached {stream_type} stream result for {document_id}"
                )
                result.update(cached_result)
                result["cache_hits"][f"{stream_type}_stream_result"] = True
                return result

            # Process using stream-specific processor
            logger.info(f"Processing {document_id} with {stream_type} stream processor")

            # Process document from specific stream using enum
            if stream_type == StreamType.SEJM.value:
                # Process legislative work-in-progress document
                stream_result = await self.dual_api_processor.process_sejm_document(
                    document_id
                )
            elif stream_type == StreamType.ELI.value:
                # Process enacted/historical law document
                stream_result = await self.dual_api_processor.process_eli_document(
                    document_id
                )
            else:
                # Should not reach here due to enum validation above
                raise ValueError(f"Invalid stream_type '{stream_type}'")

            if stream_result.success:
                result.update(
                    {
                        "status": "failed",  # Will be "completed" when embeddings are implemented
                        "act_text": stream_result.act_text,
                        "metadata": stream_result.metadata,
                        "source_used": stream_result.source_used,
                        "content_quality_score": stream_result.content_quality_score,
                        "text_length": len(stream_result.act_text),
                        "api_processing_time": stream_result.processing_time,
                        "content_extracted": bool(
                            stream_result.act_text and stream_result.act_text.strip()
                        ),
                        "metadata_extracted": bool(
                            stream_result.metadata and len(stream_result.metadata) > 0
                        ),
                        "embeddings_generated": False,  # Not yet implemented
                        "reason": "embeddings_not_generated",
                    }
                )

                # Cache the successful result
                cache_data = {
                    "status": "failed",  # Will be "completed" when embeddings are implemented
                    "act_text": stream_result.act_text,
                    "metadata": stream_result.metadata,
                    "source_used": stream_result.source_used,
                    "content_quality_score": stream_result.content_quality_score,
                    "text_length": len(stream_result.act_text),
                    "content_extracted": bool(
                        stream_result.act_text and stream_result.act_text.strip()
                    ),
                    "metadata_extracted": bool(
                        stream_result.metadata and len(stream_result.metadata) > 0
                    ),
                    "embeddings_generated": False,
                    "reason": "embeddings_not_generated",
                }
                self.cache_manager.cache_api_response(
                    stream_type,
                    f"{stream_type}_stream_document_process",
                    {"document_id": document_id, "stream": stream_type},
                    cache_data,
                )

                logger.info(
                    f"Successfully processed {document_id} via {stream_result.source_used}"
                )

            else:
                result.update(
                    {
                        "status": "failed",
                        "reason": stream_result.error_message
                        or f"{stream_type}_stream_processing_failed",
                        "source_used": stream_result.source_used,
                    }
                )
                logger.warning(
                    f"{stream_type.upper()} stream processing failed for {document_id}: {stream_result.error_message}"
                )

        except Exception as e:
            logger.error(
                f"{stream_type.upper()} stream document processing failed for {document_id}: {e}"
            )
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

    async def ingest_documents_by_stream(
        self,
        document_ids: list,
        stream_type: str,
    ) -> Dict[str, Any]:
        """Ingest documents from a specific stream (ELI or Sejm).

        Args:
            document_ids: List of document IDs to process
            stream_type: "eli" (enacted law) or "sejm" (legislative work)

        Returns:
            Ingestion statistics and results
        """
        logger.info(
            f"Starting {stream_type} stream ingestion of {len(document_ids)} documents"
        )

        # Validate stream type using enum
        try:
            StreamType(stream_type)
        except ValueError:
            valid_streams = [s.value for s in StreamType]
            raise ValueError(
                f"Invalid stream_type '{stream_type}'. Must be one of: {valid_streams}"
            )

        stats = {
            "start_time": datetime.now(UTC),
            "stream_type": stream_type,
            "documents_processed": 0,
            "documents_successful": 0,
            "documents_failed": 0,
            "documents_cached": 0,
            "stream_breakdown": {
                f"{stream_type}_successful": 0,
                f"{stream_type}_failed": 0,
                f"{stream_type}_cached": 0,
            },
            "results": [],
        }

        # Process documents with concurrency control
        semaphore = asyncio.Semaphore(self.config.parallel_workers)

        async def process_with_semaphore(doc_id):
            async with semaphore:
                return await self.process_document_by_stream(doc_id, stream_type)

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
                    stats["stream_breakdown"][f"{stream_type}_failed"] += 1
                    stats["results"].append(
                        {
                            "document_id": document_ids[i],
                            "status": "exception",
                            "error": str(result),
                        }
                    )
                elif isinstance(result, dict):
                    stats["results"].append(result)

                    # Check for partial success (content + metadata extracted, but no embeddings yet)
                    if (
                        result.get("content_extracted")
                        and result.get("metadata_extracted")
                        and result.get("reason") == "embeddings_not_generated"
                    ):
                        # This is partial success - content and metadata extracted successfully
                        stats["documents_successful"] += 1
                        stats["stream_breakdown"][f"{stream_type}_successful"] += 1

                        # Track cache hits
                        if result.get("cache_hits", {}).get(
                            f"{stream_type}_stream_result"
                        ):
                            stats["documents_cached"] += 1
                            stats["stream_breakdown"][f"{stream_type}_cached"] += 1

                    elif result.get("status") == "completed":
                        # True completion (content + metadata + embeddings)
                        stats["documents_successful"] += 1
                        stats["stream_breakdown"][f"{stream_type}_successful"] += 1

                        # Track cache hits
                        if result.get("cache_hits", {}).get(
                            f"{stream_type}_stream_result"
                        ):
                            stats["documents_cached"] += 1
                            stats["stream_breakdown"][f"{stream_type}_cached"] += 1

                    elif result.get("status") == "failed":
                        stats["documents_failed"] += 1
                        stats["stream_breakdown"][f"{stream_type}_failed"] += 1

        except Exception as e:
            logger.error(f"{stream_type.upper()} stream document ingestion failed: {e}")
            raise IngestionPipelineError(f"{stream_type} stream ingestion failed: {e}")

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
            f"{stream_type.upper()} stream ingestion completed: {stats['documents_successful']}/{stats['documents_processed']} successful"
        )
        logger.info(f"Stream breakdown: {stats['stream_breakdown']}")

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
