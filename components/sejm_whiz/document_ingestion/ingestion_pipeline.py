"""Document ingestion pipeline orchestrating ELI API, processing, and database storage."""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, UTC
from dataclasses import asdict

from sejm_whiz.database import DocumentOperations, get_database_config
from sejm_whiz.redis import get_redis_cache, get_redis_queue, JobPriority

from .config import get_ingestion_config, DocumentIngestionConfig
from .eli_client import ELIClient
from .text_processor import TextProcessor

logger = logging.getLogger(__name__)


class IngestionPipelineError(Exception):
    """Ingestion pipeline specific error."""

    pass


class DocumentIngestionPipeline:
    """Complete document ingestion pipeline from ELI API to database."""

    def __init__(self, config: Optional[DocumentIngestionConfig] = None):
        self.config = config or get_ingestion_config()
        self.text_processor = TextProcessor(config)
        self.cache = get_redis_cache()
        self.job_queue = get_redis_queue()

        # Initialize database operations
        db_config = get_database_config()
        self.db_operations = DocumentOperations(db_config)

        # Statistics
        self.stats = {
            "documents_processed": 0,
            "documents_stored": 0,
            "documents_skipped": 0,
            "documents_failed": 0,
            "start_time": None,
            "end_time": None,
        }

    async def ingest_recent_documents(self, days: int = 7) -> Dict[str, Any]:
        """Ingest recently published documents."""

        logger.info(f"Starting ingestion of documents from last {days} days")
        self.stats["start_time"] = datetime.now(UTC)

        try:
            # Get ELI client
            async with ELIClient(self.config) as eli_client:
                # Fetch recent documents
                recent_docs = await eli_client.get_recent_documents(
                    days=days, document_types=self.config.legal_document_types
                )

                logger.info(f"Found {len(recent_docs)} recent documents to process")

                # Process documents in batches
                results = await self._process_documents_batch(eli_client, recent_docs)

        except Exception as e:
            logger.error(f"Recent document ingestion failed: {e}")
            raise IngestionPipelineError(f"Ingestion failed: {e}")

        finally:
            self.stats["end_time"] = datetime.now(UTC)

        return self._compile_results(results)

    async def ingest_document_by_eli(self, eli_id: str) -> Dict[str, Any]:
        """Ingest specific document by ELI identifier."""

        logger.info(f"Ingesting document: {eli_id}")

        try:
            # Check cache first
            cached_doc = self.cache.get_document(eli_id)
            if cached_doc:
                logger.info(f"Document {eli_id} found in cache")
                return {"status": "cached", "document": cached_doc}

            # Check if already in database
            existing_doc = await self.db_operations.get_document_by_eli(eli_id)
            if existing_doc:
                logger.info(f"Document {eli_id} already exists in database")
                # Cache for future requests
                self.cache.cache_document(eli_id, asdict(existing_doc))
                return {"status": "exists", "document": asdict(existing_doc)}

            # Fetch and process document
            async with ELIClient(self.config) as eli_client:
                eli_doc = await eli_client.get_document(eli_id)

                if not eli_doc:
                    logger.warning(f"Document {eli_id} not found in ELI API")
                    return {"status": "not_found", "eli_id": eli_id}

                # Get document content
                content = await eli_client.get_document_content(eli_id)

                # Process document
                processed_doc = self.text_processor.process_document(
                    content, eli_id, eli_doc.get("source_url")
                )

                # Validate document
                is_valid, errors = self.text_processor.validate_document(processed_doc)
                if not is_valid:
                    logger.warning(f"Document {eli_id} validation failed: {errors}")
                    return {"status": "invalid", "errors": errors}

                # Store in database
                stored_doc = await self.db_operations.create_document(processed_doc)

                # Cache document
                self.cache.cache_document(eli_id, asdict(stored_doc))

                logger.info(f"Successfully ingested document: {eli_id}")
                return {"status": "ingested", "document": asdict(stored_doc)}

        except Exception as e:
            logger.error(f"Failed to ingest document {eli_id}: {e}")
            return {"status": "error", "error": str(e)}

    async def ingest_documents_batch(self, eli_ids: List[str]) -> List[Dict[str, Any]]:
        """Ingest multiple documents in batch."""

        logger.info(f"Batch ingesting {len(eli_ids)} documents")

        results = []

        # Process in smaller batches to avoid overwhelming the API
        batch_size = min(self.config.batch_size, len(eli_ids))

        for i in range(0, len(eli_ids), batch_size):
            batch = eli_ids[i : i + batch_size]

            # Process batch
            batch_results = await asyncio.gather(
                *[self.ingest_document_by_eli(eli_id) for eli_id in batch],
                return_exceptions=True,
            )

            # Handle results and exceptions
            for eli_id, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Exception processing {eli_id}: {result}")
                    results.append(
                        {"status": "error", "eli_id": eli_id, "error": str(result)}
                    )
                else:
                    results.append(result)

            # Rate limiting between batches
            await asyncio.sleep(1.0)

        return results

    async def _process_documents_batch(
        self, eli_client: ELIClient, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process a batch of documents from ELI API."""

        results = []

        for i in range(0, len(documents), self.config.batch_size):
            batch = documents[i : i + self.config.batch_size]

            logger.info(
                f"Processing batch {i // self.config.batch_size + 1} ({len(batch)} documents)"
            )

            # Process documents in parallel
            if self.config.parallel_workers > 1:
                semaphore = asyncio.Semaphore(self.config.parallel_workers)
                tasks = [
                    self._process_single_document_with_semaphore(
                        eli_client, doc, semaphore
                    )
                    for doc in batch
                ]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                batch_results = []
                for doc in batch:
                    result = await self._process_single_document(eli_client, doc)
                    batch_results.append(result)

            # Handle results
            for doc, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Exception processing document {doc.get('eli_id')}: {result}"
                    )
                    result = {"status": "error", "error": str(result)}

                results.append(result)
                self._update_stats(result)

            # Brief pause between batches
            await asyncio.sleep(0.5)

        return results

    async def _process_single_document_with_semaphore(
        self, eli_client: ELIClient, doc: Dict[str, Any], semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        """Process single document with semaphore for concurrency control."""
        async with semaphore:
            return await self._process_single_document(eli_client, doc)

    async def _process_single_document(
        self, eli_client: ELIClient, doc: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process single document from ELI API response."""

        eli_id = doc.get("eli_id")
        if not eli_id:
            return {"status": "error", "error": "Missing ELI ID"}

        try:
            # Check if already processed
            existing_doc = await self.db_operations.get_document_by_eli(eli_id)
            if existing_doc:
                return {"status": "exists", "eli_id": eli_id}

            # Get document content
            content = await eli_client.get_document_content(eli_id)
            if not content:
                return {"status": "no_content", "eli_id": eli_id}

            # Process document
            processed_doc = self.text_processor.process_document(
                content, eli_id, doc.get("source_url")
            )

            # Merge additional metadata from ELI API
            if doc.get("published_date"):
                processed_doc.published_at = datetime.fromisoformat(
                    doc["published_date"]
                )

            processed_doc.metadata.update(
                {
                    "eli_metadata": doc,
                    "ingestion_timestamp": datetime.now(UTC).isoformat(),
                }
            )

            # Validate document
            is_valid, errors = self.text_processor.validate_document(processed_doc)
            if not is_valid:
                return {"status": "invalid", "eli_id": eli_id, "errors": errors}

            # Store in database
            stored_doc = await self.db_operations.create_document(processed_doc)

            # Cache document
            self.cache.cache_document(eli_id, asdict(stored_doc))

            return {
                "status": "ingested",
                "eli_id": eli_id,
                "document_id": str(stored_doc.id),
            }

        except Exception as e:
            logger.error(f"Failed to process document {eli_id}: {e}")
            return {"status": "error", "eli_id": eli_id, "error": str(e)}

    def _update_stats(self, result: Dict[str, Any]) -> None:
        """Update processing statistics."""

        status = result.get("status")

        if status == "ingested":
            self.stats["documents_processed"] += 1
            self.stats["documents_stored"] += 1
        elif status == "exists":
            self.stats["documents_skipped"] += 1
        elif status in ["error", "invalid", "no_content"]:
            self.stats["documents_failed"] += 1

        self.stats["documents_processed"] += 1

    def _compile_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compile final ingestion results."""

        # Calculate duration
        duration = None
        if self.stats["start_time"] and self.stats["end_time"]:
            duration = (
                self.stats["end_time"] - self.stats["start_time"]
            ).total_seconds()

        # Group results by status
        status_counts = {}
        for result in results:
            status = result.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "summary": {
                "total_documents": len(results),
                "duration_seconds": duration,
                "documents_per_second": len(results) / duration if duration else 0,
                **self.stats,
            },
            "status_breakdown": status_counts,
            "results": results,
        }

    # Job queue integration
    async def enqueue_ingestion_job(
        self, eli_ids: List[str], priority: JobPriority = JobPriority.NORMAL
    ) -> str:
        """Enqueue document ingestion as background job."""

        job_id = self.job_queue.enqueue(
            task_name="document_ingestion",
            args=[eli_ids],
            priority=priority,
            max_retries=2,
        )

        logger.info(f"Enqueued ingestion job {job_id} for {len(eli_ids)} documents")
        return job_id

    async def process_ingestion_job(self, eli_ids: List[str]) -> Dict[str, Any]:
        """Process ingestion job (called by job worker)."""

        logger.info(f"Processing ingestion job for {len(eli_ids)} documents")

        try:
            results = await self.ingest_documents_batch(eli_ids)

            logger.info(f"Completed ingestion job: {len(results)} documents processed")
            return self._compile_results(results)

        except Exception as e:
            logger.error(f"Ingestion job failed: {e}")
            raise

    # Monitoring and management
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get current ingestion statistics."""
        return dict(self.stats)

    async def cleanup_failed_ingestions(self, older_than_hours: int = 24) -> int:
        """Clean up failed ingestion attempts."""

        # This would clean up any temporary data or failed attempts
        # TODO: Implement time-based filtering for items older than specified hours
        cleared_count = self.cache.clear_pattern("ingestion:failed:*")

        logger.info(f"Cleaned up {cleared_count} failed ingestion records")
        return cleared_count


# Global pipeline instance
_ingestion_pipeline: Optional[DocumentIngestionPipeline] = None


def get_ingestion_pipeline(
    config: Optional[DocumentIngestionConfig] = None,
) -> DocumentIngestionPipeline:
    """Get global document ingestion pipeline instance."""
    global _ingestion_pipeline

    if _ingestion_pipeline is None:
        _ingestion_pipeline = DocumentIngestionPipeline(config)

    return _ingestion_pipeline
