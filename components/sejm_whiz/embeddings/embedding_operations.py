"""High-level embedding operations integrating with database and cache."""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from sejm_whiz.database import DocumentOperations, get_database_config
from sejm_whiz.redis import get_redis_cache, get_redis_queue, JobPriority

from .config import get_embedding_config, EmbeddingConfig
from .herbert_embedder import get_herbert_embedder, EmbeddingResult

logger = logging.getLogger(__name__)


class EmbeddingOperations:
    """High-level embedding operations for legal documents."""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or get_embedding_config()
        self.embedder = get_herbert_embedder(config)
        self.cache = get_redis_cache()
        self.job_queue = get_redis_queue()

        # Initialize database operations
        db_config = get_database_config()
        self.db_operations = DocumentOperations(db_config)

    async def generate_document_embedding(
        self, document_id: str
    ) -> Optional[EmbeddingResult]:
        """Generate and store embedding for a document."""

        logger.info(f"Generating embedding for document: {document_id}")

        try:
            # Check cache first
            if self.config.cache_embeddings:
                cached_embedding = self.cache.get_document_embedding(document_id)
                if cached_embedding is not None:
                    logger.info(f"Found cached embedding for document: {document_id}")
                    return EmbeddingResult(
                        embedding=np.array(cached_embedding),
                        model_name=self.config.model_name,
                        processing_time=0.0,
                        token_count=0,
                        metadata={"source": "cache"},
                    )

            # Get document from database
            document = await self.db_operations.get_document_by_id(document_id)
            if not document:
                logger.warning(f"Document not found: {document_id}")
                return None

            # Generate embedding
            embedding_result = self.embedder.embed_legal_document(
                title=document.title,
                content=document.content,
                document_type=document.document_type,
            )

            # Store embedding in database
            await self.db_operations.store_document_embedding(
                document_id=document_id,
                embedding=embedding_result.embedding.tolist(),
                model_name=embedding_result.model_name,
                model_version="1.0",
                embedding_method=self.config.pooling_strategy,
                token_count=embedding_result.token_count,
                confidence_score=int(embedding_result.quality_score * 100),
            )

            # Update document with embedding
            await self.db_operations.update_document_embedding(
                document_id, embedding_result.embedding.tolist()
            )

            # Cache embedding
            if self.config.cache_embeddings:
                self.cache.cache_document_embedding(
                    document_id,
                    embedding_result.embedding.tolist(),
                    self.config.cache_ttl,
                )

            logger.info(f"Generated and stored embedding for document: {document_id}")
            return embedding_result

        except Exception as e:
            logger.error(
                f"Failed to generate embedding for document {document_id}: {e}"
            )
            return None

    async def generate_embeddings_batch(
        self, document_ids: List[str]
    ) -> Dict[str, Optional[EmbeddingResult]]:
        """Generate embeddings for multiple documents."""

        logger.info(f"Generating embeddings for {len(document_ids)} documents")

        results = {}

        # Process in smaller batches
        batch_size = min(self.config.batch_size, len(document_ids))

        for i in range(0, len(document_ids), batch_size):
            batch_ids = document_ids[i : i + batch_size]

            # Process batch concurrently
            batch_tasks = [
                self.generate_document_embedding(doc_id) for doc_id in batch_ids
            ]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle results
            for doc_id, result in zip(batch_ids, batch_results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Exception generating embedding for {doc_id}: {result}"
                    )
                    results[doc_id] = None
                else:
                    results[doc_id] = result

        success_count = sum(1 for r in results.values() if r is not None)
        logger.info(
            f"Generated {success_count}/{len(document_ids)} embeddings successfully"
        )

        return results

    async def find_similar_documents(
        self, query_text: str, limit: int = 10, threshold: float = None
    ) -> List[Dict[str, Any]]:
        """Find documents similar to query text."""

        logger.info(f"Finding similar documents for query (limit: {limit})")

        # Generate query embedding
        query_result = self.embedder.embed_text(query_text)
        query_embedding = query_result.embedding

        # Use threshold from config if not provided
        threshold = threshold or self.config.similarity_threshold

        # Search in database
        similar_docs = await self.db_operations.find_similar_documents(
            query_embedding=query_embedding.tolist(), limit=limit, threshold=threshold
        )

        # Enrich results with similarity scores
        enriched_results = []
        for doc, distance in similar_docs:
            similarity = 1.0 - distance  # Convert distance to similarity

            enriched_results.append(
                {
                    "document": {
                        "id": str(doc.id),
                        "title": doc.title,
                        "document_type": doc.document_type,
                        "legal_act_type": doc.legal_act_type,
                        "legal_domain": doc.legal_domain,
                        "published_at": doc.published_at.isoformat()
                        if doc.published_at
                        else None,
                        "eli_identifier": doc.eli_identifier,
                    },
                    "similarity_score": round(similarity, 4),
                    "distance": round(distance, 4),
                }
            )

        logger.info(f"Found {len(enriched_results)} similar documents")
        return enriched_results

    async def find_similar_by_document(
        self, document_id: str, limit: int = 10, threshold: float = None
    ) -> List[Dict[str, Any]]:
        """Find documents similar to a specific document."""

        logger.info(f"Finding documents similar to: {document_id}")

        # Get source document
        source_doc = await self.db_operations.get_document_by_id(document_id)
        if not source_doc:
            logger.warning(f"Source document not found: {document_id}")
            return []

        # Use existing embedding or generate new one
        if source_doc.embedding:
            query_embedding = source_doc.embedding
        else:
            # Generate embedding for source document
            embedding_result = await self.generate_document_embedding(document_id)
            if not embedding_result:
                logger.error(
                    f"Failed to generate embedding for source document: {document_id}"
                )
                return []
            query_embedding = embedding_result.embedding.tolist()

        # Use threshold from config if not provided
        threshold = threshold or self.config.similarity_threshold

        # Search in database (exclude source document)
        similar_docs = await self.db_operations.find_similar_by_document_id(
            document_id=document_id,
            limit=limit + 1,  # Get one extra to exclude source
            threshold=threshold,
        )

        # Filter out source document and limit results
        filtered_docs = [
            (doc, distance)
            for doc, distance in similar_docs
            if str(doc.id) != document_id
        ][:limit]

        # Enrich results
        enriched_results = []
        for doc, distance in filtered_docs:
            similarity = 1.0 - distance

            enriched_results.append(
                {
                    "document": {
                        "id": str(doc.id),
                        "title": doc.title,
                        "document_type": doc.document_type,
                        "legal_act_type": doc.legal_act_type,
                        "legal_domain": doc.legal_domain,
                        "published_at": doc.published_at.isoformat()
                        if doc.published_at
                        else None,
                        "eli_identifier": doc.eli_identifier,
                    },
                    "similarity_score": round(similarity, 4),
                    "distance": round(distance, 4),
                }
            )

        logger.info(f"Found {len(enriched_results)} similar documents to {document_id}")
        return enriched_results

    async def batch_similarity_search(
        self, queries: List[str], limit: int = 10
    ) -> List[List[Dict[str, Any]]]:
        """Perform similarity search for multiple queries."""

        logger.info(f"Performing batch similarity search for {len(queries)} queries")

        # Generate embeddings for all queries
        query_results = self.embedder.embed_texts(queries)

        # Perform searches concurrently
        search_tasks = [
            self.db_operations.find_similar_documents(
                query_embedding=result.embedding.tolist(),
                limit=limit,
                threshold=self.config.similarity_threshold,
            )
            for result in query_results
        ]

        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Process results
        final_results = []
        for query, search_result in zip(queries, search_results):
            if isinstance(search_result, Exception):
                logger.error(
                    f"Search failed for query '{query[:50]}...': {search_result}"
                )
                final_results.append([])
            else:
                # Enrich results
                enriched = []
                for doc, distance in search_result:
                    similarity = 1.0 - distance
                    enriched.append(
                        {
                            "document": {
                                "id": str(doc.id),
                                "title": doc.title,
                                "document_type": doc.document_type,
                                "similarity_score": round(similarity, 4),
                            }
                        }
                    )
                final_results.append(enriched)

        return final_results

    # Job queue integration
    async def enqueue_embedding_generation(
        self, document_ids: List[str], priority: JobPriority = JobPriority.NORMAL
    ) -> str:
        """Enqueue embedding generation as background job."""

        job_id = self.job_queue.enqueue(
            task_name="embedding_generation",
            args=[document_ids],
            priority=priority,
            max_retries=2,
        )

        logger.info(
            f"Enqueued embedding generation job {job_id} for {len(document_ids)} documents"
        )
        return job_id

    async def process_embedding_job(self, document_ids: List[str]) -> Dict[str, Any]:
        """Process embedding generation job (called by job worker)."""

        logger.info(
            f"Processing embedding generation job for {len(document_ids)} documents"
        )

        try:
            results = await self.generate_embeddings_batch(document_ids)

            success_count = sum(1 for r in results.values() if r is not None)

            logger.info(
                f"Completed embedding generation job: {success_count}/{len(document_ids)} successful"
            )

            return {
                "total_documents": len(document_ids),
                "successful": success_count,
                "failed": len(document_ids) - success_count,
                "results": {
                    doc_id: {
                        "success": result is not None,
                        "quality_score": result.quality_score if result else 0.0,
                        "token_count": result.token_count if result else 0,
                    }
                    for doc_id, result in results.items()
                },
            }

        except Exception as e:
            logger.error(f"Embedding generation job failed: {e}")
            raise

    async def update_embeddings_for_new_documents(
        self, hours_back: int = 24
    ) -> Dict[str, Any]:
        """Update embeddings for recently added documents."""

        logger.info(f"Updating embeddings for documents from last {hours_back} hours")

        # Get documents without embeddings
        documents_without_embeddings = (
            await self.db_operations.get_documents_without_embeddings(
                limit=100, hours_back=hours_back
            )
        )

        if not documents_without_embeddings:
            logger.info("No documents found without embeddings")
            return {"updated": 0, "skipped": 0}

        document_ids = [str(doc.id) for doc in documents_without_embeddings]

        # Generate embeddings
        results = await self.generate_embeddings_batch(document_ids)

        success_count = sum(1 for r in results.values() if r is not None)

        logger.info(
            f"Updated embeddings for {success_count}/{len(document_ids)} documents"
        )

        return {
            "total_found": len(document_ids),
            "updated": success_count,
            "failed": len(document_ids) - success_count,
        }

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding generation statistics."""
        # This would typically come from monitoring/metrics
        return {
            "model_name": self.config.model_name,
            "embedding_dim": self.config.embedding_dim,
            "cache_enabled": self.config.cache_embeddings,
            "device": str(self.embedder.device)
            if hasattr(self.embedder, "device")
            else "unknown",
        }


# Global operations instance
_embedding_operations: Optional[EmbeddingOperations] = None


def get_embedding_operations(
    config: Optional[EmbeddingConfig] = None,
) -> EmbeddingOperations:
    """Get global embedding operations instance."""
    global _embedding_operations

    if _embedding_operations is None:
        _embedding_operations = EmbeddingOperations(config)

    return _embedding_operations
