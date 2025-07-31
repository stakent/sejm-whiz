"""Document indexing for semantic search."""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID


from sejm_whiz.embeddings import get_herbert_embedder, get_batch_processor, BatchJob
from sejm_whiz.vector_db import get_vector_operations
from sejm_whiz.text_processing import process_legal_document
from sejm_whiz.database.models import LegalDocument

logger = logging.getLogger(__name__)


@dataclass
class IndexingResult:
    """Result from document indexing operation."""

    document_id: UUID
    success: bool
    embedding_dimensions: int
    processing_time_ms: float
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "document_id": str(self.document_id),
            "success": self.success,
            "embedding_dimensions": self.embedding_dimensions,
            "processing_time_ms": self.processing_time_ms,
            "error": self.error,
            "metadata": self.metadata or {},
        }


class DocumentIndexer:
    """Document indexing service for semantic search."""

    def __init__(
        self,
        embedder=None,
        batch_processor=None,
        vector_operations=None,
    ):
        """Initialize document indexer.

        Args:
            embedder: HerBERT embedder instance
            batch_processor: Batch processor for embeddings
            vector_operations: Vector database operations
        """
        self.embedder = embedder or get_herbert_embedder()
        self.batch_processor = batch_processor or get_batch_processor()
        self.vector_operations = vector_operations or get_vector_operations()
        self.logger = logging.getLogger(__name__)

    def index_document(
        self,
        document: LegalDocument,
        overwrite_existing: bool = False,
    ) -> IndexingResult:
        """Index a single document for semantic search.

        Args:
            document: Legal document to index
            overwrite_existing: Whether to overwrite existing embeddings

        Returns:
            IndexingResult with operation details
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Check if document already has embedding
            if not overwrite_existing and document.embedding is not None:
                self.logger.info(
                    f"Document {document.id} already has embedding, skipping"
                )
                return IndexingResult(
                    document_id=document.id,
                    success=True,
                    embedding_dimensions=len(document.embedding),
                    processing_time_ms=0.0,
                    metadata={"skipped": True, "reason": "already_indexed"},
                )

            # Process document text
            processed_doc = process_legal_document(document.content)
            clean_doc_text = processed_doc.get("clean_text", document.content)
            self.logger.debug(
                f"Processed document {document.id}: {len(clean_doc_text)} characters"
            )

            # Generate embedding
            try:
                embedding_result = self.embedder.embed_text(clean_doc_text)
            except Exception as e:
                processing_time = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds() * 1000
                return IndexingResult(
                    document_id=document.id,
                    success=False,
                    embedding_dimensions=0,
                    processing_time_ms=processing_time,
                    error=f"Failed to generate embedding: {str(e)}",
                    metadata={"stage": "embedding_generation"},
                )

            # Update document with embedding
            document.embedding = embedding_result.embedding.tolist()

            # Store in vector database
            self.vector_operations.update_document_embedding(
                document_id=document.id,
                embedding=document.embedding,
            )

            processing_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            self.logger.info(f"Successfully indexed document {document.id}")
            return IndexingResult(
                document_id=document.id,
                success=True,
                embedding_dimensions=len(document.embedding),
                processing_time_ms=processing_time,
                metadata={
                    "text_length": len(document.content),
                    "processed_length": len(clean_doc_text),
                    "document_type": document.document_type,
                    "legal_domain": document.legal_domain,
                },
            )

        except Exception as e:
            processing_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000
            error_msg = str(e)
            self.logger.error(f"Failed to index document {document.id}: {error_msg}")

            return IndexingResult(
                document_id=document.id,
                success=False,
                embedding_dimensions=0,
                processing_time_ms=processing_time,
                error=error_msg,
            )

    def batch_index_documents(
        self,
        documents: List[LegalDocument],
        batch_size: int = 32,
        overwrite_existing: bool = False,
        show_progress: bool = True,
    ) -> List[IndexingResult]:
        """Index multiple documents in batches.

        Args:
            documents: List of legal documents to index
            batch_size: Size of processing batches
            overwrite_existing: Whether to overwrite existing embeddings
            show_progress: Whether to show progress logging

        Returns:
            List of IndexingResult objects
        """
        try:
            results = []
            total_docs = len(documents)

            self.logger.info(f"Starting batch indexing of {total_docs} documents")

            # Filter documents that need indexing
            docs_to_process = []
            for doc in documents:
                if overwrite_existing or doc.embedding is None:
                    docs_to_process.append(doc)
                else:
                    # Document already indexed
                    results.append(
                        IndexingResult(
                            document_id=doc.id,
                            success=True,
                            embedding_dimensions=len(doc.embedding)
                            if doc.embedding
                            else 0,
                            processing_time_ms=0.0,
                            metadata={"skipped": True, "reason": "already_indexed"},
                        )
                    )

            if not docs_to_process:
                self.logger.info("All documents already indexed")
                return results

            self.logger.info(
                f"Processing {len(docs_to_process)} documents that need indexing"
            )

            # Process documents in batches
            for i in range(0, len(docs_to_process), batch_size):
                batch = docs_to_process[i : i + batch_size]
                batch_start = datetime.now(timezone.utc)

                if show_progress:
                    self.logger.info(
                        f"Processing batch {i // batch_size + 1}/{(len(docs_to_process) + batch_size - 1) // batch_size}"
                    )

                # Prepare texts for batch processing
                texts = []
                for doc in batch:
                    processed_doc = process_legal_document(doc.content)
                    clean_text = processed_doc.get("clean_text", doc.content)
                    texts.append(clean_text)

                # Generate embeddings for batch
                try:
                    embedding_results = self.embedder.embed_texts(texts)
                except Exception as e:
                    # If batch embedding fails, fall back to individual processing
                    self.logger.warning(
                        f"Batch embedding failed, falling back to individual processing: {e}"
                    )
                    embedding_results = []
                    for text in texts:
                        try:
                            result = self.embedder.embed_text(text)
                            embedding_results.append(result)
                        except Exception as individual_error:
                            # Create a mock failed result
                            failed_result = type(
                                "FailedResult",
                                (),
                                {"embedding": None, "error": str(individual_error)},
                            )()
                            embedding_results.append(failed_result)

                # Process results
                for j, (doc, embedding_result) in enumerate(
                    zip(batch, embedding_results)
                ):
                    processing_time = (
                        (datetime.now(timezone.utc) - batch_start).total_seconds()
                        * 1000
                        / len(batch)
                    )

                    if (
                        embedding_result
                        and hasattr(embedding_result, "embedding")
                        and embedding_result.embedding is not None
                    ):
                        # Update document with embedding
                        doc.embedding = embedding_result.embedding.tolist()

                        # Store in vector database
                        try:
                            self.vector_operations.update_document_embedding(
                                document_id=doc.id,
                                embedding=doc.embedding,
                            )

                            results.append(
                                IndexingResult(
                                    document_id=doc.id,
                                    success=True,
                                    embedding_dimensions=len(doc.embedding),
                                    processing_time_ms=processing_time,
                                    metadata={
                                        "batch_index": i // batch_size,
                                        "document_index": j,
                                        "text_length": len(doc.content),
                                        "document_type": doc.document_type,
                                    },
                                )
                            )

                        except Exception as e:
                            results.append(
                                IndexingResult(
                                    document_id=doc.id,
                                    success=False,
                                    embedding_dimensions=0,
                                    processing_time_ms=processing_time,
                                    error=f"Database update failed: {str(e)}",
                                )
                            )
                    else:
                        error_msg = (
                            embedding_result.error
                            if embedding_result
                            else "Unknown embedding error"
                        )
                        results.append(
                            IndexingResult(
                                document_id=doc.id,
                                success=False,
                                embedding_dimensions=0,
                                processing_time_ms=processing_time,
                                error=error_msg,
                            )
                        )

            successful = sum(1 for r in results if r.success)
            self.logger.info(
                f"Batch indexing completed: {successful}/{len(results)} documents indexed successfully"
            )

            return results

        except Exception as e:
            self.logger.error(f"Batch indexing failed: {e}")
            raise

    def reindex_all_documents(
        self,
        document_type: Optional[str] = None,
        legal_domain: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[IndexingResult]:
        """Reindex all documents in the database.

        Args:
            document_type: Filter by document type (optional)
            legal_domain: Filter by legal domain (optional)
            batch_size: Size of processing batches

        Returns:
            List of IndexingResult objects
        """
        try:
            # Get all documents from database
            documents = self.vector_operations.get_documents(
                document_type=document_type,
                legal_domain=legal_domain,
            )

            self.logger.info(f"Reindexing {len(documents)} documents")

            return self.batch_index_documents(
                documents=documents,
                batch_size=batch_size,
                overwrite_existing=True,  # Force reindexing
                show_progress=True,
            )

        except Exception as e:
            self.logger.error(f"Reindexing failed: {e}")
            raise

    def get_indexing_statistics(self) -> Dict[str, Any]:
        """Get statistics about document indexing.

        Returns:
            Dictionary with indexing statistics
        """
        try:
            stats = self.vector_operations.get_embedding_statistics()

            return {
                "total_documents": stats.get("total_documents", 0),
                "indexed_documents": stats.get("indexed_documents", 0),
                "indexing_percentage": stats.get("indexing_percentage", 0.0),
                "avg_embedding_dimensions": stats.get("avg_embedding_dimensions", 0),
                "document_types": stats.get("document_types", {}),
                "legal_domains": stats.get("legal_domains", {}),
            }

        except Exception as e:
            self.logger.error(f"Failed to get indexing statistics: {e}")
            return {}


# Singleton instance
_document_indexer_instance = None


def get_document_indexer() -> DocumentIndexer:
    """Get singleton document indexer instance."""
    global _document_indexer_instance
    if _document_indexer_instance is None:
        _document_indexer_instance = DocumentIndexer()
    return _document_indexer_instance
