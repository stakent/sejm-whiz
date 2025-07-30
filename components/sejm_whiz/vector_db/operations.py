"""CRUD operations for vector database."""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import select, update, delete, text
from sqlalchemy.exc import SQLAlchemyError

from sejm_whiz.database.models import LegalDocument, DocumentEmbedding
from .connection import get_vector_session

logger = logging.getLogger(__name__)


class VectorDBOperations:
    """Vector database CRUD operations."""

    def create_document_with_embedding(
        self,
        title: str,
        content: str,
        document_type: str,
        embedding: List[float],
        eli_identifier: Optional[str] = None,
        source_url: Optional[str] = None,
        legal_act_type: Optional[str] = None,
        legal_domain: Optional[str] = None,
        **kwargs,
    ) -> UUID:
        """Create a new legal document with embedding."""
        try:
            with get_vector_session() as session:
                document = LegalDocument(
                    title=title,
                    content=content,
                    document_type=document_type,
                    embedding=embedding,
                    eli_identifier=eli_identifier,
                    source_url=source_url,
                    legal_act_type=legal_act_type,
                    legal_domain=legal_domain,
                    **kwargs,
                )
                session.add(document)
                session.flush()  # Get the ID before commit
                document_id = document.id
                session.commit()

                logger.info(f"Created document with ID: {document_id}")
                return document_id

        except SQLAlchemyError as e:
            logger.error(f"Failed to create document with embedding: {e}")
            raise

    def update_document_embedding(
        self, document_id: UUID, embedding: List[float]
    ) -> bool:
        """Update document embedding."""
        try:
            with get_vector_session() as session:
                stmt = (
                    update(LegalDocument)
                    .where(LegalDocument.id == document_id)
                    .values(embedding=embedding)
                )
                result = session.execute(stmt)
                session.commit()

                success = result.rowcount > 0
                if success:
                    logger.info(f"Updated embedding for document: {document_id}")
                else:
                    logger.warning(f"No document found with ID: {document_id}")

                return success

        except SQLAlchemyError as e:
            logger.error(f"Failed to update document embedding: {e}")
            raise

    def get_document_by_id(self, document_id: UUID) -> Optional[LegalDocument]:
        """Get document by ID."""
        try:
            with get_vector_session() as session:
                stmt = select(LegalDocument).where(LegalDocument.id == document_id)
                result = session.execute(stmt)
                document = result.scalar_one_or_none()

                if document:
                    # Create a detached copy with all attributes loaded
                    doc_dict = {
                        "id": document.id,
                        "title": document.title,
                        "content": document.content,
                        "document_type": document.document_type,
                        "source_url": document.source_url,
                        "eli_identifier": document.eli_identifier,
                        "embedding": document.embedding,
                        "created_at": document.created_at,
                        "updated_at": document.updated_at,
                        "published_at": document.published_at,
                        "legal_act_type": document.legal_act_type,
                        "legal_domain": document.legal_domain,
                        "is_amendment": document.is_amendment,
                        "affects_multiple_acts": document.affects_multiple_acts,
                    }

                    # Create a new detached instance
                    detached_doc = LegalDocument(
                        **{k: v for k, v in doc_dict.items() if k != "id"}
                    )
                    detached_doc.id = doc_dict["id"]

                    return detached_doc

                return None

        except SQLAlchemyError as e:
            logger.error(f"Failed to get document by ID: {e}")
            raise

    def get_documents_by_type(
        self, document_type: str, limit: Optional[int] = None
    ) -> List[LegalDocument]:
        """Get documents by type."""
        try:
            with get_vector_session() as session:
                stmt = select(LegalDocument).where(
                    LegalDocument.document_type == document_type
                )
                if limit:
                    stmt = stmt.limit(limit)

                result = session.execute(stmt)
                documents = result.scalars().all()

                # Create detached copies
                detached_docs = []
                for doc in documents:
                    doc_dict = {
                        "id": doc.id,
                        "title": doc.title,
                        "content": doc.content,
                        "document_type": doc.document_type,
                        "source_url": doc.source_url,
                        "eli_identifier": doc.eli_identifier,
                        "embedding": doc.embedding,
                        "created_at": doc.created_at,
                        "updated_at": doc.updated_at,
                        "published_at": doc.published_at,
                        "legal_act_type": doc.legal_act_type,
                        "legal_domain": doc.legal_domain,
                        "is_amendment": doc.is_amendment,
                        "affects_multiple_acts": doc.affects_multiple_acts,
                    }

                    detached_doc = LegalDocument(
                        **{k: v for k, v in doc_dict.items() if k != "id"}
                    )
                    detached_doc.id = doc_dict["id"]
                    detached_docs.append(detached_doc)

                return detached_docs

        except SQLAlchemyError as e:
            logger.error(f"Failed to get documents by type: {e}")
            raise

    def delete_document(self, document_id: UUID) -> bool:
        """Delete document and its embeddings."""
        try:
            with get_vector_session() as session:
                # First delete related embeddings
                embedding_stmt = delete(DocumentEmbedding).where(
                    DocumentEmbedding.document_id == document_id
                )
                session.execute(embedding_stmt)

                # Then delete the document
                document_stmt = delete(LegalDocument).where(
                    LegalDocument.id == document_id
                )
                result = session.execute(document_stmt)
                session.commit()

                success = result.rowcount > 0
                if success:
                    logger.info(f"Deleted document: {document_id}")
                else:
                    logger.warning(f"No document found with ID: {document_id}")

                return success

        except SQLAlchemyError as e:
            logger.error(f"Failed to delete document: {e}")
            raise

    def create_document_embedding(
        self,
        document_id: UUID,
        embedding: List[float],
        model_name: str,
        model_version: str,
        embedding_method: str = "bag_of_embeddings",
        token_count: Optional[int] = None,
        chunk_size: Optional[int] = None,
        preprocessing_version: Optional[str] = None,
        confidence_score: Optional[int] = None,
    ) -> UUID:
        """Create a new document embedding record."""
        try:
            with get_vector_session() as session:
                doc_embedding = DocumentEmbedding(
                    document_id=document_id,
                    embedding=embedding,
                    model_name=model_name,
                    model_version=model_version,
                    embedding_method=embedding_method,
                    token_count=token_count,
                    chunk_size=chunk_size,
                    preprocessing_version=preprocessing_version,
                    confidence_score=confidence_score,
                )
                session.add(doc_embedding)
                session.flush()
                embedding_id = doc_embedding.id
                session.commit()

                logger.info(f"Created embedding record with ID: {embedding_id}")
                return embedding_id

        except SQLAlchemyError as e:
            logger.error(f"Failed to create document embedding: {e}")
            raise

    def get_document_embeddings(self, document_id: UUID) -> List[DocumentEmbedding]:
        """Get all embeddings for a document."""
        try:
            with get_vector_session() as session:
                stmt = select(DocumentEmbedding).where(
                    DocumentEmbedding.document_id == document_id
                )
                result = session.execute(stmt)
                embeddings = result.scalars().all()

                # Create detached copies
                detached_embeddings = []
                for emb in embeddings:
                    emb_dict = {
                        "id": emb.id,
                        "document_id": emb.document_id,
                        "embedding": emb.embedding,
                        "model_name": emb.model_name,
                        "model_version": emb.model_version,
                        "embedding_method": emb.embedding_method,
                        "token_count": emb.token_count,
                        "chunk_size": emb.chunk_size,
                        "preprocessing_version": emb.preprocessing_version,
                        "confidence_score": emb.confidence_score,
                        "created_at": emb.created_at,
                    }

                    detached_emb = DocumentEmbedding(
                        **{k: v for k, v in emb_dict.items() if k != "id"}
                    )
                    detached_emb.id = emb_dict["id"]
                    detached_embeddings.append(detached_emb)

                return detached_embeddings

        except SQLAlchemyError as e:
            logger.error(f"Failed to get document embeddings: {e}")
            raise

    def bulk_insert_documents(self, documents: List[Dict[str, Any]]) -> List[UUID]:
        """Bulk insert documents with embeddings."""
        try:
            with get_vector_session() as session:
                document_objects = []
                for doc_data in documents:
                    document = LegalDocument(**doc_data)
                    document_objects.append(document)

                session.add_all(document_objects)
                session.flush()  # Get IDs before commit

                document_ids = [doc.id for doc in document_objects]
                session.commit()

                logger.info(f"Bulk inserted {len(document_ids)} documents")
                return document_ids

        except SQLAlchemyError as e:
            logger.error(f"Failed to bulk insert documents: {e}")
            raise

    def count_documents(self, document_type: Optional[str] = None) -> int:
        """Count documents, optionally filtered by type."""
        try:
            with get_vector_session() as session:
                stmt = select(LegalDocument)
                if document_type:
                    stmt = stmt.where(LegalDocument.document_type == document_type)

                result = session.execute(
                    select(text("COUNT(*)")).select_from(stmt.subquery())
                )
                return result.scalar()

        except SQLAlchemyError as e:
            logger.error(f"Failed to count documents: {e}")
            raise


# Global operations instance
_vector_ops: Optional[VectorDBOperations] = None


def get_vector_operations() -> VectorDBOperations:
    """Get or create global vector operations instance."""
    global _vector_ops
    if _vector_ops is None:
        _vector_ops = VectorDBOperations()
    return _vector_ops
