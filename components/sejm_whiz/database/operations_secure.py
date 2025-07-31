"""Secure database operations for legal documents and embeddings - SQL Injection Fixes."""

import logging
from typing import List, Optional, Dict, Tuple, Any
from uuid import UUID
from datetime import datetime, timedelta
from contextlib import contextmanager

from sqlalchemy import text, and_, or_, bindparam, Integer, String, Float
from sqlalchemy.exc import IntegrityError
import numpy as np

from .models import LegalDocument, DocumentEmbedding
from .connection import get_db_session
from .config import DatabaseConfig

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base database operation error."""

    pass


class ValidationError(DatabaseError):
    """Input validation error."""

    pass


class SecurityError(DatabaseError):
    """Security-related database error."""

    pass


def validate_uuid(value: Any) -> UUID:
    """Validate and convert to UUID, preventing injection."""
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        try:
            return UUID(value)
        except ValueError:
            raise ValidationError(f"Invalid UUID format: {value}")
    raise ValidationError(f"Expected UUID, got {type(value)}")


def validate_string(
    value: Any, max_length: int = 1000, allow_none: bool = False
) -> Optional[str]:
    """Validate string input to prevent injection attacks."""
    if value is None:
        if allow_none:
            return None
        raise ValidationError("String value cannot be None")

    if not isinstance(value, str):
        raise ValidationError(f"Expected string, got {type(value)}")

    # Check for suspicious patterns that could indicate injection attempts
    suspicious_patterns = [
        ";",
        "--",
        "/*",
        "*/",
        "UNION",
        "SELECT",
        "DROP",
        "DELETE",
        "INSERT",
        "UPDATE",
        "CREATE",
        "ALTER",
        "EXEC",
        "EXECUTE",
    ]

    value_upper = value.upper()
    for pattern in suspicious_patterns:
        if pattern in value_upper:
            logger.warning(f"Suspicious pattern detected in input: {pattern}")
            # In production, you might want to sanitize or reject

    if len(value) > max_length:
        raise ValidationError(f"String too long: {len(value)} > {max_length}")

    return value


def validate_embedding(embedding: Any) -> List[float]:
    """Validate embedding vector."""
    if not isinstance(embedding, (list, np.ndarray)):
        raise ValidationError(f"Expected list or array, got {type(embedding)}")

    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()

    if not all(isinstance(x, (int, float)) for x in embedding):
        raise ValidationError("Embedding must contain only numeric values")

    if len(embedding) != 768:  # HerBERT embedding dimension
        raise ValidationError(f"Expected 768 dimensions, got {len(embedding)}")

    return [float(x) for x in embedding]


@contextmanager
def safe_db_session():
    """Context manager for safe database sessions with proper cleanup."""
    session = None
    try:
        session = get_db_session()
        yield session
        session.commit()
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Database operation failed: {e}")
        raise DatabaseError(f"Database operation failed: {str(e)}")
    finally:
        if session:
            session.close()


class SecureDocumentOperations:
    """Secure operations for legal documents with SQL injection prevention."""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def create_document(self, document_data: Dict[str, Any]) -> LegalDocument:
        """Create a new legal document with input validation."""
        try:
            # Validate required fields
            title = validate_string(document_data.get("title"), max_length=500)
            content = validate_string(document_data.get("content"), max_length=1000000)
            document_type = validate_string(
                document_data.get("document_type"), max_length=100
            )

            # Validate optional fields
            source_url = validate_string(
                document_data.get("source_url"), max_length=500, allow_none=True
            )
            eli_identifier = validate_string(
                document_data.get("eli_identifier"), max_length=200, allow_none=True
            )
            legal_act_type = validate_string(
                document_data.get("legal_act_type"), max_length=100, allow_none=True
            )
            legal_domain = validate_string(
                document_data.get("legal_domain"), max_length=100, allow_none=True
            )

            # Validate embedding if provided
            embedding = None
            if document_data.get("embedding"):
                embedding = validate_embedding(document_data["embedding"])

            with safe_db_session() as session:
                document = LegalDocument(
                    title=title,
                    content=content,
                    document_type=document_type,
                    source_url=source_url,
                    eli_identifier=eli_identifier,
                    legal_act_type=legal_act_type,
                    legal_domain=legal_domain,
                    embedding=embedding,
                    is_amendment=document_data.get("is_amendment", False),
                    affects_multiple_acts=document_data.get(
                        "affects_multiple_acts", False
                    ),
                    published_at=document_data.get("published_at"),
                )

                session.add(document)
                session.flush()
                session.refresh(document)

                self.logger.info(f"Created document: {document.id}")
                return document

        except (ValidationError, IntegrityError) as e:
            self.logger.error(f"Document creation failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error creating document: {e}")
            raise DatabaseError(f"Failed to create document: {str(e)}")

    def get_document_by_id(self, document_id: Any) -> Optional[LegalDocument]:
        """Get document by ID with proper validation."""
        try:
            doc_id = validate_uuid(document_id)

            with safe_db_session() as session:
                # Use parameterized query
                document = (
                    session.query(LegalDocument)
                    .filter(LegalDocument.id == doc_id)
                    .first()
                )

                if document:
                    self.logger.debug(f"Retrieved document: {doc_id}")
                return document

        except ValidationError as e:
            self.logger.warning(f"Invalid document ID: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving document: {e}")
            raise DatabaseError(f"Failed to retrieve document: {str(e)}")

    def get_document_by_eli(self, eli_identifier: Any) -> Optional[LegalDocument]:
        """Get document by ELI identifier with validation."""
        try:
            eli_id = validate_string(eli_identifier, max_length=200)

            with safe_db_session() as session:
                # Use parameterized query with bindparam for security
                query = session.query(LegalDocument).filter(
                    LegalDocument.eli_identifier == bindparam("eli_param", String)
                )

                document = query.params(eli_param=eli_id).first()

                if document:
                    self.logger.debug(f"Retrieved document by ELI: {eli_id}")
                return document

        except ValidationError as e:
            self.logger.warning(f"Invalid ELI identifier: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving document by ELI: {e}")
            raise DatabaseError(f"Failed to retrieve document: {str(e)}")

    def update_document_embedding(self, document_id: Any, embedding: Any) -> bool:
        """Update document embedding with validation."""
        try:
            doc_id = validate_uuid(document_id)
            validated_embedding = validate_embedding(embedding)

            with safe_db_session() as session:
                # Use parameterized update
                result = (
                    session.query(LegalDocument)
                    .filter(LegalDocument.id == doc_id)
                    .update(
                        {
                            "embedding": validated_embedding,
                            "updated_at": datetime.utcnow(),
                        },
                        synchronize_session=False,
                    )
                )

                if result > 0:
                    self.logger.info(f"Updated embedding for document: {doc_id}")
                    return True
                else:
                    self.logger.warning(
                        f"Document not found for embedding update: {doc_id}"
                    )
                    return False

        except ValidationError as e:
            self.logger.error(f"Validation failed for embedding update: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error updating embedding: {e}")
            raise DatabaseError(f"Failed to update embedding: {str(e)}")

    def search_documents(
        self,
        query: Optional[str] = None,
        document_type: Optional[str] = None,
        legal_domain: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[LegalDocument]:
        """Search documents with safe parameterized queries."""
        try:
            # Validate inputs
            if query:
                query = validate_string(query, max_length=1000)
            if document_type:
                document_type = validate_string(document_type, max_length=100)
            if legal_domain:
                legal_domain = validate_string(legal_domain, max_length=100)

            # Validate numeric parameters
            if not isinstance(limit, int) or limit < 1 or limit > 1000:
                raise ValidationError(f"Invalid limit: {limit}")
            if not isinstance(offset, int) or offset < 0:
                raise ValidationError(f"Invalid offset: {offset}")

            with safe_db_session() as session:
                query_obj = session.query(LegalDocument)

                # Use parameterized queries for all filters
                if query:
                    # Use LIKE with proper escaping
                    search_pattern = (
                        f"%{query.replace('%', '\\%').replace('_', '\\_')}%"
                    )
                    query_obj = query_obj.filter(
                        or_(
                            LegalDocument.title.ilike(
                                bindparam("title_search", String)
                            ),
                            LegalDocument.content.ilike(
                                bindparam("content_search", String)
                            ),
                        )
                    ).params(title_search=search_pattern, content_search=search_pattern)

                if document_type:
                    query_obj = query_obj.filter(
                        LegalDocument.document_type == bindparam("doc_type", String)
                    ).params(doc_type=document_type)

                if legal_domain:
                    query_obj = query_obj.filter(
                        LegalDocument.legal_domain == bindparam("legal_domain", String)
                    ).params(legal_domain=legal_domain)

                results = query_obj.offset(offset).limit(limit).all()

                self.logger.info(f"Document search returned {len(results)} results")
                return results

        except ValidationError as e:
            self.logger.error(f"Search validation failed: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Document search failed: {e}")
            raise DatabaseError(f"Search failed: {str(e)}")


class SecureVectorOperations:
    """Secure vector similarity operations with SQL injection prevention."""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def find_similar_documents(
        self,
        query_embedding: Any,
        limit: int = 10,
        threshold: float = 0.7,
        document_type: Optional[str] = None,
    ) -> List[Tuple[LegalDocument, float]]:
        """Find similar documents using secure parameterized queries."""
        try:
            # Validate inputs
            validated_embedding = validate_embedding(query_embedding)

            if not isinstance(limit, int) or limit < 1 or limit > 100:
                raise ValidationError(f"Invalid limit: {limit}")
            if (
                not isinstance(threshold, (int, float))
                or threshold < 0
                or threshold > 1
            ):
                raise ValidationError(f"Invalid threshold: {threshold}")
            if document_type:
                document_type = validate_string(document_type, max_length=100)

            with safe_db_session() as session:
                # Use parameterized query for vector operations
                # Convert embedding to proper pgvector format
                embedding_str = str(validated_embedding)

                # Build parameterized query using text() with bound parameters
                base_query = text("""
                    SELECT d.*, 
                           (1 - (d.embedding <=> :query_embedding)) as similarity
                    FROM legal_documents d
                    WHERE d.embedding IS NOT NULL
                    AND (1 - (d.embedding <=> :query_embedding)) >= :threshold
                    AND (:doc_type IS NULL OR d.document_type = :doc_type)
                    ORDER BY d.embedding <=> :query_embedding
                    LIMIT :limit_val
                """).bindparam(
                    bindparam("query_embedding", String),
                    bindparam("threshold", Float),
                    bindparam("doc_type", String),
                    bindparam("limit_val", Integer),
                )

                # Execute with bound parameters
                result = session.execute(
                    base_query,
                    {
                        "query_embedding": embedding_str,
                        "threshold": threshold,
                        "doc_type": document_type,
                        "limit_val": limit,
                    },
                )

                # Process results
                documents_with_similarity = []
                for row in result:
                    # Reconstruct LegalDocument from row data
                    doc = (
                        session.query(LegalDocument)
                        .filter(LegalDocument.id == row.id)
                        .first()
                    )
                    if doc:
                        similarity = float(row.similarity)
                        documents_with_similarity.append((doc, similarity))

                self.logger.info(
                    f"Found {len(documents_with_similarity)} similar documents"
                )
                return documents_with_similarity

        except ValidationError as e:
            self.logger.error(f"Vector search validation failed: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Vector similarity search failed: {e}")
            raise DatabaseError(f"Similarity search failed: {str(e)}")

    def find_similar_by_document_id(
        self, document_id: Any, limit: int = 10, threshold: float = 0.7
    ) -> List[Tuple[LegalDocument, float]]:
        """Find documents similar to a given document with secure queries."""
        try:
            doc_id = validate_uuid(document_id)

            with safe_db_session() as session:
                # Get source document with parameterized query
                source_doc = (
                    session.query(LegalDocument)
                    .filter(LegalDocument.id == doc_id)
                    .first()
                )

                if not source_doc or not source_doc.embedding:
                    self.logger.warning(
                        f"Source document not found or no embedding: {doc_id}"
                    )
                    return []

                # Use the source document's embedding for similarity search
                similar_docs = self.find_similar_documents(
                    query_embedding=source_doc.embedding,
                    limit=limit + 1,  # +1 to account for self-match
                    threshold=threshold,
                )

                # Filter out the source document itself
                filtered_results = [
                    (doc, similarity)
                    for doc, similarity in similar_docs
                    if doc.id != doc_id
                ][:limit]

                self.logger.info(
                    f"Found {len(filtered_results)} documents similar to {doc_id}"
                )
                return filtered_results

        except ValidationError as e:
            self.logger.error(f"Document similarity validation failed: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Document similarity search failed: {e}")
            raise DatabaseError(f"Document similarity search failed: {str(e)}")

    def batch_similarity_search(
        self, embeddings: List[Any], limit: int = 5, threshold: float = 0.7
    ) -> List[List[Tuple[LegalDocument, float]]]:
        """Perform batch similarity search with validation."""
        try:
            # Validate inputs
            validated_embeddings = [validate_embedding(emb) for emb in embeddings]

            if len(validated_embeddings) > 100:  # Prevent resource exhaustion
                raise ValidationError("Too many embeddings in batch (max 100)")

            results = []
            for embedding in validated_embeddings:
                similar_docs = self.find_similar_documents(
                    query_embedding=embedding, limit=limit, threshold=threshold
                )
                results.append(similar_docs)

            self.logger.info(
                f"Completed batch similarity search for {len(embeddings)} embeddings"
            )
            return results

        except ValidationError as e:
            self.logger.error(f"Batch similarity validation failed: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Batch similarity search failed: {e}")
            raise DatabaseError(f"Batch similarity search failed: {str(e)}")


class SecureEmbeddingOperations:
    """Secure operations for document embeddings."""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def store_document_embedding(
        self,
        document_id: Any,
        embedding: Any,
        model_name: str,
        model_version: str,
        embedding_method: str = "mean_pooling",
        token_count: Optional[int] = None,
        confidence_score: Optional[int] = None,
    ) -> DocumentEmbedding:
        """Store document embedding with validation."""
        try:
            # Validate inputs
            doc_id = validate_uuid(document_id)
            validated_embedding = validate_embedding(embedding)
            model_name = validate_string(model_name, max_length=100)
            model_version = validate_string(model_version, max_length=50)
            embedding_method = validate_string(embedding_method, max_length=100)

            if token_count is not None and (
                not isinstance(token_count, int) or token_count < 0
            ):
                raise ValidationError(f"Invalid token count: {token_count}")
            if confidence_score is not None and (
                not isinstance(confidence_score, int)
                or confidence_score < 0
                or confidence_score > 100
            ):
                raise ValidationError(f"Invalid confidence score: {confidence_score}")

            with safe_db_session() as session:
                doc_embedding = DocumentEmbedding(
                    document_id=doc_id,
                    embedding=validated_embedding,
                    model_name=model_name,
                    model_version=model_version,
                    embedding_method=embedding_method,
                    token_count=token_count,
                    confidence_score=confidence_score,
                )

                session.add(doc_embedding)
                session.flush()
                session.refresh(doc_embedding)

                self.logger.info(f"Stored embedding for document: {doc_id}")
                return doc_embedding

        except ValidationError as e:
            self.logger.error(f"Embedding storage validation failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error storing embedding: {e}")
            raise DatabaseError(f"Failed to store embedding: {str(e)}")

    def get_documents_without_embeddings(
        self, limit: int = 100, hours_back: int = 24
    ) -> List[LegalDocument]:
        """Get documents without embeddings with secure queries."""
        try:
            # Validate inputs
            if not isinstance(limit, int) or limit < 1 or limit > 1000:
                raise ValidationError(f"Invalid limit: {limit}")
            if (
                not isinstance(hours_back, int) or hours_back < 1 or hours_back > 8760
            ):  # Max 1 year
                raise ValidationError(f"Invalid hours_back: {hours_back}")

            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)

            with safe_db_session() as session:
                # Use parameterized query
                query = (
                    session.query(LegalDocument)
                    .filter(
                        and_(
                            LegalDocument.embedding.is_(None),
                            LegalDocument.created_at >= bindparam("cutoff_time"),
                        )
                    )
                    .params(cutoff_time=cutoff_time)
                    .limit(limit)
                )

                results = query.all()

                self.logger.info(f"Found {len(results)} documents without embeddings")
                return results

        except ValidationError as e:
            self.logger.error(f"Query validation failed: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error querying documents without embeddings: {e}")
            raise DatabaseError(f"Query failed: {str(e)}")


# Factory function for creating secure operations
def create_secure_operations(config: Optional[DatabaseConfig] = None) -> Dict[str, Any]:
    """Create all secure operation classes."""
    return {
        "documents": SecureDocumentOperations(config),
        "vectors": SecureVectorOperations(config),
        "embeddings": SecureEmbeddingOperations(config),
    }
