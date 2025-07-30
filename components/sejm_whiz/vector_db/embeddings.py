"""Vector similarity search operations."""

import logging
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from uuid import UUID

from sqlalchemy import select, text, func
from sqlalchemy.exc import SQLAlchemyError
import numpy as np

from sejm_whiz.database.models import LegalDocument, DocumentEmbedding
from .connection import get_vector_session

logger = logging.getLogger(__name__)


class DistanceMetric(Enum):
    """Available distance metrics for vector similarity."""

    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "inner_product"


class VectorSimilaritySearch:
    """Vector similarity search operations using pgvector."""

    def __init__(self):
        """Initialize vector similarity search."""
        self.distance_operators = {
            DistanceMetric.COSINE: "<=>",
            DistanceMetric.L2: "<->",
            DistanceMetric.INNER_PRODUCT: "<#>",
        }

    def find_similar_documents(
        self,
        query_embedding: List[float],
        limit: int = 10,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        document_type: Optional[str] = None,
        legal_domain: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> List[Tuple[LegalDocument, float]]:
        """Find similar documents using vector similarity."""
        try:
            with get_vector_session() as session:
                distance_op = self.distance_operators[distance_metric]

                # Skip the complex parameter binding and use simple raw SQL approach
                # Ensure embedding is a list of floats
                if isinstance(query_embedding, np.ndarray):
                    query_embedding = query_embedding.tolist()
                elif not isinstance(query_embedding, list):
                    query_embedding = list(query_embedding)

                # Build raw SQL query (similar to operations_secure.py but simpler)
                sql_parts = ["SELECT *, (embedding "]
                sql_parts.append(distance_op)
                sql_parts.append(" ARRAY[")
                sql_parts.append(",".join(str(float(x)) for x in query_embedding))
                sql_parts.append(
                    "]::vector) as distance FROM legal_documents WHERE embedding IS NOT NULL"
                )

                # Add filters
                if document_type:
                    sql_parts.append(f" AND document_type = '{document_type}'")

                if legal_domain:
                    sql_parts.append(f" AND legal_domain = '{legal_domain}'")

                if threshold is not None:
                    sql_parts.append(f" AND (embedding {distance_op} ARRAY[")
                    sql_parts.append(",".join(str(float(x)) for x in query_embedding))
                    sql_parts.append(f"]::vector) <= {threshold}")

                sql_parts.append(f" ORDER BY distance LIMIT {limit}")

                raw_sql = "".join(sql_parts)
                result = session.execute(text(raw_sql))

                # Process raw SQL results
                loaded_results = []
                for row in result.fetchall():
                    # Create detached copy from raw SQL row
                    doc_dict = {
                        "id": row.id,
                        "title": row.title,
                        "content": row.content,
                        "document_type": row.document_type,
                        "source_url": row.source_url,
                        "eli_identifier": row.eli_identifier,
                        "embedding": row.embedding,
                        "created_at": row.created_at,
                        "updated_at": row.updated_at,
                        "published_at": row.published_at,
                        "legal_act_type": row.legal_act_type,
                        "legal_domain": row.legal_domain,
                        "is_amendment": row.is_amendment,
                        "affects_multiple_acts": row.affects_multiple_acts,
                    }

                    # Create a new detached instance
                    detached_doc = LegalDocument(
                        **{k: v for k, v in doc_dict.items() if k != "id"}
                    )
                    detached_doc.id = doc_dict["id"]

                    loaded_results.append((detached_doc, row.distance))

                return loaded_results

        except SQLAlchemyError as e:
            logger.error(f"Failed to find similar documents: {e}")
            raise

    def find_similar_by_document_id(
        self,
        document_id: str,
        limit: int = 10,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        exclude_self: bool = True,
    ) -> List[Tuple[LegalDocument, float]]:
        """Find documents similar to a given document."""
        try:
            with get_vector_session() as session:
                # Convert string ID to UUID
                doc_uuid = (
                    UUID(document_id) if isinstance(document_id, str) else document_id
                )

                # First get the embedding of the source document
                source_stmt = select(LegalDocument.embedding).where(
                    LegalDocument.id == doc_uuid
                )
                source_result = session.execute(source_stmt)
                source_embedding = source_result.scalar_one_or_none()

                if source_embedding is None:
                    logger.warning(f"No embedding found for document: {document_id}")
                    return []

                distance_op = self.distance_operators[distance_metric]

                # Use raw SQL approach similar to find_similar_documents to avoid parameter binding issues
                if isinstance(source_embedding, np.ndarray):
                    source_embedding = source_embedding.tolist()
                elif not isinstance(source_embedding, list):
                    source_embedding = list(source_embedding)

                # Build raw SQL query
                sql_parts = ["SELECT *, (embedding "]
                sql_parts.append(distance_op)
                sql_parts.append(" ARRAY[")
                sql_parts.append(",".join(str(float(x)) for x in source_embedding))
                sql_parts.append(
                    "]::vector) as distance FROM legal_documents WHERE embedding IS NOT NULL"
                )

                if exclude_self:
                    sql_parts.append(f" AND id != '{doc_uuid}'")

                sql_parts.append(f" ORDER BY distance LIMIT {limit}")

                raw_sql = "".join(sql_parts)
                result = session.execute(text(raw_sql))

                # Process raw SQL results (similar to find_similar_documents)
                loaded_results = []
                for row in result.fetchall():
                    # Create detached copy from raw SQL row
                    doc_dict = {
                        "id": row.id,
                        "title": row.title,
                        "content": row.content,
                        "document_type": row.document_type,
                        "source_url": row.source_url,
                        "eli_identifier": row.eli_identifier,
                        "embedding": row.embedding,
                        "created_at": row.created_at,
                        "updated_at": row.updated_at,
                        "published_at": row.published_at,
                        "legal_act_type": row.legal_act_type,
                        "legal_domain": row.legal_domain,
                        "is_amendment": row.is_amendment,
                        "affects_multiple_acts": row.affects_multiple_acts,
                    }

                    # Create a new detached instance
                    detached_doc = LegalDocument(
                        **{k: v for k, v in doc_dict.items() if k != "id"}
                    )
                    detached_doc.id = doc_dict["id"]

                    loaded_results.append((detached_doc, row.distance))

                # Final filtering to ensure limit (already handled in SQL but double-check)
                loaded_results = loaded_results[:limit]

                return loaded_results

        except SQLAlchemyError as e:
            logger.error(f"Failed to find similar documents by ID: {e}")
            raise

    def batch_similarity_search(
        self,
        query_embeddings: List[List[float]],
        limit: int = 10,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
    ) -> List[List[Tuple[LegalDocument, float]]]:
        """Perform batch similarity search for multiple embeddings."""
        results = []

        for embedding in query_embeddings:
            similar_docs = self.find_similar_documents(
                query_embedding=embedding, limit=limit, distance_metric=distance_metric
            )
            results.append(similar_docs)

        return results

    def find_documents_in_range(
        self,
        query_embedding: List[float],
        min_distance: float,
        max_distance: float,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        limit: Optional[int] = None,
    ) -> List[Tuple[LegalDocument, float]]:
        """Find documents within a specific distance range."""
        try:
            with get_vector_session() as session:
                distance_op = self.distance_operators[distance_metric]

                from sqlalchemy import literal_column

                distance_col = literal_column(
                    f"embedding {distance_op} :query_embedding"
                ).label("distance")

                stmt = (
                    select(LegalDocument, distance_col)
                    .where(LegalDocument.embedding.is_not(None))
                    .where(distance_col >= min_distance)
                    .where(distance_col <= max_distance)
                    .order_by(distance_col)
                )

                if limit:
                    stmt = stmt.limit(limit)

                result = session.execute(stmt, {"query_embedding": query_embedding})

                # Load results and create detached copies
                results = []
                for row in result:
                    doc = row.LegalDocument
                    distance = row.distance

                    # Create detached copy
                    doc_dict = {
                        "id": doc.id,
                        "title": doc.title,
                        "content": doc.content,
                        "document_type": doc.document_type,
                        "source_url": doc.source_url,
                        "eli_identifier": doc.eli_identifier,
                        "embedding": doc.embedding,
                        "legal_act_type": doc.legal_act_type,
                        "legal_domain": doc.legal_domain,
                        "is_amendment": doc.is_amendment,
                        "affects_multiple_acts": doc.affects_multiple_acts,
                        "created_at": doc.created_at,
                        "updated_at": doc.updated_at,
                        "published_at": doc.published_at,
                    }

                    detached_doc = LegalDocument(
                        **{k: v for k, v in doc_dict.items() if k != "id"}
                    )
                    detached_doc.id = doc_dict["id"]

                    results.append((detached_doc, distance))

                return results

        except SQLAlchemyError as e:
            logger.error(f"Failed to find documents in range: {e}")
            raise

    def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get statistics about embeddings in the database."""
        try:
            with get_vector_session() as session:
                # Count documents with embeddings
                docs_with_embeddings = session.execute(
                    select(func.count(LegalDocument.id)).where(
                        LegalDocument.embedding.is_not(None)
                    )
                ).scalar()

                # Count total documents
                total_docs = session.execute(
                    select(func.count(LegalDocument.id))
                ).scalar()

                # Count by document type
                type_counts = session.execute(
                    select(LegalDocument.document_type, func.count(LegalDocument.id))
                    .where(LegalDocument.embedding.is_not(None))
                    .group_by(LegalDocument.document_type)
                ).all()

                # Count embedding records
                embedding_records = session.execute(
                    select(func.count(DocumentEmbedding.id))
                ).scalar()

                return {
                    "documents_with_embeddings": docs_with_embeddings,
                    "total_documents": total_docs,
                    "embedding_coverage": (
                        docs_with_embeddings / total_docs * 100 if total_docs > 0 else 0
                    ),
                    "type_distribution": dict(type_counts),
                    "embedding_records": embedding_records,
                }

        except SQLAlchemyError as e:
            logger.error(f"Failed to get embedding statistics: {e}")
            raise

    def create_vector_index(
        self,
        table_name: str = "legal_documents",
        column_name: str = "embedding",
        index_type: str = "ivfflat",
        lists: int = 100,
    ) -> bool:
        """Create vector index for improved performance."""
        try:
            with get_vector_session() as session:
                index_name = f"idx_{table_name}_{column_name}_{index_type}"

                if index_type == "ivfflat":
                    sql = text(f"""
                        CREATE INDEX CONCURRENTLY {index_name}
                        ON {table_name}
                        USING ivfflat ({column_name} vector_cosine_ops)
                        WITH (lists = {lists})
                    """)
                elif index_type == "hnsw":
                    sql = text(f"""
                        CREATE INDEX CONCURRENTLY {index_name}
                        ON {table_name}
                        USING hnsw ({column_name} vector_cosine_ops)
                    """)
                else:
                    raise ValueError(f"Unsupported index type: {index_type}")

                session.execute(sql)
                session.commit()

                logger.info(f"Created vector index: {index_name}")
                return True

        except SQLAlchemyError as e:
            logger.error(f"Failed to create vector index: {e}")
            raise


# Global similarity search instance
_similarity_search: Optional[VectorSimilaritySearch] = None


def get_similarity_search() -> VectorSimilaritySearch:
    """Get or create global similarity search instance."""
    global _similarity_search
    if _similarity_search is None:
        _similarity_search = VectorSimilaritySearch()
    return _similarity_search
