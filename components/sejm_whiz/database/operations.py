"""Database operations for legal documents and embeddings."""

import logging
from typing import List, Optional, Dict, Tuple
from uuid import UUID
from datetime import datetime

from sqlalchemy import func, text, and_, or_
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector

from .models import LegalDocument, LegalAmendment, CrossReference, DocumentEmbedding
from .connection import get_db_session

logger = logging.getLogger(__name__)


class DocumentOperations:
    """Operations for legal documents."""
    
    @staticmethod
    def create_document(
        title: str,
        content: str,
        document_type: str,
        embedding: Optional[List[float]] = None,
        **kwargs
    ) -> LegalDocument:
        """Create a new legal document."""
        with get_db_session() as session:
            document = LegalDocument(
                title=title,
                content=content,
                document_type=document_type,
                embedding=embedding,
                **kwargs
            )
            session.add(document)
            session.flush()
            session.refresh(document)
            return document
    
    @staticmethod
    def get_document_by_id(document_id: UUID) -> Optional[LegalDocument]:
        """Get document by ID."""
        with get_db_session() as session:
            return session.query(LegalDocument).filter(
                LegalDocument.id == document_id
            ).first()
    
    @staticmethod
    def get_document_by_eli(eli_identifier: str) -> Optional[LegalDocument]:
        """Get document by ELI identifier."""
        with get_db_session() as session:
            return session.query(LegalDocument).filter(
                LegalDocument.eli_identifier == eli_identifier
            ).first()
    
    @staticmethod
    def update_document_embedding(document_id: UUID, embedding: List[float]) -> bool:
        """Update document embedding."""
        with get_db_session() as session:
            result = session.query(LegalDocument).filter(
                LegalDocument.id == document_id
            ).update({"embedding": embedding, "updated_at": datetime.utcnow()})
            return result > 0
    
    @staticmethod
    def search_documents(
        query: Optional[str] = None,
        document_type: Optional[str] = None,
        legal_domain: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[LegalDocument]:
        """Search documents with filters."""
        with get_db_session() as session:
            query_obj = session.query(LegalDocument)
            
            if query:
                query_obj = query_obj.filter(
                    or_(
                        LegalDocument.title.ilike(f"%{query}%"),
                        LegalDocument.content.ilike(f"%{query}%")
                    )
                )
            
            if document_type:
                query_obj = query_obj.filter(LegalDocument.document_type == document_type)
            
            if legal_domain:
                query_obj = query_obj.filter(LegalDocument.legal_domain == legal_domain)
            
            return query_obj.offset(offset).limit(limit).all()


class VectorOperations:
    """Vector similarity operations."""
    
    @staticmethod
    def find_similar_documents(
        embedding: List[float],
        limit: int = 10,
        threshold: float = 0.7,
        document_type: Optional[str] = None
    ) -> List[Tuple[LegalDocument, float]]:
        """Find similar documents using cosine similarity."""
        with get_db_session() as session:
            # Convert embedding to pgvector format
            query_vector = str(embedding)
            
            # Build base query
            query_obj = session.query(
                LegalDocument,
                (1 - LegalDocument.embedding.cosine_distance(query_vector)).label('similarity')
            ).filter(
                LegalDocument.embedding.is_not(None)
            )
            
            # Add document type filter if specified
            if document_type:
                query_obj = query_obj.filter(LegalDocument.document_type == document_type)
            
            # Add similarity threshold
            query_obj = query_obj.filter(
                (1 - LegalDocument.embedding.cosine_distance(query_vector)) >= threshold
            )
            
            # Order by similarity and limit results
            results = query_obj.order_by(
                (1 - LegalDocument.embedding.cosine_distance(query_vector)).desc()
            ).limit(limit).all()
            
            return [(doc, float(similarity)) for doc, similarity in results]
    
    @staticmethod
    def find_similar_by_document_id(
        document_id: UUID,
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Tuple[LegalDocument, float]]:
        """Find documents similar to a given document."""
        with get_db_session() as session:
            # Get the source document's embedding
            source_doc = session.query(LegalDocument).filter(
                LegalDocument.id == document_id
            ).first()
            
            if not source_doc or not source_doc.embedding:
                return []
            
            # Use the embedding to find similar documents
            return VectorOperations.find_similar_documents(
                embedding=source_doc.embedding,
                limit=limit + 1,  # +1 to exclude self
                threshold=threshold
            )[1:]  # Remove self from results
    
    @staticmethod
    def batch_similarity_search(
        embeddings: List[List[float]],
        limit: int = 5,
        threshold: float = 0.7
    ) -> List[List[Tuple[LegalDocument, float]]]:
        """Perform batch similarity search."""
        results = []
        for embedding in embeddings:
            similar_docs = VectorOperations.find_similar_documents(
                embedding=embedding,
                limit=limit,
                threshold=threshold
            )
            results.append(similar_docs)
        return results


class AmendmentOperations:
    """Operations for legal amendments."""
    
    @staticmethod
    def create_amendment(
        document_id: UUID,
        amendment_type: str,
        amendment_text: str,
        **kwargs
    ) -> LegalAmendment:
        """Create a new legal amendment."""
        with get_db_session() as session:
            amendment = LegalAmendment(
                document_id=document_id,
                amendment_type=amendment_type,
                amendment_text=amendment_text,
                **kwargs
            )
            session.add(amendment)
            session.flush()
            session.refresh(amendment)
            return amendment
    
    @staticmethod
    def get_amendments_by_document(document_id: UUID) -> List[LegalAmendment]:
        """Get all amendments for a document."""
        with get_db_session() as session:
            return session.query(LegalAmendment).filter(
                LegalAmendment.document_id == document_id
            ).all()
    
    @staticmethod
    def get_omnibus_amendments(omnibus_bill_id: str) -> List[LegalAmendment]:
        """Get all amendments from an omnibus bill."""
        with get_db_session() as session:
            return session.query(LegalAmendment).filter(
                LegalAmendment.omnibus_bill_id == omnibus_bill_id
            ).all()
    
    @staticmethod
    def mark_multi_act_amendments(
        amendments: List[UUID],
        omnibus_bill_id: str
    ) -> int:
        """Mark amendments as part of multi-act omnibus bill."""
        with get_db_session() as session:
            result = session.query(LegalAmendment).filter(
                LegalAmendment.id.in_(amendments)
            ).update({
                "affects_multiple_acts": True,
                "omnibus_bill_id": omnibus_bill_id
            })
            return result


class CrossReferenceOperations:
    """Operations for cross-references between documents."""
    
    @staticmethod
    def create_cross_reference(
        source_document_id: UUID,
        target_document_id: UUID,
        reference_type: str,
        similarity_score: Optional[int] = None,
        **kwargs
    ) -> CrossReference:
        """Create cross-reference between documents."""
        with get_db_session() as session:
            cross_ref = CrossReference(
                source_document_id=source_document_id,
                target_document_id=target_document_id,
                reference_type=reference_type,
                similarity_score=similarity_score,
                **kwargs
            )
            session.add(cross_ref)
            session.flush()
            session.refresh(cross_ref)
            return cross_ref
    
    @staticmethod
    def get_cross_references_from_document(
        document_id: UUID,
        reference_type: Optional[str] = None
    ) -> List[CrossReference]:
        """Get cross-references from a document."""
        with get_db_session() as session:
            query_obj = session.query(CrossReference).filter(
                CrossReference.source_document_id == document_id
            )
            
            if reference_type:
                query_obj = query_obj.filter(CrossReference.reference_type == reference_type)
            
            return query_obj.all()
    
    @staticmethod
    def get_cross_references_to_document(
        document_id: UUID,
        reference_type: Optional[str] = None
    ) -> List[CrossReference]:
        """Get cross-references to a document."""
        with get_db_session() as session:
            query_obj = session.query(CrossReference).filter(
                CrossReference.target_document_id == document_id
            )
            
            if reference_type:
                query_obj = query_obj.filter(CrossReference.reference_type == reference_type)
            
            return query_obj.all()


class EmbeddingOperations:
    """Operations for document embeddings."""
    
    @staticmethod
    def store_embedding(
        document_id: UUID,
        embedding: List[float],
        model_name: str,
        model_version: str,
        embedding_method: str = "bag_of_embeddings",
        **kwargs
    ) -> DocumentEmbedding:
        """Store document embedding with metadata."""
        with get_db_session() as session:
            doc_embedding = DocumentEmbedding(
                document_id=document_id,
                embedding=embedding,
                model_name=model_name,
                model_version=model_version,
                embedding_method=embedding_method,
                **kwargs
            )
            session.add(doc_embedding)
            session.flush()
            session.refresh(doc_embedding)
            return doc_embedding
    
    @staticmethod
    def get_latest_embedding(
        document_id: UUID,
        model_name: Optional[str] = None
    ) -> Optional[DocumentEmbedding]:
        """Get latest embedding for document."""
        with get_db_session() as session:
            query_obj = session.query(DocumentEmbedding).filter(
                DocumentEmbedding.document_id == document_id
            )
            
            if model_name:
                query_obj = query_obj.filter(DocumentEmbedding.model_name == model_name)
            
            return query_obj.order_by(DocumentEmbedding.created_at.desc()).first()
    
    @staticmethod
    def bulk_similarity_search(
        embeddings: List[List[float]],
        batch_size: int = 100
    ) -> List[List[Tuple[UUID, float]]]:
        """Perform bulk similarity search for multiple embeddings."""
        results = []
        
        with get_db_session() as session:
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i + batch_size]
                batch_results = []
                
                for embedding in batch:
                    query_vector = str(embedding)
                    similar_docs = session.query(
                        DocumentEmbedding.document_id,
                        (1 - DocumentEmbedding.embedding.cosine_distance(query_vector)).label('similarity')
                    ).filter(
                        DocumentEmbedding.embedding.is_not(None)
                    ).order_by(
                        (1 - DocumentEmbedding.embedding.cosine_distance(query_vector)).desc()
                    ).limit(10).all()
                    
                    batch_results.append([(doc_id, float(sim)) for doc_id, sim in similar_docs])
                
                results.extend(batch_results)
        
        return results


class AnalyticsOperations:
    """Analytics and reporting operations."""
    
    @staticmethod
    def get_document_stats() -> Dict[str, int]:
        """Get document statistics."""
        with get_db_session() as session:
            stats = {}
            
            # Total documents
            stats["total_documents"] = session.query(LegalDocument).count()
            
            # Documents by type
            doc_types = session.query(
                LegalDocument.document_type,
                func.count(LegalDocument.id)
            ).group_by(LegalDocument.document_type).all()
            
            stats["by_type"] = {doc_type: count for doc_type, count in doc_types}
            
            # Documents with embeddings
            stats["with_embeddings"] = session.query(LegalDocument).filter(
                LegalDocument.embedding.is_not(None)
            ).count()
            
            # Multi-act amendments
            stats["multi_act_amendments"] = session.query(LegalAmendment).filter(
                LegalAmendment.affects_multiple_acts == True
            ).count()
            
            return stats
    
    @staticmethod
    def get_similarity_distribution(sample_size: int = 1000) -> List[float]:
        """Get distribution of similarity scores for analysis."""
        with get_db_session() as session:
            # Sample random documents with embeddings
            sample_docs = session.query(LegalDocument).filter(
                LegalDocument.embedding.is_not(None)
            ).order_by(func.random()).limit(sample_size).all()
            
            similarities = []
            for i, doc1 in enumerate(sample_docs[:-1]):
                doc2 = sample_docs[i + 1]
                similarity = session.query(
                    (1 - doc1.embedding.cosine_distance(str(doc2.embedding)))
                ).scalar()
                similarities.append(float(similarity))
            
            return similarities