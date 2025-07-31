"""SQLAlchemy models for legal documents and embeddings."""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    Integer,
    Boolean,
    JSON,
    Index,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class LegalDocument(Base):
    """Legal document with vector embeddings."""

    __tablename__ = "legal_documents"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Document metadata
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    document_type = Column(String(100), nullable=False)  # law, amendment, regulation
    source_url = Column(String(500))
    eli_identifier = Column(String(200), unique=True)  # ELI API identifier

    # Vector embedding (768 dimensions for HerBERT)
    embedding = Column(Vector(768))

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    published_at = Column(DateTime)

    # Legal metadata
    legal_act_type = Column(String(100))  # ustawa, kodeks, rozporządzenie
    legal_domain = Column(String(100))  # civil, criminal, administrative, etc.
    is_amendment = Column(Boolean, default=False)
    affects_multiple_acts = Column(Boolean, default=False)

    # Relationships
    amendments = relationship("LegalAmendment", back_populates="document")
    cross_references_from = relationship(
        "CrossReference",
        foreign_keys="CrossReference.source_document_id",
        back_populates="source_document",
    )
    cross_references_to = relationship(
        "CrossReference",
        foreign_keys="CrossReference.target_document_id",
        back_populates="target_document",
    )

    # Indexes
    __table_args__ = (
        Index("idx_legal_documents_type", "document_type"),
        Index("idx_legal_documents_domain", "legal_domain"),
        Index("idx_legal_documents_published", "published_at"),
        Index("idx_legal_documents_embedding", "embedding", postgresql_using="ivfflat"),
    )


class LegalAmendment(Base):
    """Legal amendments and their relationships."""

    __tablename__ = "legal_amendments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id = Column(
        UUID(as_uuid=True), ForeignKey("legal_documents.id"), nullable=False
    )

    # Amendment details
    amendment_type = Column(String(100))  # change, addition, repeal
    affected_article = Column(String(200))
    affected_paragraph = Column(String(200))
    amendment_text = Column(Text)

    # Multi-act amendment tracking
    affects_multiple_acts = Column(Boolean, default=False)
    omnibus_bill_id = Column(String(200))  # ID for grouping omnibus amendments

    # Impact assessment
    impact_score = Column(Integer, default=0)  # 0-100 scale
    complexity_level = Column(String(50))  # low, medium, high

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    effective_date = Column(DateTime)

    # Relationships
    document = relationship("LegalDocument", back_populates="amendments")

    __table_args__ = (
        Index("idx_amendments_document", "document_id"),
        Index("idx_amendments_omnibus", "omnibus_bill_id"),
        Index("idx_amendments_effective", "effective_date"),
    )


class CrossReference(Base):
    """Cross-references between legal documents."""

    __tablename__ = "cross_references"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    source_document_id = Column(
        UUID(as_uuid=True), ForeignKey("legal_documents.id"), nullable=False
    )
    target_document_id = Column(
        UUID(as_uuid=True), ForeignKey("legal_documents.id"), nullable=False
    )

    # Reference details
    reference_type = Column(String(100))  # citation, amendment, repeal
    reference_text = Column(Text)
    source_article = Column(String(200))
    target_article = Column(String(200))

    # Semantic similarity score
    similarity_score = Column(Integer)  # 0-100 scale

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    source_document = relationship("LegalDocument", foreign_keys=[source_document_id])
    target_document = relationship("LegalDocument", foreign_keys=[target_document_id])

    __table_args__ = (
        Index("idx_cross_refs_source", "source_document_id"),
        Index("idx_cross_refs_target", "target_document_id"),
        Index("idx_cross_refs_type", "reference_type"),
    )


class DocumentEmbedding(Base):
    """Separate table for embedding versions and metadata."""

    __tablename__ = "document_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id = Column(
        UUID(as_uuid=True), ForeignKey("legal_documents.id"), nullable=False
    )

    # Embedding metadata
    embedding = Column(Vector(768))
    model_name = Column(String(100))  # herbert-klej-cased-v1
    model_version = Column(String(50))
    embedding_method = Column(String(100))  # bag_of_embeddings, mean_pooling

    # Processing metadata
    token_count = Column(Integer)
    chunk_size = Column(Integer)
    preprocessing_version = Column(String(50))

    # Quality metrics
    confidence_score = Column(Integer)  # 0-100 scale

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_embeddings_document", "document_id"),
        Index("idx_embeddings_model", "model_name", "model_version"),
        Index("idx_embeddings_vector", "embedding", postgresql_using="ivfflat"),
    )


class PredictionModel(Base):
    """Model metadata and performance tracking."""

    __tablename__ = "prediction_models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Model details
    model_name = Column(String(200), nullable=False)
    model_type = Column(String(100))  # embedding_similarity, ensemble, etc.
    model_version = Column(String(50))

    # Performance metrics
    accuracy_score = Column(Integer)  # 0-100 scale
    precision_score = Column(Integer)
    recall_score = Column(Integer)
    f1_score = Column(Integer)

    # Training metadata
    training_data_size = Column(Integer)
    training_date = Column(DateTime)
    hyperparameters = Column(JSON)

    # Status
    is_active = Column(Boolean, default=False)
    deployment_env = Column(String(50))  # local, k3s, production

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_models_active", "is_active"),
        Index("idx_models_type", "model_type"),
        Index("idx_models_env", "deployment_env"),
    )
