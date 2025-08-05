"""Pydantic models for data processor pipeline."""

from datetime import date
from typing import List, Optional, Any, Union
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


class DateRange(BaseModel):
    """Date range for filtering data."""

    start: date
    end: date


class PipelineInput(BaseModel):
    """Input data for the pipeline."""

    session_id: Optional[str] = None
    date_range: Optional[DateRange] = None
    category: Optional[str] = None
    document_ids: Optional[List[str]] = Field(default_factory=list)


class ProcessedDocument(BaseModel):
    """Processed document with cleaned text."""

    processed_content: str
    # Allow additional fields from original document
    model_config = ConfigDict(extra="allow")


class DocumentWithEmbedding(BaseModel):
    """Document with generated embedding."""

    processed_content: str
    embedding: Any  # BagEmbedding object
    # Allow additional fields from original document
    model_config = ConfigDict(extra="allow")


class SejmIngestionData(BaseModel):
    """Data after Sejm ingestion step."""

    sejm_proceedings: List[Any]  # List of ProceedingSitting objects
    step_completed: str = "sejm_ingestion"
    # Include original input data
    session_id: Optional[str] = None
    date_range: Optional[DateRange] = None
    category: Optional[str] = None
    document_ids: Optional[List[str]] = Field(default_factory=list)


class EliIngestionData(BaseModel):
    """Data after ELI ingestion step."""

    eli_documents: List[Any]  # List of LegalDocument objects
    step_completed: str = "eli_ingestion"
    # Include previous data
    sejm_proceedings: Optional[List[Any]] = None
    session_id: Optional[str] = None
    date_range: Optional[DateRange] = None
    category: Optional[str] = None
    document_ids: Optional[List[str]] = Field(default_factory=list)


class TextProcessingData(BaseModel):
    """Data after text processing step."""

    processed_sejm_proceedings: Optional[List[ProcessedDocument]] = None
    processed_eli_documents: Optional[List[ProcessedDocument]] = None
    step_completed: str = "text_processing"
    # Include previous data
    sejm_proceedings: Optional[List[Any]] = None
    eli_documents: Optional[List[Any]] = None
    session_id: Optional[str] = None
    date_range: Optional[DateRange] = None
    category: Optional[str] = None
    document_ids: Optional[List[str]] = Field(default_factory=list)


class EmbeddingGenerationData(BaseModel):
    """Data after embedding generation step."""

    sejm_proceedings_embeddings: Optional[List[DocumentWithEmbedding]] = None
    eli_documents_embeddings: Optional[List[DocumentWithEmbedding]] = None
    step_completed: str = "embedding_generation"
    # Include previous data
    processed_sejm_proceedings: Optional[List[ProcessedDocument]] = None
    processed_eli_documents: Optional[List[ProcessedDocument]] = None
    sejm_proceedings: Optional[List[Any]] = None
    eli_documents: Optional[List[Any]] = None
    session_id: Optional[str] = None
    date_range: Optional[DateRange] = None
    category: Optional[str] = None
    document_ids: Optional[List[str]] = Field(default_factory=list)


class DatabaseStorageData(BaseModel):
    """Data after database storage step."""

    stored_sejm_proceedings: Optional[List[UUID]] = None
    stored_eli_documents: Optional[List[UUID]] = None
    step_completed: str = "database_storage"
    # Include previous data
    sejm_proceedings_embeddings: Optional[List[DocumentWithEmbedding]] = None
    eli_documents_embeddings: Optional[List[DocumentWithEmbedding]] = None
    processed_sejm_proceedings: Optional[List[ProcessedDocument]] = None
    processed_eli_documents: Optional[List[ProcessedDocument]] = None
    sejm_proceedings: Optional[List[Any]] = None
    eli_documents: Optional[List[Any]] = None
    session_id: Optional[str] = None
    date_range: Optional[DateRange] = None
    category: Optional[str] = None
    document_ids: Optional[List[str]] = Field(default_factory=list)


# Union type for all pipeline data stages
PipelineData = Union[
    PipelineInput,
    SejmIngestionData,
    EliIngestionData,
    TextProcessingData,
    EmbeddingGenerationData,
    DatabaseStorageData,
]
