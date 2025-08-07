"""Document ingestion component for fetching and processing Polish legal documents."""

from .config import DocumentIngestionConfig, get_ingestion_config
from sejm_whiz.eli_api.client import (
    EliApiClient as ELIClient,
    get_client as get_eli_client,
    EliApiError as ELIApiError,
)
from .text_processor import TextProcessor, ProcessedDocument, LegalStructure
from .ingestion_pipeline import (
    DocumentIngestionPipeline,
    get_ingestion_pipeline,
    IngestionPipelineError,
)

__all__ = [
    # Configuration
    "DocumentIngestionConfig",
    "get_ingestion_config",
    # ELI API client
    "ELIClient",
    "get_eli_client",
    "ELIApiError",
    # Text processing
    "TextProcessor",
    "ProcessedDocument",
    "LegalStructure",
    # Pipeline orchestration
    "DocumentIngestionPipeline",
    "get_ingestion_pipeline",
    "IngestionPipelineError",
]
