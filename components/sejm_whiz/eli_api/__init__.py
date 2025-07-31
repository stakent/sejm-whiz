"""ELI API integration component for Polish legal documents."""

from .client import EliApiClient, EliApiConfig, get_client, close_client
from .models import (
    LegalDocument,
    Amendment,
    DocumentSearchResult,
    MultiActAmendment,
    DocumentType,
    DocumentStatus,
    AmendmentType,
)
from .parser import LegalTextParser, MultiActAmendmentDetector, DocumentStructure
from .utils import (
    validate_eli_id,
    sanitize_query,
    normalize_document_type,
    extract_legal_references,
    is_amendment_document,
    clean_legal_text,
)

__all__ = [
    # Client
    "EliApiClient",
    "EliApiConfig",
    "get_client",
    "close_client",
    # Models
    "LegalDocument",
    "Amendment",
    "DocumentSearchResult",
    "MultiActAmendment",
    "DocumentType",
    "DocumentStatus",
    "AmendmentType",
    # Parser
    "LegalTextParser",
    "MultiActAmendmentDetector",
    "DocumentStructure",
    # Utils
    "validate_eli_id",
    "sanitize_query",
    "normalize_document_type",
    "extract_legal_references",
    "is_amendment_document",
    "clean_legal_text",
]
