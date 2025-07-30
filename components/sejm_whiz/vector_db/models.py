"""Vector database models - re-exports from database component."""

from sejm_whiz.database.models import (
    Base,
    LegalDocument,
    LegalAmendment,
    CrossReference,
    DocumentEmbedding,
    PredictionModel,
)

__all__ = [
    "Base",
    "LegalDocument",
    "LegalAmendment",
    "CrossReference",
    "DocumentEmbedding",
    "PredictionModel",
]
