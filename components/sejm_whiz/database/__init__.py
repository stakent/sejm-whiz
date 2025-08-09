"""Database component for sejm-whiz project.

This component provides database connectivity, models, and operations
for legal documents with vector embeddings support.
"""

from .config import DatabaseConfig, get_database_config, get_db_config
from .connection import (
    DatabaseManager,
    get_database_manager,
    init_database,
    get_db_session,
    check_database_health,
)
from .models import (
    Base,
    LegalDocument,
    LegalAmendment,
    CrossReference,
    DocumentEmbedding,
    PredictionModel,
)
from .operations import (
    DocumentOperations,
    VectorOperations,
    AmendmentOperations,
    CrossReferenceOperations,
    EmbeddingOperations,
    AnalyticsOperations,
)

__all__ = [
    # Configuration
    "DatabaseConfig",
    "get_database_config",
    "get_db_config",
    # Connection management
    "DatabaseManager",
    "get_database_manager",
    "init_database",
    "get_db_session",
    "check_database_health",
    # Models
    "Base",
    "LegalDocument",
    "LegalAmendment",
    "CrossReference",
    "DocumentEmbedding",
    "PredictionModel",
    # Operations
    "DocumentOperations",
    "VectorOperations",
    "AmendmentOperations",
    "CrossReferenceOperations",
    "EmbeddingOperations",
    "AnalyticsOperations",
]
