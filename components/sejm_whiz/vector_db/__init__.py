"""Vector database component for PostgreSQL + pgvector operations."""

from .connection import get_vector_db, get_vector_session, VectorDBConnection
from .operations import get_vector_operations, VectorDBOperations
from .embeddings import get_similarity_search, VectorSimilaritySearch, DistanceMetric
from .models import (
    LegalDocument,
    LegalAmendment,
    CrossReference,
    DocumentEmbedding,
    PredictionModel,
)
from .utils import (
    validate_embedding_dimensions,
    normalize_embedding,
    compute_cosine_similarity,
    validate_vector_db_health,
    estimate_index_parameters,
)

__all__ = [
    # Connection
    "get_vector_db",
    "get_vector_session",
    "VectorDBConnection",
    # Operations
    "get_vector_operations",
    "VectorDBOperations",
    # Similarity Search
    "get_similarity_search",
    "VectorSimilaritySearch",
    "DistanceMetric",
    # Models
    "LegalDocument",
    "LegalAmendment",
    "CrossReference",
    "DocumentEmbedding",
    "PredictionModel",
    # Utils
    "validate_embedding_dimensions",
    "normalize_embedding",
    "compute_cosine_similarity",
    "validate_vector_db_health",
    "estimate_index_parameters",
]
