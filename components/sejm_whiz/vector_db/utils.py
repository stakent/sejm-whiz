"""Utility functions for vector database operations."""

import logging
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


def validate_embedding_dimensions(
    embedding: List[float], expected_dimensions: int = 768
) -> bool:
    """Validate embedding dimensions."""
    if not isinstance(embedding, (list, np.ndarray)):
        logger.error("Embedding must be a list or numpy array")
        return False

    if len(embedding) != expected_dimensions:
        logger.error(f"Expected {expected_dimensions} dimensions, got {len(embedding)}")
        return False

    # Check for valid float values
    try:
        float_embedding = [float(x) for x in embedding]
        # Check for NaN or infinite values
        if any(not np.isfinite(x) for x in float_embedding):
            logger.error("Embedding contains NaN or infinite values")
            return False
    except (ValueError, TypeError):
        logger.error("Embedding contains non-numeric values")
        return False

    return True


def normalize_embedding(embedding: List[float]) -> List[float]:
    """Normalize embedding vector to unit length."""
    try:
        embedding_array = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding_array)

        if norm == 0:
            logger.warning("Zero vector cannot be normalized")
            return embedding

        normalized = embedding_array / norm
        return normalized.tolist()

    except Exception as e:
        logger.error(f"Failed to normalize embedding: {e}")
        return embedding


def compute_cosine_similarity(
    embedding1: List[float], embedding2: List[float]
) -> float:
    """Compute cosine similarity between two embeddings."""
    try:
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

    except Exception as e:
        logger.error(f"Failed to compute cosine similarity: {e}")
        return 0.0


def batch_normalize_embeddings(embeddings: List[List[float]]) -> List[List[float]]:
    """Normalize a batch of embeddings."""
    return [normalize_embedding(emb) for emb in embeddings]


def validate_vector_db_health() -> Dict[str, Any]:
    """Validate vector database health and configuration."""
    from .connection import get_vector_db

    try:
        vector_db = get_vector_db()

        health_status = {
            "connection": vector_db.test_connection(),
            "pgvector_extension": vector_db.test_vector_extension(),
            "vector_dimensions": vector_db.get_vector_dimensions(),
            "status": "healthy",
        }

        if not health_status["connection"]:
            health_status["status"] = "unhealthy"
            health_status["error"] = "Database connection failed"
        elif not health_status["pgvector_extension"]:
            health_status["status"] = "warning"
            health_status["warning"] = "pgvector extension not available"

        return health_status

    except Exception as e:
        logger.error(f"Vector DB health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "connection": False,
            "pgvector_extension": False,
        }


def estimate_index_parameters(
    num_documents: int, target_recall: float = 0.95
) -> Dict[str, Any]:
    """Estimate optimal index parameters based on dataset size."""
    # IVFFlat parameters
    if num_documents < 1000:
        lists = max(1, num_documents // 10)
        index_type = "ivfflat"
    elif num_documents < 100000:
        lists = max(100, num_documents // 1000)
        index_type = "ivfflat"
    else:
        # For larger datasets, consider HNSW
        lists = None
        index_type = "hnsw"

    # Probes should be a fraction of lists for good recall
    probes = max(1, lists // 10) if lists else None

    return {
        "index_type": index_type,
        "lists": lists,
        "probes": probes,
        "estimated_recall": target_recall,
        "recommendation": (
            f"Use {index_type} index"
            + (f" with {lists} lists" if lists else "")
            + (f" and {probes} probes" if probes else "")
        ),
    }
