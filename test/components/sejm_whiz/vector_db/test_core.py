"""Core tests for vector_db component."""

from sejm_whiz.vector_db import (
    get_vector_db,
    get_vector_session,
    VectorDBConnection,
    get_vector_operations,
    VectorDBOperations,
    get_similarity_search,
    VectorSimilaritySearch,
    DistanceMetric,
    LegalDocument,
    DocumentEmbedding,
    validate_embedding_dimensions,
    normalize_embedding,
    compute_cosine_similarity,
    validate_vector_db_health,
    estimate_index_parameters,
)


def test_imports():
    """Test that all main components can be imported."""
    # Connection classes
    assert VectorDBConnection is not None
    assert get_vector_db is not None
    assert get_vector_session is not None

    # Operations classes
    assert VectorDBOperations is not None
    assert get_vector_operations is not None

    # Similarity search classes
    assert VectorSimilaritySearch is not None
    assert get_similarity_search is not None
    assert DistanceMetric is not None

    # Models
    assert LegalDocument is not None
    assert DocumentEmbedding is not None

    # Utilities
    assert validate_embedding_dimensions is not None
    assert normalize_embedding is not None
    assert compute_cosine_similarity is not None
    assert validate_vector_db_health is not None
    assert estimate_index_parameters is not None


def test_distance_metric_enum():
    """Test DistanceMetric enum values."""
    assert DistanceMetric.COSINE.value == "cosine"
    assert DistanceMetric.L2.value == "l2"
    assert DistanceMetric.INNER_PRODUCT.value == "inner_product"


def test_utility_functions_basic():
    """Test basic utility function behavior."""
    # Test embedding validation
    valid_embedding = [0.1] * 768
    assert validate_embedding_dimensions(valid_embedding) is True

    invalid_embedding = [0.1] * 512
    assert validate_embedding_dimensions(invalid_embedding) is False

    # Test normalization
    embedding = [3.0, 4.0, 0.0]
    normalized = normalize_embedding(embedding)
    assert len(normalized) == 3

    # Test cosine similarity
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    similarity = compute_cosine_similarity(vec1, vec2)
    assert abs(similarity - 1.0) < 1e-6

    # Test index parameter estimation
    params = estimate_index_parameters(1000)
    assert "index_type" in params
    assert "recommendation" in params


def test_singleton_instances():
    """Test that singletons return the same instance."""
    # These should work even without database connection
    db1 = get_vector_db()
    db2 = get_vector_db()
    assert db1 is db2

    ops1 = get_vector_operations()
    ops2 = get_vector_operations()
    assert ops1 is ops2

    search1 = get_similarity_search()
    search2 = get_similarity_search()
    assert search1 is search2


def test_health_check_structure():
    """Test that health check returns expected structure."""
    health = validate_vector_db_health()

    assert isinstance(health, dict)
    assert "status" in health
    assert "connection" in health
    assert "pgvector_extension" in health

    # Status should be one of expected values
    assert health["status"] in ["healthy", "warning", "unhealthy", "error"]
