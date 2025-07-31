"""Tests for vector_db utils module."""

import numpy as np
from unittest.mock import Mock, patch

from sejm_whiz.vector_db.utils import (
    validate_embedding_dimensions,
    normalize_embedding,
    compute_cosine_similarity,
    batch_normalize_embeddings,
    validate_vector_db_health,
    estimate_index_parameters,
)


class TestEmbeddingValidation:
    """Test embedding validation functions."""

    def test_validate_embedding_dimensions_valid(self):
        """Test validation with valid embedding."""
        embedding = [0.1] * 768
        result = validate_embedding_dimensions(embedding, 768)
        assert result is True

    def test_validate_embedding_dimensions_wrong_size(self):
        """Test validation with wrong dimensions."""
        embedding = [0.1] * 512  # Wrong size
        result = validate_embedding_dimensions(embedding, 768)
        assert result is False

    def test_validate_embedding_dimensions_not_list(self):
        """Test validation with non-list input."""
        embedding = "not a list"
        result = validate_embedding_dimensions(embedding, 768)
        assert result is False

    def test_validate_embedding_dimensions_with_numpy(self):
        """Test validation with numpy array."""
        embedding = np.array([0.1] * 768)
        result = validate_embedding_dimensions(embedding, 768)
        assert result is True

    def test_validate_embedding_dimensions_with_nan(self):
        """Test validation with NaN values."""
        embedding = [0.1] * 767 + [float("nan")]
        result = validate_embedding_dimensions(embedding, 768)
        assert result is False

    def test_validate_embedding_dimensions_with_inf(self):
        """Test validation with infinite values."""
        embedding = [0.1] * 767 + [float("inf")]
        result = validate_embedding_dimensions(embedding, 768)
        assert result is False

    def test_validate_embedding_dimensions_non_numeric(self):
        """Test validation with non-numeric values."""
        embedding = [0.1] * 767 + ["not_a_number"]
        result = validate_embedding_dimensions(embedding, 768)
        assert result is False


class TestEmbeddingNormalization:
    """Test embedding normalization functions."""

    def test_normalize_embedding_valid(self):
        """Test normalizing valid embedding."""
        embedding = [3.0, 4.0, 0.0]  # Magnitude = 5
        result = normalize_embedding(embedding)
        expected = [0.6, 0.8, 0.0]

        assert len(result) == len(expected)
        for i, val in enumerate(result):
            assert abs(val - expected[i]) < 1e-6

    def test_normalize_embedding_zero_vector(self):
        """Test normalizing zero vector."""
        embedding = [0.0, 0.0, 0.0]
        result = normalize_embedding(embedding)
        assert result == embedding  # Should return original

    def test_normalize_embedding_unit_vector(self):
        """Test normalizing already unit vector."""
        embedding = [1.0, 0.0, 0.0]
        result = normalize_embedding(embedding)

        # Should remain approximately the same
        assert abs(result[0] - 1.0) < 1e-6
        assert abs(result[1] - 0.0) < 1e-6
        assert abs(result[2] - 0.0) < 1e-6

    def test_normalize_embedding_error_handling(self):
        """Test normalization error handling."""
        embedding = ["not", "numeric", "values"]
        result = normalize_embedding(embedding)
        assert result == embedding  # Should return original on error

    def test_batch_normalize_embeddings(self):
        """Test batch normalization."""
        embeddings = [[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]]

        result = batch_normalize_embeddings(embeddings)

        assert len(result) == 3
        # First embedding should be normalized
        assert abs(result[0][0] - 0.6) < 1e-6
        assert abs(result[0][1] - 0.8) < 1e-6


class TestSimilarityComputation:
    """Test similarity computation functions."""

    def test_compute_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        embedding1 = [1.0, 2.0, 3.0]
        embedding2 = [1.0, 2.0, 3.0]

        result = compute_cosine_similarity(embedding1, embedding2)
        assert abs(result - 1.0) < 1e-6

    def test_compute_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0]

        result = compute_cosine_similarity(embedding1, embedding2)
        assert abs(result - 0.0) < 1e-6

    def test_compute_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors."""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [-1.0, 0.0, 0.0]

        result = compute_cosine_similarity(embedding1, embedding2)
        assert abs(result - (-1.0)) < 1e-6

    def test_compute_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector."""
        embedding1 = [1.0, 2.0, 3.0]
        embedding2 = [0.0, 0.0, 0.0]

        result = compute_cosine_similarity(embedding1, embedding2)
        assert result == 0.0

    def test_compute_cosine_similarity_error_handling(self):
        """Test cosine similarity error handling."""
        embedding1 = ["not", "numeric"]
        embedding2 = [1.0, 2.0]

        result = compute_cosine_similarity(embedding1, embedding2)
        assert result == 0.0


class TestHealthValidation:
    """Test health validation functions."""

    @patch("sejm_whiz.vector_db.connection.get_vector_db")
    def test_validate_vector_db_health_success(self, mock_get_vector_db):
        """Test successful health validation."""
        mock_vector_db = Mock()
        mock_vector_db.test_connection.return_value = True
        mock_vector_db.test_vector_extension.return_value = True
        mock_vector_db.get_vector_dimensions.return_value = 768
        mock_get_vector_db.return_value = mock_vector_db

        result = validate_vector_db_health()

        expected = {
            "connection": True,
            "pgvector_extension": True,
            "vector_dimensions": 768,
            "status": "healthy",
        }

        assert result == expected

    @patch("sejm_whiz.vector_db.connection.get_vector_db")
    def test_validate_vector_db_health_connection_failed(self, mock_get_vector_db):
        """Test health validation with connection failure."""
        mock_vector_db = Mock()
        mock_vector_db.test_connection.return_value = False
        mock_vector_db.test_vector_extension.return_value = True
        mock_vector_db.get_vector_dimensions.return_value = 768
        mock_get_vector_db.return_value = mock_vector_db

        result = validate_vector_db_health()

        assert result["status"] == "unhealthy"
        assert result["connection"] is False
        assert "error" in result

    @patch("sejm_whiz.vector_db.connection.get_vector_db")
    def test_validate_vector_db_health_no_pgvector(self, mock_get_vector_db):
        """Test health validation without pgvector extension."""
        mock_vector_db = Mock()
        mock_vector_db.test_connection.return_value = True
        mock_vector_db.test_vector_extension.return_value = False
        mock_vector_db.get_vector_dimensions.return_value = 768
        mock_get_vector_db.return_value = mock_vector_db

        result = validate_vector_db_health()

        assert result["status"] == "warning"
        assert result["pgvector_extension"] is False
        assert "warning" in result

    @patch("sejm_whiz.vector_db.connection.get_vector_db")
    def test_validate_vector_db_health_exception(self, mock_get_vector_db):
        """Test health validation with exception."""
        mock_get_vector_db.side_effect = Exception("Database error")

        result = validate_vector_db_health()

        assert result["status"] == "error"
        assert result["connection"] is False
        assert result["pgvector_extension"] is False
        assert "error" in result


class TestIndexParameterEstimation:
    """Test index parameter estimation functions."""

    def test_estimate_index_parameters_small_dataset(self):
        """Test index parameters for small dataset."""
        result = estimate_index_parameters(500)

        assert result["index_type"] == "ivfflat"
        assert result["lists"] == 50  # 500 // 10
        assert result["probes"] == 5  # 50 // 10
        assert result["estimated_recall"] == 0.95
        assert "recommendation" in result

    def test_estimate_index_parameters_medium_dataset(self):
        """Test index parameters for medium dataset."""
        result = estimate_index_parameters(50000)

        assert result["index_type"] == "ivfflat"
        assert result["lists"] == 100  # max(100, 50000 // 1000)
        assert result["probes"] == 10
        assert result["estimated_recall"] == 0.95

    def test_estimate_index_parameters_large_dataset(self):
        """Test index parameters for large dataset."""
        result = estimate_index_parameters(200000)

        assert result["index_type"] == "hnsw"
        assert result["lists"] is None
        assert result["probes"] is None
        assert result["estimated_recall"] == 0.95

    def test_estimate_index_parameters_very_small_dataset(self):
        """Test index parameters for very small dataset."""
        result = estimate_index_parameters(5)

        assert result["index_type"] == "ivfflat"
        assert result["lists"] == 1  # max(1, 5 // 10)
        assert result["probes"] == 1

    def test_estimate_index_parameters_custom_recall(self):
        """Test index parameters with custom target recall."""
        result = estimate_index_parameters(1000, target_recall=0.99)

        assert result["estimated_recall"] == 0.99
        assert result["index_type"] == "ivfflat"
