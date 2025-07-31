"""Tests for similarity calculation functionality."""

import pytest
import numpy as np

from sejm_whiz.embeddings.similarity import (
    SimilarityCalculator,
    get_similarity_calculator,
    SimilarityResult,
    SimilarityMatrix,
    cosine_similarity,
    euclidean_distance,
    find_most_similar_embeddings,
)


class TestSimilarityCalculator:
    """Test similarity calculator."""

    @pytest.fixture
    def calculator(self):
        """Create similarity calculator instance."""
        return SimilarityCalculator()

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        np.random.seed(42)  # For reproducible tests
        return [np.random.randn(768), np.random.randn(768), np.random.randn(768)]

    def test_initialization(self, calculator):
        """Test calculator initialization."""
        assert len(calculator.supported_methods) > 0
        assert "cosine" in calculator.supported_methods
        assert "euclidean" in calculator.supported_methods
        assert "manhattan" in calculator.supported_methods

    def test_cosine_similarity_identical(self, calculator):
        """Test cosine similarity with identical embeddings."""
        embedding = np.array([1.0, 2.0, 3.0])

        result = calculator.cosine_similarity(embedding, embedding)

        assert isinstance(result, SimilarityResult)
        assert result.method == "cosine"
        assert abs(result.similarity_score - 1.0) < 1e-6
        assert abs(result.distance - 0.0) < 1e-6
        assert "dot_product" in result.metadata

    def test_cosine_similarity_orthogonal(self, calculator):
        """Test cosine similarity with orthogonal embeddings."""
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([0.0, 1.0, 0.0])

        result = calculator.cosine_similarity(embedding1, embedding2)

        assert abs(result.similarity_score - 0.0) < 1e-6
        assert abs(result.distance - 1.0) < 1e-6

    def test_cosine_similarity_opposite(self, calculator):
        """Test cosine similarity with opposite embeddings."""
        embedding1 = np.array([1.0, 2.0, 3.0])
        embedding2 = np.array([-1.0, -2.0, -3.0])

        result = calculator.cosine_similarity(embedding1, embedding2)

        assert abs(result.similarity_score - (-1.0)) < 1e-6
        assert abs(result.distance - 2.0) < 1e-6

    def test_cosine_similarity_zero_norm(self, calculator):
        """Test cosine similarity with zero norm embedding."""
        embedding1 = np.array([1.0, 2.0, 3.0])
        embedding2 = np.array([0.0, 0.0, 0.0])

        result = calculator.cosine_similarity(embedding1, embedding2)

        assert result.similarity_score == 0.0
        assert result.distance == 1.0

    def test_euclidean_similarity(self, calculator):
        """Test Euclidean distance-based similarity."""
        embedding1 = np.array([1.0, 2.0, 3.0])
        embedding2 = np.array([1.0, 2.0, 3.0])  # Identical

        result = calculator.euclidean_similarity(embedding1, embedding2)

        assert isinstance(result, SimilarityResult)
        assert result.method == "euclidean"
        assert result.distance == 0.0  # Identical embeddings
        assert result.similarity_score > 0.9  # High similarity for identical

    def test_manhattan_similarity(self, calculator):
        """Test Manhattan distance-based similarity."""
        embedding1 = np.array([1.0, 2.0, 3.0])
        embedding2 = np.array([2.0, 3.0, 4.0])  # Each component differs by 1

        result = calculator.manhattan_similarity(embedding1, embedding2)

        assert isinstance(result, SimilarityResult)
        assert result.method == "manhattan"
        assert result.distance == 3.0  # Sum of absolute differences
        assert "avg_component_diff" in result.metadata

    def test_dot_product_similarity(self, calculator):
        """Test dot product similarity."""
        # Normalized embeddings
        embedding1 = np.array([1.0, 0.0, 0.0])  # Already normalized
        embedding2 = np.array([0.707, 0.707, 0.0])  # Normalized

        result = calculator.dot_product_similarity(embedding1, embedding2)

        assert isinstance(result, SimilarityResult)
        assert result.method == "dot_product"
        assert abs(result.similarity_score - 0.707) < 0.01
        assert "embedding1_is_normalized" in result.metadata
        assert "embedding2_is_normalized" in result.metadata

    def test_legal_weighted_similarity(self, calculator):
        """Test legal weighted similarity."""
        embedding1 = np.random.randn(768)
        embedding2 = np.random.randn(768)

        result = calculator.legal_weighted_similarity(embedding1, embedding2)

        assert isinstance(result, SimilarityResult)
        assert result.method == "legal_weighted"
        assert -1.0 <= result.similarity_score <= 1.0
        assert "weights_applied" in result.metadata
        assert result.metadata["weights_applied"] == True

    def test_legal_weighted_similarity_custom_weights(self, calculator):
        """Test legal weighted similarity with custom weights."""
        embedding1 = np.array([1.0, 2.0, 3.0])
        embedding2 = np.array([1.0, 2.0, 3.0])
        weights = np.array([2.0, 1.0, 0.5])  # Different weights

        result = calculator.legal_weighted_similarity(embedding1, embedding2, weights)

        assert result.similarity_score > 0.9  # Should be high for identical embeddings

    def test_calculate_similarity_cosine(self, calculator):
        """Test generic similarity calculation with cosine method."""
        embedding1 = np.array([1.0, 2.0, 3.0])
        embedding2 = np.array([2.0, 3.0, 4.0])

        result = calculator.calculate_similarity(embedding1, embedding2, "cosine")

        assert result.method == "cosine"
        assert isinstance(result.similarity_score, float)

    def test_calculate_similarity_unsupported_method(self, calculator):
        """Test similarity calculation with unsupported method."""
        embedding1 = np.array([1.0, 2.0, 3.0])
        embedding2 = np.array([2.0, 3.0, 4.0])

        with pytest.raises(ValueError, match="Unsupported method"):
            calculator.calculate_similarity(embedding1, embedding2, "unsupported")

    def test_calculate_pairwise_similarities(self, calculator, sample_embeddings):
        """Test pairwise similarity calculation."""
        labels = ["doc1", "doc2", "doc3"]

        result = calculator.calculate_pairwise_similarities(
            sample_embeddings, labels, "cosine"
        )

        assert isinstance(result, SimilarityMatrix)
        assert result.matrix.shape == (3, 3)
        assert result.labels == labels
        assert result.method == "cosine"

        # Diagonal should be 1.0 (self-similarity)
        np.testing.assert_array_almost_equal(np.diag(result.matrix), [1.0, 1.0, 1.0])

        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(result.matrix, result.matrix.T)

        # Check metadata
        assert "n_embeddings" in result.metadata
        assert result.metadata["n_embeddings"] == 3

    def test_calculate_pairwise_similarities_no_labels(
        self, calculator, sample_embeddings
    ):
        """Test pairwise similarities without labels."""
        result = calculator.calculate_pairwise_similarities(sample_embeddings)

        assert len(result.labels) == 3
        assert all(label.startswith("embedding_") for label in result.labels)

    def test_find_most_similar(self, calculator, sample_embeddings):
        """Test finding most similar embeddings."""
        query_embedding = sample_embeddings[0]
        candidates = sample_embeddings[1:]
        labels = ["candidate1", "candidate2"]

        results = calculator.find_most_similar(
            query_embedding, candidates, labels, "cosine", top_k=2
        )

        assert len(results) == 2
        assert all(len(result) == 3 for result in results)  # (index, score, label)

        # Results should be sorted by similarity (descending)
        scores = [result[1] for result in results]
        assert scores == sorted(scores, reverse=True)

    def test_find_most_similar_no_labels(self, calculator, sample_embeddings):
        """Test finding most similar without labels."""
        query_embedding = sample_embeddings[0]
        candidates = sample_embeddings[1:]

        results = calculator.find_most_similar(query_embedding, candidates, top_k=1)

        assert len(results) == 1
        assert results[0][2].startswith("candidate_")  # Auto-generated label

    def test_calculate_similarity_statistics(self, calculator):
        """Test similarity statistics calculation."""
        similarities = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

        stats = calculator.calculate_similarity_statistics(similarities)

        assert isinstance(stats, dict)
        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "q25" in stats
        assert "q75" in stats
        assert "count" in stats

        assert stats["mean"] == 0.5
        assert stats["median"] == 0.5
        assert stats["min"] == 0.1
        assert stats["max"] == 0.9
        assert stats["count"] == 9

    def test_calculate_similarity_statistics_empty(self, calculator):
        """Test similarity statistics with empty list."""
        stats = calculator.calculate_similarity_statistics([])
        assert stats == {}

    def test_cluster_by_similarity(self, calculator):
        """Test clustering by similarity threshold."""
        # Create embeddings where some are similar
        embeddings = [
            np.array([1.0, 0.0, 0.0]),  # Group 1
            np.array([0.9, 0.1, 0.0]),  # Group 1 (similar to first)
            np.array([0.0, 1.0, 0.0]),  # Group 2
            np.array([0.0, 0.9, 0.1]),  # Group 2 (similar to third)
        ]

        clusters = calculator.cluster_by_similarity(
            embeddings, threshold=0.8, method="cosine"
        )

        assert isinstance(clusters, list)
        assert len(clusters) >= 1  # At least one cluster

        # All embeddings should be assigned to a cluster
        all_indices = set()
        for cluster in clusters:
            all_indices.update(cluster)
        assert all_indices == {0, 1, 2, 3}

    def test_generate_default_legal_weights(self, calculator):
        """Test default legal weights generation."""
        weights = calculator._generate_default_legal_weights(768)

        assert len(weights) == 768
        assert all(w > 0 for w in weights)
        # Middle dimensions should have higher weights
        middle_start = 768 // 4
        middle_end = 3 * 768 // 4
        assert np.mean(weights[middle_start:middle_end]) > np.mean(
            weights[:middle_start]
        )


def test_get_similarity_calculator():
    """Test global calculator instance."""
    calc1 = get_similarity_calculator()
    calc2 = get_similarity_calculator()

    # Should return same instance (singleton pattern)
    assert calc1 == calc2
    assert isinstance(calc1, SimilarityCalculator)


def test_cosine_similarity_convenience_function():
    """Test cosine similarity convenience function."""
    embedding1 = np.array([1.0, 2.0, 3.0])
    embedding2 = np.array([1.0, 2.0, 3.0])

    similarity = cosine_similarity(embedding1, embedding2)

    assert isinstance(similarity, float)
    assert abs(similarity - 1.0) < 1e-6  # Identical embeddings


def test_euclidean_distance_convenience_function():
    """Test Euclidean distance convenience function."""
    embedding1 = np.array([1.0, 2.0, 3.0])
    embedding2 = np.array([1.0, 2.0, 3.0])

    distance = euclidean_distance(embedding1, embedding2)

    assert isinstance(distance, float)
    assert distance == 0.0  # Identical embeddings


def test_find_most_similar_embeddings_convenience_function():
    """Test find most similar convenience function."""
    query = np.array([1.0, 0.0, 0.0])
    candidates = [
        np.array([0.9, 0.1, 0.0]),  # Very similar
        np.array([0.0, 1.0, 0.0]),  # Orthogonal
        np.array([0.8, 0.2, 0.0]),  # Similar
    ]

    results = find_most_similar_embeddings(query, candidates, top_k=2)

    assert len(results) == 2
    assert all(len(result) == 2 for result in results)  # (index, score)

    # First result should be most similar (index 0)
    assert results[0][0] == 0


class TestSimilarityResult:
    """Test SimilarityResult data class."""

    def test_creation(self):
        """Test result creation."""
        result = SimilarityResult(
            similarity_score=0.85,
            distance=0.15,
            method="cosine",
            metadata={"test": "value"},
        )

        assert result.similarity_score == 0.85
        assert result.distance == 0.15
        assert result.method == "cosine"
        assert result.metadata["test"] == "value"


class TestSimilarityMatrix:
    """Test SimilarityMatrix data class."""

    def test_creation(self):
        """Test matrix creation."""
        matrix = np.array([[1.0, 0.8, 0.6], [0.8, 1.0, 0.7], [0.6, 0.7, 1.0]])
        labels = ["doc1", "doc2", "doc3"]

        sim_matrix = SimilarityMatrix(
            matrix=matrix, labels=labels, method="cosine", metadata={"n_embeddings": 3}
        )

        assert sim_matrix.matrix.shape == (3, 3)
        assert sim_matrix.labels == labels
        assert sim_matrix.method == "cosine"
        assert sim_matrix.metadata["n_embeddings"] == 3
