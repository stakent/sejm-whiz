"""Similarity calculations for embeddings and legal documents."""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """Result of similarity calculation."""

    similarity_score: float
    distance: float
    method: str
    metadata: Dict[str, Any]


@dataclass
class SimilarityMatrix:
    """Matrix of pairwise similarities."""

    matrix: np.ndarray
    labels: List[str]
    method: str
    metadata: Dict[str, Any]


class SimilarityCalculator:
    """Calculate similarities between embeddings and documents."""

    def __init__(self):
        """Initialize similarity calculator."""
        self.supported_methods = [
            "cosine",
            "euclidean",
            "manhattan",
            "dot_product",
            "jaccard",
            "legal_weighted",
            "semantic_similarity",
        ]

    def cosine_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> SimilarityResult:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            SimilarityResult with cosine similarity score
        """
        # Ensure embeddings are numpy arrays
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)

        # Calculate cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            similarity = 0.0
        else:
            similarity = dot_product / (norm1 * norm2)

        # Ensure similarity is in valid range [-1, 1]
        similarity = np.clip(similarity, -1.0, 1.0)

        # Convert to distance (0 = identical, 2 = opposite)
        distance = 1.0 - similarity

        return SimilarityResult(
            similarity_score=float(similarity),
            distance=float(distance),
            method="cosine",
            metadata={
                "embedding1_norm": float(norm1),
                "embedding2_norm": float(norm2),
                "dot_product": float(dot_product),
            },
        )

    def euclidean_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> SimilarityResult:
        """
        Calculate Euclidean distance-based similarity.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            SimilarityResult with Euclidean-based similarity
        """
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)

        # Calculate Euclidean distance
        distance = np.linalg.norm(emb1 - emb2)

        # Convert distance to similarity (0 = identical, larger = more different)
        # Use exponential decay to convert distance to similarity [0, 1]
        similarity = np.exp(-distance / len(emb1))

        return SimilarityResult(
            similarity_score=float(similarity),
            distance=float(distance),
            method="euclidean",
            metadata={
                "raw_distance": float(distance),
                "normalized_distance": float(distance / len(emb1)),
            },
        )

    def manhattan_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> SimilarityResult:
        """
        Calculate Manhattan distance-based similarity.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            SimilarityResult with Manhattan-based similarity
        """
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)

        # Calculate Manhattan distance
        distance = np.sum(np.abs(emb1 - emb2))

        # Convert to similarity
        similarity = np.exp(-distance / (2 * len(emb1)))

        return SimilarityResult(
            similarity_score=float(similarity),
            distance=float(distance),
            method="manhattan",
            metadata={
                "raw_distance": float(distance),
                "avg_component_diff": float(distance / len(emb1)),
            },
        )

    def dot_product_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> SimilarityResult:
        """
        Calculate dot product similarity (for normalized embeddings).

        Args:
            embedding1: First embedding vector (should be normalized)
            embedding2: Second embedding vector (should be normalized)

        Returns:
            SimilarityResult with dot product similarity
        """
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)

        # Calculate dot product
        dot_product = np.dot(emb1, emb2)

        # For normalized vectors, dot product equals cosine similarity
        distance = 1.0 - dot_product

        return SimilarityResult(
            similarity_score=float(dot_product),
            distance=float(distance),
            method="dot_product",
            metadata={
                "dot_product": float(dot_product),
                "embedding1_is_normalized": bool(np.isclose(np.linalg.norm(emb1), 1.0)),
                "embedding2_is_normalized": bool(np.isclose(np.linalg.norm(emb2), 1.0)),
            },
        )

    def legal_weighted_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        legal_weights: Optional[np.ndarray] = None,
    ) -> SimilarityResult:
        """
        Calculate similarity with legal domain weighting.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            legal_weights: Optional weights for legal importance

        Returns:
            SimilarityResult with legal-weighted similarity
        """
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)

        if legal_weights is None:
            # Default weighting emphasizes certain dimensions
            legal_weights = self._generate_default_legal_weights(len(emb1))

        # Apply weights
        weighted_emb1 = emb1 * legal_weights
        weighted_emb2 = emb2 * legal_weights

        # Calculate cosine similarity on weighted embeddings
        dot_product = np.dot(weighted_emb1, weighted_emb2)
        norm1 = np.linalg.norm(weighted_emb1)
        norm2 = np.linalg.norm(weighted_emb2)

        if norm1 == 0 or norm2 == 0:
            similarity = 0.0
        else:
            similarity = dot_product / (norm1 * norm2)

        similarity = np.clip(similarity, -1.0, 1.0)
        distance = 1.0 - similarity

        return SimilarityResult(
            similarity_score=float(similarity),
            distance=float(distance),
            method="legal_weighted",
            metadata={
                "weights_applied": True,
                "weight_sum": float(np.sum(legal_weights)),
                "weight_mean": float(np.mean(legal_weights)),
            },
        )

    def calculate_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray, method: str = "cosine"
    ) -> SimilarityResult:
        """
        Calculate similarity using specified method.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            method: Similarity method to use

        Returns:
            SimilarityResult with calculated similarity
        """
        if method not in self.supported_methods:
            raise ValueError(
                f"Unsupported method: {method}. Supported: {self.supported_methods}"
            )

        if method == "cosine":
            return self.cosine_similarity(embedding1, embedding2)
        elif method == "euclidean":
            return self.euclidean_similarity(embedding1, embedding2)
        elif method == "manhattan":
            return self.manhattan_similarity(embedding1, embedding2)
        elif method == "dot_product":
            return self.dot_product_similarity(embedding1, embedding2)
        elif method == "legal_weighted":
            return self.legal_weighted_similarity(embedding1, embedding2)
        else:
            # Default to cosine
            return self.cosine_similarity(embedding1, embedding2)

    def calculate_pairwise_similarities(
        self,
        embeddings: List[np.ndarray],
        labels: Optional[List[str]] = None,
        method: str = "cosine",
    ) -> SimilarityMatrix:
        """
        Calculate pairwise similarities between all embeddings.

        Args:
            embeddings: List of embedding vectors
            labels: Optional labels for embeddings
            method: Similarity method to use

        Returns:
            SimilarityMatrix with pairwise similarities
        """
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))

        # Generate default labels if not provided
        if labels is None:
            labels = [f"embedding_{i}" for i in range(n)]

        # Calculate pairwise similarities
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    result = self.calculate_similarity(
                        embeddings[i], embeddings[j], method
                    )
                    similarity_matrix[i, j] = result.similarity_score
                    similarity_matrix[j, i] = result.similarity_score

        return SimilarityMatrix(
            matrix=similarity_matrix,
            labels=labels,
            method=method,
            metadata={
                "n_embeddings": n,
                "avg_similarity": float(
                    np.mean(similarity_matrix[np.triu_indices(n, k=1)])
                ),
                "min_similarity": float(
                    np.min(similarity_matrix[np.triu_indices(n, k=1)])
                ),
                "max_similarity": float(
                    np.max(similarity_matrix[np.triu_indices(n, k=1)])
                ),
            },
        )

    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        candidate_labels: Optional[List[str]] = None,
        method: str = "cosine",
        top_k: int = 5,
    ) -> List[Tuple[int, float, str]]:
        """
        Find most similar embeddings to a query embedding.

        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embeddings
            candidate_labels: Optional labels for candidates
            method: Similarity method to use
            top_k: Number of top results to return

        Returns:
            List of (index, similarity_score, label) tuples
        """
        if candidate_labels is None:
            candidate_labels = [
                f"candidate_{i}" for i in range(len(candidate_embeddings))
            ]

        # Calculate similarities
        similarities = []
        for i, candidate in enumerate(candidate_embeddings):
            result = self.calculate_similarity(query_embedding, candidate, method)
            similarities.append((i, result.similarity_score, candidate_labels[i]))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def calculate_similarity_statistics(
        self, similarities: List[float]
    ) -> Dict[str, float]:
        """
        Calculate statistics for a list of similarity scores.

        Args:
            similarities: List of similarity scores

        Returns:
            Dictionary with statistics
        """
        if not similarities:
            return {}

        similarities_array = np.array(similarities)

        return {
            "mean": float(np.mean(similarities_array)),
            "median": float(np.median(similarities_array)),
            "std": float(np.std(similarities_array)),
            "min": float(np.min(similarities_array)),
            "max": float(np.max(similarities_array)),
            "q25": float(np.percentile(similarities_array, 25)),
            "q75": float(np.percentile(similarities_array, 75)),
            "count": len(similarities),
        }

    def cluster_by_similarity(
        self,
        embeddings: List[np.ndarray],
        threshold: float = 0.8,
        method: str = "cosine",
    ) -> List[List[int]]:
        """
        Cluster embeddings by similarity threshold.

        Args:
            embeddings: List of embedding vectors
            threshold: Similarity threshold for clustering
            method: Similarity method to use

        Returns:
            List of clusters (each cluster is a list of indices)
        """
        n = len(embeddings)
        visited = [False] * n
        clusters = []

        for i in range(n):
            if visited[i]:
                continue

            # Start new cluster
            cluster = [i]
            visited[i] = True

            # Find all similar embeddings
            for j in range(i + 1, n):
                if visited[j]:
                    continue

                result = self.calculate_similarity(embeddings[i], embeddings[j], method)
                if result.similarity_score >= threshold:
                    cluster.append(j)
                    visited[j] = True

            clusters.append(cluster)

        return clusters

    def _generate_default_legal_weights(self, embedding_dim: int) -> np.ndarray:
        """Generate default legal weighting for embeddings."""
        # Simple heuristic: slightly emphasize middle dimensions
        weights = np.ones(embedding_dim)

        # Boost middle dimensions (often capture more semantic info)
        start_boost = embedding_dim // 4
        end_boost = 3 * embedding_dim // 4
        weights[start_boost:end_boost] *= 1.2

        # Normalize weights
        weights = weights / np.sum(weights) * embedding_dim

        return weights


# Global similarity calculator
_similarity_calculator: Optional[SimilarityCalculator] = None


def get_similarity_calculator() -> SimilarityCalculator:
    """Get global similarity calculator instance."""
    global _similarity_calculator

    if _similarity_calculator is None:
        _similarity_calculator = SimilarityCalculator()

    return _similarity_calculator


# Convenience functions
def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    calculator = get_similarity_calculator()
    result = calculator.cosine_similarity(embedding1, embedding2)
    return result.similarity_score


def euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate Euclidean distance between two embeddings."""
    calculator = get_similarity_calculator()
    result = calculator.euclidean_similarity(embedding1, embedding2)
    return result.distance


def find_most_similar_embeddings(
    query_embedding: np.ndarray, candidate_embeddings: List[np.ndarray], top_k: int = 5
) -> List[Tuple[int, float]]:
    """Find most similar embeddings to query."""
    calculator = get_similarity_calculator()
    results = calculator.find_most_similar(
        query_embedding, candidate_embeddings, top_k=top_k
    )
    return [(idx, score) for idx, score, _ in results]
