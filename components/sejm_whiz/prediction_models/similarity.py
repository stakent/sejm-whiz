"""Similarity-based prediction models using embeddings and historical data."""

import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime
import numpy as np
from abc import ABC, abstractmethod

from .config import PredictionConfig
from .core import (
    PredictionInput,
    PredictionResult,
    SimilarityMatch,
    ModelType,
    PredictionType,
)

logger = logging.getLogger(__name__)


class BaseSimilarityPredictor(ABC):
    """Base class for similarity-based prediction models."""

    def __init__(self, config: PredictionConfig):
        self.config = config
        self.similarity_threshold = config.similarity_threshold
        self.top_k = config.top_k_similar
        self.decay_factor = config.similarity_decay_factor
        self.historical_data: Dict[str, Dict[str, Any]] = {}
        self.document_embeddings: Dict[str, np.ndarray] = {}

    @abstractmethod
    def calculate_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Calculate similarity between two embeddings."""
        pass

    @abstractmethod
    def predict(self, input_data: PredictionInput) -> PredictionResult:
        """Make similarity-based prediction."""
        pass

    def add_historical_document(
        self,
        document_id: str,
        embedding: np.ndarray,
        outcome: Dict[str, Any],
        metadata: Dict[str, Any] = None,
    ) -> None:
        """Add historical document for similarity matching."""
        self.document_embeddings[document_id] = embedding
        self.historical_data[document_id] = {
            "outcome": outcome,
            "metadata": metadata or {},
            "timestamp": datetime.now(),
        }
        logger.debug(f"Added historical document {document_id}")

    def find_similar_documents(
        self, query_embedding: np.ndarray, exclude_ids: List[str] = None
    ) -> List[SimilarityMatch]:
        """Find most similar historical documents."""
        exclude_ids = exclude_ids or []
        similarities = []

        for doc_id, embedding in self.document_embeddings.items():
            if doc_id in exclude_ids:
                continue

            try:
                similarity = self.calculate_similarity(query_embedding, embedding)
                if similarity >= self.similarity_threshold:
                    historical_data = self.historical_data.get(doc_id, {})
                    match = SimilarityMatch(
                        source_document_id="query",
                        target_document_id=doc_id,
                        similarity_score=similarity,
                        similarity_type=self.__class__.__name__.lower(),
                        matched_features=["embedding"],
                        historical_outcome=historical_data.get("outcome"),
                        confidence_boost=self._calculate_confidence_boost(similarity),
                    )
                    similarities.append(match)
            except Exception as e:
                logger.error(
                    f"Error calculating similarity with document {doc_id}: {e}"
                )
                continue

        # Sort by similarity score and return top-k
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        return similarities[: self.top_k]

    def _calculate_confidence_boost(self, similarity: float) -> float:
        """Calculate confidence boost based on similarity score."""
        if similarity >= 0.95:
            return 0.3
        elif similarity >= 0.9:
            return 0.2
        elif similarity >= 0.85:
            return 0.15
        elif similarity >= 0.8:
            return 0.1
        else:
            return 0.05

    def _apply_temporal_decay(self, similarity: float, document_age_days: int) -> float:
        """Apply temporal decay to similarity score."""
        decay = np.exp(-self.decay_factor * document_age_days / 365.0)
        return similarity * decay


class CosineDistancePredictor(BaseSimilarityPredictor):
    """Similarity predictor using cosine distance."""

    def calculate_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between embeddings."""
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Calculate cosine similarity
        cosine_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(np.clip(cosine_sim, -1.0, 1.0))

    def predict(self, input_data: PredictionInput) -> PredictionResult:
        """Make prediction based on cosine similarity."""
        if input_data.embeddings is None:
            raise ValueError("Embeddings required for similarity-based prediction")

        # Find similar documents
        similar_docs = self.find_similar_documents(
            input_data.embeddings, exclude_ids=[input_data.document_id]
        )

        if not similar_docs:
            # No similar documents found
            return PredictionResult(
                prediction_id=f"cosine_{input_data.document_id}_{datetime.now().isoformat()}",
                document_id=input_data.document_id,
                prediction_type=PredictionType.SIMILARITY_BASED,
                model_type=ModelType.SIMILARITY,
                prediction=0.5,  # Default neutral prediction
                confidence=0.3,  # Low confidence
                features_used=["cosine_similarity"],
                model_version="cosine_v1.0",
                timestamp=datetime.now(),
                explanation="No similar historical documents found",
            )

        # Aggregate predictions from similar documents
        prediction_value, confidence = self._aggregate_similar_predictions(similar_docs)

        return PredictionResult(
            prediction_id=f"cosine_{input_data.document_id}_{datetime.now().isoformat()}",
            document_id=input_data.document_id,
            prediction_type=PredictionType.SIMILARITY_BASED,
            model_type=ModelType.SIMILARITY,
            prediction=prediction_value,
            confidence=confidence,
            features_used=["cosine_similarity", "historical_outcomes"],
            model_version="cosine_v1.0",
            timestamp=datetime.now(),
            explanation=f"Based on {len(similar_docs)} similar documents (avg similarity: {np.mean([m.similarity_score for m in similar_docs]):.3f})",
        )

    def _aggregate_similar_predictions(
        self, similar_docs: List[SimilarityMatch]
    ) -> Tuple[float, float]:
        """Aggregate predictions from similar documents."""
        if not similar_docs:
            return 0.5, 0.3

        # Weight predictions by similarity score
        weighted_predictions = []
        total_weight = 0.0

        for match in similar_docs:
            if match.historical_outcome:
                # Extract prediction value from historical outcome
                if isinstance(match.historical_outcome, dict):
                    pred_value = match.historical_outcome.get("prediction", 0.5)
                    if isinstance(pred_value, bool):
                        pred_value = float(pred_value)
                    elif not isinstance(pred_value, (int, float)):
                        pred_value = 0.5
                else:
                    pred_value = (
                        float(match.historical_outcome)
                        if isinstance(match.historical_outcome, (int, float, bool))
                        else 0.5
                    )

                weight = match.similarity_score
                weighted_predictions.append(pred_value * weight)
                total_weight += weight

        if total_weight == 0:
            return 0.5, 0.3

        # Calculate weighted average
        prediction = sum(weighted_predictions) / total_weight

        # Calculate confidence based on similarity scores and consistency
        avg_similarity = np.mean([m.similarity_score for m in similar_docs])
        prediction_variance = np.var(
            [
                match.historical_outcome.get("prediction", 0.5)
                if isinstance(match.historical_outcome, dict)
                else float(match.historical_outcome)
                if isinstance(match.historical_outcome, (int, float, bool))
                else 0.5
                for match in similar_docs
            ]
        )

        # Higher confidence for higher similarity and lower variance
        base_confidence = avg_similarity * 0.8
        variance_penalty = min(0.3, prediction_variance)
        confidence = max(0.1, base_confidence - variance_penalty)

        return float(prediction), float(confidence)


class EuclideanDistancePredictor(BaseSimilarityPredictor):
    """Similarity predictor using Euclidean distance."""

    def calculate_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Calculate similarity based on Euclidean distance."""
        # Calculate Euclidean distance
        distance = np.linalg.norm(embedding1 - embedding2)

        # Convert distance to similarity score (0-1 range)
        # Use exponential decay to map distance to similarity
        max_distance = np.sqrt(len(embedding1))  # Theoretical maximum distance
        normalized_distance = distance / max_distance
        similarity = np.exp(-normalized_distance)

        return float(np.clip(similarity, 0.0, 1.0))

    def predict(self, input_data: PredictionInput) -> PredictionResult:
        """Make prediction based on Euclidean distance similarity."""
        if input_data.embeddings is None:
            raise ValueError("Embeddings required for similarity-based prediction")

        # Find similar documents
        similar_docs = self.find_similar_documents(
            input_data.embeddings, exclude_ids=[input_data.document_id]
        )

        if not similar_docs:
            return PredictionResult(
                prediction_id=f"euclidean_{input_data.document_id}_{datetime.now().isoformat()}",
                document_id=input_data.document_id,
                prediction_type=PredictionType.SIMILARITY_BASED,
                model_type=ModelType.SIMILARITY,
                prediction=0.5,
                confidence=0.3,
                features_used=["euclidean_distance"],
                model_version="euclidean_v1.0",
                timestamp=datetime.now(),
                explanation="No similar historical documents found",
            )

        # Aggregate predictions
        prediction_value, confidence = self._aggregate_similar_predictions(similar_docs)

        return PredictionResult(
            prediction_id=f"euclidean_{input_data.document_id}_{datetime.now().isoformat()}",
            document_id=input_data.document_id,
            prediction_type=PredictionType.SIMILARITY_BASED,
            model_type=ModelType.SIMILARITY,
            prediction=prediction_value,
            confidence=confidence,
            features_used=["euclidean_distance", "historical_outcomes"],
            model_version="euclidean_v1.0",
            timestamp=datetime.now(),
            explanation=f"Based on {len(similar_docs)} similar documents (avg similarity: {np.mean([m.similarity_score for m in similar_docs]):.3f})",
        )

    def _aggregate_similar_predictions(
        self, similar_docs: List[SimilarityMatch]
    ) -> Tuple[float, float]:
        """Aggregate predictions from similar documents."""
        # Same implementation as CosineDistancePredictor
        if not similar_docs:
            return 0.5, 0.3

        weighted_predictions = []
        total_weight = 0.0

        for match in similar_docs:
            if match.historical_outcome:
                if isinstance(match.historical_outcome, dict):
                    pred_value = match.historical_outcome.get("prediction", 0.5)
                    if isinstance(pred_value, bool):
                        pred_value = float(pred_value)
                    elif not isinstance(pred_value, (int, float)):
                        pred_value = 0.5
                else:
                    pred_value = (
                        float(match.historical_outcome)
                        if isinstance(match.historical_outcome, (int, float, bool))
                        else 0.5
                    )

                weight = match.similarity_score
                weighted_predictions.append(pred_value * weight)
                total_weight += weight

        if total_weight == 0:
            return 0.5, 0.3

        prediction = sum(weighted_predictions) / total_weight

        avg_similarity = np.mean([m.similarity_score for m in similar_docs])
        prediction_variance = np.var(
            [
                match.historical_outcome.get("prediction", 0.5)
                if isinstance(match.historical_outcome, dict)
                else float(match.historical_outcome)
                if isinstance(match.historical_outcome, (int, float, bool))
                else 0.5
                for match in similar_docs
            ]
        )

        base_confidence = avg_similarity * 0.8
        variance_penalty = min(0.3, prediction_variance)
        confidence = max(0.1, base_confidence - variance_penalty)

        return float(prediction), float(confidence)


class HybridSimilarityPredictor(BaseSimilarityPredictor):
    """Hybrid similarity predictor combining multiple similarity measures."""

    def __init__(self, config: PredictionConfig):
        super().__init__(config)
        self.cosine_predictor = CosineDistancePredictor(config)
        self.euclidean_predictor = EuclideanDistancePredictor(config)
        self.similarity_weights = {"cosine": 0.7, "euclidean": 0.3}

    def calculate_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Calculate hybrid similarity combining multiple measures."""
        cosine_sim = self.cosine_predictor.calculate_similarity(embedding1, embedding2)
        euclidean_sim = self.euclidean_predictor.calculate_similarity(
            embedding1, embedding2
        )

        # Weighted combination
        hybrid_sim = (
            cosine_sim * self.similarity_weights["cosine"]
            + euclidean_sim * self.similarity_weights["euclidean"]
        )

        return float(hybrid_sim)

    def predict(self, input_data: PredictionInput) -> PredictionResult:
        """Make prediction using hybrid similarity approach."""
        if input_data.embeddings is None:
            raise ValueError("Embeddings required for similarity-based prediction")

        # Get predictions from both similarity measures
        cosine_pred = self.cosine_predictor.predict(input_data)
        euclidean_pred = self.euclidean_predictor.predict(input_data)

        # Combine predictions
        combined_prediction = (
            float(cosine_pred.prediction) * self.similarity_weights["cosine"]
            + float(euclidean_pred.prediction) * self.similarity_weights["euclidean"]
        )

        combined_confidence = (
            cosine_pred.confidence * self.similarity_weights["cosine"]
            + euclidean_pred.confidence * self.similarity_weights["euclidean"]
        )

        return PredictionResult(
            prediction_id=f"hybrid_{input_data.document_id}_{datetime.now().isoformat()}",
            document_id=input_data.document_id,
            prediction_type=PredictionType.SIMILARITY_BASED,
            model_type=ModelType.HYBRID,
            prediction=combined_prediction,
            confidence=combined_confidence,
            features_used=[
                "cosine_similarity",
                "euclidean_distance",
                "hybrid_combination",
            ],
            model_version="hybrid_v1.0",
            timestamp=datetime.now(),
            explanation=f"Hybrid prediction combining cosine ({cosine_pred.prediction:.3f}) and Euclidean ({euclidean_pred.prediction:.3f}) similarities",
        )

    def add_historical_document(
        self,
        document_id: str,
        embedding: np.ndarray,
        outcome: Dict[str, Any],
        metadata: Dict[str, Any] = None,
    ) -> None:
        """Add historical document to both predictors."""
        super().add_historical_document(document_id, embedding, outcome, metadata)
        self.cosine_predictor.add_historical_document(
            document_id, embedding, outcome, metadata
        )
        self.euclidean_predictor.add_historical_document(
            document_id, embedding, outcome, metadata
        )


class TemporalSimilarityPredictor(BaseSimilarityPredictor):
    """Similarity predictor with temporal weighting."""

    def __init__(self, config: PredictionConfig):
        super().__init__(config)
        self.temporal_window_days = config.temporal_window_days
        self.base_predictor = CosineDistancePredictor(config)

    def calculate_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Calculate similarity using base predictor."""
        return self.base_predictor.calculate_similarity(embedding1, embedding2)

    def predict(self, input_data: PredictionInput) -> PredictionResult:
        """Make temporally-weighted similarity prediction."""
        if input_data.embeddings is None:
            raise ValueError("Embeddings required for similarity-based prediction")

        # Find similar documents
        similar_docs = self.find_similar_documents(
            input_data.embeddings, exclude_ids=[input_data.document_id]
        )

        if not similar_docs:
            return PredictionResult(
                prediction_id=f"temporal_{input_data.document_id}_{datetime.now().isoformat()}",
                document_id=input_data.document_id,
                prediction_type=PredictionType.SIMILARITY_BASED,
                model_type=ModelType.SIMILARITY,
                prediction=0.5,
                confidence=0.3,
                features_used=["temporal_similarity"],
                model_version="temporal_v1.0",
                timestamp=datetime.now(),
                explanation="No similar historical documents found",
            )

        # Apply temporal weighting
        temporally_weighted_docs = []
        current_time = datetime.now()

        for match in similar_docs:
            doc_data = self.historical_data.get(match.target_document_id, {})
            doc_timestamp = doc_data.get("timestamp", current_time)

            if isinstance(doc_timestamp, str):
                doc_timestamp = datetime.fromisoformat(doc_timestamp)

            age_days = (current_time - doc_timestamp).days

            # Apply temporal decay if document is older than window
            if age_days > self.temporal_window_days:
                decayed_similarity = self._apply_temporal_decay(
                    match.similarity_score, age_days
                )
                match.similarity_score = decayed_similarity
                match.confidence_boost *= (
                    0.8  # Reduce confidence boost for old documents
                )

            temporally_weighted_docs.append(match)

        # Filter out documents with very low similarity after temporal decay
        temporally_weighted_docs = [
            doc
            for doc in temporally_weighted_docs
            if doc.similarity_score >= self.similarity_threshold * 0.5
        ]

        if not temporally_weighted_docs:
            return PredictionResult(
                prediction_id=f"temporal_{input_data.document_id}_{datetime.now().isoformat()}",
                document_id=input_data.document_id,
                prediction_type=PredictionType.SIMILARITY_BASED,
                model_type=ModelType.SIMILARITY,
                prediction=0.5,
                confidence=0.3,
                features_used=["temporal_similarity"],
                model_version="temporal_v1.0",
                timestamp=datetime.now(),
                explanation="No sufficiently similar documents after temporal weighting",
            )

        # Aggregate predictions with temporal weighting
        prediction_value, confidence = self._aggregate_temporal_predictions(
            temporally_weighted_docs
        )

        return PredictionResult(
            prediction_id=f"temporal_{input_data.document_id}_{datetime.now().isoformat()}",
            document_id=input_data.document_id,
            prediction_type=PredictionType.SIMILARITY_BASED,
            model_type=ModelType.SIMILARITY,
            prediction=prediction_value,
            confidence=confidence,
            features_used=[
                "temporal_similarity",
                "temporal_decay",
                "historical_outcomes",
            ],
            model_version="temporal_v1.0",
            timestamp=datetime.now(),
            explanation=f"Based on {len(temporally_weighted_docs)} temporally-weighted similar documents",
        )

    def _aggregate_temporal_predictions(
        self, similar_docs: List[SimilarityMatch]
    ) -> Tuple[float, float]:
        """Aggregate predictions with temporal weighting."""
        if not similar_docs:
            return 0.5, 0.3

        weighted_predictions = []
        total_weight = 0.0

        for match in similar_docs:
            if match.historical_outcome:
                if isinstance(match.historical_outcome, dict):
                    pred_value = match.historical_outcome.get("prediction", 0.5)
                    if isinstance(pred_value, bool):
                        pred_value = float(pred_value)
                    elif not isinstance(pred_value, (int, float)):
                        pred_value = 0.5
                else:
                    pred_value = (
                        float(match.historical_outcome)
                        if isinstance(match.historical_outcome, (int, float, bool))
                        else 0.5
                    )

                # Weight includes both similarity and temporal factors
                weight = match.similarity_score * (1.0 + match.confidence_boost)
                weighted_predictions.append(pred_value * weight)
                total_weight += weight

        if total_weight == 0:
            return 0.5, 0.3

        prediction = sum(weighted_predictions) / total_weight

        # Confidence considers both similarity and temporal factors
        avg_similarity = np.mean([m.similarity_score for m in similar_docs])
        avg_boost = np.mean([m.confidence_boost for m in similar_docs])
        confidence = min(0.95, avg_similarity * 0.8 + avg_boost)

        return float(prediction), float(confidence)


def get_similarity_predictor(
    config: PredictionConfig, predictor_type: str = "cosine"
) -> BaseSimilarityPredictor:
    """Factory function to create similarity predictors."""
    if predictor_type == "cosine":
        return CosineDistancePredictor(config)
    elif predictor_type == "euclidean":
        return EuclideanDistancePredictor(config)
    elif predictor_type == "hybrid":
        return HybridSimilarityPredictor(config)
    elif predictor_type == "temporal":
        return TemporalSimilarityPredictor(config)
    else:
        raise ValueError(f"Unknown similarity predictor type: {predictor_type}")


def create_default_similarity_predictor(
    config: PredictionConfig,
) -> BaseSimilarityPredictor:
    """Create default similarity predictor based on configuration."""
    # Use hybrid predictor as default for better performance
    predictor = HybridSimilarityPredictor(config)
    logger.info("Created default hybrid similarity predictor")
    return predictor
