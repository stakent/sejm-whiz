"""Ensemble prediction models combining multiple approaches."""

import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime
import numpy as np
from abc import ABC, abstractmethod

from .config import PredictionConfig
from .core import (
    PredictionInput,
    PredictionResult,
    EnsemblePrediction,
    ModelType,
    PredictionType,
    ModelInfo,
)

logger = logging.getLogger(__name__)


class BaseEnsemble(ABC):
    """Base class for ensemble prediction models."""

    def __init__(self, config: PredictionConfig):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.model_weights: Dict[str, float] = {}
        self.ensemble_strategy = config.ensemble_strategy
        self.is_trained = False

    @abstractmethod
    def add_model(self, name: str, model: Any, weight: float = 1.0) -> None:
        """Add a model to the ensemble."""
        pass

    @abstractmethod
    def predict(self, input_data: PredictionInput) -> EnsemblePrediction:
        """Make ensemble prediction."""
        pass

    @abstractmethod
    def train(self, training_data: Any) -> None:
        """Train the ensemble model."""
        pass


class VotingEnsemble(BaseEnsemble):
    """Voting-based ensemble that combines predictions through voting."""

    def __init__(self, config: PredictionConfig, voting_strategy: str = "soft"):
        super().__init__(config)
        self.voting_strategy = voting_strategy  # soft or hard
        self.model_info: Dict[str, ModelInfo] = {}

    def add_model(self, name: str, model: Any, weight: float = 1.0) -> None:
        """Add a model to the voting ensemble."""
        self.models[name] = model
        self.model_weights[name] = weight
        logger.info(f"Added model '{name}' to voting ensemble with weight {weight}")

    def predict(self, input_data: PredictionInput) -> EnsemblePrediction:
        """Make prediction using voting ensemble."""
        if not self.models:
            raise ValueError("No models added to ensemble")

        individual_predictions = []
        prediction_scores = []

        # Get predictions from each model
        for name, model in self.models.items():
            try:
                prediction = self._get_model_prediction(model, input_data, name)
                individual_predictions.append(prediction)

                # Extract numerical score for voting
                if isinstance(prediction.prediction, (int, float)):
                    prediction_scores.append(float(prediction.prediction))
                elif (
                    isinstance(prediction.prediction, dict)
                    and "score" in prediction.prediction
                ):
                    prediction_scores.append(float(prediction.prediction["score"]))
                else:
                    prediction_scores.append(prediction.confidence)

            except Exception as e:
                logger.error(f"Error getting prediction from model {name}: {e}")
                continue

        if not individual_predictions:
            raise RuntimeError("No successful predictions from ensemble models")

        # Combine predictions using voting strategy
        combined_prediction, combined_confidence = self._combine_predictions(
            individual_predictions, prediction_scores
        )

        # Calculate consensus score
        consensus_score = self._calculate_consensus(prediction_scores)

        return EnsemblePrediction(
            individual_predictions=individual_predictions,
            combined_prediction=combined_prediction,
            combined_confidence=combined_confidence,
            model_weights=self.model_weights.copy(),
            ensemble_strategy=f"voting_{self.voting_strategy}",
            consensus_score=consensus_score,
        )

    def _get_model_prediction(
        self, model: Any, input_data: PredictionInput, model_name: str
    ) -> PredictionResult:
        """Get prediction from an individual model."""
        # This is a placeholder - actual implementation would depend on model interface
        prediction_value = 0.5  # Placeholder
        confidence = 0.7  # Placeholder

        return PredictionResult(
            prediction_id=f"{model_name}_{input_data.document_id}_{datetime.now().isoformat()}",
            document_id=input_data.document_id,
            prediction_type=PredictionType.AMENDMENT_PROBABILITY,
            model_type=ModelType.ENSEMBLE,
            prediction=prediction_value,
            confidence=confidence,
            features_used=["ensemble_feature"],
            model_version="1.0",
            timestamp=datetime.now(),
            explanation=f"Prediction from {model_name} model",
        )

    def _combine_predictions(
        self, predictions: List[PredictionResult], scores: List[float]
    ) -> Tuple[float, float]:
        """Combine individual predictions using voting strategy."""
        if self.voting_strategy == "soft":
            # Weighted average of predictions
            weighted_sum = sum(
                score * self.model_weights.get(pred.model_version, 1.0)
                for score, pred in zip(scores, predictions)
            )
            total_weight = sum(
                self.model_weights.get(pred.model_version, 1.0) for pred in predictions
            )
            combined_prediction = (
                weighted_sum / total_weight if total_weight > 0 else 0.0
            )

            # Weighted average of confidences
            confidence_weighted_sum = sum(
                pred.confidence * self.model_weights.get(pred.model_version, 1.0)
                for pred in predictions
            )
            combined_confidence = (
                confidence_weighted_sum / total_weight if total_weight > 0 else 0.0
            )

        else:  # hard voting
            # Majority vote (for classification)
            votes = {}
            for pred in predictions:
                vote = (
                    round(float(pred.prediction))
                    if isinstance(pred.prediction, (int, float))
                    else str(pred.prediction)
                )
                votes[vote] = votes.get(vote, 0) + self.model_weights.get(
                    pred.model_version, 1.0
                )

            combined_prediction = max(votes.keys(), key=lambda k: votes[k])
            combined_confidence = max(votes.values()) / sum(votes.values())

        return combined_prediction, combined_confidence

    def _calculate_consensus(self, scores: List[float]) -> float:
        """Calculate consensus score based on prediction variance."""
        if len(scores) < 2:
            return 1.0

        variance = np.var(scores)
        # Convert variance to consensus score (lower variance = higher consensus)
        consensus = max(0.0, 1.0 - variance)
        return float(consensus)

    def train(self, training_data: Any) -> None:
        """Train the voting ensemble (trains individual models)."""
        logger.info("Training voting ensemble models")

        for name, model in self.models.items():
            try:
                if hasattr(model, "train"):
                    model.train(training_data)
                    logger.info(f"Trained model {name}")
                else:
                    logger.warning(f"Model {name} does not support training")
            except Exception as e:
                logger.error(f"Error training model {name}: {e}")

        self.is_trained = True


class StackingEnsemble(BaseEnsemble):
    """Stacking ensemble with meta-learner."""

    def __init__(self, config: PredictionConfig, meta_learner: Any = None):
        super().__init__(config)
        self.meta_learner = meta_learner
        self.base_models: Dict[str, Any] = {}
        self.meta_features_cache: Dict[str, np.ndarray] = {}

    def add_model(self, name: str, model: Any, weight: float = 1.0) -> None:
        """Add a base model to the stacking ensemble."""
        self.base_models[name] = model
        self.model_weights[name] = weight
        logger.info(f"Added base model '{name}' to stacking ensemble")

    def set_meta_learner(self, meta_learner: Any) -> None:
        """Set the meta-learner for the stacking ensemble."""
        self.meta_learner = meta_learner
        logger.info("Set meta-learner for stacking ensemble")

    def predict(self, input_data: PredictionInput) -> EnsemblePrediction:
        """Make prediction using stacking ensemble."""
        if not self.base_models:
            raise ValueError("No base models added to ensemble")
        if not self.meta_learner:
            raise ValueError("No meta-learner set for stacking ensemble")

        # Get predictions from base models
        individual_predictions = []
        meta_features = []

        for name, model in self.base_models.items():
            try:
                prediction = self._get_model_prediction(model, input_data, name)
                individual_predictions.append(prediction)

                # Collect features for meta-learner
                if isinstance(prediction.prediction, (int, float)):
                    meta_features.append(prediction.prediction)
                else:
                    meta_features.append(prediction.confidence)

            except Exception as e:
                logger.error(f"Error getting prediction from base model {name}: {e}")
                continue

        if not individual_predictions:
            raise RuntimeError("No successful predictions from base models")

        # Use meta-learner to combine predictions
        meta_input = np.array(meta_features).reshape(1, -1)
        try:
            if hasattr(self.meta_learner, "predict_proba"):
                meta_prediction = self.meta_learner.predict_proba(meta_input)[0]
                combined_prediction = float(
                    meta_prediction[1]
                )  # Assuming binary classification
                combined_confidence = max(meta_prediction)
            else:
                combined_prediction = float(self.meta_learner.predict(meta_input)[0])
                combined_confidence = 0.8  # Default confidence for regression
        except Exception as e:
            logger.error(f"Error in meta-learner prediction: {e}")
            # Fallback to simple averaging
            scores = [p.confidence for p in individual_predictions]
            combined_prediction = float(np.mean(scores))
            combined_confidence = float(
                np.mean([p.confidence for p in individual_predictions])
            )

        # Calculate consensus score
        scores = [
            float(p.prediction)
            if isinstance(p.prediction, (int, float))
            else p.confidence
            for p in individual_predictions
        ]
        consensus_score = self._calculate_consensus(scores)

        return EnsemblePrediction(
            individual_predictions=individual_predictions,
            combined_prediction=combined_prediction,
            combined_confidence=combined_confidence,
            model_weights=self.model_weights.copy(),
            ensemble_strategy="stacking",
            consensus_score=consensus_score,
        )

    def _calculate_consensus(self, scores: List[float]) -> float:
        """Calculate consensus score based on prediction variance."""
        if len(scores) < 2:
            return 1.0

        variance = np.var(scores)
        consensus = max(0.0, 1.0 - variance)
        return float(consensus)

    def train(self, training_data: Any) -> None:
        """Train the stacking ensemble (base models + meta-learner)."""
        logger.info("Training stacking ensemble")

        # First, train base models
        for name, model in self.base_models.items():
            try:
                if hasattr(model, "train"):
                    model.train(training_data)
                    logger.info(f"Trained base model {name}")
            except Exception as e:
                logger.error(f"Error training base model {name}: {e}")

        # Then train meta-learner on base model predictions
        # This is a simplified version - actual implementation would need cross-validation
        if self.meta_learner and hasattr(self.meta_learner, "fit"):
            try:
                # Generate meta-features from base models
                # This is placeholder logic
                meta_features = np.random.random(
                    (100, len(self.base_models))
                )  # Placeholder
                meta_labels = np.random.randint(0, 2, 100)  # Placeholder

                self.meta_learner.fit(meta_features, meta_labels)
                logger.info("Trained meta-learner")
            except Exception as e:
                logger.error(f"Error training meta-learner: {e}")

        self.is_trained = True


class BlendingEnsemble(BaseEnsemble):
    """Blending ensemble with holdout validation."""

    def __init__(self, config: PredictionConfig, holdout_ratio: float = 0.2):
        super().__init__(config)
        self.holdout_ratio = holdout_ratio
        self.blending_weights: Dict[str, float] = {}
        self.base_models: Dict[str, Any] = {}

    def add_model(self, name: str, model: Any, weight: float = 1.0) -> None:
        """Add a model to the blending ensemble."""
        self.base_models[name] = model
        self.model_weights[name] = weight
        logger.info(f"Added model '{name}' to blending ensemble")

    def predict(self, input_data: PredictionInput) -> EnsemblePrediction:
        """Make prediction using blending ensemble."""
        if not self.base_models:
            raise ValueError("No models added to ensemble")

        individual_predictions = []
        weighted_scores = []

        # Get predictions from each model
        for name, model in self.base_models.items():
            try:
                prediction = self._get_model_prediction(model, input_data, name)
                individual_predictions.append(prediction)

                # Apply blending weight
                blend_weight = self.blending_weights.get(
                    name, self.model_weights.get(name, 1.0)
                )
                if isinstance(prediction.prediction, (int, float)):
                    weighted_scores.append(float(prediction.prediction) * blend_weight)
                else:
                    weighted_scores.append(prediction.confidence * blend_weight)

            except Exception as e:
                logger.error(f"Error getting prediction from model {name}: {e}")
                continue

        if not individual_predictions:
            raise RuntimeError("No successful predictions from ensemble models")

        # Combine using learned blending weights
        total_weight = sum(
            self.blending_weights.get(name, self.model_weights.get(name, 1.0))
            for name in self.base_models.keys()
            if any(p.model_version == name for p in individual_predictions)
        )

        combined_prediction = (
            sum(weighted_scores) / total_weight if total_weight > 0 else 0.0
        )

        # Calculate combined confidence
        weighted_confidences = [
            pred.confidence * self.blending_weights.get(pred.model_version, 1.0)
            for pred in individual_predictions
        ]
        combined_confidence = (
            sum(weighted_confidences) / total_weight if total_weight > 0 else 0.0
        )

        # Calculate consensus
        scores = [
            float(p.prediction)
            if isinstance(p.prediction, (int, float))
            else p.confidence
            for p in individual_predictions
        ]
        consensus_score = self._calculate_consensus(scores)

        return EnsemblePrediction(
            individual_predictions=individual_predictions,
            combined_prediction=combined_prediction,
            combined_confidence=combined_confidence,
            model_weights=self.blending_weights.copy(),
            ensemble_strategy="blending",
            consensus_score=consensus_score,
        )

    def _calculate_consensus(self, scores: List[float]) -> float:
        """Calculate consensus score based on prediction variance."""
        if len(scores) < 2:
            return 1.0

        variance = np.var(scores)
        consensus = max(0.0, 1.0 - variance)
        return float(consensus)

    def train(self, training_data: Any) -> None:
        """Train the blending ensemble and learn optimal weights."""
        logger.info("Training blending ensemble")

        # Train base models on training portion
        for name, model in self.base_models.items():
            try:
                if hasattr(model, "train"):
                    model.train(training_data)
                    logger.info(f"Trained base model {name}")
            except Exception as e:
                logger.error(f"Error training base model {name}: {e}")

        # Learn blending weights on holdout data
        # This is placeholder logic - actual implementation would use validation data
        self.blending_weights = {
            name: max(0.1, np.random.random()) for name in self.base_models.keys()
        }

        # Normalize weights
        total_weight = sum(self.blending_weights.values())
        self.blending_weights = {
            name: weight / total_weight
            for name, weight in self.blending_weights.items()
        }

        logger.info(f"Learned blending weights: {self.blending_weights}")
        self.is_trained = True


def get_ensemble_model(
    config: PredictionConfig, ensemble_type: str = None
) -> BaseEnsemble:
    """Factory function to create ensemble models."""
    if ensemble_type is None:
        ensemble_type = config.ensemble_strategy

    if ensemble_type == "voting":
        return VotingEnsemble(config, voting_strategy="soft")
    elif ensemble_type == "stacking":
        return StackingEnsemble(config)
    elif ensemble_type == "blending":
        return BlendingEnsemble(config)
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")


def create_default_ensemble(config: PredictionConfig) -> BaseEnsemble:
    """Create a default ensemble with common configuration."""
    ensemble = get_ensemble_model(config)

    # Add placeholder models - actual implementation would add real models
    # ensemble.add_model("similarity_model", SimilarityModel(), weight=0.4)
    # ensemble.add_model("classification_model", ClassificationModel(), weight=0.4)
    # ensemble.add_model("temporal_model", TemporalModel(), weight=0.2)

    logger.info(f"Created default {config.ensemble_strategy} ensemble")
    return ensemble
