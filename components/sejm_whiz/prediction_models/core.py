"""Core data models and types for prediction models."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import numpy as np


class PredictionType(Enum):
    """Types of predictions supported by the system."""

    AMENDMENT_PROBABILITY = "amendment_probability"
    CATEGORY_CLASSIFICATION = "category_classification"
    IMPACT_ASSESSMENT = "impact_assessment"
    TIMELINE_PREDICTION = "timeline_prediction"
    SIMILARITY_BASED = "similarity_based"


class ModelType(Enum):
    """Types of models in the prediction system."""

    ENSEMBLE = "ensemble"
    SIMILARITY = "similarity"
    CLASSIFICATION = "classification"
    TEMPORAL = "temporal"
    HYBRID = "hybrid"


class ConfidenceLevel(Enum):
    """Confidence levels for predictions."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class PredictionInput:
    """Input data for making predictions."""

    document_id: str
    text: str
    embeddings: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    legal_category: Optional[str] = None
    historical_data: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.historical_data is None:
            self.historical_data = []


@dataclass
class PredictionResult:
    """Result of a prediction operation."""

    prediction_id: str
    document_id: str
    prediction_type: PredictionType
    model_type: ModelType
    prediction: Union[float, str, Dict[str, Any]]
    confidence: float
    confidence_level: ConfidenceLevel
    features_used: List[str]
    model_version: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    explanation: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

        # Determine confidence level from numeric confidence
        if self.confidence >= 0.9:
            self.confidence_level = ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.75:
            self.confidence_level = ConfidenceLevel.HIGH
        elif self.confidence >= 0.6:
            self.confidence_level = ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.4:
            self.confidence_level = ConfidenceLevel.LOW
        else:
            self.confidence_level = ConfidenceLevel.VERY_LOW


@dataclass
class EnsemblePrediction:
    """Result of ensemble prediction combining multiple models."""

    individual_predictions: List[PredictionResult]
    combined_prediction: Union[float, str, Dict[str, Any]]
    combined_confidence: float
    model_weights: Dict[str, float]
    ensemble_strategy: str
    consensus_score: float

    @property
    def best_individual_prediction(self) -> PredictionResult:
        """Get the individual prediction with highest confidence."""
        return max(self.individual_predictions, key=lambda p: p.confidence)

    @property
    def prediction_variance(self) -> float:
        """Calculate variance in numerical predictions."""
        numerical_predictions = [
            float(p.prediction)
            for p in self.individual_predictions
            if isinstance(p.prediction, (int, float))
        ]
        if len(numerical_predictions) < 2:
            return 0.0
        return float(np.var(numerical_predictions))


@dataclass
class TrainingData:
    """Training data for model training."""

    features: np.ndarray
    labels: np.ndarray
    document_ids: List[str]
    feature_names: List[str]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ModelMetrics:
    """Evaluation metrics for a trained model."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    cross_val_scores: Optional[List[float]] = None
    training_time: Optional[float] = None
    inference_time: Optional[float] = None


@dataclass
class ModelInfo:
    """Information about a trained model."""

    model_id: str
    model_type: ModelType
    model_version: str
    training_date: datetime
    feature_names: List[str]
    hyperparameters: Dict[str, Any]
    metrics: ModelMetrics
    model_path: Optional[str] = None
    description: Optional[str] = None

    @property
    def is_high_quality(self) -> bool:
        """Check if model meets quality thresholds."""
        return (
            self.metrics.accuracy >= 0.8
            and self.metrics.f1_score >= 0.75
            and self.metrics.precision >= 0.7
        )


@dataclass
class FeatureVector:
    """Feature vector for model input."""

    document_id: str
    features: Dict[str, Any]
    embeddings: Optional[np.ndarray] = None
    legal_features: Optional[Dict[str, Any]] = None
    temporal_features: Optional[Dict[str, Any]] = None

    def to_array(self, feature_names: List[str]) -> np.ndarray:
        """Convert to numpy array using specified feature order."""
        return np.array([self.features.get(name, 0.0) for name in feature_names])


@dataclass
class SimilarityMatch:
    """Result from similarity-based prediction."""

    source_document_id: str
    target_document_id: str
    similarity_score: float
    similarity_type: str  # cosine, euclidean, etc.
    matched_features: List[str]
    historical_outcome: Optional[Dict[str, Any]] = None
    confidence_boost: float = 0.0


@dataclass
class BatchPredictionJob:
    """Batch prediction job configuration."""

    job_id: str
    inputs: List[PredictionInput]
    prediction_types: List[PredictionType]
    model_types: List[ModelType]
    batch_size: int = 32
    use_ensemble: bool = True
    save_results: bool = True
    callback_url: Optional[str] = None


@dataclass
class BatchPredictionResult:
    """Result of batch prediction job."""

    job_id: str
    predictions: List[PredictionResult]
    ensemble_predictions: List[EnsemblePrediction]
    total_processed: int
    success_count: int
    error_count: int
    processing_time: float
    errors: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def success_rate(self) -> float:
        """Calculate success rate of batch job."""
        if self.total_processed == 0:
            return 0.0
        return self.success_count / self.total_processed
