"""Prediction models component for legal document analysis."""

# Configuration
from .config import PredictionConfig, get_prediction_config

# Core data models and types
from .core import (
    PredictionType,
    ModelType,
    ConfidenceLevel,
    PredictionInput,
    PredictionResult,
    EnsemblePrediction,
    TrainingData,
    ModelMetrics,
    ModelInfo,
    FeatureVector,
    SimilarityMatch,
    BatchPredictionJob,
    BatchPredictionResult,
)

# Ensemble models
from .ensemble import (
    BaseEnsemble,
    VotingEnsemble,
    StackingEnsemble,
    BlendingEnsemble,
    get_ensemble_model,
    create_default_ensemble,
)

# Similarity-based predictors
from .similarity import (
    BaseSimilarityPredictor,
    CosineDistancePredictor,
    EuclideanDistancePredictor,
    HybridSimilarityPredictor,
    TemporalSimilarityPredictor,
    get_similarity_predictor,
    create_default_similarity_predictor,
)

# Classification models
from .classification import (
    BaseClassifier,
    RandomForestLegalClassifier,
    GradientBoostingLegalClassifier,
    SVMLegalClassifier,
    LogisticRegressionLegalClassifier,
    TfidfEmbeddingClassifier,
    get_classifier,
    create_classifier_ensemble,
    train_classifier_ensemble,
)

__all__ = [
    # Configuration
    "PredictionConfig",
    "get_prediction_config",
    # Core types and data models
    "PredictionType",
    "ModelType",
    "ConfidenceLevel",
    "PredictionInput",
    "PredictionResult",
    "EnsemblePrediction",
    "TrainingData",
    "ModelMetrics",
    "ModelInfo",
    "FeatureVector",
    "SimilarityMatch",
    "BatchPredictionJob",
    "BatchPredictionResult",
    # Ensemble models
    "BaseEnsemble",
    "VotingEnsemble",
    "StackingEnsemble",
    "BlendingEnsemble",
    "get_ensemble_model",
    "create_default_ensemble",
    # Similarity predictors
    "BaseSimilarityPredictor",
    "CosineDistancePredictor",
    "EuclideanDistancePredictor",
    "HybridSimilarityPredictor",
    "TemporalSimilarityPredictor",
    "get_similarity_predictor",
    "create_default_similarity_predictor",
    # Classification models
    "BaseClassifier",
    "RandomForestLegalClassifier",
    "GradientBoostingLegalClassifier",
    "SVMLegalClassifier",
    "LogisticRegressionLegalClassifier",
    "TfidfEmbeddingClassifier",
    "get_classifier",
    "create_classifier_ensemble",
    "train_classifier_ensemble",
]
