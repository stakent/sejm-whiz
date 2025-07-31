"""Prediction models configuration for legal document analysis."""

import os
from typing import List, Dict
from pydantic import Field
from pydantic_settings import BaseSettings


class PredictionConfig(BaseSettings):
    """Prediction models configuration with environment variable support."""

    # Model settings
    ensemble_strategy: str = Field(
        default="voting", env="PREDICTION_ENSEMBLE_STRATEGY"
    )  # voting, stacking, blending
    model_cache_dir: str = Field(
        default="./models/prediction", env="PREDICTION_MODEL_CACHE_DIR"
    )
    device: str = Field(default="auto", env="PREDICTION_DEVICE")  # auto, cpu, cuda, mps

    # Similarity-based prediction settings
    similarity_threshold: float = Field(
        default=0.75, env="PREDICTION_SIMILARITY_THRESHOLD"
    )
    top_k_similar: int = Field(default=10, env="PREDICTION_TOP_K_SIMILAR")
    similarity_decay_factor: float = Field(
        default=0.9, env="PREDICTION_SIMILARITY_DECAY"
    )

    # Classification settings
    classification_models: List[str] = Field(
        default=["random_forest", "gradient_boosting", "svm"],
        env="PREDICTION_CLASSIFICATION_MODELS",
    )
    cross_validation_folds: int = Field(default=5, env="PREDICTION_CV_FOLDS")
    test_size: float = Field(default=0.2, env="PREDICTION_TEST_SIZE")
    random_state: int = Field(default=42, env="PREDICTION_RANDOM_STATE")

    # Ensemble settings
    ensemble_models: List[str] = Field(
        default=["similarity", "classification", "temporal"],
        env="PREDICTION_ENSEMBLE_MODELS",
    )
    ensemble_weights: Dict[str, float] = Field(
        default={"similarity": 0.4, "classification": 0.4, "temporal": 0.2},
        env="PREDICTION_ENSEMBLE_WEIGHTS",
    )

    # Feature engineering
    feature_extraction_methods: List[str] = Field(
        default=["tfidf", "embeddings", "legal_features", "temporal_features"],
        env="PREDICTION_FEATURE_METHODS",
    )
    max_features: int = Field(default=10000, env="PREDICTION_MAX_FEATURES")
    ngram_range: tuple = Field(default=(1, 3), env="PREDICTION_NGRAM_RANGE")

    # Temporal prediction settings
    temporal_window_days: int = Field(default=365, env="PREDICTION_TEMPORAL_WINDOW")
    temporal_decay_rate: float = Field(default=0.1, env="PREDICTION_TEMPORAL_DECAY")
    seasonal_adjustment: bool = Field(
        default=True, env="PREDICTION_SEASONAL_ADJUSTMENT"
    )

    # Performance settings
    batch_size: int = Field(default=32, env="PREDICTION_BATCH_SIZE")
    max_workers: int = Field(default=4, env="PREDICTION_MAX_WORKERS")
    use_multiprocessing: bool = Field(
        default=True, env="PREDICTION_USE_MULTIPROCESSING"
    )
    memory_limit_gb: float = Field(default=8.0, env="PREDICTION_MEMORY_LIMIT")

    # Training settings
    max_epochs: int = Field(default=100, env="PREDICTION_MAX_EPOCHS")
    early_stopping_patience: int = Field(
        default=10, env="PREDICTION_EARLY_STOPPING_PATIENCE"
    )
    learning_rate: float = Field(default=0.001, env="PREDICTION_LEARNING_RATE")
    weight_decay: float = Field(default=0.01, env="PREDICTION_WEIGHT_DECAY")

    # Evaluation metrics
    evaluation_metrics: List[str] = Field(
        default=["accuracy", "precision", "recall", "f1", "auc_roc"],
        env="PREDICTION_EVALUATION_METRICS",
    )
    confidence_threshold: float = Field(
        default=0.8, env="PREDICTION_CONFIDENCE_THRESHOLD"
    )

    # Legal document specific settings
    legal_categories: List[str] = Field(
        default=[
            "constitutional",
            "civil",
            "criminal",
            "administrative",
            "tax",
            "labor",
        ],
        env="PREDICTION_LEGAL_CATEGORIES",
    )

    amendment_impact_weights: Dict[str, float] = Field(
        default={
            "major_reform": 3.0,
            "amendment": 1.5,
            "technical_change": 0.8,
            "clarification": 0.5,
        },
        env="PREDICTION_AMENDMENT_WEIGHTS",
    )

    # Caching and storage
    cache_predictions: bool = Field(default=True, env="PREDICTION_CACHE_PREDICTIONS")
    cache_ttl: int = Field(default=3600, env="PREDICTION_CACHE_TTL")  # 1 hour
    save_model_checkpoints: bool = Field(
        default=True, env="PREDICTION_SAVE_CHECKPOINTS"
    )
    checkpoint_frequency: int = Field(default=10, env="PREDICTION_CHECKPOINT_FREQUENCY")

    # Monitoring and logging
    log_predictions: bool = Field(default=True, env="PREDICTION_LOG_PREDICTIONS")
    track_model_drift: bool = Field(default=True, env="PREDICTION_TRACK_DRIFT")
    drift_threshold: float = Field(default=0.1, env="PREDICTION_DRIFT_THRESHOLD")

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def model_path(self) -> str:
        """Get full model cache path."""
        return os.path.abspath(self.model_cache_dir)

    @classmethod
    def for_gpu(cls) -> "PredictionConfig":
        """Create configuration optimized for GPU processing."""
        return cls(
            device="cuda",
            batch_size=64,
            use_multiprocessing=False,  # GPU handles parallelization
            max_workers=2,
            memory_limit_gb=6.0,  # GTX 1060 6GB
        )

    @classmethod
    def for_cpu(cls) -> "PredictionConfig":
        """Create configuration optimized for CPU processing."""
        return cls(
            device="cpu",
            batch_size=16,
            use_multiprocessing=True,
            max_workers=4,
            memory_limit_gb=8.0,
        )

    @classmethod
    def for_production(cls) -> "PredictionConfig":
        """Create configuration for production environment."""
        return cls(
            device="auto",
            batch_size=32,
            cache_predictions=True,
            save_model_checkpoints=True,
            track_model_drift=True,
            confidence_threshold=0.85,
            similarity_threshold=0.8,
            max_workers=2,
            log_predictions=True,
        )

    @classmethod
    def for_training(cls) -> "PredictionConfig":
        """Create configuration optimized for model training."""
        return cls(
            device="auto",
            batch_size=16,
            max_epochs=200,
            early_stopping_patience=15,
            cross_validation_folds=10,
            save_model_checkpoints=True,
            checkpoint_frequency=5,
            use_multiprocessing=True,
        )


def get_prediction_config() -> PredictionConfig:
    """Get prediction configuration based on environment."""
    env = os.getenv("ENVIRONMENT", "development")
    device_preference = os.getenv("PREDICTION_DEVICE_PREFERENCE", "auto")
    mode = os.getenv("PREDICTION_MODE", "inference")  # inference, training

    if env == "production":
        return PredictionConfig.for_production()
    elif mode == "training":
        return PredictionConfig.for_training()
    elif device_preference == "gpu":
        return PredictionConfig.for_gpu()
    elif device_preference == "cpu":
        return PredictionConfig.for_cpu()
    else:
        return PredictionConfig()
