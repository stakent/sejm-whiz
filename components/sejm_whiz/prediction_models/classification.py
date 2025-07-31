"""Text classification models for legal document prediction."""

import logging
from typing import Dict, List, Any
from datetime import datetime
import numpy as np
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import joblib
import os

from .config import PredictionConfig
from .core import (
    PredictionInput,
    PredictionResult,
    TrainingData,
    ModelMetrics,
    ModelType,
    PredictionType,
    FeatureVector,
)

logger = logging.getLogger(__name__)


class BaseClassifier(ABC):
    """Base class for classification models."""

    def __init__(self, config: PredictionConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_extractor = None
        self.is_trained = False
        self.model_version = "1.0"
        self.feature_names: List[str] = []

    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying classification model."""
        pass

    @abstractmethod
    def extract_features(self, input_data: PredictionInput) -> FeatureVector:
        """Extract features from input data."""
        pass

    def train(self, training_data: TrainingData) -> ModelMetrics:
        """Train the classification model."""
        if self.model is None:
            self.model = self._create_model()

        # Initialize scaler if needed
        if self.scaler is None:
            self.scaler = StandardScaler()

        # Fit scaler and transform features
        X_scaled = self.scaler.fit_transform(training_data.features)
        y = training_data.labels

        # Store feature names
        self.feature_names = training_data.feature_names.copy()

        # Train model
        start_time = datetime.now()
        self.model.fit(X_scaled, y)
        training_time = (datetime.now() - start_time).total_seconds()

        # Evaluate model
        metrics = self._evaluate_model(X_scaled, y, training_time)

        self.is_trained = True
        logger.info(
            f"Trained {self.__class__.__name__} with accuracy: {metrics.accuracy:.3f}"
        )

        return metrics

    def predict(self, input_data: PredictionInput) -> PredictionResult:
        """Make prediction using the trained classifier."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Extract features
        feature_vector = self.extract_features(input_data)
        features_array = feature_vector.to_array(self.feature_names)

        # Scale features
        features_scaled = self.scaler.transform(features_array.reshape(1, -1))

        # Make prediction
        start_time = datetime.now()

        if hasattr(self.model, "predict_proba"):
            # For models that support probability prediction
            probabilities = self.model.predict_proba(features_scaled)[0]
            prediction_value = float(probabilities[1])  # Assuming binary classification
            confidence = float(max(probabilities))
        else:
            # For models that only support hard predictions
            prediction_value = float(self.model.predict(features_scaled)[0])
            confidence = 0.8  # Default confidence for hard predictions

        inference_time = (datetime.now() - start_time).total_seconds()

        return PredictionResult(
            prediction_id=f"{self.__class__.__name__.lower()}_{input_data.document_id}_{datetime.now().isoformat()}",
            document_id=input_data.document_id,
            prediction_type=PredictionType.CATEGORY_CLASSIFICATION,
            model_type=ModelType.CLASSIFICATION,
            prediction=prediction_value,
            confidence=confidence,
            features_used=self.feature_names[:10],  # Show top 10 features
            model_version=self.model_version,
            timestamp=datetime.now(),
            metadata={
                "inference_time": inference_time,
                "feature_count": len(self.feature_names),
            },
        )

    def _evaluate_model(
        self, X: np.ndarray, y: np.ndarray, training_time: float
    ) -> ModelMetrics:
        """Evaluate model performance."""
        try:
            # Cross-validation scores
            cv_scores = cross_val_score(
                self.model,
                X,
                y,
                cv=self.config.cross_validation_folds,
                scoring="accuracy",
            )

            # Split for detailed evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )

            # Make predictions on test set
            y_pred = self.model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            # AUC-ROC if model supports probability prediction
            auc_roc = None
            if hasattr(self.model, "predict_proba") and len(np.unique(y)) == 2:
                try:
                    y_proba = self.model.predict_proba(X_test)[:, 1]
                    auc_roc = roc_auc_score(y_test, y_proba)
                except Exception as e:
                    logger.warning(f"Could not calculate AUC-ROC: {e}")

            # Feature importance if available
            feature_importance = None
            if hasattr(self.model, "feature_importances_"):
                feature_importance = dict(
                    zip(self.feature_names, self.model.feature_importances_)
                )
            elif hasattr(self.model, "coef_"):
                feature_importance = dict(
                    zip(
                        self.feature_names,
                        np.abs(self.model.coef_[0])
                        if len(self.model.coef_.shape) > 1
                        else np.abs(self.model.coef_),
                    )
                )

            return ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc_roc=auc_roc,
                feature_importance=feature_importance,
                cross_val_scores=cv_scores.tolist(),
                training_time=training_time,
            )

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return ModelMetrics(
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                training_time=training_time,
            )

    def save_model(self, filepath: str) -> None:
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_version": self.model_version,
            "config": self.config.dict(),
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"Saved model to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_names = model_data["feature_names"]
        self.model_version = model_data.get("model_version", "1.0")

        self.is_trained = True
        logger.info(f"Loaded model from {filepath}")


class RandomForestLegalClassifier(BaseClassifier):
    """Random Forest classifier for legal documents."""

    def _create_model(self) -> RandomForestClassifier:
        """Create Random Forest model."""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.config.random_state,
            n_jobs=self.config.max_workers,
        )

    def extract_features(self, input_data: PredictionInput) -> FeatureVector:
        """Extract features for Random Forest."""
        features = {}

        # Text length features
        features["text_length"] = len(input_data.text)
        features["word_count"] = len(input_data.text.split())
        features["sentence_count"] = (
            input_data.text.count(".")
            + input_data.text.count("!")
            + input_data.text.count("?")
        )

        # Legal document features
        if input_data.legal_category:
            features[f"category_{input_data.legal_category}"] = 1.0

        # Keyword presence features
        legal_keywords = [
            "ustawa",
            "rozporządzenie",
            "konstytucja",
            "kodeks",
            "nowelizacja",
            "amendment",
            "article",
            "paragraph",
            "section",
            "clause",
        ]

        text_lower = input_data.text.lower()
        for keyword in legal_keywords:
            features[f"keyword_{keyword}"] = float(keyword in text_lower)

        # Metadata features
        if input_data.metadata:
            for key, value in input_data.metadata.items():
                if isinstance(value, (int, float)):
                    features[f"meta_{key}"] = float(value)
                elif isinstance(value, bool):
                    features[f"meta_{key}"] = float(value)

        return FeatureVector(
            document_id=input_data.document_id,
            features=features,
            embeddings=input_data.embeddings,
        )


class GradientBoostingLegalClassifier(BaseClassifier):
    """Gradient Boosting classifier for legal documents."""

    def _create_model(self) -> GradientBoostingClassifier:
        """Create Gradient Boosting model."""
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.config.random_state,
        )

    def extract_features(self, input_data: PredictionInput) -> FeatureVector:
        """Extract features for Gradient Boosting."""
        # Same feature extraction as Random Forest
        features = {}

        features["text_length"] = len(input_data.text)
        features["word_count"] = len(input_data.text.split())
        features["sentence_count"] = (
            input_data.text.count(".")
            + input_data.text.count("!")
            + input_data.text.count("?")
        )

        if input_data.legal_category:
            features[f"category_{input_data.legal_category}"] = 1.0

        legal_keywords = [
            "ustawa",
            "rozporządzenie",
            "konstytucja",
            "kodeks",
            "nowelizacja",
            "amendment",
            "article",
            "paragraph",
            "section",
            "clause",
        ]

        text_lower = input_data.text.lower()
        for keyword in legal_keywords:
            features[f"keyword_{keyword}"] = float(keyword in text_lower)

        if input_data.metadata:
            for key, value in input_data.metadata.items():
                if isinstance(value, (int, float)):
                    features[f"meta_{key}"] = float(value)
                elif isinstance(value, bool):
                    features[f"meta_{key}"] = float(value)

        return FeatureVector(
            document_id=input_data.document_id,
            features=features,
            embeddings=input_data.embeddings,
        )


class SVMLegalClassifier(BaseClassifier):
    """Support Vector Machine classifier for legal documents."""

    def _create_model(self) -> SVC:
        """Create SVM model."""
        return SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,  # Enable probability prediction
            random_state=self.config.random_state,
        )

    def extract_features(self, input_data: PredictionInput) -> FeatureVector:
        """Extract features for SVM."""
        features = {}

        # Statistical features
        features["text_length"] = len(input_data.text)
        features["word_count"] = len(input_data.text.split())
        features["avg_word_length"] = (
            np.mean([len(word) for word in input_data.text.split()])
            if input_data.text.split()
            else 0
        )

        # Character-level features
        features["uppercase_ratio"] = (
            sum(1 for c in input_data.text if c.isupper()) / len(input_data.text)
            if input_data.text
            else 0
        )
        features["digit_ratio"] = (
            sum(1 for c in input_data.text if c.isdigit()) / len(input_data.text)
            if input_data.text
            else 0
        )
        features["punctuation_ratio"] = (
            sum(1 for c in input_data.text if not c.isalnum() and not c.isspace())
            / len(input_data.text)
            if input_data.text
            else 0
        )

        # Legal-specific features
        if input_data.legal_category:
            features[f"category_{input_data.legal_category}"] = 1.0

        # Legal structure indicators
        legal_markers = ["art.", "par.", "§", "ust.", "pkt", "lit."]
        for marker in legal_markers:
            features[f"marker_{marker}"] = float(marker in input_data.text.lower())

        return FeatureVector(
            document_id=input_data.document_id,
            features=features,
            embeddings=input_data.embeddings,
        )


class LogisticRegressionLegalClassifier(BaseClassifier):
    """Logistic Regression classifier for legal documents."""

    def _create_model(self) -> LogisticRegression:
        """Create Logistic Regression model."""
        return LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="liblinear",
            random_state=self.config.random_state,
            max_iter=1000,
        )

    def extract_features(self, input_data: PredictionInput) -> FeatureVector:
        """Extract features for Logistic Regression."""
        features = {}

        # Basic text features
        features["text_length"] = len(input_data.text)
        features["word_count"] = len(input_data.text.split())

        # TF-IDF features would be added here in a real implementation
        # For now, using simple keyword-based features

        # Legal terminology features
        legal_terms = [
            "ustawa",
            "rozporządzenie",
            "nowelizacja",
            "zmiana",
            "uchylenie",
            "dodanie",
            "zastąpienie",
            "skreślenie",
            "wejście w życie",
        ]

        text_lower = input_data.text.lower()
        for term in legal_terms:
            features[f"term_{term}"] = float(term in text_lower)

        # Document structure features
        features["has_articles"] = float(
            "art." in text_lower or "artykuł" in text_lower
        )
        features["has_paragraphs"] = float(
            "§" in input_data.text or "paragraf" in text_lower
        )
        features["has_amendments"] = float(
            "nowelizacja" in text_lower or "zmienia" in text_lower
        )

        return FeatureVector(
            document_id=input_data.document_id,
            features=features,
            embeddings=input_data.embeddings,
        )


class TfidfEmbeddingClassifier(BaseClassifier):
    """Classifier using TF-IDF features combined with embeddings."""

    def __init__(self, config: PredictionConfig):
        super().__init__(config)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=config.max_features,
            ngram_range=config.ngram_range,
            stop_words=None,  # Keep all words for legal documents
            lowercase=True,
        )
        self.base_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=config.random_state,
            n_jobs=config.max_workers,
        )

    def _create_model(self) -> Any:
        """Create the underlying model."""
        return self.base_classifier

    def extract_features(self, input_data: PredictionInput) -> FeatureVector:
        """Extract TF-IDF and embedding features."""
        features = {}

        # Basic features
        features["text_length"] = len(input_data.text)
        features["word_count"] = len(input_data.text.split())

        # Add embedding features if available
        if input_data.embeddings is not None:
            for i, val in enumerate(
                input_data.embeddings[:50]
            ):  # Use first 50 embedding dimensions
                features[f"embedding_{i}"] = float(val)

        return FeatureVector(
            document_id=input_data.document_id,
            features=features,
            embeddings=input_data.embeddings,
        )

    def train(self, training_data: TrainingData) -> ModelMetrics:
        """Train with TF-IDF features."""
        # For now, proceed with standard training
        # TF-IDF features would be integrated here in full implementation
        return super().train(training_data)


def get_classifier(config: PredictionConfig, classifier_type: str) -> BaseClassifier:
    """Factory function to create classifiers."""
    classifiers = {
        "random_forest": RandomForestLegalClassifier,
        "gradient_boosting": GradientBoostingLegalClassifier,
        "svm": SVMLegalClassifier,
        "logistic_regression": LogisticRegressionLegalClassifier,
        "tfidf_embedding": TfidfEmbeddingClassifier,
    }

    if classifier_type not in classifiers:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    return classifiers[classifier_type](config)


def create_classifier_ensemble(config: PredictionConfig) -> List[BaseClassifier]:
    """Create ensemble of multiple classifiers."""
    classifiers = []

    for model_name in config.classification_models:
        try:
            classifier = get_classifier(config, model_name)
            classifiers.append(classifier)
            logger.info(f"Added {model_name} to classifier ensemble")
        except Exception as e:
            logger.error(f"Error creating classifier {model_name}: {e}")

    return classifiers


def train_classifier_ensemble(
    classifiers: List[BaseClassifier], training_data: TrainingData
) -> Dict[str, ModelMetrics]:
    """Train multiple classifiers and return their metrics."""
    results = {}

    for classifier in classifiers:
        try:
            logger.info(f"Training {classifier.__class__.__name__}")
            metrics = classifier.train(training_data)
            results[classifier.__class__.__name__] = metrics
        except Exception as e:
            logger.error(f"Error training {classifier.__class__.__name__}: {e}")

    return results
