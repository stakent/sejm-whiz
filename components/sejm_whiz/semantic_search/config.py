"""Configuration for semantic search operations."""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from sejm_whiz.vector_db import DistanceMetric


class SearchMode(Enum):
    """Search operation modes."""

    SEMANTIC_ONLY = "semantic_only"
    CROSS_REGISTER = "cross_register"
    HYBRID = "hybrid"
    LEGAL_FOCUSED = "legal_focused"


class RelevanceMode(Enum):
    """Relevance calculation modes."""

    COSINE_SIMILARITY = "cosine_similarity"
    EUCLIDEAN_DISTANCE = "euclidean_distance"
    DOT_PRODUCT = "dot_product"
    HYBRID_SCORING = "hybrid_scoring"


@dataclass
class SearchConfig:
    """Configuration for semantic search operations."""

    # Basic search parameters
    max_results: int = field(
        default_factory=lambda: int(os.getenv("SEARCH_MAX_RESULTS", "50"))
    )
    similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("SEARCH_SIMILARITY_THRESHOLD", "0.7"))
    )
    distance_metric: DistanceMetric = field(
        default_factory=lambda: DistanceMetric.COSINE
    )
    search_mode: SearchMode = field(default_factory=lambda: SearchMode.HYBRID)
    relevance_mode: RelevanceMode = field(
        default_factory=lambda: RelevanceMode.HYBRID_SCORING
    )

    # Performance optimization
    enable_caching: bool = field(
        default_factory=lambda: os.getenv("SEARCH_ENABLE_CACHING", "true").lower()
        == "true"
    )
    cache_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("SEARCH_CACHE_TTL", "3600"))
    )
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("SEARCH_BATCH_SIZE", "100"))
    )
    parallel_processing: bool = field(
        default_factory=lambda: os.getenv("SEARCH_PARALLEL_PROCESSING", "true").lower()
        == "true"
    )

    # Cross-register matching
    enable_cross_register: bool = field(
        default_factory=lambda: os.getenv(
            "SEARCH_ENABLE_CROSS_REGISTER", "true"
        ).lower()
        == "true"
    )
    cross_register_boost: float = field(
        default_factory=lambda: float(os.getenv("SEARCH_CROSS_REGISTER_BOOST", "1.2"))
    )

    # Legal domain specificity
    legal_term_boost: float = field(
        default_factory=lambda: float(os.getenv("SEARCH_LEGAL_TERM_BOOST", "1.5"))
    )
    document_type_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "ustawa": 1.5,  # Acts
            "rozporządzenie": 1.3,  # Regulations
            "obwieszczenie": 1.1,  # Announcements
            "uchwała": 1.2,  # Resolutions
            "inne": 1.0,  # Other
        }
    )

    # Temporal relevance
    enable_temporal_boost: bool = field(
        default_factory=lambda: os.getenv(
            "SEARCH_ENABLE_TEMPORAL_BOOST", "true"
        ).lower()
        == "true"
    )
    temporal_decay_factor: float = field(
        default_factory=lambda: float(os.getenv("SEARCH_TEMPORAL_DECAY", "0.1"))
    )
    temporal_window_days: int = field(
        default_factory=lambda: int(os.getenv("SEARCH_TEMPORAL_WINDOW", "365"))
    )

    # Query processing
    enable_query_expansion: bool = field(
        default_factory=lambda: os.getenv(
            "SEARCH_ENABLE_QUERY_EXPANSION", "true"
        ).lower()
        == "true"
    )
    query_expansion_terms: int = field(
        default_factory=lambda: int(os.getenv("SEARCH_QUERY_EXPANSION_TERMS", "5"))
    )

    # Result filtering
    min_document_length: int = field(
        default_factory=lambda: int(os.getenv("SEARCH_MIN_DOC_LENGTH", "100"))
    )
    max_document_age_days: Optional[int] = field(
        default_factory=lambda: (
            int(env_val)
            if (env_val := os.getenv("SEARCH_MAX_DOC_AGE_DAYS")) is not None
            else None
        )
    )
    allowed_document_types: Optional[List[str]] = field(default_factory=lambda: None)

    # Advanced features
    enable_semantic_clustering: bool = field(
        default_factory=lambda: os.getenv("SEARCH_ENABLE_CLUSTERING", "false").lower()
        == "true"
    )
    clustering_threshold: float = field(
        default_factory=lambda: float(os.getenv("SEARCH_CLUSTERING_THRESHOLD", "0.8"))
    )

    # Debug and monitoring
    enable_debug_logging: bool = field(
        default_factory=lambda: os.getenv("SEARCH_DEBUG_LOGGING", "false").lower()
        == "true"
    )
    track_search_analytics: bool = field(
        default_factory=lambda: os.getenv("SEARCH_TRACK_ANALYTICS", "true").lower()
        == "true"
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_results <= 0:
            raise ValueError("max_results must be positive")

        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be non-negative")


@dataclass
class RankingParameters:
    """Parameters for result ranking and scoring."""

    # Base scoring weights
    similarity_weight: float = 0.6
    temporal_weight: float = 0.2
    document_type_weight: float = 0.1
    legal_domain_weight: float = 0.1

    # Boosting factors
    exact_match_boost: float = 2.0
    partial_match_boost: float = 1.5
    legal_term_boost: float = 1.3
    recent_document_boost: float = 1.2

    # Penalty factors
    low_quality_penalty: float = 0.8
    duplicate_penalty: float = 0.5
    length_penalty_threshold: int = 10000  # Characters
    length_penalty_factor: float = 0.9

    def __post_init__(self):
        """Validate ranking parameters."""
        total_weight = (
            self.similarity_weight
            + self.temporal_weight
            + self.document_type_weight
            + self.legal_domain_weight
        )

        if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError(f"Ranking weights must sum to 1.0, got {total_weight}")


def get_default_search_config() -> SearchConfig:
    """Get default search configuration."""
    return SearchConfig()


def get_production_search_config() -> SearchConfig:
    """Get production-optimized search configuration."""
    return SearchConfig(
        max_results=100,
        similarity_threshold=0.75,
        enable_caching=True,
        cache_ttl_seconds=7200,  # 2 hours
        batch_size=200,
        parallel_processing=True,
        enable_cross_register=True,
        enable_temporal_boost=True,
        enable_query_expansion=True,
        track_search_analytics=True,
    )


def get_development_search_config() -> SearchConfig:
    """Get development-friendly search configuration."""
    return SearchConfig(
        max_results=20,
        similarity_threshold=0.6,
        enable_caching=False,
        batch_size=50,
        parallel_processing=False,
        enable_debug_logging=True,
        track_search_analytics=False,
    )


def get_default_ranking_parameters() -> RankingParameters:
    """Get default ranking parameters."""
    return RankingParameters()


def get_legal_focused_ranking_parameters() -> RankingParameters:
    """Get ranking parameters optimized for legal document search."""
    return RankingParameters(
        similarity_weight=0.5,
        temporal_weight=0.1,
        document_type_weight=0.2,
        legal_domain_weight=0.2,
        legal_term_boost=2.0,
        exact_match_boost=3.0,
    )
