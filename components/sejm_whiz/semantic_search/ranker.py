"""Result ranking and scoring for semantic search."""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum

from .search_engine import SearchResult

logger = logging.getLogger(__name__)


class RankingStrategy(Enum):
    """Available ranking strategies."""

    SIMILARITY_ONLY = "similarity_only"
    TEMPORAL_BOOST = "temporal_boost"
    DOCUMENT_TYPE_BOOST = "document_type_boost"
    LEGAL_DOMAIN_BOOST = "legal_domain_boost"
    COMPOSITE = "composite"


@dataclass
class RankingConfig:
    """Configuration for result ranking."""

    strategy: RankingStrategy = RankingStrategy.COMPOSITE

    # Temporal boost parameters
    temporal_decay_days: int = 365
    temporal_weight: float = 0.2
    recency_boost_days: int = 30
    recency_boost_factor: float = 1.5

    # Document type boost parameters
    document_type_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "ustawa": 1.2,
            "kodeks": 1.3,
            "rozporządzenie": 1.0,
            "dekret": 0.9,
            "uchwała": 0.8,
        }
    )

    # Legal domain boost parameters
    legal_domain_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "prawo konstytucyjne": 1.3,
            "prawo cywilne": 1.2,
            "prawo karne": 1.2,
            "prawo administracyjne": 1.1,
            "prawo pracy": 1.0,
            "prawo podatkowe": 1.0,
        }
    )

    # Amendment boost parameters
    amendment_boost: float = 1.1
    multi_act_amendment_boost: float = 1.2

    # Passage matching boost
    passage_match_weight: float = 0.3

    # Minimum and maximum final scores
    min_score: float = 0.0
    max_score: float = 1.0


class ResultRanker:
    """Result ranking and scoring service."""

    def __init__(self, config: Optional[RankingConfig] = None):
        """Initialize result ranker.

        Args:
            config: Ranking configuration
        """
        self.config = config or RankingConfig()
        self.logger = logging.getLogger(__name__)

    def rank_results(
        self,
        results: List[SearchResult],
        query: Optional[str] = None,
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> List[SearchResult]:
        """Rank search results using configured strategy.

        Args:
            results: List of search results to rank
            query: Original search query (optional)
            custom_weights: Custom weighting factors (optional)

        Returns:
            Ranked list of search results
        """
        try:
            if not results:
                return results

            self.logger.info(
                f"Ranking {len(results)} search results using {self.config.strategy.value}"
            )

            # Apply ranking strategy
            if self.config.strategy == RankingStrategy.SIMILARITY_ONLY:
                ranked_results = self._rank_by_similarity(results)
            elif self.config.strategy == RankingStrategy.TEMPORAL_BOOST:
                ranked_results = self._rank_with_temporal_boost(results)
            elif self.config.strategy == RankingStrategy.DOCUMENT_TYPE_BOOST:
                ranked_results = self._rank_with_document_type_boost(results)
            elif self.config.strategy == RankingStrategy.LEGAL_DOMAIN_BOOST:
                ranked_results = self._rank_with_legal_domain_boost(results)
            else:  # COMPOSITE
                ranked_results = self._rank_composite(results, query, custom_weights)

            self.logger.debug(
                f"Ranking completed, top result score: {ranked_results[0].similarity_score:.4f}"
            )
            return ranked_results

        except Exception as e:
            self.logger.error(f"Result ranking failed: {e}")
            # Return original results if ranking fails
            return results

    def _rank_by_similarity(self, results: List[SearchResult]) -> List[SearchResult]:
        """Rank results by similarity score only."""
        return sorted(results, key=lambda r: r.similarity_score, reverse=True)

    def _rank_with_temporal_boost(
        self, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Rank results with temporal boosting for recent documents."""
        current_time = datetime.now(UTC)

        for result in results:
            temporal_score = self._calculate_temporal_score(
                result.document.published_at, current_time
            )

            # Apply temporal boost
            boosted_score = (
                result.similarity_score * (1 - self.config.temporal_weight)
                + temporal_score * self.config.temporal_weight
            )

            # Update result metadata
            result.search_metadata["temporal_score"] = temporal_score
            result.search_metadata["boosted_score"] = boosted_score
            result.similarity_score = boosted_score

        return sorted(results, key=lambda r: r.similarity_score, reverse=True)

    def _rank_with_document_type_boost(
        self, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Rank results with document type boosting."""
        for result in results:
            doc_type = result.document.document_type
            type_weight = self.config.document_type_weights.get(doc_type, 1.0)

            boosted_score = result.similarity_score * type_weight

            # Update result metadata
            result.search_metadata["document_type_weight"] = type_weight
            result.search_metadata["boosted_score"] = boosted_score
            result.similarity_score = boosted_score

        return sorted(results, key=lambda r: r.similarity_score, reverse=True)

    def _rank_with_legal_domain_boost(
        self, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Rank results with legal domain boosting."""
        for result in results:
            domain = result.document.legal_domain
            domain_weight = self.config.legal_domain_weights.get(domain, 1.0)

            boosted_score = result.similarity_score * domain_weight

            # Update result metadata
            result.search_metadata["legal_domain_weight"] = domain_weight
            result.search_metadata["boosted_score"] = boosted_score
            result.similarity_score = boosted_score

        return sorted(results, key=lambda r: r.similarity_score, reverse=True)

    def _rank_composite(
        self,
        results: List[SearchResult],
        query: Optional[str] = None,
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> List[SearchResult]:
        """Rank results using composite scoring strategy."""
        current_time = datetime.now(UTC)

        for result in results:
            # Base similarity score
            base_score = result.similarity_score

            # Temporal score
            temporal_score = self._calculate_temporal_score(
                result.document.published_at, current_time
            )

            # Document type weight
            doc_type = result.document.document_type
            type_weight = self.config.document_type_weights.get(doc_type, 1.0)

            # Legal domain weight
            domain = result.document.legal_domain
            domain_weight = self.config.legal_domain_weights.get(domain, 1.0)

            # Amendment boost
            amendment_boost = 1.0
            if result.document.is_amendment:
                amendment_boost = self.config.amendment_boost
                if result.document.affects_multiple_acts:
                    amendment_boost = self.config.multi_act_amendment_boost

            # Passage matching boost
            passage_boost = 1.0
            if result.matched_passages and query:
                passage_boost = (
                    1.0
                    + self._calculate_passage_relevance(result.matched_passages, query)
                    * self.config.passage_match_weight
                )

            # Apply custom weights if provided
            if custom_weights:
                type_weight *= custom_weights.get("document_type", 1.0)
                domain_weight *= custom_weights.get("legal_domain", 1.0)
                temporal_score *= custom_weights.get("temporal", 1.0)

            # Calculate composite score
            composite_score = (
                (
                    base_score * (1 - self.config.temporal_weight)
                    + temporal_score * self.config.temporal_weight
                )
                * type_weight
                * domain_weight
                * amendment_boost
                * passage_boost
            )

            # Clamp to configured range
            composite_score = max(
                self.config.min_score, min(self.config.max_score, composite_score)
            )

            # Update result metadata
            result.search_metadata.update(
                {
                    "base_similarity_score": base_score,
                    "temporal_score": temporal_score,
                    "document_type_weight": type_weight,
                    "legal_domain_weight": domain_weight,
                    "amendment_boost": amendment_boost,
                    "passage_boost": passage_boost,
                    "composite_score": composite_score,
                }
            )

            result.similarity_score = composite_score

        return sorted(results, key=lambda r: r.similarity_score, reverse=True)

    def _calculate_temporal_score(
        self, published_at: Optional[datetime], current_time: datetime
    ) -> float:
        """Calculate temporal relevance score."""
        if not published_at:
            return 0.5  # Neutral score for documents without publication date

        days_old = (current_time - published_at).days

        # Apply recency boost for very recent documents
        if days_old <= self.config.recency_boost_days:
            return min(
                1.0,
                self.config.recency_boost_factor
                * (1.0 - days_old / self.config.recency_boost_days),
            )

        # Apply temporal decay for older documents
        if days_old > self.config.temporal_decay_days:
            return 0.1  # Minimum score for very old documents

        # Linear decay
        decay_factor = 1.0 - (days_old - self.config.recency_boost_days) / (
            self.config.temporal_decay_days - self.config.recency_boost_days
        )
        return max(0.1, decay_factor)

    def _calculate_passage_relevance(self, passages: List[str], query: str) -> float:
        """Calculate relevance score based on passage matching."""
        if not passages or not query:
            return 0.0

        query_terms = set(query.lower().split())
        total_relevance = 0.0

        for passage in passages:
            passage_terms = set(passage.lower().split())
            overlap = len(query_terms.intersection(passage_terms))
            relevance = overlap / len(query_terms) if query_terms else 0.0
            total_relevance += relevance

        return min(1.0, total_relevance / len(passages))

    def get_ranking_explanation(self, result: SearchResult) -> Dict[str, Any]:
        """Get detailed explanation of ranking factors for a result.

        Args:
            result: Search result to explain

        Returns:
            Dictionary with ranking explanation
        """
        metadata = result.search_metadata

        explanation = {
            "final_score": result.similarity_score,
            "ranking_strategy": self.config.strategy.value,
            "factors": {
                "base_similarity": metadata.get(
                    "base_similarity_score", result.similarity_score
                ),
                "temporal_score": metadata.get("temporal_score"),
                "document_type_weight": metadata.get("document_type_weight"),
                "legal_domain_weight": metadata.get("legal_domain_weight"),
                "amendment_boost": metadata.get("amendment_boost"),
                "passage_boost": metadata.get("passage_boost"),
            },
            "document_info": {
                "id": str(result.document.id),
                "type": result.document.document_type,
                "domain": result.document.legal_domain,
                "is_amendment": result.document.is_amendment,
                "affects_multiple_acts": result.document.affects_multiple_acts,
                "published_at": result.document.published_at.isoformat()
                if result.document.published_at
                else None,
            },
        }

        return explanation


# Singleton instance
_result_ranker_instance = None


def get_result_ranker(config: Optional[RankingConfig] = None) -> ResultRanker:
    """Get singleton result ranker instance."""
    global _result_ranker_instance
    if _result_ranker_instance is None or config is not None:
        _result_ranker_instance = ResultRanker(config)
    return _result_ranker_instance
