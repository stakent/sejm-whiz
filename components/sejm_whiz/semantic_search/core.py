"""High-level semantic search integration and services."""

import logging
from typing import List, Optional, Dict, Any, Union
from uuid import UUID

from .search_engine import get_search_engine, SearchResult
from .indexer import get_document_indexer, IndexingResult
from .ranker import get_result_ranker, RankingConfig, RankingStrategy
from .cross_register import get_cross_register_matcher
from sejm_whiz.vector_db import DistanceMetric
from sejm_whiz.database.models import LegalDocument

logger = logging.getLogger(__name__)


class SemanticSearchService:
    """High-level semantic search service integrating all components."""

    def __init__(
        self,
        search_engine=None,
        indexer=None,
        ranker=None,
        cross_register_matcher=None,
        ranking_config=None,
    ):
        """Initialize semantic search service.

        Args:
            search_engine: Semantic search engine instance
            indexer: Document indexer instance
            ranker: Result ranker instance
            cross_register_matcher: Cross-register matcher instance
            ranking_config: Ranking configuration
        """
        self.search_engine = search_engine or get_search_engine()
        self.indexer = indexer or get_document_indexer()
        self.ranker = ranker or get_result_ranker(ranking_config)
        self.cross_register_matcher = (
            cross_register_matcher or get_cross_register_matcher()
        )
        self.logger = logging.getLogger(__name__)

    def search_documents(
        self,
        query: str,
        limit: int = 10,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        document_type: Optional[str] = None,
        legal_domain: Optional[str] = None,
        similarity_threshold: float = 0.0,
        ranking_strategy: RankingStrategy = RankingStrategy.COMPOSITE,
        include_cross_register: bool = False,
    ) -> List[SearchResult]:
        """Comprehensive document search with ranking and cross-register matching.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            distance_metric: Vector distance metric to use
            document_type: Filter by document type (optional)
            legal_domain: Filter by legal domain (optional)
            similarity_threshold: Minimum similarity score threshold
            ranking_strategy: Strategy for ranking results
            include_cross_register: Whether to include cross-register analysis

        Returns:
            List of ranked SearchResult objects
        """
        try:
            self.logger.info(
                f"Starting comprehensive search for query: '{query[:100]}...'"
            )

            # Perform semantic search
            search_results = self.search_engine.search(
                query=query,
                limit=limit * 2,  # Get more candidates for ranking
                distance_metric=distance_metric,
                document_type=document_type,
                legal_domain=legal_domain,
                similarity_threshold=similarity_threshold,
                include_passages=True,
            )

            if not search_results:
                self.logger.info("No search results found")
                return []

            # Apply ranking
            if ranking_strategy != RankingStrategy.SIMILARITY_ONLY:
                # Update ranker strategy if needed
                if self.ranker.config.strategy != ranking_strategy:
                    ranking_config = RankingConfig(strategy=ranking_strategy)
                    self.ranker = get_result_ranker(ranking_config)

                search_results = self.ranker.rank_results(search_results, query)

            # Limit results after ranking
            search_results = search_results[:limit]

            # Add cross-register analysis if requested
            if include_cross_register:
                search_results = self._add_cross_register_analysis(
                    query, search_results
                )

            self.logger.info(f"Search completed with {len(search_results)} results")
            return search_results

        except Exception as e:
            self.logger.error(f"Comprehensive search failed: {e}")
            raise

    def find_similar_documents(
        self,
        document_id: UUID,
        limit: int = 10,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        ranking_strategy: RankingStrategy = RankingStrategy.TEMPORAL_BOOST,
    ) -> List[SearchResult]:
        """Find documents similar to a given document with ranking.

        Args:
            document_id: ID of the reference document
            limit: Maximum number of results to return
            distance_metric: Vector distance metric to use
            ranking_strategy: Strategy for ranking results

        Returns:
            List of ranked SearchResult objects
        """
        try:
            # Find similar documents
            search_results = self.search_engine.find_similar_to_document(
                document_id=document_id,
                limit=limit * 2,
                distance_metric=distance_metric,
                exclude_self=True,
            )

            if not search_results:
                return []

            # Apply ranking
            if ranking_strategy != RankingStrategy.SIMILARITY_ONLY:
                ranking_config = RankingConfig(strategy=ranking_strategy)
                ranker = get_result_ranker(ranking_config)
                search_results = ranker.rank_results(search_results)

            return search_results[:limit]

        except Exception as e:
            self.logger.error(f"Similar documents search failed: {e}")
            raise

    def index_documents(
        self,
        documents: List[LegalDocument],
        batch_size: int = 32,
        overwrite_existing: bool = False,
    ) -> List[IndexingResult]:
        """Index documents for semantic search.

        Args:
            documents: List of legal documents to index
            batch_size: Size of processing batches
            overwrite_existing: Whether to overwrite existing embeddings

        Returns:
            List of IndexingResult objects
        """
        try:
            return self.indexer.batch_index_documents(
                documents=documents,
                batch_size=batch_size,
                overwrite_existing=overwrite_existing,
                show_progress=True,
            )

        except Exception as e:
            self.logger.error(f"Document indexing failed: {e}")
            raise

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search system statistics.

        Returns:
            Dictionary with system statistics
        """
        try:
            indexing_stats = self.indexer.get_indexing_statistics()

            return {
                "indexing": indexing_stats,
                "search_engine": {
                    "available": True,
                    "embedding_model": "allegro/herbert-klej-cased-v1",
                    "embedding_dimensions": 768,
                },
                "ranking": {
                    "strategy": self.ranker.config.strategy.value,
                    "temporal_decay_days": self.ranker.config.temporal_decay_days,
                    "document_type_weights": self.ranker.config.document_type_weights,
                },
                "cross_register": {
                    "available": True,
                    "mapping_count": len(self.cross_register_matcher.register_mappings),
                },
            }

        except Exception as e:
            self.logger.error(f"Failed to get search statistics: {e}")
            return {}

    def _add_cross_register_analysis(
        self, query: str, search_results: List[SearchResult]
    ) -> List[SearchResult]:
        """Add cross-register analysis to search results."""
        try:
            for result in search_results:
                # Perform cross-register matching between query and document
                matches = self.cross_register_matcher.match_registers(
                    legal_text=result.document.content,
                    parliamentary_text=query,
                    similarity_threshold=0.6,
                    confidence_threshold=0.5,
                )

                # Add cross-register matches to result metadata
                if matches:
                    result.search_metadata["cross_register_matches"] = [
                        match.to_dict()
                        for match in matches[:3]  # Top 3 matches
                    ]
                    result.search_metadata["cross_register_score"] = max(
                        match.similarity_score for match in matches
                    )

                    # Boost similarity score based on cross-register matches
                    max_cross_score = max(match.similarity_score for match in matches)
                    boost_factor = 1.0 + (max_cross_score * 0.1)  # 10% max boost
                    result.similarity_score *= boost_factor
                else:
                    result.search_metadata["cross_register_matches"] = []
                    result.search_metadata["cross_register_score"] = 0.0

            return search_results

        except Exception as e:
            self.logger.warning(f"Cross-register analysis failed: {e}")
            return search_results


# High-level convenience functions


def process_search_query(
    query: str,
    limit: int = 10,
    document_type: Optional[str] = None,
    legal_domain: Optional[str] = None,
    ranking_strategy: RankingStrategy = RankingStrategy.COMPOSITE,
) -> List[Dict[str, Any]]:
    """Process a search query and return results as dictionaries.

    Args:
        query: Search query text
        limit: Maximum number of results to return
        document_type: Filter by document type (optional)
        legal_domain: Filter by legal domain (optional)
        ranking_strategy: Strategy for ranking results

    Returns:
        List of search result dictionaries
    """
    service = get_semantic_search_service()
    results = service.search_documents(
        query=query,
        limit=limit,
        document_type=document_type,
        legal_domain=legal_domain,
        ranking_strategy=ranking_strategy,
    )

    return [result.to_dict() for result in results]


def search_similar_documents(
    document_id: Union[str, UUID],
    limit: int = 10,
    ranking_strategy: RankingStrategy = RankingStrategy.TEMPORAL_BOOST,
) -> List[Dict[str, Any]]:
    """Search for documents similar to a given document.

    Args:
        document_id: ID of the reference document
        limit: Maximum number of results to return
        ranking_strategy: Strategy for ranking results

    Returns:
        List of search result dictionaries
    """
    if isinstance(document_id, str):
        document_id = UUID(document_id)

    service = get_semantic_search_service()
    results = service.find_similar_documents(
        document_id=document_id,
        limit=limit,
        ranking_strategy=ranking_strategy,
    )

    return [result.to_dict() for result in results]


# Singleton instance
_semantic_search_service_instance = None


def get_semantic_search_service(
    ranking_config: Optional[RankingConfig] = None,
) -> SemanticSearchService:
    """Get singleton semantic search service instance."""
    global _semantic_search_service_instance
    if _semantic_search_service_instance is None or ranking_config is not None:
        _semantic_search_service_instance = SemanticSearchService(
            ranking_config=ranking_config
        )
    return _semantic_search_service_instance
