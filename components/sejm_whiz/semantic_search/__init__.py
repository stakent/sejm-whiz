"""Semantic search component for legal document retrieval using embeddings."""

from .search_engine import SemanticSearchEngine, get_search_engine, SearchResult
from .indexer import DocumentIndexer, get_document_indexer, IndexingResult
from .ranker import ResultRanker, get_result_ranker, RankingConfig
from .cross_register import (
    CrossRegisterMatcher,
    get_cross_register_matcher,
    MatchResult,
)
from .core import (
    process_search_query,
    search_similar_documents,
    get_semantic_search_service,
)

__all__ = [
    # Main search engine
    "SemanticSearchEngine",
    "get_search_engine",
    "SearchResult",
    # Document indexing
    "DocumentIndexer",
    "get_document_indexer",
    "IndexingResult",
    # Result ranking
    "ResultRanker",
    "get_result_ranker",
    "RankingConfig",
    # Cross-register matching
    "CrossRegisterMatcher",
    "get_cross_register_matcher",
    "MatchResult",
    # High-level API
    "process_search_query",
    "search_similar_documents",
    "get_semantic_search_service",
]
