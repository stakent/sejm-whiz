"""Tests for semantic search core integration."""

import pytest
from unittest.mock import Mock, patch
from uuid import uuid4

from sejm_whiz.semantic_search.core import (
    SemanticSearchService,
    get_semantic_search_service,
    process_search_query,
    search_similar_documents,
)
from sejm_whiz.semantic_search.search_engine import SearchResult
from sejm_whiz.semantic_search.indexer import IndexingResult
from sejm_whiz.semantic_search.ranker import RankingStrategy, RankingConfig
from sejm_whiz.semantic_search.cross_register import MatchResult
from sejm_whiz.vector_db import DistanceMetric
from sejm_whiz.database.models import LegalDocument


class TestSemanticSearchService:
    """Test semantic search service integration."""
    
    @pytest.fixture
    def mock_search_engine(self):
        """Mock search engine.""" 
        engine = Mock()
        
        # Mock search result
        doc = LegalDocument(
            id=uuid4(),
            title="Test Document",
            content="Test content",
            document_type="ustawa",
            legal_domain="prawo cywilne",
        )
        
        search_result = SearchResult(
            document=doc,
            similarity_score=0.8,
            embedding_distance=0.2,
            matched_passages=["Test passage"],
            search_metadata={"test": "metadata"},
        )
        
        engine.search.return_value = [search_result]
        engine.find_similar_to_document.return_value = [search_result]
        
        return engine
    
    @pytest.fixture
    def mock_indexer(self):
        """Mock document indexer."""
        indexer = Mock()
        
        indexing_result = IndexingResult(
            document_id=uuid4(),
            success=True,
            embedding_dimensions=768,
            processing_time_ms=100.0,
        )
        
        indexer.batch_index_documents.return_value = [indexing_result]
        indexer.get_indexing_statistics.return_value = {
            "total_documents": 100,
            "indexed_documents": 80,
            "indexing_percentage": 80.0,
        }
        
        return indexer
    
    @pytest.fixture
    def mock_ranker(self):
        """Mock result ranker."""
        ranker = Mock()
        ranker.config = RankingConfig()
        
        # Mock ranking returns same results
        def mock_rank_results(results, query=None):
            return results
        
        ranker.rank_results.side_effect = mock_rank_results
        return ranker
    
    @pytest.fixture
    def mock_cross_register_matcher(self):
        """Mock cross-register matcher."""
        matcher = Mock()
        matcher.register_mappings = {"test": ["mapping"]}
        
        match_result = MatchResult(
            formal_text="formal text",
            informal_text="informal text",
            similarity_score=0.7,
            match_type="direct_mapping",
            confidence=0.8,
            normalized_forms={"formal": "formal", "informal": "informal"},
            key_terms=["test"],
            metadata={"test": "data"},
        )
        
        matcher.match_registers.return_value = [match_result]
        return matcher
    
    @pytest.fixture
    def search_service(self, mock_search_engine, mock_indexer, mock_ranker, mock_cross_register_matcher):
        """Create search service with mocked dependencies."""
        return SemanticSearchService(
            search_engine=mock_search_engine,
            indexer=mock_indexer,
            ranker=mock_ranker,
            cross_register_matcher=mock_cross_register_matcher,
        )
    
    def test_search_documents_basic(self, search_service, mock_search_engine):
        """Test basic document search."""
        query = "test legal query"
        
        results = search_service.search_documents(query, limit=5)
        
        # Verify search engine was called
        mock_search_engine.search.assert_called_once()
        call_args = mock_search_engine.search.call_args
        assert call_args[1]["query"] == query
        assert call_args[1]["limit"] == 10  # 5 * 2 for ranking candidates
        assert call_args[1]["include_passages"] is True
        
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
    
    def test_search_documents_with_filters(self, search_service, mock_search_engine):
        """Test search with filters."""
        results = search_service.search_documents(
            query="test query",
            document_type="ustawa",
            legal_domain="prawo cywilne",
            similarity_threshold=0.7,
        )
        
        # Verify filters were passed
        call_args = mock_search_engine.search.call_args
        assert call_args[1]["document_type"] == "ustawa"
        assert call_args[1]["legal_domain"] == "prawo cywilne"
        assert call_args[1]["similarity_threshold"] == 0.7
        
        assert len(results) == 1
    
    def test_find_similar_documents(self, search_service, mock_search_engine):
        """Test finding similar documents."""
        document_id = uuid4()
        
        results = search_service.find_similar_documents(
            document_id=document_id,
            limit=5,
        )
        
        # Verify search engine was called
        mock_search_engine.find_similar_to_document.assert_called_once()
        call_args = mock_search_engine.find_similar_to_document.call_args
        assert call_args[1]["document_id"] == document_id
        assert call_args[1]["limit"] == 10  # 5 * 2 for ranking candidates
        assert call_args[1]["exclude_self"] is True
        
        assert len(results) == 1
    
    def test_index_documents(self, search_service, mock_indexer):
        """Test document indexing."""
        documents = [
            LegalDocument(id=uuid4(), title="Doc 1", content="Content 1"),
            LegalDocument(id=uuid4(), title="Doc 2", content="Content 2"),
        ]
        
        results = search_service.index_documents(
            documents=documents,
            batch_size=32,
            overwrite_existing=True,
        )
        
        # Verify indexer was called
        mock_indexer.batch_index_documents.assert_called_once()
        call_args = mock_indexer.batch_index_documents.call_args
        assert call_args[1]["documents"] == documents
        assert call_args[1]["batch_size"] == 32
        assert call_args[1]["overwrite_existing"] is True
        
        assert len(results) == 1
        assert isinstance(results[0], IndexingResult)
    
    def test_get_search_statistics(self, search_service, mock_indexer, mock_ranker, mock_cross_register_matcher):
        """Test getting search statistics."""
        stats = search_service.get_search_statistics()
        
        # Verify indexer was called
        mock_indexer.get_indexing_statistics.assert_called_once()
        
        # Check structure
        assert "indexing" in stats
        assert "search_engine" in stats
        assert "ranking" in stats
        assert "cross_register" in stats
        
        # Check search engine info
        assert stats["search_engine"]["available"] is True
        assert stats["search_engine"]["embedding_model"] == "allegro/herbert-klej-cased-v1"
        
        # Check ranking info
        assert stats["ranking"]["strategy"] == RankingStrategy.COMPOSITE.value
        
        # Check cross-register info
        assert stats["cross_register"]["available"] is True
        assert stats["cross_register"]["mapping_count"] == 1


def test_process_search_query():
    """Test process_search_query function."""
    with patch('sejm_whiz.semantic_search.core.get_semantic_search_service') as mock_get_service:
        mock_service = Mock()
        mock_result = Mock()
        mock_result.to_dict.return_value = {"test": "result"}
        mock_service.search_documents.return_value = [mock_result]
        mock_get_service.return_value = mock_service
        
        results = process_search_query(
            query="test query",
            limit=5,
            document_type="ustawa",
        )
        
        # Verify service was called correctly
        mock_service.search_documents.assert_called_once()
        call_args = mock_service.search_documents.call_args
        assert call_args[1]["query"] == "test query"
        assert call_args[1]["limit"] == 5
        assert call_args[1]["document_type"] == "ustawa"
        
        # Check results
        assert len(results) == 1
        assert results[0] == {"test": "result"}


def test_search_similar_documents():
    """Test search_similar_documents function."""
    document_id = uuid4()
    
    with patch('sejm_whiz.semantic_search.core.get_semantic_search_service') as mock_get_service:
        mock_service = Mock()
        mock_result = Mock()
        mock_result.to_dict.return_value = {"test": "result"}
        mock_service.find_similar_documents.return_value = [mock_result]
        mock_get_service.return_value = mock_service
        
        # Test with UUID
        results = search_similar_documents(document_id, limit=3)
        
        mock_service.find_similar_documents.assert_called_once()
        call_args = mock_service.find_similar_documents.call_args
        assert call_args[1]["document_id"] == document_id
        assert call_args[1]["limit"] == 3
        
        assert len(results) == 1
        assert results[0] == {"test": "result"}


def test_get_semantic_search_service_singleton():
    """Test singleton pattern for semantic search service."""
    service1 = get_semantic_search_service()
    service2 = get_semantic_search_service()
    
    assert service1 is service2
