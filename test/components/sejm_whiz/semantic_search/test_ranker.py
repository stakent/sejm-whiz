"""Tests for result ranker."""

import pytest
from unittest.mock import Mock
from uuid import uuid4
from datetime import datetime, timedelta

from sejm_whiz.semantic_search.ranker import (
    ResultRanker,
    RankingConfig,
    RankingStrategy,
    get_result_ranker,
)
from sejm_whiz.semantic_search.search_engine import SearchResult
from sejm_whiz.database.models import LegalDocument


class TestResultRanker:
    """Test result ranker functionality."""
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample legal documents with different characteristics."""
        now = datetime.utcnow()
        
        documents = []
        
        # Recent important document
        doc1 = LegalDocument(
            id=uuid4(),
            title="Recent Constitution Amendment",
            content="Constitutional law content",
            document_type="ustawa",
            legal_domain="prawo konstytucyjne",
            published_at=now - timedelta(days=10),
            is_amendment=True,
            affects_multiple_acts=True,
        )
        documents.append(doc1)
        
        # Old civil law document
        doc2 = LegalDocument(
            id=uuid4(),
            title="Old Civil Code",
            content="Civil law content",
            document_type="kodeks",
            legal_domain="prawo cywilne",
            published_at=now - timedelta(days=500),
            is_amendment=False,
            affects_multiple_acts=False,
        )
        documents.append(doc2)
        
        # Medium age administrative document
        doc3 = LegalDocument(
            id=uuid4(),
            title="Administrative Regulation",
            content="Administrative law content",
            document_type="rozporządzenie",
            legal_domain="prawo administracyjne",
            published_at=now - timedelta(days=100),
            is_amendment=False,
            affects_multiple_acts=False,
        )
        documents.append(doc3)
        
        return documents
    
    @pytest.fixture
    def sample_search_results(self, sample_documents):
        """Create sample search results."""
        results = []
        
        for i, doc in enumerate(sample_documents):
            result = SearchResult(
                document=doc,
                similarity_score=0.8 - (i * 0.1),  # Decreasing similarity
                embedding_distance=0.2 + (i * 0.1),
                matched_passages=[f"Passage {i+1}"],
                search_metadata={"base_score": 0.8 - (i * 0.1)},
            )
            results.append(result)
        
        return results
    
    @pytest.fixture
    def default_config(self):
        """Create default ranking configuration."""
        return RankingConfig()
    
    @pytest.fixture
    def ranker(self, default_config):
        """Create ranker with default configuration."""
        return ResultRanker(default_config)
    
    def test_rank_by_similarity_only(self, ranker, sample_search_results):
        """Test ranking by similarity score only."""
        ranker.config.strategy = RankingStrategy.SIMILARITY_ONLY
        
        ranked_results = ranker.rank_results(sample_search_results)
        
        # Should be ordered by similarity score (descending)
        assert len(ranked_results) == 3
        assert ranked_results[0].similarity_score >= ranked_results[1].similarity_score
        assert ranked_results[1].similarity_score >= ranked_results[2].similarity_score
    
    def test_rank_with_temporal_boost(self, ranker, sample_search_results):
        """Test ranking with temporal boosting."""
        ranker.config.strategy = RankingStrategy.TEMPORAL_BOOST
        
        ranked_results = ranker.rank_results(sample_search_results)
        
        # Recent document should be boosted
        assert len(ranked_results) == 3
        
        # Check that temporal scores were calculated
        for result in ranked_results:
            assert "temporal_score" in result.search_metadata
            assert "boosted_score" in result.search_metadata
        
        # Most recent document should have highest temporal score
        recent_result = next(r for r in ranked_results if "Recent" in r.document.title)
        old_result = next(r for r in ranked_results if "Old" in r.document.title)
        
        assert recent_result.search_metadata["temporal_score"] > old_result.search_metadata["temporal_score"]
    
    def test_rank_with_document_type_boost(self, ranker, sample_search_results):
        """Test ranking with document type boosting."""
        ranker.config.strategy = RankingStrategy.DOCUMENT_TYPE_BOOST
        
        ranked_results = ranker.rank_results(sample_search_results)
        
        # Check that document type weights were applied
        for result in ranked_results:
            assert "document_type_weight" in result.search_metadata
            assert "boosted_score" in result.search_metadata
        
        # Constitution (ustawa) should have weight 1.2
        constitution_result = next(r for r in ranked_results if r.document.document_type == "ustawa")
        assert constitution_result.search_metadata["document_type_weight"] == 1.2
        
        # Code (kodeks) should have weight 1.3
        code_result = next(r for r in ranked_results if r.document.document_type == "kodeks")
        assert code_result.search_metadata["document_type_weight"] == 1.3
    
    def test_rank_with_legal_domain_boost(self, ranker, sample_search_results):
        """Test ranking with legal domain boosting."""
        ranker.config.strategy = RankingStrategy.LEGAL_DOMAIN_BOOST
        
        ranked_results = ranker.rank_results(sample_search_results)
        
        # Check that legal domain weights were applied
        for result in ranked_results:
            assert "legal_domain_weight" in result.search_metadata
            assert "boosted_score" in result.search_metadata
        
        # Constitutional law should have weight 1.3
        const_result = next(r for r in ranked_results if r.document.legal_domain == "prawo konstytucyjne")
        assert const_result.search_metadata["legal_domain_weight"] == 1.3
    
    def test_rank_composite(self, ranker, sample_search_results):
        """Test composite ranking strategy."""
        ranker.config.strategy = RankingStrategy.COMPOSITE
        
        ranked_results = ranker.rank_results(sample_search_results, query="test query")
        
        # Should have all ranking factors
        for result in ranked_results:
            metadata = result.search_metadata
            assert "base_similarity_score" in metadata
            assert "temporal_score" in metadata
            assert "document_type_weight" in metadata
            assert "legal_domain_weight" in metadata
            assert "amendment_boost" in metadata
            assert "passage_boost" in metadata
            assert "composite_score" in metadata
        
        # Amendment should have boost
        amendment_result = next(r for r in ranked_results if r.document.is_amendment)
        assert amendment_result.search_metadata["amendment_boost"] > 1.0
        
        # Multi-act amendment should have higher boost
        multi_act_result = next(r for r in ranked_results if r.document.affects_multiple_acts)
        assert multi_act_result.search_metadata["amendment_boost"] >= ranker.config.multi_act_amendment_boost
    
    def test_calculate_temporal_score(self, ranker):
        """Test temporal score calculation."""
        current_time = datetime.utcnow()
        
        # Recent document (within recency boost period)
        recent_date = current_time - timedelta(days=10)
        recent_score = ranker._calculate_temporal_score(recent_date, current_time)
        assert recent_score > 0.8  # Should have high score (10 days = 1.0)
        
        # Medium age document
        medium_date = current_time - timedelta(days=100)
        medium_score = ranker._calculate_temporal_score(medium_date, current_time)
        assert 0.1 < medium_score < 0.8
        
        # Very old document
        old_date = current_time - timedelta(days=500)
        old_score = ranker._calculate_temporal_score(old_date, current_time)
        assert old_score == 0.1  # Minimum score
        
        # No publication date
        none_score = ranker._calculate_temporal_score(None, current_time)
        assert none_score == 0.5  # Neutral score
    
    def test_calculate_passage_relevance(self, ranker):
        """Test passage relevance calculation."""
        query = "legal article paragraph"
        passages = [
            "This passage contains legal and article terms.",
            "Another passage with paragraph and legal content.",
            "Irrelevant passage without query terms.",
        ]
        
        relevance = ranker._calculate_passage_relevance(passages, query)
        
        assert 0.0 <= relevance <= 1.0
        assert relevance > 0  # Should have some relevance due to matching terms
        
        # Test with no passages
        empty_relevance = ranker._calculate_passage_relevance([], query)
        assert empty_relevance == 0.0
        
        # Test with empty query
        no_query_relevance = ranker._calculate_passage_relevance(passages, "")
        assert no_query_relevance == 0.0
    
    def test_custom_weights(self, ranker, sample_search_results):
        """Test ranking with custom weights."""
        custom_weights = {
            "document_type": 2.0,
            "legal_domain": 0.5,
            "temporal": 1.5,
        }
        
        ranked_results = ranker.rank_results(
            sample_search_results,
            query="test query",
            custom_weights=custom_weights
        )
        
        # Custom weights should be applied
        for result in ranked_results:
            # Document type weights should be multiplied by custom weight
            base_doc_weight = ranker.config.document_type_weights.get(result.document.document_type, 1.0)
            expected_doc_weight = base_doc_weight * custom_weights["document_type"]
            assert abs(result.search_metadata["document_type_weight"] - expected_doc_weight) < 0.001
    
    def test_get_ranking_explanation(self, ranker, sample_search_results):
        """Test ranking explanation generation."""
        # Rank results first
        ranked_results = ranker.rank_results(sample_search_results)
        
        # Get explanation for first result
        explanation = ranker.get_ranking_explanation(ranked_results[0])
        
        assert "final_score" in explanation
        assert "ranking_strategy" in explanation
        assert "factors" in explanation
        assert "document_info" in explanation
        
        # Check factors
        factors = explanation["factors"]
        assert "base_similarity" in factors
        assert "temporal_score" in factors
        assert "document_type_weight" in factors
        
        # Check document info
        doc_info = explanation["document_info"]
        assert "id" in doc_info
        assert "type" in doc_info
        assert "domain" in doc_info
    
    def test_empty_results(self, ranker):
        """Test ranking with empty results list."""
        ranked_results = ranker.rank_results([])
        assert ranked_results == []
    
    def test_ranking_config_defaults(self):
        """Test default ranking configuration values."""
        config = RankingConfig()
        
        assert config.strategy == RankingStrategy.COMPOSITE
        assert config.temporal_decay_days == 365
        assert config.temporal_weight == 0.2
        assert config.recency_boost_days == 30
        assert config.recency_boost_factor == 1.5
        
        # Check document type weights
        assert config.document_type_weights["ustawa"] == 1.2
        assert config.document_type_weights["kodeks"] == 1.3
        
        # Check legal domain weights
        assert config.legal_domain_weights["prawo konstytucyjne"] == 1.3
        assert config.legal_domain_weights["prawo cywilne"] == 1.2
    
    def test_score_clamping(self, ranker, sample_search_results):
        """Test that scores are clamped to configured range."""
        # Set tight score range
        ranker.config.min_score = 0.3
        ranker.config.max_score = 0.9
        
        ranked_results = ranker.rank_results(sample_search_results)
        
        # All scores should be within range
        for result in ranked_results:
            assert ranker.config.min_score <= result.similarity_score <= ranker.config.max_score
    
    def test_get_result_ranker_singleton(self):
        """Test singleton pattern for result ranker."""
        ranker1 = get_result_ranker()
        ranker2 = get_result_ranker()
        
        assert ranker1 is ranker2
    
    def test_get_result_ranker_with_config(self):
        """Test creating ranker with custom configuration."""
        custom_config = RankingConfig(strategy=RankingStrategy.TEMPORAL_BOOST)
        ranker = get_result_ranker(custom_config)
        
        assert ranker.config.strategy == RankingStrategy.TEMPORAL_BOOST
    
    def test_ranking_failure_recovery(self, ranker, sample_search_results):
        """Test that ranking failures return original results."""
        # Mock a failure in ranking
        original_rank_composite = ranker._rank_composite
        
        def failing_rank_composite(*args, **kwargs):
            raise Exception("Ranking failed")
        
        ranker._rank_composite = failing_rank_composite
        
        # Should return original results on failure
        ranked_results = ranker.rank_results(sample_search_results)
        assert ranked_results == sample_search_results
        
        # Restore original method
        ranker._rank_composite = original_rank_composite