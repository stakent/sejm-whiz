"""Tests for semantic search engine."""

import pytest
from unittest.mock import Mock, patch
from uuid import uuid4
from datetime import datetime

import numpy as np

from sejm_whiz.semantic_search.search_engine import (
    SemanticSearchEngine,
    SearchResult,
    get_search_engine,
)
from sejm_whiz.vector_db import DistanceMetric
from sejm_whiz.database.models import LegalDocument


class TestSemanticSearchEngine:
    """Test semantic search engine functionality."""

    @pytest.fixture
    def mock_embedder(self):
        """Mock HerBERT embedder."""
        embedder = Mock()
        embedding_result = Mock()
        embedding_result.success = True
        embedding_result.embedding = np.random.random(768)
        embedding_result.error = None
        embedder.generate_embedding.return_value = embedding_result
        return embedder

    @pytest.fixture
    def mock_similarity_search(self):
        """Mock vector similarity search."""
        similarity_search = Mock()

        # Mock document
        doc = LegalDocument(
            id=uuid4(),
            title="Test Document",
            content="Test legal document content with articles and paragraphs.",
            document_type="ustawa",
            legal_domain="prawo cywilne",
            published_at=datetime.utcnow(),
            is_amendment=False,
            affects_multiple_acts=False,
        )

        similarity_search.find_similar_documents.return_value = [(doc, 0.2)]
        similarity_search.find_similar_to_document.return_value = [(doc, 0.15)]
        return similarity_search

    @pytest.fixture
    def search_engine(self, mock_embedder, mock_similarity_search):
        """Create search engine with mocked dependencies."""
        return SemanticSearchEngine(
            embedder=mock_embedder,
            similarity_search=mock_similarity_search,
        )

    def test_search_basic(self, search_engine, mock_embedder, mock_similarity_search):
        """Test basic search functionality."""
        query = "Test query about legal matters"

        with patch(
            "sejm_whiz.semantic_search.search_engine.process_legal_document"
        ) as mock_process:
            mock_processed = {"clean_text": "processed query text"}
            mock_process.return_value = mock_processed

            results = search_engine.search(query, limit=5)

        # Verify embedder was called
        mock_embedder.generate_embedding.assert_called_once_with("processed query text")

        # Verify similarity search was called
        mock_similarity_search.find_similar_documents.assert_called_once()

        # Check results
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].document.title == "Test Document"
        assert results[0].similarity_score > 0

    def test_search_with_filters(self, search_engine):
        """Test search with document type and legal domain filters."""
        query = "Test query"

        with patch(
            "sejm_whiz.semantic_search.search_engine.process_legal_document"
        ) as mock_process:
            mock_processed = {"clean_text": "processed query text"}
            mock_process.return_value = mock_processed

            results = search_engine.search(
                query,
                document_type="ustawa",
                legal_domain="prawo cywilne",
                similarity_threshold=0.5,
            )

        # Verify filters were passed to similarity search
        call_args = search_engine.similarity_search.find_similar_documents.call_args
        assert call_args[1]["document_type"] == "ustawa"
        assert call_args[1]["legal_domain"] == "prawo cywilne"

        assert len(results) == 1

    def test_search_distance_metrics(self, search_engine):
        """Test search with different distance metrics."""
        query = "Test query"

        with patch(
            "sejm_whiz.semantic_search.search_engine.process_legal_document"
        ) as mock_process:
            mock_processed = {"clean_text": "processed query text"}
            mock_process.return_value = mock_processed

            # Test cosine distance
            results_cosine = search_engine.search(
                query, distance_metric=DistanceMetric.COSINE
            )
            assert len(results_cosine) == 1
            assert results_cosine[0].similarity_score == 0.8  # 1.0 - 0.2

            # Test L2 distance
            results_l2 = search_engine.search(query, distance_metric=DistanceMetric.L2)
            assert len(results_l2) == 1
            # L2 similarity = 1.0 / (1.0 + distance)
            expected_l2_score = 1.0 / (1.0 + 0.2)
            assert abs(results_l2[0].similarity_score - expected_l2_score) < 0.001

    def test_search_with_passages(self, search_engine):
        """Test search with passage extraction."""
        query = "legal matters"

        with patch(
            "sejm_whiz.semantic_search.search_engine.process_legal_document"
        ) as mock_process:
            mock_processed = {"clean_text": "processed query text"}
            mock_process.return_value = mock_processed

            results = search_engine.search(query, include_passages=True)

        assert len(results) == 1
        # Should have extracted passages containing query terms
        assert len(results[0].matched_passages) > 0

    def test_batch_search(self, search_engine):
        """Test batch search functionality."""
        queries = ["First query", "Second query", "Third query"]

        with patch(
            "sejm_whiz.semantic_search.search_engine.process_legal_document"
        ) as mock_process:
            mock_processed = {"clean_text": "processed query text"}
            mock_process.return_value = mock_processed

            results = search_engine.batch_search(queries, limit=3)

        assert len(results) == 3  # One result list per query
        for query_results in results:
            assert len(query_results) == 1
            assert isinstance(query_results[0], SearchResult)

    def test_find_similar_to_document(self, search_engine):
        """Test finding documents similar to a given document."""
        document_id = uuid4()

        results = search_engine.find_similar_to_document(
            document_id=document_id,
            limit=5,
            exclude_self=True,
        )

        # Verify similarity search was called
        search_engine.similarity_search.find_similar_to_document.assert_called_once_with(
            document_id=document_id,
            limit=6,  # limit + 1 for exclude_self
            distance_metric=DistanceMetric.COSINE,
        )

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)

    def test_search_embedding_failure(self, search_engine, mock_embedder):
        """Test search behavior when embedding generation fails."""
        # Mock embedding failure
        embedding_result = Mock()
        embedding_result.success = False
        embedding_result.error = "Embedding generation failed"
        mock_embedder.generate_embedding.return_value = embedding_result

        with patch(
            "sejm_whiz.semantic_search.search_engine.process_legal_document"
        ) as mock_process:
            mock_processed = {"clean_text": "processed query text"}
            mock_process.return_value = mock_processed

            with pytest.raises(ValueError, match="Failed to generate query embedding"):
                search_engine.search("test query")

    def test_extract_relevant_passages(self, search_engine):
        """Test passage extraction functionality."""
        query = "legal article paragraph"
        document_text = (
            "This is the first sentence about legal matters. "
            "The article discusses various legal paragraphs. "
            "This sentence is not relevant. "
            "Another article about legal issues and paragraphs."
        )

        passages = search_engine._extract_relevant_passages(
            query, document_text, max_passages=2
        )

        assert len(passages) <= 2
        # Should contain passages with query terms
        for passage in passages:
            assert any(
                term in passage.lower() for term in ["legal", "article", "paragraph"]
            )

    def test_search_result_to_dict(self):
        """Test SearchResult to_dict conversion."""
        doc = LegalDocument(
            id=uuid4(),
            title="Test Document",
            content="Test content",
            document_type="ustawa",
            legal_domain="prawo cywilne",
            published_at=datetime.utcnow(),
        )

        result = SearchResult(
            document=doc,
            similarity_score=0.85,
            embedding_distance=0.15,
            matched_passages=["Test passage"],
            search_metadata={"test": "metadata"},
        )

        result_dict = result.to_dict()

        assert result_dict["document_id"] == str(doc.id)
        assert result_dict["title"] == "Test Document"
        assert result_dict["similarity_score"] == 0.85
        assert result_dict["matched_passages"] == ["Test passage"]
        assert result_dict["search_metadata"] == {"test": "metadata"}

    def test_get_search_engine_singleton(self):
        """Test singleton pattern for search engine."""
        engine1 = get_search_engine()
        engine2 = get_search_engine()

        assert engine1 is engine2

    def test_similarity_threshold_filtering(self, search_engine):
        """Test filtering results by similarity threshold."""
        # Mock multiple documents with different distances
        high_sim_doc = Mock()
        high_sim_doc.id = uuid4()
        high_sim_doc.title = "High similarity"
        high_sim_doc.content = "High similarity document content"

        med_sim_doc = Mock()
        med_sim_doc.id = uuid4()
        med_sim_doc.title = "Medium similarity"
        med_sim_doc.content = "Medium similarity document content"

        low_sim_doc = Mock()
        low_sim_doc.id = uuid4()
        low_sim_doc.title = "Low similarity"
        low_sim_doc.content = "Low similarity document content"

        docs_with_distances = [
            (high_sim_doc, 0.1),  # similarity = 0.9
            (med_sim_doc, 0.4),  # similarity = 0.6
            (low_sim_doc, 0.8),  # similarity = 0.2
        ]

        search_engine.similarity_search.find_similar_documents.return_value = (
            docs_with_distances
        )

        with patch(
            "sejm_whiz.semantic_search.search_engine.process_legal_document"
        ) as mock_process:
            mock_processed = {"clean_text": "processed query text"}
            mock_process.return_value = mock_processed

            results = search_engine.search("test query", similarity_threshold=0.5)

        # Should only return documents with similarity >= 0.5
        assert len(results) == 2  # Only high and medium similarity docs
        assert all(r.similarity_score >= 0.5 for r in results)
