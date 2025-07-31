"""Tests for vector_db embeddings module."""

import pytest
from unittest.mock import Mock, patch
from uuid import uuid4

from sejm_whiz.vector_db.embeddings import (
    VectorSimilaritySearch,
    DistanceMetric,
    get_similarity_search,
)
from sejm_whiz.database.models import LegalDocument


class TestDistanceMetric:
    """Test DistanceMetric enum."""

    def test_distance_metric_values(self):
        """Test distance metric enum values."""
        assert DistanceMetric.COSINE.value == "cosine"
        assert DistanceMetric.L2.value == "l2"
        assert DistanceMetric.INNER_PRODUCT.value == "inner_product"


class TestVectorSimilaritySearch:
    """Test VectorSimilaritySearch class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.search = VectorSimilaritySearch()
        self.sample_embedding = [0.1] * 768
        self.document_id = str(uuid4())

    def test_init(self):
        """Test VectorSimilaritySearch initialization."""
        assert self.search.distance_operators[DistanceMetric.COSINE] == "<=>"
        assert self.search.distance_operators[DistanceMetric.L2] == "<->"
        assert self.search.distance_operators[DistanceMetric.INNER_PRODUCT] == "<#>"

    @patch("sejm_whiz.vector_db.embeddings.get_vector_session")
    @patch("sejm_whiz.vector_db.embeddings.select")
    @patch("sejm_whiz.vector_db.embeddings.text")
    def test_find_similar_documents(self, mock_text, mock_select, mock_get_session):
        """Test finding similar documents."""
        mock_session = Mock()
        mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = Mock(return_value=None)

        # Mock query results - create mock rows with proper attributes
        mock_row1 = Mock()
        mock_row1.id = str(uuid4())
        mock_row1.title = "Test Doc 1"
        mock_row1.content = "Test content 1"
        mock_row1.document_type = "law"
        mock_row1.legal_domain = "civil"
        mock_row1.created_at = None
        mock_row1.updated_at = None
        mock_row1.embedding = self.sample_embedding
        mock_row1.distance = 0.1

        mock_row2 = Mock()
        mock_row2.id = str(uuid4())
        mock_row2.title = "Test Doc 2"
        mock_row2.content = "Test content 2"
        mock_row2.document_type = "law"
        mock_row2.legal_domain = "civil"
        mock_row2.created_at = None
        mock_row2.updated_at = None
        mock_row2.embedding = self.sample_embedding
        mock_row2.distance = 0.2

        mock_result = Mock()
        mock_result.fetchall.return_value = [mock_row1, mock_row2]
        mock_session.execute.return_value = mock_result

        result = self.search.find_similar_documents(
            query_embedding=self.sample_embedding,
            limit=2,
            distance_metric=DistanceMetric.COSINE,
        )

        assert len(result) == 2
        assert result[0][0].title == "Test Doc 1"
        assert result[0][1] == 0.1
        assert result[1][0].title == "Test Doc 2"
        assert result[1][1] == 0.2
        mock_session.execute.assert_called_once()

    @patch("sejm_whiz.vector_db.embeddings.get_vector_session")
    def test_find_similar_documents_with_filters(self, mock_get_session):
        """Test finding similar documents with filters."""
        mock_session = Mock()
        mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = Mock(return_value=None)

        mock_result = Mock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        # Patch the actual SQL construction to avoid mock comparison issues
        with (
            patch("sejm_whiz.vector_db.embeddings.text") as mock_text,
            patch("sejm_whiz.vector_db.embeddings.select") as mock_select,
        ):
            # Mock text to return something that can be compared
            mock_distance_expr = Mock()
            mock_distance_expr.__le__ = Mock(return_value=Mock())
            mock_text.return_value = mock_distance_expr

            result = self.search.find_similar_documents(
                query_embedding=self.sample_embedding,
                limit=10,
                distance_metric=DistanceMetric.L2,
                document_type="law",
                legal_domain="civil",
                threshold=0.5,
            )

            assert result == []
            mock_session.execute.assert_called_once()

    @patch("sejm_whiz.vector_db.embeddings.get_vector_session")
    @patch("sejm_whiz.vector_db.embeddings.select")
    @patch("sejm_whiz.vector_db.embeddings.text")
    def test_find_similar_by_document_id(
        self, mock_text, mock_select, mock_get_session
    ):
        """Test finding similar documents by document ID."""
        mock_session = Mock()
        mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = Mock(return_value=None)

        # Mock source embedding query result
        source_embedding = [0.2] * 768
        mock_embedding_result = Mock()
        mock_embedding_result.scalar_one_or_none.return_value = source_embedding

        # Mock similar documents query result for raw SQL
        mock_similar_row = Mock()
        mock_similar_row.id = str(uuid4())
        mock_similar_row.title = "Similar Doc"
        mock_similar_row.content = "Similar content"
        mock_similar_row.document_type = "law"
        mock_similar_row.source_url = None
        mock_similar_row.eli_identifier = None
        mock_similar_row.legal_domain = "civil"
        mock_similar_row.legal_act_type = None
        mock_similar_row.is_amendment = False
        mock_similar_row.affects_multiple_acts = False
        mock_similar_row.created_at = None
        mock_similar_row.updated_at = None
        mock_similar_row.published_at = None
        mock_similar_row.embedding = source_embedding
        mock_similar_row.distance = 0.1

        # Mock the result object that has fetchall() method
        mock_similar_result = Mock()
        mock_similar_result.fetchall.return_value = [mock_similar_row]

        mock_session.execute.side_effect = [
            mock_embedding_result,  # Source embedding
            mock_similar_result,  # Raw SQL result with fetchall
        ]

        result = self.search.find_similar_by_document_id(
            document_id=self.document_id, limit=5, distance_metric=DistanceMetric.COSINE
        )

        assert len(result) == 1
        assert result[0][0].title == "Similar Doc"
        assert result[0][1] == 0.1  # distance
        assert mock_session.execute.call_count == 2

    @patch("sejm_whiz.vector_db.embeddings.get_vector_session")
    @patch("sejm_whiz.vector_db.embeddings.select")
    def test_find_similar_by_document_id_no_embedding(
        self, mock_select, mock_get_session
    ):
        """Test finding similar documents when source has no embedding."""
        mock_session = Mock()
        mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = Mock(return_value=None)

        # Mock no embedding found
        mock_session.execute.return_value.scalar_one_or_none.return_value = None

        result = self.search.find_similar_by_document_id(
            document_id=self.document_id, limit=5
        )

        assert result == []
        mock_session.execute.assert_called_once()

    def test_batch_similarity_search(self):
        """Test batch similarity search."""
        embeddings = [[0.1] * 768, [0.2] * 768, [0.3] * 768]

        with patch.object(self.search, "find_similar_documents") as mock_find:
            mock_find.side_effect = [[(Mock(), 0.1)], [(Mock(), 0.2)], [(Mock(), 0.3)]]

            results = self.search.batch_similarity_search(
                query_embeddings=embeddings,
                limit=1,
                distance_metric=DistanceMetric.COSINE,
            )

        assert len(results) == 3
        assert mock_find.call_count == 3

    @patch("sejm_whiz.vector_db.embeddings.get_vector_session")
    def test_find_documents_in_range(self, mock_get_session):
        """Test finding documents within distance range."""
        mock_session = Mock()
        mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = Mock(return_value=None)

        mock_doc_row = Mock()
        mock_doc_row.id = str(uuid4())
        mock_doc_row.title = "Range Doc"
        mock_doc_row.content = "Range content"
        mock_doc_row.document_type = "law"
        mock_doc_row.legal_domain = "civil"
        mock_doc_row.created_at = None
        mock_doc_row.updated_at = None
        mock_doc_row.embedding = self.sample_embedding
        mock_doc_row.distance = 0.3

        # Create a mock row that looks like SQLAlchemy result row for ORM query
        mock_result_row = Mock()
        mock_result_row.LegalDocument = Mock(spec=LegalDocument)
        mock_result_row.LegalDocument.id = mock_doc_row.id
        mock_result_row.LegalDocument.title = mock_doc_row.title
        mock_result_row.LegalDocument.content = mock_doc_row.content
        mock_result_row.LegalDocument.document_type = mock_doc_row.document_type
        mock_result_row.LegalDocument.source_url = None
        mock_result_row.LegalDocument.eli_identifier = None
        mock_result_row.LegalDocument.legal_domain = mock_doc_row.legal_domain
        mock_result_row.LegalDocument.created_at = mock_doc_row.created_at
        mock_result_row.LegalDocument.updated_at = mock_doc_row.updated_at
        mock_result_row.LegalDocument.embedding = mock_doc_row.embedding
        mock_result_row.distance = mock_doc_row.distance

        mock_result = [mock_result_row]
        mock_session.execute.return_value = mock_result

        # Patch the SQL construction to avoid comparison issues
        with (
            patch("sejm_whiz.vector_db.embeddings.text") as mock_text,
            patch("sejm_whiz.vector_db.embeddings.select") as mock_select,
        ):
            # Mock text to return something that supports comparisons
            mock_distance_expr = Mock()
            mock_distance_expr.__ge__ = Mock(return_value=Mock())
            mock_distance_expr.__le__ = Mock(return_value=Mock())
            mock_text.return_value = mock_distance_expr

            result = self.search.find_documents_in_range(
                query_embedding=self.sample_embedding,
                min_distance=0.2,
                max_distance=0.4,
                distance_metric=DistanceMetric.COSINE,
                limit=10,
            )

            assert len(result) == 1
            assert result[0][0].title == "Range Doc"
            assert result[0][1] == 0.3
            mock_session.execute.assert_called_once()

    @patch("sejm_whiz.vector_db.embeddings.get_vector_session")
    @patch("sejm_whiz.vector_db.embeddings.select")
    @patch("sejm_whiz.vector_db.embeddings.func")
    def test_get_embedding_statistics(self, mock_func, mock_select, mock_get_session):
        """Test getting embedding statistics."""
        mock_session = Mock()
        mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = Mock(return_value=None)

        # Mock the different execute calls - fix the third one to return a mock with .all()
        type_counts_result = Mock()
        type_counts_result.all.return_value = [("law", 50), ("regulation", 25)]

        mock_session.execute.side_effect = [
            Mock(scalar=Mock(return_value=75)),  # docs_with_embeddings
            Mock(scalar=Mock(return_value=100)),  # total_docs
            type_counts_result,  # type_counts (needs .all() method)
            Mock(scalar=Mock(return_value=80)),  # embedding_records
        ]

        result = self.search.get_embedding_statistics()

        expected = {
            "documents_with_embeddings": 75,
            "total_documents": 100,
            "embedding_coverage": 75.0,
            "type_distribution": {"law": 50, "regulation": 25},
            "embedding_records": 80,
        }

        assert result == expected
        assert mock_session.execute.call_count == 4

    @patch("sejm_whiz.vector_db.embeddings.get_vector_session")
    @patch("sejm_whiz.vector_db.embeddings.text")
    def test_create_vector_index_ivfflat(self, mock_text, mock_get_session):
        """Test creating IVFFlat vector index."""
        mock_session = Mock()
        mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = Mock(return_value=None)

        result = self.search.create_vector_index(
            table_name="legal_documents",
            column_name="embedding",
            index_type="ivfflat",
            lists=100,
        )

        assert result is True
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()

    @patch("sejm_whiz.vector_db.embeddings.get_vector_session")
    @patch("sejm_whiz.vector_db.embeddings.text")
    def test_create_vector_index_hnsw(self, mock_text, mock_get_session):
        """Test creating HNSW vector index."""
        mock_session = Mock()
        mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = Mock(return_value=None)

        result = self.search.create_vector_index(
            table_name="legal_documents", column_name="embedding", index_type="hnsw"
        )

        assert result is True
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()

    @patch("sejm_whiz.vector_db.embeddings.get_vector_session")
    def test_create_vector_index_unsupported_type(self, mock_get_session):
        """Test creating vector index with unsupported type."""
        mock_session = Mock()
        mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = Mock(return_value=None)

        with pytest.raises(ValueError, match="Unsupported index type"):
            self.search.create_vector_index(index_type="unsupported")


class TestGlobalFunctions:
    """Test global helper functions."""

    @patch("sejm_whiz.vector_db.embeddings.VectorSimilaritySearch")
    def test_get_similarity_search_singleton(self, mock_search_class):
        """Test get_similarity_search returns singleton."""
        mock_instance = Mock()
        mock_search_class.return_value = mock_instance

        # Clear the global instance
        import sejm_whiz.vector_db.embeddings

        sejm_whiz.vector_db.embeddings._similarity_search = None

        # First call should create instance
        result1 = get_similarity_search()
        assert result1 == mock_instance
        mock_search_class.assert_called_once()

        # Second call should return same instance
        result2 = get_similarity_search()
        assert result2 == mock_instance
        assert result1 is result2
        # Should not create another instance
        mock_search_class.assert_called_once()
