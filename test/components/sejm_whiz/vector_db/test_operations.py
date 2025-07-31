"""Tests for vector_db operations module."""

from unittest.mock import Mock, patch
from uuid import uuid4, UUID

from sejm_whiz.vector_db.operations import VectorDBOperations, get_vector_operations
from sejm_whiz.database.models import LegalDocument


class TestVectorDBOperations:
    """Test VectorDBOperations class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ops = VectorDBOperations()
        self.sample_embedding = [0.1] * 768
        self.document_id = uuid4()

    @patch("sejm_whiz.vector_db.operations.get_vector_session")
    def test_create_document_with_embedding(self, mock_get_session):
        """Test creating document with embedding."""
        mock_session = Mock()
        mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = Mock(return_value=None)

        # Mock the document object with an ID
        mock_document = Mock()
        mock_document.id = self.document_id
        mock_session.add.return_value = None
        mock_session.flush.return_value = None
        mock_session.commit.return_value = None

        # Patch LegalDocument constructor
        with patch(
            "sejm_whiz.vector_db.operations.LegalDocument", return_value=mock_document
        ):
            result = self.ops.create_document_with_embedding(
                title="Test Document",
                content="Test content",
                document_type="law",
                embedding=self.sample_embedding,
            )

        assert result == self.document_id
        mock_session.add.assert_called_once_with(mock_document)
        mock_session.flush.assert_called_once()
        mock_session.commit.assert_called_once()

    @patch("sejm_whiz.vector_db.operations.get_vector_session")
    @patch("sejm_whiz.vector_db.operations.update")
    def test_update_document_embedding(self, mock_update, mock_get_session):
        """Test updating document embedding."""
        mock_session = Mock()
        mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = Mock(return_value=None)

        mock_result = Mock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        result = self.ops.update_document_embedding(
            document_id=self.document_id, embedding=self.sample_embedding
        )

        assert result is True
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()

    @patch("sejm_whiz.vector_db.operations.get_vector_session")
    @patch("sejm_whiz.vector_db.operations.select")
    def test_get_document_by_id(self, mock_select, mock_get_session):
        """Test getting document by ID."""
        mock_session = Mock()
        mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = Mock(return_value=None)

        # Create a mock document with proper attributes
        mock_document = Mock(spec=LegalDocument)
        mock_document.id = self.document_id
        mock_document.title = "Test Document"
        mock_document.content = "Test content"
        mock_document.document_type = "law"
        mock_document.source_url = "http://example.com"
        mock_document.eli_identifier = "eli:pl:test"
        mock_document.embedding = [0.1] * 768
        mock_document.created_at = None
        mock_document.updated_at = None
        mock_document.published_at = None
        mock_document.legal_act_type = None
        mock_document.legal_domain = "civil"
        mock_document.is_amendment = False
        mock_document.affects_multiple_acts = False

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_document
        mock_session.execute.return_value = mock_result

        result = self.ops.get_document_by_id(self.document_id)

        # Check that we get a proper LegalDocument with expected attributes
        assert result is not None
        assert result.id == self.document_id
        assert result.title == "Test Document"
        assert result.content == "Test content"
        assert result.document_type == "law"
        mock_session.execute.assert_called_once()

    @patch("sejm_whiz.vector_db.operations.get_vector_session")
    @patch("sejm_whiz.vector_db.operations.select")
    def test_get_documents_by_type(self, mock_select, mock_get_session):
        """Test getting documents by type."""
        mock_session = Mock()
        mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = Mock(return_value=None)

        # Create mock documents with proper attributes
        mock_documents = []
        for i in range(3):
            mock_doc = Mock(spec=LegalDocument)
            mock_doc.id = UUID(f"00000000-0000-0000-0000-00000000000{i}")
            mock_doc.title = f"Test Document {i}"
            mock_doc.content = f"Test content {i}"
            mock_doc.document_type = "law"
            mock_doc.source_url = f"http://example.com/{i}"
            mock_doc.eli_identifier = f"eli:pl:test:{i}"
            mock_doc.embedding = [0.1 * (i + 1)] * 768
            mock_doc.created_at = None
            mock_doc.updated_at = None
            mock_doc.published_at = None
            mock_doc.legal_act_type = None
            mock_doc.legal_domain = "civil"
            mock_doc.is_amendment = False
            mock_doc.affects_multiple_acts = False
            mock_documents.append(mock_doc)

        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = mock_documents
        mock_session.execute.return_value = mock_result

        result = self.ops.get_documents_by_type("law", limit=10)

        # Check that we get the right number of documents with expected attributes
        assert len(result) == 3
        assert all(doc.document_type == "law" for doc in result)
        assert result[0].title == "Test Document 0"
        assert result[1].title == "Test Document 1"
        assert result[2].title == "Test Document 2"
        mock_session.execute.assert_called_once()

    @patch("sejm_whiz.vector_db.operations.get_vector_session")
    @patch("sejm_whiz.vector_db.operations.delete")
    def test_delete_document(self, mock_delete, mock_get_session):
        """Test deleting document."""
        mock_session = Mock()
        mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = Mock(return_value=None)

        mock_result = Mock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        result = self.ops.delete_document(self.document_id)

        assert result is True
        # Should be called twice (embeddings + document)
        assert mock_session.execute.call_count == 2
        mock_session.commit.assert_called_once()

    @patch("sejm_whiz.vector_db.operations.get_vector_session")
    def test_create_document_embedding(self, mock_get_session):
        """Test creating document embedding record."""
        mock_session = Mock()
        mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = Mock(return_value=None)

        embedding_id = uuid4()
        mock_embedding = Mock()
        mock_embedding.id = embedding_id

        with patch(
            "sejm_whiz.vector_db.operations.DocumentEmbedding",
            return_value=mock_embedding,
        ):
            result = self.ops.create_document_embedding(
                document_id=self.document_id,
                embedding=self.sample_embedding,
                model_name="herbert-klej-cased-v1",
                model_version="1.0",
            )

        assert result == embedding_id
        mock_session.add.assert_called_once_with(mock_embedding)
        mock_session.flush.assert_called_once()
        mock_session.commit.assert_called_once()

    @patch("sejm_whiz.vector_db.operations.get_vector_session")
    def test_bulk_insert_documents(self, mock_get_session):
        """Test bulk inserting documents."""
        mock_session = Mock()
        mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = Mock(return_value=None)

        # Create mock documents with IDs
        mock_docs = []
        expected_ids = []
        for i in range(3):
            doc_id = uuid4()
            mock_doc = Mock()
            mock_doc.id = doc_id
            mock_docs.append(mock_doc)
            expected_ids.append(doc_id)

        with patch(
            "sejm_whiz.vector_db.operations.LegalDocument", side_effect=mock_docs
        ):
            documents = [
                {
                    "title": f"Doc {i}",
                    "content": f"Content {i}",
                    "document_type": "law",
                    "embedding": self.sample_embedding,
                }
                for i in range(3)
            ]

            result = self.ops.bulk_insert_documents(documents)

        assert result == expected_ids
        mock_session.add_all.assert_called_once()
        mock_session.flush.assert_called_once()
        mock_session.commit.assert_called_once()


class TestGlobalFunctions:
    """Test global helper functions."""

    @patch("sejm_whiz.vector_db.operations.VectorDBOperations")
    def test_get_vector_operations_singleton(self, mock_ops_class):
        """Test get_vector_operations returns singleton."""
        mock_instance = Mock()
        mock_ops_class.return_value = mock_instance

        # Clear the global instance
        import sejm_whiz.vector_db.operations

        sejm_whiz.vector_db.operations._vector_ops = None

        # First call should create instance
        result1 = get_vector_operations()
        assert result1 == mock_instance
        mock_ops_class.assert_called_once()

        # Second call should return same instance
        result2 = get_vector_operations()
        assert result2 == mock_instance
        assert result1 is result2
        # Should not create another instance
        mock_ops_class.assert_called_once()
