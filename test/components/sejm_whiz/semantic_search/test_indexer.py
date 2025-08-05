"""Tests for document indexer."""

import pytest
from unittest.mock import Mock, patch
from uuid import uuid4
from datetime import datetime, UTC

import numpy as np

from sejm_whiz.semantic_search.indexer import (
    DocumentIndexer,
    IndexingResult,
    get_document_indexer,
)
from sejm_whiz.database.models import LegalDocument


class TestDocumentIndexer:
    """Test document indexer functionality."""

    @pytest.fixture
    def mock_embedder(self):
        """Mock HerBERT embedder."""
        embedder = Mock()
        embedding_result = Mock()
        embedding_result.embedding = np.random.random(768)
        embedder.embed_text.return_value = embedding_result
        embedder.embed_texts.return_value = [embedding_result]
        return embedder

    @pytest.fixture
    def mock_batch_processor(self):
        """Mock batch processor."""
        processor = Mock()
        batch_result = Mock()
        batch_result.results = []
        processor.batch_generate_embeddings.return_value = batch_result
        return processor

    @pytest.fixture
    def mock_vector_operations(self):
        """Mock vector database operations."""
        operations = Mock()
        operations.update_document_embedding.return_value = True
        operations.get_documents.return_value = []
        operations.get_embedding_statistics.return_value = {
            "total_documents": 100,
            "indexed_documents": 80,
            "indexing_percentage": 80.0,
            "avg_embedding_dimensions": 768,
            "document_types": {"ustawa": 50, "rozporządzenie": 30},
            "legal_domains": {"prawo cywilne": 40, "prawo karne": 40},
        }
        return operations

    @pytest.fixture
    def indexer(self, mock_embedder, mock_batch_processor, mock_vector_operations):
        """Create indexer with mocked dependencies."""
        return DocumentIndexer(
            embedder=mock_embedder,
            batch_processor=mock_batch_processor,
            vector_operations=mock_vector_operations,
        )

    @pytest.fixture
    def sample_document(self):
        """Create sample legal document."""
        return LegalDocument(
            id=uuid4(),
            title="Test Legal Document",
            content="This is a test legal document with various articles and paragraphs.",
            document_type="ustawa",
            legal_domain="prawo cywilne",
            published_at=datetime.now(UTC),
            is_amendment=False,
            affects_multiple_acts=False,
            embedding=None,  # Not yet indexed
        )

    def test_index_document_success(
        self, indexer, sample_document, mock_embedder, mock_vector_operations
    ):
        """Test successful document indexing."""
        with patch(
            "sejm_whiz.semantic_search.indexer.process_legal_document"
        ) as mock_process:
            mock_processed = {"clean_text": "processed legal text"}
            mock_process.return_value = mock_processed

            result = indexer.index_document(sample_document)

        # Verify processing and embedding generation
        mock_process.assert_called_once_with(sample_document.content)
        mock_embedder.embed_text.assert_called_once_with("processed legal text")

        # Verify database update
        mock_vector_operations.update_document_embedding.assert_called_once()

        # Check result
        assert result.success is True
        assert result.document_id == sample_document.id
        assert result.embedding_dimensions == 768
        assert result.processing_time_ms > 0
        assert result.error is None

        # Check document was updated with embedding
        assert sample_document.embedding is not None
        assert len(sample_document.embedding) == 768

    def test_index_document_already_indexed(self, indexer, sample_document):
        """Test indexing document that already has embedding."""
        # Set existing embedding
        sample_document.embedding = [0.1] * 768

        result = indexer.index_document(sample_document, overwrite_existing=False)

        # Should skip indexing
        assert result.success is True
        assert result.processing_time_ms == 0.0
        assert result.metadata["skipped"] is True
        assert result.metadata["reason"] == "already_indexed"

    def test_index_document_overwrite_existing(
        self, indexer, sample_document, mock_embedder
    ):
        """Test overwriting existing embedding."""
        # Set existing embedding
        sample_document.embedding = [0.1] * 768

        with patch(
            "sejm_whiz.semantic_search.indexer.process_legal_document"
        ) as mock_process:
            mock_processed = {"clean_text": "processed legal text"}
            mock_process.return_value = mock_processed

            result = indexer.index_document(sample_document, overwrite_existing=True)

        # Should reindex
        assert result.success is True
        assert result.processing_time_ms > 0
        mock_embedder.embed_text.assert_called_once()

    def test_index_document_embedding_failure(
        self, indexer, sample_document, mock_embedder
    ):
        """Test indexing failure due to embedding generation."""
        # Mock embedding failure by raising an exception
        mock_embedder.embed_text.side_effect = Exception("Embedding generation failed")

        with patch(
            "sejm_whiz.semantic_search.indexer.process_legal_document"
        ) as mock_process:
            mock_processed = {"clean_text": "processed legal text"}
            mock_process.return_value = mock_processed

            result = indexer.index_document(sample_document)

        # Should fail
        assert result.success is False
        assert (
            result.error == "Failed to generate embedding: Embedding generation failed"
        )
        assert result.embedding_dimensions == 0

    def test_batch_index_documents(
        self, indexer, mock_embedder, mock_vector_operations
    ):
        """Test batch document indexing."""
        # Create sample documents
        documents = []
        for i in range(5):
            doc = LegalDocument(
                id=uuid4(),
                title=f"Document {i}",
                content=f"Content of document {i}",
                document_type="ustawa",
                legal_domain="prawo cywilne",
                embedding=None,
            )
            documents.append(doc)

        # Mock successful embedding results
        embedding_results = []
        for _ in range(5):
            result = Mock()
            result.embedding = np.random.random(768)
            embedding_results.append(result)

        # Mock embedder to return results
        mock_embedder.embed_texts.return_value = embedding_results

        with patch(
            "sejm_whiz.semantic_search.indexer.process_legal_document"
        ) as mock_process:
            mock_processed = {"clean_text": "processed text"}
            mock_process.return_value = mock_processed

            results = indexer.batch_index_documents(documents, batch_size=3)

        # Should have results for all documents
        assert len(results) == 5
        assert all(r.success for r in results)

        # Check database updates
        assert mock_vector_operations.update_document_embedding.call_count == 5

    def test_batch_index_with_existing_embeddings(
        self, indexer, mock_embedder, mock_vector_operations
    ):
        """Test batch indexing with mix of indexed and non-indexed documents."""
        documents = []

        # Document already indexed
        doc1 = LegalDocument(
            id=uuid4(),
            title="Already indexed",
            content="Content 1",
            embedding=[0.1] * 768,
        )
        documents.append(doc1)

        # Document not indexed
        doc2 = LegalDocument(
            id=uuid4(),
            title="Not indexed",
            content="Content 2",
            embedding=None,
        )
        documents.append(doc2)

        # Mock embedding for the non-indexed document
        embedding_result = Mock()
        embedding_result.embedding = np.random.random(768)

        # Mock embedder to return result for the one document that needs indexing
        mock_embedder.embed_texts.return_value = [embedding_result]

        with patch(
            "sejm_whiz.semantic_search.indexer.process_legal_document"
        ) as mock_process:
            mock_processed = {"clean_text": "processed text"}
            mock_process.return_value = mock_processed

            results = indexer.batch_index_documents(documents, overwrite_existing=False)

        assert len(results) == 2

        # First document should be skipped
        assert results[0].metadata["skipped"] is True

        # Second document should be processed
        assert results[1].success is True
        assert results[1].metadata.get("skipped") is None

    def test_reindex_all_documents(
        self, indexer, mock_embedder, mock_vector_operations
    ):
        """Test reindexing all documents."""
        # Mock documents from database
        documents = []
        for i in range(3):
            doc = LegalDocument(
                id=uuid4(),
                title=f"Document {i}",
                content=f"Content {i}",
                document_type="ustawa",
                embedding=[0.1] * 768,  # Already indexed
            )
            documents.append(doc)

        mock_vector_operations.get_documents.return_value = documents

        # Mock embedder for reindexing
        embedding_results = [Mock(embedding=np.random.random(768)) for _ in range(3)]
        mock_embedder.embed_texts.return_value = embedding_results

        with patch(
            "sejm_whiz.semantic_search.indexer.process_legal_document"
        ) as mock_process:
            mock_processed = {"clean_text": "processed text"}
            mock_process.return_value = mock_processed

            results = indexer.reindex_all_documents(document_type="ustawa")

        # Should get documents with filter
        mock_vector_operations.get_documents.assert_called_once_with(
            document_type="ustawa",
            legal_domain=None,
        )

        # Should reindex all documents (overwrite_existing=True)
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_get_indexing_statistics(self, indexer, mock_vector_operations):
        """Test getting indexing statistics."""
        stats = indexer.get_indexing_statistics()

        mock_vector_operations.get_embedding_statistics.assert_called_once()

        assert stats["total_documents"] == 100
        assert stats["indexed_documents"] == 80
        assert stats["indexing_percentage"] == 80.0
        assert stats["avg_embedding_dimensions"] == 768
        assert "document_types" in stats
        assert "legal_domains" in stats

    def test_get_indexing_statistics_failure(self, indexer, mock_vector_operations):
        """Test handling of statistics retrieval failure."""
        mock_vector_operations.get_embedding_statistics.side_effect = Exception(
            "Database error"
        )

        stats = indexer.get_indexing_statistics()

        # Should return empty dict on failure
        assert stats == {}

    def test_indexing_result_to_dict(self):
        """Test IndexingResult to_dict conversion."""
        result = IndexingResult(
            document_id=uuid4(),
            success=True,
            embedding_dimensions=768,
            processing_time_ms=150.5,
            error=None,
            metadata={"test": "data"},
        )

        result_dict = result.to_dict()

        assert result_dict["document_id"] == str(result.document_id)
        assert result_dict["success"] is True
        assert result_dict["embedding_dimensions"] == 768
        assert result_dict["processing_time_ms"] == 150.5
        assert result_dict["error"] is None
        assert result_dict["metadata"] == {"test": "data"}

    def test_get_document_indexer_singleton(self):
        """Test singleton pattern for document indexer."""
        indexer1 = get_document_indexer()
        indexer2 = get_document_indexer()

        assert indexer1 is indexer2

    def test_batch_index_partial_failure(
        self, indexer, mock_embedder, mock_vector_operations
    ):
        """Test batch indexing with partial failures."""
        documents = []
        for i in range(3):
            doc = LegalDocument(
                id=uuid4(),
                title=f"Document {i}",
                content=f"Content {i}",
                embedding=None,
            )
            documents.append(doc)

        # Mock mixed success/failure results
        embedding_results = [
            Mock(embedding=np.random.random(768)),  # Success
            Mock(embedding=None, error="Processing failed"),  # Failure
            Mock(embedding=np.random.random(768)),  # Success but DB will fail
        ]

        mock_embedder.embed_texts.return_value = embedding_results

        # Mock database failure for one document
        def mock_update_embedding(document_id, embedding):
            if document_id == documents[2].id:
                raise Exception("Database error")
            return True

        mock_vector_operations.update_document_embedding.side_effect = (
            mock_update_embedding
        )

        with patch(
            "sejm_whiz.semantic_search.indexer.process_legal_document"
        ) as mock_process:
            mock_processed = {"clean_text": "processed text"}
            mock_process.return_value = mock_processed

            results = indexer.batch_index_documents(documents)

        assert len(results) == 3

        # First document: success
        assert results[0].success is True

        # Second document: embedding failure
        assert results[1].success is False
        assert "Processing failed" in results[1].error

        # Third document: database failure
        assert results[2].success is False
        assert "Database update failed" in results[2].error
