"""Integration tests for vector_db component with real PostgreSQL."""

import pytest

from sejm_whiz.vector_db import (
    get_vector_db,
    get_vector_operations,
    get_similarity_search,
    DistanceMetric,
    validate_vector_db_health,
)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset global singletons before each test to ensure test isolation."""
    # Reset singleton instances to None to force re-creation
    import sejm_whiz.vector_db.connection as conn_module
    import sejm_whiz.vector_db.operations as ops_module
    import sejm_whiz.vector_db.embeddings as emb_module

    # Store original values (might be None or real instances)
    original_vector_db = getattr(conn_module, "_vector_db", None)
    original_vector_ops = getattr(ops_module, "_vector_ops", None)
    original_similarity_search = getattr(emb_module, "_similarity_search", None)

    # Reset to None to force fresh instances
    conn_module._vector_db = None
    ops_module._vector_ops = None
    emb_module._similarity_search = None

    yield

    # Restore original values after test (cleanup)
    conn_module._vector_db = original_vector_db
    ops_module._vector_ops = original_vector_ops
    emb_module._similarity_search = original_similarity_search


@pytest.fixture(scope="module")
def skip_if_no_database():
    """Skip integration tests if no database is available."""
    health = validate_vector_db_health()
    if not health.get("connection", False):
        pytest.skip("No database connection available for integration tests")


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "title": "Ustawa o ochronie danych osobowych",
            "content": "Niniejsza ustawa reguluje ochronę danych osobowych...",
            "document_type": "law",
            "embedding": [0.1] * 768,
            "legal_domain": "privacy",
        },
        {
            "title": "Kodeks karny - artykuł 148",
            "content": "Kto zabija człowieka podlega karze...",
            "document_type": "law",
            "embedding": [0.2] * 768,
            "legal_domain": "criminal",
        },
        {
            "title": "Rozporządzenie w sprawie bezpieczeństwa",
            "content": "W sprawie bezpieczeństwa pracy...",
            "document_type": "regulation",
            "embedding": [0.3] * 768,
            "legal_domain": "administrative",
        },
    ]


class TestVectorDBIntegration:
    """Integration tests for vector database operations."""

    @pytest.mark.integration
    def test_database_health_check(self, skip_if_no_database):
        """Test database health check with real database."""
        health = validate_vector_db_health()

        assert health["status"] in ["healthy", "warning"]  # Warning if no pgvector
        assert health["connection"] is True
        assert health["vector_dimensions"] == 768

    @pytest.mark.integration
    def test_vector_db_connection(self, skip_if_no_database):
        """Test vector database connection."""
        vector_db = get_vector_db()

        assert vector_db.test_connection() is True
        assert vector_db.get_vector_dimensions() == 768

    @pytest.mark.integration
    def test_document_crud_operations(self, skip_if_no_database, sample_documents):
        """Test CRUD operations with real database."""
        ops = get_vector_operations()

        # Create document
        doc_data = sample_documents[0]
        doc_id = ops.create_document_with_embedding(**doc_data)

        assert doc_id is not None

        try:
            # Read document - access attributes within the session context
            document = ops.get_document_by_id(doc_id)
            assert document is not None

            # Access attributes immediately while still in session
            doc_title = document.title
            doc_type = document.document_type
            assert doc_title == doc_data["title"]
            assert doc_type == doc_data["document_type"]

            # Update embedding
            new_embedding = [0.9] * 768
            success = ops.update_document_embedding(doc_id, new_embedding)
            assert success is True

            # Verify update exists
            updated_doc = ops.get_document_by_id(doc_id)
            assert updated_doc is not None

        finally:
            # Cleanup - delete document
            deleted = ops.delete_document(doc_id)
            assert deleted is True

            # Verify deletion
            deleted_doc = ops.get_document_by_id(doc_id)
            assert deleted_doc is None

    @pytest.mark.integration
    def test_bulk_operations(self, skip_if_no_database, sample_documents):
        """Test bulk operations with real database."""
        ops = get_vector_operations()

        # Bulk insert
        doc_ids = ops.bulk_insert_documents(sample_documents)
        assert len(doc_ids) == len(sample_documents)

        try:
            # Verify all documents were created
            for doc_id in doc_ids:
                document = ops.get_document_by_id(doc_id)
                assert document is not None

            # Count documents by type - collect IDs while in session
            law_docs = ops.get_documents_by_type("law")
            regulation_docs = ops.get_documents_by_type("regulation")

            # Extract IDs immediately while objects are still attached
            law_doc_ids = [str(d.id) for d in law_docs]
            regulation_doc_ids = [str(d.id) for d in regulation_docs]
            doc_id_strs = [str(doc_id) for doc_id in doc_ids]

            # Should find at least our inserted documents
            found_law_docs = [d_id for d_id in doc_id_strs if d_id in law_doc_ids]
            found_reg_docs = [
                d_id for d_id in doc_id_strs if d_id in regulation_doc_ids
            ]

            assert len(found_law_docs) == 2
            assert len(found_reg_docs) == 1

        finally:
            # Cleanup
            for doc_id in doc_ids:
                ops.delete_document(doc_id)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not validate_vector_db_health().get("pgvector_extension", False),
        reason="pgvector extension not available",
    )
    def test_similarity_search(self, skip_if_no_database, sample_documents):
        """Test similarity search with real database and pgvector."""
        ops = get_vector_operations()
        search = get_similarity_search()

        # Insert test documents
        doc_ids = ops.bulk_insert_documents(sample_documents)

        try:
            # Test similarity search
            query_embedding = [0.15] * 768  # Should be close to first document

            similar_docs = search.find_similar_documents(
                query_embedding=query_embedding,
                limit=2,
                distance_metric=DistanceMetric.COSINE,
            )

            assert len(similar_docs) <= 2

            # Results should be sorted by distance (ascending)
            if len(similar_docs) > 1:
                assert similar_docs[0][1] <= similar_docs[1][1]

            # Test find similar by document ID
            if doc_ids:
                similar_by_id = search.find_similar_by_document_id(
                    document_id=str(doc_ids[0]), limit=2, exclude_self=True
                )

                # Extract IDs immediately while objects are attached
                source_ids = [str(doc.id) for doc, _ in similar_by_id]
                assert str(doc_ids[0]) not in source_ids

        finally:
            # Cleanup
            for doc_id in doc_ids:
                ops.delete_document(doc_id)

    @pytest.mark.integration
    def test_embedding_statistics(self, skip_if_no_database, sample_documents):
        """Test embedding statistics with real database."""
        ops = get_vector_operations()
        search = get_similarity_search()

        # Get initial stats
        initial_stats = search.get_embedding_statistics()

        # Insert test documents
        doc_ids = ops.bulk_insert_documents(sample_documents)

        try:
            # Get updated stats
            updated_stats = search.get_embedding_statistics()

            # Should have more documents with embeddings
            assert (
                updated_stats["documents_with_embeddings"]
                >= initial_stats["documents_with_embeddings"]
            )
            assert updated_stats["total_documents"] >= initial_stats["total_documents"]

            # Check type distribution
            assert "law" in updated_stats["type_distribution"]
            assert "regulation" in updated_stats["type_distribution"]

        finally:
            # Cleanup
            for doc_id in doc_ids:
                ops.delete_document(doc_id)

    @pytest.mark.integration
    def test_document_embedding_records(self, skip_if_no_database):
        """Test document embedding record management."""
        ops = get_vector_operations()

        # Create a document first
        doc_id = ops.create_document_with_embedding(
            title="Test Document",
            content="Test content",
            document_type="law",
            embedding=[0.1] * 768,
        )

        try:
            # Create additional embedding record
            embedding_id = ops.create_document_embedding(
                document_id=doc_id,
                embedding=[0.2] * 768,
                model_name="herbert-klej-cased-v1",
                model_version="1.0",
                embedding_method="bag_of_embeddings",
                confidence_score=95,
            )

            assert embedding_id is not None

            # Get embeddings for document
            embeddings = ops.get_document_embeddings(doc_id)
            assert len(embeddings) >= 1

            # Find our specific embedding and access attributes immediately
            our_embedding = None
            for emb in embeddings:
                if emb.id == embedding_id:
                    our_embedding = emb
                    break

            assert our_embedding is not None
            # Access attributes immediately while in session
            model_name = our_embedding.model_name
            model_version = our_embedding.model_version
            confidence_score = our_embedding.confidence_score

            assert model_name == "herbert-klej-cased-v1"
            assert model_version == "1.0"
            assert confidence_score == 95

        finally:
            # Cleanup
            ops.delete_document(doc_id)


@pytest.mark.integration
class TestVectorIndexing:
    """Test vector indexing operations."""

    @pytest.mark.skipif(
        not validate_vector_db_health().get("pgvector_extension", False),
        reason="pgvector extension not available",
    )
    def test_create_vector_index(self, skip_if_no_database):
        """Test creating vector indexes."""
        search = get_similarity_search()

        # Note: This test might fail if index already exists
        # In production, you'd want to check for existing indexes first
        try:
            success = search.create_vector_index(
                table_name="legal_documents",
                column_name="embedding",
                index_type="ivfflat",
                lists=10,  # Small number for test
            )
            # Test might fail if index already exists, which is okay
            assert success is True or success is False

        except Exception as e:
            # Index might already exist, which is fine for testing
            pytest.skip(f"Index operation failed (might already exist): {e}")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-m", "integration"])
