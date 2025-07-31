"""Integration tests for ELI API component."""

import pytest
from unittest.mock import AsyncMock, patch

from sejm_whiz.eli_api import (
    EliApiClient,
    EliApiConfig,
    LegalDocument,
    DocumentSearchResult,
    validate_eli_id,
    sanitize_query,
    LegalTextParser,
    MultiActAmendmentDetector,
)


class TestEliApiIntegration:
    """Integration tests for ELI API component."""

    @pytest.mark.asyncio
    async def test_client_search_and_parse_workflow(self):
        """Test complete workflow: search, fetch, and parse document."""
        config = EliApiConfig(rate_limit=0)  # No rate limiting for tests

        # Mock API responses
        mock_search_response = {
            "documents": [
                {
                    "eli_id": "pl/test/2023/1",
                    "title": "Test Amendment Act",
                    "type": "ustawa",
                    "status": "obowiązująca",
                }
            ],
            "total": 1,
        }

        mock_document_response = {
            "eli_id": "pl/test/2023/1",
            "title": "Test Amendment Act",
            "type": "ustawa",
            "status": "obowiązująca",
            "published_date": "2023-01-01T00:00:00Z",
        }

        mock_content = """
        <html><body>
        <h1>Ustawa o zmianie niektórych ustaw</h1>
        <p>Art. 1. W ustawie z dnia 1 stycznia 2020 r. wprowadza się zmiany...</p>
        <p>Art. 2. W Kodeksie cywilnym zmienia się art. 123...</p>
        </body></html>
        """

        async with EliApiClient(config) as client:
            with patch.object(client, "_make_request") as mock_request:
                # Mock search request
                mock_request.return_value = mock_search_response

                # Search for documents
                search_result = await client.search_documents("amendment")

                assert isinstance(search_result, DocumentSearchResult)
                assert len(search_result.documents) == 1
                assert search_result.documents[0].eli_id == "pl/test/2023/1"

                # Mock document fetch
                mock_request.return_value = mock_document_response

                # Fetch specific document
                document = await client.get_document("pl/test/2023/1")

                assert isinstance(document, LegalDocument)
                assert document.eli_id == "pl/test/2023/1"

                # Mock content fetch
                with (
                    patch.object(client, "_rate_limiter") as mock_limiter,
                    patch.object(client, "_ensure_client"),
                ):
                    mock_limiter.acquire = AsyncMock()
                    mock_response = AsyncMock()
                    mock_response.status_code = 200
                    mock_response.text = mock_content
                    mock_response.raise_for_status = AsyncMock()

                    client._client = AsyncMock()
                    client._client.get.return_value = mock_response

                    # Fetch document content
                    content = await client.get_document_content(
                        "pl/test/2023/1", "html"
                    )

                    assert len(content) > 0
                    assert "Art. 1" in content

                    # Parse content
                    parser = LegalTextParser()
                    structure = parser.parse_html_content(content)

                    assert structure.title
                    assert len(structure.articles) >= 2

                    # Detect multi-act amendment
                    detector = MultiActAmendmentDetector()
                    multi_act = detector.detect_multi_act_amendments(document, content)

                    assert multi_act is not None  # Should detect multi-act amendment
                    assert multi_act.complexity_score >= 2

    def test_utility_functions_integration(self):
        """Test utility functions work together."""
        # Test query sanitization and validation
        raw_query = "  test'; DROP TABLE users; --  "
        sanitized = sanitize_query(raw_query)

        assert sanitized != raw_query
        assert "DROP TABLE" not in sanitized
        assert len(sanitized.strip()) > 0

        # Test ELI ID validation and formatting
        test_eli_ids = [
            "pl/test/2023/1",
            "PL/TEST/2023/1",
            "invalid-eli-id",
            "us/test/2023/1",
        ]

        valid_count = sum(1 for eli_id in test_eli_ids if validate_eli_id(eli_id))
        assert valid_count == 2  # Only first two should be valid

    @pytest.mark.asyncio
    async def test_error_handling_chain(self):
        """Test error handling across component functions."""
        config = EliApiConfig(rate_limit=0)

        async with EliApiClient(config) as client:
            # Test validation errors
            with pytest.raises(Exception):  # Should raise validation error
                await client.search_documents(limit=0)

            with pytest.raises(Exception):  # Should raise validation error
                await client.get_document("invalid-eli-id")

            # Test API errors
            with patch.object(client, "_make_request") as mock_request:
                from sejm_whiz.eli_api.client import EliNotFoundError

                mock_request.side_effect = EliNotFoundError("Not found")

                with pytest.raises(EliNotFoundError):
                    await client.get_document("pl/nonexistent/2023/1")

    def test_model_integration(self):
        """Test model creation and validation integration."""
        # Test creating document from API response
        api_data = {
            "eli_id": "pl/test/2023/1",
            "title": "Test Integration Act",
            "type": "ustawa",
            "status": "obowiązująca",
            "published_date": "2023-01-01T00:00:00Z",
            "keywords": ["integration", "test"],
        }

        document = LegalDocument.from_api_response(api_data)

        assert document.eli_id == "pl/test/2023/1"
        assert document.title == "Test Integration Act"
        assert len(document.keywords) == 2

        # Test document methods
        assert document.is_in_force() is True

        # Test converting back to dict
        doc_dict = document.to_dict()
        assert doc_dict["eli_id"] == "pl/test/2023/1"
        assert "created_at" not in doc_dict

    def test_parser_integration_with_models(self):
        """Test parser integration with models."""
        # Create sample document
        document = LegalDocument(
            eli_id="pl/omnibus/2023/1",
            title="Complex Amendment Act",
            document_type="ustawa",
            status="obowiązująca",
        )

        # Complex legal content
        content = """
        <html><body>
        <h1>Ustawa o zmianie niektórych ustaw w zakresie modernizacji prawa</h1>

        <h2>Rozdział I. Postanowienia ogólne</h2>
        <p>Art. 1. Zakres ustawy.</p>

        <h2>Rozdział II. Zmiany w ustawach</h2>
        <p>Art. 2. W ustawie z dnia 1 stycznia 2020 r. o procedurach administracyjnych
        wprowadza się następujące zmiany:</p>
        <p>1) w art. 123 zmienia się § 1;</p>
        <p>2) dodaje się art. 123a w brzmieniu...</p>

        <p>Art. 3. W Kodeksie postępowania cywilnego zmienia się art. 456.</p>

        <p>Art. 4. W ustawie z dnia 15 lutego 2021 r. o ochronie środowiska
        uchyla się art. 789.</p>
        </body></html>
        """

        # Parse structure
        parser = LegalTextParser()
        structure = parser.parse_html_content(content)

        assert len(structure.chapters) >= 2
        assert len(structure.articles) >= 4
        assert len(structure.cross_references) >= 3

        # Detect multi-act amendment
        detector = MultiActAmendmentDetector()
        multi_act = detector.detect_multi_act_amendments(document, content)

        assert multi_act is not None
        assert multi_act.complexity_score >= 3
        assert multi_act.is_omnibus() is True
        assert len(multi_act.affected_acts) >= 3

    @pytest.mark.asyncio
    async def test_batch_operations(self):
        """Test batch operations integration."""
        config = EliApiConfig(rate_limit=0)
        eli_ids = ["pl/test/2023/1", "pl/test/2023/2", "pl/nonexistent/2023/1"]

        async with EliApiClient(config) as client:
            with patch.object(client, "get_document") as mock_get:

                def mock_get_document(eli_id):
                    if "nonexistent" in eli_id:
                        from sejm_whiz.eli_api.client import EliNotFoundError

                        raise EliNotFoundError("Not found")

                    return LegalDocument(
                        eli_id=eli_id,
                        title=f"Test Document {eli_id[-1]}",
                        document_type="ustawa",
                        status="obowiązująca",
                    )

                mock_get.side_effect = mock_get_document

                results = await client.batch_get_documents(eli_ids)

                assert len(results) == 3
                assert results[0] is not None  # First document found
                assert results[1] is not None  # Second document found
                assert results[2] is None  # Third document not found


def test_component_imports():
    """Test that all component imports work correctly."""
    # Test importing main classes
    from sejm_whiz.eli_api import (
        EliApiClient,
        EliApiConfig,
        LegalTextParser,
        MultiActAmendmentDetector,
        validate_eli_id,
        sanitize_query,
        normalize_document_type,
        extract_legal_references,
        is_amendment_document,
        clean_legal_text,
    )

    # Test that classes can be instantiated
    config = EliApiConfig()
    assert config.base_url

    client = EliApiClient(config)
    assert client.config == config

    parser = LegalTextParser()
    assert parser is not None

    detector = MultiActAmendmentDetector()
    assert detector is not None

    # Test utility functions
    assert validate_eli_id("pl/test/2023/1") is True
    assert sanitize_query("test query") == "test query"
    assert normalize_document_type("ustawa") == "ustawa"
    assert isinstance(extract_legal_references("art. 123"), list)
    assert is_amendment_document("ustawa o zmianie") is True
    assert clean_legal_text("test") == "test"
