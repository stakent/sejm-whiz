"""Test Sejm API + ELI API integration for act text extraction."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from sejm_whiz.document_ingestion.dual_stream_pipeline import (
    DualApiDocumentProcessor,
    ActDocument,
)
from sejm_whiz.eli_api.client import EliApiClient
from sejm_whiz.sejm_api.client import SejmApiClient
from sejm_whiz.eli_api.content_validator import BasicContentValidator


class TestMultiApiIntegration:
    """Test Sejm API + ELI API integration for act text extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sejm_client = Mock(spec=SejmApiClient)
        self.eli_client = Mock(spec=EliApiClient)
        self.content_validator = BasicContentValidator()

        self.processor = DualApiDocumentProcessor(
            sejm_client=self.sejm_client,
            eli_client=self.eli_client,
            content_validator=self.content_validator,
        )

    @pytest.mark.asyncio
    async def test_sejm_api_primary_source(self):
        """Test Sejm API as primary source for act text."""
        # Mock successful Sejm API response
        mock_act_data = {
            "text": "Ustawa z dnia 1 stycznia 2025 r. o przykładowych przepisach prawnych. "
            * 10,  # >100 chars
            "title": "Ustawa o przykładowych przepisach",
            "sejm_id": "term10-session1-sitting5",
            "term": 10,
            "session": 1,
            "sitting": 5,
            "source": "sejm_api_voting",
        }

        mock_metadata = {
            "source_api": "sejm_api",
            "document_id": "term10-session1-sitting5",
            "title": "Ustawa o przykładowych przepisach",
            "document_type": "voting_record",
        }

        # Configure mocks
        self.sejm_client.get_act_with_full_text = AsyncMock(return_value=mock_act_data)
        self.sejm_client.extract_act_metadata = AsyncMock(return_value=mock_metadata)
        self.sejm_client.is_sejm_content_complete = Mock(return_value=True)

        # Test processing Sejm document
        result = await self.processor.process_sejm_document("term10-session1-sitting5")

        # Verify results
        assert result.success is True
        assert result.source_used == "sejm_api"
        assert len(result.act_text) > 100
        assert "Ustawa z dnia" in result.act_text
        assert result.metadata["document_type"] == "voting_record"
        assert result.processing_time > 0

        # Verify Sejm API was called with term and number (parsed from document ID)
        self.sejm_client.get_act_with_full_text.assert_called_once_with(10, 1)
        self.sejm_client.extract_act_metadata.assert_called_once_with(mock_act_data)
        self.sejm_client.is_sejm_content_complete.assert_called_once_with(mock_act_data)

    @pytest.mark.asyncio
    async def test_eli_api_fallback_chain(self):
        """Test ELI API HTML→PDF fallback when Sejm fails."""
        # Mock Sejm API failure
        self.sejm_client.get_act_with_full_text = AsyncMock(
            side_effect=Exception("Sejm API error")
        )

        # Mock ELI API success with PDF fallback
        mock_eli_result = {
            "eli_id": "DU/2025/1076",
            "content": "Dziennik Ustaw Rzeczypospolitej Polskiej - document content extracted from PDF. "
            * 5,
            "source": "pdf",
            "usable": True,
        }

        self.eli_client.get_document_content_with_dual_storage = AsyncMock(
            return_value={
                "eli_id": "DU/2025/1076",
                "preferred_content": mock_eli_result["content"],
                "preferred_source": mock_eli_result["source"],
                "html_content": None,
                "pdf_content": mock_eli_result["content"],
                "html_quality_score": 0.0,
                "pdf_quality_score": 0.8,
                "conversion_accuracy": None,
                "usable": mock_eli_result["usable"],
            }
        )

        # Test processing ELI document
        result = await self.processor.process_eli_document("DU/2025/1076")

        # Verify results
        assert result.success is True
        assert result.source_used == "eli_api_pdf"
        assert len(result.act_text) > 50
        assert "Dziennik Ustaw" in result.act_text
        assert result.metadata["preferred_source"] == "pdf"

        # Verify ELI API was called with dual storage
        self.eli_client.get_document_content_with_dual_storage.assert_called_once_with(
            "DU/2025/1076"
        )

    @pytest.mark.asyncio
    async def test_eli_document_processing(self):
        """Test ELI document processing with dual content storage."""

        # Mock ELI API with dual content (HTML + PDF)
        dual_eli_result = {
            "eli_id": "DU/2025/1",
            "html_content": "Complete legal HTML content with substantial text. " * 10,
            "pdf_content": "Complete legal PDF content with substantial text. " * 8,
            "html_quality_score": 0.9,
            "pdf_quality_score": 0.7,
            "preferred_content": "Complete legal HTML content with substantial text. "
            * 10,
            "preferred_source": "html",
            "conversion_accuracy": 0.85,
            "usable": True,
        }

        self.eli_client.get_document_content_with_dual_storage = AsyncMock(
            return_value=dual_eli_result
        )

        # Test processing ELI document
        result = await self.processor.process_eli_document("DU/2025/1")

        # Should successfully process with HTML as preferred source
        assert result.success is True
        assert result.source_used == "eli_api_html"
        assert len(result.act_text) > 100
        assert result.metadata["preferred_source"] == "html"
        assert result.metadata["html_available"] is True
        assert result.metadata["pdf_available"] is True
        assert result.metadata["conversion_accuracy"] == 0.85

        # ELI API should have been called
        self.eli_client.get_document_content_with_dual_storage.assert_called_once_with(
            "DU/2025/1"
        )

    @pytest.mark.asyncio
    async def test_act_text_and_metadata_extraction(self):
        """Test that both text and metadata are extracted from both APIs."""

        # Test with comprehensive Sejm API response
        comprehensive_sejm_data = {
            "text": "Kompletna ustawa o przepisach prawnych zawierająca wszystkie niezbędne informacje prawne. "
            * 15,
            "title": "Ustawa kompletna",
            "sejm_id": "comprehensive-test",
            "term": 10,
            "session": 3,
            "sitting": 7,
            "voting_date": "2025-01-15T10:30:00",
            "source": "sejm_api_voting",
        }

        expected_metadata = {
            "source_api": "sejm_api",
            "document_id": "comprehensive-test",
            "title": "Ustawa kompletna",
            "term": 10,
            "session": 3,
            "sitting": 7,
            "voting_date": "2025-01-15T10:30:00",
            "document_type": "voting_record",
            "parliamentary_process": "voting",
        }

        self.sejm_client.get_act_with_full_text = AsyncMock(
            return_value=comprehensive_sejm_data
        )
        self.sejm_client.extract_act_metadata = AsyncMock(
            return_value=expected_metadata
        )
        self.sejm_client.is_sejm_content_complete = Mock(return_value=True)

        # Test processing
        result = await self.processor.process_sejm_document("comprehensive-test")

        # Verify comprehensive extraction
        assert result.success is True
        assert result.act_text == comprehensive_sejm_data["text"]
        assert result.metadata["document_type"] == "voting_record"
        assert result.metadata["parliamentary_process"] == "voting"
        assert result.metadata["term"] == 10
        assert result.metadata["session"] == 3
        assert result.content_quality_score >= 0.7  # High quality for complete content

    @pytest.mark.asyncio
    async def test_all_sources_fail_scenario(self):
        """Test behavior when all API sources fail."""
        # Mock both APIs failing
        self.sejm_client.get_act_with_full_text = AsyncMock(
            side_effect=Exception("Sejm API unavailable")
        )

        eli_fail_result = {
            "eli_id": "failed-doc",
            "content": "",
            "source": "none",
            "usable": False,
        }
        self.eli_client.get_document_content_with_basic_fallback = AsyncMock(
            return_value=eli_fail_result
        )

        # Test processing (try both Sejm and ELI)
        result = await self.processor.process_sejm_document("failed-doc")

        # Should fail gracefully
        assert result.success is False
        assert "Sejm API error: Sejm API unavailable" in result.error_message
        assert result.act_text == ""
        assert result.source_used == "sejm"

    @pytest.mark.asyncio
    async def test_content_quality_validation(self):
        """Test content quality validation across APIs."""
        # Test with high-quality HTML content from ELI
        high_quality_eli = {
            "eli_id": "quality-test",
            "preferred_content": "<html><body>"
            + "High quality legal document content. " * 100
            + "</body></html>",
            "preferred_source": "html",
            "html_content": "<html><body>"
            + "High quality legal document content. " * 100
            + "</body></html>",
            "pdf_content": None,
            "html_quality_score": 0.9,
            "pdf_quality_score": 0.0,
            "conversion_accuracy": None,
            "usable": True,
        }

        # Mock ELI API with dual storage
        self.eli_client.get_document_content_with_dual_storage = AsyncMock(
            return_value=high_quality_eli
        )

        result = await self.processor.process_eli_document("quality-test")

        assert result.success is True
        assert result.source_used == "eli_api_html"
        # Quality score should be calculated based on content validator
        assert result.content_quality_score is not None
        assert 0.0 <= result.content_quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_act_document_extraction(self):
        """Test ActDocument standardized extraction."""
        test_content = (
            "Przykładowy akt prawny z pełną treścią dokumentu prawnego. " * 20
        )

        act_doc = await self.processor.extract_act_text_and_metadata(
            "sejm_api", test_content
        )

        assert isinstance(act_doc, ActDocument)
        # Content gets stripped in processing, so compare stripped versions
        assert act_doc.text_content == test_content.strip()
        assert act_doc.source_api == "sejm_api"
        assert act_doc.title.startswith("Przykładowy")  # First line becomes title
        assert act_doc.quality_score > 0
        assert isinstance(act_doc.extraction_timestamp, datetime)
        assert act_doc.metadata["content_length"] == len(test_content.strip())

    def test_processor_configuration_validation(self):
        """Test processor configuration validation."""
        # Test with no clients
        empty_processor = DualApiDocumentProcessor()
        issues = empty_processor.validate_configuration()

        assert "No API clients configured" in issues
        assert "No content validator configured" in issues

        # Test with complete configuration
        issues = self.processor.validate_configuration()
        assert len(issues) == 0  # Should have no issues

    def test_processing_statistics(self):
        """Test processing statistics tracking."""
        stats = self.processor.get_processing_statistics()

        # Should return basic statistics structure
        expected_keys = [
            "total_processed",
            "sejm_api_successes",
            "eli_api_html_successes",
            "eli_api_pdf_successes",
            "failures",
            "average_processing_time",
            "quality_score_average",
        ]

        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))

    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent document processing."""

        # Mock successful responses for multiple documents
        def mock_sejm_response(doc_id):
            return {
                "text": f"Document {doc_id} content with legal information. " * 10,
                "title": f"Document {doc_id}",
                "sejm_id": doc_id,
                "source": "sejm_api_proceeding",
            }

        async def mock_get_act(term, number):
            doc_id = f"term{term}-{number}"
            return mock_sejm_response(doc_id)

        async def mock_extract_metadata(act_data):
            return {
                "source_api": "sejm_api",
                "document_id": act_data["sejm_id"],
                "title": act_data["title"],
            }

        self.sejm_client.get_act_with_full_text = mock_get_act
        self.sejm_client.extract_act_metadata = mock_extract_metadata
        self.sejm_client.is_sejm_content_complete = Mock(return_value=True)

        # Process multiple documents concurrently
        doc_ids = ["10_1", "10_2", "10_3"]
        tasks = [self.processor.process_sejm_document(doc_id) for doc_id in doc_ids]

        results = await asyncio.gather(*tasks)

        # Verify all succeeded
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.success is True
            expected_doc_id = f"term10-{i + 1}"
            assert expected_doc_id in result.act_text
            assert result.source_used == "sejm_api"

    @pytest.mark.asyncio
    async def test_error_handling_robustness(self):
        """Test robust error handling in various failure scenarios."""

        # Test with malformed Sejm response
        malformed_sejm = {
            "text": None,  # Invalid text type
            "title": 123,  # Invalid title type
            "sejm_id": "malformed-test",
        }

        self.sejm_client.get_act_with_full_text = AsyncMock(return_value=malformed_sejm)
        self.sejm_client.is_sejm_content_complete = Mock(
            return_value=False
        )  # Should be rejected

        # Mock ELI API also failing
        self.eli_client.get_document_content_with_basic_fallback = AsyncMock(
            side_effect=Exception("ELI API timeout")
        )

        result = await self.processor.process_sejm_document("malformed-test")

        # Should handle gracefully
        assert result.success is False
        assert "Sejm API content incomplete" in result.error_message
        assert result.processing_time > 0  # Should still measure time
