"""Basic PDF conversion tests for interim milestone."""

import pytest
from unittest.mock import Mock, patch

from sejm_whiz.eli_api.pdf_converter import BasicPDFConverter
from sejm_whiz.eli_api.content_validator import BasicContentValidator


class TestBasicPDFConversion:
    """Basic PDF conversion tests for interim milestone."""

    def setup_method(self):
        """Set up test fixtures."""
        self.pdf_converter = BasicPDFConverter()
        self.content_validator = BasicContentValidator()

    def test_pdf_converter_initialization(self):
        """Test PDF converter initializes correctly."""
        converter = BasicPDFConverter()
        assert converter.engine == "pdfplumber"

    def test_pdf_converter_invalid_engine(self):
        """Test PDF converter rejects invalid engine."""
        with pytest.raises(ValueError, match="Unsupported engine"):
            BasicPDFConverter(engine="invalid_engine")

    def create_mock_pdf_content(self, text_content: str) -> bytes:
        """Create mock PDF content for testing."""
        # This creates a simple mock that represents PDF bytes
        return f"PDF_MOCK_CONTENT:{text_content}".encode("utf-8")

    @patch("pdfplumber.open")
    @pytest.mark.asyncio
    async def test_pdf_to_text_basic_functionality(self, mock_pdfplumber_open):
        """Test that PDF conversion produces any usable text."""
        # Mock pdfplumber response
        mock_page = Mock()
        mock_page.extract_text.return_value = (
            "Test document content with Polish characters: ąćęłńóśźż"
        )

        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)

        mock_pdfplumber_open.return_value = mock_pdf

        # Test conversion
        pdf_content = self.create_mock_pdf_content("test content")
        result = await self.pdf_converter.convert_pdf_to_text(pdf_content)

        # Verify result
        assert result is not None
        assert len(result) > 0
        assert "Test document content" in result
        assert "ąćęłńóśźż" in result  # Polish characters preserved

        # Verify pdfplumber was called correctly
        mock_pdfplumber_open.assert_called_once()

    def test_minimum_text_extraction(self):
        """Test that minimum text thresholds are met."""
        # Test with text meeting minimum threshold
        good_text = "This is a test document with sufficient content to meet the minimum character threshold for PDF extraction validation."
        assert self.pdf_converter.is_conversion_acceptable(good_text, min_chars=50)

        # Test with text below threshold
        short_text = "Too short"
        assert not self.pdf_converter.is_conversion_acceptable(short_text, min_chars=50)

        # Test with empty text
        assert not self.pdf_converter.is_conversion_acceptable("", min_chars=50)
        assert not self.pdf_converter.is_conversion_acceptable(None, min_chars=50)

    def test_polish_character_handling(self):
        """Basic test for Polish character preservation."""
        polish_text = "Dokument zawiera polskie znaki: ąćęłńóśźż ĄĆĘŁŃÓŚŹŻ"

        # Test quality info extraction
        quality_info = self.pdf_converter.get_text_quality_info(polish_text)

        assert quality_info["char_count"] > 0
        assert quality_info["word_count"] > 0
        assert quality_info["has_polish_chars"] is True
        assert quality_info["quality_score"] > 0

        # Test with non-Polish text
        english_text = "This document contains only English characters"
        quality_info_en = self.pdf_converter.get_text_quality_info(english_text)
        assert quality_info_en["has_polish_chars"] is False

    @patch("pdfplumber.open")
    @pytest.mark.asyncio
    async def test_conversion_fallback_chain(self, mock_pdfplumber_open):
        """Test HTML→PDF fallback workflow."""
        from sejm_whiz.eli_api.client import EliApiClient

        # Mock successful PDF conversion
        mock_page = Mock()
        mock_page.extract_text.return_value = (
            "Successfully extracted PDF content with legal text."
        )

        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)

        mock_pdfplumber_open.return_value = mock_pdf

        # Create ELI client with mock PDF converter
        eli_client = EliApiClient()

        # Test the fallback method
        with (
            patch.object(eli_client, "get_document_content") as mock_get_content,
            patch.object(eli_client, "_get_document_content_raw") as mock_pdf_fetch,
        ):
            # Mock HTML fetch returns content that fails validation (too short)
            def mock_get_content_side_effect(eli_id, format_type):
                if format_type == "html":
                    return "Too short"  # Will fail HTML validation (< 100 chars)
                elif format_type == "pdf":
                    return "PDF text content"  # Will be ignored, we use _get_document_content_raw for PDF
                return None

            mock_get_content.side_effect = mock_get_content_side_effect

            # Mock PDF fetch success
            mock_pdf_fetch.return_value = self.create_mock_pdf_content("legal document")

            # Test the fallback chain
            result = await eli_client.get_document_content_with_basic_fallback(
                "DU/2025/1076"
            )

            assert result["usable"] is True
            assert result["source"] == "pdf"
            assert len(result["content"]) > 50  # Meets minimum threshold
            assert "Successfully extracted PDF content" in result["content"]

    @patch("pdfplumber.open")
    @pytest.mark.asyncio
    async def test_pdf_conversion_error_handling(self, mock_pdfplumber_open):
        """Test PDF conversion handles errors gracefully."""
        # Mock pdfplumber throwing an exception
        mock_pdfplumber_open.side_effect = Exception("PDF parsing failed")

        pdf_content = self.create_mock_pdf_content("test")

        with pytest.raises(Exception):
            await self.pdf_converter.convert_pdf_to_text(pdf_content)

    def test_content_validator_integration(self):
        """Test PDF converter works with content validator."""
        # Test HTML content validation
        good_html = "<html><body>" + "A" * 150 + "</body></html>"  # 150+ chars
        assert self.content_validator.is_html_content_usable(good_html)

        short_html = "<html><body>Too short</body></html>"  # < 100 chars
        assert not self.content_validator.is_html_content_usable(short_html)

        # Test PDF text validation
        good_pdf_text = (
            "This is a sufficiently long PDF text content for validation purposes."
        )
        assert self.content_validator.is_pdf_text_usable(good_pdf_text)

        short_pdf_text = "Too short"
        assert not self.content_validator.is_pdf_text_usable(short_pdf_text)

    def test_content_quality_scoring(self):
        """Test content quality scoring for different sources."""
        html_content = "<html><body>" + "Quality HTML content " * 20 + "</body></html>"
        pdf_content = "Quality PDF text content " * 15

        html_score = self.content_validator.get_content_quality_score(
            html_content, "html"
        )
        pdf_score = self.content_validator.get_content_quality_score(pdf_content, "pdf")

        assert 0.0 <= html_score <= 1.0
        assert 0.0 <= pdf_score <= 1.0

        # HTML should have different scoring than PDF
        assert html_score != pdf_score

    @pytest.mark.asyncio
    async def test_multiple_pages_pdf(self):
        """Test PDF with multiple pages."""
        with patch("pdfplumber.open") as mock_pdfplumber_open:
            # Mock multiple pages
            mock_page1 = Mock()
            mock_page1.extract_text.return_value = (
                "Page 1 content with legal information."
            )

            mock_page2 = Mock()
            mock_page2.extract_text.return_value = (
                "Page 2 content with additional legal details."
            )

            mock_pdf = Mock()
            mock_pdf.pages = [mock_page1, mock_page2]
            mock_pdf.__enter__ = Mock(return_value=mock_pdf)
            mock_pdf.__exit__ = Mock(return_value=False)

            mock_pdfplumber_open.return_value = mock_pdf

            pdf_content = self.create_mock_pdf_content("multi-page document")
            result = await self.pdf_converter.convert_pdf_to_text(pdf_content)

            # Should combine both pages
            assert "Page 1 content" in result
            assert "Page 2 content" in result
            assert (
                len(result.split("\n\n")) == 2
            )  # Two pages separated by double newline

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test empty quality info
        empty_quality = self.pdf_converter.get_text_quality_info("")
        assert empty_quality["char_count"] == 0
        assert empty_quality["word_count"] == 0
        assert empty_quality["quality_score"] == 0.0

        # Test None input
        none_quality = self.pdf_converter.get_text_quality_info(None)
        assert none_quality["char_count"] == 0

        # Test content validator with edge cases
        assert not self.content_validator.is_html_content_usable("")
        assert not self.content_validator.is_html_content_usable(None)
        assert not self.content_validator.is_pdf_text_usable("")
        assert not self.content_validator.is_pdf_text_usable(None)

        # Test unknown source
        assert not self.content_validator.validate_content_by_source(
            "content", "unknown_source"
        )

    @pytest.mark.asyncio
    async def test_performance_acceptable(self):
        """Test that PDF conversion completes within reasonable time."""
        import time

        with patch("pdfplumber.open") as mock_pdfplumber_open:
            # Mock reasonable response time
            mock_page = Mock()
            mock_page.extract_text.return_value = "Performance test content " * 50

            mock_pdf = Mock()
            mock_pdf.pages = [mock_page]
            mock_pdf.__enter__ = Mock(return_value=mock_pdf)
            mock_pdf.__exit__ = Mock(return_value=False)

            mock_pdfplumber_open.return_value = mock_pdf

            pdf_content = self.create_mock_pdf_content("performance test")

            start_time = time.time()
            result = await self.pdf_converter.convert_pdf_to_text(pdf_content)
            end_time = time.time()

            # Should complete within reasonable time (10 seconds as per interim goal)
            processing_time = end_time - start_time
            assert processing_time < 10.0  # Interim goal: <10 seconds

            # Should produce usable result
            assert len(result) > 50  # Meets minimum threshold
