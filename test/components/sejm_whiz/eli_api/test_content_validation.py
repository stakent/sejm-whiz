"""Content validation tests for interim milestone."""

from sejm_whiz.eli_api.content_validator import BasicContentValidator


class TestContentValidation:
    """Test content validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = BasicContentValidator()

    def test_validator_initialization(self):
        """Test validator initializes with correct defaults."""
        validator = BasicContentValidator()
        assert validator.min_html_chars == 100
        assert validator.min_pdf_chars == 50

        # Test custom thresholds
        custom_validator = BasicContentValidator(min_html_chars=200, min_pdf_chars=75)
        assert custom_validator.min_html_chars == 200
        assert custom_validator.min_pdf_chars == 75

    def test_html_content_validation(self):
        """Test HTML content validation logic."""
        # Good HTML content
        good_html = "<html><body>" + "A" * 150 + "</body></html>"
        assert self.validator.is_html_content_usable(good_html)

        # Too short HTML
        short_html = "<html><body>Short</body></html>"
        assert not self.validator.is_html_content_usable(short_html)

        # Empty content
        assert not self.validator.is_html_content_usable("")
        assert not self.validator.is_html_content_usable(None)

        # Error page content
        error_html = "<html><body>" + "A" * 150 + "404 Not Found</body></html>"
        assert not self.validator.is_html_content_usable(error_html)

        # Polish error page
        polish_error_html = (
            "<html><body>" + "A" * 150 + "Błąd - niedostępny</body></html>"
        )
        assert not self.validator.is_html_content_usable(polish_error_html)

    def test_pdf_text_validation(self):
        """Test PDF text validation logic."""
        # Good PDF text
        good_pdf = "This is a substantial PDF text content with enough characters for validation."
        assert self.validator.is_pdf_text_usable(good_pdf)

        # Too short PDF text
        short_pdf = "Short"
        assert not self.validator.is_pdf_text_usable(short_pdf)

        # Empty content
        assert not self.validator.is_pdf_text_usable("")
        assert not self.validator.is_pdf_text_usable(None)

        # Mostly gibberish content
        gibberish = "asdkfjh asldfkjh alskdfj laksjdf lkasjdf lkasjdf alskdjf"
        # This might still pass basic validation, but quality score should be low
        if self.validator.is_pdf_text_usable(gibberish):
            quality_score = self.validator.get_content_quality_score(gibberish, "pdf")
            assert quality_score < 0.8  # Should have low quality

    def test_content_source_priority(self):
        """Test content source priority ordering."""
        priority = self.validator.get_content_source_priority()
        assert priority == ["html", "pdf"]
        assert len(priority) == 2

    def test_validate_content_by_source(self):
        """Test source-specific validation."""
        html_content = "<html><body>" + "A" * 120 + "</body></html>"
        pdf_content = "Good PDF text content with sufficient length for validation."

        # Test HTML validation
        assert self.validator.validate_content_by_source(html_content, "html")
        assert not self.validator.validate_content_by_source("short", "html")

        # Test PDF validation
        assert self.validator.validate_content_by_source(pdf_content, "pdf")
        assert not self.validator.validate_content_by_source("short", "pdf")

        # Test unknown source
        assert not self.validator.validate_content_by_source(html_content, "unknown")

    def test_content_quality_scoring(self):
        """Test content quality scoring functionality."""
        # High quality HTML
        high_quality_html = (
            "<html><body>"
            + "High quality legal document content. " * 50
            + "</body></html>"
        )
        html_score = self.validator.get_content_quality_score(high_quality_html, "html")
        assert 0.8 <= html_score <= 1.0

        # High quality PDF
        high_quality_pdf = "High quality legal document text content. " * 30
        pdf_score = self.validator.get_content_quality_score(high_quality_pdf, "pdf")
        assert 0.8 <= pdf_score <= 1.0

        # Low quality content
        low_quality = "x" * 60  # Repetitive, low quality
        low_score = self.validator.get_content_quality_score(low_quality, "pdf")
        assert low_score < 0.8

        # Empty content
        empty_score = self.validator.get_content_quality_score("", "html")
        assert empty_score == 0.0

    def test_error_page_detection(self):
        """Test error page detection logic."""
        # Test public method indirectly through HTML validation
        error_indicators = [
            "404 Not Found",
            "Error occurred",
            "błąd systemu",
            "niedostępny dokument",
            "Access Denied",
            "Forbidden",
            "Unauthorized",
        ]

        for indicator in error_indicators:
            error_content = (
                "<html><body>" + "A" * 100 + indicator + "A" * 50 + "</body></html>"
            )
            assert not self.validator.is_html_content_usable(
                error_content
            ), f"Failed to detect error: {indicator}"

    def test_readable_content_detection(self):
        """Test readable content detection."""
        # Test through quality scoring since _has_readable_content is private
        readable_content = (
            "This is normal readable text with proper words and sentences."
        )
        readable_score = self.validator.get_content_quality_score(
            readable_content, "pdf"
        )

        symbols_only = "!@#$%^&*()_+{}[]|\\:;<>?,./"
        symbols_score = self.validator.get_content_quality_score(symbols_only, "pdf")

        # Readable content should score higher
        assert readable_score > symbols_score

    def test_gibberish_detection(self):
        """Test gibberish detection logic."""
        # Normal text should pass
        normal_text = "This document contains normal legal language and proper sentence structure."
        normal_score = self.validator.get_content_quality_score(normal_text, "pdf")

        # Gibberish should score lower
        gibberish_text = (
            "asdfkl jklasdf qwerty uiop zxcvbn mnbvcx qazwsx edcrfv tgbyhn ujmik olp"
        )
        gibberish_score = self.validator.get_content_quality_score(
            gibberish_text, "pdf"
        )

        assert normal_score > gibberish_score

    def test_polish_content_handling(self):
        """Test handling of Polish language content."""
        polish_content = """
        Ustawa z dnia 15 stycznia 2025 roku o przepisach prawnych zawierających
        polskie znaki diakrytyczne: ąćęłńóśźż. Artykuł 1. Przepisy niniejszej
        ustawy mają zastosowanie do wszystkich dokumentów prawnych publikowanych
        w Dzienniku Ustaw Rzeczypospolitej Polskiej.
        """

        # Should validate successfully
        assert self.validator.is_pdf_text_usable(polish_content)

        # Should get reasonable quality score (286 chars = 0.286 base + readable bonus = ~0.31)
        quality_score = self.validator.get_content_quality_score(polish_content, "pdf")
        assert quality_score > 0.3  # Should be reasonable quality for length
        assert quality_score < 0.4  # Not too high for short content

        # Test with longer Polish content for higher quality score
        long_polish_content = polish_content * 4  # ~1144 characters
        long_quality_score = self.validator.get_content_quality_score(
            long_polish_content, "pdf"
        )
        assert long_quality_score > 0.7  # Should be high quality for longer content

        # HTML version
        polish_html = f"<html><body>{polish_content}</body></html>"
        assert self.validator.is_html_content_usable(polish_html)

    def test_edge_cases_and_boundaries(self):
        """Test edge cases and boundary conditions."""
        # Test exactly at minimum thresholds
        exact_html = (
            "<html><body>"
            + "A" * (100 - len("<html><body></body></html>"))
            + "</body></html>"
        )
        assert self.validator.is_html_content_usable(exact_html)

        exact_pdf = "A" * 50
        assert self.validator.is_pdf_text_usable(exact_pdf)

        # Test just below thresholds
        below_html = (
            "<html><body>"
            + "A" * (99 - len("<html><body></body></html>"))
            + "</body></html>"
        )
        assert not self.validator.is_html_content_usable(below_html)

        below_pdf = "A" * 49
        assert not self.validator.is_pdf_text_usable(below_pdf)

    def test_non_string_inputs(self):
        """Test handling of non-string inputs."""
        # Test various non-string types
        invalid_inputs = [None, 123, [], {}, True, False]

        for invalid_input in invalid_inputs:
            assert not self.validator.is_html_content_usable(invalid_input)
            assert not self.validator.is_pdf_text_usable(invalid_input)
            assert (
                self.validator.get_content_quality_score(invalid_input, "html") == 0.0
            )

    def test_whitespace_handling(self):
        """Test proper whitespace handling."""
        # Content with lots of whitespace should still validate if core content is sufficient
        whitespace_html = (
            "<html><body>\n\n\n    " + "A" * 120 + "    \n\n\n</body></html>"
        )
        assert self.validator.is_html_content_usable(whitespace_html)

        whitespace_pdf = (
            "\n\n\n    "
            + "Good content here with enough text to pass validation."
            + "    \n\n\n"
        )
        assert self.validator.is_pdf_text_usable(whitespace_pdf)

        # But pure whitespace should fail
        pure_whitespace = "\n\n\n\t\t\t      \r\r\r"
        assert not self.validator.is_pdf_text_usable(pure_whitespace)

    def test_html_tags_requirement(self):
        """Test that HTML content must actually contain HTML tags."""
        # Plain text without HTML tags should be less favorable
        plain_text = "This is just plain text without any HTML tags but with sufficient length for validation."

        # While it might still pass basic length validation, it should have lower quality
        html_score = self.validator.get_content_quality_score(plain_text, "html")

        # Same content with HTML tags should score better or equal
        html_content = f"<html><body><p>{plain_text}</p></body></html>"
        tagged_score = self.validator.get_content_quality_score(html_content, "html")

        # HTML with proper tags should score at least as well, typically better
        assert tagged_score >= html_score
