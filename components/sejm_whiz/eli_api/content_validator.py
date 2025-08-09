"""Basic content validation for ELI API document processing."""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class BasicContentValidator:
    """Simple content validation for interim goal."""

    MINIMUM_HTML_CHARS = 100
    MINIMUM_PDF_TEXT_CHARS = 50

    def __init__(
        self, min_html_chars: Optional[int] = None, min_pdf_chars: Optional[int] = None
    ):
        """Initialize content validator with custom thresholds.

        Args:
            min_html_chars: Minimum characters for HTML content to be considered usable
            min_pdf_chars: Minimum characters for PDF text to be considered usable
        """
        self.min_html_chars = min_html_chars or self.MINIMUM_HTML_CHARS
        self.min_pdf_chars = min_pdf_chars or self.MINIMUM_PDF_TEXT_CHARS

        logger.debug(
            f"ContentValidator initialized: HTML≥{self.min_html_chars}, PDF≥{self.min_pdf_chars}"
        )

    def is_html_content_usable(self, content: str) -> bool:
        """Check if HTML content meets minimum usability requirements.

        Args:
            content: HTML content to validate

        Returns:
            True if HTML content is usable
        """
        if not content or not isinstance(content, str):
            return False

        cleaned_content = content.strip()
        char_count = len(cleaned_content)

        # Basic HTML content checks
        is_long_enough = char_count >= self.min_html_chars
        has_content = bool(cleaned_content)

        # Additional HTML-specific checks
        has_html_tags = "<" in content and ">" in content
        is_not_error_page = not self._looks_like_error_page(content)

        result = is_long_enough and has_content and is_not_error_page

        logger.debug(
            f"HTML validation: {char_count} chars, tags={has_html_tags}, "
            f"not_error={is_not_error_page}, usable={result}"
        )

        return result

    def is_pdf_text_usable(self, text: str) -> bool:
        """Check if PDF-extracted text meets minimum usability requirements.

        Args:
            text: PDF-extracted text to validate

        Returns:
            True if PDF text is usable
        """
        if not text or not isinstance(text, str):
            return False

        cleaned_text = text.strip()
        char_count = len(cleaned_text)

        # Basic PDF text checks
        is_long_enough = char_count >= self.min_pdf_chars
        has_content = bool(cleaned_text)

        # Additional text quality checks
        has_readable_content = self._has_readable_content(text)
        not_mostly_gibberish = self._not_mostly_gibberish(text)

        result = (
            is_long_enough
            and has_content
            and has_readable_content
            and not_mostly_gibberish
        )

        logger.debug(
            f"PDF text validation: {char_count} chars, readable={has_readable_content}, "
            f"not_gibberish={not_mostly_gibberish}, usable={result}"
        )

        return result

    def get_content_source_priority(self) -> List[str]:
        """Return content source priority: PDF first, then HTML.

        Changed to PDF-first because:
        - 2025 documents are PDF-only (no HTML available)
        - Consistent extraction approach across all years
        - PDF is the authoritative source for legal documents

        Returns:
            List of content sources in priority order
        """
        return ["pdf", "html"]

    def validate_content_by_source(self, content: str, source: str) -> bool:
        """Validate content based on its source type.

        Args:
            content: Content to validate
            source: Source type ('html' or 'pdf')

        Returns:
            True if content is usable for its source type
        """
        if source == "html":
            return self.is_html_content_usable(content)
        elif source == "pdf":
            return self.is_pdf_text_usable(content)
        else:
            logger.warning(f"Unknown content source: {source}")
            return False

    def get_content_quality_score(self, content: str, source: str) -> float:
        """Get basic quality score for content (0.0 to 1.0).

        NOTE: This scoring function is NOT FINISHED and subject to change when we get
        actual content to work with. Current implementation is preliminary.

        ASSUMPTION: ELI API returns correct HTTP status codes (not 200) for API errors.
        Error detection here focuses on content-level issues, not HTTP-level errors.

        Args:
            content: Content to score
            source: Source type ('html' or 'pdf')

        Returns:
            Quality score between 0.0 and 1.0
        """
        if not content or not isinstance(content, str):
            return 0.0

        char_count = len(content.strip())

        # Base score from character count
        if source == "html":
            base_score = min(1.0, char_count / 2000.0)  # 2000 chars = 1.0 for HTML
        else:  # PDF
            base_score = min(1.0, char_count / 1000.0)  # 1000 chars = 1.0 for PDF

        # Apply quality modifiers
        if self._has_readable_content(content):
            base_score *= 1.1

        if not self._not_mostly_gibberish(content):
            base_score *= 0.5

        if source == "html" and self._looks_like_error_page(content):
            base_score *= 0.1

        return min(1.0, base_score)

    def _looks_like_error_page(self, content: str) -> bool:
        """Check if HTML content looks like an error page.

        Args:
            content: HTML content to check

        Returns:
            True if content appears to be an error page
        """
        error_indicators = [
            "404",
            "not found",
            "error",
            "błąd",
            "niedostępny",
            "access denied",
            "forbidden",
            "unauthorized",
        ]

        content_lower = content.lower()
        return any(indicator in content_lower for indicator in error_indicators)

    def _has_readable_content(self, text: str) -> bool:
        """Check if text contains readable content (not just whitespace/symbols).

        Args:
            text: Text to check

        Returns:
            True if text has readable content
        """
        # Count alphabetic characters
        alpha_chars = sum(1 for c in text if c.isalpha())
        total_chars = len(text.strip())

        if total_chars == 0:
            return False

        # At least 10% of characters should be alphabetic (lowered for legal documents with lots of numbers/formatting)
        alpha_ratio = alpha_chars / total_chars
        return alpha_ratio >= 0.10

    def _not_mostly_gibberish(self, text: str) -> bool:
        """Check if text is not mostly gibberish/random characters.

        Args:
            text: Text to check

        Returns:
            True if text is not mostly gibberish
        """
        words = text.split()
        if not words:
            return False

        # Check for reasonable word lengths
        reasonable_words = [w for w in words if 2 <= len(w) <= 50]
        reasonable_ratio = len(reasonable_words) / len(words)

        # At least 50% of words should be reasonable length
        return reasonable_ratio >= 0.5
