"""HTML and text cleaning utilities for Polish legal documents."""

import re
from bs4 import BeautifulSoup


class HTMLCleaner:
    """Cleans HTML content from legal documents."""

    def __init__(self):
        # Common HTML patterns in legal documents
        self.html_patterns = [
            (r"<[^>]+>", ""),  # Remove all HTML tags
            (r"&nbsp;", " "),  # Non-breaking spaces
            (r"&amp;", "&"),  # HTML entities
            (r"&lt;", "<"),
            (r"&gt;", ">"),
            (r"&quot;", '"'),
            (r"&#\d+;", ""),  # Numeric HTML entities
        ]

        # Legal document specific cleaning patterns
        self.legal_patterns = [
            (r"\s*\n\s*\n\s*", "\n\n"),  # Multiple newlines
            (r"[ \t]+", " "),  # Multiple spaces/tabs
            (r"^\s+|\s+$", ""),  # Leading/trailing whitespace
        ]

    def clean_html(self, text: str) -> str:
        """Remove HTML tags and entities from text."""
        if not text:
            return ""

        # Use BeautifulSoup for robust HTML parsing
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()

        # Apply regex patterns for remaining HTML entities
        for pattern, replacement in self.html_patterns:
            text = re.sub(pattern, replacement, text)

        # Convert remaining non-breaking spaces to regular spaces
        text = text.replace("\xa0", " ")

        return text

    def clean_formatting(self, text: str) -> str:
        """Clean formatting artifacts from legal text."""
        if not text:
            return ""

        for pattern, replacement in self.legal_patterns:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)

        return text.strip()

    def clean_document(self, text: str) -> str:
        """Complete cleaning pipeline for legal documents."""
        text = self.clean_html(text)
        text = self.clean_formatting(text)
        return text


class TextCleaner:
    """Advanced text cleaning for Polish legal documents."""

    def __init__(self):
        self.html_cleaner = HTMLCleaner()

        # Polish legal document specific patterns
        self.noise_patterns = [
            # Page numbers and references
            (r"^\s*strona\s+\d+\s*$", ""),
            (r"^\s*str\.\s+\d+\s*$", ""),
            # Headers and footers
            (r"^\s*DZIENNIK\s+USTAW.*$", ""),
            (r"^\s*Poz\.\s+\d+.*$", ""),
            # Excessive punctuation
            (r"\.{3,}", "..."),
            (r"-{3,}", "---"),
            # Multiple spaces after punctuation
            (r"([.!?])\s{2,}", r"\1 "),
        ]

    def remove_noise(self, text: str) -> str:
        """Remove common noise patterns from legal text."""
        if not text:
            return ""

        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            # Apply noise removal patterns
            for pattern, replacement in self.noise_patterns:
                line = re.sub(
                    pattern, replacement, line, flags=re.IGNORECASE | re.MULTILINE
                )

            # Keep non-empty lines
            if line.strip():
                cleaned_lines.append(line.strip())

        return "\n".join(cleaned_lines)

    def clean_text(self, text: str) -> str:
        """Complete text cleaning pipeline."""
        text = self.html_cleaner.clean_document(text)
        text = self.remove_noise(text)
        return text
