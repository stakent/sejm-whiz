"""Basic PDF converter for ELI API document fallback processing."""

import asyncio
import logging
import pdfplumber

logger = logging.getLogger(__name__)


class BasicPDFConverter:
    """Simple PDF-to-text converter for interim milestone."""

    def __init__(self, engine: str = "pdfplumber"):
        """Initialize PDF converter with specified engine.

        Args:
            engine: PDF processing engine ('pdfplumber' only for now)
        """
        self.engine = engine
        if engine != "pdfplumber":
            raise ValueError(
                f"Unsupported engine: {engine}. Only 'pdfplumber' supported."
            )

    async def convert_pdf_to_text(self, pdf_content: bytes) -> str:
        """Convert PDF content to text using pdfplumber.

        Args:
            pdf_content: Raw PDF bytes

        Returns:
            Extracted text content

        Raises:
            Exception: If PDF conversion fails
        """
        try:
            # Run PDF processing in thread pool to avoid blocking
            text = await asyncio.get_event_loop().run_in_executor(
                None, self._extract_text_sync, pdf_content
            )
            return text
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise

    def _extract_text_sync(self, pdf_content: bytes) -> str:
        """Synchronous PDF text extraction using pdfplumber.

        Args:
            pdf_content: Raw PDF bytes

        Returns:
            Extracted text content
        """
        text_parts = []

        try:
            import io

            pdf_file = io.BytesIO(pdf_content)

            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                            logger.debug(
                                f"Extracted {len(page_text)} chars from page {page_num + 1}"
                            )
                        else:
                            logger.warning(f"No text found on page {page_num + 1}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract text from page {page_num + 1}: {e}"
                        )
                        continue

            full_text = "\n\n".join(text_parts)
            logger.info(
                f"PDF conversion completed: {len(full_text)} total characters from {len(text_parts)} pages"
            )

            return full_text

        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            raise

    def is_conversion_acceptable(self, text: str, min_chars: int = 50) -> bool:
        """Check if PDF conversion result meets minimum quality threshold.

        Args:
            text: Extracted text to validate
            min_chars: Minimum character count threshold

        Returns:
            True if text meets minimum quality requirements
        """
        if not text or not isinstance(text, str):
            return False

        cleaned_text = text.strip()
        char_count = len(cleaned_text)

        # Basic quality checks
        meets_length = char_count >= min_chars
        has_content = bool(cleaned_text)

        logger.debug(
            f"Conversion quality check: {char_count} chars, meets_length={meets_length}"
        )

        return meets_length and has_content

    def get_text_quality_info(self, text: str) -> dict:
        """Get basic quality metrics for extracted text.

        Args:
            text: Extracted text to analyze

        Returns:
            Dictionary with quality metrics
        """
        if not text:
            return {
                "char_count": 0,
                "word_count": 0,
                "line_count": 0,
                "has_polish_chars": False,
                "quality_score": 0.0,
            }

        # Basic text metrics
        char_count = len(text.strip())
        words = text.split()
        word_count = len(words)
        line_count = len(text.splitlines())

        # Check for Polish characters (basic test)
        polish_chars = set("ąćęłńóśźż")
        has_polish_chars = bool(polish_chars.intersection(text.lower()))

        # Simple quality score (0.0 to 1.0)
        quality_score = min(1.0, char_count / 1000.0)  # Normalize to 1000 chars = 1.0

        return {
            "char_count": char_count,
            "word_count": word_count,
            "line_count": line_count,
            "has_polish_chars": has_polish_chars,
            "quality_score": quality_score,
        }
