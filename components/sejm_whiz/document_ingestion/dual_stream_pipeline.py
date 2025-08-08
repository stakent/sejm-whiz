"""Dual-API document processing pipeline for unified Sejm and ELI API integration."""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DocumentResult:
    """Result of document processing from any API source."""

    document_id: str
    act_text: str
    metadata: Dict[str, Any]
    source_used: str
    success: bool
    processing_time: float
    error_message: Optional[str] = None
    content_quality_score: Optional[float] = None


@dataclass
class ActDocument:
    """Standardized legal act document representation."""

    document_id: str
    title: str
    text_content: str
    metadata: Dict[str, Any]
    source_api: str
    extraction_timestamp: datetime
    quality_score: float


class DualApiDocumentProcessor:
    """Process documents from both Sejm API and ELI API."""

    def __init__(self, sejm_client=None, eli_client=None, content_validator=None):
        """Initialize dual-API document processor.

        Args:
            sejm_client: Sejm API client instance
            eli_client: ELI API client instance
            content_validator: Content validation instance
        """
        self.sejm_client = sejm_client
        self.eli_client = eli_client
        self.content_validator = content_validator

        logger.info("DualApiDocumentProcessor initialized")

    async def process_eli_document(self, document_id: str) -> DocumentResult:
        """Process enacted/historical law document from ELI API.

        ELI API contains documents that represent law in effect at present or in the past.
        These are finalized legal texts with binding legal effect.

        Args:
            document_id: ELI document identifier to process

        Returns:
            DocumentResult with processing outcome from ELI API
        """
        start_time = datetime.now()

        result = DocumentResult(
            document_id=document_id,
            act_text="",
            metadata={},
            source_used="eli",
            success=False,
            processing_time=0.0,
        )

        try:
            # Process from ELI API with dual content storage
            if self.eli_client:
                eli_result = await self._try_eli_api_with_dual_storage(document_id)
                if eli_result.success:
                    result = eli_result
                else:
                    # Copy error message from failed attempt
                    result.error_message = eli_result.error_message

        except Exception as e:
            logger.error(f"ELI document processing failed for {document_id}: {e}")
            result.error_message = str(e)

        finally:
            result.processing_time = (datetime.now() - start_time).total_seconds()

        return result

    async def process_sejm_document(self, document_id: str) -> DocumentResult:
        """Process work-in-progress legislative document from Sejm API.

        Sejm API contains documents that represent law still in development/legislative process.
        These are proposed legal texts that may become binding law in the future.

        Args:
            document_id: Sejm document identifier to process

        Returns:
            DocumentResult with processing outcome from Sejm API
        """
        start_time = datetime.now()

        result = DocumentResult(
            document_id=document_id,
            act_text="",
            metadata={},
            source_used="sejm",
            success=False,
            processing_time=0.0,
        )

        try:
            # Process from Sejm API
            if self.sejm_client:
                sejm_result = await self._try_sejm_api(document_id)
                if sejm_result.success:
                    result = sejm_result
                else:
                    # Copy error message from failed attempt
                    result.error_message = sejm_result.error_message

        except Exception as e:
            logger.error(f"Sejm document processing failed for {document_id}: {e}")
            result.error_message = str(e)

        finally:
            result.processing_time = (datetime.now() - start_time).total_seconds()

        return result

    async def _try_sejm_api(self, document_id: str) -> DocumentResult:
        """Try to get document from Sejm API.

        Args:
            document_id: Document ID to fetch

        Returns:
            DocumentResult from Sejm API processing
        """
        result = DocumentResult(
            document_id=document_id,
            act_text="",
            metadata={},
            source_used="sejm_api",
            success=False,
            processing_time=0.0,
        )

        try:
            # Parse document_id into term and number (temporary for compatibility)
            # TODO: Update callers to pass term and number directly
            if "_" in document_id:
                term_str, number_str = document_id.split("_", 1)
                term = int(term_str)
                number = int(number_str)
            else:
                # Fallback defaults
                term = 10
                number = int(document_id) if document_id.isdigit() else 1

            # Get act with full text from Sejm API using term and number
            sejm_data = await self.sejm_client.get_act_with_full_text(term, number)

            if self.sejm_client.is_sejm_content_complete(sejm_data):
                # Extract text and metadata
                act_text = sejm_data.get("text", "")
                metadata = await self.sejm_client.extract_act_metadata(sejm_data)

                # Validate content quality using ContentValidator (same logic as ELI HTML)
                # NOTE: This scoring is NOT FINISHED and subject to change when we get
                # actual content to work with. Using ContentValidator for consistency.
                quality_score = 0.5  # Default moderate quality for Sejm API
                if self.content_validator:
                    # Use same sophisticated scoring as ELI content
                    quality_score = self.content_validator.get_content_quality_score(
                        act_text,
                        "html",  # Treat Sejm API text as HTML-like content
                    )

                result.act_text = act_text
                result.metadata = metadata
                result.content_quality_score = quality_score
                result.success = True

                logger.debug(
                    f"Sejm API success: {len(act_text)} chars, quality={quality_score}"
                )
            else:
                result.error_message = "Sejm API content incomplete"

        except Exception as e:
            result.error_message = f"Sejm API error: {str(e)}"
            logger.error(f"Document {document_id} processing failed: {e}")

        return result

    async def _try_eli_api_with_dual_storage(self, document_id: str) -> DocumentResult:
        """Try to get document from ELI API with dual HTML+PDF storage.

        Fetches and stores BOTH HTML and PDF content when available for quality validation.

        Args:
            document_id: ELI document ID to fetch

        Returns:
            DocumentResult from ELI API dual content processing
        """
        result = DocumentResult(
            document_id=document_id,
            act_text="",
            metadata={},
            source_used="eli_api",
            success=False,
            processing_time=0.0,
        )

        try:
            # Try ELI API with dual content storage
            eli_data = await self.eli_client.get_document_content_with_dual_storage(
                document_id
            )

            if eli_data.get("usable", False):
                # Use preferred content (best quality between HTML and PDF)
                preferred_content = eli_data.get("preferred_content", "")
                preferred_source = eli_data.get("preferred_source", "unknown")

                # Calculate quality score based on content validator
                # NOTE: This scoring is NOT FINISHED and subject to change when we get
                # actual content to work with. Current implementation is preliminary.
                quality_score = 0.5  # Default moderate quality
                if self.content_validator:
                    quality_score = self.content_validator.get_content_quality_score(
                        preferred_content, preferred_source
                    )

                result.act_text = preferred_content
                result.metadata = {
                    "eli_id": document_id,
                    "preferred_source": preferred_source,
                    "processing_method": "eli_api_dual_storage",
                    "html_available": bool(eli_data.get("html_content")),
                    "pdf_available": bool(eli_data.get("pdf_content")),
                    "html_quality_score": eli_data.get("html_quality_score", 0.0),
                    "pdf_quality_score": eli_data.get("pdf_quality_score", 0.0),
                    "conversion_accuracy": eli_data.get("conversion_accuracy"),
                    "dual_content_analysis": {
                        "html_length": len(eli_data.get("html_content") or ""),
                        "pdf_length": len(eli_data.get("pdf_content") or ""),
                        "quality_comparison": {
                            "html": eli_data.get("html_quality_score", 0.0),
                            "pdf": eli_data.get("pdf_quality_score", 0.0),
                        },
                    },
                }
                result.content_quality_score = quality_score
                result.source_used = f"eli_api_{preferred_source}"
                result.success = True

                logger.debug(
                    f"ELI API dual storage success: {len(preferred_content)} chars from {preferred_source}, quality={quality_score}"
                )
            else:
                result.error_message = (
                    "ELI API returned no usable content from either HTML or PDF"
                )

        except Exception as e:
            result.error_message = f"ELI API dual storage error: {str(e)}"
            logger.error(f"ELI API dual storage error for {document_id}: {e}")

        return result

    async def extract_act_text_and_metadata(
        self, source: str, content: str
    ) -> ActDocument:
        """Extract standardized act document from raw content.

        Args:
            source: Source identifier (API name)
            content: Raw content to process

        Returns:
            Standardized ActDocument representation
        """
        try:
            # Basic text extraction (simplified for interim goal)
            text_content = content.strip()

            # Extract basic metadata from content
            metadata = {
                "source_api": source,
                "content_length": len(text_content),
                "extraction_method": "basic_text_extraction",
                "processed_at": datetime.now().isoformat(),
            }

            # Simple title extraction (take first non-empty line)
            lines = [line.strip() for line in text_content.splitlines() if line.strip()]
            title = lines[0] if lines else "Untitled Document"

            # Basic quality scoring
            quality_score = 0.5
            if self.content_validator:
                quality_score = self.content_validator.get_content_quality_score(
                    content, source
                )

            act_document = ActDocument(
                document_id="",  # Will be set by caller
                title=title,
                text_content=text_content,
                metadata=metadata,
                source_api=source,
                extraction_timestamp=datetime.now(),
                quality_score=quality_score,
            )

            logger.debug(
                f"Extracted act document: {len(text_content)} chars, quality={quality_score}"
            )
            return act_document

        except Exception as e:
            logger.error(f"Act extraction failed: {e}")
            raise

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get basic processing statistics for monitoring.

        Returns:
            Dictionary with processing statistics
        """
        # This would track statistics in a real implementation
        # For interim goal, return basic placeholder
        return {
            "total_processed": 0,
            "sejm_api_successes": 0,
            "eli_api_html_successes": 0,
            "eli_api_pdf_successes": 0,
            "failures": 0,
            "average_processing_time": 0.0,
            "quality_score_average": 0.0,
        }

    def validate_configuration(self) -> List[str]:
        """Validate processor configuration and return any issues.

        Returns:
            List of configuration issues (empty if valid)
        """
        issues = []

        if not self.sejm_client and not self.eli_client:
            issues.append("No API clients configured")

        if not self.content_validator:
            issues.append("No content validator configured")

        return issues
