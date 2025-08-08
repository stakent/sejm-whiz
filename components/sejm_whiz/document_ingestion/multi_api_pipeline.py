"""Multi-API document processing pipeline for unified Sejm and ELI API integration."""

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


class MultiApiDocumentProcessor:
    """Process documents from both Sejm API and ELI API."""

    def __init__(self, sejm_client=None, eli_client=None, content_validator=None):
        """Initialize multi-API document processor.

        Args:
            sejm_client: Sejm API client instance
            eli_client: ELI API client instance
            content_validator: Content validation instance
        """
        self.sejm_client = sejm_client
        self.eli_client = eli_client
        self.content_validator = content_validator

        logger.info("MultiApiDocumentProcessor initialized")

    async def process_document_from_any_source(
        self, document_id: str
    ) -> DocumentResult:
        """Try to get content from available APIs with source priority.

        Priority order:
        1. Sejm API (typically more reliable for full text)
        2. ELI API with HTML→PDF fallback

        Args:
            document_id: Document identifier to process

        Returns:
            DocumentResult with processing outcome
        """
        start_time = datetime.now()

        result = DocumentResult(
            document_id=document_id,
            act_text="",
            metadata={},
            source_used="none",
            success=False,
            processing_time=0.0,
        )

        try:
            # Try Sejm API first (typically more reliable for full text)
            if self.sejm_client:
                sejm_result = await self._try_sejm_api(document_id)
                if sejm_result.success:
                    result = sejm_result
                    logger.info(f"Successfully processed {document_id} via Sejm API")
                else:
                    logger.debug(
                        f"Sejm API failed for {document_id}: {sejm_result.error_message}"
                    )

            # Try ELI API with fallback if Sejm failed
            if not result.success and self.eli_client:
                eli_result = await self._try_eli_api_with_fallback(document_id)
                if eli_result.success:
                    result = eli_result
                    logger.info(f"Successfully processed {document_id} via ELI API")
                else:
                    logger.debug(
                        f"ELI API failed for {document_id}: {eli_result.error_message}"
                    )

            # Mark as failed if all sources failed
            if not result.success:
                result.error_message = "All API sources failed"
                logger.warning(f"All sources failed for document {document_id}")

        except Exception as e:
            result.error_message = f"Processing error: {str(e)}"
            logger.error(f"Unexpected error processing {document_id}: {e}")

        finally:
            # Calculate processing time
            end_time = datetime.now()
            result.processing_time = (end_time - start_time).total_seconds()

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
            # Get act with full text from Sejm API
            sejm_data = await self.sejm_client.get_act_with_full_text(document_id)

            if self.sejm_client.is_sejm_content_complete(sejm_data):
                # Extract text and metadata
                act_text = sejm_data.get("text", "")
                metadata = await self.sejm_client.extract_act_metadata(sejm_data)

                # Validate content quality if validator available
                quality_score = 1.0  # Default high quality for Sejm API
                if self.content_validator:
                    if (
                        len(act_text.strip()) >= 100
                    ):  # Reasonable threshold for Sejm content
                        quality_score = 0.9  # High quality
                    else:
                        quality_score = 0.3  # Lower quality but still usable

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
            logger.error(f"Sejm API error for {document_id}: {e}")

        return result

    async def _try_eli_api_with_fallback(self, document_id: str) -> DocumentResult:
        """Try to get document from ELI API with HTML→PDF fallback.

        Args:
            document_id: ELI document ID to fetch

        Returns:
            DocumentResult from ELI API processing
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
            # Try ELI API with basic fallback
            eli_data = await self.eli_client.get_document_content_with_basic_fallback(
                document_id
            )

            if eli_data.get("usable", False):
                content = eli_data.get("content", "")
                content_source = eli_data.get("source", "unknown")

                # Calculate quality score based on content validator
                quality_score = 0.5  # Default moderate quality
                if self.content_validator:
                    quality_score = self.content_validator.get_content_quality_score(
                        content, content_source
                    )

                result.act_text = content
                result.metadata = {
                    "eli_id": document_id,
                    "content_source": content_source,
                    "processing_method": f"eli_api_{content_source}",
                }
                result.content_quality_score = quality_score
                result.source_used = f"eli_api_{content_source}"
                result.success = True

                logger.debug(
                    f"ELI API success: {len(content)} chars from {content_source}, quality={quality_score}"
                )
            else:
                result.error_message = "ELI API returned no usable content"

        except Exception as e:
            result.error_message = f"ELI API error: {str(e)}"
            logger.error(f"ELI API error for {document_id}: {e}")

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
