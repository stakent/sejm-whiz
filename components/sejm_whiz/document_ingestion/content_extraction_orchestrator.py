"""Content extraction orchestrator for multi-phase document processing."""

import asyncio
import logging
from datetime import datetime, UTC
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from sejm_whiz.eli_api.client import EliApiClient
from sejm_whiz.sejm_api.client import SejmApiClient
from sejm_whiz.eli_api.enhanced_content_validator import EnhancedContentValidator

logger = logging.getLogger(__name__)


@dataclass
class ExtractionAttempt:
    """Result of a single extraction attempt."""

    source_type: str
    success: bool
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time_ms: float = 0.0
    quality_score: float = 0.0


@dataclass
class DocumentExtractionResult:
    """Complete result of document extraction process."""

    document_id: str
    status: str  # "success", "manual_review_required"
    attempts_made: List[ExtractionAttempt] = field(default_factory=list)
    final_content: Optional[str] = None
    final_metadata: Optional[Dict[str, Any]] = None
    manual_review_required: bool = False
    manual_review_context: Optional[Dict[str, Any]] = None
    total_processing_time_ms: float = 0.0


class ContentExtractionOrchestrator:
    """Orchestrates multiple content extraction attempts with guaranteed outcome."""

    def __init__(self):
        self.primary_sources = {
            "eli_client": EliApiClient(),
            "sejm_client": SejmApiClient(),
        }
        self.content_validator = EnhancedContentValidator()

        # Quality thresholds for different content tiers
        self.quality_thresholds = {
            "high": 0.8,  # Complete text + full metadata
            "medium": 0.6,  # Partial text + basic metadata
            "low": 0.3,  # Summary + metadata only
            "summary": 0.1,  # Metadata-based summary
        }

    async def extract_document_content(
        self, document_id: str, document_url: str = ""
    ) -> DocumentExtractionResult:
        """Guaranteed content extraction or manual review flagging.

        Two-phase processing:
        Phase 1: Primary sources (ELI + Sejm APIs with metadata)
        Phase 2: Manual review flagging with rich context
        """
        start_time = asyncio.get_event_loop().time()

        result = DocumentExtractionResult(document_id=document_id, status="processing")

        try:
            logger.info(f"Starting content extraction orchestration for {document_id}")

            # Phase 1: Primary sources (current implementation)
            primary_result = await self._try_primary_sources(document_id, document_url)
            result.attempts_made.extend(primary_result.attempts_made)

            if primary_result.success:
                result.final_content = primary_result.final_content
                result.final_metadata = primary_result.final_metadata
                result.status = "success"
                logger.info(f"Primary source extraction successful for {document_id}")
                return result

            # Phase 2: Flag for manual review
            logger.info(
                f"All extraction methods failed, flagging for manual review: {document_id}"
            )
            result.status = "manual_review_required"
            result.manual_review_required = True
            result.manual_review_context = self._prepare_manual_review_context(result)

        except Exception as e:
            logger.error(
                f"Content extraction orchestration failed for {document_id}: {e}"
            )
            result.status = "manual_review_required"
            result.manual_review_required = True
            result.manual_review_context = {
                "failure_type": "orchestration_exception",
                "error_message": str(e),
                "requires_technical_review": True,
            }

        finally:
            result.total_processing_time_ms = (
                asyncio.get_event_loop().time() - start_time
            ) * 1000

        return result

    async def _try_primary_sources(
        self, document_id: str, document_url: str
    ) -> DocumentExtractionResult:
        """Try primary API sources (ELI and Sejm)."""
        result = DocumentExtractionResult(document_id=document_id, status="processing")

        # Try ELI API with fallback
        eli_attempt = await self._try_eli_api_source(document_id, document_url)
        result.attempts_made.append(eli_attempt)

        if (
            eli_attempt.success
            and eli_attempt.quality_score >= self.quality_thresholds["low"]
        ):
            result.final_content = eli_attempt.content
            result.final_metadata = eli_attempt.metadata
            result.status = "success"
            return result

        # Try Sejm API
        sejm_attempt = await self._try_sejm_api_source(document_id)
        result.attempts_made.append(sejm_attempt)

        if (
            sejm_attempt.success
            and sejm_attempt.quality_score >= self.quality_thresholds["low"]
        ):
            result.final_content = sejm_attempt.content
            result.final_metadata = sejm_attempt.metadata
            result.status = "success"
            return result

        result.status = "failed"
        return result

    async def _try_eli_api_source(
        self, document_id: str, document_url: str
    ) -> ExtractionAttempt:
        """Try ELI API with PDF fallback."""
        start_time = asyncio.get_event_loop().time()
        attempt = ExtractionAttempt(source_type="eli_api_with_fallback", success=False)

        try:
            eli_client = self.primary_sources["eli_client"]

            # Use enhanced ELI client with PDF fallback
            result = await eli_client.get_document_content_with_basic_fallback(
                document_id
            )

            if result and result.get("content"):
                content = result["content"]
                content_type = result.get("content_type", "unknown")
                metadata = result.get("metadata", {})

                attempt.success = True
                attempt.content = content
                attempt.metadata = {
                    "content_type": content_type,
                    "source": "eli_api",
                    **metadata,
                }

                # Calculate quality score
                attempt.quality_score = self._calculate_content_quality_score(
                    content, metadata, "eli_api"
                )

                logger.info(
                    f"ELI API extraction successful for {document_id} (quality: {attempt.quality_score:.3f})"
                )
            else:
                attempt.error_message = "ELI API returned no usable content"

        except Exception as e:
            logger.error(f"ELI API extraction failed for {document_id}: {e}")
            attempt.error_message = str(e)

        finally:
            attempt.processing_time_ms = (
                asyncio.get_event_loop().time() - start_time
            ) * 1000

        return attempt

    async def _try_sejm_api_source(self, document_id: str) -> ExtractionAttempt:
        """Try Sejm API for document content."""
        start_time = asyncio.get_event_loop().time()
        attempt = ExtractionAttempt(source_type="sejm_api", success=False)

        try:
            sejm_client = self.primary_sources["sejm_client"]

            # Try to get act data from Sejm API
            act_data = await sejm_client.get_act_with_full_text(document_id)

            if act_data and sejm_client.is_sejm_content_complete(act_data):
                content = act_data.get("text", "")
                metadata = sejm_client.extract_act_metadata(act_data)

                attempt.success = True
                attempt.content = content
                attempt.metadata = {"source": "sejm_api", **metadata}

                # Calculate quality score
                attempt.quality_score = self._calculate_content_quality_score(
                    content, metadata, "sejm_api"
                )

                logger.info(
                    f"Sejm API extraction successful for {document_id} (quality: {attempt.quality_score:.3f})"
                )
            else:
                attempt.error_message = "Sejm API returned incomplete content"

        except Exception as e:
            logger.error(f"Sejm API extraction failed for {document_id}: {e}")
            attempt.error_message = str(e)

        finally:
            attempt.processing_time_ms = (
                asyncio.get_event_loop().time() - start_time
            ) * 1000

        return attempt

    # REMOVED: Metadata summary generation
    # ELI API provides rich metadata (type, title, etc.) directly
    # We store the metadata received along with the document - no need to reconstruct

    def _calculate_content_quality_score(
        self, content: str, metadata: Dict[str, Any], source_type: str
    ) -> float:
        """Calculate quality score for content."""
        if not content:
            return 0.0

        score = 0.0
        content_length = len(content.strip())

        # Base score from content length
        if content_length >= 1000:
            score += 0.4
        elif content_length >= 500:
            score += 0.3
        elif content_length >= 200:
            score += 0.2
        elif content_length >= 50:
            score += 0.1

        # Bonus for source quality
        source_bonuses = {
            "eli_api": 0.3,
            "sejm_api": 0.25,
            "metadata_summary": 0.05,
        }

        score += source_bonuses.get(source_type, 0.0)

        # Bonus for metadata completeness
        if metadata:
            metadata_items = len([v for v in metadata.values() if v])
            if metadata_items >= 5:
                score += 0.2
            elif metadata_items >= 3:
                score += 0.15
            elif metadata_items >= 1:
                score += 0.1

        # Penalty for reconstruction flags
        if metadata and metadata.get("requires_manual_review"):
            score -= 0.1

        # Polish language content patterns (basic check)
        polish_chars = ["ą", "ć", "ę", "ł", "ń", "ó", "ś", "ź", "ż"]
        if any(char in content.lower() for char in polish_chars):
            score += 0.1

        # Legal document structure patterns
        legal_patterns = ["art.", "artykuł", "ustawa", "rozdział", "§", "ust."]
        pattern_matches = sum(
            1 for pattern in legal_patterns if pattern.lower() in content.lower()
        )
        if pattern_matches >= 3:
            score += 0.1
        elif pattern_matches >= 1:
            score += 0.05

        return min(score, 1.0)  # Cap at 1.0

    def _prepare_manual_review_context(
        self, result: DocumentExtractionResult
    ) -> Dict[str, Any]:
        """Prepare comprehensive context for manual review."""
        context = {
            "document_id": result.document_id,
            "flagged_at": datetime.now(UTC).isoformat(),
            "total_attempts": len(result.attempts_made),
            "processing_time_ms": result.total_processing_time_ms,
            "failure_summary": self._summarize_failures(result.attempts_made),
            "suggested_actions": self._suggest_manual_actions(result.attempts_made),
            "priority": self._determine_review_priority(result.document_id),
            "estimated_effort": self._estimate_manual_effort(result.attempts_made),
        }

        return context

    def _summarize_failures(self, attempts: List[ExtractionAttempt]) -> Dict[str, Any]:
        """Summarize why extraction attempts failed."""
        summary = {
            "total_attempts": len(attempts),
            "sources_tried": [attempt.source_type for attempt in attempts],
            "main_errors": [],
        }

        # Collect unique error messages
        errors = set()
        for attempt in attempts:
            if not attempt.success and attempt.error_message:
                errors.add(attempt.error_message)

        summary["main_errors"] = list(errors)
        summary["success_count"] = len([a for a in attempts if a.success])
        summary["quality_scores"] = [a.quality_score for a in attempts if a.success]

        return summary

    def _suggest_manual_actions(self, attempts: List[ExtractionAttempt]) -> List[str]:
        """Suggest actions for manual reviewers."""
        suggestions = []

        # Analyze attempt patterns
        sources_tried = [attempt.source_type for attempt in attempts]

        if "eli_api_with_fallback" in sources_tried:
            suggestions.append("Check if document is available directly on ELI website")

        if "alternative_wayback_machine" in sources_tried:
            suggestions.append(
                "Manual search in Wayback Machine with different date ranges"
            )

        if any("alternative_" in source for source in sources_tried):
            suggestions.append(
                "Search for document in additional legal databases (ISAP, Lex)"
            )

        if any("metadata" in source for source in sources_tried):
            suggestions.append(
                "Manual transcription may be required if document exists in scanned format"
            )

        # Generic suggestions
        suggestions.extend(
            [
                "Verify document ID format and existence",
                "Check if document has been superseded or amended",
                "Contact source institution for document availability",
            ]
        )

        return suggestions

    def _determine_review_priority(self, document_id: str) -> str:
        """Determine manual review priority based on document importance."""
        # Basic priority determination from ELI ID patterns
        try:
            parts = document_id.split("/")
            if len(parts) >= 3:
                doc_type = parts[0]
                year = int(parts[1])

                # High priority: Recent constitutional/legal changes
                if doc_type == "DU" and year >= 2023:
                    return "high"
                # Medium priority: Other recent documents
                elif year >= 2022:
                    return "medium"
                # Low priority: Historical documents
                else:
                    return "low"
        except (ValueError, IndexError):
            pass

        return "medium"  # Default priority

    def _estimate_manual_effort(self, attempts: List[ExtractionAttempt]) -> str:
        """Estimate manual effort required for review."""
        successful_attempts = [a for a in attempts if a.success]

        if successful_attempts:
            # Some content was extracted but quality was insufficient
            avg_quality = sum(a.quality_score for a in successful_attempts) / len(
                successful_attempts
            )
            if avg_quality >= 0.3:
                return "quick"  # Just needs quality improvement
            else:
                return "medium"  # Needs significant enhancement
        else:
            # No content was extracted
            return "complex"  # Full manual transcription needed
