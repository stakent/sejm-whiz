"""Enhanced content validation with multi-tier quality assessment."""

import re
import logging
from typing import Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ContentQualityAssessment:
    """Result of content quality assessment."""

    tier: str = "insufficient"  # "high", "medium", "low", "summary", "insufficient"
    usable: bool = False
    quality_score: float = 0.0
    character_count: int = 0
    sentence_count: int = 0
    metadata_completeness: float = 0.0
    language_confidence: float = 0.0
    legal_structure_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    validation_details: Dict[str, Any] = field(default_factory=dict)


class EnhancedContentValidator:
    """Enhanced validation with multiple quality tiers and comprehensive assessment."""

    # Quality tier requirements
    QUALITY_TIERS = {
        "high": {
            "min_chars": 500,
            "min_sentences": 10,
            "min_legal_patterns": 3,
            "metadata_complete": True,
            "min_quality_score": 0.7,
        },
        "medium": {
            "min_chars": 200,
            "min_sentences": 5,
            "min_legal_patterns": 2,
            "metadata_partial": True,
            "min_quality_score": 0.5,
        },
        "low": {
            "min_chars": 50,
            "min_sentences": 2,
            "min_legal_patterns": 1,
            "metadata_minimal": True,
            "min_quality_score": 0.3,
        },
        "summary": {
            "min_chars": 20,
            "min_sentences": 1,
            "metadata_only": True,
            "min_quality_score": 0.1,
        },
    }

    def __init__(self):
        # Polish legal document patterns
        self.legal_patterns = {
            "article_markers": [r"\bart\.\s*\d+", r"artykuł\s+\d+", r"art\s+\d+"],
            "section_markers": [r"§\s*\d+", r"paragraf\s+\d+", r"ust\.\s*\d+"],
            "chapter_markers": [r"rozdział\s+[IVX\d]+", r"część\s+[IVX\d]+"],
            "legal_terms": [
                "ustawa",
                "rozporządzenie",
                "konstytucja",
                "kodeks",
                "przepis",
                "prawo",
                "dziennik ustaw",
                "monitor polski",
            ],
            "structure_words": [
                "określa",
                "stanowi",
                "wprowadza",
                "nowelizuje",
                "uchyla",
                "wchodzi w życie",
                "traci moc",
            ],
        }

        # Polish character patterns for language detection
        self.polish_chars = ["ą", "ć", "ę", "ł", "ń", "ó", "ś", "ź", "ż"]

        # Error indicators that suggest low-quality content
        self.error_indicators = [
            "404 not found",
            "page not found",
            "błąd",
            "error",
            "nie znaleziono",
            "strona niedostępna",
            "access denied",
            "unauthorized",
        ]

    def assess_content_quality(
        self, content: str, metadata: Dict[str, Any], source_type: str = "unknown"
    ) -> ContentQualityAssessment:
        """Perform comprehensive multi-tier quality assessment."""
        assessment = ContentQualityAssessment()

        try:
            # Basic content metrics
            assessment.character_count = len(content.strip())
            assessment.sentence_count = self._count_sentences(content)

            # Calculate component scores
            length_score = self._assess_content_length(content)
            language_score = self._assess_polish_language_patterns(content)
            structure_score = self._assess_legal_document_structure(content)
            metadata_score = self._assess_metadata_completeness(metadata)
            error_penalty = self._assess_error_content(content)

            # Combine scores with weights
            assessment.quality_score = (
                length_score * 0.25
                + language_score * 0.25
                + structure_score * 0.25
                + metadata_score * 0.15
                + self._get_source_bonus(source_type) * 0.1
            ) - error_penalty

            assessment.quality_score = max(0.0, min(1.0, assessment.quality_score))

            # Store component scores
            assessment.language_confidence = language_score
            assessment.legal_structure_score = structure_score
            assessment.metadata_completeness = metadata_score

            # Determine quality tier
            assessment.tier = self._determine_quality_tier(
                assessment, content, metadata
            )
            assessment.usable = assessment.tier != "insufficient"

            # Generate recommendations
            assessment.recommendations = self._generate_improvement_recommendations(
                assessment, content, metadata
            )

            # Store validation details
            assessment.validation_details = {
                "length_score": length_score,
                "language_score": language_score,
                "structure_score": structure_score,
                "metadata_score": metadata_score,
                "source_bonus": self._get_source_bonus(source_type),
                "error_penalty": error_penalty,
                "legal_patterns_found": self._count_legal_patterns(content),
                "polish_char_ratio": self._calculate_polish_char_ratio(content),
                "source_type": source_type,
            }

            logger.debug(
                f"Content quality assessment: tier={assessment.tier}, "
                f"score={assessment.quality_score:.3f}, usable={assessment.usable}"
            )

        except Exception as e:
            logger.error(f"Content quality assessment failed: {e}")
            assessment.tier = "insufficient"
            assessment.usable = False
            assessment.recommendations = ["Technical error in quality assessment"]

        return assessment

    def _count_sentences(self, content: str) -> int:
        """Count sentences in content using Polish sentence patterns."""
        # Split on sentence endings, but be careful with abbreviations
        sentence_endings = r"[.!?]+(?:\s|$)"

        # Remove common abbreviations to avoid false sentence breaks
        cleaned_content = re.sub(
            r"\b(?:art|ust|pkt|lit|nr|ul|al|tel|fax)\.\s*",
            "",
            content,
            flags=re.IGNORECASE,
        )

        sentences = re.split(sentence_endings, cleaned_content)
        # Filter out very short "sentences" (likely artifacts)
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        return len(meaningful_sentences)

    def _assess_content_length(self, content: str) -> float:
        """Assess content quality based on length."""
        length = len(content.strip())

        if length >= 2000:
            return 1.0
        elif length >= 1000:
            return 0.8
        elif length >= 500:
            return 0.6
        elif length >= 200:
            return 0.4
        elif length >= 50:
            return 0.2
        else:
            return 0.0

    def _assess_polish_language_patterns(self, content: str) -> float:
        """Assess Polish language patterns and character usage."""
        if not content:
            return 0.0

        content_lower = content.lower()

        # Count Polish characters
        polish_char_count = sum(
            1 for char in content_lower if char in self.polish_chars
        )
        polish_char_ratio = polish_char_count / len(content) if content else 0

        # Count Polish legal terms
        legal_term_count = sum(
            1 for term in self.legal_patterns["legal_terms"] if term in content_lower
        )

        # Combine scores
        char_score = min(
            1.0, polish_char_ratio * 50
        )  # Polish chars should be ~2% of text
        term_score = min(1.0, legal_term_count * 0.2)  # More legal terms = better

        return (char_score + term_score) / 2

    def _assess_legal_document_structure(self, content: str) -> float:
        """Assess legal document structure patterns."""
        if not content:
            return 0.0

        structure_score = 0.0
        content_lower = content.lower()

        # Check for article markers
        article_patterns = self.legal_patterns["article_markers"]
        article_matches = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in article_patterns
        )
        if article_matches > 0:
            structure_score += 0.3

        # Check for section markers
        section_patterns = self.legal_patterns["section_markers"]
        section_matches = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in section_patterns
        )
        if section_matches > 0:
            structure_score += 0.2

        # Check for chapter markers
        chapter_patterns = self.legal_patterns["chapter_markers"]
        chapter_matches = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in chapter_patterns
        )
        if chapter_matches > 0:
            structure_score += 0.2

        # Check for legal structure words
        structure_words = self.legal_patterns["structure_words"]
        structure_word_count = sum(
            1 for word in structure_words if word in content_lower
        )
        if structure_word_count >= 2:
            structure_score += 0.3
        elif structure_word_count >= 1:
            structure_score += 0.15

        return min(1.0, structure_score)

    def _count_legal_patterns(self, content: str) -> int:
        """Count legal patterns found in content."""
        pattern_count = 0

        for pattern_group in self.legal_patterns.values():
            if isinstance(pattern_group, list):
                for pattern in pattern_group:
                    if isinstance(pattern, str):
                        # Direct string search
                        if pattern in content.lower():
                            pattern_count += 1
                    else:
                        # Regex pattern
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        pattern_count += len(matches)

        return pattern_count

    def _calculate_polish_char_ratio(self, content: str) -> float:
        """Calculate ratio of Polish characters in content."""
        if not content:
            return 0.0

        polish_count = sum(1 for char in content.lower() if char in self.polish_chars)
        return polish_count / len(content)

    def _assess_metadata_completeness(self, metadata: Dict[str, Any]) -> float:
        """Assess metadata completeness and quality."""
        if not metadata:
            return 0.0

        # Key metadata fields for legal documents
        important_fields = ["title", "document_type", "source", "eli_id", "date", "url"]

        present_fields = 0
        for field_name in important_fields:
            value = metadata.get(field_name)
            if value and str(value).strip():
                present_fields += 1

        completeness = present_fields / len(important_fields)

        # Bonus for additional useful metadata
        bonus_fields = [
            "content_type",
            "processing_method",
            "quality_indicators",
            "language",
            "keywords",
        ]

        bonus_present = sum(
            1
            for field in bonus_fields
            if metadata.get(field) and str(metadata.get(field)).strip()
        )

        bonus_score = min(0.2, bonus_present * 0.05)

        return min(1.0, completeness + bonus_score)

    def _assess_error_content(self, content: str) -> float:
        """Check for error indicators that suggest low-quality content."""
        if not content:
            return 0.0

        content_lower = content.lower()
        penalty = 0.0

        for error_indicator in self.error_indicators:
            if error_indicator in content_lower:
                penalty += 0.3  # Heavy penalty for error content

        # Additional heuristics for poor content
        if len(content.strip()) < 20:
            penalty += 0.2

        # Check for excessive repetition (might indicate corrupted content)
        words = content.lower().split()
        if len(words) > 10:
            unique_words = set(words)
            repetition_ratio = len(words) / len(unique_words)
            if repetition_ratio > 3:  # Same words repeated too often
                penalty += 0.1

        return min(1.0, penalty)

    def _get_source_bonus(self, source_type: str) -> float:
        """Get quality bonus based on source reliability."""
        source_bonuses = {
            "eli_api": 0.3,
            "sejm_api": 0.25,
            "historical_cache": 0.2,
            "wayback_machine": 0.15,
            "document_registry": 0.1,
            "metadata_reconstruction": 0.05,
            "unknown": 0.0,
        }

        return source_bonuses.get(source_type, 0.0)

    def _determine_quality_tier(
        self,
        assessment: ContentQualityAssessment,
        content: str,
        metadata: Dict[str, Any],
    ) -> str:
        """Determine quality tier based on assessment results."""

        # Check each tier in order of quality (high to low)
        for tier_name, requirements in self.QUALITY_TIERS.items():
            if self._meets_tier_requirements(
                assessment, content, metadata, requirements
            ):
                return tier_name

        return "insufficient"

    def _meets_tier_requirements(
        self,
        assessment: ContentQualityAssessment,
        content: str,
        metadata: Dict[str, Any],
        requirements: Dict[str, Any],
    ) -> bool:
        """Check if content meets specific tier requirements."""

        # Character count requirement
        if assessment.character_count < requirements.get("min_chars", 0):
            return False

        # Sentence count requirement
        if assessment.sentence_count < requirements.get("min_sentences", 0):
            return False

        # Quality score requirement
        if assessment.quality_score < requirements.get("min_quality_score", 0):
            return False

        # Legal pattern requirement
        legal_pattern_count = self._count_legal_patterns(content)
        if legal_pattern_count < requirements.get("min_legal_patterns", 0):
            return False

        # Metadata requirements
        if (
            requirements.get("metadata_complete")
            and assessment.metadata_completeness < 0.8
        ):
            return False
        elif (
            requirements.get("metadata_partial")
            and assessment.metadata_completeness < 0.5
        ):
            return False
        elif (
            requirements.get("metadata_minimal")
            and assessment.metadata_completeness < 0.2
        ):
            return False
        elif requirements.get("metadata_only") and not metadata:
            return False

        return True

    def _generate_improvement_recommendations(
        self,
        assessment: ContentQualityAssessment,
        content: str,
        metadata: Dict[str, Any],
    ) -> List[str]:
        """Generate specific recommendations for content improvement."""
        recommendations = []

        # Content length recommendations
        if assessment.character_count < 200:
            recommendations.append(
                "Content too short - try alternative sources or manual review"
            )
        elif assessment.character_count < 500:
            recommendations.append(
                "Content appears incomplete - verify full document was retrieved"
            )

        # Language pattern recommendations
        if assessment.language_confidence < 0.3:
            recommendations.append(
                "Low Polish language confidence - verify document language and encoding"
            )

        # Legal structure recommendations
        if assessment.legal_structure_score < 0.3:
            recommendations.append(
                "Weak legal document structure - may be informal document or fragment"
            )
        elif assessment.legal_structure_score < 0.6:
            recommendations.append(
                "Some legal patterns found but structure could be clearer"
            )

        # Metadata recommendations
        if assessment.metadata_completeness < 0.3:
            recommendations.append(
                "Insufficient metadata - gather document details from alternative sources"
            )
        elif assessment.metadata_completeness < 0.6:
            recommendations.append(
                "Metadata partially complete - enrich with additional document information"
            )

        # Quality tier specific recommendations
        if assessment.tier == "insufficient":
            recommendations.append(
                "Document requires manual review or alternative extraction method"
            )
        elif assessment.tier == "summary":
            recommendations.append(
                "Only summary information available - consider manual content enhancement"
            )
        elif assessment.tier == "low":
            recommendations.append(
                "Basic content available but may benefit from additional sources"
            )

        # Source-specific recommendations
        validation_details = assessment.validation_details
        if validation_details.get("error_penalty", 0) > 0.1:
            recommendations.append(
                "Content contains error indicators - verify source quality"
            )

        return recommendations

    def is_content_usable_for_tier(
        self, content: str, metadata: Dict[str, Any], required_tier: str
    ) -> bool:
        """Check if content meets minimum requirements for specified tier."""
        assessment = self.assess_content_quality(content, metadata)

        # Define tier hierarchy
        tier_hierarchy = ["high", "medium", "low", "summary", "insufficient"]

        required_index = (
            tier_hierarchy.index(required_tier)
            if required_tier in tier_hierarchy
            else len(tier_hierarchy)
        )
        actual_index = (
            tier_hierarchy.index(assessment.tier)
            if assessment.tier in tier_hierarchy
            else len(tier_hierarchy)
        )

        return actual_index <= required_index

    def get_quality_summary(self, assessment: ContentQualityAssessment) -> str:
        """Generate human-readable quality summary."""
        summary_parts = [
            f"Tier: {assessment.tier.upper()}",
            f"Score: {assessment.quality_score:.3f}",
            f"Length: {assessment.character_count} chars",
            f"Sentences: {assessment.sentence_count}",
        ]

        if assessment.language_confidence > 0:
            summary_parts.append(
                f"Polish confidence: {assessment.language_confidence:.2f}"
            )

        if assessment.legal_structure_score > 0:
            summary_parts.append(
                f"Legal structure: {assessment.legal_structure_score:.2f}"
            )

        return " | ".join(summary_parts)
