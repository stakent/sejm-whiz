"""Cross-register matching between formal legal language and parliamentary proceedings."""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass
import re

from sejm_whiz.embeddings import get_similarity_calculator
from sejm_whiz.text_processing import normalize_legal_text, process_legal_document

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result from cross-register matching operation."""

    formal_text: str
    informal_text: str
    similarity_score: float
    match_type: str
    confidence: float
    normalized_forms: Dict[str, str]
    key_terms: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "formal_text": self.formal_text,
            "informal_text": self.informal_text,
            "similarity_score": self.similarity_score,
            "match_type": self.match_type,
            "confidence": self.confidence,
            "normalized_forms": self.normalized_forms,
            "key_terms": self.key_terms,
            "metadata": self.metadata,
        }


class CrossRegisterMatcher:
    """Cross-register matching between legal and parliamentary language."""

    def __init__(self, similarity_calculator=None):
        """Initialize cross-register matcher.

        Args:
            similarity_calculator: Similarity calculator instance
        """
        self.similarity_calculator = (
            similarity_calculator or get_similarity_calculator()
        )
        self.logger = logging.getLogger(__name__)

        # Legal-to-parliamentary term mappings
        self.register_mappings = {
            # Legal formalities -> Parliamentary language
            "niniejsza ustawa": ["ta ustawa", "obecna ustawa", "przedmiotowa ustawa"],
            "w rozumieniu niniejszej ustawy": ["według tej ustawy", "w tej ustawie"],
            "stosuje się przepisy": [
                "obowiązują przepisy",
                "mają zastosowanie przepisy",
            ],
            "z zastrzeżeniem": ["oprócz przypadków", "z wyjątkiem"],
            "podlega karze": ["grozi kara", "można ukarać"],
            "organ właściwy": ["właściwy urząd", "odpowiedni organ"],
            "w terminie": ["do czasu", "w ciągu"],
            "zgodnie z": ["według", "na podstawie"],
            "w zakresie": ["jeśli chodzi o", "w sprawie"],
            "jednakże": ["ale", "jednak", "natomiast"],
            # Parliamentary -> Legal
            "mówimy o": ["dotyczy", "odnosi się do"],
            "chodzi o to": ["celem jest", "ma na celu"],
            "trzeba": ["należy", "powinien"],
            "można": ["ma prawo", "jest uprawniony"],
            "nie wolno": ["zabronione jest", "nie można"],
            "wszyscy": ["każdy", "osoby"],
            "każdy wie": ["powszechnie wiadomo", "oczywiste jest"],
        }

        # Compile regex patterns for efficiency
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for term matching."""
        self.formal_patterns = {}
        self.informal_patterns = {}

        for formal, informal_list in self.register_mappings.items():
            # Create patterns for formal terms
            self.formal_patterns[formal] = re.compile(
                r"\b" + re.escape(formal) + r"\b", re.IGNORECASE
            )

            # Create patterns for informal terms
            for informal in informal_list:
                self.informal_patterns[informal] = re.compile(
                    r"\b" + re.escape(informal) + r"\b", re.IGNORECASE
                )

    def match_registers(
        self,
        legal_text: str,
        parliamentary_text: str,
        similarity_threshold: float = 0.7,
        confidence_threshold: float = 0.6,
    ) -> List[MatchResult]:
        """Match formal legal language with informal parliamentary language.

        Args:
            legal_text: Formal legal document text
            parliamentary_text: Parliamentary proceedings text
            similarity_threshold: Minimum similarity score for matches
            confidence_threshold: Minimum confidence score for matches

        Returns:
            List of MatchResult objects
        """
        try:
            matches = []

            # Process and normalize texts
            processed_legal = process_legal_document(legal_text)
            processed_parl = process_legal_document(parliamentary_text)

            legal_normalized = normalize_legal_text(
                processed_legal.get("clean_text", legal_text)
            )
            parl_normalized = normalize_legal_text(
                processed_parl.get("clean_text", parliamentary_text)
            )

            self.logger.debug(
                f"Processing texts: legal={len(legal_normalized)} chars, parliamentary={len(parl_normalized)} chars"
            )

            # Find direct term mappings
            direct_matches = self._find_direct_mappings(
                legal_normalized, parl_normalized
            )
            matches.extend(direct_matches)

            # Find semantic similarities using embeddings
            semantic_matches = self._find_semantic_matches(
                legal_normalized, parl_normalized, similarity_threshold
            )
            matches.extend(semantic_matches)

            # Find structural patterns
            structural_matches = self._find_structural_patterns(
                legal_normalized, parl_normalized
            )
            matches.extend(structural_matches)

            # Filter by confidence threshold
            filtered_matches = [
                m for m in matches if m.confidence >= confidence_threshold
            ]

            # Sort by similarity score
            filtered_matches.sort(key=lambda m: m.similarity_score, reverse=True)

            self.logger.info(f"Found {len(filtered_matches)} cross-register matches")
            return filtered_matches

        except Exception as e:
            self.logger.error(f"Cross-register matching failed: {e}")
            raise

    def _find_direct_mappings(
        self, legal_text: str, parl_text: str
    ) -> List[MatchResult]:
        """Find direct term mappings between registers."""
        matches = []

        for formal_term, informal_terms in self.register_mappings.items():
            formal_pattern = self.formal_patterns[formal_term]

            # Check if formal term appears in legal text
            formal_matches = formal_pattern.findall(legal_text)
            if not formal_matches:
                continue

            # Check for informal equivalents in parliamentary text
            for informal_term in informal_terms:
                informal_pattern = self.informal_patterns[informal_term]
                informal_matches = informal_pattern.findall(parl_text)

                if informal_matches:
                    # Calculate similarity score based on context
                    similarity_score = self._calculate_context_similarity(
                        legal_text, parl_text, formal_term, informal_term
                    )

                    confidence = 0.9  # High confidence for direct mappings

                    match = MatchResult(
                        formal_text=formal_term,
                        informal_text=informal_term,
                        similarity_score=similarity_score,
                        match_type="direct_mapping",
                        confidence=confidence,
                        normalized_forms={
                            "formal": formal_term,
                            "informal": informal_term,
                        },
                        key_terms=[formal_term, informal_term],
                        metadata={
                            "formal_occurrences": len(formal_matches),
                            "informal_occurrences": len(informal_matches),
                            "match_method": "pattern_matching",
                        },
                    )

                    matches.append(match)

        return matches

    def _find_semantic_matches(
        self, legal_text: str, parl_text: str, threshold: float
    ) -> List[MatchResult]:
        """Find semantic matches using embedding similarity."""
        matches = []

        try:
            # Split texts into sentences for better matching
            legal_sentences = [
                s.strip() for s in legal_text.split(".") if len(s.strip()) > 20
            ]
            parl_sentences = [
                s.strip() for s in parl_text.split(".") if len(s.strip()) > 20
            ]

            # Limit to prevent excessive computation
            legal_sentences = legal_sentences[:50]
            parl_sentences = parl_sentences[:50]

            # Calculate similarities between sentence pairs
            for legal_sent in legal_sentences:
                for parl_sent in parl_sentences:
                    try:
                        similarity_result = (
                            self.similarity_calculator.calculate_similarity(
                                legal_sent, parl_sent
                            )
                        )

                        if similarity_result.similarity_score >= threshold:
                            # Extract key terms from both sentences
                            legal_terms = self._extract_key_terms(legal_sent)
                            parl_terms = self._extract_key_terms(parl_sent)

                            confidence = min(
                                0.8, similarity_result.similarity_score + 0.1
                            )

                            match = MatchResult(
                                formal_text=legal_sent[:200] + "..."
                                if len(legal_sent) > 200
                                else legal_sent,
                                informal_text=parl_sent[:200] + "..."
                                if len(parl_sent) > 200
                                else parl_sent,
                                similarity_score=similarity_result.similarity_score,
                                match_type="semantic_similarity",
                                confidence=confidence,
                                normalized_forms={
                                    "formal": legal_sent,
                                    "informal": parl_sent,
                                },
                                key_terms=list(set(legal_terms + parl_terms)),
                                metadata={
                                    "embedding_similarity": similarity_result.similarity_score,
                                    "legal_sentence_length": len(legal_sent),
                                    "parliamentary_sentence_length": len(parl_sent),
                                    "match_method": "embedding_similarity",
                                },
                            )

                            matches.append(match)

                    except Exception as e:
                        self.logger.warning(
                            f"Failed to calculate similarity for sentence pair: {e}"
                        )
                        continue

        except Exception as e:
            self.logger.warning(f"Semantic matching failed: {e}")

        return matches

    def _find_structural_patterns(
        self, legal_text: str, parl_text: str
    ) -> List[MatchResult]:
        """Find structural pattern matches between texts."""
        matches = []

        # Define structural patterns
        patterns = {
            "article_reference": {
                "formal": r"art\.\s*(\d+)",
                "informal": r"artykuł[a-ząę]*\s*(\d+|pierwszy|drugi|trzeci)",
                "confidence": 0.85,
            },
            "paragraph_reference": {
                "formal": r"§\s*(\d+)",
                "informal": r"paragraf\s*(\d+|pierwszy|drugi|trzeci)",
                "confidence": 0.85,
            },
            "legal_obligation": {
                "formal": r"(należy|powinien|zobowiązany)",
                "informal": r"(trzeba|musi|powinien|ma obowiązek)",
                "confidence": 0.75,
            },
            "legal_prohibition": {
                "formal": r"(zabrania się|nie wolno|zabronione)",
                "informal": r"(nie można|nie wolno|zakaz)",
                "confidence": 0.75,
            },
        }

        for pattern_name, pattern_config in patterns.items():
            formal_pattern = re.compile(pattern_config["formal"], re.IGNORECASE)
            informal_pattern = re.compile(pattern_config["informal"], re.IGNORECASE)

            formal_matches = formal_pattern.findall(legal_text)
            informal_matches = informal_pattern.findall(parl_text)

            if formal_matches and informal_matches:
                # Create match for pattern category
                similarity_score = min(
                    0.9, len(formal_matches) * len(informal_matches) * 0.1
                )

                match = MatchResult(
                    formal_text=f"Pattern: {pattern_name} (formal)",
                    informal_text=f"Pattern: {pattern_name} (informal)",
                    similarity_score=similarity_score,
                    match_type="structural_pattern",
                    confidence=pattern_config["confidence"],
                    normalized_forms={
                        "formal": str(formal_matches),
                        "informal": str(informal_matches),
                    },
                    key_terms=[pattern_name],
                    metadata={
                        "pattern_name": pattern_name,
                        "formal_matches": formal_matches,
                        "informal_matches": informal_matches,
                        "match_method": "structural_pattern",
                    },
                )

                matches.append(match)

        return matches

    def _calculate_context_similarity(
        self, legal_text: str, parl_text: str, formal_term: str, informal_term: str
    ) -> float:
        """Calculate context similarity around matched terms."""
        try:
            # Extract context around terms (±100 characters)
            formal_contexts = self._extract_contexts(legal_text, formal_term, 100)
            informal_contexts = self._extract_contexts(parl_text, informal_term, 100)

            if not formal_contexts or not informal_contexts:
                return 0.5  # Default similarity

            # Calculate similarity between contexts
            max_similarity = 0.0
            for formal_ctx in formal_contexts:
                for informal_ctx in informal_contexts:
                    try:
                        result = self.similarity_calculator.calculate_similarity(
                            formal_ctx, informal_ctx
                        )
                        max_similarity = max(max_similarity, result.similarity_score)
                    except Exception:
                        continue

            return max_similarity

        except Exception as e:
            self.logger.warning(f"Context similarity calculation failed: {e}")
            return 0.5

    def _extract_contexts(self, text: str, term: str, context_size: int) -> List[str]:
        """Extract contexts around a term."""
        contexts = []
        pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)

        for match in pattern.finditer(text):
            start = max(0, match.start() - context_size)
            end = min(len(text), match.end() + context_size)
            context = text[start:end]
            contexts.append(context)

        return contexts

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key legal terms from text."""
        # Simple key term extraction based on common legal patterns
        legal_terms = []

        # Legal structure terms
        structure_patterns = [
            r"art\.\s*\d+",
            r"§\s*\d+",
            r"ust\.\s*\d+",
            r"pkt\s*\d+",
            r"rozdział\s*\w+",
            r"dział\s*\w+",
        ]

        # Legal action terms
        action_patterns = [
            r"należy",
            r"powinien",
            r"zobowiązany",
            r"uprawniony",
            r"zabrania się",
            r"nie wolno",
            r"dozwolone",
        ]

        all_patterns = structure_patterns + action_patterns

        for pattern in all_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            legal_terms.extend(matches)

        return list(set(legal_terms))


# Singleton instance
_cross_register_matcher_instance = None


def get_cross_register_matcher() -> CrossRegisterMatcher:
    """Get singleton cross-register matcher instance."""
    global _cross_register_matcher_instance
    if _cross_register_matcher_instance is None:
        _cross_register_matcher_instance = CrossRegisterMatcher()
    return _cross_register_matcher_instance
