"""Query processing and normalization for semantic search."""

import re
import logging
from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from sejm_whiz.text_processing import (
    clean_text,
    normalize_legal_text,
    tokenize_polish_legal,
)
from sejm_whiz.text_processing.legal_parser import LegalReference

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of search queries."""

    GENERAL = "general"
    LEGAL_REFERENCE = "legal_reference"
    CONCEPT_SEARCH = "concept_search"
    TEMPORAL_SEARCH = "temporal_search"
    CROSS_REGISTER = "cross_register"


@dataclass
class ProcessedQuery:
    """Processed and normalized search query."""

    original_query: str
    normalized_query: str
    query_type: QueryType
    legal_terms: List[str] = field(default_factory=list)
    legal_references: List[LegalReference] = field(default_factory=list)
    temporal_filters: Dict[str, Any] = field(default_factory=dict)
    expanded_terms: List[str] = field(default_factory=list)
    search_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "query_type": self.query_type.value,
            "legal_terms": self.legal_terms,
            "legal_references": [ref.__dict__ for ref in self.legal_references],
            "temporal_filters": self.temporal_filters,
            "expanded_terms": self.expanded_terms,
            "search_metadata": self.search_metadata,
        }


class QueryProcessor:
    """Processes and normalizes search queries for legal document search."""

    def __init__(self):
        """Initialize the query processor."""
        self._legal_terms_dict = self._load_legal_terms_dictionary()
        self._legal_synonyms = self._load_legal_synonyms()
        self._temporal_patterns = self._load_temporal_patterns()
        self._legal_reference_patterns = self._load_legal_reference_patterns()

    def process_query(self, query: str, expand_terms: bool = True) -> ProcessedQuery:
        """
        Process and normalize a search query.

        Args:
            query: Raw search query
            expand_terms: Whether to expand query with synonyms

        Returns:
            Processed query with normalization and metadata
        """
        logger.debug(f"Processing query: {query}")

        # Basic normalization
        normalized = self._normalize_query(query)

        # Detect query type
        query_type = self._detect_query_type(query, normalized)

        # Extract legal terms
        legal_terms = self._extract_legal_terms(normalized)

        # Extract legal references
        legal_references = self._extract_legal_references(query)

        # Extract temporal filters
        temporal_filters = self._extract_temporal_filters(query)

        # Expand terms with synonyms
        expanded_terms = []
        if expand_terms:
            expanded_terms = self._expand_query_terms(normalized, legal_terms)

        # Build search metadata
        search_metadata = {
            "processing_timestamp": datetime.utcnow().isoformat(),
            "has_legal_terms": len(legal_terms) > 0,
            "has_references": len(legal_references) > 0,
            "has_temporal_filters": len(temporal_filters) > 0,
            "expansion_count": len(expanded_terms),
        }

        return ProcessedQuery(
            original_query=query,
            normalized_query=normalized,
            query_type=query_type,
            legal_terms=legal_terms,
            legal_references=legal_references,
            temporal_filters=temporal_filters,
            expanded_terms=expanded_terms,
            search_metadata=search_metadata,
        )

    def _normalize_query(self, query: str) -> str:
        """Normalize query text for better search."""
        # Basic text cleaning
        normalized = clean_text(query)

        # Legal text normalization
        normalized = normalize_legal_text(normalized)

        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized).strip()

        # Convert to lowercase for processing
        normalized = normalized.lower()

        logger.debug(f"Normalized query: {normalized}")
        return normalized

    def _detect_query_type(self, original: str, normalized: str) -> QueryType:
        """Detect the type of search query."""
        # Check for legal references
        if self._has_legal_reference_pattern(original):
            return QueryType.LEGAL_REFERENCE

        # Check for temporal patterns
        if self._has_temporal_pattern(original):
            return QueryType.TEMPORAL_SEARCH

        # Check for cross-register indicators
        if self._has_cross_register_indicators(normalized):
            return QueryType.CROSS_REGISTER

        # Check for legal concept search
        if self._has_legal_concepts(normalized):
            return QueryType.CONCEPT_SEARCH

        return QueryType.GENERAL

    def _extract_legal_terms(self, normalized_query: str) -> List[str]:
        """Extract legal terms from the query."""
        legal_terms = []

        # Tokenize the query
        tokens = tokenize_polish_legal(normalized_query)

        # Find legal terms using dictionary
        for token in tokens:
            if token.lower() in self._legal_terms_dict:
                legal_terms.append(token.lower())

        # Find multi-word legal terms
        for term in self._legal_terms_dict:
            if len(term.split()) > 1 and term in normalized_query:
                legal_terms.append(term)

        return list(set(legal_terms))  # Remove duplicates

    def _extract_legal_references(self, query: str) -> List[LegalReference]:
        """Extract legal document references from the query."""
        references = []

        for pattern_name, pattern in self._legal_reference_patterns.items():
            matches = re.finditer(pattern, query, re.IGNORECASE)

            for match in matches:
                ref = self._parse_legal_reference(match, pattern_name)
                if ref:
                    references.append(ref)

        return references

    def _extract_temporal_filters(self, query: str) -> Dict[str, Any]:
        """Extract temporal filters from the query."""
        temporal_filters = {}

        for filter_type, patterns in self._temporal_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    temporal_filters[filter_type] = self._parse_temporal_match(
                        match, filter_type
                    )

        return temporal_filters

    def _expand_query_terms(
        self, normalized_query: str, legal_terms: List[str]
    ) -> List[str]:
        """Expand query terms with synonyms and related terms."""
        expanded_terms = []

        # Expand legal terms with synonyms
        for term in legal_terms:
            if term in self._legal_synonyms:
                expanded_terms.extend(self._legal_synonyms[term])

        # Add common legal variations
        tokens = normalized_query.split()
        for token in tokens:
            variations = self._generate_legal_variations(token)
            expanded_terms.extend(variations)

        return list(set(expanded_terms))  # Remove duplicates

    def _load_legal_terms_dictionary(self) -> Set[str]:
        """Load dictionary of legal terms."""
        return {
            # Basic legal terms
            "ustawa",
            "kodeks",
            "rozporządzenie",
            "konstytucja",
            "orzeczenie",
            "wyrok",
            "uchwała",
            "obwieszczenie",
            "artykuł",
            "paragraf",
            "ustęp",
            "punkt",
            "litera",
            # Legal concepts
            "prawo",
            "przepis",
            "norma",
            "regulacja",
            "zobowiązanie",
            "odpowiedzialność",
            "sankcja",
            "kara",
            "grzywna",
            "postępowanie",
            "proces",
            "sprawa",
            "wniosek",
            # Legal entities
            "sąd",
            "trybunał",
            "prokuratura",
            "policja",
            "minister",
            "ministerstwo",
            "urząd",
            "gmina",
            "powiat",
            "województwo",
            "sejm",
            "senat",
            # Legal procedures
            "nowelizacja",
            "zmiana",
            "uchylenie",
            "dodanie",
            "wejście w życie",
            "vacatio legis",
            "publikacja",
            "ogłoszenie",
            "promulgacja",
            # Common legal phrases
            "zgodnie z",
            "na podstawie",
            "w związku z",
            "z zastrzeżeniem",
            "nie dotyczy",
            "stosuje się",
        }

    def _load_legal_synonyms(self) -> Dict[str, List[str]]:
        """Load synonyms for legal terms."""
        return {
            "ustawa": ["akt prawny", "prawo", "regulacja"],
            "kodeks": ["zbiór przepisów", "kodyfikacja"],
            "rozporządzenie": ["rozp.", "akt wykonawczy"],
            "artykuł": ["art.", "art", "przepis"],
            "paragraf": ["§", "par.", "paragraf"],
            "ustęp": ["ust.", "ustęp"],
            "punkt": ["pkt", "pkt.", "punkt"],
            "sąd": ["trybunał", "organ sądowy", "instancja"],
            "prawo": ["przepis", "norma", "regulacja"],
            "kara": ["sankcja", "grzywna", "penalizacja"],
        }

    def _load_temporal_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for temporal query detection."""
        return {
            "year": [
                r"\b(19|20)\d{2}\b",
                r"\b\d{4}\s*rok",
                r"\bw\s*roku\s*(\d{4})\b",
            ],
            "date_range": [
                r"\bod\s+(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4})\s+do\s+(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4})",
                r"\bpomiędzy\s+(\d{4})\s*[ai]\s*(\d{4})",
            ],
            "relative_time": [
                r"\bw\s*ostatni[ch]*\s*(\d+)\s*(rok|lat|miesiąc|dni)",
                r"\bprzed\s*(\d+)\s*(rok|lat|miesiąc|dni)",
                r"\bpo\s*(\d{4})",
            ],
        }

    def _load_legal_reference_patterns(self) -> Dict[str, str]:
        """Load patterns for legal reference detection."""
        return {
            "article": r"\bart\.\s*(\d+)(?:\s*ust\.\s*(\d+))?(?:\s*pkt\s*(\d+))?",
            "paragraph": r"\b§\s*(\d+)(?:\s*ust\.\s*(\d+))?",
            "act_reference": r"\bustaw[aąy]\s+z\s+dnia\s+(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4})",
            "regulation": r"\brozporządzeni[aeu]\s+(.+?)(?:\s+z\s+dnia\s+(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}))?",
        }

    def _has_legal_reference_pattern(self, query: str) -> bool:
        """Check if query contains legal reference patterns."""
        for pattern in self._legal_reference_patterns.values():
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False

    def _has_temporal_pattern(self, query: str) -> bool:
        """Check if query contains temporal patterns."""
        for patterns in self._temporal_patterns.values():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return True
        return False

    def _has_cross_register_indicators(self, query: str) -> bool:
        """Check if query indicates cross-register search."""
        indicators = [
            "posiedzenie",
            "debata",
            "głosowanie",
            "komisja",
            "poseł",
            "senator",
            "marszałek",
            "interpelacja",
            "projekt ustawy",
            "nowelizacja",
            "wniosek",
        ]
        return any(indicator in query for indicator in indicators)

    def _has_legal_concepts(self, query: str) -> bool:
        """Check if query contains legal concepts."""
        concept_indicators = [
            "odpowiedzialność",
            "prawo",
            "obowiązek",
            "uprawnienie",
            "procedura",
            "postępowanie",
            "orzekanie",
            "interpretacja",
        ]
        return any(concept in query for concept in concept_indicators)

    def _parse_legal_reference(
        self, match: re.Match, pattern_name: str
    ) -> Optional[LegalReference]:
        """Parse a legal reference from regex match."""
        try:
            if pattern_name == "article":
                article = match.group(1)
                paragraph = match.group(2) if len(match.groups()) > 1 else None
                point = match.group(3) if len(match.groups()) > 2 else None

                return LegalReference(
                    document_type="article",
                    document_title="",
                    article=article,
                    paragraph=paragraph,
                    point=point,
                    full_reference=match.group(0),
                )

            elif pattern_name == "act_reference":
                date = match.group(1)
                return LegalReference(
                    document_type="ustawa",
                    document_title=f"ustawa z dnia {date}",
                    full_reference=match.group(0),
                )

            # Add more parsing logic for other patterns

        except Exception as e:
            logger.warning(f"Error parsing legal reference: {e}")

        return None

    def _parse_temporal_match(self, match: re.Match, filter_type: str) -> Any:
        """Parse temporal information from regex match."""
        if filter_type == "year":
            return int(match.group(0))
        elif filter_type == "relative_time":
            # Parse relative time expressions
            return {"text": match.group(0), "parsed": True}
        else:
            return match.group(0)

    def _generate_legal_variations(self, term: str) -> List[str]:
        """Generate legal variations of a term."""
        variations = []

        # Common Polish legal abbreviations
        abbreviations = {
            "artykuł": ["art.", "art"],
            "paragraf": ["§", "par."],
            "ustęp": ["ust.", "ust"],
            "punkt": ["pkt", "pkt."],
            "ustawa": ["u.", "ust."],
            "rozporządzenie": ["rozp.", "rozp"],
        }

        if term in abbreviations:
            variations.extend(abbreviations[term])

        # Reverse lookup for abbreviations
        for full_term, abbrevs in abbreviations.items():
            if term in abbrevs:
                variations.append(full_term)

        return variations


def get_query_processor() -> QueryProcessor:
    """Get a query processor instance."""
    return QueryProcessor()


# Module-level processor instance
_processor_instance: Optional[QueryProcessor] = None


def process_search_query(query: str, expand_terms: bool = True) -> ProcessedQuery:
    """
    Process a search query using the default processor.

    Args:
        query: Raw search query
        expand_terms: Whether to expand query with synonyms

    Returns:
        Processed query with normalization and metadata
    """
    global _processor_instance

    if _processor_instance is None:
        _processor_instance = QueryProcessor()

    return _processor_instance.process_query(query, expand_terms)
