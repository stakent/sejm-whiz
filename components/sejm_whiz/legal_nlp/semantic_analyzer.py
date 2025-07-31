"""Semantic analysis utilities for legal documents."""

from typing import Dict, List, Any
from dataclasses import dataclass
import re


@dataclass
class SemanticField:
    """Represents a semantic field or domain in legal text."""

    field_name: str
    terms: List[str]
    weight: float
    context_terms: List[str] = None

    def __post_init__(self):
        if self.context_terms is None:
            self.context_terms = []


class LegalSemanticAnalyzer:
    """Advanced semantic analysis for Polish legal documents."""

    def __init__(self):
        """Initialize semantic analyzer with legal domain knowledge."""
        self._init_semantic_fields()
        self._init_semantic_patterns()

    def _init_semantic_fields(self):
        """Initialize semantic fields for legal domains."""
        self.semantic_fields = {
            "civil_law": SemanticField(
                field_name="civil_law",
                terms=[
                    "własność",
                    "posiadanie",
                    "użytkowanie",
                    "spadek",
                    "dziedziczenie",
                    "umowa",
                    "zobowiązanie",
                    "szkoda",
                    "odszkodowanie",
                    "odpowiedzialność",
                    "małżeństwo",
                    "rozwód",
                    "alimenty",
                    "kuratela",
                    "opieka",
                ],
                weight=1.0,
                context_terms=[
                    "kodeks cywilny",
                    "prawo cywilne",
                    "stosunki cywilnoprawne",
                ],
            ),
            "criminal_law": SemanticField(
                field_name="criminal_law",
                terms=[
                    "przestępstwo",
                    "wykroczenie",
                    "kara",
                    "grzywna",
                    "więzienie",
                    "napaść",
                    "kradzież",
                    "oszustwo",
                    "zabójstwo",
                    "przemoc",
                    "prokurator",
                    "oskarżyciel",
                    "podejrzany",
                    "świadek",
                ],
                weight=1.0,
                context_terms=["kodeks karny", "prawo karne", "postępowanie karne"],
            ),
            "administrative_law": SemanticField(
                field_name="administrative_law",
                terms=[
                    "organ administracji",
                    "decyzja administracyjna",
                    "postępowanie administracyjne",
                    "odwołanie",
                    "skarga",
                    "samorząd",
                    "gmina",
                    "powiat",
                    "województwo",
                    "pozwolenie",
                    "koncesja",
                    "licencja",
                    "rejestracja",
                    "kontrola",
                ],
                weight=1.0,
                context_terms=[
                    "kodeks postępowania administracyjnego",
                    "prawo administracyjne",
                ],
            ),
            "constitutional_law": SemanticField(
                field_name="constitutional_law",
                terms=[
                    "konstytucja",
                    "prawa podstawowe",
                    "wolności obywatelskie",
                    "sejm",
                    "senat",
                    "prezydent",
                    "rada ministrów",
                    "trybunał konstytucyjny",
                    "demokracja",
                    "suwerenność",
                    "rozdział władz",
                ],
                weight=1.0,
                context_terms=["konstytucja", "ustrój", "system prawny"],
            ),
            "tax_law": SemanticField(
                field_name="tax_law",
                terms=[
                    "podatek",
                    "składka",
                    "opłata",
                    "danina",
                    "VAT",
                    "CIT",
                    "PIT",
                    "ordynacja podatkowa",
                    "zobowiązanie podatkowe",
                    "ulga podatkowa",
                    "kontrola podatkowa",
                    "interpretacja podatkowa",
                ],
                weight=1.0,
                context_terms=["prawo podatkowe", "ordynacja podatkowa", "fiskus"],
            ),
            "labor_law": SemanticField(
                field_name="labor_law",
                terms=[
                    "umowa o pracę",
                    "stosunek pracy",
                    "pracodawca",
                    "pracownik",
                    "wynagrodzenie",
                    "urlop",
                    "zwolnienie",
                    "wypowiedzenie",
                    "kodeks pracy",
                    "czas pracy",
                    "bezpieczeństwo pracy",
                    "związki zawodowe",
                ],
                weight=1.0,
                context_terms=["kodeks pracy", "prawo pracy", "stosunki pracy"],
            ),
        }

    def _init_semantic_patterns(self):
        """Initialize patterns for semantic analysis."""
        self.semantic_patterns = {
            "causal_relations": [
                r"(?:z powodu|na skutek|w wyniku|spowodowane przez)\s+(.+?)(?:\.|,|;)",
                r"(?:prowadzi do|skutkuje|powoduje|wynika z)\s+(.+?)(?:\.|,|;)",
                r"(?:przyczyna|skutek|następstwo)\s+(.+?)(?:\.|,|;)",
            ],
            "temporal_relations": [
                r"(?:przed|po|podczas|w trakcie|w czasie)\s+(.+?)(?:\.|,|;)",
                r"(?:następnie|później|wcześniej|równocześnie)\s+(.+?)(?:\.|,|;)",
                r"(?:od momentu|do czasu|w okresie)\s+(.+?)(?:\.|,|;)",
            ],
            "modal_expressions": [
                r"(?:może|powinien|musi|jest obowiązany|ma prawo)\s+(.+?)(?:\.|,|;)",
                r"(?:wolno|można|należy|trzeba|konieczne jest)\s+(.+?)(?:\.|,|;)",
                r"(?:dopuszczalne|niedopuszczalne|zakazane|dozwolone)\s+(.+?)(?:\.|,|;)",
            ],
            "conditional_relations": [
                r"(?:pod warunkiem|w przypadku|jeśli|gdy)\s+(.+?)(?:\.|,|;)",
                r"(?:o ile|chyba że|z wyjątkiem)\s+(.+?)(?:\.|,|;)",
            ],
        }

        # Compile patterns
        self.compiled_semantic_patterns = {}
        for pattern_type, patterns in self.semantic_patterns.items():
            self.compiled_semantic_patterns[pattern_type] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                for pattern in patterns
            ]

    def identify_semantic_fields(self, text: str) -> Dict[str, float]:
        """Identify and score semantic fields present in the text."""
        if not text:
            return {}

        text_lower = text.lower()
        field_scores = {}

        for field_name, field_info in self.semantic_fields.items():
            score = 0.0

            # Score based on field terms
            for term in field_info.terms:
                # Count exact matches
                exact_matches = text_lower.count(term.lower())
                score += exact_matches * field_info.weight

                # Count partial matches (for compound terms)
                if " " in term:
                    words = term.lower().split()
                    if all(word in text_lower for word in words):
                        score += 0.5 * field_info.weight

            # Boost score for context terms
            for context_term in field_info.context_terms:
                if context_term.lower() in text_lower:
                    score += 2.0 * field_info.weight

            # Normalize score by text length
            text_length_factor = len(text.split()) / 1000  # Normalize per 1000 words
            normalized_score = score / max(1.0, text_length_factor)

            if normalized_score > 0:
                field_scores[field_name] = round(normalized_score, 2)

        return dict(sorted(field_scores.items(), key=lambda x: x[1], reverse=True))

    def extract_semantic_relations(self, text: str) -> Dict[str, List[Dict]]:
        """Extract semantic relations from text."""
        if not text:
            return {}

        relations = {}

        for relation_type, patterns in self.compiled_semantic_patterns.items():
            relations[relation_type] = []

            for pattern in patterns:
                for match in pattern.finditer(text):
                    relation_info = {
                        "relation_text": match.group(0),
                        "extracted_content": match.group(1)
                        if match.groups()
                        else match.group(0),
                        "position": {"start": match.start(), "end": match.end()},
                        "context": self._extract_context(
                            text, match.start(), match.end()
                        ),
                    }
                    relations[relation_type].append(relation_info)

        return relations

    def _extract_context(
        self, text: str, start: int, end: int, window: int = 100
    ) -> str:
        """Extract context around a text span."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()

    def analyze_conceptual_density(self, text: str) -> Dict[str, Any]:
        """Analyze the conceptual density of legal text."""
        if not text:
            return {
                "total_words": 0,
                "legal_terms_count": 0,
                "unique_legal_terms": 0,
                "density_score": 0.0,
                "unique_terms_ratio": 0.0,
                "legal_terms_found": [],
            }

        words = text.split()
        total_words = len(words)

        if total_words == 0:
            return {
                "total_words": 0,
                "legal_terms_count": 0,
                "unique_legal_terms": 0,
                "density_score": 0.0,
                "unique_terms_ratio": 0.0,
                "legal_terms_found": [],
            }

        # Count legal terms
        legal_term_count = 0
        legal_terms_found = set()

        text_lower = text.lower()

        for field_info in self.semantic_fields.values():
            for term in field_info.terms:
                term_lower = term.lower()
                if term_lower in text_lower:
                    # Count occurrences
                    count = text_lower.count(term_lower)
                    legal_term_count += count
                    legal_terms_found.add(term)

        # Calculate density metrics
        density_score = legal_term_count / total_words
        unique_terms_ratio = len(legal_terms_found) / total_words

        return {
            "total_words": total_words,
            "legal_terms_count": legal_term_count,
            "unique_legal_terms": len(legal_terms_found),
            "density_score": round(density_score, 4),
            "unique_terms_ratio": round(unique_terms_ratio, 4),
            "legal_terms_found": list(legal_terms_found)[:20],  # Limit to first 20
        }

    def extract_legal_definitions(self, text: str) -> List[Dict[str, Any]]:
        """Extract legal definitions using semantic patterns."""
        if not text:
            return []

        definition_patterns = [
            # Standard definition patterns
            r"(\w+(?:\s+\w+)*)\s+(?:oznacza|znaczy|to)\s+(.+?)(?:\.|;)",
            r"(?:przez|pod pojęciem)\s+(\w+(?:\s+\w+)*)\s+(?:rozumie się|należy rozumieć)\s+(.+?)(?:\.|;)",
            r"(\w+(?:\s+\w+)*)\s+(?:w rozumieniu|na potrzeby)\s+(?:niniejszej ustawy|niniejszego aktu|przepisów)\s+(?:to|oznacza)\s+(.+?)(?:\.|;)",
            # Enumeration patterns
            r"(?:należy przez nie rozumieć|oznacza):\s*(?:\n\s*)?(.+?)(?:\.|;)",
        ]

        definitions = []

        for pattern in definition_patterns:
            compiled_pattern = re.compile(
                pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL
            )

            for match in compiled_pattern.finditer(text):
                groups = match.groups()

                if len(groups) >= 2:
                    term = groups[0].strip()
                    definition = groups[1].strip()

                    # Clean up the definition
                    definition = re.sub(r"\s+", " ", definition)

                    definitions.append(
                        {
                            "term": term,
                            "definition": definition,
                            "position": {"start": match.start(), "end": match.end()},
                            "context": self._extract_context(
                                text, match.start(), match.end(), 150
                            ),
                            "confidence": self._calculate_definition_confidence(
                                term, definition
                            ),
                        }
                    )

        # Remove duplicates and sort by confidence
        unique_definitions = self._deduplicate_definitions(definitions)
        return sorted(unique_definitions, key=lambda x: x["confidence"], reverse=True)

    def _calculate_definition_confidence(self, term: str, definition: str) -> float:
        """Calculate confidence score for a legal definition."""
        base_confidence = 0.7

        # Boost for longer, more detailed definitions
        length_boost = min(0.2, len(definition.split()) / 50)

        # Boost for presence of legal terminology
        legal_term_boost = 0.0
        definition_lower = definition.lower()
        for field_info in self.semantic_fields.values():
            for legal_term in field_info.terms:
                if legal_term.lower() in definition_lower:
                    legal_term_boost += 0.05

        legal_term_boost = min(0.2, legal_term_boost)

        # Penalty for very short terms or definitions
        short_penalty = 0.0
        if len(term) < 3 or len(definition.split()) < 3:
            short_penalty = -0.2

        confidence = base_confidence + length_boost + legal_term_boost + short_penalty
        return max(0.0, min(1.0, confidence))

    def _deduplicate_definitions(self, definitions: List[Dict]) -> List[Dict]:
        """Remove duplicate definitions based on term similarity."""
        if not definitions:
            return []

        unique_definitions = []
        seen_terms = set()

        for def_info in definitions:
            term_lower = def_info["term"].lower().strip()

            # Simple deduplication - more sophisticated methods could be added
            if term_lower not in seen_terms:
                seen_terms.add(term_lower)
                unique_definitions.append(def_info)

        return unique_definitions

    def analyze_argumentative_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the argumentative structure of legal text."""
        if not text:
            return {
                "arguments": {
                    "premises": [],
                    "conclusions": [],
                    "counterarguments": [],
                    "justifications": [],
                },
                "argument_counts": {
                    "premises": 0,
                    "conclusions": 0,
                    "counterarguments": 0,
                    "justifications": 0,
                },
                "total_argumentative_elements": 0,
                "argumentative_complexity": 0.0,
            }

        # Patterns for argumentative elements
        argumentative_patterns = {
            "premises": [
                r"(?:ponieważ|gdyż|bowiem|albowiem|z uwagi na to, że)\s+(.+?)(?:\.|,|;)",
                r"(?:mając na uwadze|uwzględniając|w związku z tym, że)\s+(.+?)(?:\.|,|;)",
            ],
            "conclusions": [
                r"(?:dlatego|w związku z tym|stąd|zatem|wobec tego)\s+(.+?)(?:\.|,|;)",
                r"(?:z tego wynika|można stwierdzić|należy uznać)\s+(.+?)(?:\.|,|;)",
            ],
            "counterarguments": [
                r"(?:jednak|jednakże|niemniej|mimo to|pomimo)\s+(.+?)(?:\.|,|;)",
                r"(?:z drugiej strony|natomiast|ale|lecz)\s+(.+?)(?:\.|,|;)",
            ],
            "justifications": [
                r"(?:uzasadnienie|podstawa prawna|ratio legis)\s+(.+?)(?:\.|,|;)",
                r"(?:cel ustawy|zamierzenie prawodawcy)\s+(.+?)(?:\.|,|;)",
            ],
        }

        arguments = {}

        for arg_type, patterns in argumentative_patterns.items():
            arguments[arg_type] = []

            for pattern in patterns:
                compiled_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)

                for match in compiled_pattern.finditer(text):
                    arguments[arg_type].append(
                        {
                            "text": match.group(1)
                            if match.groups()
                            else match.group(0),
                            "full_match": match.group(0),
                            "position": {"start": match.start(), "end": match.end()},
                            "context": self._extract_context(
                                text, match.start(), match.end()
                            ),
                        }
                    )

        # Calculate argumentative complexity
        total_args = sum(len(args) for args in arguments.values())
        complexity_score = min(1.0, total_args / 20)  # Normalize to 0-1

        return {
            "arguments": arguments,
            "argument_counts": {
                arg_type: len(args) for arg_type, args in arguments.items()
            },
            "total_argumentative_elements": total_args,
            "argumentative_complexity": round(complexity_score, 2),
        }

    def perform_comprehensive_semantic_analysis(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive semantic analysis of legal text."""
        if not text:
            return {}

        return {
            "semantic_fields": self.identify_semantic_fields(text),
            "semantic_relations": self.extract_semantic_relations(text),
            "conceptual_density": self.analyze_conceptual_density(text),
            "legal_definitions": self.extract_legal_definitions(text)[:10],  # Top 10
            "argumentative_structure": self.analyze_argumentative_structure(text),
            "analysis_metadata": {
                "text_length": len(text),
                "word_count": len(text.split()),
                "analysis_timestamp": None,  # Could be added if needed
            },
        }
