"""Specialized parser for Polish legal document structures."""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class LegalDocumentType(Enum):
    """Types of legal documents."""

    USTAWA = "ustawa"  # Law/Act
    ROZPORZADZENIE = "rozporządzenie"  # Regulation
    KODEKS = "kodeks"  # Code
    KONSTYTUCJA = "konstytucja"  # Constitution
    ORZECZENIE = "orzeczenie"  # Court ruling
    UCHWALA = "uchwała"  # Resolution


@dataclass
class LegalReference:
    """Represents a reference to a legal document or provision."""

    document_type: str
    document_title: str
    article: Optional[str] = None
    paragraph: Optional[str] = None
    point: Optional[str] = None
    letter: Optional[str] = None
    full_reference: str = ""


@dataclass
class LegalProvision:
    """Represents a legal provision (article, paragraph, etc.)."""

    provision_type: str  # article, paragraph, point, letter
    number: str
    title: Optional[str] = None
    content: str = ""
    sub_provisions: List["LegalProvision"] = None

    def __post_init__(self):
        if self.sub_provisions is None:
            self.sub_provisions = []


class PolishLegalParser:
    """Parser for Polish legal document structures."""

    def __init__(self):
        # Document type detection patterns
        self.document_patterns = {
            "ustawa": r"ustawa\s+z\s+dnia\s+\d{1,2}\s+\w+\s+\d{4}\s+r\.",
            "rozporządzenie": r"rozporządzenie\s+(?:ministra|rady\s+ministrów|prezesa\s+rady\s+ministrów)",
            "kodeks": r"kodeks\s+(?:cywilny|karny|postępowania\s+(?:cywilnego|karnego|administracyjnego))",
            "konstytucja": r"konstytucja\s+rzeczypospolitej\s+polskiej",
            "orzeczenie": r"(?:wyrok|postanowienie|orzeczenie)\s+(?:sądu|trybunału)",
            "uchwała": r"uchwała\s+(?:rady\s+ministrów|sejmu|senatu)",
        }

        # Provision structure patterns
        self.provision_patterns = {
            "article": r"(?:art\.|artykuł)\s*(\d+[a-z]?)",
            "paragraph": r"§\s*(\d+[a-z]?)",
            "ustep": r"(?<=\n)\s*(\d+)\.\s*",  # Paragraph within article (at line start)
            "point": r"(\d+)\)\s*",  # Numbered point
            "letter": r"([a-z])\)\s*",  # Lettered point
            "chapter": r"(?:rozdział|rozdz\.|rozd\.)\s*([IVX]+|\d+)",
            "section": r"(?:\bdział\b|sekcja)\s*([IVX]+|\d+)",
            "title": r"(?:tytuł)\s*([IVX]+|\d+)",
        }

        # Reference extraction patterns
        self.reference_patterns = [
            # Complex references: art. 123 ust. 2 pkt 3 lit. a
            r"art\.\s*(\d+[a-z]?)(?:\s+ust\.\s*(\d+))?(?:\s+pkt\s*(\d+))?(?:\s+lit\.\s*([a-z]))?",
            # Paragraph references: § 123 ust. 2
            r"§\s*(\d+[a-z]?)(?:\s+ust\.\s*(\d+))?",
            # Simple article references
            r"artykuł\s+(\d+[a-z]?)",
        ]

        # Compile patterns
        self.compiled_document_patterns = {
            doc_type: re.compile(pattern, re.IGNORECASE)
            for doc_type, pattern in self.document_patterns.items()
        }

        self.compiled_provision_patterns = {
            prov_type: re.compile(pattern, re.IGNORECASE)
            for prov_type, pattern in self.provision_patterns.items()
        }

        self.compiled_reference_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.reference_patterns
        ]

    def detect_document_type(self, text: str) -> Optional[LegalDocumentType]:
        """Detect the type of legal document."""
        if not text:
            return None

        # Check first few paragraphs for document type indicators
        sample_text = text[:1000].lower()

        for doc_type, pattern in self.compiled_document_patterns.items():
            if pattern.search(sample_text):
                try:
                    return LegalDocumentType(doc_type)
                except ValueError:
                    continue

        return None

    def extract_document_metadata(self, text: str) -> Dict[str, str]:
        """Extract metadata from document header."""
        metadata = {}

        # Extract title (usually in first few lines)
        lines = text.split("\n")[:20]  # Check first 20 lines

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for date patterns
            date_match = re.search(r"(\d{1,2}\s+\w+\s+\d{4})", line)
            if date_match:
                metadata["date"] = date_match.group(1)

            # Look for journal references (Dz.U.)
            journal_match = re.search(
                r"dz\.?\s*u\.?\s*(?:nr\s*)?(\d+)", line, re.IGNORECASE
            )
            if journal_match:
                metadata["journal_number"] = journal_match.group(1)

            # Look for position number
            pos_match = re.search(r"poz\.?\s*(\d+)", line, re.IGNORECASE)
            if pos_match:
                metadata["position"] = pos_match.group(1)

        return metadata

    def parse_provisions(self, text: str) -> List[LegalProvision]:
        """Parse legal provisions from text."""
        provisions = []

        # Find all provision markers
        markers = []
        for prov_type, pattern in self.compiled_provision_patterns.items():
            for match in pattern.finditer(text):
                markers.append(
                    {
                        "type": prov_type,
                        "number": match.group(1),
                        "start": match.start(),
                        "end": match.end(),
                        "match": match.group(0),
                    }
                )

        # Sort markers by position
        markers.sort(key=lambda x: x["start"])

        # Create provisions from markers
        for i, marker in enumerate(markers):
            start = marker["end"]
            end = markers[i + 1]["start"] if i + 1 < len(markers) else len(text)

            content = text[start:end].strip()
            if content:
                provisions.append(
                    LegalProvision(
                        provision_type=marker["type"],
                        number=marker["number"],
                        content=content,
                    )
                )

        return provisions

    def _parse_single_provision(self, text: str) -> Optional[LegalProvision]:
        """Parse a single legal provision."""
        text = text.strip()
        if not text:
            return None

        # Try to match each provision type
        for prov_type, pattern in self.compiled_provision_patterns.items():
            match = pattern.match(text)
            if match:
                number = match.group(1)
                content = text[match.end() :].strip()

                return LegalProvision(
                    provision_type=prov_type, number=number, content=content
                )

        return None

    def extract_references(self, text: str) -> List[LegalReference]:
        """Extract legal references from text."""
        references = []

        for pattern in self.compiled_reference_patterns:
            for match in pattern.finditer(text):
                groups = match.groups()

                # Create reference based on matched groups
                ref = LegalReference(
                    document_type="",  # Will be inferred from context
                    document_title="",
                    full_reference=match.group(0),
                )

                # Parse groups based on pattern structure
                if len(groups) >= 1 and groups[0]:
                    ref.article = groups[0]
                if len(groups) >= 2 and groups[1]:
                    ref.paragraph = groups[1]
                if len(groups) >= 3 and groups[2]:
                    ref.point = groups[2]
                if len(groups) >= 4 and groups[3]:
                    ref.letter = groups[3]

                references.append(ref)

        return references

    def parse_document_structure(self, text: str) -> Dict[str, Any]:
        """Parse complete document structure."""
        if not text:
            return {}

        return {
            "document_type": self.detect_document_type(text),
            "metadata": self.extract_document_metadata(text),
            "provisions": self.parse_provisions(text),
            "references": self.extract_references(text),
            "structure_elements": self._extract_structure_elements(text),
        }

    def _extract_structure_elements(self, text: str) -> Dict[str, List[Dict]]:
        """Extract structural elements like chapters, sections, etc."""
        elements = {
            "chapters": [],
            "sections": [],
            "titles": [],
            "articles": [],
            "paragraphs": [],
        }

        # Map provision types to element categories
        type_mapping = {
            "chapter": "chapters",
            "section": "sections",
            "title": "titles",
            "article": "articles",
            "paragraph": "paragraphs",
        }

        for prov_type, pattern in self.compiled_provision_patterns.items():
            if prov_type in type_mapping:
                category = type_mapping[prov_type]

                for match in pattern.finditer(text):
                    elements[category].append(
                        {
                            "number": match.group(1),
                            "position": match.start(),
                            "text": match.group(0),
                        }
                    )

        return elements


class LegalDocumentAnalyzer:
    """High-level analyzer for legal documents."""

    def __init__(self):
        self.parser = PolishLegalParser()

    def analyze_document(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive analysis of a legal document."""
        if not text:
            return {}

        structure = self.parser.parse_document_structure(text)

        # Add analysis metrics
        analysis = {
            "document_info": {
                "type": structure.get("document_type"),
                "metadata": structure.get("metadata", {}),
                "length": len(text),
                "word_count": len(text.split()),
            },
            "structure": structure.get("structure_elements", {}),
            "provisions": {
                "count": len(structure.get("provisions", [])),
                "types": self._count_provision_types(structure.get("provisions", [])),
            },
            "references": {
                "count": len(structure.get("references", [])),
                "references": structure.get("references", []),
            },
        }

        return analysis

    def _count_provision_types(
        self, provisions: List[LegalProvision]
    ) -> Dict[str, int]:
        """Count provisions by type."""
        counts = {}
        for provision in provisions:
            prov_type = provision.provision_type
            counts[prov_type] = counts.get(prov_type, 0) + 1
        return counts
