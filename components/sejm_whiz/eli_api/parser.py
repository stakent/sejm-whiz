"""Document structure extraction and parsing utilities for legal documents."""

import re
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass
from bs4 import BeautifulSoup
from html import unescape

from .models import LegalDocument, MultiActAmendment

logger = logging.getLogger(__name__)


@dataclass
class DocumentStructure:
    """Extracted structure of a legal document."""

    title: str
    preamble: Optional[str]
    chapters: List["Chapter"]
    articles: List["Article"]
    final_provisions: Optional[str]
    attachments: List["Attachment"]

    # Metadata
    cross_references: List[str]
    legal_citations: List[str]
    amendment_indicators: List[str]


@dataclass
class Chapter:
    """Legal document chapter."""

    number: str
    title: str
    articles: List["Article"]


@dataclass
class Article:
    """Legal document article."""

    number: str
    title: Optional[str]
    paragraphs: List["Paragraph"]
    points: List["Point"]


@dataclass
class Paragraph:
    """Legal document paragraph."""

    number: str
    content: str
    points: List["Point"]


@dataclass
class Point:
    """Legal document point."""

    number: str
    content: str
    subpoints: List["Subpoint"]


@dataclass
class Subpoint:
    """Legal document subpoint."""

    letter: str
    content: str


@dataclass
class Attachment:
    """Legal document attachment."""

    number: str
    title: str
    content: str


class LegalTextParser:
    """Parser for extracting structure from Polish legal documents."""

    def __init__(self):
        # Polish legal document patterns
        self.chapter_pattern = re.compile(
            r"Rozdział\s+([IVXLCDM]+)\s*\.?\s*(.*)", re.IGNORECASE
        )
        self.article_pattern = re.compile(
            r"Art\.\s*(\d+[a-z]?)\s*\.?\s*(.*)", re.IGNORECASE
        )
        self.paragraph_pattern = re.compile(r"§\s*(\d+)\s*\.?\s*(.*)", re.IGNORECASE)
        self.point_pattern = re.compile(r"(\d+)\)\s*(.*)", re.IGNORECASE)
        self.subpoint_pattern = re.compile(r"([a-z])\)\s*(.*)", re.IGNORECASE)

        # Cross-reference patterns
        self.act_reference_pattern = re.compile(
            r"ustaw[aeęiy]?\s+z\s+dnia\s+(\d{1,2}\s+\w+\s+\d{4})\s+r\.\s*(?:o\s+)?(.*?)(?:\s*\(|\s*$)",
            re.IGNORECASE,
        )
        self.article_reference_pattern = re.compile(
            r"art\.\s*(\d+[a-z]?)", re.IGNORECASE
        )
        self.paragraph_reference_pattern = re.compile(r"§\s*(\d+)", re.IGNORECASE)

        # Amendment indicators
        self.amendment_indicators = [
            r"zmienia\s+się",
            r"dodaje\s+się",
            r"uchyla\s+się",
            r"zastępuje\s+się",
            r"w\s+miejsce\s+.*?\s+wprowadza\s+się",
            r"po\s+.*?\s+dodaje\s+się",
            r"otrzymuje\s+brzmienie",
        ]

    def parse_html_content(self, html_content: str) -> DocumentStructure:
        """Parse HTML content of legal document."""

        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style"]):
                element.decompose()

            # Extract title from h1 tag first, then title tag as fallback
            title = None
            h1_tag = soup.find("h1")
            if h1_tag:
                title = h1_tag.get_text().strip()
            else:
                title_tag = soup.find("title")
                if title_tag:
                    title = title_tag.get_text().strip()

            # Extract text content
            text_content = soup.get_text()
            text_content = unescape(text_content)

            # Parse the structure
            structure = self.parse_text_content(text_content)

            # Override title if we found a better one from HTML tags
            if title:
                structure.title = title

            return structure

        except Exception as e:
            logger.error(f"Failed to parse HTML content: {e}")
            # Fallback to plain text parsing
            plain_text = re.sub(r"<[^>]+>", "", html_content)
            return self.parse_text_content(plain_text)

    def parse_text_content(self, text_content: str) -> DocumentStructure:
        """Parse plain text content of legal document."""

        lines = text_content.split("\n")
        lines = [line.strip() for line in lines if line.strip()]

        # Extract title (usually first significant line)
        title = lines[0] if lines else "Untitled Document"

        # Find document structure
        chapters = self._extract_chapters(lines)
        articles = self._extract_articles(lines)

        # Extract metadata
        cross_references = self._extract_cross_references(text_content)
        legal_citations = self._extract_legal_citations(text_content)
        amendment_indicators = self._find_amendment_indicators(text_content)

        # Extract other sections
        preamble = self._extract_preamble(lines)
        final_provisions = self._extract_final_provisions(lines)
        attachments = self._extract_attachments(lines)

        return DocumentStructure(
            title=title,
            preamble=preamble,
            chapters=chapters,
            articles=articles,
            final_provisions=final_provisions,
            attachments=attachments,
            cross_references=cross_references,
            legal_citations=legal_citations,
            amendment_indicators=amendment_indicators,
        )

    def _extract_chapters(self, lines: List[str]) -> List[Chapter]:
        """Extract chapters from document lines."""

        chapters = []
        current_chapter = None

        for line in lines:
            chapter_match = self.chapter_pattern.match(line)
            if chapter_match:
                if current_chapter:
                    chapters.append(current_chapter)

                current_chapter = Chapter(
                    number=chapter_match.group(1),
                    title=chapter_match.group(2).strip(),
                    articles=[],
                )

        if current_chapter:
            chapters.append(current_chapter)

        return chapters

    def _extract_articles(self, lines: List[str]) -> List[Article]:
        """Extract articles from document lines."""

        articles = []
        current_article = None

        i = 0
        while i < len(lines):
            line = lines[i]
            article_match = self.article_pattern.match(line)

            if article_match:
                if current_article:
                    articles.append(current_article)

                article_number = article_match.group(1)
                article_title = (
                    article_match.group(2).strip() if article_match.group(2) else None
                )

                # Extract paragraphs for this article
                paragraphs, i = self._extract_paragraphs(lines, i + 1)

                current_article = Article(
                    number=article_number,
                    title=article_title,
                    paragraphs=paragraphs,
                    points=[],
                )
                continue

            i += 1

        if current_article:
            articles.append(current_article)

        return articles

    def _extract_paragraphs(
        self, lines: List[str], start_index: int
    ) -> Tuple[List[Paragraph], int]:
        """Extract paragraphs starting from given index."""

        paragraphs = []
        i = start_index

        while i < len(lines):
            line = lines[i]

            # Stop if we hit next article
            if self.article_pattern.match(line):
                break

            paragraph_match = self.paragraph_pattern.match(line)
            if paragraph_match:
                paragraph_number = paragraph_match.group(1)
                paragraph_content = paragraph_match.group(2).strip()

                # Extract points for this paragraph
                points, i = self._extract_points(lines, i + 1)

                paragraphs.append(
                    Paragraph(
                        number=paragraph_number,
                        content=paragraph_content,
                        points=points,
                    )
                )
                continue

            i += 1

        return paragraphs, i

    def _extract_points(
        self, lines: List[str], start_index: int
    ) -> Tuple[List[Point], int]:
        """Extract points starting from given index."""

        points = []
        i = start_index

        while i < len(lines):
            line = lines[i]

            # Stop if we hit next article or paragraph
            if self.article_pattern.match(line) or self.paragraph_pattern.match(line):
                break

            point_match = self.point_pattern.match(line)
            if point_match:
                point_number = point_match.group(1)
                point_content = point_match.group(2).strip()

                # Extract subpoints
                subpoints, i = self._extract_subpoints(lines, i + 1)

                points.append(
                    Point(
                        number=point_number, content=point_content, subpoints=subpoints
                    )
                )
                continue

            i += 1

        return points, i

    def _extract_subpoints(
        self, lines: List[str], start_index: int
    ) -> Tuple[List[Subpoint], int]:
        """Extract subpoints starting from given index."""

        subpoints = []
        i = start_index

        while i < len(lines):
            line = lines[i]

            # Stop if we hit next article, paragraph, or point
            if (
                self.article_pattern.match(line)
                or self.paragraph_pattern.match(line)
                or self.point_pattern.match(line)
            ):
                break

            subpoint_match = self.subpoint_pattern.match(line)
            if subpoint_match:
                subpoint_letter = subpoint_match.group(1)
                subpoint_content = subpoint_match.group(2).strip()

                subpoints.append(
                    Subpoint(letter=subpoint_letter, content=subpoint_content)
                )

            i += 1

        return subpoints, i

    def _extract_cross_references(self, text: str) -> List[str]:
        """Extract cross-references to other legal acts."""

        references = []

        # More flexible patterns for cross-references
        cross_ref_patterns = [
            # Full act references with dates (more specific to avoid greedy matching)
            r"ustaw[aeęiy]?\s+z\s+dnia\s+\d{1,2}\s+\w+\s+\d{4}\s+r\.\s+o\s+[^\.]+?(?=\s|$)",
            # General act references (W ustawie...)
            r"[Ww]\s+ustawie\s+z\s+dnia\s+[^\.]+",
            r"[Ww]\s+ustawie\s+[^\.]+",
            # References to codes
            r"[Ww]\s+[Kk]odeksie\s+[^\.]+",
            r"[Kk]odeks\s+\w+[^\.]*",
            # References to regulations
            r"rozporządzeni[aeęiy]?\s+[^\.]+",
            # Constitutional references
            r"konstytucj[aeęiy]?\s+[^\.]*",
        ]

        for pattern in cross_ref_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                reference = match.group(0).strip()
                if reference and len(reference) > 5:  # Avoid very short matches
                    references.append(reference)

        return list(set(references))  # Remove duplicates

    def _extract_legal_citations(self, text: str) -> List[str]:
        """Extract legal citations (articles, paragraphs)."""

        citations = []

        # Find article citations
        article_matches = self.article_reference_pattern.finditer(text)
        for match in article_matches:
            citations.append(f"art. {match.group(1)}")

        # Find paragraph citations
        paragraph_matches = self.paragraph_reference_pattern.finditer(text)
        for match in paragraph_matches:
            citations.append(f"§ {match.group(1)}")

        return list(set(citations))  # Remove duplicates

    def _find_amendment_indicators(self, text: str) -> List[str]:
        """Find indicators of amendments in text."""

        indicators = []

        for pattern in self.amendment_indicators:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(match.group(0))

        return indicators

    def _extract_preamble(self, lines: List[str]) -> Optional[str]:
        """Extract document preamble."""

        preamble_lines = []

        for line in lines:
            # Stop at first article
            if self.article_pattern.match(line):
                break

            # Skip title line
            if line == lines[0]:
                continue

            # Skip empty lines and chapter headers
            if not line or self.chapter_pattern.match(line):
                continue

            preamble_lines.append(line)

        return "\n".join(preamble_lines) if preamble_lines else None

    def _extract_final_provisions(self, lines: List[str]) -> Optional[str]:
        """Extract final provisions section."""

        final_provisions_start = None

        for i, line in enumerate(lines):
            if re.search(r"przepisy\s+końcowe", line, re.IGNORECASE):
                final_provisions_start = i
                break

        if final_provisions_start is not None:
            final_lines = lines[final_provisions_start:]
            return "\n".join(final_lines)

        return None

    def _extract_attachments(self, lines: List[str]) -> List[Attachment]:
        """Extract document attachments."""

        attachments = []
        attachment_pattern = re.compile(r"załącznik\s+nr\s+(\d+)", re.IGNORECASE)

        current_attachment = None
        expecting_title = False

        for line in lines:
            attachment_match = attachment_pattern.match(line)
            if attachment_match:
                if current_attachment:
                    attachments.append(current_attachment)

                current_attachment = Attachment(
                    number=attachment_match.group(1), title=line, content=""
                )
                expecting_title = True
            elif current_attachment:
                if expecting_title and line.strip():
                    # First non-empty line after attachment header is the title
                    current_attachment.title += f" - {line}"
                    expecting_title = False
                else:
                    current_attachment.content += line + "\n"

        if current_attachment:
            attachments.append(current_attachment)

        return attachments


class MultiActAmendmentDetector:
    """Detector for amendments affecting multiple legal acts."""

    def __init__(self):
        self.act_patterns = [
            r"ustaw[aeęiy]?\s+z\s+dnia\s+\d+[^\.]*?r\.",  # Acts with dates
            r"ustaw[aeęiy]?\s+z\s+dnia\s+\d+[^\.]*?o\s+[^\.]+",  # Acts about something
            r"kodeks[ua]?\s+\w+[^\.]*",  # Codes
            r"rozporządzeni[aeęiy]?\s+[^\.]+",  # Regulations
            r"konstytucj[aeęiy]?\s+[^\.]*",  # Constitution
            r"w\s+ustawie\s+[^\.]+",  # In the act...
            r"w\s+kodeksie\s+\w+[^\.]*",  # In the code...
        ]

        self.omnibus_indicators = [
            r"przepisy\s+wprowadzające",
            r"zmiany\s+w\s+różnych\s+ustawach",
            r"nowelizacja\s+.*?\s+ustaw",
            r"kompleksowa\s+nowelizacja",
        ]

    def detect_multi_act_amendments(
        self, document: LegalDocument, content: str
    ) -> Optional[MultiActAmendment]:
        """Detect if document is a multi-act amendment."""

        affected_acts = self._find_affected_acts(content)

        if len(affected_acts) < 2:
            return None

        # Check for omnibus indicators
        is_omnibus = any(
            re.search(pattern, content, re.IGNORECASE)
            for pattern in self.omnibus_indicators
        )

        cross_references = self._extract_cross_references(content)
        impact_assessment = self._assess_impact(affected_acts, content)

        return MultiActAmendment(
            eli_id=document.eli_id,
            title=document.title,
            affected_acts=affected_acts,
            complexity_score=len(affected_acts),
            published_date=document.published_date,
            effective_date=document.effective_date,
            cross_references=cross_references,
            impact_assessment=impact_assessment,
        )

    def _find_affected_acts(self, content: str) -> List[str]:
        """Find all legal acts affected by amendment."""

        affected_acts = set()

        for pattern in self.act_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                affected_acts.add(match.group(0))

        return list(affected_acts)

    def _extract_cross_references(self, content: str) -> List[str]:
        """Extract cross-references between affected acts."""

        # This is a simplified implementation
        # In practice, this would use more sophisticated NLP techniques

        cross_ref_patterns = [
            r"w\s+związku\s+z\s+.*?art\.\s*\d+",
            r"zgodnie\s+z\s+.*?§\s*\d+",
            r"na\s+podstawie\s+.*?ust\.\s*\d+",
        ]

        cross_references = []

        for pattern in cross_ref_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                cross_references.append(match.group(0))

        return cross_references

    def _assess_impact(self, affected_acts: List[str], content: str) -> str:
        """Assess the impact of multi-act amendment."""

        impact_indicators = {
            "high": [
                r"fundamentalne\s+zmiany",
                r"kompleksowa\s+reforma",
                r"znaczące\s+modyfikacje",
            ],
            "medium": [
                r"dostosowanie\s+przepisów",
                r"korekta\s+regulacji",
                r"aktualizacja\s+przepisów",
            ],
            "low": [
                r"drobne\s+zmiany",
                r"techniczne\s+poprawki",
                r"ujednolicenie\s+terminologii",
            ],
        }

        for level, patterns in impact_indicators.items():
            if any(re.search(pattern, content, re.IGNORECASE) for pattern in patterns):
                return level

        # Default assessment based on number of affected acts
        if len(affected_acts) >= 10:
            return "high"
        elif len(affected_acts) >= 5:
            return "medium"
        else:
            return "low"
