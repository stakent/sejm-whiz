"""Text processing and extraction for legal documents."""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from bs4 import BeautifulSoup
import hashlib

from .config import get_ingestion_config, DocumentIngestionConfig

logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """Processed legal document with extracted content and metadata."""
    title: str
    content: str
    document_type: str
    eli_identifier: Optional[str] = None
    source_url: Optional[str] = None
    published_at: Optional[datetime] = None
    legal_act_type: Optional[str] = None
    legal_domain: Optional[str] = None
    is_amendment: bool = False
    affects_multiple_acts: bool = False
    quality_score: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LegalStructure:
    """Extracted legal document structure."""
    chapters: List[Dict[str, Any]]
    articles: List[Dict[str, Any]]
    paragraphs: List[Dict[str, Any]]
    sections: List[Dict[str, Any]]
    references: List[Dict[str, Any]]


class TextProcessor:
    """Legal document text processing and content extraction."""
    
    def __init__(self, config: Optional[DocumentIngestionConfig] = None):
        self.config = config or get_ingestion_config()
        
        # Polish legal document patterns
        self.legal_patterns = {
            # Document types
            "ustawa": re.compile(r"ustawa\s+z\s+dnia\s+(\d{1,2}\s+\w+\s+\d{4})", re.IGNORECASE),
            "kodeks": re.compile(r"kodeks\s+(\w+)", re.IGNORECASE),
            "rozporządzenie": re.compile(r"rozporządzenie\s+(.+?)\s+z\s+dnia\s+(\d{1,2}\s+\w+\s+\d{4})", re.IGNORECASE),
            "konstytucja": re.compile(r"konstytucja\s+rzeczypospolitej\s+polskiej", re.IGNORECASE),
            
            # Legal structure
            "article": re.compile(r"art\.\s*(\d+[a-z]?)", re.IGNORECASE),
            "paragraph": re.compile(r"§\s*(\d+)", re.IGNORECASE),
            "section": re.compile(r"ust\.\s*(\d+)", re.IGNORECASE),
            "chapter": re.compile(r"rozdział\s+([IVX]+|\d+)", re.IGNORECASE),
            "point": re.compile(r"pkt\s+(\d+)", re.IGNORECASE),
            
            # References
            "law_reference": re.compile(r"ustawy?\s+z\s+dnia\s+(\d{1,2}\s+\w+\s+\d{4})", re.IGNORECASE),
            "article_reference": re.compile(r"art\.\s*(\d+[a-z]?)\s+ustawy", re.IGNORECASE),
            "regulation_reference": re.compile(r"rozporządzenia?\s+(.+?)\s+z\s+dnia\s+(\d{1,2}\s+\w+\s+\d{4})", re.IGNORECASE),
            
            # Dates
            "date": re.compile(r"(\d{1,2})\s+(\w+)\s+(\d{4})", re.IGNORECASE),
            "effective_date": re.compile(r"wchodzi\s+w\s+życie\s+(.+?)(?:\.|$)", re.IGNORECASE),
            
            # Amendments
            "amendment_marker": re.compile(r"nowelizacja|zmiana|uchylenie|dodanie|skreślenie", re.IGNORECASE),
            "omnibus_marker": re.compile(r"ustawa\s+o\s+zmianie\s+(.+?)\s+oraz\s+(.+)", re.IGNORECASE)
        }
        
        # Polish months mapping
        self.polish_months = {
            "stycznia": 1, "lutego": 2, "marca": 3, "kwietnia": 4,
            "maja": 5, "czerwca": 6, "lipca": 7, "sierpnia": 8,
            "września": 9, "października": 10, "listopada": 11, "grudnia": 12
        }
    
    def process_document(self, raw_content: str, eli_id: Optional[str] = None, source_url: Optional[str] = None) -> ProcessedDocument:
        """Process raw document content into structured format."""
        
        logger.info(f"Processing document: {eli_id}")
        
        # Clean and extract text
        cleaned_content = self._clean_html(raw_content)
        
        # Extract metadata
        metadata = self._extract_metadata(cleaned_content)
        
        # Determine document type
        document_type = self._identify_document_type(cleaned_content)
        
        # Extract title
        title = self._extract_title(cleaned_content)
        
        # Extract publication date
        published_at = self._extract_publication_date(cleaned_content)
        
        # Determine legal classification
        legal_act_type = self._classify_legal_act(cleaned_content, document_type)
        legal_domain = self._identify_legal_domain(cleaned_content)
        
        # Check if amendment
        is_amendment = self._is_amendment(cleaned_content)
        affects_multiple_acts = self._affects_multiple_acts(cleaned_content)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(cleaned_content, title, eli_id)
        
        document = ProcessedDocument(
            title=title,
            content=cleaned_content,
            document_type=document_type,
            eli_identifier=eli_id,
            source_url=source_url,
            published_at=published_at,
            legal_act_type=legal_act_type,
            legal_domain=legal_domain,
            is_amendment=is_amendment,
            affects_multiple_acts=affects_multiple_acts,
            quality_score=quality_score,
            metadata=metadata
        )
        
        logger.info(f"Processed document: {title} (quality: {quality_score:.2f})")
        return document
    
    def _clean_html(self, html_content: str) -> str:
        """Clean HTML content and extract text."""
        
        if not html_content.strip():
            return ""
        
        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean whitespace
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n\s*\n', '\n\n', text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"HTML cleaning failed: {e}")
            # Fallback to simple regex cleaning
            text = re.sub(r'<[^>]+>', '', html_content)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
    
    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from document content."""
        
        metadata = {}
        
        # Extract document structure
        if self.config.extract_structure:
            metadata['structure'] = self._extract_structure(content)
        
        # Extract references
        if self.config.extract_references:
            metadata['references'] = self._extract_references(content)
        
        # Content statistics
        metadata['content_stats'] = {
            'character_count': len(content),
            'word_count': len(content.split()),
            'paragraph_count': content.count('\n\n') + 1,
            'article_count': len(self.legal_patterns['article'].findall(content)),
            'section_count': len(self.legal_patterns['section'].findall(content))
        }
        
        return metadata
    
    def _extract_structure(self, content: str) -> LegalStructure:
        """Extract legal document structure."""
        
        structure = LegalStructure(
            chapters=[],
            articles=[],
            paragraphs=[],
            sections=[],
            references=[]
        )
        
        # Extract chapters
        for match in self.legal_patterns['chapter'].finditer(content):
            structure.chapters.append({
                'number': match.group(1),
                'position': match.start(),
                'text': content[match.start():match.start()+200].strip()
            })
        
        # Extract articles
        for match in self.legal_patterns['article'].finditer(content):
            structure.articles.append({
                'number': match.group(1),
                'position': match.start(),
                'text': content[match.start():match.start()+500].strip()
            })
        
        # Extract paragraphs
        for match in self.legal_patterns['paragraph'].finditer(content):
            structure.paragraphs.append({
                'number': match.group(1),
                'position': match.start(),
                'text': content[match.start():match.start()+300].strip()
            })
        
        # Extract sections
        for match in self.legal_patterns['section'].finditer(content):
            structure.sections.append({
                'number': match.group(1),
                'position': match.start(),
                'text': content[match.start():match.start()+400].strip()
            })
        
        return structure
    
    def _extract_references(self, content: str) -> List[Dict[str, Any]]:
        """Extract legal references from content."""
        
        references = []
        
        # Law references
        for match in self.legal_patterns['law_reference'].finditer(content):
            references.append({
                'type': 'law',
                'date': match.group(1),
                'position': match.start(),
                'text': match.group(0)
            })
        
        # Article references
        for match in self.legal_patterns['article_reference'].finditer(content):
            references.append({
                'type': 'article',
                'article': match.group(1),
                'position': match.start(),
                'text': match.group(0)
            })
        
        # Regulation references
        for match in self.legal_patterns['regulation_reference'].finditer(content):
            references.append({
                'type': 'regulation',
                'authority': match.group(1),
                'date': match.group(2),
                'position': match.start(),
                'text': match.group(0)
            })
        
        return references
    
    def _identify_document_type(self, content: str) -> str:
        """Identify the type of legal document."""
        
        content_lower = content.lower()
        
        # Check each document type pattern
        for doc_type, pattern in self.legal_patterns.items():
            if doc_type in ['ustawa', 'kodeks', 'rozporządzenie', 'konstytucja']:
                if pattern.search(content):
                    return doc_type
        
        # Fallback based on keywords
        if 'konstytucja' in content_lower:
            return 'konstytucja'
        elif 'kodeks' in content_lower:
            return 'kodeks'
        elif 'rozporządzenie' in content_lower:
            return 'rozporządzenie'
        elif 'ustawa' in content_lower:
            return 'ustawa'
        else:
            return 'unknown'
    
    def _extract_title(self, content: str) -> str:
        """Extract document title."""
        
        lines = content.split('\n')
        
        # Look for title in first few lines
        for line in lines[:10]:
            line = line.strip()
            if line and len(line) > 10:
                # Check if it looks like a title
                if any(keyword in line.lower() for keyword in ['ustawa', 'kodeks', 'rozporządzenie', 'konstytucja']):
                    return line[:500]  # Limit title length
        
        # Fallback to first substantial line
        for line in lines:
            line = line.strip()
            if len(line) > 20:
                return line[:500]
        
        return "Untitled Document"
    
    def _extract_publication_date(self, content: str) -> Optional[datetime]:
        """Extract publication date from content."""
        
        for match in self.legal_patterns['date'].finditer(content):
            try:
                day = int(match.group(1))
                month_name = match.group(2).lower()
                year = int(match.group(3))
                
                if month_name in self.polish_months:
                    month = self.polish_months[month_name]
                    return datetime(year, month, day)
                    
            except (ValueError, KeyError):
                continue
        
        return None
    
    def _classify_legal_act(self, content: str, document_type: str) -> str:
        """Classify the legal act type."""
        
        if document_type in ['ustawa', 'kodeks', 'rozporządzenie', 'konstytucja']:
            return document_type
        
        # Additional classification based on content
        content_lower = content.lower()
        
        if 'kodeks cywilny' in content_lower:
            return 'kodeks'
        elif 'kodeks karny' in content_lower:
            return 'kodeks'
        elif 'kodeks pracy' in content_lower:
            return 'kodeks'
        
        return document_type
    
    def _identify_legal_domain(self, content: str) -> str:
        """Identify legal domain/area."""
        
        content_lower = content.lower()
        
        # Domain keywords
        domains = {
            'civil': ['cywilny', 'własność', 'umowa', 'spadek', 'rodzina'],
            'criminal': ['karny', 'przestępstwo', 'kara', 'sąd karny'],
            'administrative': ['administracyjny', 'urząd', 'procedura administracyjna'],
            'commercial': ['handlowy', 'spółka', 'przedsiębiorstwo', 'gospodarcz'],
            'labor': ['pracy', 'pracownik', 'pracodawca', 'zatrudnienie'],
            'tax': ['podatkowy', 'podatek', 'vat', 'pit'],
            'constitutional': ['konstytucyjny', 'konstytucja', 'podstawowe prawa']
        }
        
        for domain, keywords in domains.items():
            if any(keyword in content_lower for keyword in keywords):
                return domain
        
        return 'general'
    
    def _is_amendment(self, content: str) -> bool:
        """Check if document is an amendment."""
        
        return bool(self.legal_patterns['amendment_marker'].search(content))
    
    def _affects_multiple_acts(self, content: str) -> bool:
        """Check if document affects multiple legal acts."""
        
        # Check for omnibus bill pattern
        if self.legal_patterns['omnibus_marker'].search(content):
            return True
        
        # Count different law references
        law_refs = self.legal_patterns['law_reference'].findall(content)
        return len(set(law_refs)) > 1
    
    def _calculate_quality_score(self, content: str, title: str, eli_id: Optional[str]) -> float:
        """Calculate document quality score."""
        
        score = 0.0
        
        # Content length check
        if self.config.min_text_length <= len(content) <= self.config.max_text_length:
            score += 0.2
        
        # Title quality
        if title and title != "Untitled Document" and len(title) > 10:
            score += 0.2
        
        # ELI identifier presence
        if eli_id:
            score += 0.2
        
        # Legal structure indicators
        if self.legal_patterns['article'].search(content):
            score += 0.2
        
        # Document type identification
        content_lower = content.lower()
        if any(doc_type in content_lower for doc_type in ['ustawa', 'kodeks', 'rozporządzenie']):
            score += 0.2
        
        return min(score, 1.0)
    
    def create_content_hash(self, content: str) -> str:
        """Create hash of document content for deduplication."""
        
        # Normalize content for hashing
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def validate_document(self, document: ProcessedDocument) -> Tuple[bool, List[str]]:
        """Validate processed document quality."""
        
        errors = []
        
        # Required fields
        if not document.title or document.title == "Untitled Document":
            errors.append("Missing or invalid title")
        
        if not document.content or len(document.content) < self.config.min_text_length:
            errors.append(f"Content too short (min: {self.config.min_text_length} chars)")
        
        if len(document.content) > self.config.max_text_length:
            errors.append(f"Content too long (max: {self.config.max_text_length} chars)")
        
        # ELI identifier requirement
        if self.config.require_eli_identifier and not document.eli_identifier:
            errors.append("Missing ELI identifier")
        
        # Quality score requirement
        if document.quality_score < self.config.min_quality_score:
            errors.append(f"Quality score too low: {document.quality_score:.2f} < {self.config.min_quality_score}")
        
        # Legal structure validation
        if self.config.validate_legal_structure:
            if document.document_type == 'unknown':
                errors.append("Could not identify document type")
            
            # Check for basic legal structure
            if not self.legal_patterns['article'].search(document.content):
                errors.append("No articles found in legal document")
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.warning(f"Document validation failed: {', '.join(errors)}")
        
        return is_valid, errors