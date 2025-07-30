"""Named entity recognition for Polish legal documents."""

import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

def _try_import_spacy():
    """Lazy import of spacy to avoid numpy compatibility issues."""
    try:
        import spacy
        return True, spacy
    except (ImportError, ValueError) as e:
        # ValueError can occur due to numpy compatibility issues
        return False, None


class LegalEntityType(Enum):
    """Types of legal entities."""
    LAW_REFERENCE = "LAW_REFERENCE"
    ARTICLE_REFERENCE = "ARTICLE_REFERENCE"
    PARAGRAPH_REFERENCE = "PARAGRAPH_REFERENCE"
    COURT_NAME = "COURT_NAME"
    LEGAL_PERSON = "LEGAL_PERSON"
    DATE = "DATE"
    MONEY = "MONEY"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    PERSON = "PERSON"


@dataclass
class LegalEntity:
    """Represents a legal entity found in text."""
    text: str
    entity_type: LegalEntityType
    start: int
    end: int
    confidence: float = 1.0
    metadata: Optional[Dict] = None


class PolishLegalNER:
    """Named Entity Recognition for Polish legal documents."""
    
    def __init__(self, model_name: str = "pl_core_news_sm"):
        self.model_name = model_name
        self.nlp = None
        self._spacy_available = None
        self._spacy = None
        
        # Legal reference patterns
        self.legal_patterns = {
            'law_reference': [
                r'ustawa\s+z\s+dnia\s+\d{1,2}\s+\w+\s+\d{4}\s+r\.',
                r'rozporządzenie\s+(?:Ministra|Rady\s+Ministrów)\s+z\s+dnia\s+\d{1,2}\s+\w+\s+\d{4}\s+r\.',
                r'kodeks\s+(?:cywilny|karny|postępowania\s+(?:cywilnego|karnego))',
                r'konstytucja\s+rzeczypospolitej\s+polskiej',
            ],
            
            'article_reference': [
                r'art\.\s*\d+(?:\s*ust\.\s*\d+)?(?:\s*pkt\s*\d+)?',
                r'artykuł\s+\d+(?:\s+ustęp\s+\d+)?(?:\s+punkt\s+\d+)?',
            ],
            
            'paragraph_reference': [
                r'§\s*\d+(?:\s*ust\.\s*\d+)?',
                r'paragraf\s+\d+(?:\s+ustęp\s+\d+)?',
            ],
            
            'court_name': [
                r'sąd\s+najwyższy',
                r'trybunał\s+konstytucyjny',
                r'naczelny\s+sąd\s+administracyjny',
                r'wojewódzki\s+sąd\s+administracyjny',
                r'sąd\s+(?:rejonowy|okręgowy|apelacyjny)',
                r'sąd\s+\w+\s+w\s+\w+',
            ],
            
            'legal_person': [
                r'spółka\s+(?:z\s+ograniczoną\s+odpowiedzialnością|akcyjna)',
                r'(?:sp\.\s*z\s*o\.o\.|s\.a\.)',
                r'fundacja\s+\w+',
                r'stowarzyszenie\s+\w+',
            ],
        }
        
        # Compile patterns
        self.compiled_patterns = {}
        for category, patterns in self.legal_patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def _ensure_spacy(self):
        """Ensure spacy is loaded if available."""
        if self._spacy_available is None:
            self._spacy_available, self._spacy = _try_import_spacy()
            
            if self._spacy_available:
                try:
                    self.nlp = self._spacy.load(self.model_name)
                except OSError:
                    self.nlp = None
        
        return self._spacy_available
    
    def extract_legal_references(self, text: str) -> List[LegalEntity]:
        """Extract legal references from text."""
        entities = []
        
        for category, patterns in self.compiled_patterns.items():
            entity_type = getattr(LegalEntityType, category.upper())
            
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entities.append(LegalEntity(
                        text=match.group(),
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end()
                    ))
        
        return entities
    
    def extract_spacy_entities(self, text: str) -> List[LegalEntity]:
        """Extract entities using spaCy NER."""
        entities = []
        
        if not self._ensure_spacy() or not self.nlp:
            return entities
        
        doc = self.nlp(text)
        
        # Map spaCy labels to legal entity types
        label_mapping = {
            'PER': LegalEntityType.PERSON,
            'ORG': LegalEntityType.ORGANIZATION,
            'LOC': LegalEntityType.LOCATION,
            'DATE': LegalEntityType.DATE,
            'MONEY': LegalEntityType.MONEY,
        }
        
        for ent in doc.ents:
            if ent.label_ in label_mapping:
                entities.append(LegalEntity(
                    text=ent.text,
                    entity_type=label_mapping[ent.label_],
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.8
                ))
        
        return entities
    
    def extract_all_entities(self, text: str) -> List[LegalEntity]:
        """Extract all entities from text."""
        if not text:
            return []
        
        entities = []
        entities.extend(self.extract_legal_references(text))
        entities.extend(self.extract_spacy_entities(text))
        
        # Remove overlapping entities (keep longer ones)
        entities = self._remove_overlaps(entities)
        
        return sorted(entities, key=lambda x: x.start)
    
    def _remove_overlaps(self, entities: List[LegalEntity]) -> List[LegalEntity]:
        """Remove overlapping entities, keeping the longer ones."""
        if not entities:
            return []
        
        # Sort by start position, then by length (descending)
        sorted_entities = sorted(entities, key=lambda x: (x.start, -(x.end - x.start)))
        
        non_overlapping = []
        for entity in sorted_entities:
            # Check if this entity overlaps with any already selected
            overlaps = False
            for selected in non_overlapping:
                if (entity.start < selected.end and entity.end > selected.start):
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping.append(entity)
        
        return non_overlapping
    
    def get_entity_statistics(self, entities: List[LegalEntity]) -> Dict[str, int]:
        """Get statistics about extracted entities."""
        stats = {}
        for entity in entities:
            entity_type = entity.entity_type.value
            stats[entity_type] = stats.get(entity_type, 0) + 1
        
        return stats


class LegalEntityExtractor:
    """High-level interface for legal entity extraction."""
    
    def __init__(self, model_name: str = "pl_core_news_sm"):
        self.ner = PolishLegalNER(model_name)
    
    def extract_entities(self, text: str) -> Dict[str, any]:
        """Extract entities and return structured results."""
        entities = self.ner.extract_all_entities(text)
        
        # Group entities by type
        grouped_entities = {}
        for entity in entities:
            entity_type = entity.entity_type.value
            if entity_type not in grouped_entities:
                grouped_entities[entity_type] = []
            grouped_entities[entity_type].append({
                'text': entity.text,
                'start': entity.start,
                'end': entity.end,
                'confidence': entity.confidence
            })
        
        return {
            'entities': [
                {
                    'text': entity.text,
                    'type': entity.entity_type.value,
                    'start': entity.start,
                    'end': entity.end,
                    'confidence': entity.confidence
                }
                for entity in entities
            ],
            'grouped_entities': grouped_entities,
            'statistics': self.ner.get_entity_statistics(entities)
        }