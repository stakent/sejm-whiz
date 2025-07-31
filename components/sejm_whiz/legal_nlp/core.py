"""Core legal NLP functionality for advanced text analysis."""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import re


class LegalConceptType(Enum):
    """Types of legal concepts that can be identified."""
    LEGAL_PRINCIPLE = "legal_principle"
    LEGAL_DEFINITION = "legal_definition"
    OBLIGATION = "obligation"
    PROHIBITION = "prohibition"
    RIGHT = "right"
    PENALTY = "penalty"
    PROCEDURE = "procedure"
    CONDITION = "condition"
    EXCEPTION = "exception"


@dataclass
class LegalConcept:
    """Represents a legal concept identified in text."""
    concept_type: LegalConceptType
    text: str
    start_pos: int
    end_pos: int
    confidence: float
    context: str = ""
    related_articles: List[str] = None
    
    def __post_init__(self):
        if self.related_articles is None:
            self.related_articles = []


@dataclass
class SemanticRelation:
    """Represents a semantic relationship between legal concepts."""
    source_concept: str
    target_concept: str
    relation_type: str
    confidence: float
    evidence_text: str = ""


@dataclass 
class LegalAmendment:
    """Represents a legal amendment or change."""
    amendment_type: str  # addition, modification, deletion
    target_provision: str
    original_text: str
    amended_text: str
    effective_date: Optional[str] = None
    rationale: str = ""


class LegalNLPAnalyzer:
    """Advanced NLP analyzer for Polish legal documents."""
    
    def __init__(self):
        """Initialize the legal NLP analyzer."""
        self._init_patterns()
        self._init_legal_vocabulary()
    
    def _init_patterns(self):
        """Initialize regex patterns for legal concept detection."""
        self.concept_patterns = {
            LegalConceptType.LEGAL_DEFINITION: [
                r'(?:oznacza|znaczy|rozumie się|definiuje się jako)\s+(.+?)(?:\.|;)',
                r'(?:w rozumieniu niniejszej ustawy|w rozumieniu przepisów)\s+(.+?)(?:\.|;)',
                r'(?:przez|pod pojęciem)\s+(.+?)\s+(?:rozumie się|należy rozumieć)'
            ],
            LegalConceptType.LEGAL_PRINCIPLE: [
                r'(?:jest|stanowi)\s+(?:państwem|zasadą|podstawą)\s+(.+?)(?:\.|;)',
                r'(?:zasada|podstawa|reguła|norma)\s+(.+?)(?:\.|;)',
                r'(?:urzeczywistnia|realizuje|gwarantuje)\s+(?:zasady|podstawy)\s+(.+?)(?:\.|;)',
                r'(?:władza|zwierzchnictwo|suwerenność)\s+(.+?)(?:\.|;)'
            ],
            LegalConceptType.OBLIGATION: [
                r'(?:jest obowiązany|ma obowiązek|powinien|zobowiązuje się)\s+(.+?)(?:\.|;)',
                r'(?:obowiązek|zobowiązanie)\s+(?:do\s+)?(.+?)(?:\.|;)',
                r'(?:należy|trzeba|konieczne jest)\s+(.+?)(?:\.|;)'
            ],
            LegalConceptType.PROHIBITION: [
                r'(?:zabrania się|nie wolno|nie można|zabronione jest)\s+(.+?)(?:\.|;)',
                r'(?:zakaz|prohibicja)\s+(.+?)(?:\.|;)',
                r'(?:nie ma prawa|nie jest uprawniony)\s+(.+?)(?:\.|;)'
            ],
            LegalConceptType.RIGHT: [
                r'(?:ma prawo|jest uprawniony|przysługuje mu prawo)\s+(.+?)(?:\.|;)',
                r'(?:prawo do|uprawnienie do)\s+(.+?)(?:\.|;)',
                r'(?:może|wolno mu)\s+(.+?)(?:\.|;)'
            ],
            LegalConceptType.PENALTY: [
                r'(?:podlega karze|karze się|grozi kara)\s+(.+?)(?:\.|;)',
                r'(?:kara|sankcja|grzywna)\s+(?:w wysokości\s+)?(.+?)(?:\.|;)',
                r'(?:zostanie ukarany|poniesie odpowiedzialność)\s+(.+?)(?:\.|;)'
            ],
            LegalConceptType.CONDITION: [
                r'(?:pod warunkiem|w przypadku gdy|jeżeli|gdy)\s+(.+?)(?:\.|;|,)',
                r'(?:warunek|przesłanka)\s+(.+?)(?:\.|;)',
                r'(?:wymaga się|konieczne jest spełnienie)\s+(.+?)(?:\.|;)'
            ],
            LegalConceptType.EXCEPTION: [
                r'(?:z wyjątkiem|nie dotyczy|nie ma zastosowania)\s+(.+?)(?:\.|;)',
                r'(?:wyjątek|odstępstwo)\s+(?:od\s+)?(.+?)(?:\.|;)',
                r'(?:za wyjątkiem|oprócz)\s+(.+?)(?:\.|;)'
            ]
        }
        
        # Amendment detection patterns
        self.amendment_patterns = {
            'addition': [
                r'(?:dodaje się|wprowadza się|dopisuje się)\s+(.+?)(?:\.|;)',
                r'(?:nowy|dodatkowy)\s+(?:artykuł|paragraf|punkt|ustęp)\s+(.+?)(?:\.|;)'
            ],
            'modification': [
                r'(?:zmienia się|modyfikuje się)\s+(.+?)\s+(?:na|przez)\s+(.+?)(?:\.|;)',
                r'(?:w miejsce|zamiast)\s+(.+?)\s+(?:wstawia się|wprowadza się)\s+(.+?)(?:\.|;)',
                r'(?:słowa|wyrazy|tekst)\s+[„"](.+?)[„"]\s+zastępuje się\s+(?:słowami|wyrazami|tekstem)\s+[„"](.+?)[„"]',
                r'(?:zastępuje się)\s+(.+?)\s+(?:słowami|wyrazami|tekstem)\s+(.+?)(?:\.|;)'
            ],
            'deletion': [
                r'(?:usuwa się|skreśla się|uchyla się)\s+(.+?)(?:\.|;)',
                r'(?:traci moc|zostaje uchylony)\s+(.+?)(?:\.|;)'
            ]
        }
        
        # Compile patterns
        self.compiled_concept_patterns = {}
        for concept_type, patterns in self.concept_patterns.items():
            self.compiled_concept_patterns[concept_type] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                for pattern in patterns
            ]
        
        self.compiled_amendment_patterns = {}
        for amendment_type, patterns in self.amendment_patterns.items():
            self.compiled_amendment_patterns[amendment_type] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                for pattern in patterns
            ]
    
    def _init_legal_vocabulary(self):
        """Initialize legal vocabulary and terminology."""
        self.legal_terms = {
            'procedural': [
                'postępowanie', 'procedura', 'tryb', 'wniosek', 'odwołanie',
                'skarga', 'protest', 'sprzeciw', 'rozpoznanie', 'rozstrzygnięcie'
            ],
            'institutional': [
                'sąd', 'trybunał', 'urząd', 'minister', 'wojewoda', 'starosta',
                'komisja', 'rada', 'organ', 'instytucja', 'jednostka'
            ],
            'temporal': [
                'termin', 'okres', 'czas', 'dzień', 'miesiąc', 'rok',
                'natychmiast', 'niezwłocznie', 'w terminie', 'do dnia'
            ],
            'conditional': [
                'warunek', 'przesłanka', 'wymóg', 'kryterium', 'podstawa',
                'uzasadnienie', 'powód', 'okoliczność', 'sytuacja'
            ]
        }
        
        # Legal connectors and discourse markers
        self.discourse_markers = {
            'addition': ['ponadto', 'dodatkowo', 'również', 'oraz', 'a także'],
            'contrast': ['jednak', 'natomiast', 'ale', 'lecz', 'jednakże'],
            'consequence': ['w związku z tym', 'w wyniku', 'skutkiem', 'dlatego'],
            'condition': ['jeżeli', 'gdy', 'w przypadku gdy', 'pod warunkiem'],
            'exception': ['z wyjątkiem', 'oprócz', 'za wyjątkiem', 'nie dotyczy']
        }
    
    def extract_legal_concepts(self, text: str, context_window: int = 100) -> List[LegalConcept]:
        """Extract legal concepts from text."""
        if not text:
            return []
        
        concepts = []
        
        for concept_type, patterns in self.compiled_concept_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Extract context around the match
                    context_start = max(0, start_pos - context_window)
                    context_end = min(len(text), end_pos + context_window)
                    context = text[context_start:context_end]
                    
                    # Calculate confidence based on pattern complexity and context
                    confidence = self._calculate_concept_confidence(match, context, concept_type)
                    
                    concept = LegalConcept(
                        concept_type=concept_type,
                        text=match.group(0),
                        start_pos=start_pos,
                        end_pos=end_pos,
                        confidence=confidence,
                        context=context.strip()
                    )
                    
                    concepts.append(concept)
        
        # Remove overlapping concepts, keeping the one with higher confidence
        concepts = self._remove_overlapping_concepts(concepts)
        
        return sorted(concepts, key=lambda x: x.start_pos)
    
    def _calculate_concept_confidence(self, match, context: str, concept_type: LegalConceptType) -> float:
        """Calculate confidence score for a legal concept."""
        base_confidence = 0.7
        
        # Boost confidence if legal terminology is present in context
        legal_term_boost = 0
        for category, terms in self.legal_terms.items():
            for term in terms:
                if term.lower() in context.lower():
                    legal_term_boost += 0.05
        
        # Boost confidence for specific concept types
        type_boost = {
            LegalConceptType.LEGAL_DEFINITION: 0.1,
            LegalConceptType.OBLIGATION: 0.05,
            LegalConceptType.PROHIBITION: 0.05,
            LegalConceptType.PENALTY: 0.08
        }.get(concept_type, 0)
        
        # Penalty for very short matches
        length_penalty = 0 if len(match.group(0)) > 20 else -0.1
        
        confidence = min(1.0, base_confidence + legal_term_boost + type_boost + length_penalty)
        return max(0.0, confidence)
    
    def _remove_overlapping_concepts(self, concepts: List[LegalConcept]) -> List[LegalConcept]:
        """Remove overlapping concepts, keeping those with higher confidence."""
        if not concepts:
            return []
        
        # Sort by start position
        sorted_concepts = sorted(concepts, key=lambda x: x.start_pos)
        filtered_concepts = []
        
        for concept in sorted_concepts:
            # Check if this concept overlaps with any already added concept
            overlaps = False
            for existing in filtered_concepts:
                if (concept.start_pos < existing.end_pos and 
                    concept.end_pos > existing.start_pos):
                    # There's an overlap
                    if concept.confidence <= existing.confidence:
                        overlaps = True
                        break
                    else:
                        # Remove the existing concept as this one has higher confidence
                        filtered_concepts.remove(existing)
                        break
            
            if not overlaps:
                filtered_concepts.append(concept)
        
        return filtered_concepts
    
    def detect_amendments(self, text: str) -> List[LegalAmendment]:
        """Detect legal amendments in text."""
        if not text:
            return []
        
        amendments = []
        
        for amendment_type, patterns in self.compiled_amendment_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    groups = match.groups()
                    
                    amendment = LegalAmendment(
                        amendment_type=amendment_type,
                        target_provision="",  # Will be extracted from context
                        original_text="",
                        amended_text="",
                        rationale=match.group(0)
                    )
                    
                    # Extract details based on amendment type
                    if amendment_type == 'modification' and len(groups) >= 2:
                        amendment.original_text = groups[0] if groups[0] else ""
                        amendment.amended_text = groups[1] if groups[1] else ""
                    elif len(groups) >= 1:
                        amendment.amended_text = groups[0] if groups[0] else ""
                    
                    amendments.append(amendment)
        
        return amendments
    
    def analyze_semantic_relations(self, concepts: List[LegalConcept]) -> List[SemanticRelation]:
        """Analyze semantic relationships between legal concepts."""
        if len(concepts) < 2:
            return []
        
        relations = []
        
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                # Skip if concepts are too far apart
                if abs(concept1.start_pos - concept2.start_pos) > 500:
                    continue
                
                # Determine relationship type
                relation_type = self._determine_relation_type(concept1, concept2)
                if relation_type:
                    confidence = self._calculate_relation_confidence(concept1, concept2, relation_type)
                    
                    relation = SemanticRelation(
                        source_concept=concept1.text,
                        target_concept=concept2.text,
                        relation_type=relation_type,
                        confidence=confidence,
                        evidence_text=self._extract_relation_evidence(concept1, concept2)
                    )
                    
                    relations.append(relation)
        
        return sorted(relations, key=lambda x: x.confidence, reverse=True)
    
    def _determine_relation_type(self, concept1: LegalConcept, concept2: LegalConcept) -> Optional[str]:
        """Determine the type of relationship between two concepts."""
        type1, type2 = concept1.concept_type, concept2.concept_type
        
        # Define relationship rules
        relation_rules = {
            (LegalConceptType.CONDITION, LegalConceptType.OBLIGATION): "conditional_obligation",
            (LegalConceptType.CONDITION, LegalConceptType.RIGHT): "conditional_right",
            (LegalConceptType.OBLIGATION, LegalConceptType.PENALTY): "violation_penalty",
            (LegalConceptType.PROHIBITION, LegalConceptType.PENALTY): "violation_penalty",
            (LegalConceptType.RIGHT, LegalConceptType.PROCEDURE): "exercise_procedure",
            (LegalConceptType.LEGAL_DEFINITION, LegalConceptType.OBLIGATION): "definition_application",
            (LegalConceptType.EXCEPTION, LegalConceptType.OBLIGATION): "exception_to_rule",
            (LegalConceptType.EXCEPTION, LegalConceptType.PROHIBITION): "exception_to_rule"
        }
        
        return relation_rules.get((type1, type2)) or relation_rules.get((type2, type1))
    
    def _calculate_relation_confidence(self, concept1: LegalConcept, concept2: LegalConcept, relation_type: str) -> float:
        """Calculate confidence for a semantic relation."""
        base_confidence = 0.6
        
        # Distance penalty - closer concepts are more likely to be related
        distance = abs(concept1.start_pos - concept2.start_pos)
        distance_penalty = min(0.3, distance / 1000)
        
        # Concept confidence boost
        concept_confidence_boost = (concept1.confidence + concept2.confidence) / 4
        
        # Relation type specific boosts
        type_boosts = {
            "violation_penalty": 0.2,
            "conditional_obligation": 0.15,
            "conditional_right": 0.15,
            "definition_application": 0.1
        }
        
        type_boost = type_boosts.get(relation_type, 0)
        
        confidence = base_confidence + concept_confidence_boost + type_boost - distance_penalty
        return max(0.0, min(1.0, confidence))
    
    def _extract_relation_evidence(self, concept1: LegalConcept, concept2: LegalConcept) -> str:
        """Extract text evidence for a semantic relation."""
        # Find the text span that covers both concepts
        start = min(concept1.start_pos, concept2.start_pos)
        end = max(concept1.end_pos, concept2.end_pos)
        
        # Expand context slightly
        start = max(0, start - 20)
        end = min(len(concept1.context), end + 20)
        
        return concept1.context[start:end].strip()
    
    def analyze_discourse_structure(self, text: str) -> Dict[str, Any]:
        """Analyze discourse structure and rhetorical patterns in legal text."""
        if not text:
            return {}
        
        discourse_analysis = {
            'discourse_markers': self._find_discourse_markers(text),
            'argument_structure': self._analyze_argument_structure(text),
            'temporal_structure': self._analyze_temporal_structure(text),
            'conditional_structure': self._analyze_conditional_structure(text)
        }
        
        return discourse_analysis
    
    def _find_discourse_markers(self, text: str) -> Dict[str, List[Dict]]:
        """Find discourse markers in text."""
        markers = {}
        
        for marker_type, marker_list in self.discourse_markers.items():
            markers[marker_type] = []
            
            for marker in marker_list:
                pattern = re.compile(r'\b' + re.escape(marker) + r'\b', re.IGNORECASE)
                for match in pattern.finditer(text):
                    markers[marker_type].append({
                        'marker': match.group(0),
                        'position': match.start(),
                        'context': text[max(0, match.start()-50):match.end()+50]
                    })
        
        return markers
    
    def _analyze_argument_structure(self, text: str) -> Dict[str, Any]:
        """Analyze argumentative structure of legal text."""
        # Look for argumentative patterns
        argument_patterns = {
            'premise': [r'(?:ponieważ|gdyż|bowiem|albowiem)\s+(.+?)(?:\.|;)'],
            'conclusion': [r'(?:dlatego|w związku z tym|stąd)\s+(.+?)(?:\.|;)'],
            'justification': [r'(?:uzasadnienie|podstawa|powód)\s+(.+?)(?:\.|;)']
        }
        
        arguments = {}
        for arg_type, patterns in argument_patterns.items():
            arguments[arg_type] = []
            for pattern in patterns:
                compiled_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                for match in compiled_pattern.finditer(text):
                    arguments[arg_type].append({
                        'text': match.group(1) if match.groups() else match.group(0),
                        'position': match.start(),
                        'full_match': match.group(0)
                    })
        
        return arguments
    
    def _analyze_temporal_structure(self, text: str) -> List[Dict]:
        """Analyze temporal expressions and their relationships."""
        temporal_patterns = [
            r'(?:od dnia|do dnia|w dniu|dnia)\s+(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{4})',
            r'(?:w terminie|do terminu)\s+(\d+\s+(?:dni|miesięcy|lat))',
            r'(?:natychmiast|niezwłocznie|bez zwłoki)',
            r'(?:wchodzi w życie|traci moc|obowiązuje od)\s+(.+?)(?:\.|;)'
        ]
        
        temporal_expressions = []
        for pattern in temporal_patterns:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            for match in compiled_pattern.finditer(text):
                temporal_expressions.append({
                    'expression': match.group(0),
                    'position': match.start(),
                    'type': 'temporal_reference'
                })
        
        return temporal_expressions
    
    def _analyze_conditional_structure(self, text: str) -> List[Dict]:
        """Analyze conditional structures in legal text."""
        conditional_patterns = [
            r'(?:jeżeli|gdy|w przypadku gdy)\s+(.+?)(?:,\s*to\s*)?(.+?)(?:\.|;)',
            r'(?:pod warunkiem)\s+(.+?)(?:\.|;)',
            r'(?:w sytuacji gdy|w okolicznościach gdy)\s+(.+?)(?:\.|;)'
        ]
        
        conditionals = []
        for pattern in conditional_patterns:
            compiled_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for match in compiled_pattern.finditer(text):
                groups = match.groups()
                conditionals.append({
                    'condition': groups[0] if groups and groups[0] else '',
                    'consequence': groups[1] if len(groups) > 1 and groups[1] else '',
                    'full_text': match.group(0),
                    'position': match.start()
                })
        
        return conditionals
    
    def perform_comprehensive_analysis(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive legal NLP analysis."""
        if not text:
            return {}
        
        # Extract concepts
        concepts = self.extract_legal_concepts(text)
        
        # Analyze relations
        relations = self.analyze_semantic_relations(concepts)
        
        # Detect amendments
        amendments = self.detect_amendments(text)
        
        # Analyze discourse
        discourse = self.analyze_discourse_structure(text)
        
        return {
            'legal_concepts': {
                'count': len(concepts),
                'concepts': [
                    {
                        'type': concept.concept_type.value,
                        'text': concept.text,
                        'confidence': concept.confidence,
                        'position': {'start': concept.start_pos, 'end': concept.end_pos}
                    }
                    for concept in concepts
                ],
                'by_type': self._group_concepts_by_type(concepts)
            },
            'semantic_relations': {
                'count': len(relations),
                'relations': [
                    {
                        'source': relation.source_concept,
                        'target': relation.target_concept,
                        'type': relation.relation_type,
                        'confidence': relation.confidence
                    }
                    for relation in relations
                ]
            },
            'amendments': {
                'count': len(amendments),
                'amendments': [
                    {
                        'type': amendment.amendment_type,
                        'target': amendment.target_provision,
                        'original': amendment.original_text,
                        'amended': amendment.amended_text
                    }
                    for amendment in amendments
                ]
            },
            'discourse_structure': discourse,
            'analysis_summary': {
                'complexity_score': self._calculate_complexity_score(concepts, relations, discourse),
                'main_concepts': self._identify_main_concepts(concepts),
                'key_relations': self._identify_key_relations(relations)
            }
        }
    
    def _group_concepts_by_type(self, concepts: List[LegalConcept]) -> Dict[str, int]:
        """Group concepts by type and count them."""
        type_counts = {}
        for concept in concepts:
            concept_type = concept.concept_type.value
            type_counts[concept_type] = type_counts.get(concept_type, 0) + 1
        return type_counts
    
    def _calculate_complexity_score(self, concepts: List[LegalConcept], relations: List[SemanticRelation], discourse: Dict) -> float:
        """Calculate a complexity score for the legal text."""
        # Base score from concept density
        concept_score = min(1.0, len(concepts) / 50)
        
        # Relation complexity
        relation_score = min(1.0, len(relations) / 20)
        
        # Discourse complexity (presence of multiple discourse marker types)
        discourse_markers = discourse.get('discourse_markers', {})
        discourse_score = min(1.0, len([t for t, m in discourse_markers.items() if m]) / 5)
        
        # Weighted average
        complexity = (concept_score * 0.4 + relation_score * 0.4 + discourse_score * 0.2)
        return round(complexity, 2)
    
    def _identify_main_concepts(self, concepts: List[LegalConcept]) -> List[str]:
        """Identify the main legal concepts based on confidence and frequency."""
        if not concepts:
            return []
        
        # Sort by confidence and take top concepts
        sorted_concepts = sorted(concepts, key=lambda x: x.confidence, reverse=True)
        main_concepts = [concept.concept_type.value for concept in sorted_concepts[:5]]
        
        return list(set(main_concepts))  # Remove duplicates while preserving order somewhat
    
    def _identify_key_relations(self, relations: List[SemanticRelation]) -> List[str]:
        """Identify key semantic relations."""
        if not relations:
            return []
        
        # Sort by confidence and take top relations
        sorted_relations = sorted(relations, key=lambda x: x.confidence, reverse=True)
        key_relations = [relation.relation_type for relation in sorted_relations[:3]]
        
        return list(set(key_relations))