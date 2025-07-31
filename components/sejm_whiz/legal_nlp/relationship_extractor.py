"""Extract and analyze relationships between legal entities and concepts."""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from collections import defaultdict


class RelationshipType(Enum):
    """Types of relationships between legal entities."""

    DEFINES = "defines"
    MODIFIES = "modifies"
    REFERENCES = "references"
    SUPERSEDES = "supersedes"
    IMPLEMENTS = "implements"
    DELEGATES = "delegates"
    REQUIRES = "requires"
    PROHIBITS = "prohibits"
    PERMITS = "permits"
    ESTABLISHES = "establishes"
    REPEALS = "repeals"
    EXTENDS = "extends"
    LIMITS = "limits"
    CONDITIONAL_ON = "conditional_on"
    APPLIES_TO = "applies_to"


@dataclass
class LegalEntity:
    """Represents a legal entity (person, institution, concept, etc.)."""

    entity_type: str
    name: str
    aliases: List[str] = None
    attributes: Dict[str, str] = None
    position: Tuple[int, int] = None  # (start, end) in text

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.attributes is None:
            self.attributes = {}


@dataclass
class LegalRelationship:
    """Represents a relationship between legal entities."""

    source_entity: LegalEntity
    target_entity: LegalEntity
    relationship_type: RelationshipType
    confidence: float
    evidence_text: str
    context: str = ""
    modifiers: List[str] = None
    temporal_info: Optional[str] = None

    def __post_init__(self):
        if self.modifiers is None:
            self.modifiers = []


class LegalRelationshipExtractor:
    """Extract relationships between legal entities in Polish legal texts."""

    def __init__(self):
        """Initialize the relationship extractor."""
        self._init_entity_patterns()
        self._init_relationship_patterns()
        self._init_legal_vocabulary()

    def _init_entity_patterns(self):
        """Initialize patterns for detecting legal entities."""
        self.entity_patterns = {
            "person": [
                r"(?:osoba|osoby)\s+(?:fizyczna|fizyczne|prawna|prawne)",
                r"(?:obywatel|obywatele)\s+(?:polski|polscy|polskich)",
                r"(?:podmiot|podmioty)\s+(?:gospodarczy|gospodarcze)",
                r"(?:przedsiębiorca|przedsiębiorcy)",
                r"(?:konsument|konsumenci)",
                r"(?:pracownik|pracownicy|pracodawca|pracodawcy)",
            ],
            "institution": [
                r"(?:sąd|sądy)\s+(?:rejonowy|okręgowy|apelacyjny|najwyższy)",
                r"(?:trybunał|trybunały)\s+(?:konstytucyjny|stanu|administracyjny)",
                r"(?:minister|ministra|ministerstwo)",
                r"(?:urząd|urzędy)\s+(?:gminy|miasta|powiatu|wojewódzki)",
                r"(?:rada|rady)\s+(?:gminy|miasta|powiatu|ministrów)",
                r"(?:komisja|komisje)\s+(?:europejska|nadzoru|kontroli)",
                r"(?:prokurator|prokuratura)",
                r"(?:policja|straż)\s+(?:miejska|pożarna|graniczna)",
            ],
            "legal_act": [
                r"(?:ustawa|ustawy)\s+(?:z dnia\s+\d+|\w+)",
                r"(?:rozporządzenie|rozporządzenia)\s+(?:ministra|rady ministrów)",
                r"(?:kodeks|kodeksy)\s+(?:cywilny|karny|pracy|postępowania)",
                r"(?:konstytucja|konstytucji)\s+(?:rzeczypospolitej polskiej)",
                r"(?:traktat|traktaty|konwencja|konwencje)",
                r"(?:dyrektywa|dyrektywy|rozporządzenie|rozporządzenia)\s+(?:unii europejskiej|ue)",
            ],
            "legal_concept": [
                r"(?:prawo|prawa)\s+(?:własności|autorskie|administracyjne|konstytucyjne)",
                r"(?:zobowiązanie|zobowiązania)\s+(?:umowne|pozaumowne)",
                r"(?:odpowiedzialność|odpowiedzialności)\s+(?:cywilna|karna|administracyjna)",
                r"(?:postępowanie|postępowania)\s+(?:cywilne|karne|administracyjne)",
                r"(?:orzeczenie|orzeczenia|wyrok|wyroki)",
                r"(?:umowa|umowy|kontrakt|kontrakty)",
                r"(?:decyzja|decyzje)\s+(?:administracyjna|administracyjne)",
            ],
            "temporal_entity": [
                r"(?:termin|terminy)\s+(?:płatności|wykonania|przedawnienia)",
                r"(?:okres|okresy)\s+(?:próbny|wypowiedzenia|ochronny)",
                r"(?:data|daty)\s+(?:wejścia w życie|uchylenia|zmiany)",
            ],
        }

        # Compile patterns
        self.compiled_entity_patterns = {}
        for entity_type, patterns in self.entity_patterns.items():
            self.compiled_entity_patterns[entity_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def _init_relationship_patterns(self):
        """Initialize patterns for detecting relationships."""
        self.relationship_patterns = {
            RelationshipType.DEFINES: [
                r"(.+?)\s+(?:oznacza|definiuje się jako|to)\s+(.+?)(?:\.|;)",
                r"(?:przez|pod pojęciem)\s+(.+?)\s+(?:rozumie się|należy rozumieć)\s+(.+?)(?:\.|;)",
            ],
            RelationshipType.MODIFIES: [
                r"(.+?)\s+(?:zmienia|modyfikuje|nowelizuje)\s+(.+?)(?:\.|;)",
                r"(?:w|we)\s+(.+?)\s+(?:wprowadza się zmiany|dokonuje się zmian)\s+(.+?)(?:\.|;)",
            ],
            RelationshipType.REFERENCES: [
                r"(.+?)\s+(?:odnosi się do|nawiązuje do|powołuje się na)\s+(.+?)(?:\.|;)",
                r"(?:zgodnie z|na podstawie|w świetle)\s+(.+?)\s+(.+?)(?:\.|;)",
                r"(.+?)\s+(?:o którym mowa w|określony w|wskazany w)\s+(.+?)(?:\.|;)",
            ],
            RelationshipType.SUPERSEDES: [
                r"(.+?)\s+(?:zastępuje|uchyla|znosi)\s+(.+?)(?:\.|;)",
                r"(?:w miejsce|zamiast)\s+(.+?)\s+(?:wprowadza się|ustala się)\s+(.+?)(?:\.|;)",
            ],
            RelationshipType.IMPLEMENTS: [
                r"(.+?)\s+(?:wykonuje|realizuje|wdraża)\s+(.+?)(?:\.|;)",
                r"(?:w wykonaniu|dla realizacji)\s+(.+?)\s+(.+?)(?:\.|;)",
            ],
            RelationshipType.DELEGATES: [
                r"(.+?)\s+(?:upoważnia|deleguje|przekazuje uprawnienia)\s+(.+?)(?:\.|;)",
                r"(?:uprawnienie do|kompetencja do)\s+(.+?)\s+(?:przysługuje|należy do)\s+(.+?)(?:\.|;)",
            ],
            RelationshipType.REQUIRES: [
                r"(.+?)\s+(?:wymaga|potrzebuje|jest uzależnione od)\s+(.+?)(?:\.|;)",
                r"(?:warunkiem|przesłanką)\s+(.+?)\s+(?:jest|stanowi)\s+(.+?)(?:\.|;)",
            ],
            RelationshipType.PROHIBITS: [
                r"(.+?)\s+(?:zabrania|zakazuje|nie pozwala na)\s+(.+?)(?:\.|;)",
                r"(?:zakaz|prohibicja)\s+(.+?)\s+(?:dotyczy|obejmuje)\s+(.+?)(?:\.|;)",
            ],
            RelationshipType.PERMITS: [
                r"(.+?)\s+(?:pozwala|zezwala|umożliwia)\s+(.+?)(?:\.|;)",
                r"(?:możliwość|prawo do)\s+(.+?)\s+(?:przysługuje|należy do)\s+(.+?)(?:\.|;)",
            ],
            RelationshipType.ESTABLISHES: [
                r"(.+?)\s+(?:ustanawia|tworzy|powołuje)\s+(.+?)(?:\.|;)",
                r"(?:powstaje|tworzy się|ustanawia się)\s+(.+?)\s+(?:w ramach|przez)\s+(.+?)(?:\.|;)",
            ],
            RelationshipType.REPEALS: [
                r"(.+?)\s+(?:uchyla się|traci moc|zostaje zniesiony)\s+(?:przez|w związku z)\s+(.+?)(?:\.|;)",
                r"(?:uchyleniu|zniesieniu)\s+(.+?)\s+(?:służy|dokonuje)\s+(.+?)(?:\.|;)",
            ],
            RelationshipType.EXTENDS: [
                r"(.+?)\s+(?:rozszerza|poszerza|wydłuża)\s+(.+?)(?:\.|;)",
                r"(?:rozszerzenie|poszerzenie)\s+(.+?)\s+(?:następuje przez|dokonuje się poprzez)\s+(.+?)(?:\.|;)",
            ],
            RelationshipType.LIMITS: [
                r"(.+?)\s+(?:ogranicza|zawę ża|limituje)\s+(.+?)(?:\.|;)",
                r"(?:ograniczenie|zawężenie)\s+(.+?)\s+(?:następuje przez|wynika z)\s+(.+?)(?:\.|;)",
            ],
            RelationshipType.CONDITIONAL_ON: [
                r"(.+?)\s+(?:pod warunkiem|w zależności od|uzależnione od)\s+(.+?)(?:\.|;)",
                r"(?:jeżeli|gdy|w przypadku gdy)\s+(.+?)\s+(?:to|wówczas)\s+(.+?)(?:\.|;)",
            ],
            RelationshipType.APPLIES_TO: [
                r"(.+?)\s+(?:dotyczy|odnosi się do|ma zastosowanie do)\s+(.+?)(?:\.|;)",
                r"(?:zakres|zasięg)\s+(.+?)\s+(?:obejmuje|dotyczy)\s+(.+?)(?:\.|;)",
            ],
        }

        # Compile relationship patterns
        self.compiled_relationship_patterns = {}
        for rel_type, patterns in self.relationship_patterns.items():
            self.compiled_relationship_patterns[rel_type] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                for pattern in patterns
            ]

    def _init_legal_vocabulary(self):
        """Initialize legal vocabulary for context analysis."""
        self.legal_vocabulary = {
            "modifiers": [
                "bezwarunkowo",
                "warunkowo",
                "tymczasowo",
                "ostatecznie",
                "natychmiast",
                "niezwłocznie",
                "stopniowo",
                "częściowo",
                "całkowicie",
                "w pełni",
                "ograniczenie",
                "wyjątkowo",
            ],
            "temporal_markers": [
                "od dnia",
                "do dnia",
                "w dniu",
                "z dniem",
                "w terminie",
                "przed upływem",
                "po upływie",
                "w ciągu",
                "nie później niż",
                "nie wcześniej niż",
                "równocześnie",
                "jednocześnie",
            ],
            "legal_operators": [
                "zgodnie z",
                "na podstawie",
                "w związku z",
                "w świetle",
                "z zastrzeżeniem",
                "z wyłączeniem",
                "z wyjątkiem",
                "pod rygorem",
                "w granicach",
                "w zakresie",
            ],
        }

    def extract_entities(self, text: str) -> List[LegalEntity]:
        """Extract legal entities from text."""
        if not text:
            return []

        entities = []

        for entity_type, patterns in self.compiled_entity_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entity = LegalEntity(
                        entity_type=entity_type,
                        name=match.group(0),
                        position=(match.start(), match.end()),
                    )
                    entities.append(entity)

        # Remove overlapping entities
        entities = self._remove_overlapping_entities(entities)

        # Add additional attributes based on context
        entities = self._enrich_entities(entities, text)

        return sorted(entities, key=lambda x: x.position[0])

    def _remove_overlapping_entities(
        self, entities: List[LegalEntity]
    ) -> List[LegalEntity]:
        """Remove overlapping entities, keeping the most specific ones."""
        if not entities:
            return []

        # Sort by position
        sorted_entities = sorted(entities, key=lambda x: x.position[0])
        filtered_entities = []

        for entity in sorted_entities:
            # Check for overlap with existing entities
            overlaps = False
            for existing in filtered_entities:
                if (
                    entity.position[0] < existing.position[1]
                    and entity.position[1] > existing.position[0]
                ):
                    # Overlap detected - keep the longer/more specific entity
                    entity_length = entity.position[1] - entity.position[0]
                    existing_length = existing.position[1] - existing.position[0]

                    if entity_length <= existing_length:
                        overlaps = True
                        break
                    else:
                        # Remove the existing shorter entity
                        filtered_entities.remove(existing)
                        break

            if not overlaps:
                filtered_entities.append(entity)

        return filtered_entities

    def _enrich_entities(
        self, entities: List[LegalEntity], text: str
    ) -> List[LegalEntity]:
        """Enrich entities with additional attributes from context."""
        for entity in entities:
            start, end = entity.position

            # Extract context around entity
            context_start = max(0, start - 100)
            context_end = min(len(text), end + 100)
            context = text[context_start:context_end]

            # Look for modifiers
            modifiers = []
            for modifier in self.legal_vocabulary["modifiers"]:
                if modifier.lower() in context.lower():
                    modifiers.append(modifier)

            # Look for temporal information
            temporal_info = None
            for temporal_marker in self.legal_vocabulary["temporal_markers"]:
                if temporal_marker.lower() in context.lower():
                    temporal_info = temporal_marker
                    break

            # Add attributes
            entity.attributes["modifiers"] = modifiers
            entity.attributes["temporal_info"] = temporal_info
            entity.attributes["context"] = context.strip()

        return entities

    def extract_relationships(
        self, text: str, entities: Optional[List[LegalEntity]] = None
    ) -> List[LegalRelationship]:
        """Extract relationships between legal entities."""
        if not text:
            return []

        if entities is None:
            entities = self.extract_entities(text)

        relationships = []

        # Extract relationships using patterns
        for rel_type, patterns in self.compiled_relationship_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    groups = match.groups()

                    if len(groups) >= 2:
                        source_text = groups[0].strip()
                        target_text = groups[1].strip()

                        # Try to match with existing entities
                        source_entity = self._find_matching_entity(
                            source_text, entities
                        )
                        target_entity = self._find_matching_entity(
                            target_text, entities
                        )

                        # Create entities if not found
                        if not source_entity:
                            source_entity = LegalEntity(
                                entity_type="concept",
                                name=source_text,
                                position=(match.start(1), match.end(1)),
                            )

                        if not target_entity:
                            target_entity = LegalEntity(
                                entity_type="concept",
                                name=target_text,
                                position=(match.start(2), match.end(2)),
                            )

                        # Extract context and modifiers
                        context = self._extract_relationship_context(
                            text, match.start(), match.end()
                        )
                        modifiers = self._extract_modifiers(context)
                        temporal_info = self._extract_temporal_info(context)

                        # Calculate confidence
                        confidence = self._calculate_relationship_confidence(
                            match, context, rel_type, source_entity, target_entity
                        )

                        relationship = LegalRelationship(
                            source_entity=source_entity,
                            target_entity=target_entity,
                            relationship_type=rel_type,
                            confidence=confidence,
                            evidence_text=match.group(0),
                            context=context,
                            modifiers=modifiers,
                            temporal_info=temporal_info,
                        )

                        relationships.append(relationship)

        # Remove low-confidence relationships
        relationships = [rel for rel in relationships if rel.confidence >= 0.3]

        return sorted(relationships, key=lambda x: x.confidence, reverse=True)

    def _find_matching_entity(
        self, text: str, entities: List[LegalEntity]
    ) -> Optional[LegalEntity]:
        """Find an entity that matches the given text."""
        text_lower = text.lower().strip()

        for entity in entities:
            entity_name_lower = entity.name.lower().strip()

            # Exact match
            if text_lower == entity_name_lower:
                return entity

            # Partial match (if one contains the other)
            if text_lower in entity_name_lower or entity_name_lower in text_lower:
                return entity

            # Check aliases
            for alias in entity.aliases:
                if text_lower == alias.lower() or alias.lower() in text_lower:
                    return entity

        return None

    def _extract_relationship_context(
        self, text: str, start: int, end: int, window: int = 150
    ) -> str:
        """Extract context around a relationship."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()

    def _extract_modifiers(self, context: str) -> List[str]:
        """Extract modifiers from relationship context."""
        modifiers = []
        context_lower = context.lower()

        for modifier in self.legal_vocabulary["modifiers"]:
            if modifier.lower() in context_lower:
                modifiers.append(modifier)

        return modifiers

    def _extract_temporal_info(self, context: str) -> Optional[str]:
        """Extract temporal information from relationship context."""
        context_lower = context.lower()

        for temporal_marker in self.legal_vocabulary["temporal_markers"]:
            if temporal_marker.lower() in context_lower:
                return temporal_marker

        return None

    def _calculate_relationship_confidence(
        self,
        match,
        context: str,
        rel_type: RelationshipType,
        source_entity: LegalEntity,
        target_entity: LegalEntity,
    ) -> float:
        """Calculate confidence score for a relationship."""
        base_confidence = 0.6

        # Pattern match quality
        pattern_boost = 0.1 if len(match.group(0)) > 30 else 0.0

        # Context quality (presence of legal vocabulary)
        context_boost = 0
        context_lower = context.lower()
        for category, terms in self.legal_vocabulary.items():
            for term in terms:
                if term.lower() in context_lower:
                    context_boost += 0.02

        context_boost = min(0.2, context_boost)

        # Entity type compatibility
        entity_boost = 0
        if (
            source_entity.entity_type != "concept"
            and target_entity.entity_type != "concept"
        ):
            entity_boost = 0.1

        # Relationship type specific boosts
        type_boosts = {
            RelationshipType.DEFINES: 0.15,
            RelationshipType.MODIFIES: 0.1,
            RelationshipType.REFERENCES: 0.05,
            RelationshipType.SUPERSEDES: 0.12,
            RelationshipType.REQUIRES: 0.08,
        }

        type_boost = type_boosts.get(rel_type, 0)

        confidence = (
            base_confidence + pattern_boost + context_boost + entity_boost + type_boost
        )
        return max(0.0, min(1.0, confidence))

    def analyze_relationship_network(
        self, relationships: List[LegalRelationship]
    ) -> Dict[str, Any]:
        """Analyze the network of relationships between legal entities."""
        if not relationships:
            return {}

        # Build entity frequency map
        entity_frequency = defaultdict(int)
        relationship_types = defaultdict(int)

        for rel in relationships:
            entity_frequency[rel.source_entity.name] += 1
            entity_frequency[rel.target_entity.name] += 1
            relationship_types[rel.relationship_type.value] += 1

        # Find central entities (high frequency)
        central_entities = sorted(
            entity_frequency.items(), key=lambda x: x[1], reverse=True
        )[:10]

        # Analyze relationship patterns
        relationship_patterns = self._analyze_relationship_patterns(relationships)

        # Calculate network metrics
        network_metrics = {
            "total_relationships": len(relationships),
            "unique_entities": len(entity_frequency),
            "relationship_density": len(relationships)
            / max(1, len(entity_frequency) ** 2),
            "average_confidence": sum(rel.confidence for rel in relationships)
            / len(relationships),
        }

        return {
            "network_metrics": network_metrics,
            "central_entities": central_entities,
            "relationship_type_distribution": dict(relationship_types),
            "relationship_patterns": relationship_patterns,
            "high_confidence_relationships": [
                {
                    "source": rel.source_entity.name,
                    "target": rel.target_entity.name,
                    "type": rel.relationship_type.value,
                    "confidence": rel.confidence,
                }
                for rel in relationships[:10]  # Top 10 by confidence
            ],
        }

    def _analyze_relationship_patterns(
        self, relationships: List[LegalRelationship]
    ) -> Dict[str, Any]:
        """Analyze patterns in the relationship network."""
        patterns = {"chains": [], "cycles": [], "hubs": [], "clusters": []}

        # Build adjacency map
        adjacency = defaultdict(list)
        for rel in relationships:
            adjacency[rel.source_entity.name].append(
                {
                    "target": rel.target_entity.name,
                    "type": rel.relationship_type.value,
                    "confidence": rel.confidence,
                }
            )

        # Find hubs (entities with many outgoing relationships)
        for entity, connections in adjacency.items():
            if len(connections) >= 3:
                patterns["hubs"].append(
                    {
                        "entity": entity,
                        "connection_count": len(connections),
                        "connections": connections[:5],  # Top 5 connections
                    }
                )

        # Find chains (A -> B -> C patterns)
        for entity, connections in adjacency.items():
            for connection in connections:
                target = connection["target"]
                if target in adjacency:
                    # Found a chain
                    patterns["chains"].append(
                        {
                            "chain": [entity, target],
                            "extended_to": list(adjacency[target])[
                                :3
                            ],  # Show next 3 connections
                        }
                    )

        return patterns

    def perform_comprehensive_relationship_analysis(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive relationship analysis of legal text."""
        if not text:
            return {}

        # Extract entities and relationships
        entities = self.extract_entities(text)
        relationships = self.extract_relationships(text, entities)

        # Analyze network
        network_analysis = self.analyze_relationship_network(relationships)

        return {
            "entities": {
                "count": len(entities),
                "by_type": self._group_entities_by_type(entities),
                "entities": [
                    {
                        "type": entity.entity_type,
                        "name": entity.name,
                        "attributes": entity.attributes,
                    }
                    for entity in entities[:20]  # Limit to first 20
                ],
            },
            "relationships": {
                "count": len(relationships),
                "by_type": {
                    rel_type.value: len(
                        [r for r in relationships if r.relationship_type == rel_type]
                    )
                    for rel_type in RelationshipType
                },
                "relationships": [
                    {
                        "source": rel.source_entity.name,
                        "target": rel.target_entity.name,
                        "type": rel.relationship_type.value,
                        "confidence": rel.confidence,
                        "modifiers": rel.modifiers,
                    }
                    for rel in relationships[:15]  # Top 15 relationships
                ],
            },
            "network_analysis": network_analysis,
        }

    def _group_entities_by_type(self, entities: List[LegalEntity]) -> Dict[str, int]:
        """Group entities by type and count them."""
        type_counts = defaultdict(int)
        for entity in entities:
            type_counts[entity.entity_type] += 1
        return dict(type_counts)
