"""Legal NLP component for advanced analysis of Polish legal documents."""

from .core import (
    LegalNLPAnalyzer,
    LegalConcept,
    LegalConceptType,
    SemanticRelation,
    LegalAmendment,
)
from .semantic_analyzer import LegalSemanticAnalyzer, SemanticField
from .relationship_extractor import (
    LegalRelationshipExtractor,
    LegalEntity,
    LegalRelationship,
    RelationshipType,
)


# Main analyzer class that combines all functionality
class ComprehensiveLegalAnalyzer:
    """Comprehensive legal document analyzer combining all NLP capabilities."""

    def __init__(self):
        """Initialize all analyzers."""
        self.nlp_analyzer = LegalNLPAnalyzer()
        self.semantic_analyzer = LegalSemanticAnalyzer()
        self.relationship_extractor = LegalRelationshipExtractor()

    def analyze_document(self, text: str) -> dict:
        """Perform comprehensive analysis of a legal document."""
        if not text:
            return {}

        # Core NLP analysis
        nlp_results = self.nlp_analyzer.perform_comprehensive_analysis(text)

        # Semantic analysis
        semantic_results = (
            self.semantic_analyzer.perform_comprehensive_semantic_analysis(text)
        )

        # Relationship analysis
        relationship_results = (
            self.relationship_extractor.perform_comprehensive_relationship_analysis(
                text
            )
        )

        return {
            "legal_nlp_analysis": nlp_results,
            "semantic_analysis": semantic_results,
            "relationship_analysis": relationship_results,
            "document_metadata": {
                "text_length": len(text),
                "word_count": len(text.split()),
                "analysis_components": ["nlp", "semantic", "relationships"],
            },
        }

    def extract_key_insights(self, text: str) -> dict:
        """Extract key insights from legal document analysis."""
        analysis = self.analyze_document(text)

        # Extract top insights
        insights = {
            "main_legal_concepts": [],
            "key_relationships": [],
            "semantic_fields": [],
            "amendments": [],
            "complexity_indicators": {},
        }

        # Extract main concepts
        if "legal_nlp_analysis" in analysis:
            nlp_data = analysis["legal_nlp_analysis"]
            if "legal_concepts" in nlp_data:
                concepts = nlp_data["legal_concepts"].get("concepts", [])
                insights["main_legal_concepts"] = [
                    {"type": c["type"], "confidence": c["confidence"]}
                    for c in sorted(
                        concepts, key=lambda x: x["confidence"], reverse=True
                    )[:5]
                ]

            if "amendments" in nlp_data:
                insights["amendments"] = nlp_data["amendments"].get("amendments", [])

        # Extract key relationships
        if "relationship_analysis" in analysis:
            rel_data = analysis["relationship_analysis"]
            if "relationships" in rel_data:
                relationships = rel_data["relationships"].get("relationships", [])
                insights["key_relationships"] = [
                    {"type": r["type"], "confidence": r["confidence"]}
                    for r in sorted(
                        relationships, key=lambda x: x["confidence"], reverse=True
                    )[:5]
                ]

        # Extract semantic fields
        if "semantic_analysis" in analysis:
            sem_data = analysis["semantic_analysis"]
            if "semantic_fields" in sem_data:
                insights["semantic_fields"] = list(sem_data["semantic_fields"].items())[
                    :5
                ]

        # Calculate complexity
        insights["complexity_indicators"] = {
            "concept_density": len(insights["main_legal_concepts"]),
            "relationship_density": len(insights["key_relationships"]),
            "semantic_diversity": len(insights["semantic_fields"]),
        }

        return insights


# Convenience functions for quick analysis
def analyze_legal_concepts(text: str) -> dict:
    """Quick function to analyze legal concepts in text."""
    analyzer = LegalNLPAnalyzer()
    concepts = analyzer.extract_legal_concepts(text)
    return {
        "concepts": [
            {
                "type": concept.concept_type.value,
                "text": concept.text,
                "confidence": concept.confidence,
            }
            for concept in concepts
        ]
    }


def extract_semantic_fields(text: str) -> dict:
    """Quick function to identify semantic fields in text."""
    analyzer = LegalSemanticAnalyzer()
    return analyzer.identify_semantic_fields(text)


def extract_legal_relationships(text: str) -> dict:
    """Quick function to extract legal relationships from text."""
    extractor = LegalRelationshipExtractor()
    relationships = extractor.extract_relationships(text)
    return {
        "relationships": [
            {
                "source": rel.source_entity.name,
                "target": rel.target_entity.name,
                "type": rel.relationship_type.value,
                "confidence": rel.confidence,
            }
            for rel in relationships
        ]
    }


# Export all main classes and functions
__all__ = [
    # Core classes
    "LegalNLPAnalyzer",
    "LegalSemanticAnalyzer",
    "LegalRelationshipExtractor",
    "ComprehensiveLegalAnalyzer",
    # Data classes
    "LegalConcept",
    "LegalConceptType",
    "SemanticRelation",
    "LegalAmendment",
    "SemanticField",
    "LegalEntity",
    "LegalRelationship",
    "RelationshipType",
    # Convenience functions
    "analyze_legal_concepts",
    "extract_semantic_fields",
    "extract_legal_relationships",
]
