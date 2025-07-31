"""Tests for legal_nlp core functionality."""

import pytest
from sejm_whiz.legal_nlp.core import (
    LegalNLPAnalyzer, 
    LegalConcept, 
    LegalConceptType, 
    SemanticRelation, 
    LegalAmendment
)


class TestLegalNLPAnalyzer:
    """Test cases for LegalNLPAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = LegalNLPAnalyzer()
        
        # Sample legal text in Polish
        self.sample_text = """
        Ustawa z dnia 23 kwietnia 1964 r. - Kodeks cywilny (Dz.U. Nr 16, poz. 93)
        
        Art. 1. Osoba fizyczna ma prawo do posiadania rzeczy na własność, o ile przepisy ustawy nie stanowią inaczej.
        
        Art. 2. Zabrania się naruszania prawa własności bez zgody właściciela.
        
        Art. 3. W przypadku naruszenia prawa własności, właściciel ma prawo do odszkodowania.
        Jeżeli szkoda została wyrządzona umyślnie, sprawca podlega karze grzywny w wysokości do 10000 złotych.
        
        Art. 4. Niniejsza ustawa definiuje pojęcie „właściciel" jako osobę posiadającą tytuł prawny do rzeczy.
        """
        
        self.amendment_text = """
        W ustawie z dnia 15 lutego 1992 r. o podatku dochodowym od osób prawnych wprowadza się następujące zmiany:
        1) w art. 12 ust. 1 słowa „5 lat" zastępuje się słowami „7 lat";
        2) dodaje się art. 12a w brzmieniu: „Art. 12a. Nowe przepisy wchodzą w życie z dniem 1 stycznia 2024 r.";
        3) usuwa się art. 15 ust. 3.
        """
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer is not None
        assert hasattr(self.analyzer, 'concept_patterns')
        assert hasattr(self.analyzer, 'amendment_patterns')
    
    def test_extract_legal_concepts_basic(self):
        """Test basic legal concept extraction."""
        concepts = self.analyzer.extract_legal_concepts(self.sample_text)
        
        assert len(concepts) > 0
        
        # Check that we found some expected concept types
        concept_types = [concept.concept_type for concept in concepts]
        
        # Should find rights, prohibitions, and definitions
        assert any(ct == LegalConceptType.RIGHT for ct in concept_types)
        assert any(ct == LegalConceptType.PROHIBITION for ct in concept_types)
    
    def test_extract_legal_concepts_empty_text(self):
        """Test concept extraction with empty text."""
        concepts = self.analyzer.extract_legal_concepts("")
        assert concepts == []
        
        concepts = self.analyzer.extract_legal_concepts(None)
        assert concepts == []
    
    def test_legal_concept_properties(self):
        """Test properties of extracted legal concepts."""
        concepts = self.analyzer.extract_legal_concepts(self.sample_text)
        
        for concept in concepts:
            assert isinstance(concept, LegalConcept)
            assert isinstance(concept.concept_type, LegalConceptType)
            assert isinstance(concept.text, str)
            assert len(concept.text) > 0
            assert 0 <= concept.confidence <= 1.0
            assert concept.start_pos >= 0
            assert concept.end_pos > concept.start_pos
            assert isinstance(concept.context, str)
    
    def test_detect_amendments(self):
        """Test amendment detection."""
        amendments = self.analyzer.detect_amendments(self.amendment_text)
        
        assert len(amendments) > 0
        
        # Check amendment types
        amendment_types = [amendment.amendment_type for amendment in amendments]
        assert 'modification' in amendment_types
        assert 'addition' in amendment_types
        assert 'deletion' in amendment_types
    
    def test_detect_amendments_empty_text(self):
        """Test amendment detection with empty text."""
        amendments = self.analyzer.detect_amendments("")
        assert amendments == []
    
    def test_amendment_properties(self):
        """Test properties of detected amendments."""
        amendments = self.analyzer.detect_amendments(self.amendment_text)
        
        for amendment in amendments:
            assert isinstance(amendment, LegalAmendment)
            assert amendment.amendment_type in ['addition', 'modification', 'deletion']
            assert isinstance(amendment.rationale, str)
    
    def test_analyze_semantic_relations(self):
        """Test semantic relation analysis."""
        concepts = self.analyzer.extract_legal_concepts(self.sample_text)
        relations = self.analyzer.analyze_semantic_relations(concepts)
        
        # May or may not find relations depending on concept proximity
        assert isinstance(relations, list)
        
        for relation in relations:
            assert isinstance(relation, SemanticRelation)
            assert isinstance(relation.source_concept, str)
            assert isinstance(relation.target_concept, str)
            assert isinstance(relation.relation_type, str)
            assert 0 <= relation.confidence <= 1.0
    
    def test_analyze_semantic_relations_few_concepts(self):
        """Test semantic relations with insufficient concepts."""
        # Create minimal concepts
        concept = LegalConcept(
            concept_type=LegalConceptType.RIGHT,
            text="test",
            start_pos=0,
            end_pos=4,
            confidence=0.8
        )
        
        relations = self.analyzer.analyze_semantic_relations([concept])
        assert relations == []
    
    def test_analyze_discourse_structure(self):
        """Test discourse structure analysis."""
        discourse = self.analyzer.analyze_discourse_structure(self.sample_text)
        
        assert isinstance(discourse, dict)
        assert 'discourse_markers' in discourse
        assert 'argument_structure' in discourse
        assert 'temporal_structure' in discourse
        assert 'conditional_structure' in discourse
        
        # Check discourse markers structure
        markers = discourse['discourse_markers']
        assert isinstance(markers, dict)
        
        # Check conditional structure (should find "jeżeli" pattern)
        conditionals = discourse['conditional_structure']
        assert isinstance(conditionals, list)
    
    def test_analyze_discourse_structure_empty_text(self):
        """Test discourse analysis with empty text."""
        discourse = self.analyzer.analyze_discourse_structure("")
        assert discourse == {}
    
    def test_perform_comprehensive_analysis(self):
        """Test comprehensive analysis."""
        analysis = self.analyzer.perform_comprehensive_analysis(self.sample_text)
        
        assert isinstance(analysis, dict)
        assert 'legal_concepts' in analysis
        assert 'semantic_relations' in analysis
        assert 'amendments' in analysis
        assert 'discourse_structure' in analysis
        assert 'analysis_summary' in analysis
        
        # Check legal concepts section
        concepts_section = analysis['legal_concepts']
        assert 'count' in concepts_section
        assert 'concepts' in concepts_section
        assert 'by_type' in concepts_section
        
        # Check analysis summary
        summary = analysis['analysis_summary']
        assert 'complexity_score' in summary
        assert 'main_concepts' in summary
        assert 'key_relations' in summary
        assert isinstance(summary['complexity_score'], (int, float))
    
    def test_perform_comprehensive_analysis_empty_text(self):
        """Test comprehensive analysis with empty text."""
        analysis = self.analyzer.perform_comprehensive_analysis("")
        assert analysis == {}
    
    def test_calculate_complexity_score(self):
        """Test complexity score calculation."""
        # Create sample data
        concepts = [
            LegalConcept(LegalConceptType.RIGHT, "test1", 0, 5, 0.8),
            LegalConcept(LegalConceptType.OBLIGATION, "test2", 10, 15, 0.7),
        ]
        relations = []
        discourse = {'discourse_markers': {'addition': [], 'contrast': []}}
        
        score = self.analyzer._calculate_complexity_score(concepts, relations, discourse)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1.0
    
    def test_identify_main_concepts(self):
        """Test main concept identification."""
        concepts = [
            LegalConcept(LegalConceptType.RIGHT, "high conf", 0, 8, 0.9),
            LegalConcept(LegalConceptType.OBLIGATION, "med conf", 10, 18, 0.6),
            LegalConcept(LegalConceptType.PENALTY, "low conf", 20, 28, 0.3),
        ]
        
        main_concepts = self.analyzer._identify_main_concepts(concepts)
        
        assert isinstance(main_concepts, list)
        assert len(main_concepts) <= 5
        
        # Should prioritize higher confidence concepts
        if main_concepts:
            assert 'right' in main_concepts  # Should include highest confidence
    
    def test_identify_key_relations(self):
        """Test key relation identification."""
        relations = [
            SemanticRelation("A", "B", "test_relation", 0.8),
            SemanticRelation("C", "D", "another_relation", 0.6),
        ]
        
        key_relations = self.analyzer._identify_key_relations(relations)
        
        assert isinstance(key_relations, list)
        assert len(key_relations) <= 3
        
        if key_relations:
            assert "test_relation" in key_relations  # Should include highest confidence
    
    def test_group_concepts_by_type(self):
        """Test concept grouping by type."""
        concepts = [
            LegalConcept(LegalConceptType.RIGHT, "test1", 0, 5, 0.8),
            LegalConcept(LegalConceptType.RIGHT, "test2", 10, 15, 0.7),
            LegalConcept(LegalConceptType.OBLIGATION, "test3", 20, 25, 0.6),
        ]
        
        grouped = self.analyzer._group_concepts_by_type(concepts)
        
        assert isinstance(grouped, dict)
        assert grouped['right'] == 2
        assert grouped['obligation'] == 1
    
    def test_concept_confidence_calculation(self):
        """Test concept confidence calculation."""
        # Mock a regex match object
        class MockMatch:
            def group(self, n=0):
                return "test legal concept"
        
        match = MockMatch()
        context = "legal context with legal terminology"
        concept_type = LegalConceptType.LEGAL_DEFINITION
        
        confidence = self.analyzer._calculate_concept_confidence(match, context, concept_type)
        
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1.0
    
    def test_remove_overlapping_concepts(self):
        """Test removal of overlapping concepts."""
        concepts = [
            LegalConcept(LegalConceptType.RIGHT, "overlap test", 0, 12, 0.8),
            LegalConcept(LegalConceptType.OBLIGATION, "overlap", 0, 7, 0.6),  # Overlapping
            LegalConcept(LegalConceptType.PENALTY, "separate", 20, 28, 0.7),  # Separate
        ]
        
        filtered = self.analyzer._remove_overlapping_concepts(concepts)
        
        # Should keep higher confidence overlapping concept and separate concept
        assert len(filtered) == 2
        
        # Should keep the higher confidence concept (first one)
        texts = [c.text for c in filtered]
        assert "overlap test" in texts
        assert "separate" in texts
        assert "overlap" not in texts


class TestLegalConceptType:
    """Test LegalConceptType enum."""
    
    def test_concept_type_values(self):
        """Test that all concept types have expected values."""
        expected_types = [
            'legal_principle', 'legal_definition', 'obligation', 'prohibition',
            'right', 'penalty', 'procedure', 'condition', 'exception'
        ]
        
        actual_types = [ct.value for ct in LegalConceptType]
        
        for expected in expected_types:
            assert expected in actual_types


class TestLegalConcept:
    """Test LegalConcept dataclass."""
    
    def test_legal_concept_creation(self):
        """Test LegalConcept instantiation."""
        concept = LegalConcept(
            concept_type=LegalConceptType.RIGHT,
            text="test concept",
            start_pos=10,
            end_pos=20,
            confidence=0.85,
            context="test context"
        )
        
        assert concept.concept_type == LegalConceptType.RIGHT
        assert concept.text == "test concept"
        assert concept.start_pos == 10
        assert concept.end_pos == 20
        assert concept.confidence == 0.85
        assert concept.context == "test context"
        assert concept.related_articles == []  # Default empty list
    
    def test_legal_concept_with_articles(self):
        """Test LegalConcept with related articles."""
        concept = LegalConcept(
            concept_type=LegalConceptType.OBLIGATION,
            text="test",
            start_pos=0,
            end_pos=4,
            confidence=0.7,
            related_articles=["art. 1", "art. 2"]
        )
        
        assert concept.related_articles == ["art. 1", "art. 2"]


class TestSemanticRelation:
    """Test SemanticRelation dataclass."""
    
    def test_semantic_relation_creation(self):
        """Test SemanticRelation instantiation."""
        relation = SemanticRelation(
            source_concept="concept A",
            target_concept="concept B", 
            relation_type="test_relation",
            confidence=0.75,
            evidence_text="evidence"
        )
        
        assert relation.source_concept == "concept A"
        assert relation.target_concept == "concept B"
        assert relation.relation_type == "test_relation"
        assert relation.confidence == 0.75
        assert relation.evidence_text == "evidence"


class TestLegalAmendment:
    """Test LegalAmendment dataclass."""
    
    def test_legal_amendment_creation(self):
        """Test LegalAmendment instantiation."""
        amendment = LegalAmendment(
            amendment_type="modification",
            target_provision="art. 5",
            original_text="old text",
            amended_text="new text",
            effective_date="2024-01-01",
            rationale="test rationale"
        )
        
        assert amendment.amendment_type == "modification"
        assert amendment.target_provision == "art. 5"
        assert amendment.original_text == "old text"
        assert amendment.amended_text == "new text"
        assert amendment.effective_date == "2024-01-01"
        assert amendment.rationale == "test rationale"
    
    def test_legal_amendment_minimal(self):
        """Test LegalAmendment with minimal required fields."""
        amendment = LegalAmendment(
            amendment_type="addition",
            target_provision="art. 10",
            original_text="",
            amended_text="new provision"
        )
        
        assert amendment.amendment_type == "addition"
        assert amendment.target_provision == "art. 10"
        assert amendment.original_text == ""
        assert amendment.amended_text == "new provision"
        assert amendment.effective_date is None
        assert amendment.rationale == ""