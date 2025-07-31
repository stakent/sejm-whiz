"""Integration tests for legal_nlp component."""

import pytest
from sejm_whiz.legal_nlp import (
    ComprehensiveLegalAnalyzer,
    analyze_legal_concepts,
    extract_semantic_fields,
    extract_legal_relationships
)


class TestComprehensiveLegalAnalyzer:
    """Test cases for ComprehensiveLegalAnalyzer integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ComprehensiveLegalAnalyzer()
        
        # Comprehensive legal text for integration testing
        self.comprehensive_text = """
        Ustawa z dnia 23 kwietnia 1964 r. - Kodeks cywilny (Dz.U. Nr 16, poz. 93)
        
        Art. 1. Osoba fizyczna ma prawo do posiadania rzeczy na własność, 
        o ile przepisy ustawy nie stanowią inaczej.
        
        Art. 2. Zabrania się naruszania prawa własności bez zgody właściciela.
        Jeżeli dojdzie do naruszenia, właściciel może żądać przywrócenia stanu poprzedniego.
        
        Art. 3. Minister sprawiedliwości definiuje pojęcie „własność" jako prawo rzeczowe
        dające właścicielowi pełną władzę nad rzeczą.
        
        Art. 4. Sąd rozpoznaje sprawy dotyczące sporów własnościowych.
        Ponieważ prawo własności jest fundamentalne, dlatego jego ochrona ma pierwszeństwo.
        
        Art. 5. W ustawie z dnia 15 lutego 1992 r. wprowadza się następujące zmiany:
        1) słowa „5 lat" zastępuje się słowami „7 lat";
        2) dodaje się nowy artykuł.
        """
    
    def test_analyzer_initialization(self):
        """Test comprehensive analyzer initialization."""
        assert self.analyzer is not None
        assert hasattr(self.analyzer, 'nlp_analyzer')
        assert hasattr(self.analyzer, 'semantic_analyzer')
        assert hasattr(self.analyzer, 'relationship_extractor')
    
    def test_analyze_document_comprehensive(self):
        """Test comprehensive document analysis."""
        analysis = self.analyzer.analyze_document(self.comprehensive_text)
        
        assert isinstance(analysis, dict)
        
        # Check main sections
        assert 'legal_nlp_analysis' in analysis
        assert 'semantic_analysis' in analysis
        assert 'relationship_analysis' in analysis
        assert 'document_metadata' in analysis
        
        # Check document metadata
        metadata = analysis['document_metadata']
        assert metadata['text_length'] > 0
        assert metadata['word_count'] > 0
        assert 'analysis_components' in metadata
        assert len(metadata['analysis_components']) == 3
    
    def test_analyze_document_empty_text(self):
        """Test comprehensive analysis with empty text."""
        analysis = self.analyzer.analyze_document("")
        assert analysis == {}
        
        analysis = self.analyzer.analyze_document(None)
        assert analysis == {}
    
    def test_legal_nlp_analysis_section(self):
        """Test legal NLP analysis section structure."""
        analysis = self.analyzer.analyze_document(self.comprehensive_text)
        nlp_section = analysis['legal_nlp_analysis']
        
        assert isinstance(nlp_section, dict)
        assert 'legal_concepts' in nlp_section
        assert 'semantic_relations' in nlp_section
        assert 'amendments' in nlp_section
        assert 'discourse_structure' in nlp_section
        assert 'analysis_summary' in nlp_section
        
        # Check legal concepts
        concepts = nlp_section['legal_concepts']
        assert 'count' in concepts
        assert 'concepts' in concepts
        assert 'by_type' in concepts
        assert isinstance(concepts['count'], int)
        assert isinstance(concepts['concepts'], list)
        assert isinstance(concepts['by_type'], dict)
    
    def test_semantic_analysis_section(self):
        """Test semantic analysis section structure."""
        analysis = self.analyzer.analyze_document(self.comprehensive_text)
        semantic_section = analysis['semantic_analysis']
        
        assert isinstance(semantic_section, dict)
        assert 'semantic_fields' in semantic_section
        assert 'semantic_relations' in semantic_section
        assert 'conceptual_density' in semantic_section
        assert 'legal_definitions' in semantic_section
        assert 'argumentative_structure' in semantic_section
        assert 'analysis_metadata' in semantic_section
    
    def test_relationship_analysis_section(self):
        """Test relationship analysis section structure."""
        analysis = self.analyzer.analyze_document(self.comprehensive_text)
        relationship_section = analysis['relationship_analysis']
        
        assert isinstance(relationship_section, dict)
        assert 'entities' in relationship_section
        assert 'relationships' in relationship_section
        assert 'network_analysis' in relationship_section
        
        # Check entities subsection
        entities = relationship_section['entities']
        assert 'count' in entities
        assert 'by_type' in entities
        assert 'entities' in entities
    
    def test_extract_key_insights(self):
        """Test key insights extraction."""
        insights = self.analyzer.extract_key_insights(self.comprehensive_text)
        
        assert isinstance(insights, dict)
        assert 'main_legal_concepts' in insights
        assert 'key_relationships' in insights
        assert 'semantic_fields' in insights
        assert 'amendments' in insights
        assert 'complexity_indicators' in insights
        
        # Check complexity indicators
        complexity = insights['complexity_indicators']
        assert 'concept_density' in complexity
        assert 'relationship_density' in complexity
        assert 'semantic_diversity' in complexity
        assert isinstance(complexity['concept_density'], int)
        assert isinstance(complexity['relationship_density'], int)
        assert isinstance(complexity['semantic_diversity'], int)
    
    def test_extract_key_insights_empty_text(self):
        """Test key insights extraction with empty text."""
        insights = self.analyzer.extract_key_insights("")
        
        # Should return empty structure
        assert insights['main_legal_concepts'] == []
        assert insights['key_relationships'] == []
        assert insights['semantic_fields'] == []
        assert insights['amendments'] == []
        assert insights['complexity_indicators']['concept_density'] == 0
    
    def test_insights_data_structure(self):
        """Test structure of extracted insights."""
        insights = self.analyzer.extract_key_insights(self.comprehensive_text)
        
        # Check main legal concepts structure
        for concept in insights['main_legal_concepts']:
            assert isinstance(concept, dict)
            assert 'type' in concept
            assert 'confidence' in concept
            assert isinstance(concept['confidence'], (int, float))
        
        # Check key relationships structure
        for relationship in insights['key_relationships']:
            assert isinstance(relationship, dict)
            assert 'type' in relationship
            assert 'confidence' in relationship
            assert isinstance(relationship['confidence'], (int, float))
        
        # Check semantic fields structure
        for field_name, score in insights['semantic_fields']:
            assert isinstance(field_name, str)
            assert isinstance(score, (int, float))
    
    def test_cross_component_integration(self):
        """Test that different analysis components work together correctly."""
        analysis = self.analyzer.analyze_document(self.comprehensive_text)
        
        # Extract data from different components
        nlp_concepts = analysis['legal_nlp_analysis']['legal_concepts']['concepts']
        semantic_fields = analysis['semantic_analysis']['semantic_fields']
        entities = analysis['relationship_analysis']['entities']['entities']
        
        # Verify that we have meaningful data from each component
        assert len(nlp_concepts) > 0  # Should find legal concepts
        assert len(semantic_fields) > 0  # Should identify semantic fields
        assert len(entities) > 0  # Should extract entities
        
        # Check that the analysis makes sense for legal text
        # Should identify civil law as main semantic field
        if semantic_fields:
            top_field = max(semantic_fields.items(), key=lambda x: x[1])
            assert top_field[0] in ['civil_law', 'constitutional_law', 'administrative_law']


class TestConvenienceFunctions:
    """Test convenience functions for quick analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_text = """
        Osoba fizyczna ma prawo do ochrony danych osobowych.
        Zabrania się przetwarzania danych bez zgody zainteresowanego.
        Minister cyfryzacji definiuje pojęcie „dane osobowe" jako informacje o osobie fizycznej.
        """
    
    def test_analyze_legal_concepts_function(self):
        """Test analyze_legal_concepts convenience function."""
        result = analyze_legal_concepts(self.sample_text)
        
        assert isinstance(result, dict)
        assert 'concepts' in result
        assert isinstance(result['concepts'], list)
        
        for concept in result['concepts']:
            assert isinstance(concept, dict)
            assert 'type' in concept
            assert 'text' in concept
            assert 'confidence' in concept
            assert isinstance(concept['confidence'], (int, float))
    
    def test_analyze_legal_concepts_empty_text(self):
        """Test analyze_legal_concepts with empty text."""
        result = analyze_legal_concepts("")
        assert result == {'concepts': []}
    
    def test_extract_semantic_fields_function(self):
        """Test extract_semantic_fields convenience function."""
        result = extract_semantic_fields(self.sample_text)
        
        assert isinstance(result, dict)
        # Should be field_name -> score mapping
        for field_name, score in result.items():
            assert isinstance(field_name, str)
            assert isinstance(score, (int, float))
            assert score > 0
    
    def test_extract_semantic_fields_empty_text(self):
        """Test extract_semantic_fields with empty text."""
        result = extract_semantic_fields("")
        assert result == {}
    
    def test_extract_legal_relationships_function(self):
        """Test extract_legal_relationships convenience function."""
        result = extract_legal_relationships(self.sample_text)
        
        assert isinstance(result, dict)
        assert 'relationships' in result
        assert isinstance(result['relationships'], list)
        
        for relationship in result['relationships']:
            assert isinstance(relationship, dict)
            assert 'source' in relationship
            assert 'target' in relationship
            assert 'type' in relationship
            assert 'confidence' in relationship
            assert isinstance(relationship['confidence'], (int, float))
    
    def test_extract_legal_relationships_empty_text(self):
        """Test extract_legal_relationships with empty text."""
        result = extract_legal_relationships("")
        assert result == {'relationships': []}


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""
    
    def test_polish_civil_code_analysis(self):
        """Test analysis of Polish civil code text."""
        civil_code_text = """
        Kodeks cywilny z dnia 23 kwietnia 1964 r.
        
        Art. 140. Czynność prawna sprzeczna z ustawą albo mająca na celu obejście ustawy 
        jest nieważna, chyba że właściwy przepis przewiduje inny skutek.
        
        Art. 141. Nieważna jest również czynność prawna sprzeczna z zasadami współżycia społecznego.
        
        Art. 142. Jeżeli nieważność dotyczy tylko części czynności prawnej, 
        czynność pozostaje ważna co do pozostałych części, 
        gdy można przypuszczać, że byłaby dokonana również bez części obciążonej nieważnością.
        """
        
        analyzer = ComprehensiveLegalAnalyzer()
        analysis = analyzer.analyze_document(civil_code_text)
        
        # Should identify this as civil law
        semantic_fields = analysis['semantic_analysis']['semantic_fields']
        assert 'civil_law' in semantic_fields
        
        # Should find legal concepts
        concepts = analysis['legal_nlp_analysis']['legal_concepts']['concepts']
        assert len(concepts) > 0
        
        # Should extract some entities
        entities = analysis['relationship_analysis']['entities']['entities']
        assert len(entities) > 0
    
    def test_constitutional_law_analysis(self):
        """Test analysis of constitutional law text."""
        constitution_text = """
        Konstytucja Rzeczypospolitej Polskiej z dnia 2 kwietnia 1997 r.
        
        Art. 2. Rzeczpospolita Polska jest demokratycznym państwem prawnym, 
        urzeczywistniającym zasady sprawiedliwości społecznej.
        
        Art. 3. Rzeczpospolita Polska jest państwem unitarnym.
        
        Art. 4. Władza zwierzchnia w Rzeczypospolitej Polskiej należy do Narodu.
        Naród sprawuje władzę przez swoich przedstawicieli lub bezpośrednio.
        """
        
        analyzer = ComprehensiveLegalAnalyzer()
        analysis = analyzer.analyze_document(constitution_text)
        
        # Should identify constitutional law elements
        semantic_fields = analysis['semantic_analysis']['semantic_fields']
        assert 'constitutional_law' in semantic_fields
        
        # Should find legal principles and definitions
        concepts = analysis['legal_nlp_analysis']['legal_concepts']['concepts']
        concept_types = [c['type'] for c in concepts]
        expected_types = ['legal_principle', 'legal_definition']
        found_expected = [ct for ct in expected_types if ct in concept_types]
        assert len(found_expected) > 0
    
    def test_amendment_text_analysis(self):
        """Test analysis of legal amendment text."""
        amendment_text = """
        Ustawa z dnia 15 marca 2023 r. o zmianie ustawy - Kodeks postępowania cywilnego
        
        Art. 1. W ustawie z dnia 17 listopada 1964 r. - Kodeks postępowania cywilnego 
        wprowadza się następujące zmiany:
        
        1) w art. 205 § 1 słowa „30 dni" zastępuje się słowami „45 dni";
        2) dodaje się art. 205a w brzmieniu: „Art. 205a. Nowe przepisy stosuje się od dnia 1 stycznia 2024 r.";
        3) uchyla się art. 206 § 3.
        """
        
        analyzer = ComprehensiveLegalAnalyzer()
        analysis = analyzer.analyze_document(amendment_text)
        
        # Should detect amendments
        amendments = analysis['legal_nlp_analysis']['amendments']['amendments']
        assert len(amendments) > 0
        
        # Should find different types of amendments
        amendment_types = [a['type'] for a in amendments]
        assert 'modification' in amendment_types
        assert 'addition' in amendment_types
        assert 'deletion' in amendment_types
    
    def test_performance_with_long_text(self):
        """Test performance and correctness with longer legal text."""
        # Create a longer text by repeating patterns
        long_text = """
        Kodeks cywilny reguluje stosunki między osobami fizycznymi i prawnymi.
        """ * 50  # Repeat to create longer text
        
        analyzer = ComprehensiveLegalAnalyzer()
        
        # Should complete without errors
        analysis = analyzer.analyze_document(long_text)
        
        assert isinstance(analysis, dict)
        assert len(analysis) > 0
        
        # Should scale reasonably with text length
        metadata = analysis['document_metadata']
        assert metadata['text_length'] > 1000
        assert metadata['word_count'] > 100