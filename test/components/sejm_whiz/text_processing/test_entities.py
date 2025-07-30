"""Tests for entity extraction functionality."""

import pytest
from sejm_whiz.text_processing.entities import (
    PolishLegalNER, 
    LegalEntityExtractor, 
    LegalEntity, 
    LegalEntityType
)


class TestPolishLegalNER:
    """Test legal named entity recognition."""
    
    def setup_method(self):
        self.ner = PolishLegalNER()
    
    def test_extract_law_references(self):
        """Test extraction of law references."""
        text = """
        Zgodnie z ustawą z dnia 23 kwietnia 1964 r. - Kodeks cywilny,
        rozporządzenie Ministra z dnia 15 marca 2023 r. oraz
        Konstytucja Rzeczypospolitej Polskiej.
        """
        
        entities = self.ner.extract_legal_references(text)
        law_refs = [e for e in entities if e.entity_type == LegalEntityType.LAW_REFERENCE]
        
        assert len(law_refs) >= 2  # Should find at least kodeks and konstytucja
        
        # Check that we found some specific references
        found_texts = [entity.text.lower() for entity in law_refs]
        assert any("kodeks" in text for text in found_texts)
        assert any("konstytucja" in text for text in found_texts)
    
    def test_extract_article_references(self):
        """Test extraction of article references."""
        text = """
        Zgodnie z art. 123 ust. 2 pkt 3 kodeksu cywilnego oraz
        artykuł 45 ustawy podstawowej.
        """
        
        entities = self.ner.extract_legal_references(text)
        article_refs = [e for e in entities if e.entity_type == LegalEntityType.ARTICLE_REFERENCE]
        
        assert len(article_refs) >= 2
        
        # Check that we found the specific articles
        found_texts = [entity.text for entity in article_refs]
        assert any("123" in text for text in found_texts)
        assert any("45" in text for text in found_texts)
    
    def test_extract_paragraph_references(self):
        """Test extraction of paragraph references."""
        text = "§ 15 ust. 2 oraz paragraf 23 rozporządzenia."
        
        entities = self.ner.extract_legal_references(text)
        para_refs = [e for e in entities if e.entity_type == LegalEntityType.PARAGRAPH_REFERENCE]
        
        assert len(para_refs) >= 2
        
        found_texts = [entity.text for entity in para_refs]
        assert any("15" in text for text in found_texts)
        assert any("23" in text for text in found_texts)
    
    def test_extract_court_names(self):
        """Test extraction of court names."""
        text = """
        Sąd Najwyższy wydał orzeczenie, które zostało potwierdzone przez
        Trybunał Konstytucyjny. Wojewódzki Sąd Administracyjny w Warszawie
        oraz Sąd Rejonowy w Krakowie.
        """
        
        entities = self.ner.extract_legal_references(text)
        court_refs = [e for e in entities if e.entity_type == LegalEntityType.COURT_NAME]
        
        assert len(court_refs) >= 3
        
        found_texts = [entity.text.lower() for entity in court_refs]
        assert any("najwyższy" in text for text in found_texts)
        assert any("konstytucyjny" in text for text in found_texts)
        assert any("administracyjny" in text for text in found_texts)
    
    def test_extract_legal_persons(self):
        """Test extraction of legal persons."""
        text = """
        Spółka z ograniczoną odpowiedzialnością "Test" sp. z o.o.
        oraz spółka akcyjna "Example" S.A. i Fundacja Pomocy.
        """
        
        entities = self.ner.extract_legal_references(text)
        legal_persons = [e for e in entities if e.entity_type == LegalEntityType.LEGAL_PERSON]
        
        assert len(legal_persons) >= 2
        
        found_texts = [entity.text.lower() for entity in legal_persons]
        assert any("sp. z o.o." in text or "spółka" in text for text in found_texts)
        assert any("s.a." in text or "akcyjna" in text for text in found_texts)
    
    def test_remove_overlaps(self):
        """Test overlap removal functionality."""
        # Create overlapping entities
        entities = [
            LegalEntity("art. 123", LegalEntityType.ARTICLE_REFERENCE, 0, 8),
            LegalEntity("art. 123 ust. 2", LegalEntityType.ARTICLE_REFERENCE, 0, 14),
            LegalEntity("separate entity", LegalEntityType.LAW_REFERENCE, 20, 35)
        ]
        
        result = self.ner._remove_overlaps(entities)
        
        # Should keep the longer overlapping entity and the separate one
        assert len(result) == 2
        assert any("ust. 2" in entity.text for entity in result)
        assert any("separate entity" in entity.text for entity in result)
    
    def test_get_entity_statistics(self):
        """Test entity statistics generation."""
        entities = [
            LegalEntity("art. 1", LegalEntityType.ARTICLE_REFERENCE, 0, 6),
            LegalEntity("art. 2", LegalEntityType.ARTICLE_REFERENCE, 10, 16),
            LegalEntity("§ 1", LegalEntityType.PARAGRAPH_REFERENCE, 20, 23),
            LegalEntity("kodeks", LegalEntityType.LAW_REFERENCE, 30, 36)
        ]
        
        stats = self.ner.get_entity_statistics(entities)
        
        assert stats[LegalEntityType.ARTICLE_REFERENCE.value] == 2
        assert stats[LegalEntityType.PARAGRAPH_REFERENCE.value] == 1
        assert stats[LegalEntityType.LAW_REFERENCE.value] == 1


class TestLegalEntityExtractor:
    """Test high-level entity extraction interface."""
    
    def setup_method(self):
        self.extractor = LegalEntityExtractor()
    
    def test_extract_entities_complete(self):
        """Test complete entity extraction."""
        text = """
        Art. 123 ustawy z dnia 15 marca 2023 r. stanowi, że Sąd Najwyższy
        może wydać orzeczenie. § 45 rozporządzenia określa procedury
        dla spółki z o.o. "Test Company".
        """
        
        result = self.extractor.extract_entities(text)
        
        # Check structure
        assert 'entities' in result
        assert 'grouped_entities' in result
        assert 'statistics' in result
        
        # Check that entities were found
        assert len(result['entities']) > 0
        
        # Check that entities have proper structure
        for entity in result['entities']:
            assert 'text' in entity
            assert 'type' in entity
            assert 'start' in entity
            assert 'end' in entity
            assert 'confidence' in entity
        
        # Check grouped entities
        assert len(result['grouped_entities']) > 0
        
        # Check statistics
        assert isinstance(result['statistics'], dict)
        assert sum(result['statistics'].values()) == len(result['entities'])
    
    def test_extract_entities_empty_input(self):
        """Test handling of empty input."""
        result = self.extractor.extract_entities("")
        
        assert result['entities'] == []
        assert result['grouped_entities'] == {}
        assert result['statistics'] == {}
    
    def test_extract_entities_no_legal_content(self):
        """Test extraction from text without legal content."""
        text = "This is just regular text without any legal references."
        result = self.extractor.extract_entities(text)
        
        # Might still find some entities from spaCy NER (like organizations)
        # but should handle gracefully
        assert 'entities' in result
        assert 'grouped_entities' in result
        assert 'statistics' in result
    
    def test_grouped_entities_structure(self):
        """Test that grouped entities are properly structured."""
        text = "Art. 1 i art. 2 oraz § 5 kodeksu."
        result = self.extractor.extract_entities(text)
        
        grouped = result['grouped_entities']
        
        # If articles were found, they should be grouped properly
        if 'ARTICLE_REFERENCE' in grouped:
            articles = grouped['ARTICLE_REFERENCE']
            for article in articles:
                assert 'text' in article
                assert 'start' in article
                assert 'end' in article
                assert 'confidence' in article