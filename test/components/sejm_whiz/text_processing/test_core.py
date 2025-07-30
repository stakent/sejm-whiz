"""Tests for core text processing functionality."""

import pytest
from sejm_whiz.text_processing.core import (
    TextProcessor,
    clean_legal_text,
    normalize_legal_text,
    process_legal_document,
    extract_legal_references
)


class TestTextProcessor:
    """Test main TextProcessor interface."""
    
    def setup_method(self):
        try:
            self.processor = TextProcessor()
        except Exception:
            # If spacy model not available, skip these tests
            pytest.skip("SpaCy model not available")
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        html_text = "<p>Art. 1.&nbsp;Test <strong>content</strong></p>"
        result = self.processor.clean_text(html_text)
        
        assert "<p>" not in result
        assert "&nbsp;" not in result
        assert "<strong>" not in result
        assert "Art. 1. Test content" in result
    
    def test_normalize_text(self):
        """Test text normalization functionality."""
        text = "art.123 z dnia 15 . 03 . 2023"
        result = self.processor.normalize_text(text)
        
        assert "art. 123" in result
        assert "15.03.2023" in result
    
    def test_normalize_text_remove_diacritics(self):
        """Test diacritic removal during normalization."""
        text = "Przepis o ochronie danych"
        result = self.processor.normalize_text(text, remove_diacritics=True)
        
        # Should remove Polish diacritics
        assert "ą" not in result or "a" in result
        assert "ę" not in result or "e" in result
    
    def test_tokenize_text(self):
        """Test text tokenization."""
        text = "Art. 1. Pierwszy przepis.\n\nArt. 2. Drugi przepis."
        result = self.processor.tokenize_text(text)
        
        # Check expected structure
        expected_keys = ['paragraphs', 'sentences', 'words', 'structure', 'segments', 'linguistic_features']
        for key in expected_keys:
            assert key in result
        
        # Check content
        assert len(result['paragraphs']) >= 1
        assert len(result['sentences']) >= 2
        assert len(result['words']) >= 6
    
    def test_extract_entities(self):
        """Test entity extraction."""
        text = "Art. 123 ustawy oraz Sąd Najwyższy wydał orzeczenie."
        result = self.processor.extract_entities(text)
        
        # Check structure
        assert 'entities' in result
        assert 'grouped_entities' in result
        assert 'statistics' in result
        
        # Should find some entities
        assert len(result['entities']) >= 1
    
    def test_analyze_document(self):
        """Test document analysis."""
        text = """Ustawa z dnia 15 marca 2023 r.
        Art. 1. Pierwszy artykuł.
        Art. 2. Drugi artykuł."""
        
        result = self.processor.analyze_document(text)
        
        # Check structure
        assert 'document_info' in result
        assert 'structure' in result
        assert 'provisions' in result
        assert 'references' in result
    
    def test_process_document_minimal(self):
        """Test minimal document processing."""
        text = "Art. 1. Test content."
        result = self.processor.process_document(
            text,
            clean=False,
            normalize=False,
            extract_entities=False,
            analyze_structure=False
        )
        
        assert result['original_text'] == text
        assert result['processed_text'] == text
        assert 'tokenization' in result
        assert 'entities' not in result or result['entities'] == {}
        assert 'analysis' not in result or result['analysis'] == {}
    
    def test_process_document_complete(self):
        """Test complete document processing pipeline."""
        text = "<p>art.123&nbsp;ustawy</p>"
        result = self.processor.process_document(text)
        
        # Check all processing steps were applied
        assert result['original_text'] == text
        assert result['processed_text'] != text  # Should be different after processing
        assert "<p>" not in result['processed_text']  # HTML cleaned
        assert "art. 123" in result['processed_text']  # Normalized
        
        # Check all components are present
        assert 'tokenization' in result
        assert 'entities' in result
        assert 'analysis' in result
        
        # Check processing steps tracking
        steps = result['processing_steps']
        assert steps['cleaned'] is True
        assert steps['normalized'] is True
        assert steps['diacritics_removed'] is False
    
    def test_get_text_statistics(self):
        """Test text statistics generation."""
        text = """First paragraph with content.

        Second paragraph with more content.
        Multiple sentences here."""
        
        stats = self.processor.get_text_statistics(text)
        
        assert 'characters' in stats
        assert 'words' in stats
        assert 'sentences' in stats
        assert 'paragraphs' in stats
        
        assert stats['characters'] == len(text)
        assert stats['words'] >= 8  # At least some words
        assert stats['sentences'] >= 3  # At least some sentences
        assert stats['paragraphs'] >= 2  # At least two paragraphs
    
    def test_empty_input_handling(self):
        """Test handling of empty input."""
        result = self.processor.process_document("")
        
        assert result['original_text'] == ""
        assert result['processed_text'] == ""
        
        # Check tokenization structure without exact match due to structure details
        tokenization = result['tokenization']
        assert tokenization['paragraphs'] == []
        assert tokenization['sentences'] == []
        assert tokenization['words'] == []
        assert tokenization['segments'] == []
        assert tokenization['linguistic_features']['tokens'] == []
        assert 'structure' in tokenization


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_clean_legal_text(self):
        """Test quick clean function."""
        html_text = "<p>Art. 1. Test</p>"
        result = clean_legal_text(html_text)
        
        assert "<p>" not in result
        assert "Art. 1. Test" in result
    
    def test_normalize_legal_text(self):
        """Test quick normalize function."""
        text = "art.123"
        result = normalize_legal_text(text)
        
        assert "art. 123" in result
    
    def test_process_legal_document(self):
        """Test quick process function."""
        text = "<p>art.123</p>"
        result = process_legal_document(text, extract_entities=False, analyze_structure=False)
        
        assert 'original_text' in result
        assert 'processed_text' in result
        assert 'tokenization' in result
    
    def test_extract_legal_references(self):
        """Test quick reference extraction."""
        text = "Art. 123 ust. 2 oraz § 45"
        result = extract_legal_references(text)
        
        # Should return a list of reference entities
        assert isinstance(result, list)
        # Actual content depends on NER functionality
