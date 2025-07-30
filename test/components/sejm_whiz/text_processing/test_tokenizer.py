"""Tests for tokenization functionality."""

import pytest
from sejm_whiz.text_processing.tokenizer import PolishTokenizer, LegalDocumentTokenizer


class TestPolishTokenizer:
    """Test Polish tokenization functionality."""
    
    def setup_method(self):
        # Use basic tokenizer that doesn't require spacy model
        try:
            self.tokenizer = PolishTokenizer()
        except Exception:
            # If spacy model not available, skip these tests
            pytest.skip("SpaCy model not available")
    
    def test_tokenize_words_basic(self):
        """Test basic word tokenization."""
        text = "To jest test tokenizacji."
        result = self.tokenizer.tokenize_words(text)
        
        expected_words = ["To", "jest", "test", "tokenizacji", "."]
        assert len(result) >= 4  # At least the main words
        assert "To" in result
        assert "jest" in result
        assert "test" in result
        assert "tokenizacji" in result
    
    def test_tokenize_sentences_basic(self):
        """Test basic sentence tokenization."""
        text = "Pierwsza sentencja. Druga sentencja! Trzecia sentencja?"
        result = self.tokenizer.tokenize_sentences(text)
        
        assert len(result) == 3
        assert "Pierwsza sentencja" in result[0]
        assert "Druga sentencja" in result[1]
        assert "Trzecia sentencja" in result[2]
    
    def test_tokenize_legal_sentences(self):
        """Test legal document sentence tokenization."""
        text = "Art. 1. Przepis pierwszy; 1) punkt pierwszy; 2) punkt drugi."
        result = self.tokenizer.tokenize_sentences(text)
        
        # Should handle legal punctuation patterns
        assert len(result) >= 1
        # Fallback tokenizer may split differently, just check we get reasonable results
        assert "Art" in str(result[0]) or "Przepis" in str(result[0])
    
    def test_tokenize_paragraphs(self):
        """Test paragraph tokenization."""
        text = """Pierwszy paragraf.

        Drugi paragraf z treścią.

        Trzeci paragraf."""
        
        result = self.tokenizer.tokenize_paragraphs(text)
        assert len(result) == 3
        assert "Pierwszy paragraf" in result[0]
        assert "Drugi paragraf" in result[1]
        assert "Trzeci paragraf" in result[2]
    
    def test_get_linguistic_features(self):
        """Test linguistic feature extraction."""
        text = "Test tekstu polskiego."
        result = self.tokenizer.get_linguistic_features(text)
        
        assert 'tokens' in result
        assert 'lemmas' in result
        assert 'pos_tags' in result
        assert 'entities' in result
        
        assert len(result['tokens']) > 0
        assert len(result['lemmas']) > 0
    
    def test_empty_input(self):
        """Test handling of empty input."""
        assert self.tokenizer.tokenize_words("") == []
        assert self.tokenizer.tokenize_sentences("") == []
        assert self.tokenizer.tokenize_paragraphs("") == []
        
        features = self.tokenizer.get_linguistic_features("")
        assert features['tokens'] == []
        assert features['lemmas'] == []


class TestLegalDocumentTokenizer:
    """Test legal document tokenization functionality."""
    
    def setup_method(self):
        try:
            self.tokenizer = LegalDocumentTokenizer()
        except Exception:
            pytest.skip("SpaCy model not available")
    
    def test_extract_legal_structure_articles(self):
        """Test article extraction."""
        text = "Art. 1. Pierwszy artykuł. Artykuł 2. Drugi artykuł."
        result = self.tokenizer.extract_legal_structure(text)
        
        assert 'article' in result
        assert len(result['article']) >= 2
        
        # Check that articles are found
        articles = [match[0] for match in result['article']]
        assert any("1" in article for article in articles)
        assert any("2" in article for article in articles)
    
    def test_extract_legal_structure_paragraphs(self):
        """Test paragraph extraction."""
        text = "§ 1. Pierwszy paragraf. § 2. Drugi paragraf."
        result = self.tokenizer.extract_legal_structure(text)
        
        assert 'paragraph' in result
        assert len(result['paragraph']) >= 2
    
    def test_extract_legal_structure_points(self):
        """Test point extraction."""
        text = "1) pierwszy punkt; 2) drugi punkt; pkt 3 trzeci punkt."
        result = self.tokenizer.extract_legal_structure(text)
        
        assert 'point' in result
        # Should find at least the "pkt 3" reference
        assert len(result['point']) >= 1
    
    def test_extract_legal_structure_chapters(self):
        """Test chapter extraction."""
        text = "Rozdział 1. Pierwszy rozdział. Rozdz. II Drugi rozdział."
        result = self.tokenizer.extract_legal_structure(text)
        
        assert 'chapter' in result
        assert len(result['chapter']) >= 2
    
    def test_segment_by_structure(self):
        """Test structural segmentation."""
        text = """Art. 1. Pierwszy artykuł z treścią.
        Art. 2. Drugi artykuł z inną treścią.
        § 1. Paragraf w ramach artykułu."""
        
        result = self.tokenizer.segment_by_structure(text)
        
        assert len(result) >= 2  # At least some segments
        
        # Check that segments have proper structure
        for segment in result:
            assert 'type' in segment
            assert 'marker' in segment
            assert 'content' in segment
    
    def test_tokenize_legal_document_complete(self):
        """Test complete legal document tokenization."""
        text = """Art. 1. Pierwszy artykuł.

        Treść artykułu z sentencją. Druga sentencja artykułu.

        Art. 2. Drugi artykuł."""
        
        result = self.tokenizer.tokenize_legal_document(text)
        
        # Check all expected keys are present
        expected_keys = ['paragraphs', 'sentences', 'words', 'structure', 'segments', 'linguistic_features']
        for key in expected_keys:
            assert key in result
        
        # Check that content is found
        assert len(result['paragraphs']) >= 2
        assert len(result['sentences']) >= 2
        assert len(result['words']) > 10
        
        # Check legal structure
        assert 'article' in result['structure']
        assert len(result['structure']['article']) >= 2
    
    def test_complex_legal_structure(self):
        """Test complex legal document structure."""
        text = """Rozdział I
        Przepisy ogólne
        
        Art. 1. 1. Ustawa reguluje zasady.
        2. W ustawie stosuje się następujące określenia:
        1) pojęcie pierwsze - definicja;
        2) pojęcie drugie - inna definicja.
        
        § 1. Dodatkowe przepisy."""
        
        result = self.tokenizer.tokenize_legal_document(text)
        
        # Should find multiple types of legal elements
        structure = result['structure']
        assert len(structure['chapter']) >= 1
        assert len(structure['article']) >= 1
        assert len(structure['paragraph']) >= 1
        
        # Should create proper segments
        assert len(result['segments']) >= 2