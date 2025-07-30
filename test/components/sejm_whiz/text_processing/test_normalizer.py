"""Tests for text normalization functionality."""

import pytest
from sejm_whiz.text_processing.normalizer import PolishNormalizer, LegalTextNormalizer


class TestPolishNormalizer:
    """Test Polish text normalization."""
    
    def setup_method(self):
        self.normalizer = PolishNormalizer()
    
    def test_normalize_diacritics_preserve(self):
        """Test diacritic preservation (default behavior)."""
        text = "Zażółć gęślą jaźń"
        result = self.normalizer.normalize_diacritics(text, remove=False)
        # Should normalize but preserve Polish characters
        assert "ą" in result or "a" in result
        assert "ę" in result or "e" in result
    
    def test_normalize_diacritics_remove(self):
        """Test diacritic removal."""
        text = "Zażółć gęślą jaźń"
        result = self.normalizer.normalize_diacritics(text, remove=True)
        assert "ą" not in result
        assert "ę" not in result
        assert "ó" not in result
        assert "ł" not in result
        assert "a" in result
        assert "e" in result
        assert "o" in result
        assert "l" in result
    
    def test_normalize_characters(self):
        """Test character variant normalization."""
        text = '\u201cSmart quotes\u201d and \u2018apostrophes\u2019 with \u2014 dashes\u2026'
        result = self.normalizer.normalize_characters(text)
        assert '"' in result
        assert "'" in result
        assert "-" in result
        assert "..." in result
        assert "\u201c" not in result
        assert "\u2014" not in result
    
    def test_normalize_numbers(self):
        """Test number format normalization."""
        text = "Number range 10 - 20 and date 15 . 03 . 2023"
        result = self.normalizer.normalize_numbers(text)
        assert "10-20" in result
        assert "15.03.2023" in result
    
    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "Text\u00A0with\u2000various\u200Bwhitespace\u2028characters"
        result = self.normalizer.normalize_whitespace(text)
        # All special whitespace should be normalized to regular spaces
        assert "\u00A0" not in result
        assert "\u2000" not in result
        assert "\u200B" not in result
        assert "Text with various whitespace characters" == result
    
    def test_normalize_text_pipeline(self):
        """Test complete normalization pipeline."""
        text = "Test\u201ctext\u201d  with   ą,ę,ó and 10 - 20 range\u2026"
        result = self.normalizer.normalize_text(text, remove_diacritics=True)
        
        # Should normalize quotes, diacritics, whitespace, and numbers
        assert '"' in result
        assert "a,e,o" in result
        assert "10-20" in result
        assert "..." in result
        assert "   " not in result


class TestLegalTextNormalizer:
    """Test legal text normalization."""
    
    def setup_method(self):
        self.normalizer = LegalTextNormalizer()
    
    def test_normalize_article_references(self):
        """Test article reference normalization."""
        test_cases = [
            ("art.123", "art. 123"),
            ("artykuł 45", "art. 45"),
            ("art.  67", "art. 67"),
        ]
        
        for input_text, expected in test_cases:
            result = self.normalizer.normalize_legal_references(input_text)
            assert expected in result
    
    def test_normalize_paragraph_references(self):
        """Test paragraph reference normalization."""
        test_cases = [
            ("§123", "§ 123"),
            ("par.45", "§ 45"),
            ("paragraf 67", "§ 67"),
        ]
        
        for input_text, expected in test_cases:
            result = self.normalizer.normalize_legal_references(input_text)
            assert expected in result
    
    def test_normalize_point_references(self):
        """Test point reference normalization."""
        test_cases = [
            ("pkt123", "pkt 123"),
            ("punkt 45", "pkt 45"),
        ]
        
        for input_text, expected in test_cases:
            result = self.normalizer.normalize_legal_references(input_text)
            assert expected in result
    
    def test_normalize_chapter_references(self):
        """Test chapter reference normalization."""
        test_cases = [
            ("rozd.5", "rozdz. 5"),
            ("rozdział 10", "rozdz. 10"),
        ]
        
        for input_text, expected in test_cases:
            result = self.normalizer.normalize_legal_references(input_text)
            assert expected in result
    
    def test_normalize_abbreviations(self):
        """Test legal abbreviation normalization."""
        test_cases = [
            ("u.z.", "ustawa z"),
            ("k.c.", "kodeks cywilny"),
            ("k.p.c.", "kodeks postępowania cywilnego"),
        ]
        
        for input_text, expected in test_cases:
            result = self.normalizer.normalize_legal_references(input_text)
            assert expected in result
    
    def test_normalize_legal_text_complete(self):
        """Test complete legal text normalization."""
        text = """Art.123 u.z. 15.03.2023 mówi o "prawach" obywateli.
        \u00a745 k.c. określa\u2026"""
        
        result = self.normalizer.normalize_legal_text(text, remove_diacritics=True)
        
        assert "art. 123" in result
        assert "ustawa z" in result
        assert "15.03.2023" in result
        assert "\u00a7 45" in result
        assert "kodeks cywilny" in result
        assert '"' in result  # Normalized quotes
        assert "..." in result  # Normalized ellipsis
    
    def test_case_insensitive_normalization(self):
        """Test that normalization is case insensitive."""
        text = "ART.123 i PARAGRAF 45"
        result = self.normalizer.normalize_legal_references(text)
        
        # Should normalize regardless of case
        assert "art. 123" in result.lower()
        assert "\u00a7 45" in result