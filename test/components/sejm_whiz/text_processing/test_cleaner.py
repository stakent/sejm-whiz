"""Tests for text cleaning functionality."""

import pytest
from sejm_whiz.text_processing.cleaner import HTMLCleaner, TextCleaner


class TestHTMLCleaner:
    """Test HTML cleaning functionality."""
    
    def setup_method(self):
        self.cleaner = HTMLCleaner()
    
    def test_clean_html_basic(self):
        """Test basic HTML tag removal."""
        html_text = "<p>Test paragraph</p>"
        expected = "Test paragraph"
        result = self.cleaner.clean_html(html_text)
        assert result == expected
    
    def test_clean_html_complex(self):
        """Test complex HTML structure removal."""
        html_text = """
        <div class="legal-text">
            <h1>Ustawa</h1>
            <p>Art. 1. <strong>Przepis</strong> prawny.</p>
            <ul>
                <li>Punkt pierwszy</li>
                <li>Punkt drugi</li>
            </ul>
        </div>
        """
        result = self.cleaner.clean_html(html_text)
        assert "<" not in result
        assert ">" not in result
        assert "Ustawa" in result
        assert "Art. 1." in result
        assert "Przepis prawny" in result
    
    def test_clean_html_entities(self):
        """Test HTML entity removal."""
        html_text = "Test&nbsp;with&amp;entities&lt;tag&gt;"
        result = self.cleaner.clean_html(html_text)
        assert "&nbsp;" not in result
        assert "&amp;" not in result
        assert "&lt;" not in result
        assert "&gt;" not in result
    
    def test_clean_formatting(self):
        """Test formatting cleanup."""
        text = "Text   with\n\n\nmultiple\t\tspaces"
        result = self.cleaner.clean_formatting(text)
        assert "   " not in result
        assert "\n\n\n" not in result
        assert "\t\t" not in result
    
    def test_clean_document_pipeline(self):
        """Test complete document cleaning pipeline."""
        document = """
        <html>
            <body>
                <p>Art.&nbsp;1.&amp;nbsp;Test</p>
                <p>   Another   paragraph   </p>
            </body>
        </html>
        """
        result = self.cleaner.clean_document(document)
        assert "<" not in result
        assert "&nbsp;" not in result
        assert "Art. 1." in result
        assert result.strip() == result  # No leading/trailing whitespace
    
    def test_empty_input(self):
        """Test handling of empty input."""
        assert self.cleaner.clean_html("") == ""
        assert self.cleaner.clean_formatting("") == ""
        assert self.cleaner.clean_document("") == ""


class TestTextCleaner:
    """Test advanced text cleaning functionality."""
    
    def setup_method(self):
        self.cleaner = TextCleaner()
    
    def test_remove_noise_page_numbers(self):
        """Test removal of page numbers."""
        text = """Art. 1. Test
        strona 5
        Art. 2. Another test
        str. 10
        Art. 3. Final test"""
        
        result = self.cleaner.remove_noise(text)
        assert "strona 5" not in result
        assert "str. 10" not in result
        assert "Art. 1. Test" in result
        assert "Art. 2. Another test" in result
    
    def test_remove_noise_headers(self):
        """Test removal of document headers."""
        text = """DZIENNIK USTAW RP
        Art. 1. Test content
        Poz. 123
        Art. 2. More content"""
        
        result = self.cleaner.remove_noise(text)
        assert "DZIENNIK USTAW" not in result
        assert "Poz. 123" not in result
        assert "Art. 1. Test content" in result
    
    def test_clean_punctuation(self):
        """Test punctuation normalization."""
        text = "Test....text---with.....excessive...punctuation"
        result = self.cleaner.remove_noise(text)
        assert "....." not in result
        assert "---" in result  # Should be normalized to exactly 3 dashes
        assert "..." in result
    
    def test_clean_text_pipeline(self):
        """Test complete text cleaning pipeline."""
        text = """
        <p>DZIENNIK USTAW</p>
        <div>Art. 1. Test&nbsp;content</div>
        strona 15
        <p>Art. 2. More content</p>
        """
        
        result = self.cleaner.clean_text(text)
        assert "<" not in result
        assert "DZIENNIK USTAW" not in result
        assert "strona 15" not in result
        assert "Art. 1. Test content" in result
        assert "Art. 2. More content" in result
    
    def test_preserve_legal_structure(self):
        """Test that legal structure is preserved."""
        text = """Art. 1.
        1. Pierwszy punkt
        2. Drugi punkt
        § 1. Paragraf
        Art. 2. Kolejny artykuł"""
        
        result = self.cleaner.clean_text(text)
        assert "Art. 1." in result
        assert "1. Pierwszy punkt" in result
        assert "§ 1. Paragraf" in result
        assert "Art. 2." in result