"""Tests for legal document parsing functionality."""

import pytest
from sejm_whiz.text_processing.legal_parser import (
    PolishLegalParser,
    LegalDocumentAnalyzer,
    LegalDocumentType,
    LegalReference,
    LegalProvision
)


class TestPolishLegalParser:
    """Test Polish legal document parsing."""
    
    def setup_method(self):
        self.parser = PolishLegalParser()
    
    def test_detect_document_type_ustawa(self):
        """Test detection of ustawa (law) documents."""
        text = "Ustawa z dnia 15 marca 2023 r. o ochronie danych osobowych..."
        result = self.parser.detect_document_type(text)
        assert result == LegalDocumentType.USTAWA
    
    def test_detect_document_type_rozporzadzenie(self):
        """Test detection of rozporządzenie (regulation) documents."""
        text = "Rozporządzenie Ministra Sprawiedliwości z dnia 10 lutego 2023 r..."
        result = self.parser.detect_document_type(text)
        assert result == LegalDocumentType.ROZPORZADZENIE
    
    def test_detect_document_type_kodeks(self):
        """Test detection of kodeks (code) documents."""
        text = "Kodeks cywilny - przepisy ogólne..."
        result = self.parser.detect_document_type(text)
        assert result == LegalDocumentType.KODEKS
    
    def test_detect_document_type_konstytucja(self):
        """Test detection of konstytucja (constitution) documents."""
        text = "Konstytucja Rzeczypospolitej Polskiej z dnia 2 kwietnia 1997 r..."
        result = self.parser.detect_document_type(text)
        assert result == LegalDocumentType.KONSTYTUCJA
    
    def test_detect_document_type_orzeczenie(self):
        """Test detection of orzeczenie (court ruling) documents."""
        text = "Wyrok Sądu Najwyższego z dnia 15 marca 2023 r..."
        result = self.parser.detect_document_type(text)
        assert result == LegalDocumentType.ORZECZENIE
    
    def test_detect_document_type_none(self):
        """Test handling of unrecognized document types."""
        text = "This is just regular text without legal document markers."
        result = self.parser.detect_document_type(text)
        assert result is None
    
    def test_extract_document_metadata(self):
        """Test metadata extraction."""
        text = """Ustawa z dnia 15 marca 2023 r.
        o ochronie danych osobowych
        (Dz.U. nr 45, poz. 123)"""
        
        metadata = self.parser.extract_document_metadata(text)
        
        assert 'date' in metadata
        assert '15 marca 2023' in metadata['date']
        assert 'journal_number' in metadata
        assert metadata['journal_number'] == '45'
        assert 'position' in metadata
        assert metadata['position'] == '123'
    
    def test_parse_provisions_articles(self):
        """Test parsing of article provisions."""
        text = """Art. 1. Pierwszy artykuł o podstawowych definicjach.
        Art. 2. Drugi artykuł o zakresie stosowania."""
        
        provisions = self.parser.parse_provisions(text)
        
        assert len(provisions) >= 2
        
        # Find article provisions
        articles = [p for p in provisions if p.provision_type == 'article']
        assert len(articles) >= 2
        
        # Check content
        assert any('1' in p.number for p in articles)
        assert any('2' in p.number for p in articles)
        assert any('podstawowych definicjach' in p.content for p in articles)
    
    def test_parse_provisions_paragraphs(self):
        """Test parsing of paragraph provisions."""
        text = """§ 15. Pierwszy paragraf.
        § 20. Drugi paragraf z treścią."""
        
        provisions = self.parser.parse_provisions(text)
        paragraphs = [p for p in provisions if p.provision_type == 'paragraph']
        
        assert len(paragraphs) >= 2
        assert any('15' in p.number for p in paragraphs)
        assert any('20' in p.number for p in paragraphs)
    
    def test_parse_provisions_chapters(self):
        """Test parsing of chapter provisions."""
        text = """Rozdział 1. Przepisy ogólne
        Rozdz. II Przepisy szczególne"""
        
        provisions = self.parser.parse_provisions(text)
        chapters = [p for p in provisions if p.provision_type == 'chapter']
        
        assert len(chapters) >= 2
        assert any('1' in p.number for p in chapters)
        assert any('II' in p.number for p in chapters)
    
    def test_extract_references_simple(self):
        """Test extraction of simple legal references."""
        text = "Zgodnie z art. 123 kodeksu cywilnego oraz § 45 rozporządzenia."
        
        references = self.parser.extract_references(text)
        
        assert len(references) >= 2
        
        # Check article reference
        article_refs = [r for r in references if r.article]
        assert len(article_refs) >= 1
        assert any('123' in r.article for r in article_refs)
    
    def test_extract_references_complex(self):
        """Test extraction of complex legal references."""
        text = "Art. 123 ust. 2 pkt 3 lit. a oraz art. 456 ust. 1"
        
        references = self.parser.extract_references(text)
        
        assert len(references) >= 2
        
        # Find the complex reference
        complex_refs = [r for r in references if r.point or r.letter]
        assert len(complex_refs) >= 1
        
        # Check that complex structure is parsed
        complex_ref = complex_refs[0]
        if complex_ref.point:
            assert '3' in complex_ref.point
        if complex_ref.letter:
            assert 'a' in complex_ref.letter
    
    def test_parse_document_structure_complete(self):
        """Test complete document structure parsing."""
        text = """Ustawa z dnia 15 marca 2023 r.
        o testowaniu
        (Dz.U. nr 45, poz. 123)
        
        Rozdział I
        Przepisy ogólne
        
        Art. 1. Definicje podstawowe.
        Art. 2. Zgodnie z art. 1 ust. 2...
        
        § 1. Dodatkowe przepisy."""
        
        structure = self.parser.parse_document_structure(text)
        
        # Check all components are present
        assert 'document_type' in structure
        assert 'metadata' in structure
        assert 'provisions' in structure
        assert 'references' in structure
        assert 'structure_elements' in structure
        
        # Check document type
        assert structure['document_type'] == LegalDocumentType.USTAWA
        
        # Check metadata
        assert '15 marca 2023' in structure['metadata']['date']
        
        # Check provisions
        assert len(structure['provisions']) >= 3  # Chapter, articles, paragraph
        
        # Check references
        assert len(structure['references']) >= 1  # Art. 1 ust. 2 reference
        
        # Check structure elements
        elements = structure['structure_elements']
        assert len(elements['chapters']) >= 1
        assert len(elements['articles']) >= 2
        assert len(elements['paragraphs']) >= 1


class TestLegalDocumentAnalyzer:
    """Test high-level document analysis."""
    
    def setup_method(self):
        self.analyzer = LegalDocumentAnalyzer()
    
    def test_analyze_document_complete(self):
        """Test complete document analysis."""
        text = """Ustawa z dnia 15 marca 2023 r.
        o testowaniu
        
        Art. 1. Pierwszy artykuł z definicjami podstawowymi.
        Art. 2. Drugi artykuł odnoszący się do art. 1.
        
        § 1. Paragraf wykonawczy."""
        
        analysis = self.analyzer.analyze_document(text)
        
        # Check main structure
        assert 'document_info' in analysis
        assert 'structure' in analysis
        assert 'provisions' in analysis
        assert 'references' in analysis
        
        # Check document info
        doc_info = analysis['document_info']
        assert doc_info['type'] == LegalDocumentType.USTAWA
        assert doc_info['length'] > 0
        assert doc_info['word_count'] > 0
        
        # Check provisions analysis
        provisions = analysis['provisions']
        assert provisions['count'] >= 3
        assert 'types' in provisions
        assert provisions['types']['article'] >= 2
        
        # Check references analysis
        references = analysis['references']
        assert references['count'] >= 1  # Reference to art. 1
    
    def test_analyze_document_empty(self):
        """Test analysis of empty document."""
        analysis = self.analyzer.analyze_document("")
        
        assert analysis == {}
    
    def test_count_provision_types(self):
        """Test provision type counting."""
        provisions = [
            LegalProvision('article', '1', content='Test'),
            LegalProvision('article', '2', content='Test'),
            LegalProvision('paragraph', '1', content='Test'),
            LegalProvision('chapter', 'I', content='Test')
        ]
        
        counts = self.analyzer._count_provision_types(provisions)
        
        assert counts['article'] == 2
        assert counts['paragraph'] == 1
        assert counts['chapter'] == 1
    
    def test_analyze_complex_document(self):
        """Test analysis of complex legal document."""
        text = """Kodeks cywilny
        
        Księga pierwsza
        Część ogólna
        
        Tytuł I
        Przepisy wstępne
        
        Rozdział 1
        Zasady ogólne
        
        Art. 1. § 1. Przepisy Kodeksu cywilnego regulują stosunki cywilnoprawne.
        § 2. Stosowanie przepisów art. 1 § 1 wymaga uwzględnienia...
        
        Art. 2. Dalsze przepisy."""
        
        analysis = self.analyzer.analyze_document(text)
        
        # Should detect kodeks
        assert analysis['document_info']['type'] == LegalDocumentType.KODEKS
        
        # Should find multiple structural elements
        structure = analysis['structure']
        assert len(structure['chapters']) >= 1
        assert len(structure['articles']) >= 2
        assert len(structure['paragraphs']) >= 2
        
        # Should count provisions correctly
        assert analysis['provisions']['count'] >= 4
        
        # Should find internal references
        assert analysis['references']['count'] >= 1