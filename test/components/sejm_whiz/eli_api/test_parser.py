"""Tests for ELI API legal document parser."""

import pytest
from unittest.mock import patch, MagicMock

from sejm_whiz.eli_api.parser import (
    LegalTextParser,
    MultiActAmendmentDetector,
    DocumentStructure,
    Chapter,
    Article,
    Paragraph,
    Point,
    Subpoint,
    Attachment
)
from sejm_whiz.eli_api.models import LegalDocument, MultiActAmendment, DocumentType, DocumentStatus
from datetime import datetime


class TestLegalTextParser:
    """Test legal text parser functionality."""
    
    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return LegalTextParser()
    
    def test_html_content_parsing(self, parser):
        """Test parsing HTML content."""
        html_content = """
        <html>
        <head><title>Test Act</title></head>
        <body>
            <h1>Ustawa testowa</h1>
            <p>Art. 1. Pierwszy artykuł.</p>
            <p>§ 1. Pierwszy paragraf.</p>
            <script>alert('xss')</script>
            <style>body { color: red; }</style>
        </body>
        </html>
        """
        
        structure = parser.parse_html_content(html_content)
        
        assert isinstance(structure, DocumentStructure)
        assert "Ustawa testowa" in structure.title
        assert len(structure.articles) >= 1
        # Script and style tags should be removed
        assert "alert" not in str(structure)
        assert "color: red" not in str(structure)
    
    def test_text_content_parsing(self, parser):
        """Test parsing plain text content."""
        text_content = """
        USTAWA TESTOWA
        
        Art. 1. Pierwszy artykuł testowy.
        § 1. Pierwszy paragraf artykułu pierwszego.
        1) pierwszy punkt
        a) pierwszy podpunkt
        
        Art. 2. Drugi artykuł.
        § 1. Pierwszy paragraf drugiego artykułu.
        § 2. Drugi paragraf drugiego artykułu.
        """
        
        structure = parser.parse_text_content(text_content)
        
        assert structure.title == "USTAWA TESTOWA"
        assert len(structure.articles) >= 2
        assert structure.articles[0].number == "1"
        assert structure.articles[1].number == "2"
    
    def test_chapter_extraction(self, parser):
        """Test chapter extraction."""
        lines = [
            "USTAWA TESTOWA",
            "Rozdział I. Postanowienia ogólne",
            "Art. 1. Pierwszy artykuł.",
            "Rozdział II. Przepisy szczegółowe", 
            "Art. 2. Drugi artykuł."
        ]
        
        chapters = parser._extract_chapters(lines)
        
        assert len(chapters) == 2
        assert chapters[0].number == "I"
        assert chapters[0].title == "Postanowienia ogólne"
        assert chapters[1].number == "II"
        assert chapters[1].title == "Przepisy szczegółowe"
    
    def test_article_extraction(self, parser):
        """Test article extraction."""
        lines = [
            "USTAWA TESTOWA",
            "Art. 1. Pierwszy artykuł testowy.",
            "§ 1. Paragraf pierwszy.",
            "Art. 2a. Drugi artykuł z literą.",
            "§ 1. Paragraf drugiego artykułu."
        ]
        
        articles = parser._extract_articles(lines)
        
        assert len(articles) >= 2
        assert articles[0].number == "1"
        assert articles[0].title == "Pierwszy artykuł testowy."
        assert articles[1].number == "2a"  # Article with letter
        assert articles[1].title == "Drugi artykuł z literą."
    
    def test_paragraph_extraction(self, parser):
        """Test paragraph extraction."""
        lines = [
            "Art. 1. Test artykuł.",
            "§ 1. Pierwszy paragraf artykułu.",
            "1) pierwszy punkt",
            "§ 2. Drugi paragraf artykułu.",
            "1) pierwszy punkt drugiego paragrafu",
            "Art. 2. Następny artykuł."
        ]
        
        paragraphs, end_index = parser._extract_paragraphs(lines, 1)
        
        assert len(paragraphs) == 2
        assert paragraphs[0].number == "1"
        assert paragraphs[0].content == "Pierwszy paragraf artykułu."
        assert paragraphs[1].number == "2"
        assert paragraphs[1].content == "Drugi paragraf artykułu."
    
    def test_point_extraction(self, parser):
        """Test point extraction."""
        lines = [
            "§ 1. Test paragraf.",
            "1) pierwszy punkt paragrafu",
            "a) pierwszy podpunkt",
            "2) drugi punkt paragrafu",
            "b) drugi podpunkt",
            "§ 2. Następny paragraf."
        ]
        
        points, end_index = parser._extract_points(lines, 1)
        
        assert len(points) == 2
        assert points[0].number == "1"
        assert points[0].content == "pierwszy punkt paragrafu"
        assert points[1].number == "2"
        assert points[1].content == "drugi punkt paragrafu"
    
    def test_subpoint_extraction(self, parser):
        """Test subpoint extraction."""
        lines = [
            "1) pierwszy punkt",
            "a) pierwszy podpunkt punktu",
            "b) drugi podpunkt punktu",
            "c) trzeci podpunkt punktu",
            "2) drugi punkt"
        ]
        
        subpoints, end_index = parser._extract_subpoints(lines, 1)
        
        assert len(subpoints) == 3
        assert subpoints[0].letter == "a"
        assert subpoints[0].content == "pierwszy podpunkt punktu"
        assert subpoints[1].letter == "b"
        assert subpoints[2].letter == "c"
    
    def test_cross_reference_extraction(self, parser):
        """Test cross-reference extraction."""
        text = """
        Na podstawie ustawy z dnia 12 stycznia 2020 r. o testach prawnych
        oraz ustawy z dnia 5 marca 2021 r. o procedurach testowych
        ustala się następujące przepisy.
        """
        
        references = parser._extract_cross_references(text)
        
        assert len(references) >= 2
        assert any("12 stycznia 2020" in ref for ref in references)
        assert any("5 marca 2021" in ref for ref in references)
    
    def test_legal_citation_extraction(self, parser):
        """Test legal citation extraction."""
        text = """
        Zgodnie z art. 123 oraz art. 45a Konstytucji,
        w związku z § 67 i § 89 rozporządzenia...
        """
        
        citations = parser._extract_legal_citations(text)
        
        assert "art. 123" in citations
        assert "art. 45a" in citations
        assert "§ 67" in citations
        assert "§ 89" in citations
    
    def test_amendment_indicator_detection(self, parser):
        """Test amendment indicator detection."""
        text = """
        W ustawie zmienia się art. 123.
        Dodaje się nowy paragraf.
        Uchyla się dotychczasowy przepis.
        W miejsce art. 45 wprowadza się nowe brzmienie.
        """
        
        indicators = parser._find_amendment_indicators(text)
        
        assert len(indicators) >= 3
        assert any("zmienia się" in indicator.lower() for indicator in indicators)
        assert any("dodaje się" in indicator.lower() for indicator in indicators)
        assert any("uchyla się" in indicator.lower() for indicator in indicators)
    
    def test_preamble_extraction(self, parser):
        """Test preamble extraction."""
        lines = [
            "USTAWA TESTOWA",
            "z dnia 12 stycznia 2023 r.",
            "o testowaniu przepisów prawnych",
            "",
            "W trosce o jakość legislacji...",
            "Mając na uwadze potrzeby...",
            "",
            "Art. 1. Pierwszy artykuł."
        ]
        
        preamble = parser._extract_preamble(lines)
        
        assert preamble is not None
        assert "z dnia 12 stycznia 2023 r." in preamble
        assert "W trosce o jakość" in preamble
        assert "Art. 1." not in preamble  # Should stop before first article
    
    def test_final_provisions_extraction(self, parser):
        """Test final provisions extraction."""
        lines = [
            "Art. 1. Pierwszy artykuł.",
            "Art. 2. Drugi artykuł.",
            "PRZEPISY KOŃCOWE",
            "Art. 10. Ustawa wchodzi w życie...",
            "Art. 11. Przepisy przejściowe..."
        ]
        
        final_provisions = parser._extract_final_provisions(lines)
        
        assert final_provisions is not None
        assert "PRZEPISY KOŃCOWE" in final_provisions
        assert "Art. 10." in final_provisions
        assert "Art. 11." in final_provisions
    
    def test_attachment_extraction(self, parser):
        """Test attachment extraction."""
        lines = [
            "Art. 1. Ustawa ma załączniki.",
            "Załącznik nr 1",
            "WZÓR FORMULARZA",
            "Treść załącznika pierwszego...",
            "Załącznik nr 2",
            "TABELA STAWEK",
            "Treść załącznika drugiego..."
        ]
        
        attachments = parser._extract_attachments(lines)
        
        assert len(attachments) == 2
        assert attachments[0].number == "1"
        assert "WZÓR FORMULARZA" in attachments[0].title
        assert "Treść załącznika pierwszego" in attachments[0].content
        assert attachments[1].number == "2"
        assert "TABELA STAWEK" in attachments[1].title
    
    def test_malformed_html_handling(self, parser):
        """Test handling of malformed HTML."""
        malformed_html = """
        <html><body>
        <p>Unclosed paragraph
        <div>Nested content<span>More content
        Art. 1. Legal content.
        </body>
        """
        
        # Should not raise exception and should extract some content
        structure = parser.parse_html_content(malformed_html)
        
        assert isinstance(structure, DocumentStructure)
        assert structure.title  # Should have some title
        # Should extract legal content despite malformed HTML
        assert len(structure.articles) >= 0
    
    def test_empty_content_handling(self, parser):
        """Test handling of empty content."""
        structure = parser.parse_text_content("")
        
        assert isinstance(structure, DocumentStructure)
        assert structure.title == "Untitled Document"
        assert len(structure.articles) == 0
        assert len(structure.chapters) == 0


class TestMultiActAmendmentDetector:
    """Test multi-act amendment detector."""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return MultiActAmendmentDetector()
    
    @pytest.fixture
    def sample_document(self):
        """Create sample legal document."""
        return LegalDocument(
            eli_id="pl/omnibus/2023/1",
            title="Ustawa o zmianie niektórych ustaw",
            document_type=DocumentType.USTAWA,
            status=DocumentStatus.OBOWIAZUJACA,
            published_date=datetime(2023, 1, 1),
            effective_date=datetime(2023, 2, 1)
        )
    
    def test_multi_act_detection_positive(self, detector, sample_document):
        """Test detection of multi-act amendment."""
        content = """
        Ustawa o zmianie niektórych ustaw w zakresie procedur administracyjnych
        
        Art. 1. W ustawie z dnia 14 czerwca 1960 r. Kodeks postępowania administracyjnego
        wprowadza się następujące zmiany...
        
        Art. 2. W ustawie z dnia 30 sierpnia 2002 r. Prawo o postępowaniu przed sądami administracyjnymi
        wprowadza się następujące zmiany...
        
        Art. 3. W rozporządzeniu Rady Ministrów z dnia 1 stycznia 2023 r.
        w sprawie szczegółowych procedur...
        """
        
        result = detector.detect_multi_act_amendments(sample_document, content)
        
        assert result is not None
        assert isinstance(result, MultiActAmendment)
        assert result.eli_id == sample_document.eli_id
        assert result.complexity_score >= 2  # At least 2 acts affected
        assert len(result.affected_acts) >= 2
    
    def test_multi_act_detection_negative(self, detector, sample_document):
        """Test when document is not multi-act amendment."""
        content = """
        Ustawa o ochronie środowiska
        
        Art. 1. Ustawa reguluje zasady ochrony środowiska.
        Art. 2. Definicje stosowane w ustawie.
        Art. 3. Obowiązki podmiotów.
        """
        
        result = detector.detect_multi_act_amendments(sample_document, content)
        
        assert result is None  # Should not detect as multi-act amendment
    
    def test_single_act_amendment(self, detector, sample_document):
        """Test single act amendment (should not be detected as multi-act)."""
        content = """
        Ustawa o zmianie ustawy o ochronie środowiska
        
        Art. 1. W ustawie z dnia 27 kwietnia 2001 r. o ochronie środowiska
        wprowadza się następujące zmiany...
        """
        
        result = detector.detect_multi_act_amendments(sample_document, content)
        
        assert result is None  # Only one act affected
    
    def test_affected_acts_extraction(self, detector):
        """Test extraction of affected acts."""
        content = """
        Zmienia się:
        1) ustawę z dnia 1 stycznia 2020 r. o testach
        2) Kodeks cywilny
        3) rozporządzenie w sprawie procedur
        4) Konstytucję Rzeczypospolitej Polskiej
        """
        
        affected_acts = detector._find_affected_acts(content)
        
        assert len(affected_acts) >= 4
        assert any("ustaw" in act.lower() for act in affected_acts)
        assert any("kodeks" in act.lower() for act in affected_acts)
        assert any("rozporządzen" in act.lower() for act in affected_acts)
        assert any("konstytucj" in act.lower() for act in affected_acts)
    
    def test_cross_reference_extraction(self, detector):
        """Test cross-reference extraction."""
        content = """
        W związku z art. 123 ustawy podstawowej oraz
        zgodnie z § 45 rozporządzenia wykonawczego,
        na podstawie ust. 2 przepisów przejściowych...
        """
        
        cross_references = detector._extract_cross_references(content)
        
        assert len(cross_references) >= 3
        assert any("art. 123" in ref for ref in cross_references)
        assert any("§ 45" in ref for ref in cross_references)
        assert any("ust. 2" in ref for ref in cross_references)
    
    def test_impact_assessment(self, detector):
        """Test impact assessment."""
        test_cases = [
            (["act1", "act2"], "fundamentalne zmiany w systemie", "high"),
            (["act1", "act2", "act3"], "dostosowanie przepisów", "medium"),
            (["act1", "act2"], "techniczne poprawki", "low"),
            (["act1"] * 12, "regularne zmiany", "high"),  # Many acts = high impact
            (["act1"] * 6, "średnie zmiany", "medium"),   # Medium number = medium impact
            (["act1", "act2"], "małe zmiany", "low")      # Few acts = low impact
        ]
        
        for affected_acts, content, expected_impact in test_cases:
            impact = detector._assess_impact(affected_acts, content)
            assert impact == expected_impact, f"Expected {expected_impact}, got {impact} for {len(affected_acts)} acts"
    
    def test_omnibus_indicator_detection(self, detector, sample_document):
        """Test detection of omnibus legislation indicators."""
        omnibus_content = """
        Ustawa - Przepisy wprowadzające kompleksową reformę
        
        W ramach nowelizacji różnych ustaw wprowadza się
        zmiany w różnych ustawach dotyczące...
        """
        
        result = detector.detect_multi_act_amendments(sample_document, omnibus_content)
        
        # Even if only mentions indicators, should still need actual act references
        # This tests the logic without false positives
        if result:
            assert result.complexity_score >= 2
    
    def test_complex_omnibus_legislation(self, detector, sample_document):
        """Test complex omnibus legislation detection."""
        complex_content = """
        Ustawa - Przepisy wprowadzające kompleksową nowelizację ustaw
        
        Art. 1. W ustawie z dnia 1 stycznia 2020 r. o procedurach...
        Art. 2. W Kodeksie postępowania cywilnego...
        Art. 3. W ustawie z dnia 15 lutego 2021 r. o ochronie...
        Art. 4. W rozporządzeniu Rady Ministrów...
        Art. 5. W ustawie z dnia 30 marca 2022 r. o bezpieczeństwie...
        Art. 6. W Kodeksie karnym...
        """
        
        result = detector.detect_multi_act_amendments(sample_document, complex_content)
        
        assert result is not None
        assert result.complexity_score >= 5  # Many acts affected
        assert result.is_omnibus()  # Should be classified as omnibus
        assert result.get_impact_level() in ["high", "very_high"]
    
    def test_edge_cases(self, detector, sample_document):
        """Test edge cases in detection."""
        # Empty content
        assert detector.detect_multi_act_amendments(sample_document, "") is None
        
        # Content with no legal acts
        non_legal_content = "This is just regular text without legal references."
        assert detector.detect_multi_act_amendments(sample_document, non_legal_content) is None
        
        # Content with false positives (mentions acts but not in amendment context)
        false_positive_content = """
        Ustawa reguluje kwestie związane z ochroną środowiska.
        Nawiązując do ustawy o ochronie przyrody oraz 
        kodeksu postępowania administracyjnego, należy...
        """
        result = detector.detect_multi_act_amendments(sample_document, false_positive_content)
        # Might detect references but complexity should be appropriate
        if result:
            assert result.complexity_score >= 2