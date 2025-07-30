"""Tests for ELI API models."""

import pytest
from datetime import datetime, timedelta
from pydantic import ValidationError

from sejm_whiz.eli_api.models import (
    LegalDocument,
    Amendment,
    DocumentSearchResult,
    MultiActAmendment,
    DocumentStatistics,
    DocumentType,
    DocumentStatus,
    AmendmentType
)


class TestDocumentType:
    """Test DocumentType enum."""
    
    def test_document_type_values(self):
        """Test document type enum values."""
        assert DocumentType.USTAWA == "ustawa"
        assert DocumentType.ROZPORZADZENIE == "rozporządzenie"
        assert DocumentType.KODEKS == "kodeks"
        assert DocumentType.KONSTYTUCJA == "konstytucja"
        assert DocumentType.DEKRET == "dekret"
        assert DocumentType.UCHWALA == "uchwała"


class TestDocumentStatus:
    """Test DocumentStatus enum."""
    
    def test_document_status_values(self):
        """Test document status enum values."""
        assert DocumentStatus.OBOWIAZUJACA == "obowiązująca"
        assert DocumentStatus.UCHYLONA == "uchylona"
        assert DocumentStatus.WYGASLA == "wygasła"
        assert DocumentStatus.PROJEKT == "projekt"


class TestAmendmentType:
    """Test AmendmentType enum."""
    
    def test_amendment_type_values(self):
        """Test amendment type enum values."""
        assert AmendmentType.NOWELIZACJA == "nowelizacja"
        assert AmendmentType.ZMIANA == "zmiana"
        assert AmendmentType.UCHYLENIE == "uchylenie"
        assert AmendmentType.DODANIE == "dodanie"


class TestLegalDocument:
    """Test LegalDocument model."""
    
    def test_minimal_valid_document(self):
        """Test creating document with minimal required fields."""
        doc = LegalDocument(
            eli_id="pl/test/2023/1",
            title="Test Act",
            document_type=DocumentType.USTAWA,
            status=DocumentStatus.OBOWIAZUJACA
        )
        
        assert doc.eli_id == "pl/test/2023/1"
        assert doc.title == "Test Act"
        assert doc.document_type == DocumentType.USTAWA
        assert doc.status == DocumentStatus.OBOWIAZUJACA
        assert doc.language == "pl"
        assert doc.format == "html"
        assert isinstance(doc.created_at, datetime)
        assert isinstance(doc.updated_at, datetime)
    
    def test_complete_document(self):
        """Test creating document with all fields."""
        published_date = datetime(2023, 1, 1)
        effective_date = datetime(2023, 2, 1)
        
        doc = LegalDocument(
            eli_id="pl/test/2023/1",
            title="Comprehensive Test Act",
            document_type=DocumentType.USTAWA,
            status=DocumentStatus.OBOWIAZUJACA,
            published_date=published_date,
            effective_date=effective_date,
            publisher="Sejm RP",
            journal_reference="Dz.U. 2023 nr 1 poz. 1",
            journal_year=2023,
            journal_number=1,
            journal_position=1,
            content_url="https://api.sejm.gov.pl/eli/pl/test/2023/1/content",
            metadata_url="https://api.sejm.gov.pl/eli/pl/test/2023/1",
            keywords=["test", "act", "legal"],
            subject_areas=["civil law", "administrative law"],
            amending_documents=["pl/amendment/2023/1"],
            amended_documents=["pl/old/2022/1"],
            language="pl",
            format="html"
        )
        
        assert doc.published_date == published_date
        assert doc.effective_date == effective_date
        assert doc.publisher == "Sejm RP"
        assert doc.journal_year == 2023
        assert len(doc.keywords) == 3
        assert len(doc.subject_areas) == 2
    
    def test_eli_id_validation(self):
        """Test ELI ID validation."""
        # Valid ELI IDs
        valid_ids = ["pl/test/2023/1", "PL/TEST/2023/1", "pl/complex/path/2023/1"]
        
        for eli_id in valid_ids:
            doc = LegalDocument(
                eli_id=eli_id,
                title="Test",
                document_type=DocumentType.USTAWA,
                status=DocumentStatus.OBOWIAZUJACA
            )
            assert doc.eli_id == eli_id.strip()
        
        # Invalid ELI IDs
        invalid_ids = ["", "   ", "us/test/2023/1", "test/2023/1", None]
        
        for eli_id in invalid_ids:
            with pytest.raises(ValidationError):
                LegalDocument(
                    eli_id=eli_id,
                    title="Test",
                    document_type=DocumentType.USTAWA,
                    status=DocumentStatus.OBOWIAZUJACA
                )
    
    def test_title_validation(self):
        """Test title validation."""
        # Valid titles
        doc = LegalDocument(
            eli_id="pl/test/2023/1",
            title="  Valid Title  ",  # Should be trimmed
            document_type=DocumentType.USTAWA,
            status=DocumentStatus.OBOWIAZUJACA
        )
        assert doc.title == "Valid Title"
        
        # Invalid titles
        invalid_titles = ["", "   ", None]
        
        for title in invalid_titles:
            with pytest.raises(ValidationError):
                LegalDocument(
                    eli_id="pl/test/2023/1",
                    title=title,
                    document_type=DocumentType.USTAWA,
                    status=DocumentStatus.OBOWIAZUJACA
                )
    
    def test_journal_year_validation(self):
        """Test journal year validation."""
        current_year = datetime.now().year
        
        # Valid years
        valid_years = [1918, 2000, current_year, current_year + 1]
        
        for year in valid_years:
            doc = LegalDocument(
                eli_id="pl/test/2023/1",
                title="Test",
                document_type=DocumentType.USTAWA,
                status=DocumentStatus.OBOWIAZUJACA,
                journal_year=year
            )
            assert doc.journal_year == year
        
        # Invalid years
        invalid_years = [1917, current_year + 2, -1, 0]
        
        for year in invalid_years:
            with pytest.raises(ValidationError):
                LegalDocument(
                    eli_id="pl/test/2023/1",
                    title="Test",
                    document_type=DocumentType.USTAWA,
                    status=DocumentStatus.OBOWIAZUJACA,
                    journal_year=year
                )
    
    def test_from_api_response(self):
        """Test creating document from API response."""
        api_data = {
            "eli_id": "pl/test/2023/1",
            "title": "Test Act from API",
            "type": "ustawa",
            "status": "obowiązująca",
            "published_date": "2023-01-01T00:00:00Z",
            "effective_date": "2023-02-01T10:30:00+01:00",
            "publisher": "Sejm RP",
            "journal_year": 2023,
            "keywords": ["test", "api"],
            "subject_areas": ["civil law"]
        }
        
        doc = LegalDocument.from_api_response(api_data)
        
        assert doc.eli_id == "pl/test/2023/1"
        assert doc.title == "Test Act from API"
        assert doc.document_type == DocumentType.USTAWA
        assert doc.status == DocumentStatus.OBOWIAZUJACA
        assert doc.published_date.year == 2023
        assert doc.effective_date.year == 2023
        assert doc.publisher == "Sejm RP"
        assert len(doc.keywords) == 2
    
    def test_from_api_response_invalid_type(self):
        """Test handling invalid document type in API response."""
        api_data = {
            "eli_id": "pl/test/2023/1",
            "title": "Test Act",
            "type": "unknown_type",  # Invalid type
            "status": "obowiązująca"
        }
        
        doc = LegalDocument.from_api_response(api_data)
        
        # Should default to USTAWA
        assert doc.document_type == DocumentType.USTAWA
    
    def test_from_api_response_invalid_status(self):
        """Test handling invalid status in API response."""
        api_data = {
            "eli_id": "pl/test/2023/1",
            "title": "Test Act",
            "type": "ustawa",
            "status": "unknown_status"  # Invalid status
        }
        
        doc = LegalDocument.from_api_response(api_data)
        
        # Should default to OBOWIAZUJACA
        assert doc.status == DocumentStatus.OBOWIAZUJACA
    
    def test_to_dict(self):
        """Test converting document to dictionary."""
        doc = LegalDocument(
            eli_id="pl/test/2023/1",
            title="Test Act",
            document_type=DocumentType.USTAWA,
            status=DocumentStatus.OBOWIAZUJACA,
            keywords=["test", "dict"]
        )
        
        doc_dict = doc.to_dict()
        
        assert doc_dict["eli_id"] == "pl/test/2023/1"
        assert doc_dict["title"] == "Test Act"
        assert doc_dict["document_type"] == "ustawa"
        assert doc_dict["status"] == "obowiązująca"
        assert "created_at" not in doc_dict  # Should be excluded
        assert "updated_at" not in doc_dict  # Should be excluded
    
    def test_is_in_force(self):
        """Test checking if document is in force."""
        # Document in force
        doc_in_force = LegalDocument(
            eli_id="pl/test/2023/1",
            title="Test Act",
            document_type=DocumentType.USTAWA,
            status=DocumentStatus.OBOWIAZUJACA
        )
        assert doc_in_force.is_in_force() is True
        
        # Document repealed
        doc_repealed = LegalDocument(
            eli_id="pl/test/2023/2",
            title="Repealed Act",
            document_type=DocumentType.USTAWA,
            status=DocumentStatus.UCHYLONA
        )
        assert doc_repealed.is_in_force() is False
        
        # Document with repeal date
        doc_with_repeal = LegalDocument(
            eli_id="pl/test/2023/3",
            title="Test Act",
            document_type=DocumentType.USTAWA,
            status=DocumentStatus.OBOWIAZUJACA,
            repeal_date=datetime(2023, 12, 31)
        )
        assert doc_with_repeal.is_in_force() is False
    
    def test_is_recent(self):
        """Test checking if document is recent."""
        # Recent document
        recent_date = datetime.now() - timedelta(days=10)
        doc_recent = LegalDocument(
            eli_id="pl/test/2023/1",
            title="Recent Act",
            document_type=DocumentType.USTAWA,
            status=DocumentStatus.OBOWIAZUJACA,
            published_date=recent_date
        )
        assert doc_recent.is_recent(days=30) is True
        assert doc_recent.is_recent(days=5) is False
        
        # Old document
        old_date = datetime.now() - timedelta(days=100)
        doc_old = LegalDocument(
            eli_id="pl/test/2023/2",
            title="Old Act",
            document_type=DocumentType.USTAWA,
            status=DocumentStatus.OBOWIAZUJACA,
            published_date=old_date
        )
        assert doc_old.is_recent(days=30) is False
        
        # Document without publication date
        doc_no_date = LegalDocument(
            eli_id="pl/test/2023/3",
            title="No Date Act",
            document_type=DocumentType.USTAWA,
            status=DocumentStatus.OBOWIAZUJACA
        )
        assert doc_no_date.is_recent(days=30) is False


class TestAmendment:
    """Test Amendment model."""
    
    def test_minimal_valid_amendment(self):
        """Test creating amendment with minimal required fields."""
        amendment = Amendment(
            eli_id="pl/amendment/2023/1",
            target_eli_id="pl/target/2023/1",
            amendment_type=AmendmentType.NOWELIZACJA,
            title="Test Amendment"
        )
        
        assert amendment.eli_id == "pl/amendment/2023/1"
        assert amendment.target_eli_id == "pl/target/2023/1"
        assert amendment.amendment_type == AmendmentType.NOWELIZACJA
        assert amendment.title == "Test Amendment"
    
    def test_complete_amendment(self):
        """Test creating amendment with all fields."""
        published_date = datetime(2023, 1, 1)
        effective_date = datetime(2023, 2, 1)
        
        amendment = Amendment(
            eli_id="pl/amendment/2023/1",
            target_eli_id="pl/target/2023/1",
            amendment_type=AmendmentType.NOWELIZACJA,
            title="Comprehensive Amendment",
            description="Amendment description",
            published_date=published_date,
            effective_date=effective_date,
            affected_articles=["1", "2", "3a"],
            affected_paragraphs=["1", "2"],
            change_summary="Summary of changes",
            legal_basis="Article 5 of Constitution"
        )
        
        assert amendment.description == "Amendment description"
        assert amendment.published_date == published_date
        assert amendment.effective_date == effective_date
        assert len(amendment.affected_articles) == 3
        assert len(amendment.affected_paragraphs) == 2
        assert amendment.change_summary == "Summary of changes"
        assert amendment.legal_basis == "Article 5 of Constitution"
    
    def test_eli_id_validation(self):
        """Test ELI ID validation for amendments."""
        # Invalid ELI IDs
        invalid_ids = ["", "   ", "us/test/2023/1", None]
        
        for eli_id in invalid_ids:
            with pytest.raises(ValidationError):
                Amendment(
                    eli_id=eli_id,
                    target_eli_id="pl/target/2023/1",
                    amendment_type=AmendmentType.NOWELIZACJA,
                    title="Test"
                )
            
            with pytest.raises(ValidationError):
                Amendment(
                    eli_id="pl/amendment/2023/1",
                    target_eli_id=eli_id,
                    amendment_type=AmendmentType.NOWELIZACJA,
                    title="Test"
                )
    
    def test_from_api_response(self):
        """Test creating amendment from API response."""
        api_data = {
            "eli_id": "pl/amendment/2023/1",
            "target_eli_id": "pl/target/2023/1",
            "amendment_type": "nowelizacja",
            "title": "API Amendment",
            "published_date": "2023-01-01T00:00:00Z",
            "affected_articles": ["1", "2"],
            "change_summary": "API summary"
        }
        
        amendment = Amendment.from_api_response(api_data)
        
        assert amendment.eli_id == "pl/amendment/2023/1"
        assert amendment.target_eli_id == "pl/target/2023/1"
        assert amendment.amendment_type == AmendmentType.NOWELIZACJA
        assert amendment.title == "API Amendment"
        assert len(amendment.affected_articles) == 2
        assert amendment.change_summary == "API summary"


class TestDocumentSearchResult:
    """Test DocumentSearchResult model."""
    
    def test_search_result_creation(self):
        """Test creating search result."""
        documents = [
            LegalDocument(
                eli_id="pl/test/2023/1",
                title="Test Act 1",
                document_type=DocumentType.USTAWA,
                status=DocumentStatus.OBOWIAZUJACA
            ),
            LegalDocument(
                eli_id="pl/test/2023/2",
                title="Test Act 2",
                document_type=DocumentType.USTAWA,
                status=DocumentStatus.OBOWIAZUJACA
            )
        ]
        
        result = DocumentSearchResult(
            documents=documents,
            total=10,
            offset=0,
            limit=2
        )
        
        assert len(result.documents) == 2
        assert result.total == 10
        assert result.offset == 0
        assert result.limit == 2
    
    def test_has_more(self):
        """Test checking if there are more results."""
        documents = [
            LegalDocument(
                eli_id="pl/test/2023/1",
                title="Test Act",
                document_type=DocumentType.USTAWA,
                status=DocumentStatus.OBOWIAZUJACA
            )
        ]
        
        # Has more results
        result_has_more = DocumentSearchResult(
            documents=documents,
            total=10,
            offset=0,
            limit=1
        )
        assert result_has_more.has_more() is True
        
        # No more results
        result_no_more = DocumentSearchResult(
            documents=documents,
            total=1,
            offset=0,
            limit=1
        )
        assert result_no_more.has_more() is False
    
    def test_next_offset(self):
        """Test calculating next offset."""
        result = DocumentSearchResult(
            documents=[],
            total=100,
            offset=20,
            limit=10
        )
        
        assert result.next_offset() == 30


class TestMultiActAmendment:
    """Test MultiActAmendment model."""
    
    def test_minimal_valid_multi_act_amendment(self):
        """Test creating multi-act amendment with minimal fields."""
        amendment = MultiActAmendment(
            eli_id="pl/omnibus/2023/1",
            title="Omnibus Amendment",
            affected_acts=["pl/act1/2020/1", "pl/act2/2021/1"],
            complexity_score=2
        )
        
        assert amendment.eli_id == "pl/omnibus/2023/1"
        assert amendment.title == "Omnibus Amendment"
        assert len(amendment.affected_acts) == 2
        assert amendment.complexity_score == 2
    
    def test_complexity_score_validation(self):
        """Test complexity score validation."""
        # Valid complexity score
        amendment = MultiActAmendment(
            eli_id="pl/omnibus/2023/1",
            title="Test",
            affected_acts=["pl/act1/2020/1", "pl/act2/2021/1"],
            complexity_score=2
        )
        assert amendment.complexity_score == 2
        
        # Invalid complexity score (too low)
        with pytest.raises(ValidationError):
            MultiActAmendment(
                eli_id="pl/omnibus/2023/1",
                title="Test",
                affected_acts=["pl/act1/2020/1"],
                complexity_score=1
            )
    
    def test_is_omnibus(self):
        """Test checking if amendment is omnibus legislation."""
        # Not omnibus (affects 2 acts)
        amendment_simple = MultiActAmendment(
            eli_id="pl/amendment/2023/1",
            title="Simple Amendment",
            affected_acts=["pl/act1/2020/1", "pl/act2/2021/1"],
            complexity_score=2
        )
        assert amendment_simple.is_omnibus() is False
        
        # Omnibus (affects 3+ acts)
        amendment_omnibus = MultiActAmendment(
            eli_id="pl/omnibus/2023/1",
            title="Omnibus Amendment",
            affected_acts=["pl/act1/2020/1", "pl/act2/2021/1", "pl/act3/2022/1"],
            complexity_score=3
        )
        assert amendment_omnibus.is_omnibus() is True
    
    def test_get_impact_level(self):
        """Test getting impact level based on complexity."""
        test_cases = [
            (2, "low"),
            (3, "medium"),
            (4, "medium"),
            (5, "high"),
            (9, "high"),
            (10, "very_high"),
            (15, "very_high")
        ]
        
        for complexity, expected_level in test_cases:
            affected_acts = [f"pl/act{i}/2020/1" for i in range(complexity)]
            amendment = MultiActAmendment(
                eli_id="pl/test/2023/1",
                title="Test Amendment",
                affected_acts=affected_acts,
                complexity_score=complexity
            )
            assert amendment.get_impact_level() == expected_level


class TestDocumentStatistics:
    """Test DocumentStatistics model."""
    
    def test_document_statistics_creation(self):
        """Test creating document statistics."""
        stats = DocumentStatistics(
            total_documents=1000,
            documents_by_type={
                DocumentType.USTAWA: 500,
                DocumentType.ROZPORZADZENIE: 400,
                DocumentType.KODEKS: 100
            },
            documents_by_status={
                DocumentStatus.OBOWIAZUJACA: 800,
                DocumentStatus.UCHYLONA: 200
            },
            documents_by_year={
                2023: 100,
                2022: 150,
                2021: 200
            },
            recent_documents_count=50,
            active_documents_count=800
        )
        
        assert stats.total_documents == 1000
        assert stats.documents_by_type[DocumentType.USTAWA] == 500
        assert stats.documents_by_status[DocumentStatus.OBOWIAZUJACA] == 800
        assert stats.documents_by_year[2023] == 100
        assert stats.recent_documents_count == 50
        assert stats.active_documents_count == 800
        assert isinstance(stats.last_updated, datetime)