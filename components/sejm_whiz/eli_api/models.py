"""Pydantic models for ELI API legal documents."""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class DocumentType(str, Enum):
    """Legal document types."""

    USTAWA = "ustawa"  # Act/Law
    ROZPORZADZENIE = "rozporządzenie"  # Regulation
    KODEKS = "kodeks"  # Code
    KONSTYTUCJA = "konstytucja"  # Constitution
    DEKRET = "dekret"  # Decree
    UCHWALA = "uchwała"  # Resolution


class DocumentStatus(str, Enum):
    """Document status."""

    OBOWIAZUJACA = "obowiązująca"  # In force
    UCHYLONA = "uchylona"  # Repealed
    WYGASLA = "wygasła"  # Expired
    PROJEKT = "projekt"  # Draft


class AmendmentType(str, Enum):
    """Amendment types."""

    NOWELIZACJA = "nowelizacja"  # Amendment
    ZMIANA = "zmiana"  # Change
    UCHYLENIE = "uchylenie"  # Repeal
    DODANIE = "dodanie"  # Addition


class LegalDocument(BaseModel):
    """Model for legal document from ELI API."""

    eli_id: str = Field(..., description="European Legislation Identifier")
    title: str = Field(..., description="Document title")
    document_type: DocumentType = Field(..., description="Type of legal document")
    status: DocumentStatus = Field(..., description="Current status of document")

    published_date: Optional[datetime] = Field(None, description="Publication date")
    effective_date: Optional[datetime] = Field(
        None, description="Date when document becomes effective"
    )
    repeal_date: Optional[datetime] = Field(
        None, description="Date when document was repealed"
    )

    publisher: Optional[str] = Field(None, description="Publishing authority")
    journal_reference: Optional[str] = Field(
        None, description="Official journal reference"
    )
    journal_year: Optional[int] = Field(None, description="Journal year")
    journal_number: Optional[int] = Field(None, description="Journal number")
    journal_position: Optional[int] = Field(None, description="Position in journal")

    content_url: Optional[str] = Field(None, description="URL to document content")
    metadata_url: Optional[str] = Field(None, description="URL to document metadata")

    keywords: List[str] = Field(default_factory=list, description="Document keywords")
    subject_areas: List[str] = Field(
        default_factory=list, description="Legal subject areas"
    )

    amending_documents: List[str] = Field(
        default_factory=list, description="ELI IDs of amending documents"
    )
    amended_documents: List[str] = Field(
        default_factory=list, description="ELI IDs of amended documents"
    )

    language: str = Field(default="pl", description="Document language")
    format: str = Field(default="html", description="Content format")

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @field_validator("eli_id")
    @classmethod
    def validate_eli_id(cls, v):
        """Validate ELI ID format."""
        if not v or len(v.strip()) == 0:
            raise ValueError("ELI ID cannot be empty")

        # Polish ELI ID format: DU/YYYY/NUMBER or MP/YYYY/NUMBER
        # Where DU = Dziennik Ustaw, MP = Monitor Polski
        stripped = v.strip()
        if not (stripped.startswith(("DU/", "MP/")) and len(stripped.split("/")) >= 3):
            # More flexible validation - accept any reasonable ELI ID format
            if "/" not in stripped:
                raise ValueError("ELI ID must contain at least one '/' separator")

        return stripped

    @field_validator("title")
    @classmethod
    def validate_title(cls, v):
        """Validate document title."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Document title cannot be empty")

        return v.strip()

    @field_validator("journal_year")
    @classmethod
    def validate_journal_year(cls, v):
        """Validate journal year."""
        if v is not None:
            current_year = datetime.now().year
            if v < 1918 or v > current_year + 1:  # Poland regained independence in 1918
                raise ValueError(
                    f"Journal year must be between 1918 and {current_year + 1}"
                )
        return v

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "LegalDocument":
        """Create LegalDocument from ELI API response."""

        # Parse dates - API uses different field names
        published_date = None
        for date_field in ["promulgation", "published_date", "announcementDate"]:
            if data.get(date_field):
                try:
                    date_str = data[date_field]
                    if "T" in date_str:
                        published_date = datetime.fromisoformat(
                            date_str.replace("Z", "+00:00")
                        )
                    else:
                        published_date = datetime.strptime(date_str, "%Y-%m-%d")
                    break
                except (ValueError, AttributeError):
                    continue

        effective_date = None
        for date_field in ["entryIntoForce", "effective_date", "validFrom"]:
            if data.get(date_field):
                try:
                    date_str = data[date_field]
                    if "T" in date_str:
                        effective_date = datetime.fromisoformat(
                            date_str.replace("Z", "+00:00")
                        )
                    else:
                        effective_date = datetime.strptime(date_str, "%Y-%m-%d")
                    break
                except (ValueError, AttributeError):
                    continue

        repeal_date = None
        if data.get("repeal_date"):
            try:
                date_str = data["repeal_date"]
                if "T" in date_str:
                    repeal_date = datetime.fromisoformat(
                        date_str.replace("Z", "+00:00")
                    )
                else:
                    repeal_date = datetime.strptime(date_str, "%Y-%m-%d")
            except (ValueError, AttributeError):
                pass

        # Parse document type - API uses Polish names
        doc_type_str = data.get("type", "").lower()
        try:
            document_type = DocumentType(doc_type_str)
        except ValueError:
            # Map common Polish document types
            type_mapping = {
                "rozporządzenie": DocumentType.ROZPORZADZENIE,
                "ustawa": DocumentType.USTAWA,
                "uchwała": DocumentType.UCHWALA,
                "obwieszczenie": DocumentType.ROZPORZADZENIE,  # Treat as regulation
                "komunikat": DocumentType.ROZPORZADZENIE,
                "postanowienie": DocumentType.ROZPORZADZENIE,
                "zarządzenie": DocumentType.ROZPORZADZENIE,
            }
            document_type = type_mapping.get(doc_type_str, DocumentType.USTAWA)

        # Parse status - API uses Polish names
        status_str = data.get("status", "obowiązujący").lower()
        status_mapping = {
            "obowiązujący": DocumentStatus.OBOWIAZUJACA,
            "akt jednorazowy": DocumentStatus.OBOWIAZUJACA,
            "akt indywidualny": DocumentStatus.OBOWIAZUJACA,
            "akt objęty tekstem jednolitym": DocumentStatus.OBOWIAZUJACA,
            "bez statusu": DocumentStatus.OBOWIAZUJACA,
            "uchylony": DocumentStatus.UCHYLONA,
            "wygasły": DocumentStatus.WYGASLA,
        }
        status = status_mapping.get(status_str, DocumentStatus.OBOWIAZUJACA)

        # Extract keywords from API response
        keywords = []
        if data.get("keywords"):
            keywords.extend(data["keywords"])
        if data.get("keywordsNames"):
            keywords.extend(data["keywordsNames"])

        return cls(
            eli_id=data.get("ELI", data.get("eli_id", "")),
            title=data.get("title", ""),
            document_type=document_type,
            status=status,
            published_date=published_date,
            effective_date=effective_date,
            repeal_date=repeal_date,
            publisher=data.get("publisher"),
            journal_reference=data.get("displayAddress", data.get("journal_reference")),
            journal_year=data.get("year", data.get("journal_year")),
            journal_number=data.get("volume", data.get("journal_number")),
            journal_position=data.get("pos", data.get("journal_position")),
            content_url=data.get("content_url"),
            metadata_url=data.get("metadata_url"),
            keywords=keywords,
            subject_areas=data.get("subject_areas", []),
            amending_documents=data.get("amending_documents", []),
            amended_documents=data.get("amended_documents", []),
            language=data.get("language", "pl"),
            format=data.get("format", "html"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return self.model_dump(exclude={"created_at", "updated_at"})

    def is_in_force(self) -> bool:
        """Check if document is currently in force."""
        return self.status == DocumentStatus.OBOWIAZUJACA and self.repeal_date is None

    def is_recent(self, days: int = 30) -> bool:
        """Check if document was published recently."""
        if not self.published_date:
            return False

        cutoff_date = datetime.now() - timedelta(days=days)
        return self.published_date >= cutoff_date


class Amendment(BaseModel):
    """Model for legal document amendment."""

    eli_id: str = Field(..., description="ELI ID of the amendment")
    target_eli_id: str = Field(..., description="ELI ID of the amended document")
    amendment_type: AmendmentType = Field(..., description="Type of amendment")

    title: str = Field(..., description="Amendment title")
    description: Optional[str] = Field(None, description="Amendment description")

    published_date: Optional[datetime] = Field(
        None, description="Amendment publication date"
    )
    effective_date: Optional[datetime] = Field(
        None, description="Date when amendment becomes effective"
    )

    affected_articles: List[str] = Field(
        default_factory=list, description="List of affected articles"
    )
    affected_paragraphs: List[str] = Field(
        default_factory=list, description="List of affected paragraphs"
    )

    change_summary: Optional[str] = Field(None, description="Summary of changes")
    legal_basis: Optional[str] = Field(None, description="Legal basis for amendment")

    @field_validator("eli_id", "target_eli_id")
    @classmethod
    def validate_eli_ids(cls, v):
        """Validate ELI ID format."""
        if not v or len(v.strip()) == 0:
            raise ValueError("ELI ID cannot be empty")

        if not v.startswith(("pl/", "PL/")):
            raise ValueError("ELI ID must start with Polish country code")

        return v.strip()

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "Amendment":
        """Create Amendment from ELI API response."""

        # Parse dates
        published_date = None
        if data.get("published_date"):
            try:
                published_date = datetime.fromisoformat(
                    data["published_date"].replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                pass

        effective_date = None
        if data.get("effective_date"):
            try:
                effective_date = datetime.fromisoformat(
                    data["effective_date"].replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                pass

        # Parse amendment type
        amendment_type_str = data.get("amendment_type", "").lower()
        try:
            amendment_type = AmendmentType(amendment_type_str)
        except ValueError:
            # Default to nowelizacja if type not recognized
            amendment_type = AmendmentType.NOWELIZACJA

        return cls(
            eli_id=data.get("eli_id", ""),
            target_eli_id=data.get("target_eli_id", ""),
            amendment_type=amendment_type,
            title=data.get("title", ""),
            description=data.get("description"),
            published_date=published_date,
            effective_date=effective_date,
            affected_articles=data.get("affected_articles", []),
            affected_paragraphs=data.get("affected_paragraphs", []),
            change_summary=data.get("change_summary"),
            legal_basis=data.get("legal_basis"),
        )


class DocumentSearchResult(BaseModel):
    """Model for document search results."""

    documents: List[LegalDocument] = Field(..., description="List of found documents")
    total: int = Field(..., description="Total number of matching documents")
    offset: int = Field(0, description="Search offset")
    limit: int = Field(100, description="Search limit")

    def has_more(self) -> bool:
        """Check if there are more results available."""
        return self.offset + len(self.documents) < self.total

    def next_offset(self) -> int:
        """Get offset for next page of results."""
        return self.offset + self.limit


class DocumentStatistics(BaseModel):
    """Statistics about legal documents."""

    total_documents: int = Field(0, description="Total number of documents")
    documents_by_type: Dict[DocumentType, int] = Field(default_factory=dict)
    documents_by_status: Dict[DocumentStatus, int] = Field(default_factory=dict)
    documents_by_year: Dict[int, int] = Field(default_factory=dict)

    recent_documents_count: int = Field(
        0, description="Documents published in last 30 days"
    )
    active_documents_count: int = Field(0, description="Currently active documents")

    last_updated: datetime = Field(default_factory=datetime.now)


class MultiActAmendment(BaseModel):
    """Model for amendments affecting multiple legal acts."""

    eli_id: str = Field(..., description="ELI ID of the omnibus amendment")
    title: str = Field(..., description="Amendment title")

    affected_acts: List[str] = Field(
        ..., description="List of ELI IDs of affected acts"
    )
    complexity_score: int = Field(..., description="Number of affected acts")

    published_date: Optional[datetime] = Field(None, description="Publication date")
    effective_date: Optional[datetime] = Field(None, description="Effective date")

    cross_references: List[str] = Field(
        default_factory=list, description="Cross-references between acts"
    )
    impact_assessment: Optional[str] = Field(
        None, description="Impact assessment summary"
    )

    @field_validator("complexity_score")
    @classmethod
    def validate_complexity_score(cls, v):
        """Validate complexity score."""
        if v < 2:
            raise ValueError("Multi-act amendment must affect at least 2 acts")
        return v

    def is_omnibus(self) -> bool:
        """Check if this is an omnibus legislation."""
        return self.complexity_score >= 3

    def get_impact_level(self) -> str:
        """Get impact level based on complexity score."""
        if self.complexity_score >= 10:
            return "very_high"
        elif self.complexity_score >= 5:
            return "high"
        elif self.complexity_score >= 3:
            return "medium"
        else:
            return "low"
