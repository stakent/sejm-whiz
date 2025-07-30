from datetime import datetime, date
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class VotingResult(str, Enum):
    """Voting result types."""

    FOR = "za"
    AGAINST = "przeciw"
    ABSTAIN = "wstrzymal"
    ABSENT = "nieobecny"
    NOT_VOTING = "nie_glosowal"


class DeputyStatus(str, Enum):
    """Deputy status types."""

    ACTIVE = "aktywny"
    INACTIVE = "nieaktywny"
    SUSPENDED = "zawieszony"


class ProcessingStage(str, Enum):
    """Legislative processing stages."""

    FIRST_READING = "pierwsze_czytanie"
    COMMITTEE = "komisja"
    SECOND_READING = "drugie_czytanie"
    THIRD_READING = "trzecie_czytanie"
    SENATE = "senat"
    PRESIDENT = "prezydent"
    ENACTED = "uchwalone"
    REJECTED = "odrzucone"


class SejmApiResponse(BaseModel):
    """Base response model for Sejm API."""

    success: bool = True
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class Session(BaseModel):
    """Parliamentary session model."""

    id: int
    term: int = Field(..., description="Parliamentary term number")
    session_number: int = Field(..., alias="sessionNumber")
    start_date: Optional[date] = Field(None, alias="startDate")
    end_date: Optional[date] = Field(None, alias="endDate")
    title: Optional[str] = None
    description: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class Sitting(BaseModel):
    """Parliamentary sitting model."""

    id: int
    term: int
    session: int
    sitting_number: int = Field(..., alias="sittingNumber")
    date: date
    start_time: Optional[str] = Field(None, alias="startTime")
    end_time: Optional[str] = Field(None, alias="endTime")
    title: Optional[str] = None
    agenda_items: Optional[List[Dict[str, Any]]] = Field(None, alias="agendaItems")

    model_config = ConfigDict(populate_by_name=True)


class Deputy(BaseModel):
    """Deputy (member of parliament) model."""

    id: int
    first_name: str = Field(..., alias="firstName")
    last_name: str = Field(..., alias="lastName")
    email: Optional[str] = None
    birth_date: Optional[date] = Field(None, alias="birthDate")
    birth_location: Optional[str] = Field(None, alias="birthLocation")
    profession: Optional[str] = None
    education: Optional[str] = None
    number_of_votes: Optional[int] = Field(None, alias="numberOfVotes")
    voivodeship: Optional[str] = None
    district_number: Optional[int] = Field(None, alias="districtNumber")
    district_name: Optional[str] = Field(None, alias="districtName")
    club: Optional[str] = None
    status: Optional[DeputyStatus] = None

    model_config = ConfigDict(populate_by_name=True)

    @property
    def full_name(self) -> str:
        """Get deputy's full name."""
        return f"{self.first_name} {self.last_name}"


class Committee(BaseModel):
    """Parliamentary committee model."""

    id: int
    code: str
    name: str
    name_genitive: Optional[str] = Field(None, alias="nameGenitive")
    appointment_date: Optional[date] = Field(None, alias="appointmentDate")
    composition_date: Optional[date] = Field(None, alias="compositionDate")
    phone: Optional[str] = None
    scope: Optional[str] = None
    type: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class VotingOption(BaseModel):
    """Individual voting option in a voting."""

    option_id: int = Field(..., alias="optionId")
    option_index: int = Field(..., alias="optionIndex")
    description: str
    votes_count: int = Field(..., alias="votesCount")

    model_config = ConfigDict(populate_by_name=True)


class DeputyVote(BaseModel):
    """Individual deputy's vote."""

    deputy_id: int = Field(..., alias="deputyId")
    deputy_name: str = Field(..., alias="deputyName")
    club: Optional[str] = None
    vote: VotingResult

    model_config = ConfigDict(populate_by_name=True)


class Voting(BaseModel):
    """Parliamentary voting model."""

    id: int
    term: int
    session: int
    sitting: int
    voting_number: int = Field(..., alias="votingNumber")
    date: date
    time: Optional[str] = None
    title: str
    description: Optional[str] = None
    topic: Optional[str] = None
    kind: Optional[str] = None
    yes_votes: int = Field(0, alias="yesVotes")
    no_votes: int = Field(0, alias="noVotes")
    abstain_votes: int = Field(0, alias="abstainVotes")
    not_participating: int = Field(0, alias="notParticipating")
    total_votes: int = Field(0, alias="totalVotes")
    success: Optional[bool] = None
    options: Optional[List[VotingOption]] = None
    deputy_votes: Optional[List[DeputyVote]] = Field(None, alias="deputyVotes")

    model_config = ConfigDict(populate_by_name=True)

    @property
    def participation_rate(self) -> float:
        """Calculate voting participation rate."""
        if self.total_votes == 0:
            return 0.0
        return (self.yes_votes + self.no_votes + self.abstain_votes) / self.total_votes


class Interpellation(BaseModel):
    """Parliamentary interpellation/question model."""

    id: int
    term: int
    number: str
    title: str
    receipt_date: date = Field(..., alias="receiptDate")
    last_modified: Optional[datetime] = Field(None, alias="lastModified")
    from_deputy: str = Field(..., alias="fromDeputy")
    to_member: str = Field(..., alias="toMember")
    body: Optional[str] = None
    reply: Optional[str] = None
    reply_date: Optional[date] = Field(None, alias="replyDate")

    model_config = ConfigDict(populate_by_name=True)


class ProcessingStepInfo(BaseModel):
    """Information about a single processing step."""

    step_id: int = Field(..., alias="stepId")
    step_name: str = Field(..., alias="stepName")
    stage: Optional[ProcessingStage] = None
    date: Optional[date] = None
    description: Optional[str] = None
    comment: Optional[str] = None
    committee: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class ProcessingInfo(BaseModel):
    """Legislative processing information model."""

    id: int
    term: int
    print_number: str = Field(..., alias="printNumber")
    title: str
    description: Optional[str] = None
    document_type: Optional[str] = Field(None, alias="documentType")
    document_date: Optional[date] = Field(None, alias="documentDate")
    receipt_date: Optional[date] = Field(None, alias="receiptDate")
    origin: Optional[str] = None
    rcl_number: Optional[str] = Field(None, alias="rclNumber")
    urgent: bool = False
    processing_steps: List[ProcessingStepInfo] = Field(
        default_factory=list, alias="processingSteps"
    )
    current_stage: Optional[ProcessingStage] = Field(None, alias="currentStage")

    model_config = ConfigDict(populate_by_name=True)

    @property
    def is_active(self) -> bool:
        """Check if the processing is still active."""
        return self.current_stage not in [
            ProcessingStage.ENACTED,
            ProcessingStage.REJECTED,
        ]

    @property
    def days_in_processing(self) -> Optional[int]:
        """Calculate number of days in processing."""
        if not self.receipt_date:
            return None
        return (datetime.now().date() - self.receipt_date).days


class ProcessingDocument(BaseModel):
    """Document related to legislative processing."""

    id: int
    title: str
    description: Optional[str] = None
    document_type: str = Field(..., alias="documentType")
    url: Optional[str] = None
    date: Optional[date] = None

    model_config = ConfigDict(populate_by_name=True)


class AgendaItem(BaseModel):
    """Parliamentary agenda item."""

    id: int
    point: str
    title: str
    description: Optional[str] = None
    processing_id: Optional[int] = Field(None, alias="processingId")
    print_number: Optional[str] = Field(None, alias="printNumber")

    model_config = ConfigDict(populate_by_name=True)


# API pagination models


class PaginationInfo(BaseModel):
    """Pagination information for API responses."""

    total: int
    limit: int
    offset: int
    has_next: bool = Field(..., alias="hasNext")
    has_previous: bool = Field(..., alias="hasPrevious")

    model_config = ConfigDict(populate_by_name=True)


class PaginatedResponse(BaseModel):
    """Base paginated response model."""

    pagination: PaginationInfo
    data: List[Dict[str, Any]]


# Search and filter models


class DeputyFilter(BaseModel):
    """Filter model for deputy searches."""

    term: Optional[int] = None
    club: Optional[str] = None
    voivodeship: Optional[str] = None
    status: Optional[DeputyStatus] = None
    active_only: bool = Field(True, alias="activeOnly")

    model_config = ConfigDict(populate_by_name=True)


class VotingFilter(BaseModel):
    """Filter model for voting searches."""

    term: Optional[int] = None
    session: Optional[int] = None
    sitting: Optional[int] = None
    date_from: Optional[date] = Field(None, alias="dateFrom")
    date_to: Optional[date] = Field(None, alias="dateTo")
    topic_contains: Optional[str] = Field(None, alias="topicContains")

    model_config = ConfigDict(populate_by_name=True)


class ProcessingFilter(BaseModel):
    """Filter model for processing searches."""

    term: Optional[int] = None
    document_type: Optional[str] = Field(None, alias="documentType")
    stage: Optional[ProcessingStage] = None
    urgent_only: bool = Field(False, alias="urgentOnly")
    active_only: bool = Field(True, alias="activeOnly")
    date_from: Optional[date] = Field(None, alias="dateFrom")
    date_to: Optional[date] = Field(None, alias="dateTo")

    model_config = ConfigDict(populate_by_name=True)


# Statistics models


class VotingStatistics(BaseModel):
    """Voting statistics for analysis."""

    total_votings: int = Field(..., alias="totalVotings")
    successful_votings: int = Field(..., alias="successfulVotings")
    average_participation: float = Field(..., alias="averageParticipation")
    most_active_deputy: Optional[str] = Field(None, alias="mostActiveDeputy")
    most_active_club: Optional[str] = Field(None, alias="mostActiveClub")

    model_config = ConfigDict(populate_by_name=True)


class ProcessingStatistics(BaseModel):
    """Processing statistics for analysis."""

    total_processes: int = Field(..., alias="totalProcesses")
    active_processes: int = Field(..., alias="activeProcesses")
    enacted_laws: int = Field(..., alias="enactedLaws")
    rejected_proposals: int = Field(..., alias="rejectedProposals")
    average_processing_days: float = Field(..., alias="averageProcessingDays")

    model_config = ConfigDict(populate_by_name=True)
