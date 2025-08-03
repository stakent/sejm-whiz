import pytest
from datetime import datetime, date
from pydantic import ValidationError

from sejm_whiz.sejm_api.models import (
    Session,
    ProceedingSitting,
    Voting,
    Deputy,
    Committee,
    Interpellation,
    Processing,
    Proceeding,
    VotingResult,
    DeputyStatus,
    ProcessingStage,
    VotingStatistics,
    ProcessingStatistics,
    VotingOption,
    DeputyVote,
)


class TestSejmApiModels:
    """Test cases for Pydantic models used in Sejm API."""

    def test_voting_result_enum(self):
        """Test VotingResult enum values."""
        assert VotingResult.FOR == "za"
        assert VotingResult.AGAINST == "przeciw"
        assert VotingResult.ABSTAIN == "wstrzymal"
        assert VotingResult.ABSENT == "nieobecny"
        assert VotingResult.NOT_VOTING == "nie_glosowal"

    def test_deputy_status_enum(self):
        """Test DeputyStatus enum values."""
        assert DeputyStatus.ACTIVE == "aktywny"
        assert DeputyStatus.INACTIVE == "nieaktywny"
        assert DeputyStatus.SUSPENDED == "zawieszony"

    def test_processing_stage_enum(self):
        """Test ProcessingStage enum values."""
        assert ProcessingStage.FIRST_READING == "pierwsze_czytanie"
        assert ProcessingStage.COMMITTEE == "komisja"
        assert ProcessingStage.SECOND_READING == "drugie_czytanie"
        assert ProcessingStage.THIRD_READING == "trzecie_czytanie"
        assert ProcessingStage.SENATE == "senat"
        assert ProcessingStage.PRESIDENT == "prezydent"
        assert ProcessingStage.ENACTED == "uchwalone"
        assert ProcessingStage.REJECTED == "odrzucone"

    def test_session_model(self):
        """Test Session model creation and validation."""
        session_data = {
            "id": 1,
            "term": 10,
            "sessionNumber": 15,
            "startDate": "2023-01-15",
            "endDate": "2023-01-20",
            "title": "XV sesja Sejmu X kadencji",
        }

        session = Session.model_validate(session_data)

        assert session.id == 1
        assert session.term == 10
        assert session.session_number == 15
        assert session.start_date == date(2023, 1, 15)
        assert session.end_date == date(2023, 1, 20)
        assert session.title == "XV sesja Sejmu X kadencji"

    def test_session_model_with_alias(self):
        """Test Session model using field aliases."""
        session_data = {
            "id": 2,
            "term": 10,
            "session_number": 16,  # Using Python field name instead of alias
            "start_date": "2023-02-15",
            "title": "Test Session",
        }

        session = Session.model_validate(session_data)

        assert session.id == 2
        assert session.session_number == 16
        assert session.start_date == date(2023, 2, 15)

    def test_proceeding_sitting_model(self):
        """Test ProceedingSitting model creation and validation."""
        sitting_data = {
            "id": 1,
            "term": 10,
            "session": 15,
            "sittingNumber": 1,
            "date": "2023-01-15",
            "startTime": "10:00",
            "endTime": "18:30",
            "title": "1. dzień XV sesji",
        }

        proceeding_sitting = ProceedingSitting.model_validate(sitting_data)

        assert proceeding_sitting.id == 1
        assert proceeding_sitting.term == 10
        assert proceeding_sitting.session == 15
        assert proceeding_sitting.sitting_number == 1
        assert proceeding_sitting.date == date(2023, 1, 15)
        assert proceeding_sitting.start_time == "10:00"
        assert proceeding_sitting.end_time == "18:30"
        assert proceeding_sitting.title == "1. dzień XV sesji"

    def test_deputy_model(self):
        """Test Deputy model creation and validation."""
        deputy_data = {
            "id": 123,
            "firstName": "Jan",
            "lastName": "Kowalski",
            "email": "jan.kowalski@sejm.pl",
            "birthDate": "1975-03-15",
            "birthLocation": "Warszawa",
            "profession": "prawnik",
            "numberOfVotes": 45678,
            "voivodeship": "mazowieckie",
            "districtNumber": 4,
            "districtName": "Warszawa",
            "club": "Platforma Obywatelska",
            "status": "aktywny",
        }

        deputy = Deputy.model_validate(deputy_data)

        assert deputy.id == 123
        assert deputy.first_name == "Jan"
        assert deputy.last_name == "Kowalski"
        assert deputy.full_name == "Jan Kowalski"
        assert deputy.email == "jan.kowalski@sejm.pl"
        assert deputy.birth_date == date(1975, 3, 15)
        assert deputy.birth_location == "Warszawa"
        assert deputy.profession == "prawnik"
        assert deputy.number_of_votes == 45678
        assert deputy.voivodeship == "mazowieckie"
        assert deputy.district_number == 4
        assert deputy.district_name == "Warszawa"
        assert deputy.club == "Platforma Obywatelska"
        assert deputy.status == DeputyStatus.ACTIVE

    def test_deputy_full_name_property(self):
        """Test Deputy full_name property."""
        deputy_data = {"id": 1, "firstName": "Anna", "lastName": "Nowak"}

        deputy = Deputy.model_validate(deputy_data)
        assert deputy.full_name == "Anna Nowak"

    def test_voting_model(self):
        """Test Voting model creation and validation."""
        voting_data = {
            "id": 1,
            "term": 10,
            "session": 15,
            "sitting": 1,
            "votingNumber": 5,
            "date": "2023-01-15",
            "time": "14:30",
            "title": "Głosowanie nad ustawą budżetową",
            "description": "Trzecie czytanie ustawy budżetowej na rok 2023",
            "topic": "Ustawa budżetowa na rok 2023",
            "kind": "nominalne",
            "yesVotes": 245,
            "noVotes": 201,
            "abstainVotes": 14,
            "notParticipating": 0,
            "totalVotes": 460,
            "success": True,
        }

        voting = Voting.model_validate(voting_data)

        assert voting.id == 1
        assert voting.term == 10
        assert voting.session == 15
        assert voting.proceeding_sitting == 1
        assert voting.voting_number == 5
        assert voting.date == date(2023, 1, 15)
        assert voting.time == "14:30"
        assert voting.title == "Głosowanie nad ustawą budżetową"
        assert voting.yes_votes == 245
        assert voting.no_votes == 201
        assert voting.abstain_votes == 14
        assert voting.not_participating == 0
        assert voting.total_votes == 460
        assert voting.success is True

    def test_voting_participation_rate(self):
        """Test Voting participation_rate property."""
        voting_data = {
            "id": 1,
            "term": 10,
            "session": 1,
            "sitting": 1,
            "votingNumber": 1,
            "date": "2023-01-15",
            "title": "Test Voting",
            "yesVotes": 200,
            "noVotes": 150,
            "abstainVotes": 50,
            "totalVotes": 460,
        }

        voting = Voting.model_validate(voting_data)
        expected_rate = (200 + 150 + 50) / 460
        assert abs(voting.participation_rate - expected_rate) < 0.001

    def test_voting_participation_rate_zero_total(self):
        """Test Voting participation_rate with zero total votes."""
        voting_data = {
            "id": 1,
            "term": 10,
            "session": 1,
            "sitting": 1,
            "votingNumber": 1,
            "date": "2023-01-15",
            "title": "Test Voting",
            "totalVotes": 0,
        }

        voting = Voting.model_validate(voting_data)
        assert voting.participation_rate == 0.0

    def test_committee_model(self):
        """Test Committee model creation and validation."""
        committee_data = {
            "id": 1,
            "code": "SIL",
            "name": "Komisja Spraw Wewnętrznych i Administracji",
            "nameGenitive": "Komisji Spraw Wewnętrznych i Administracji",
            "appointmentDate": "2019-11-12",
            "compositionDate": "2019-11-20",
            "phone": "+48 22 694 25 67",
            "scope": "Sprawy wewnętrzne, administracja publiczna, samorząd terytorialny",
            "type": "stała",
        }

        committee = Committee.model_validate(committee_data)

        assert committee.id == 1
        assert committee.code == "SIL"
        assert committee.name == "Komisja Spraw Wewnętrznych i Administracji"
        assert committee.name_genitive == "Komisji Spraw Wewnętrznych i Administracji"
        assert committee.appointment_date == date(2019, 11, 12)
        assert committee.composition_date == date(2019, 11, 20)
        assert committee.phone == "+48 22 694 25 67"
        assert (
            committee.scope
            == "Sprawy wewnętrzne, administracja publiczna, samorząd terytorialny"
        )
        assert committee.type == "stała"

    def test_interpellation_model(self):
        """Test Interpellation model creation and validation."""
        interpellation_data = {
            "id": 1,
            "term": 10,
            "number": "INT-123/2023",
            "title": "Interpelacja w sprawie ochrony środowiska",
            "receiptDate": "2023-01-15",
            "lastModified": "2023-01-20T10:30:00",
            "fromDeputy": "Jan Kowalski",
            "toMember": "Minister Klimatu i Środowiska",
            "body": "Treść interpelacji...",
            "reply": "Odpowiedź ministra...",
            "replyDate": "2023-02-15",
        }

        interpellation = Interpellation.model_validate(interpellation_data)

        assert interpellation.id == 1
        assert interpellation.term == 10
        assert interpellation.number == "INT-123/2023"
        assert interpellation.title == "Interpelacja w sprawie ochrony środowiska"
        assert interpellation.receipt_date == date(2023, 1, 15)
        assert interpellation.last_modified == datetime(2023, 1, 20, 10, 30, 0)
        assert interpellation.from_deputy == "Jan Kowalski"
        assert interpellation.to_member == "Minister Klimatu i Środowiska"
        assert interpellation.body == "Treść interpelacji..."
        assert interpellation.reply == "Odpowiedź ministra..."
        assert interpellation.reply_date == date(2023, 2, 15)

    def test_processing_model(self):
        """Test Processing model creation and validation."""
        processing_data = {
            "id": 1,
            "term": 10,
            "number": 123,
            "printNumber": "123",
            "title": "Ustawa o zmianie ustawy o podatku dochodowym",
            "description": "Projekt ustawy zmieniającej przepisy podatkowe",
            "documentType": "rządowy projekt ustawy",
            "documentDate": "2023-01-10",
            "receiptDate": "2023-01-15",
            "origin": "Rada Ministrów",
            "rclNumber": "RC-123",
            "urgent": False,
            "currentStage": "pierwsze_czytanie",
        }

        processing = Processing.model_validate(processing_data)

        assert processing.id == 1
        assert processing.term == 10
        assert processing.number == 123
        assert processing.print_number == "123"
        assert processing.title == "Ustawa o zmianie ustawy o podatku dochodowym"
        assert (
            processing.description == "Projekt ustawy zmieniającej przepisy podatkowe"
        )
        assert processing.document_type == "rządowy projekt ustawy"
        assert processing.document_date == date(2023, 1, 10)
        assert processing.receipt_date == date(2023, 1, 15)
        assert processing.origin == "Rada Ministrów"
        assert processing.rcl_number == "RC-123"
        assert processing.urgent is False
        assert processing.current_stage == ProcessingStage.FIRST_READING

    def test_processing_is_active_property(self):
        """Test Processing is_active property."""
        # Active processing
        processing_data = {
            "id": 1,
            "term": 10,
            "number": 123,
            "printNumber": "123",
            "title": "Test",
            "documentType": "projekt ustawy",
            "currentStage": "pierwsze_czytanie",
        }

        processing = Processing.model_validate(processing_data)
        assert processing.is_active is True

        # Enacted processing (not active)
        processing_data["currentStage"] = "uchwalone"
        processing = Processing.model_validate(processing_data)
        assert processing.is_active is False

        # Rejected processing (not active)
        processing_data["currentStage"] = "odrzucone"
        processing = Processing.model_validate(processing_data)
        assert processing.is_active is False

    def test_processing_days_in_processing_property(self):
        """Test Processing days_in_processing property."""
        from datetime import datetime, timedelta

        # Processing with receipt date
        receipt_date = (datetime.now() - timedelta(days=30)).date()
        processing_data = {
            "id": 1,
            "term": 10,
            "number": 123,
            "printNumber": "123",
            "title": "Test",
            "documentType": "projekt ustawy",
            "currentStage": "pierwsze_czytanie",
            "receiptDate": receipt_date.isoformat(),
        }

        processing = Processing.model_validate(processing_data)
        assert processing.days_in_processing == 30

        # Processing without receipt date
        processing_data["receiptDate"] = None
        processing = Processing.model_validate(processing_data)
        assert processing.days_in_processing is None

    def test_voting_option_model(self):
        """Test VotingOption model creation and validation."""
        option_data = {
            "optionId": 1,
            "optionIndex": 1,
            "description": "Za przyjęciem",
            "votesCount": 245,
        }

        option = VotingOption.model_validate(option_data)

        assert option.option_id == 1
        assert option.option_index == 1
        assert option.description == "Za przyjęciem"
        assert option.votes_count == 245

    def test_deputy_vote_model(self):
        """Test DeputyVote model creation and validation."""
        vote_data = {
            "deputyId": 123,
            "deputyName": "Jan Kowalski",
            "club": "Platforma Obywatelska",
            "vote": "za",
        }

        vote = DeputyVote.model_validate(vote_data)

        assert vote.deputy_id == 123
        assert vote.deputy_name == "Jan Kowalski"
        assert vote.club == "Platforma Obywatelska"
        assert vote.vote == VotingResult.FOR

    def test_voting_statistics_model(self):
        """Test VotingStatistics model creation and validation."""
        stats_data = {
            "totalVotings": 150,
            "successfulVotings": 120,
            "averageParticipation": 0.87,
            "mostActiveDeputy": "Jan Kowalski",
            "mostActiveClub": "Platforma Obywatelska",
        }

        stats = VotingStatistics.model_validate(stats_data)

        assert stats.total_votings == 150
        assert stats.successful_votings == 120
        assert stats.average_participation == 0.87
        assert stats.most_active_deputy == "Jan Kowalski"
        assert stats.most_active_club == "Platforma Obywatelska"

    def test_processing_statistics_model(self):
        """Test ProcessingStatistics model creation and validation."""
        stats_data = {
            "totalProcesses": 200,
            "activeProcesses": 45,
            "enactedLaws": 120,
            "rejectedProposals": 35,
            "averageProcessingDays": 89.5,
        }

        stats = ProcessingStatistics.model_validate(stats_data)

        assert stats.total_processes == 200
        assert stats.active_processes == 45
        assert stats.enacted_laws == 120
        assert stats.rejected_proposals == 35
        assert stats.average_processing_days == 89.5

    def test_model_validation_errors(self):
        """Test model validation with invalid data."""
        # Missing required field
        with pytest.raises(ValidationError):
            Session.model_validate({"id": 1})  # Missing term

        # Invalid enum value
        with pytest.raises(ValidationError):
            Deputy.model_validate(
                {
                    "id": 1,
                    "firstName": "Jan",
                    "lastName": "Kowalski",
                    "status": "invalid_status",
                }
            )

        # Invalid date format
        with pytest.raises(ValidationError):
            Session.model_validate(
                {"id": 1, "term": 10, "sessionNumber": 1, "startDate": "invalid-date"}
            )

    def test_model_default_values(self):
        """Test model default values."""
        # Voting with minimal data should use defaults
        voting_data = {
            "id": 1,
            "term": 10,
            "session": 1,
            "sitting": 1,
            "votingNumber": 1,
            "date": "2023-01-15",
            "title": "Test",
        }

        voting = Voting.model_validate(voting_data)

        assert voting.yes_votes == 0
        assert voting.no_votes == 0
        assert voting.abstain_votes == 0
        assert voting.not_participating == 0
        assert voting.total_votes == 0
        assert voting.success is None
