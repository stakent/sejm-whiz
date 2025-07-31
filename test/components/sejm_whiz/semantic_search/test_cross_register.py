"""Tests for cross-register matcher."""

import pytest
from unittest.mock import Mock, patch

from sejm_whiz.semantic_search.cross_register import (
    CrossRegisterMatcher,
    MatchResult,
    get_cross_register_matcher,
)


class TestCrossRegisterMatcher:
    """Test cross-register matcher functionality."""

    @pytest.fixture
    def mock_similarity_calculator(self):
        """Mock similarity calculator."""
        calculator = Mock()
        similarity_result = Mock()
        similarity_result.similarity_score = 0.8
        calculator.calculate_similarity.return_value = similarity_result
        return calculator

    @pytest.fixture
    def matcher(self, mock_similarity_calculator):
        """Create matcher with mocked dependencies."""
        return CrossRegisterMatcher(similarity_calculator=mock_similarity_calculator)

    @pytest.fixture
    def legal_text(self):
        """Sample formal legal text."""
        return """
        Art. 1. W rozumieniu niniejszej ustawy należy stosować przepisy kodeksu cywilnego.
        § 1. Organ właściwy w terminie 30 dni zobowiązany jest do wydania decyzji.
        Zgodnie z art. 15 ust. 2 zabrania się naruszania praw obywatelskich.
        """

    @pytest.fixture
    def parliamentary_text(self):
        """Sample informal parliamentary text."""
        return """
        Według tej ustawy trzeba stosować przepisy kodeksu cywilnego.
        Właściwy urząd w ciągu 30 dni musi wydać decyzję.
        Na podstawie artykułu 15 nie wolno naruszać praw obywatelskich.
        """

    def test_match_registers_direct_mappings(
        self, matcher, legal_text, parliamentary_text
    ):
        """Test direct term mapping between registers."""
        with patch(
            "sejm_whiz.semantic_search.cross_register.process_legal_document"
        ) as mock_process:
            mock_processed = {"clean_text": legal_text}
            mock_process.return_value = mock_processed

            with patch(
                "sejm_whiz.semantic_search.cross_register.normalize_legal_text"
            ) as mock_normalize:
                # Return the appropriate text for each call
                mock_normalize.side_effect = [legal_text, parliamentary_text]

                matches = matcher.match_registers(legal_text, parliamentary_text)

        # Should find direct mappings
        assert len(matches) > 0

        # Check for specific mappings
        mapping_texts = [(m.formal_text, m.informal_text) for m in matches]

        # Should find "w rozumieniu niniejszej ustawy" -> "według tej ustawy"
        assert any(
            "w rozumieniu niniejszej ustawy" in formal
            and "według tej ustawy" in informal
            for formal, informal in mapping_texts
        )

        # Check that we have some direct mappings
        direct_mappings = [m for m in matches if m.match_type == "direct_mapping"]
        assert len(direct_mappings) > 0, "Should find some direct mappings"

        # Check properties of direct mappings
        for match in direct_mappings:
            assert match.confidence > 0.8  # High confidence for direct mappings
            assert len(match.key_terms) >= 2

    def test_match_registers_semantic_similarity(
        self, matcher, legal_text, parliamentary_text, mock_similarity_calculator
    ):
        """Test semantic similarity matching."""
        with patch(
            "sejm_whiz.semantic_search.cross_register.process_legal_document"
        ) as mock_process:
            mock_processed = {"clean_text": legal_text}
            mock_process.return_value = mock_processed

            with patch(
                "sejm_whiz.semantic_search.cross_register.normalize_legal_text"
            ) as mock_normalize:
                # Use side effect that returns the input text unchanged
                mock_normalize.side_effect = lambda x: x

                matches = matcher.match_registers(
                    legal_text, parliamentary_text, similarity_threshold=0.7
                )

        # Should have some semantic matches
        semantic_matches = [m for m in matches if m.match_type == "semantic_similarity"]
        assert len(semantic_matches) > 0

        # Check semantic match properties
        for match in semantic_matches:
            assert match.similarity_score >= 0.7
            assert match.confidence <= 0.8  # Lower confidence than direct mappings
            assert "embedding_similarity" in match.metadata

    def test_match_registers_structural_patterns(
        self, matcher, legal_text, parliamentary_text
    ):
        """Test structural pattern matching."""
        with patch(
            "sejm_whiz.semantic_search.cross_register.process_legal_document"
        ) as mock_process:
            mock_processed = {"clean_text": legal_text}
            mock_process.return_value = mock_processed

            with patch(
                "sejm_whiz.semantic_search.cross_register.normalize_legal_text"
            ) as mock_normalize:
                # Return the appropriate text for each call
                mock_normalize.side_effect = [legal_text, parliamentary_text]

                matches = matcher.match_registers(legal_text, parliamentary_text)

        # Should find structural patterns
        structural_matches = [
            m for m in matches if m.match_type == "structural_pattern"
        ]
        assert len(structural_matches) > 0

        # Check for article reference pattern
        article_matches = [
            m for m in structural_matches if "article_reference" in m.key_terms
        ]
        assert len(article_matches) > 0

        # Check for legal obligation pattern
        obligation_matches = [
            m for m in structural_matches if "legal_obligation" in m.key_terms
        ]
        assert len(obligation_matches) > 0

    def test_confidence_threshold_filtering(
        self, matcher, legal_text, parliamentary_text
    ):
        """Test filtering by confidence threshold."""
        with patch(
            "sejm_whiz.semantic_search.cross_register.process_legal_document"
        ) as mock_process:
            mock_processed = {"clean_text": legal_text}
            mock_process.return_value = mock_processed

            with patch(
                "sejm_whiz.semantic_search.cross_register.normalize_legal_text"
            ) as mock_normalize:
                # Use side effect that returns the input text unchanged
                mock_normalize.side_effect = lambda x: x

                # High confidence threshold
                high_conf_matches = matcher.match_registers(
                    legal_text, parliamentary_text, confidence_threshold=0.8
                )

                # Low confidence threshold
                low_conf_matches = matcher.match_registers(
                    legal_text, parliamentary_text, confidence_threshold=0.5
                )

        # Should have fewer matches with high confidence threshold
        assert len(high_conf_matches) <= len(low_conf_matches)

        # All high confidence matches should meet threshold
        for match in high_conf_matches:
            assert match.confidence >= 0.8

    def test_similarity_threshold_filtering(
        self, matcher, legal_text, parliamentary_text
    ):
        """Test filtering by similarity threshold."""
        with patch(
            "sejm_whiz.semantic_search.cross_register.process_legal_document"
        ) as mock_process:
            mock_processed = {"clean_text": legal_text}
            mock_process.return_value = mock_processed

            with patch(
                "sejm_whiz.semantic_search.cross_register.normalize_legal_text"
            ) as mock_normalize:
                # Use side effect that returns the input text unchanged
                mock_normalize.side_effect = lambda x: x

                # High similarity threshold
                high_sim_matches = matcher.match_registers(
                    legal_text, parliamentary_text, similarity_threshold=0.9
                )

                # Low similarity threshold
                low_sim_matches = matcher.match_registers(
                    legal_text, parliamentary_text, similarity_threshold=0.5
                )

        # Should have fewer matches with high similarity threshold
        assert len(high_sim_matches) <= len(low_sim_matches)

    def test_extract_contexts(self, matcher):
        """Test context extraction around terms."""
        text = "This is a test sentence with niniejsza ustawa in the middle and more text after that."
        term = "niniejsza ustawa"

        contexts = matcher._extract_contexts(text, term, context_size=20)

        assert len(contexts) == 1
        assert term in contexts[0]
        assert len(contexts[0]) <= len(text)  # Should be substring of original

    def test_extract_key_terms(self, matcher):
        """Test key legal term extraction."""
        text = "Art. 15 ust. 2 zawiera przepisy dotyczące § 3 pkt 1. Należy stosować zgodnie z rozdział III."

        key_terms = matcher._extract_key_terms(text)

        # Should extract legal structure terms
        assert any("art." in term.lower() for term in key_terms)
        assert any("ust." in term.lower() for term in key_terms)
        assert any("§" in term for term in key_terms)
        assert any("pkt" in term.lower() for term in key_terms)

        # Should extract legal action terms
        assert any("należy" in term.lower() for term in key_terms)

    def test_calculate_context_similarity(self, matcher, mock_similarity_calculator):
        """Test context similarity calculation."""
        legal_text = "Artykuł pierwszy zawiera podstawowe definicje prawne."
        parl_text = "Pierwszy artykuł ma podstawowe definicje prawne."
        formal_term = "artykuł pierwszy"
        informal_term = "pierwszy artykuł"

        similarity = matcher._calculate_context_similarity(
            legal_text, parl_text, formal_term, informal_term
        )

        assert 0.0 <= similarity <= 1.0
        # Should have called similarity calculator
        mock_similarity_calculator.calculate_similarity.assert_called()

    def test_register_mappings_compilation(self, matcher):
        """Test that regex patterns are compiled correctly."""
        # Check that patterns are compiled
        assert len(matcher.formal_patterns) > 0
        assert len(matcher.informal_patterns) > 0

        # Test pattern matching
        formal_pattern = matcher.formal_patterns["niniejsza ustawa"]
        text = "W niniejsza ustawa określa zasady."
        matches = formal_pattern.findall(text)
        assert len(matches) > 0

    def test_match_result_to_dict(self):
        """Test MatchResult to_dict conversion."""
        match = MatchResult(
            formal_text="niniejsza ustawa",
            informal_text="ta ustawa",
            similarity_score=0.85,
            match_type="direct_mapping",
            confidence=0.9,
            normalized_forms={"formal": "niniejsza ustawa", "informal": "ta ustawa"},
            key_terms=["niniejsza ustawa", "ta ustawa"],
            metadata={"test": "data"},
        )

        match_dict = match.to_dict()

        assert match_dict["formal_text"] == "niniejsza ustawa"
        assert match_dict["informal_text"] == "ta ustawa"
        assert match_dict["similarity_score"] == 0.85
        assert match_dict["match_type"] == "direct_mapping"
        assert match_dict["confidence"] == 0.9
        assert match_dict["metadata"] == {"test": "data"}

    def test_get_cross_register_matcher_singleton(self):
        """Test singleton pattern for cross-register matcher."""
        matcher1 = get_cross_register_matcher()
        matcher2 = get_cross_register_matcher()

        assert matcher1 is matcher2

    def test_empty_text_handling(self, matcher):
        """Test handling of empty or very short texts."""
        with patch(
            "sejm_whiz.semantic_search.cross_register.process_legal_document"
        ) as mock_process:
            mock_processed = Mock()
            mock_processed.clean_text = ""
            mock_process.return_value = mock_processed

            with patch(
                "sejm_whiz.semantic_search.cross_register.normalize_legal_text"
            ) as mock_normalize:
                mock_normalize.side_effect = ["", ""]

                matches = matcher.match_registers("", "")

        # Should handle empty texts gracefully
        assert len(matches) == 0

    def test_similarity_calculation_failure(
        self, matcher, legal_text, parliamentary_text, mock_similarity_calculator
    ):
        """Test handling of similarity calculation failures."""
        # Mock similarity calculation failure
        mock_similarity_calculator.calculate_similarity.side_effect = Exception(
            "Similarity failed"
        )

        with patch(
            "sejm_whiz.semantic_search.cross_register.process_legal_document"
        ) as mock_process:
            mock_processed = {"clean_text": legal_text}
            mock_process.return_value = mock_processed

            with patch(
                "sejm_whiz.semantic_search.cross_register.normalize_legal_text"
            ) as mock_normalize:
                # Return the appropriate text for each call
                mock_normalize.side_effect = [legal_text, parliamentary_text]

                # Should not fail completely
                matches = matcher.match_registers(legal_text, parliamentary_text)

        # Should still have direct mappings and structural patterns
        assert len(matches) > 0

        # Should not have semantic similarity matches
        semantic_matches = [m for m in matches if m.match_type == "semantic_similarity"]
        assert len(semantic_matches) == 0

    def test_register_mapping_coverage(self, matcher):
        """Test coverage of register mappings."""
        mappings = matcher.register_mappings

        # Should have mappings for common legal-parliamentary pairs
        assert "niniejsza ustawa" in mappings
        assert "stosuje się przepisy" in mappings
        assert "podlega karze" in mappings
        assert "organ właściwy" in mappings

        # Should have reverse mappings (parliamentary -> legal)
        assert "mówimy o" in mappings
        assert "trzeba" in mappings
        assert "można" in mappings
