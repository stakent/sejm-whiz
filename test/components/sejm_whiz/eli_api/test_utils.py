"""Tests for ELI API utilities."""

from sejm_whiz.eli_api.utils import (
    validate_eli_id,
    sanitize_query,
    normalize_document_type,
    extract_legal_references,
    parse_date_string,
    format_eli_id,
    calculate_document_complexity,
    is_amendment_document,
    extract_affected_articles,
    clean_legal_text,
)


class TestValidateEliId:
    """Test ELI ID validation."""

    def test_valid_eli_ids(self):
        """Test valid ELI ID formats."""
        valid_ids = [
            "pl/test/2023/1",
            "PL/TEST/2023/1",
            "pl/complex/path/with/multiple/segments/2023/1",
            "pl/act/2023/123a",
            "pl/regulation/2023/456/text",
        ]

        for eli_id in valid_ids:
            assert validate_eli_id(eli_id) is True, f"Should be valid: {eli_id}"

    def test_invalid_eli_ids(self):
        """Test invalid ELI ID formats."""
        invalid_ids = [
            "",  # Empty string
            "   ",  # Whitespace only
            None,  # None value
            123,  # Not a string
            "us/test/2023/1",  # Wrong country code
            "test/2023/1",  # No country code
            "pl",  # Too short
            "pl/",  # Too short
            "pl/<script>alert('xss')</script>/2023/1",  # Potential XSS
        ]

        for eli_id in invalid_ids:
            assert validate_eli_id(eli_id) is False, f"Should be invalid: {eli_id}"

    def test_whitespace_handling(self):
        """Test whitespace handling in ELI IDs."""
        assert validate_eli_id("  pl/test/2023/1  ") is True
        assert validate_eli_id("\tpl/test/2023/1\n") is True


class TestSanitizeQuery:
    """Test query sanitization."""

    def test_normal_queries(self):
        """Test normal query sanitization."""
        test_cases = [
            ("simple query", "simple query"),
            ("  whitespace  query  ", "whitespace query"),
            ("query\twith\ttabs", "query with tabs"),
            ("query\nwith\nnewlines", "query with newlines"),
            ("", ""),
            ("   ", ""),
        ]

        for input_query, expected in test_cases:
            result = sanitize_query(input_query)
            assert result == expected, (
                f"Input: {input_query!r}, Expected: {expected!r}, Got: {result!r}"
            )

    def test_dangerous_patterns(self):
        """Test removal of dangerous patterns."""
        dangerous_queries = [
            "'; DROP TABLE users;",
            "'; DELETE FROM documents;",
            "'; INSERT INTO malicious VALUES ('bad');",
            "'; UPDATE users SET admin=1;",
            "UNION SELECT password FROM users",
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "vbscript:msgbox('xss')",
            "onload=alert('xss')",
            "onerror=alert('xss')",
        ]

        for query in dangerous_queries:
            result = sanitize_query(query)
            # Should not contain the original dangerous pattern
            assert query.lower() not in result.lower(), (
                f"Dangerous pattern not removed: {query}"
            )

    def test_control_characters(self):
        """Test removal of control characters."""
        query_with_controls = "test\x00\x01\x1f\x7f\x9fquery"
        result = sanitize_query(query_with_controls)
        assert result == "testquery"

    def test_length_limit(self):
        """Test query length limitation."""
        long_query = "a" * 1000
        result = sanitize_query(long_query)
        assert len(result) <= 500

    def test_invalid_input(self):
        """Test handling of invalid input."""
        assert sanitize_query(None) == ""
        assert sanitize_query(123) == ""
        assert sanitize_query([]) == ""


class TestNormalizeDocumentType:
    """Test document type normalization."""

    def test_valid_normalizations(self):
        """Test valid document type normalizations."""
        test_cases = [
            ("ustawa", "ustawa"),
            ("USTAWA", "ustawa"),
            ("act", "ustawa"),
            ("law", "ustawa"),
            ("ustawy", "ustawa"),
            ("ustaw", "ustawa"),
            ("rozporządzenie", "rozporządzenie"),
            ("rozporzadzenie", "rozporządzenie"),
            ("regulation", "rozporządzenie"),
            ("rozporządzeń", "rozporządzenie"),
            ("kodeks", "kodeks"),
            ("code", "kodeks"),
            ("kodeksu", "kodeks"),
            ("kodeksy", "kodeks"),
            ("konstytucja", "konstytucja"),
            ("constitution", "konstytucja"),
            ("konstytucji", "konstytucja"),
            ("dekret", "dekret"),
            ("decree", "dekret"),
            ("dekretu", "dekret"),
            ("dekrety", "dekret"),
            ("uchwała", "uchwała"),
            ("uchwala", "uchwała"),
            ("resolution", "uchwała"),
            ("uchwały", "uchwała"),
        ]

        for input_type, expected in test_cases:
            result = normalize_document_type(input_type)
            assert result == expected, (
                f"Input: {input_type}, Expected: {expected}, Got: {result}"
            )

    def test_invalid_types(self):
        """Test handling of invalid document types."""
        invalid_types = [None, "", "   ", 123, "unknown_type", "invalid"]

        for doc_type in invalid_types:
            result = normalize_document_type(doc_type)
            assert result is None, f"Should return None for: {doc_type}"

    def test_whitespace_handling(self):
        """Test whitespace handling in document types."""
        assert normalize_document_type("  ustawa  ") == "ustawa"
        assert normalize_document_type("\tustawa\n") == "ustawa"


class TestExtractLegalReferences:
    """Test legal reference extraction."""

    def test_article_references(self):
        """Test extraction of article references."""
        text = "Zgodnie z art. 123 oraz art. 45a Konstytucji..."
        references = extract_legal_references(text)

        assert "art. 123" in references
        assert "art. 45a" in references

    def test_paragraph_references(self):
        """Test extraction of paragraph references."""
        text = "W § 1 i § 23 rozporządzenia..."
        references = extract_legal_references(text)

        assert "§ 1" in references
        assert "§ 23" in references

    def test_point_references(self):
        """Test extraction of point references."""
        text = "Według pkt 1) oraz punkt 3) załącznika..."
        references = extract_legal_references(text)

        assert any("pkt 1)" in ref for ref in references)
        assert any("punkt 3)" in ref for ref in references)

    def test_act_references(self):
        """Test extraction of act references."""
        text = "Na podstawie ustawy z dnia 12 stycznia 2023 r. o testach..."
        references = extract_legal_references(text)

        assert any("ustawy z dnia" in ref for ref in references)

    def test_code_references(self):
        """Test extraction of code references."""
        text = "Kodeks cywilny oraz Kodeks karny stanowią..."
        references = extract_legal_references(text)

        assert "Kodeks cywilny" in references
        assert "Kodeks karny" in references

    def test_constitution_references(self):
        """Test extraction of constitution references."""
        text = "Konstytucja Rzeczypospolitej Polskiej oraz konstytucja..."
        references = extract_legal_references(text)

        assert any("konstytucj" in ref.lower() for ref in references)

    def test_duplicate_removal(self):
        """Test removal of duplicate references."""
        text = "art. 123, art. 123, art. 123"
        references = extract_legal_references(text)

        # Should contain only one instance
        art_123_count = sum(1 for ref in references if "art. 123" in ref)
        assert art_123_count == 1

    def test_empty_text(self):
        """Test handling of empty text."""
        assert extract_legal_references("") == []
        assert extract_legal_references(None) == []
        assert extract_legal_references("   ") == []

    def test_case_insensitive(self):
        """Test case-insensitive extraction."""
        text = "ART. 123 oraz Art. 456"
        references = extract_legal_references(text)

        assert len(references) >= 2
        assert any("123" in ref for ref in references)
        assert any("456" in ref for ref in references)


class TestParseDateString:
    """Test Polish date string parsing."""

    def test_valid_dates(self):
        """Test parsing valid Polish dates."""
        test_cases = [
            ("12 stycznia 2023", "2023-01-12"),
            ("1 lutego 2023", "2023-02-01"),
            ("15 marca 2023", "2023-03-15"),
            ("30 kwietnia 2023", "2023-04-30"),
            ("25 maja 2023", "2023-05-25"),
            ("10 czerwca 2023", "2023-06-10"),
            ("4 lipca 2023", "2023-07-04"),
            ("20 sierpnia 2023", "2023-08-20"),
            ("3 września 2023", "2023-09-03"),
            ("31 października 2023", "2023-10-31"),
            ("11 listopada 2023", "2023-11-11"),
            ("24 grudnia 2023", "2023-12-24"),
        ]

        for input_date, expected in test_cases:
            result = parse_date_string(input_date)
            assert result == expected, (
                f"Input: {input_date}, Expected: {expected}, Got: {result}"
            )

    def test_invalid_dates(self):
        """Test handling of invalid dates."""
        invalid_dates = [
            None,
            "",
            "   ",
            "32 stycznia 2023",  # Invalid day
            "1 unknown_month 2023",  # Invalid month
            "not a date",
            "2023-01-01",  # Not Polish format
            123,
        ]

        for date_str in invalid_dates:
            result = parse_date_string(date_str)
            assert result is None, f"Should return None for: {date_str}"

    def test_case_sensitivity(self):
        """Test case sensitivity in month names."""
        test_cases = ["12 STYCZNIA 2023", "12 Stycznia 2023", "12 stycznia 2023"]

        for date_str in test_cases:
            result = parse_date_string(date_str)
            assert result == "2023-01-12"


class TestFormatEliId:
    """Test ELI ID formatting."""

    def test_normal_formatting(self):
        """Test normal ELI ID formatting."""
        test_cases = [
            ("pl/test/2023/1", "pl/test/2023/1"),
            ("PL/test/2023/1", "pl/test/2023/1"),  # Normalize country code
            ("pl//test///2023//1", "pl/test/2023/1"),  # Remove duplicate slashes
            ("pl/test/2023/1/", "pl/test/2023/1"),  # Remove trailing slash
            ("  pl/test/2023/1  ", "pl/test/2023/1"),  # Trim whitespace
        ]

        for input_id, expected in test_cases:
            result = format_eli_id(input_id)
            assert result == expected, (
                f"Input: {input_id}, Expected: {expected}, Got: {result}"
            )

    def test_invalid_input(self):
        """Test handling of invalid input."""
        assert format_eli_id(None) == ""
        assert format_eli_id("") == ""
        assert format_eli_id("   ") == ""
        assert format_eli_id(123) == ""


class TestCalculateDocumentComplexity:
    """Test document complexity calculation."""

    def test_simple_document(self):
        """Test complexity of simple document."""
        simple_text = "Art. 1. Ustawa wchodzi w życie."
        complexity = calculate_document_complexity(simple_text)

        assert 0 <= complexity <= 100
        assert complexity < 50  # Should be relatively low

    def test_complex_document(self):
        """Test complexity of complex document."""
        complex_text = (
            """
        Art. 1. Nowelizacja ustawy.
        Art. 2. Rozporządzenie wykonawcze.
        Art. 3. Delegacja ustawowa.
        Art. 4. Przepisy przejściowe.
        Art. 5. Vacatio legis.

        § 1. Pierwsze przepisy.
        § 2. Drugie przepisy.
        § 3. Trzecie przepisy.

        Zgodnie z art. 123 Konstytucji oraz art. 456 kodeksu cywilnego,
        w związku z § 7 rozporządzenia...
        """
            * 10
        )  # Make it longer

        complexity = calculate_document_complexity(complex_text)

        assert 0 <= complexity <= 100
        assert complexity > 50  # Should be relatively high

    def test_empty_document(self):
        """Test complexity of empty document."""
        assert calculate_document_complexity("") == 0
        assert calculate_document_complexity(None) == 0
        assert calculate_document_complexity("   ") == 0

    def test_complexity_bounds(self):
        """Test that complexity is always within bounds."""
        # Very long document
        very_long_text = "Art. 1. " * 10000 + "nowelizacja " * 1000
        complexity = calculate_document_complexity(very_long_text)

        assert 0 <= complexity <= 100


class TestIsAmendmentDocument:
    """Test amendment document detection."""

    def test_amendment_indicators(self):
        """Test detection of amendment indicators."""
        amendment_texts = [
            "Ustawa o zmianie ustawy o testach",
            "Nowelizacja ustawy testowej",
            "Przepisy wprowadzające do kodeksu",
            "W ustawie z dnia... wprowadza się następujące zmiany",
            "Zmienia się ustawę o...",
        ]

        for text in amendment_texts:
            assert is_amendment_document(text) is True, (
                f"Should detect amendment: {text}"
            )

    def test_non_amendment_documents(self):
        """Test non-amendment documents."""
        non_amendment_texts = [
            "Ustawa o ochronie środowiska",
            "Kodeks postępowania cywilnego",
            "Rozporządzenie w sprawie bezpieczeństwa",
            "Konstytucja Rzeczypospolitej Polskiej",
        ]

        for text in non_amendment_texts:
            assert is_amendment_document(text) is False, (
                f"Should not detect amendment: {text}"
            )

    def test_title_and_content(self):
        """Test using both title and content."""
        title = "Ustawa o zmianie niektórych ustaw"
        content = "Art. 1. W ustawie wprowadza się zmiany..."

        assert is_amendment_document(content, title) is True

    def test_empty_input(self):
        """Test handling of empty input."""
        assert is_amendment_document("") is False
        assert is_amendment_document("", "") is False
        assert is_amendment_document(None, None) is False


class TestExtractAffectedArticles:
    """Test extraction of affected articles."""

    def test_amendment_patterns(self):
        """Test extraction from amendment language."""
        text = """
        W art. 123 zmienia się...
        W art. 45a dodaje się...
        Art. 67 otrzymuje brzmienie...
        Uchyla się art. 89.
        Art. 12 i art. 34 zastępuje się...
        """

        articles = extract_affected_articles(text)

        expected_articles = ["12", "123", "34", "45a", "67", "89"]
        for article in expected_articles:
            assert article in articles, f"Should extract article: {article}"

    def test_sorting(self):
        """Test that articles are sorted correctly."""
        text = "art. 123, art. 1, art. 45a, art. 2"
        articles = extract_affected_articles(text)

        # Should be sorted by length first, then alphabetically
        # ["1", "2", "45a", "123"]
        assert articles.index("1") < articles.index("123")
        assert articles.index("2") < articles.index("45a")

    def test_duplicates_removed(self):
        """Test that duplicate articles are removed."""
        text = "art. 123 zmienia się... art. 123 otrzymuje brzmienie..."
        articles = extract_affected_articles(text)

        # Should contain only one instance of "123"
        assert articles.count("123") == 1

    def test_empty_text(self):
        """Test handling of empty text."""
        assert extract_affected_articles("") == []
        assert extract_affected_articles(None) == []
        assert extract_affected_articles("No articles here") == []


class TestCleanLegalText:
    """Test legal text cleaning."""

    def test_html_removal(self):
        """Test HTML tag removal."""
        html_text = "<p>Test <strong>legal</strong> text</p>"
        cleaned = clean_legal_text(html_text)
        assert cleaned == "Test legal text"

    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        messy_text = "Test   text\n\n\n\nwith    spacing"
        cleaned = clean_legal_text(messy_text)
        assert "   " not in cleaned  # No triple spaces
        assert "\n\n\n" not in cleaned  # No triple newlines

    def test_punctuation_normalization(self):
        """Test punctuation normalization."""
        text_with_punctuation = "Test... text---- more..."
        cleaned = clean_legal_text(text_with_punctuation)
        assert "..." in cleaned  # Should normalize to max 3 dots
        assert "----" not in cleaned  # Should normalize to max 2 dashes

    def test_unicode_normalization(self):
        """Test Unicode character normalization."""
        unicode_text = "Test\u00a0text\u2013with\u2014special\u2019chars"
        cleaned = clean_legal_text(unicode_text)

        assert "\u00a0" not in cleaned  # Non-breaking space should be normalized
        assert "\u2013" not in cleaned  # En dash should be normalized
        assert "\u2014" not in cleaned  # Em dash should be normalized
        assert "\u2019" not in cleaned  # Right single quote should be normalized

    def test_empty_input(self):
        """Test handling of empty input."""
        assert clean_legal_text("") == ""
        assert clean_legal_text(None) == ""
        assert clean_legal_text("   ") == ""

    def test_polish_diacritics_preserved(self):
        """Test that Polish diacritics are preserved."""
        polish_text = "ąćęłńóśźż ĄĆĘŁŃÓŚŹŻ"
        cleaned = clean_legal_text(polish_text)
        assert "ą" in cleaned
        assert "ć" in cleaned
        assert "Ą" in cleaned
        assert "Ć" in cleaned
