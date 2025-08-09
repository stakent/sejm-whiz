"""Helper functions for ELI API integration."""

import re
from typing import List, Optional
from urllib.parse import quote, unquote


def validate_eli_id(eli_id: str) -> bool:
    """Validate ELI (European Legislation Identifier) format.

    Args:
        eli_id: The ELI identifier to validate

    Returns:
        True if valid ELI ID format, False otherwise
    """
    if not eli_id or not isinstance(eli_id, str):
        return False

    eli_id = eli_id.strip()

    # Basic validation - accept both full ELI format (pl/...) and Polish short format (DU/..., MP/...)
    if not (
        eli_id.startswith(("pl/", "PL/")) or re.match(r"^[A-Z]{1,3}/\d{4}/\d+", eli_id)
    ):
        return False

    # Must have minimum length
    if len(eli_id) < 5:  # e.g., "pl/x"
        return False

    # Check for dangerous characters
    dangerous_chars = ["<", ">", '"', "'", "&", "\x00", "\n", "\r", "\t"]
    if any(char in eli_id for char in dangerous_chars):
        return False

    # Check for script tags or other suspicious patterns
    suspicious_patterns = [
        r"<script",
        r"javascript:",
        r"vbscript:",
        r"onload\s*=",
        r"onerror\s*=",
    ]

    for pattern in suspicious_patterns:
        if re.search(pattern, eli_id, re.IGNORECASE):
            return False

    # Should not contain invalid characters for URLs
    try:
        # Test if it can be URL encoded/decoded properly
        encoded = quote(eli_id, safe="/")
        decoded = unquote(encoded)
        return decoded == eli_id
    except Exception:
        return False


def sanitize_query(query: str) -> str:
    """Sanitize search query for API requests.

    Args:
        query: Raw search query

    Returns:
        Sanitized query string
    """
    if not query or not isinstance(query, str):
        return ""

    # Remove potentially dangerous characters and patterns
    query = query.strip()

    # Remove control characters but preserve space-like characters as spaces
    query = re.sub(
        r"[\x00-\x08\x0e-\x1f\x7f-\x9f]", "", query
    )  # Remove control chars but keep tab/newline
    query = re.sub(r"[\t\n\r\f\v]", " ", query)  # Convert tabs/newlines to spaces
    query = re.sub(r"\s+", " ", query)  # Normalize multiple spaces

    # Remove SQL injection patterns (basic protection)
    dangerous_patterns = [
        r"';\s*DROP\s+TABLE",
        r"';\s*DELETE\s+FROM",
        r"';\s*INSERT\s+INTO",
        r"';\s*UPDATE\s+",
        r"UNION\s+SELECT",
        r"<script[^>]*>",
        r"javascript:",
        r"vbscript:",
        r"onload\s*=",
        r"onerror\s*=",
    ]

    for pattern in dangerous_patterns:
        query = re.sub(pattern, "", query, flags=re.IGNORECASE)

    # Limit length
    if len(query) > 500:
        query = query[:500]

    return query.strip()


def normalize_document_type(doc_type: str) -> Optional[str]:
    """Normalize document type to standard Polish legal document types.

    Args:
        doc_type: Raw document type string

    Returns:
        Normalized document type or None if invalid
    """
    if not doc_type or not isinstance(doc_type, str):
        return None

    doc_type = doc_type.lower().strip()

    # Mapping of variations to standard types
    type_mappings = {
        "act": "ustawa",
        "law": "ustawa",
        "ustawy": "ustawa",
        "ustaw": "ustawa",
        "ustawa": "ustawa",
        "regulation": "rozporządzenie",
        "rozporzadzenie": "rozporządzenie",
        "rozporządzenie": "rozporządzenie",
        "rozporządzeń": "rozporządzenie",
        "code": "kodeks",
        "kodeks": "kodeks",
        "kodeksu": "kodeks",
        "kodeksy": "kodeks",
        "constitution": "konstytucja",
        "konstytucja": "konstytucja",
        "konstytucji": "konstytucja",
        "decree": "dekret",
        "dekret": "dekret",
        "dekretu": "dekret",
        "dekrety": "dekret",
        "resolution": "uchwała",
        "uchwala": "uchwała",
        "uchwała": "uchwała",
        "uchwały": "uchwała",
    }

    return type_mappings.get(doc_type)


def extract_legal_references(text: str) -> List[str]:
    """Extract legal references from text.

    Args:
        text: Text content to analyze

    Returns:
        List of legal references found
    """
    if not text or not isinstance(text, str):
        return []

    references = []

    # Patterns for different types of legal references
    patterns = [
        # Article references: art. 123, art. 45a
        r"art\.\s*(\d+[a-z]?)",
        # Paragraph references: § 1, § 23
        r"§\s*(\d+)",
        # Point references: pkt 1), punkt 3)
        r"(?:pkt|punkt)\s*(\d+)\)",
        # Act references: ustawa z dnia...
        r"ustaw[aeęiy]?\s+z\s+dnia\s+\d{1,2}\s+\w+\s+\d{4}\s+r\.",
        # Code references: Kodeks cywilny, Kodeks karny
        r"[Kk]odeks\s+\w+",
        # Constitution references
        r"[Kk]onstytucj[aeęiy]?\s+(?:Rzeczypospolitej\s+Polskiej)?",
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            references.append(match.group(0))

    # Remove duplicates while preserving order
    seen = set()
    unique_references = []
    for ref in references:
        ref_normalized = ref.lower().strip()
        if ref_normalized not in seen:
            seen.add(ref_normalized)
            unique_references.append(ref.strip())

    return unique_references


def parse_date_string(date_str: str) -> Optional[str]:
    """Parse Polish date string to ISO format.

    Args:
        date_str: Polish date string (e.g., "12 marca 2023")

    Returns:
        ISO date string or None if parsing fails
    """
    if not date_str or not isinstance(date_str, str):
        return None

    # Polish month names to numbers
    polish_months = {
        "stycznia": "01",
        "lutego": "02",
        "marca": "03",
        "kwietnia": "04",
        "maja": "05",
        "czerwca": "06",
        "lipca": "07",
        "sierpnia": "08",
        "września": "09",
        "października": "10",
        "listopada": "11",
        "grudnia": "12",
    }

    # Pattern for Polish date: "12 marca 2023"
    pattern = r"(\d{1,2})\s+(\w+)\s+(\d{4})"
    match = re.search(pattern, date_str.lower())

    if match:
        day_num = int(match.group(1))
        month_name = match.group(2)
        year = match.group(3)

        # Validate day number
        if day_num < 1 or day_num > 31:
            return None

        day = str(day_num).zfill(2)
        month = polish_months.get(month_name)
        if month:
            # Additional validation for month-day combinations
            if month in ["02"] and day_num > 29:  # February
                return None
            elif (
                month in ["04", "06", "09", "11"] and day_num > 30
            ):  # April, June, Sept, Nov
                return None
            return f"{year}-{month}-{day}"

    return None


def format_eli_id(eli_id: str) -> str:
    """Format ELI ID consistently.

    Args:
        eli_id: Raw ELI identifier

    Returns:
        Formatted ELI ID
    """
    if not eli_id or not isinstance(eli_id, str):
        return ""

    eli_id = eli_id.strip()

    # Ensure lowercase country code
    if eli_id.startswith("PL/"):
        eli_id = "pl/" + eli_id[3:]

    # Remove duplicate slashes
    eli_id = re.sub(r"/+", "/", eli_id)

    # Remove trailing slash
    eli_id = eli_id.rstrip("/")

    return eli_id


def calculate_document_complexity(text: str) -> int:
    """Calculate complexity score for legal document.

    Args:
        text: Document text content

    Returns:
        Complexity score (0-100)
    """
    if not text or not isinstance(text, str):
        return 0

    score = 0

    # Length factor (normalized)
    length_score = min(len(text) / 10000, 20)  # Max 20 points for length
    score += length_score

    # Article count
    article_count = len(re.findall(r"art\.\s*\d+", text, re.IGNORECASE))
    article_score = min(article_count * 2, 30)  # Max 30 points for articles
    score += article_score

    # Cross-references
    references = extract_legal_references(text)
    ref_score = min(len(references), 25)  # Max 25 points for references
    score += ref_score

    # Complex legal terms
    complex_terms = [
        r"nowelizacja",
        r"rozporządzenie",
        r"delegacja\s+ustawowa",
        r"przepisy\s+przejściowe",
        r"vacatio\s+legis",
        r"konstytucyjność",
    ]

    complex_term_count = sum(
        len(re.findall(pattern, text, re.IGNORECASE)) for pattern in complex_terms
    )
    complex_score = min(complex_term_count * 3, 25)  # Max 25 points
    score += complex_score

    return min(int(score), 100)


def is_amendment_document(text: str, title: str = "") -> bool:
    """Check if document is an amendment to existing law.

    Args:
        text: Document text content
        title: Document title

    Returns:
        True if document appears to be an amendment
    """
    if not text and not title:
        return False

    content = f"{title} {text}".lower()

    amendment_indicators = [
        r"ustaw[aeęiy]?\s+o\s+zmian[ieę]",  # "ustawa o zmianie"
        r"ustaw[aeęiy]?\s+.*?\s+zmienia",
        r"nowelizacj[aeęiy]?\s+ustaw[ywieęy]",
        r"zmian[aeęy]?\s+w\s+ustaw[ieę]",
        r"o\s+zmian[ieę]\s+ustaw[ywieęy]",
        r"przepisy\s+wprowadzające",
        r"zmienia\s+się\s+ustaw[ęę]",
        r"w\s+ustaw[ieę].*?wprowadza\s+się",
    ]

    return any(
        re.search(pattern, content, re.IGNORECASE) for pattern in amendment_indicators
    )


def extract_affected_articles(text: str) -> List[str]:
    """Extract articles that are affected by amendments.

    Args:
        text: Document text content

    Returns:
        List of affected article numbers
    """
    if not text or not isinstance(text, str):
        return []

    affected_articles = []

    # Patterns for amendment language and general article references
    patterns = [
        r"w\s+art\.\s*(\d+[a-z]?)\s+.*?zmienia\s+się",
        r"w\s+art\.\s*(\d+[a-z]?)\s+.*?dodaje\s+się",
        r"art\.\s*(\d+[a-z]?)\s+otrzymuje\s+brzmienie",
        r"uchyla\s+się\s+art\.\s*(\d+[a-z]?)",
        r"art\.\s*(\d+[a-z]?)\s+.*?zastępuje\s+się",
        r"art\.\s*(\d+[a-z]?)\s+i\s+art\.\s*(\d+[a-z]?)\s+zastępuje\s+się",  # Multiple articles
        r"art\.\s*(\d+[a-z]?)\s+zmienia\s+się",
        r"art\.\s*(\d+[a-z]?)\s+dodaje\s+się",
        r"art\.\s*(\d+[a-z]?)(?:\s|,|$)",  # Simple article reference
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Handle patterns with multiple groups
            for i in range(1, match.lastindex + 1 if match.lastindex else 2):
                try:
                    article_num = match.group(i)
                    if article_num and article_num not in affected_articles:
                        affected_articles.append(article_num)
                except IndexError:
                    break

    return sorted(affected_articles, key=lambda x: (len(x), x))


def clean_legal_text(text: str) -> str:
    """Clean and normalize legal text content.

    Args:
        text: Raw legal text

    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove HTML tags if present
    text = re.sub(r"<[^>]+>", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)

    # Remove excessive punctuation
    text = re.sub(r"\.{3,}", "...", text)
    text = re.sub(r"-{2,}", "--", text)

    # Normalize Polish diacritics consistency
    # (keeping original characters but removing any encoding artifacts)
    text = text.replace("\u00a0", " ")  # Non-breaking space
    text = text.replace("\u2013", "-")  # En dash
    text = text.replace("\u2014", "--")  # Em dash
    text = text.replace("\u2019", "'")  # Right single quotation mark
    text = text.replace("\u201c", '"')  # Left double quotation mark
    text = text.replace("\u201d", '"')  # Right double quotation mark

    return text.strip()
