"""Tokenization utilities for Polish legal documents."""

import re
from typing import List, Tuple, Dict


def _try_import_spacy():
    """Lazy import of spacy to avoid numpy compatibility issues."""
    try:
        import spacy
        from spacy.lang.pl import Polish

        return True, spacy, Polish
    except (ImportError, ValueError):
        # ValueError can occur due to numpy compatibility issues
        return False, None, None


# Don't import spacy at module level - do it lazily


class PolishTokenizer:
    """Polish language tokenizer with legal document support."""

    def __init__(self, model_name: str = "pl_core_news_sm"):
        self.model_name = model_name
        self.nlp = None
        self._spacy_available = None
        self._spacy = None
        self._polish = None

        # Legal document specific sentence boundaries
        self.sentence_patterns = [
            r"(?<=\.)\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ])",  # After period before capital
            r"(?<=;)\s+(?=\d+\))",  # After semicolon before numbered point
            r"(?<=:)\s*\n",  # After colon with newline
        ]

        # Paragraph detection patterns
        self.paragraph_patterns = [
            r"\n\s*\n",  # Double newline
            r"\n\s*(?=\d+\.)",  # Before numbered list
            r"\n\s*(?=[A-ZĄĆĘŁŃÓŚŹŻ]{2,})",  # Before capitalized section
        ]

    def _ensure_spacy(self):
        """Ensure spacy is loaded if available."""
        if self._spacy_available is None:
            self._spacy_available, self._spacy, self._polish = _try_import_spacy()

            if self._spacy_available:
                try:
                    self.nlp = self._spacy.load(self.model_name)
                except OSError:
                    # Fallback to basic Polish tokenizer if model not available
                    self.nlp = self._polish() if self._polish else None

        return self._spacy_available

    def tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if not text:
            return []

        if not self._ensure_spacy() or not self.nlp:
            # Fallback to simple word splitting with basic punctuation handling
            import string

            words = []
            for word in text.split():
                # Separate punctuation from words
                if word and word[-1] in string.punctuation:
                    if len(word) > 1:
                        words.extend([word[:-1], word[-1]])
                    else:
                        words.append(word)
                else:
                    words.append(word)
            return [w for w in words if w]

        doc = self.nlp(text)
        return [token.text for token in doc if not token.is_space]

    def tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences with legal document awareness."""
        if not text:
            return []

        if not self._ensure_spacy() or not self.nlp:
            # Fallback to simple sentence splitting
            # Split on sentence endings but not on common abbreviations
            sentences = re.split(
                r"(?<!Art)(?<!art)(?<!§)\.(?=\s+[A-ZĄĆĘŁŃÓŚŹŻ])|[!?]+", text
            )
            return [s.strip() for s in sentences if s.strip()]

        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        # Additional sentence splitting for legal patterns
        enhanced_sentences = []
        for sentence in sentences:
            # Apply legal-specific sentence patterns
            parts = [sentence]
            for pattern in self.sentence_patterns:
                new_parts = []
                for part in parts:
                    new_parts.extend(re.split(pattern, part))
                parts = [p.strip() for p in new_parts if p.strip()]
            enhanced_sentences.extend(parts)

        return enhanced_sentences

    def tokenize_paragraphs(self, text: str) -> List[str]:
        """Tokenize text into paragraphs."""
        if not text:
            return []

        # Split by paragraph patterns
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def get_linguistic_features(self, text: str) -> Dict[str, List[str]]:
        """Extract linguistic features from text."""
        if not text:
            return {"tokens": [], "lemmas": [], "pos_tags": [], "entities": []}

        if not self._ensure_spacy() or not self.nlp:
            # Fallback with basic tokenization
            tokens = text.split()
            return {
                "tokens": tokens,
                "lemmas": tokens,  # Same as tokens without proper lemmatization
                "pos_tags": ["UNKNOWN"] * len(tokens),
                "entities": [],
            }

        doc = self.nlp(text)

        return {
            "tokens": [token.text for token in doc],
            "lemmas": [token.lemma_ for token in doc],
            "pos_tags": [token.pos_ for token in doc],
            "entities": [(ent.text, ent.label_) for ent in doc.ents],
        }


class LegalDocumentTokenizer:
    """Specialized tokenizer for legal document structure."""

    def __init__(self, model_name: str = "pl_core_news_sm"):
        self.tokenizer = PolishTokenizer(model_name)

        # Legal document structure patterns
        self.structure_patterns = {
            "article": r"(?:art\.|artykuł)\s*(\d+)",
            "paragraph": r"§\s*(\d+)",
            "point": r"(?:pkt|punkt)\s*(\d+)",
            "letter": r"([a-z])\)",
            "chapter": r"(?:rozdz\.|rozdział)\s*(\d+|[IVX]+)",
            "section": r"(?:dział|sekcja)\s*(\d+|[IVX]+)",
        }

    def extract_legal_structure(self, text: str) -> Dict[str, List[Tuple[str, int]]]:
        """Extract legal document structure elements."""
        if not text:
            return {key: [] for key in self.structure_patterns.keys()}

        structure = {}

        for element_type, pattern in self.structure_patterns.items():
            matches = []
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append((match.group(), match.start()))
            structure[element_type] = matches

        return structure

    def segment_by_structure(self, text: str) -> List[Dict[str, str]]:
        """Segment text by legal document structure."""
        if not text:
            return []

        # Find all structural markers
        markers = []
        for element_type, pattern in self.structure_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                markers.append(
                    {
                        "type": element_type,
                        "text": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                    }
                )

        # Sort markers by position
        markers.sort(key=lambda x: x["start"])

        # Create segments
        segments = []
        for i, marker in enumerate(markers):
            start = marker["end"]
            end = markers[i + 1]["start"] if i + 1 < len(markers) else len(text)

            content = text[start:end].strip()
            if content:
                segments.append(
                    {
                        "type": marker["type"],
                        "marker": marker["text"],
                        "content": content,
                    }
                )

        return segments

    def tokenize_legal_document(self, text: str) -> Dict[str, any]:
        """Complete tokenization for legal documents."""
        return {
            "paragraphs": self.tokenizer.tokenize_paragraphs(text),
            "sentences": self.tokenizer.tokenize_sentences(text),
            "words": self.tokenizer.tokenize_words(text),
            "structure": self.extract_legal_structure(text),
            "segments": self.segment_by_structure(text),
            "linguistic_features": self.tokenizer.get_linguistic_features(text),
        }
