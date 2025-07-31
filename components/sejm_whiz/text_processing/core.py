"""Core text processing interface for Polish legal documents."""

from typing import Dict, List, Any
from .cleaner import TextCleaner, HTMLCleaner
from .normalizer import LegalTextNormalizer, PolishNormalizer
from . import tokenizer
from . import entities
from .legal_parser import LegalDocumentAnalyzer


class TextProcessor:
    """Main text processing interface for Polish legal documents."""

    def __init__(self, spacy_model: str = "pl_core_news_sm"):
        """Initialize text processor with all components."""
        self.cleaner = TextCleaner()
        self.normalizer = LegalTextNormalizer()
        self.tokenizer = tokenizer.LegalDocumentTokenizer(spacy_model)
        self.entity_extractor = entities.LegalEntityExtractor(spacy_model)
        self.document_analyzer = LegalDocumentAnalyzer()
        self.spacy_model = spacy_model

    def clean_text(self, text: str) -> str:
        """Clean HTML and formatting from text."""
        return self.cleaner.clean_text(text)

    def normalize_text(self, text: str, remove_diacritics: bool = False) -> str:
        """Normalize Polish legal text."""
        return self.normalizer.normalize_legal_text(text, remove_diacritics)

    def tokenize_text(self, text: str) -> Dict[str, Any]:
        """Tokenize text into various units."""
        return self.tokenizer.tokenize_legal_document(text)

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract named entities from text."""
        return self.entity_extractor.extract_entities(text)

    def analyze_document(self, text: str) -> Dict[str, Any]:
        """Analyze legal document structure."""
        return self.document_analyzer.analyze_document(text)

    def process_document(
        self,
        text: str,
        clean: bool = True,
        normalize: bool = True,
        remove_diacritics: bool = False,
        extract_entities: bool = True,
        analyze_structure: bool = True,
    ) -> Dict[str, Any]:
        """Complete processing pipeline for legal documents."""
        if not text:
            result = {
                "original_text": text,
                "processed_text": "",
                "processing_steps": {
                    "cleaned": clean,
                    "normalized": normalize,
                    "diacritics_removed": remove_diacritics,
                },
            }

            # Always provide tokenization
            result["tokenization"] = self.tokenize_text("")

            # Add other components if requested
            if extract_entities:
                result["entities"] = self.extract_entities("")

            if analyze_structure:
                result["analysis"] = self.analyze_document("")

            return result

        processed_text = text

        # Apply cleaning if requested
        if clean:
            processed_text = self.clean_text(processed_text)

        # Apply normalization if requested
        if normalize:
            processed_text = self.normalize_text(processed_text, remove_diacritics)

        result = {
            "original_text": text,
            "processed_text": processed_text,
            "processing_steps": {
                "cleaned": clean,
                "normalized": normalize,
                "diacritics_removed": remove_diacritics,
            },
        }

        # Tokenization (always performed on processed text)
        result["tokenization"] = self.tokenize_text(processed_text)

        # Entity extraction if requested
        if extract_entities:
            result["entities"] = self.extract_entities(processed_text)

        # Document analysis if requested
        if analyze_structure:
            result["analysis"] = self.analyze_document(processed_text)

        return result

    def get_text_statistics(self, text: str) -> Dict[str, int]:
        """Get basic statistics about the text."""
        if not text:
            return {"characters": 0, "words": 0, "sentences": 0, "paragraphs": 0}

        tokenization = self.tokenize_text(text)

        return {
            "characters": len(text),
            "words": len(tokenization.get("words", [])),
            "sentences": len(tokenization.get("sentences", [])),
            "paragraphs": len(tokenization.get("paragraphs", [])),
        }


# Convenience functions for quick access
def clean_legal_text(text: str) -> str:
    """Quick function to clean legal text."""
    processor = TextProcessor()
    return processor.clean_text(text)


def normalize_legal_text(text: str, remove_diacritics: bool = False) -> str:
    """Quick function to normalize legal text."""
    processor = TextProcessor()
    return processor.normalize_text(text, remove_diacritics)


def process_legal_document(text: str, **kwargs) -> Dict[str, Any]:
    """Quick function to process a complete legal document."""
    processor = TextProcessor()
    return processor.process_document(text, **kwargs)


def extract_legal_references(text: str) -> List[Dict[str, Any]]:
    """Quick function to extract legal references."""
    processor = TextProcessor()
    entities = processor.extract_entities(text)

    # Filter for legal reference entities
    legal_refs = []
    for entity in entities.get("entities", []):
        if "REFERENCE" in entity.get("type", ""):
            legal_refs.append(entity)

    return legal_refs


# Export main classes and functions
__all__ = [
    "TextProcessor",
    "TextCleaner",
    "HTMLCleaner",
    "LegalTextNormalizer",
    "PolishNormalizer",
    "LegalDocumentTokenizer",
    "PolishTokenizer",
    "LegalEntityExtractor",
    "LegalDocumentAnalyzer",
    "clean_legal_text",
    "normalize_legal_text",
    "process_legal_document",
    "extract_legal_references",
]
