"""Text processing component for Polish legal documents."""

from .cleaner import TextCleaner, HTMLCleaner
from .normalizer import LegalTextNormalizer, PolishNormalizer
from .legal_parser import LegalDocumentAnalyzer

# Core functions that may use spacy - import lazily
def _get_core_functions():
    from .core import (
        TextProcessor,
        clean_legal_text,
        normalize_legal_text,
        process_legal_document,
        extract_legal_references
    )
    return TextProcessor, clean_legal_text, normalize_legal_text, process_legal_document, extract_legal_references

def _get_tokenizer_classes():
    from .tokenizer import LegalDocumentTokenizer, PolishTokenizer
    return LegalDocumentTokenizer, PolishTokenizer

def _get_entity_classes():
    from .entities import LegalEntityExtractor
    return LegalEntityExtractor

# Lazy import properties
class LazyImport:
    def __init__(self, import_func, name):
        self.import_func = import_func
        self.name = name
        self._cached = None
    
    def __call__(self, *args, **kwargs):
        if self._cached is None:
            imports = self.import_func()
            if isinstance(imports, tuple):
                self._cached = {name: cls for name, cls in zip(self.name if isinstance(self.name, list) else [self.name], imports)}
            else:
                self._cached = {self.name: imports}
        
        if isinstance(self.name, list):
            return self._cached
        return self._cached[self.name]

# Export safe imports directly
__all__ = [
    "TextCleaner",
    "HTMLCleaner",
    "LegalTextNormalizer",
    "PolishNormalizer",
    "LegalDocumentAnalyzer"
]

# Add lazy imports to globals for backward compatibility
def __getattr__(name):
    if name in ["TextProcessor", "clean_legal_text", "normalize_legal_text", "process_legal_document", "extract_legal_references"]:
        TextProcessor, clean_legal_text, normalize_legal_text, process_legal_document, extract_legal_references = _get_core_functions()
        globals().update({
            "TextProcessor": TextProcessor,
            "clean_legal_text": clean_legal_text,
            "normalize_legal_text": normalize_legal_text,
            "process_legal_document": process_legal_document,
            "extract_legal_references": extract_legal_references
        })
        return globals()[name]
    elif name in ["LegalDocumentTokenizer", "PolishTokenizer"]:
        LegalDocumentTokenizer, PolishTokenizer = _get_tokenizer_classes()
        globals().update({
            "LegalDocumentTokenizer": LegalDocumentTokenizer,
            "PolishTokenizer": PolishTokenizer
        })
        return globals()[name]
    elif name == "LegalEntityExtractor":
        LegalEntityExtractor, = _get_entity_classes()
        globals()["LegalEntityExtractor"] = LegalEntityExtractor
        return LegalEntityExtractor
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
