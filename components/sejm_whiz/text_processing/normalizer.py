"""Text normalization utilities for Polish legal documents."""

import re
import unicodedata
from typing import Dict, List, Optional


class PolishNormalizer:
    """Normalizes Polish text, handling diacritics and character variations."""
    
    def __init__(self):
        # Polish diacritic mappings
        self.diacritic_map = {
            'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n',
            'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z',
            'Ą': 'A', 'Ć': 'C', 'Ę': 'E', 'Ł': 'L', 'Ń': 'N',
            'Ó': 'O', 'Ś': 'S', 'Ź': 'Z', 'Ż': 'Z'
        }
        
        # Alternative character representations
        self.char_variants = {
            '\u201c': '"',  # Left double quotation mark
            '\u201d': '"',  # Right double quotation mark
            '\u2018': "'",  # Left single quotation mark
            '\u2019': "'",  # Right single quotation mark
            '\u2013': '-',  # En dash
            '\u2014': '-',  # Em dash
            '\u2026': '...',  # Ellipsis
        }
        
        # Number normalization patterns
        self.number_patterns = [
            (r'\b(\d+)\s*[-–—]\s*(\d+)\b', r'\1-\2'),  # Number ranges
            (r'\b(\d{1,2})\s*\.\s*(\d{1,2})\s*\.\s*(\d{4})\b', r'\1.\2.\3'),  # Dates
        ]
    
    def normalize_diacritics(self, text: str, remove: bool = False) -> str:
        """Normalize or remove Polish diacritics."""
        if not text:
            return ""
        
        if remove:
            # Remove diacritics completely
            for polish_char, latin_char in self.diacritic_map.items():
                text = text.replace(polish_char, latin_char)
        else:
            # Normalize using Unicode NFKD
            text = unicodedata.normalize('NFKD', text)
        
        return text
    
    def normalize_characters(self, text: str) -> str:
        """Normalize character variants to standard forms."""
        if not text:
            return ""
        
        for variant_char, standard_char in self.char_variants.items():
            text = text.replace(variant_char, standard_char)
        
        return text
    
    def normalize_numbers(self, text: str) -> str:
        """Normalize number and date formats."""
        if not text:
            return ""
        
        for pattern, replacement in self.number_patterns:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters."""
        if not text:
            return ""
        
        # Replace various whitespace characters with standard space
        text = re.sub(r'[\u00A0\u2000-\u200B\u2028\u2029]', ' ', text)
        
        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def normalize_text(self, text: str, remove_diacritics: bool = False) -> str:
        """Complete normalization pipeline."""
        text = self.normalize_characters(text)
        text = self.normalize_diacritics(text, remove=remove_diacritics)
        text = self.normalize_numbers(text)
        text = self.normalize_whitespace(text)
        return text


class LegalTextNormalizer:
    """Specialized normalizer for Polish legal text."""
    
    def __init__(self):
        self.polish_normalizer = PolishNormalizer()
        
        # Legal document specific normalizations
        self.legal_patterns = [
            # Standardize article references
            (r'\bart\.\s*(\d+)', r'art. \1'),
            (r'\bartykuł\s+(\d+)', r'art. \1'),
            
            # Standardize paragraph references
            (r'§\s*(\d+)', r'§ \1'),
            (r'\bpar\.\s*(\d+)', r'§ \1'),
            (r'\bparagraf\s+(\d+)', r'§ \1'),
            
            # Standardize point references
            (r'\bpkt\s*(\d+)', r'pkt \1'),
            (r'\bpunkt\s+(\d+)', r'pkt \1'),
            
            # Standardize chapter references
            (r'\brozd\.\s*(\d+)', r'rozdz. \1'),
            (r'\brozdział\s+(\d+)', r'rozdz. \1'),
            
            # Standardize common legal abbreviations
            (r'\bu\.\s*z\.?', 'ustawa z'),
            (r'\bk\.\s*c\.?', 'kodeks cywilny'),
            (r'\bk\.\s*p\.\s*c\.?', 'kodeks postępowania cywilnego'),
        ]
    
    def normalize_legal_references(self, text: str) -> str:
        """Normalize legal references to standard format."""
        if not text:
            return ""
        
        for pattern, replacement in self.legal_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def normalize_legal_text(self, text: str, remove_diacritics: bool = False) -> str:
        """Complete legal text normalization."""
        text = self.polish_normalizer.normalize_text(text, remove_diacritics)
        text = self.normalize_legal_references(text)
        return text