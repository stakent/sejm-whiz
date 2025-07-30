"""Bag of embeddings implementation for Polish legal documents."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from collections import Counter
import re

from .herbert_encoder import HerBERTEncoder, get_herbert_encoder
from .config import EmbeddingConfig, get_embedding_config

logger = logging.getLogger(__name__)


@dataclass
class BagEmbeddingResult:
    """Result of bag of embeddings generation."""
    document_embedding: np.ndarray
    token_embeddings: List[np.ndarray]
    token_weights: List[float]
    tokens: List[str]
    vocabulary_size: int
    coverage_score: float
    processing_time: float
    metadata: Dict[str, Any]


class BagEmbeddingsGenerator:
    """Generate bag of embeddings for Polish legal documents."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize bag embeddings generator."""
        self.config = config or get_embedding_config()
        self.encoder = get_herbert_encoder(config)
        
        # Polish legal term patterns for weighting
        self.legal_term_patterns = {
            'act_reference': re.compile(r'\bustawa\s+z\s+dnia\s+[\d\s\w]+', re.IGNORECASE),
            'article': re.compile(r'\bart\.\s*\d+[a-z]?', re.IGNORECASE),
            'paragraph': re.compile(r'§\s*\d+', re.IGNORECASE),
            'section': re.compile(r'\bust\.\s*\d+', re.IGNORECASE),
            'definition': re.compile(r'oznacza|rozumie się|należy przez to rozumieć', re.IGNORECASE),
            'amendment': re.compile(r'zmienia się|uchyla się|dodaje się|skreśla się', re.IGNORECASE),
            'legal_entity': re.compile(r'minister|rząd|parlament|sejm|senat|prezydent', re.IGNORECASE),
            'legal_procedure': re.compile(r'procedura|postępowanie|proces|tryb', re.IGNORECASE)
        }
        
        # Term importance weights
        self.term_weights = {
            'act_reference': 2.5,
            'article': 2.0,
            'paragraph': 1.8,
            'section': 1.5,
            'definition': 2.2,
            'amendment': 2.0,
            'legal_entity': 1.3,
            'legal_procedure': 1.4
        }
        
        # Stop words for Polish legal text
        self.polish_stopwords = {
            'i', 'a', 'o', 'z', 'w', 'na', 'do', 'od', 'po', 'za', 'ze', 'we', 'to', 'ta', 'te', 'ty',
            'jest', 'są', 'był', 'była', 'było', 'były', 'będzie', 'będą', 'ma', 'mają', 'miał', 'miała',
            'nie', 'już', 'tylko', 'także', 'również', 'oraz', 'albo', 'lub', 'ani', 'czy', 'że', 'żeby',
            'gdy', 'kiedy', 'gdzie', 'jak', 'dlaczego', 'dlatego', 'ponieważ', 'jednak', 'lecz', 'ale'
        }
    
    def generate_bag_embedding(self, 
                             text: str,
                             title: str = "",
                             use_tf_idf: bool = True,
                             use_legal_weighting: bool = True) -> BagEmbeddingResult:
        """
        Generate bag of embeddings for a document.
        
        Args:
            text: Document text content
            title: Document title (optional)
            use_tf_idf: Whether to use TF-IDF weighting
            use_legal_weighting: Whether to apply legal term weighting
            
        Returns:
            BagEmbeddingResult with document embedding and metadata
        """
        logger.info("Generating bag of embeddings for document")
        
        # Combine title and text
        full_text = f"{title} {text}".strip() if title else text
        
        # Tokenize and clean
        tokens = self._tokenize_text(full_text)
        
        if not tokens:
            logger.warning("No tokens found in text")
            return self._create_empty_result()
        
        # Generate token embeddings
        logger.info(f"Generating embeddings for {len(tokens)} tokens")
        token_embeddings = self.encoder.encode(tokens)
        
        # Calculate token weights
        token_weights = self._calculate_token_weights(
            tokens, full_text, use_tf_idf, use_legal_weighting
        )
        
        # Create weighted document embedding
        document_embedding = self._create_weighted_embedding(token_embeddings, token_weights)
        
        # Calculate coverage score
        coverage_score = self._calculate_coverage_score(tokens, full_text)
        
        # Create vocabulary statistics
        vocab_size = len(set(tokens))
        
        result = BagEmbeddingResult(
            document_embedding=document_embedding,
            token_embeddings=token_embeddings,
            token_weights=token_weights,
            tokens=tokens,
            vocabulary_size=vocab_size,
            coverage_score=coverage_score,
            processing_time=0.0,  # Would track this properly in production
            metadata={
                'total_tokens': len(tokens),
                'unique_tokens': vocab_size,
                'avg_token_length': np.mean([len(token) for token in tokens]),
                'use_tf_idf': use_tf_idf,
                'use_legal_weighting': use_legal_weighting,
                'text_length': len(full_text)
            }
        )
        
        logger.info(f"Generated bag embedding with {vocab_size} unique tokens")
        return result
    
    def generate_bag_embeddings_batch(self, 
                                    documents: List[Dict[str, str]],
                                    use_tf_idf: bool = True,
                                    use_legal_weighting: bool = True) -> List[BagEmbeddingResult]:
        """
        Generate bag of embeddings for multiple documents.
        
        Args:
            documents: List of dicts with 'text' and optional 'title' keys
            use_tf_idf: Whether to use TF-IDF weighting
            use_legal_weighting: Whether to apply legal term weighting
            
        Returns:
            List of BagEmbeddingResult objects
        """
        logger.info(f"Generating bag embeddings for {len(documents)} documents")
        
        results = []
        for i, doc in enumerate(documents):
            try:
                text = doc.get('text', '')
                title = doc.get('title', '')
                
                result = self.generate_bag_embedding(
                    text=text,
                    title=title,
                    use_tf_idf=use_tf_idf,
                    use_legal_weighting=use_legal_weighting
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to generate bag embedding for document {i}: {e}")
                results.append(self._create_empty_result())
        
        logger.info(f"Generated {len(results)} bag embeddings")
        return results
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into meaningful tokens for legal documents."""
        if not text:
            return []
        
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split on whitespace and punctuation, but preserve legal markers
        tokens = re.findall(r'\w+|\d+|\$\d+|\bart\.\s*\d+[a-z]?', text.lower())
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Skip very short tokens
            if len(token) < 2:
                continue
            
            # Skip stop words (except legal terms)
            if token in self.polish_stopwords and not self._is_legal_term(token):
                continue
            
            # Skip pure numbers (unless they're legal references)
            if token.isdigit() and len(token) < 4:
                continue
            
            filtered_tokens.append(token)
        
        return filtered_tokens
    
    def _calculate_token_weights(self, 
                               tokens: List[str], 
                               text: str,
                               use_tf_idf: bool,
                               use_legal_weighting: bool) -> List[float]:
        """Calculate weights for tokens based on various strategies."""
        token_weights = np.ones(len(tokens), dtype=float)
        
        if use_tf_idf:
            # Simple TF-IDF approximation
            token_counts = Counter(tokens)
            total_tokens = len(tokens)
            
            for i, token in enumerate(tokens):
                tf = token_counts[token] / total_tokens
                # Simple IDF approximation (would use corpus stats in production)
                idf = np.log(1 + 1 / (1 + token_counts[token]))
                token_weights[i] = tf * idf
        
        if use_legal_weighting:
            # Apply legal term weighting
            for i, token in enumerate(tokens):
                legal_weight = self._get_legal_weight(token, text)
                token_weights[i] *= legal_weight
        
        # Normalize weights
        if np.sum(token_weights) > 0:
            token_weights = token_weights / np.sum(token_weights)
        
        return token_weights.tolist()
    
    def _get_legal_weight(self, token: str, text: str) -> float:
        """Get legal importance weight for a token."""
        weight = 1.0
        
        # Check if token is part of legal patterns
        for pattern_name, pattern in self.legal_term_patterns.items():
            if pattern.search(token) or token in text.lower():
                # Check if this token appears in context of this pattern
                context_matches = pattern.findall(text)
                if any(token in match.lower() for match in context_matches):
                    weight *= self.term_weights.get(pattern_name, 1.0)
        
        # Additional weights for specific legal terms
        if self._is_legal_term(token):
            weight *= 1.5
        
        # Boost longer tokens (often more informative)
        if len(token) > 6:
            weight *= 1.2
        elif len(token) > 10:
            weight *= 1.4
        
        return weight
    
    def _is_legal_term(self, token: str) -> bool:
        """Check if token is a legal term that should be preserved."""
        legal_terms = {
            'ustawa', 'rozporządzenie', 'artykuł', 'paragraf', 'ustęp', 'punkt',
            'konstytucja', 'kodeks', 'prawo', 'minister', 'rząd', 'parlament',
            'sejm', 'senat', 'prezydent', 'sąd', 'trybunał', 'prokuratura'
        }
        return token.lower() in legal_terms
    
    def _create_weighted_embedding(self, 
                                 token_embeddings: List[np.ndarray], 
                                 token_weights: List[float]) -> np.ndarray:
        """Create weighted document embedding from token embeddings."""
        if not token_embeddings:
            return np.zeros(self.encoder.get_embedding_dimension())
        
        # Convert to numpy arrays
        embeddings_array = np.array(token_embeddings)
        weights_array = np.array(token_weights).reshape(-1, 1)
        
        # Weighted average
        weighted_embedding = np.sum(embeddings_array * weights_array, axis=0)
        
        # Normalize
        norm = np.linalg.norm(weighted_embedding)
        if norm > 0:
            weighted_embedding = weighted_embedding / norm
        
        return weighted_embedding
    
    def _calculate_coverage_score(self, tokens: List[str], text: str) -> float:
        """Calculate how well tokens cover the original text."""
        if not tokens or not text:
            return 0.0
        
        # Simple coverage based on character coverage
        text_chars = set(text.lower().replace(' ', ''))
        token_chars = set(''.join(tokens))
        
        if not text_chars:
            return 0.0
        
        coverage = len(token_chars.intersection(text_chars)) / len(text_chars)
        return min(coverage, 1.0)
    
    def _create_empty_result(self) -> BagEmbeddingResult:
        """Create empty result for error cases."""
        return BagEmbeddingResult(
            document_embedding=np.zeros(self.encoder.get_embedding_dimension()),
            token_embeddings=[],
            token_weights=[],
            tokens=[],
            vocabulary_size=0,
            coverage_score=0.0,
            processing_time=0.0,
            metadata={'error': 'Failed to generate embeddings'}
        )
    
    def get_token_importance_analysis(self, result: BagEmbeddingResult) -> Dict[str, Any]:
        """Analyze token importance in the bag embedding result."""
        if not result.tokens:
            return {'error': 'No tokens to analyze'}
        
        # Sort tokens by weight
        token_weight_pairs = list(zip(result.tokens, result.token_weights))
        token_weight_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate statistics
        weights_array = np.array(result.token_weights)
        
        analysis = {
            'most_important_tokens': token_weight_pairs[:10],
            'least_important_tokens': token_weight_pairs[-5:],
            'weight_statistics': {
                'mean': float(np.mean(weights_array)),
                'std': float(np.std(weights_array)),
                'min': float(np.min(weights_array)),
                'max': float(np.max(weights_array))
            },
            'vocabulary_diversity': result.vocabulary_size / len(result.tokens) if result.tokens else 0,
            'coverage_score': result.coverage_score
        }
        
        return analysis


# Global bag embeddings generator
_bag_embeddings_generator: Optional[BagEmbeddingsGenerator] = None


def get_bag_embeddings_generator(config: Optional[EmbeddingConfig] = None) -> BagEmbeddingsGenerator:
    """Get global bag embeddings generator instance."""
    global _bag_embeddings_generator
    
    if _bag_embeddings_generator is None:
        _bag_embeddings_generator = BagEmbeddingsGenerator(config)
    
    return _bag_embeddings_generator