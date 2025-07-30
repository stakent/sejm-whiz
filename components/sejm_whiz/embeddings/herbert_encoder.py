"""Simplified HerBERT encoder interface for Polish legal text."""

import numpy as np
from typing import List, Optional, Union
import logging

from .herbert_embedder import HerBERTEmbedder, get_herbert_embedder, EmbeddingResult
from .config import EmbeddingConfig, get_embedding_config

logger = logging.getLogger(__name__)


class HerBERTEncoder:
    """Simplified interface for HerBERT encoding operations."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize HerBERT encoder with configuration."""
        self.config = config or get_embedding_config()
        self.embedder = get_herbert_embedder(config)
    
    def encode(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Encode text(s) to embeddings.
        
        Args:
            texts: Single text string or list of texts
            
        Returns:
            Single embedding array or list of embeddings
        """
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts
        
        # Generate embeddings
        results = self.embedder.embed_texts(text_list)
        embeddings = [result.embedding for result in results]
        
        return embeddings[0] if is_single else embeddings
    
    def encode_with_metadata(self, texts: Union[str, List[str]]) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """
        Encode text(s) with full metadata.
        
        Args:
            texts: Single text string or list of texts
            
        Returns:
            Single EmbeddingResult or list of EmbeddingResults
        """
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts
        
        results = self.embedder.embed_texts(text_list)
        
        return results[0] if is_single else results
    
    def encode_legal_document(self, 
                            title: str, 
                            content: str, 
                            document_type: str = "unknown") -> EmbeddingResult:
        """
        Encode legal document with specialized processing.
        
        Args:
            title: Document title
            content: Document content
            document_type: Type of legal document
            
        Returns:
            EmbeddingResult with legal document-specific processing
        """
        return self.embedder.embed_legal_document(title, content, document_type)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this encoder."""
        return self.config.embedding_dim
    
    def get_model_name(self) -> str:
        """Get the name of the underlying model."""
        return self.config.model_name
    
    def is_ready(self) -> bool:
        """Check if the encoder is ready to process text."""
        return self.embedder._model_loaded
    
    def preload(self) -> None:
        """Preload the model to avoid delays on first encoding."""
        if not self.embedder._model_loaded:
            logger.info("Preloading HerBERT model...")
            self.embedder._initialize_model()
            logger.info("HerBERT model preloaded successfully")
    
    def cleanup(self) -> None:
        """Clean up resources used by the encoder."""
        self.embedder.cleanup()


# Global encoder instance
_herbert_encoder: Optional[HerBERTEncoder] = None


def get_herbert_encoder(config: Optional[EmbeddingConfig] = None) -> HerBERTEncoder:
    """Get global HerBERT encoder instance."""
    global _herbert_encoder
    
    if _herbert_encoder is None:
        _herbert_encoder = HerBERTEncoder(config)
    
    return _herbert_encoder