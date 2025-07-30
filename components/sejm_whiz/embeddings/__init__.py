"""Embedding generation component using HerBERT for Polish legal documents."""

from .config import EmbeddingConfig, get_embedding_config
from .herbert_embedder import HerBERTEmbedder, get_herbert_embedder, EmbeddingResult
from .embedding_operations import EmbeddingOperations, get_embedding_operations

__all__ = [
    # Configuration
    "EmbeddingConfig",
    "get_embedding_config",
    
    # HerBERT embedder
    "HerBERTEmbedder",
    "get_herbert_embedder", 
    "EmbeddingResult",
    
    # High-level operations
    "EmbeddingOperations",
    "get_embedding_operations"
]
