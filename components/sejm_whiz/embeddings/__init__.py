"""Embedding generation component using HerBERT for Polish legal documents."""

from .config import EmbeddingConfig, get_embedding_config
from .herbert_embedder import HerBERTEmbedder, get_herbert_embedder, EmbeddingResult
from .herbert_encoder import HerBERTEncoder, get_herbert_encoder
from .bag_embeddings import (
    BagEmbeddingsGenerator, get_bag_embeddings_generator, 
    BagEmbeddingResult
)
from .similarity import (
    SimilarityCalculator, get_similarity_calculator,
    SimilarityResult, SimilarityMatrix,
    cosine_similarity, euclidean_distance, find_most_similar_embeddings
)
from .batch_processor import (
    BatchProcessor, get_batch_processor, BatchJob, BatchResult,
    batch_encode_texts, batch_generate_bag_embeddings, batch_calculate_similarities
)

# Optional import for embedding_operations (requires redis dependency)
try:
    from .embedding_operations import EmbeddingOperations, get_embedding_operations
    _embedding_operations_available = True
except ImportError:
    EmbeddingOperations = None
    get_embedding_operations = None
    _embedding_operations_available = False

__all__ = [
    # Configuration
    "EmbeddingConfig",
    "get_embedding_config",
    
    # HerBERT embedder and encoder
    "HerBERTEmbedder",
    "get_herbert_embedder", 
    "EmbeddingResult",
    "HerBERTEncoder",
    "get_herbert_encoder",
    
    # Bag of embeddings
    "BagEmbeddingsGenerator",
    "get_bag_embeddings_generator",
    "BagEmbeddingResult",
    
    # Similarity calculations
    "SimilarityCalculator",
    "get_similarity_calculator",
    "SimilarityResult",
    "SimilarityMatrix",
    "cosine_similarity",
    "euclidean_distance",
    "find_most_similar_embeddings",
    
    # Batch processing
    "BatchProcessor",
    "get_batch_processor",
    "BatchJob",
    "BatchResult",
    "batch_encode_texts",
    "batch_generate_bag_embeddings",
    "batch_calculate_similarities",
    
    # High-level operations (optional, requires redis)
    "EmbeddingOperations",
    "get_embedding_operations"
]
