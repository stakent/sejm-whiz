"""Embedding generation configuration for HerBERT and other models."""

import os
from typing import Optional, List, Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings


class EmbeddingConfig(BaseSettings):
    """Embedding configuration with environment variable support."""
    
    # Model settings
    model_name: str = Field(default="allegro/herbert-klej-cased-v1", env="EMBEDDING_MODEL_NAME")
    model_cache_dir: str = Field(default="./models", env="EMBEDDING_MODEL_CACHE_DIR")
    device: str = Field(default="auto", env="EMBEDDING_DEVICE")  # auto, cpu, cuda, mps
    
    # HerBERT specific settings
    max_length: int = Field(default=512, env="EMBEDDING_MAX_LENGTH")
    embedding_dim: int = Field(default=768, env="EMBEDDING_DIM")
    pooling_strategy: str = Field(default="mean", env="EMBEDDING_POOLING")  # mean, cls, max
    
    # Text preprocessing
    normalize_text: bool = Field(default=True, env="EMBEDDING_NORMALIZE_TEXT")
    remove_special_chars: bool = Field(default=False, env="EMBEDDING_REMOVE_SPECIAL_CHARS")
    lowercase: bool = Field(default=False, env="EMBEDDING_LOWERCASE")  # HerBERT is cased
    
    # Chunking settings for long documents
    chunk_size: int = Field(default=400, env="EMBEDDING_CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="EMBEDDING_CHUNK_OVERLAP")
    chunk_strategy: str = Field(default="sentence", env="EMBEDDING_CHUNK_STRATEGY")  # sentence, token, fixed
    
    # Batch processing
    batch_size: int = Field(default=16, env="EMBEDDING_BATCH_SIZE")
    max_workers: int = Field(default=4, env="EMBEDDING_MAX_WORKERS")
    
    # Performance settings
    use_fast_tokenizer: bool = Field(default=True, env="EMBEDDING_USE_FAST_TOKENIZER")
    compile_model: bool = Field(default=False, env="EMBEDDING_COMPILE_MODEL")  # PyTorch 2.0 compile
    use_half_precision: bool = Field(default=False, env="EMBEDDING_USE_HALF_PRECISION")
    
    # Quality control
    min_text_length: int = Field(default=10, env="EMBEDDING_MIN_TEXT_LENGTH")
    max_text_length: int = Field(default=50000, env="EMBEDDING_MAX_TEXT_LENGTH")
    similarity_threshold: float = Field(default=0.8, env="EMBEDDING_SIMILARITY_THRESHOLD")
    
    # Caching
    cache_embeddings: bool = Field(default=True, env="EMBEDDING_CACHE_EMBEDDINGS")
    cache_ttl: int = Field(default=86400, env="EMBEDDING_CACHE_TTL")  # 24 hours
    
    # Legal document specific settings
    legal_section_weights: Dict[str, float] = Field(
        default={
            "title": 2.0,
            "article": 1.5,
            "paragraph": 1.0,
            "definition": 1.8,
            "amendment": 1.3
        },
        env="EMBEDDING_LEGAL_SECTION_WEIGHTS"
    )
    
    # Alternative models for comparison
    alternative_models: List[str] = Field(
        default=[
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "sentence-transformers/LaBSE"
        ],
        env="EMBEDDING_ALTERNATIVE_MODELS"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def model_path(self) -> str:
        """Get full model path including cache directory."""
        return os.path.join(self.model_cache_dir, self.model_name.replace("/", "_"))
    
    @classmethod
    def for_gpu(cls) -> "EmbeddingConfig":
        """Create configuration optimized for GPU processing."""
        return cls(
            device="cuda",
            batch_size=32,
            use_half_precision=True,
            compile_model=True,
            max_workers=2  # Fewer workers for GPU
        )
    
    @classmethod
    def for_cpu(cls) -> "EmbeddingConfig":
        """Create configuration optimized for CPU processing."""
        return cls(
            device="cpu",
            batch_size=8,
            use_half_precision=False,
            compile_model=False,
            max_workers=4
        )
    
    @classmethod
    def for_production(cls) -> "EmbeddingConfig":
        """Create configuration for production environment."""
        return cls(
            device="auto",
            batch_size=16,
            cache_embeddings=True,
            compile_model=True,
            similarity_threshold=0.85,
            chunk_size=300,  # Smaller chunks for better precision
            max_workers=2
        )


def get_embedding_config() -> EmbeddingConfig:
    """Get embedding configuration based on environment."""
    env = os.getenv("ENVIRONMENT", "development")
    device_preference = os.getenv("EMBEDDING_DEVICE_PREFERENCE", "auto")
    
    if env == "production":
        return EmbeddingConfig.for_production()
    elif device_preference == "gpu":
        return EmbeddingConfig.for_gpu()
    elif device_preference == "cpu":
        return EmbeddingConfig.for_cpu()
    else:
        return EmbeddingConfig()