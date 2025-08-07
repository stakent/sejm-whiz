"""Cache configuration for persistent storage."""

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class CacheConfig(BaseSettings):
    """Configuration for persistent disc cache system."""

    # Base cache directory
    cache_root: str = Field(default="/var/lib/sejm-whiz/cache", env="CACHE_ROOT_DIR")

    # API response cache settings
    api_cache_ttl: int = Field(default=86400, env="API_CACHE_TTL")  # 24 hours
    max_cache_size_mb: int = Field(default=1024, env="MAX_CACHE_SIZE_MB")  # 1GB

    # Cache cleanup settings
    cleanup_interval: int = Field(default=3600, env="CACHE_CLEANUP_INTERVAL")  # 1 hour
    max_age_days: int = Field(default=30, env="CACHE_MAX_AGE_DAYS")

    # Compression settings
    compress_responses: bool = Field(default=True, env="CACHE_COMPRESS")
    compression_level: int = Field(default=6, env="CACHE_COMPRESSION_LEVEL")

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def api_responses_dir(self) -> Path:
        """API responses cache directory."""
        return Path(self.cache_root) / "api-responses"

    @property
    def sejm_api_dir(self) -> Path:
        """Sejm API responses cache directory."""
        return self.api_responses_dir / "sejm"

    @property
    def eli_api_dir(self) -> Path:
        """ELI API responses cache directory."""
        return self.api_responses_dir / "eli"

    @property
    def processed_data_dir(self) -> Path:
        """Processed data cache directory."""
        return Path(self.cache_root) / "processed-data"

    @property
    def embeddings_dir(self) -> Path:
        """Embeddings cache directory."""
        return Path(self.cache_root) / "embeddings"

    @property
    def metadata_dir(self) -> Path:
        """Metadata cache directory."""
        return Path(self.cache_root) / "metadata"

    def ensure_directories(self) -> None:
        """Create cache directories if they don't exist."""
        directories = [
            self.api_responses_dir,
            self.sejm_api_dir,
            self.eli_api_dir,
            self.processed_data_dir,
            self.embeddings_dir,
            self.metadata_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_baremetal(cls) -> "CacheConfig":
        """Create configuration for baremetal deployment."""
        return cls(cache_root="/var/lib/sejm-whiz/cache")

    @classmethod
    def for_local_dev(cls) -> "CacheConfig":
        """Create configuration for local development."""
        return cls(cache_root="./cache")


def get_cache_config() -> CacheConfig:
    """Get cache configuration based on environment."""
    deployment_env = os.getenv("DEPLOYMENT_ENV", "local")

    if deployment_env in ["baremetal", "p7_baremetal", "p7"]:
        return CacheConfig.for_baremetal()
    else:
        return CacheConfig.for_local_dev()
