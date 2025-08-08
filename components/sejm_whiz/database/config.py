"""Database configuration for k3s deployment and local development."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class DatabaseConfig(BaseSettings):
    """Database configuration with environment variable support."""

    # Database connection settings
    host: str = Field(default="localhost", env="DATABASE_HOST")
    port: int = Field(default=5432, env="DATABASE_PORT")
    database: str = Field(default="sejm_whiz", env="DATABASE_NAME")
    username: str = Field(default="sejm_whiz_user", env="DATABASE_USER")
    password: str | None = Field(default="sejm_whiz_password", env="DATABASE_PASSWORD")

    # Connection pool settings
    pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DATABASE_POOL_TIMEOUT")
    pool_recycle: int = Field(default=3600, env="DATABASE_POOL_RECYCLE")

    # SSL settings
    ssl_mode: str = Field(default="prefer", env="DATABASE_SSL_MODE")

    # pgvector settings
    vector_dimensions: int = Field(
        default=768, env="VECTOR_DIMENSIONS"
    )  # HerBERT embedding size

    class Config:
        env_file = ".env"
        case_sensitive = False
        env_prefix = ""

    @property
    def database_url(self) -> str:
        """Construct database URL for SQLAlchemy."""
        if self.host == "/var/run/postgresql":
            # Unix socket connection
            return f"postgresql:///{self.database}?host={self.host}&sslmode={self.ssl_mode}"
        elif self.password:
            auth_part = f"{self.username}:{self.password}@"
        else:
            auth_part = f"{self.username}@"
        return (
            f"postgresql://{auth_part}"
            f"{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )

    @property
    def async_database_url(self) -> str:
        """Construct async database URL for SQLAlchemy."""
        if self.password:
            auth_part = f"{self.username}:{self.password}@"
        else:
            auth_part = f"{self.username}@"
        return (
            f"postgresql+asyncpg://{auth_part}"
            f"{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )

    @classmethod
    def for_k3s(cls) -> "DatabaseConfig":
        """Create configuration for k3s deployment."""
        return cls(
            host="postgresql-pgvector.sejm-whiz.svc.cluster.local",
            port=5432,
            database="sejm_whiz",
            username="sejm_whiz_user",
            password=os.getenv("POSTGRES_PASSWORD", "sejm_whiz_password"),
            ssl_mode=os.getenv(
                "DATABASE_SSL_MODE", "prefer"
            ),  # Allow env override, default to prefer for k3s
        )

    @classmethod
    def for_local_dev(cls) -> "DatabaseConfig":
        """Create configuration for local development."""
        return cls(
            host="localhost",
            port=5432,
            database="sejm_whiz",
            username="sejm_whiz_user",
            password="sejm_whiz_password",
            ssl_mode="prefer",
        )

    @classmethod
    def for_baremetal(cls) -> "DatabaseConfig":
        """Create configuration for baremetal deployment with system PostgreSQL."""
        return cls(
            host="localhost",
            port=5432,
            database="sejm_whiz",
            username="sejm_whiz_user",
            password="sejm_whiz_password",
            ssl_mode="disable",
        )

    @classmethod
    def for_p7(cls) -> "DatabaseConfig":
        """Create configuration for p7 server deployment."""
        return cls(
            host="p7",
            port=5432,
            database="sejm_whiz",
            username="sejm_whiz_user",
            password="sejm_whiz_password",
            ssl_mode="disable",
        )


# Global configuration instance - initialized lazily
_db_config_cache = None


def get_db_config() -> DatabaseConfig:
    """Get cached database configuration."""
    global _db_config_cache
    if _db_config_cache is None:
        _db_config_cache = get_database_config()
    return _db_config_cache


# For backward compatibility
db_config = None  # Will be set on first access


def get_database_config() -> DatabaseConfig:
    """Get database configuration based on environment."""
    # Check CLI environment first (from --env flag)
    cli_env = os.getenv("SEJM_WHIZ_CLI_ENV", "")
    deployment_env = os.getenv("DEPLOYMENT_ENV", "local")

    # Prioritize CLI environment setting
    env = cli_env or deployment_env

    if env == "k3s":
        return DatabaseConfig.for_k3s()
    elif env == "baremetal":
        return DatabaseConfig.for_baremetal()
    elif env == "p7":
        return DatabaseConfig.for_p7()
    elif env in ["docker-compose", "development"]:
        # Use environment variables for Docker Compose and development
        return DatabaseConfig(
            host=os.getenv("DATABASE_HOST", "localhost"),
            port=int(os.getenv("DATABASE_PORT", "5432")),
            database=os.getenv("DATABASE_NAME", "sejm_whiz"),
            username=os.getenv("DATABASE_USER", "sejm_whiz_user"),
            password=os.getenv("DATABASE_PASSWORD", "sejm_whiz_password"),
            ssl_mode=os.getenv("DATABASE_SSL_MODE", "prefer"),
        )
    else:
        return DatabaseConfig.for_local_dev()


def create_database_engine(config: Optional[DatabaseConfig] = None):
    """Create SQLAlchemy engine with proper configuration."""
    if config is None:
        config = get_database_config()

    engine = create_engine(
        config.database_url,
        pool_size=config.pool_size,
        max_overflow=config.max_overflow,
        pool_timeout=config.pool_timeout,
        pool_recycle=config.pool_recycle,
        echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
    )

    return engine


def create_session_factory(engine):
    """Create SQLAlchemy session factory."""
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)
