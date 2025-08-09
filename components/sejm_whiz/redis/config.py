"""Redis configuration for caching and job queue functionality."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class RedisConfig(BaseSettings):
    """Redis configuration with environment variable support."""

    # Redis connection settings
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    db: int = Field(default=0, env="REDIS_DB")

    # Connection pool settings
    max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    socket_timeout: float = Field(default=5.0, env="REDIS_SOCKET_TIMEOUT")
    socket_connect_timeout: float = Field(default=5.0, env="REDIS_CONNECT_TIMEOUT")

    # Cache settings
    cache_ttl: int = Field(default=3600, env="REDIS_CACHE_TTL")  # 1 hour

    # Job queue settings
    job_queue_name: str = Field(default="sejm_whiz_jobs", env="REDIS_JOB_QUEUE")
    result_ttl: int = Field(default=86400, env="REDIS_RESULT_TTL")  # 24 hours
    job_timeout: int = Field(default=1800, env="REDIS_JOB_TIMEOUT")  # 30 minutes

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def redis_url(self) -> str:
        """Construct Redis URL for connections."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

    @classmethod
    def for_k3s(cls) -> "RedisConfig":
        """Create configuration for k3s deployment."""
        return cls(
            host="redis.sejm-whiz.svc.cluster.local",
            port=6379,
            password=os.getenv("REDIS_PASSWORD", "redis_password"),
            db=0,
        )

    @classmethod
    def for_local_dev(cls) -> "RedisConfig":
        """Create configuration for local development."""
        return cls(host="localhost", port=6379, password="redis_password", db=0)

    @classmethod
    def for_baremetal(cls) -> "RedisConfig":
        """Create configuration for baremetal deployment (no password)."""
        return cls(host="localhost", port=6379, password=None, db=0)

    @classmethod
    def for_p7(cls) -> "RedisConfig":
        """Create configuration for p7 server deployment."""
        return cls(host="p7", port=6379, password=None, db=0)


def get_redis_config() -> RedisConfig:
    """Get Redis configuration based on environment."""
    import socket

    # Check hostname FIRST (most reliable indicator)
    hostname = socket.gethostname()
    if hostname == "p7":
        return RedisConfig.for_p7()

    # Check CLI environment (from --env flag)
    cli_env = os.getenv("SEJM_WHIZ_CLI_ENV", "")
    deployment_env = os.getenv("DEPLOYMENT_ENV", "local")

    # Prioritize CLI environment setting
    env = cli_env or deployment_env

    if env == "k3s":
        return RedisConfig.for_k3s()
    elif env == "baremetal":
        return RedisConfig.for_baremetal()
    elif env == "p7":
        return RedisConfig.for_p7()
    else:
        return RedisConfig.for_local_dev()
