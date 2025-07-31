"""Redis component for caching and job queue functionality."""

from .config import RedisConfig, get_redis_config
from .connection import (
    RedisConnection,
    get_redis_connection,
    get_redis_client,
    check_redis_health,
)
from .cache import RedisCache, get_redis_cache
from .queue import RedisJobQueue, get_redis_queue, Job, JobStatus, JobPriority

__all__ = [
    # Configuration
    "RedisConfig",
    "get_redis_config",
    # Connection management
    "RedisConnection",
    "get_redis_connection",
    "get_redis_client",
    "check_redis_health",
    # Caching
    "RedisCache",
    "get_redis_cache",
    # Job queue
    "RedisJobQueue",
    "get_redis_queue",
    "Job",
    "JobStatus",
    "JobPriority",
]
