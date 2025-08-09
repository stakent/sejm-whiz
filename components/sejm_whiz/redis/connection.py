"""Redis connection management and health checks."""

import redis
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

from .config import get_redis_config, RedisConfig

logger = logging.getLogger(__name__)


class RedisConnection:
    """Redis connection manager with health checks and reconnection logic."""

    def __init__(self, config: Optional[RedisConfig] = None):
        self.config = config or get_redis_config()
        self._client: Optional[redis.Redis] = None
        self._connection_pool: Optional[redis.ConnectionPool] = None

    @property
    def client(self) -> redis.Redis:
        """Get Redis client, creating connection if needed."""
        if self._client is None:
            self._connect()
        return self._client

    def _connect(self) -> None:
        """Create Redis connection pool and client."""
        try:
            # Create connection pool
            self._connection_pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=True,
            )

            # Create Redis client
            self._client = redis.Redis(connection_pool=self._connection_pool)

            # Test connection
            self._client.ping()
            logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def disconnect(self) -> None:
        """Close Redis connection."""
        if self._connection_pool:
            self._connection_pool.disconnect()
            self._connection_pool = None
        self._client = None
        logger.info("Disconnected from Redis")

    def reconnect(self) -> None:
        """Reconnect to Redis."""
        self.disconnect()
        self._connect()

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive Redis health check."""
        health_status = {
            "connection": False,
            "ping": False,
            "memory_usage": None,
            "connected_clients": None,
            "keyspace_hits": None,
            "keyspace_misses": None,
            "config": {
                "host": self.config.host,
                "port": self.config.port,
                "db": self.config.db,
            },
            "error": None,
        }

        try:
            # Test basic connection
            client = self.client
            health_status["connection"] = True

            # Test ping
            ping_result = client.ping()
            health_status["ping"] = ping_result

            # Get Redis info
            info = client.info()
            health_status["memory_usage"] = info.get("used_memory_human")
            health_status["connected_clients"] = info.get("connected_clients")
            health_status["keyspace_hits"] = info.get("keyspace_hits")
            health_status["keyspace_misses"] = info.get("keyspace_misses")

            logger.info("Redis health check passed")

        except Exception as e:
            health_status["error"] = str(e)
            logger.error(f"Redis health check failed: {e}")

        return health_status

    @contextmanager
    def pipeline(self):
        """Context manager for Redis pipeline operations."""
        pipe = self.client.pipeline()
        try:
            yield pipe
        except Exception as e:
            logger.error(f"Pipeline operation failed: {e}")
            raise
        finally:
            pipe.reset()


# Global Redis connection instance - initialized lazily
_redis_connection: Optional[RedisConnection] = None


def get_redis_connection(config: Optional[RedisConfig] = None) -> RedisConnection:
    """Get global Redis connection instance."""
    global _redis_connection

    if _redis_connection is None or config is not None:
        _redis_connection = RedisConnection(config)

    return _redis_connection


def get_redis_client(config: Optional[RedisConfig] = None) -> redis.Redis:
    """Get Redis client directly."""
    return get_redis_connection(config).client


def check_redis_health(config: Optional[RedisConfig] = None) -> Dict[str, Any]:
    """Perform Redis health check."""
    try:
        connection = get_redis_connection(config)
        return connection.health_check()
    except Exception as e:
        return {
            "connection": False,
            "ping": False,
            "error": str(e),
            "config": {"host": "unknown", "port": "unknown"},
        }
