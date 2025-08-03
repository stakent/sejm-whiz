"""Redis caching operations for document and embedding caching."""

import json
import pickle
import logging
from typing import Any, Optional, Dict, List

from .connection import get_redis_client
from .config import get_redis_config, RedisConfig

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis-based caching operations for sejm-whiz."""

    def __init__(self, config: Optional[RedisConfig] = None):
        self.config = config or get_redis_config()
        self._client = get_redis_client(config)

    # Basic cache operations
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a key-value pair in cache with optional TTL."""
        try:
            ttl = ttl or self.config.cache_ttl

            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = pickle.dumps(value)

            result = self._client.setex(key, ttl, serialized_value)
            logger.debug(f"Cached key '{key}' with TTL {ttl}s")
            return result

        except Exception as e:
            logger.error(f"Failed to cache key '{key}': {e}")
            return False

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache by key."""
        try:
            value = self._client.get(key)
            if value is None:
                return None

            # Try JSON deserialization first, then pickle
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return pickle.loads(value)

        except Exception as e:
            logger.error(f"Failed to get key '{key}': {e}")
            return None

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            result = self._client.delete(key)
            logger.debug(f"Deleted key '{key}'")
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to delete key '{key}': {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return bool(self._client.exists(key))
        except Exception as e:
            logger.error(f"Failed to check existence of key '{key}': {e}")
            return False

    def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key."""
        try:
            result = self._client.expire(key, ttl)
            logger.debug(f"Set TTL {ttl}s for key '{key}'")
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to set TTL for key '{key}': {e}")
            return False

    # Document caching operations
    def cache_document(
        self, document_id: str, document_data: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """Cache legal document data."""
        key = f"document:{document_id}"
        return self.set(key, document_data, ttl)

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get cached legal document data."""
        key = f"document:{document_id}"
        return self.get(key)

    def cache_document_embedding(
        self, document_id: str, embedding: List[float], ttl: Optional[int] = None
    ) -> bool:
        """Cache document embedding vector."""
        key = f"embedding:{document_id}"
        return self.set(key, embedding, ttl)

    def get_document_embedding(self, document_id: str) -> Optional[List[float]]:
        """Get cached document embedding vector."""
        key = f"embedding:{document_id}"
        return self.get(key)

    # Search result caching
    def cache_search_results(
        self, query_hash: str, results: List[Dict[str, Any]], ttl: Optional[int] = None
    ) -> bool:
        """Cache search results for similarity queries."""
        key = f"search:{query_hash}"
        ttl = ttl or (self.config.cache_ttl // 2)  # Shorter TTL for search results
        return self.set(key, results, ttl)

    def get_search_results(self, query_hash: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results."""
        key = f"search:{query_hash}"
        return self.get(key)

    # ELI API response caching
    def cache_eli_response(
        self, eli_id: str, response_data: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """Cache ELI API response data."""
        key = f"eli:{eli_id}"
        ttl = ttl or (self.config.cache_ttl * 24)  # Longer TTL for ELI data
        return self.set(key, response_data, ttl)

    def get_eli_response(self, eli_id: str) -> Optional[Dict[str, Any]]:
        """Get cached ELI API response data."""
        key = f"eli:{eli_id}"
        return self.get(key)

    # Session and user caching
    def cache_user_session(
        self, session_id: str, session_data: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """Cache user session data."""
        key = f"session:{session_id}"
        ttl = ttl or 3600  # 1 hour for sessions
        return self.set(key, session_data, ttl)

    def get_user_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user session data."""
        key = f"session:{session_id}"
        return self.get(key)

    # Batch operations
    def mget(self, keys: List[str]) -> List[Optional[Any]]:
        """Get multiple keys at once."""
        try:
            values = self._client.mget(keys)
            results = []

            for value in values:
                if value is None:
                    results.append(None)
                else:
                    try:
                        results.append(json.loads(value))
                    except (json.JSONDecodeError, TypeError):
                        results.append(pickle.loads(value))

            return results

        except Exception as e:
            logger.error(f"Failed to get multiple keys: {e}")
            return [None] * len(keys)

    def mset(self, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple key-value pairs at once."""
        try:
            with self._client.pipeline() as pipe:
                for key, value in data.items():
                    # Serialize value
                    if isinstance(value, (dict, list)):
                        serialized_value = json.dumps(value)
                    else:
                        serialized_value = pickle.dumps(value)

                    pipe.setex(key, ttl or self.config.cache_ttl, serialized_value)

                results = pipe.execute()
                return all(results)

        except Exception as e:
            logger.error(f"Failed to set multiple keys: {e}")
            return False

    # Cache management
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching a pattern."""
        try:
            keys = self._client.keys(pattern)
            if keys:
                deleted = self._client.delete(*keys)
                logger.info(f"Deleted {deleted} keys matching pattern '{pattern}'")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Failed to clear pattern '{pattern}': {e}")
            return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            info = self._client.info()
            return {
                "used_memory": info.get("used_memory_human"),
                "used_memory_peak": info.get("used_memory_peak_human"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "hit_rate": self._calculate_hit_rate(info),
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}

    def _calculate_hit_rate(self, info: Dict[str, Any]) -> float:
        """Calculate cache hit rate."""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses

        if total == 0:
            return 0.0

        return round((hits / total) * 100, 2)


# Global cache instance
_redis_cache: Optional[RedisCache] = None


def get_redis_cache(config: Optional[RedisConfig] = None) -> RedisCache:
    """Get global Redis cache instance."""
    global _redis_cache

    if _redis_cache is None:
        _redis_cache = RedisCache(config)

    return _redis_cache
