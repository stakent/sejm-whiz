"""Cache manager for persistent storage of API responses and processed data."""

import json
import gzip
import hashlib
import logging
from datetime import datetime, UTC, timedelta
from pathlib import Path
from typing import Any, Optional, Dict, List
import pickle

from .config import CacheConfig, get_cache_config

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages persistent disc cache for API responses and processed data."""

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or get_cache_config()
        self.config.ensure_directories()
        logger.info(f"Cache manager initialized with root: {self.config.cache_root}")

    def _get_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key from endpoint and parameters."""
        # Create deterministic hash from endpoint and params
        content = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cache_file_path(self, cache_type: str, cache_key: str) -> Path:
        """Get cache file path for given type and key."""
        if cache_type == "sejm":
            base_dir = self.config.sejm_api_dir
        elif cache_type == "eli":
            base_dir = self.config.eli_api_dir
        elif cache_type == "processed":
            base_dir = self.config.processed_data_dir
        elif cache_type == "embeddings":
            base_dir = self.config.embeddings_dir
        elif cache_type == "metadata":
            base_dir = self.config.metadata_dir
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")

        # Use first 2 chars of hash for subdirectory to avoid too many files in one dir
        subdir = cache_key[:2]
        cache_dir = base_dir / subdir
        cache_dir.mkdir(parents=True, exist_ok=True)

        extension = ".gz" if self.config.compress_responses else ".json"
        return cache_dir / f"{cache_key}{extension}"

    def _write_cache_file(self, file_path: Path, data: Any) -> None:
        """Write data to cache file with optional compression."""
        try:
            if self.config.compress_responses:
                with gzip.open(
                    file_path,
                    "wt",
                    encoding="utf-8",
                    compresslevel=self.config.compression_level,
                ) as f:
                    if isinstance(data, (dict, list)):
                        json.dump(
                            data,
                            f,
                            ensure_ascii=False,
                            indent=None,
                            separators=(",", ":"),
                        )
                    else:
                        f.write(str(data))
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    if isinstance(data, (dict, list)):
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    else:
                        f.write(str(data))

            logger.debug(f"Cached data to {file_path}")

        except Exception as e:
            logger.error(f"Failed to write cache file {file_path}: {e}")
            raise

    def _read_cache_file(self, file_path: Path) -> Any:
        """Read data from cache file with optional decompression."""
        try:
            if file_path.suffix == ".gz":
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    return json.load(f)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)

        except Exception as e:
            logger.error(f"Failed to read cache file {file_path}: {e}")
            raise

    def _is_cache_valid(self, file_path: Path) -> bool:
        """Check if cache file is still valid based on TTL."""
        if not file_path.exists():
            return False

        try:
            file_age = datetime.now(UTC) - datetime.fromtimestamp(
                file_path.stat().st_mtime, UTC
            )
            return file_age.total_seconds() < self.config.api_cache_ttl
        except Exception as e:
            logger.warning(f"Failed to check cache validity for {file_path}: {e}")
            return False

    def cache_api_response(
        self, api_type: str, endpoint: str, params: Dict[str, Any], response_data: Any
    ) -> str:
        """Cache API response data."""
        cache_key = self._get_cache_key(endpoint, params)
        file_path = self._get_cache_file_path(api_type, cache_key)

        # Prepare cache entry with metadata
        cache_entry = {
            "endpoint": endpoint,
            "params": params,
            "cached_at": datetime.now(UTC).isoformat(),
            "data": response_data,
        }

        self._write_cache_file(file_path, cache_entry)
        logger.info(f"Cached {api_type} API response: {endpoint} -> {cache_key[:8]}...")
        return cache_key

    def get_cached_api_response(
        self, api_type: str, endpoint: str, params: Dict[str, Any]
    ) -> Optional[Any]:
        """Retrieve cached API response if available and valid."""
        cache_key = self._get_cache_key(endpoint, params)
        file_path = self._get_cache_file_path(api_type, cache_key)

        if not self._is_cache_valid(file_path):
            return None

        try:
            cache_entry = self._read_cache_file(file_path)
            logger.info(
                f"Cache hit for {api_type} API: {endpoint} -> {cache_key[:8]}..."
            )
            return cache_entry.get("data")
        except Exception as e:
            logger.warning(f"Failed to read cache for {cache_key}: {e}")
            return None

    def cache_processed_data(self, data_type: str, identifier: str, data: Any) -> str:
        """Cache processed data (embeddings, extracted text, etc.)."""
        cache_key = f"{data_type}_{identifier}"
        file_path = self._get_cache_file_path("processed", cache_key)

        cache_entry = {
            "data_type": data_type,
            "identifier": identifier,
            "cached_at": datetime.now(UTC).isoformat(),
            "data": data,
        }

        self._write_cache_file(file_path, cache_entry)
        logger.info(f"Cached processed data: {data_type}/{identifier}")
        return cache_key

    def get_cached_processed_data(
        self, data_type: str, identifier: str
    ) -> Optional[Any]:
        """Retrieve cached processed data."""
        cache_key = f"{data_type}_{identifier}"
        file_path = self._get_cache_file_path("processed", cache_key)

        if not self._is_cache_valid(file_path):
            return None

        try:
            cache_entry = self._read_cache_file(file_path)
            logger.info(f"Cache hit for processed data: {data_type}/{identifier}")
            return cache_entry.get("data")
        except Exception as e:
            logger.warning(f"Failed to read processed data cache: {e}")
            return None

    def cache_embeddings(self, document_id: str, embeddings: List[float]) -> str:
        """Cache document embeddings."""
        cache_key = f"doc_embeddings_{document_id}"
        file_path = self._get_cache_file_path("embeddings", cache_key)

        # Use pickle for binary data like embeddings for better performance
        try:
            with open(file_path.with_suffix(".pkl"), "wb") as f:
                pickle.dump(
                    {
                        "document_id": document_id,
                        "cached_at": datetime.now(UTC).isoformat(),
                        "embeddings": embeddings,
                    },
                    f,
                )
            logger.info(f"Cached embeddings for document {document_id}")
            return cache_key
        except Exception as e:
            logger.error(f"Failed to cache embeddings: {e}")
            raise

    def get_cached_embeddings(self, document_id: str) -> Optional[List[float]]:
        """Retrieve cached embeddings."""
        cache_key = f"doc_embeddings_{document_id}"
        file_path = self._get_cache_file_path("embeddings", cache_key).with_suffix(
            ".pkl"
        )

        if not self._is_cache_valid(file_path):
            return None

        try:
            with open(file_path, "rb") as f:
                cache_entry = pickle.load(f)
            logger.info(f"Cache hit for embeddings: {document_id}")
            return cache_entry.get("embeddings")
        except Exception as e:
            logger.warning(f"Failed to read embeddings cache: {e}")
            return None

    def cleanup_expired_cache(self) -> Dict[str, int]:
        """Remove expired cache files."""
        stats = {"removed": 0, "errors": 0, "size_freed_mb": 0}
        cutoff_time = datetime.now(UTC) - timedelta(days=self.config.max_age_days)

        for cache_type in ["sejm", "eli", "processed", "embeddings", "metadata"]:
            try:
                if cache_type == "sejm":
                    base_dir = self.config.sejm_api_dir
                elif cache_type == "eli":
                    base_dir = self.config.eli_api_dir
                elif cache_type == "processed":
                    base_dir = self.config.processed_data_dir
                elif cache_type == "embeddings":
                    base_dir = self.config.embeddings_dir
                else:
                    base_dir = self.config.metadata_dir

                for file_path in base_dir.rglob("*"):
                    if file_path.is_file():
                        try:
                            file_time = datetime.fromtimestamp(
                                file_path.stat().st_mtime, UTC
                            )
                            if file_time < cutoff_time:
                                file_size = file_path.stat().st_size
                                file_path.unlink()
                                stats["removed"] += 1
                                stats["size_freed_mb"] += file_size / (1024 * 1024)
                        except Exception as e:
                            logger.warning(
                                f"Failed to remove expired cache file {file_path}: {e}"
                            )
                            stats["errors"] += 1

            except Exception as e:
                logger.error(f"Failed to cleanup cache type {cache_type}: {e}")
                stats["errors"] += 1

        logger.info(f"Cache cleanup completed: {stats}")
        return stats

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache usage statistics."""
        stats = {
            "cache_root": str(self.config.cache_root),
            "total_files": 0,
            "total_size_mb": 0,
            "by_type": {},
        }

        cache_types = {
            "sejm": self.config.sejm_api_dir,
            "eli": self.config.eli_api_dir,
            "processed": self.config.processed_data_dir,
            "embeddings": self.config.embeddings_dir,
            "metadata": self.config.metadata_dir,
        }

        for cache_type, base_dir in cache_types.items():
            type_stats = {"files": 0, "size_mb": 0}

            try:
                for file_path in base_dir.rglob("*"):
                    if file_path.is_file():
                        file_size = file_path.stat().st_size
                        type_stats["files"] += 1
                        type_stats["size_mb"] += file_size / (1024 * 1024)

            except Exception as e:
                logger.warning(f"Failed to get stats for {cache_type}: {e}")

            stats["by_type"][cache_type] = type_stats
            stats["total_files"] += type_stats["files"]
            stats["total_size_mb"] += type_stats["size_mb"]

        return stats


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager(config: Optional[CacheConfig] = None) -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager

    if _cache_manager is None:
        _cache_manager = CacheManager(config)

    return _cache_manager
