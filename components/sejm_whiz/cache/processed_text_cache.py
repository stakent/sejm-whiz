"""Specialized cache for processed text with versioning and pipeline stage tracking."""

import hashlib
import json
import logging
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import gzip

from .config import CacheConfig, get_cache_config

logger = logging.getLogger(__name__)


class ProcessedTextCache:
    """
    Advanced cache for processed text with pipeline stage tracking and versioning.

    This cache supports:
    - Multiple processing stages (raw extraction, cleaning, normalization, etc.)
    - Text versioning based on processing parameters
    - Fast lookup by document ID and processing stage
    - Metadata tracking for processing pipeline
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or get_cache_config()
        self.config.ensure_directories()

        # Create subdirectories for processed text
        self.processed_text_dir = self.config.processed_data_dir / "processed_text"
        self.processing_metadata_dir = self.config.metadata_dir / "processing"

        # Ensure subdirectories exist
        for dir_path in [self.processed_text_dir, self.processing_metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info("Processed text cache initialized")

    def _get_processing_hash(self, processing_params: Dict[str, Any]) -> str:
        """Generate hash from processing parameters."""
        # Create deterministic hash from processing parameters
        params_str = json.dumps(processing_params, sort_keys=True)
        return hashlib.sha256(params_str.encode()).hexdigest()[:16]  # Use shorter hash

    def _get_cache_file_path(
        self, document_id: str, stage: str, processing_hash: str
    ) -> Path:
        """Get cache file path for processed text."""
        # Organization: processed_text/{stage}/{doc_id_prefix}/{document_id}_{processing_hash}.txt.gz

        # Use first 2 chars of document_id for subdirectory organization
        doc_prefix = document_id[:2] if len(document_id) >= 2 else "misc"

        stage_dir = self.processed_text_dir / stage / doc_prefix
        stage_dir.mkdir(parents=True, exist_ok=True)

        extension = ".txt.gz" if self.config.compress_responses else ".txt"
        filename = f"{document_id}_{processing_hash}{extension}"

        return stage_dir / filename

    def _get_metadata_file_path(self, document_id: str) -> Path:
        """Get metadata file path for document processing history."""
        doc_prefix = document_id[:2] if len(document_id) >= 2 else "misc"
        metadata_dir = self.processing_metadata_dir / doc_prefix
        metadata_dir.mkdir(parents=True, exist_ok=True)
        return metadata_dir / f"{document_id}_processing.json"

    def _write_text_file(self, file_path: Path, text: str) -> None:
        """Write text to cache file with optional compression."""
        try:
            if self.config.compress_responses:
                with gzip.open(
                    file_path,
                    "wt",
                    encoding="utf-8",
                    compresslevel=self.config.compression_level,
                ) as f:
                    f.write(text)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)

            logger.debug(f"Cached processed text to {file_path}")

        except Exception as e:
            logger.error(f"Failed to write text file {file_path}: {e}")
            raise

    def _read_text_file(self, file_path: Path) -> str:
        """Read text from cache file with optional decompression."""
        try:
            if file_path.suffix == ".gz":
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    return f.read()
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()

        except Exception as e:
            logger.error(f"Failed to read text file {file_path}: {e}")
            raise

    def _update_processing_metadata(
        self,
        document_id: str,
        stage: str,
        processing_hash: str,
        processing_params: Dict[str, Any],
        cache_file_path: Path,
        text_length: int,
        processing_time_ms: Optional[float] = None,
    ) -> None:
        """Update processing metadata for document."""
        metadata_path = self._get_metadata_file_path(document_id)

        # Load existing metadata or create new
        metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(
                    f"Failed to read existing metadata for {document_id}: {e}"
                )

        # Initialize structure
        if "processing_stages" not in metadata:
            metadata["processing_stages"] = {}
        if "document_id" not in metadata:
            metadata["document_id"] = document_id

        # Update stage information
        stage_info = {
            "processing_hash": processing_hash,
            "processing_params": processing_params,
            "cache_file_path": str(cache_file_path),
            "text_length": text_length,
            "processing_time_ms": processing_time_ms,
            "cached_at": datetime.now(UTC).isoformat(),
            "file_size": cache_file_path.stat().st_size
            if cache_file_path.exists()
            else 0,
        }

        metadata["processing_stages"][stage] = stage_info
        metadata["last_updated"] = datetime.now(UTC).isoformat()

        # Write metadata
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.debug(f"Updated processing metadata for {document_id}/{stage}")
        except Exception as e:
            logger.error(f"Failed to write processing metadata for {document_id}: {e}")
            raise

    def cache_processed_text(
        self,
        document_id: str,
        stage: str,
        processed_text: str,
        processing_params: Dict[str, Any],
        processing_time_ms: Optional[float] = None,
    ) -> Tuple[str, str]:
        """
        Cache processed text for a specific processing stage.

        Args:
            document_id: Unique document identifier
            stage: Processing stage name (e.g., 'raw_extraction', 'cleaned', 'normalized')
            processed_text: The processed text content
            processing_params: Parameters used for processing (for versioning)
            processing_time_ms: Time taken for processing (for performance tracking)

        Returns:
            Tuple of (processing_hash, cache_file_path)
        """
        # Validate inputs
        if not document_id or not stage or not processed_text:
            raise ValueError("document_id, stage, and processed_text are required")

        # Generate processing hash from parameters
        processing_hash = self._get_processing_hash(processing_params)

        # Get cache file path
        cache_file_path = self._get_cache_file_path(document_id, stage, processing_hash)

        # Write processed text
        self._write_text_file(cache_file_path, processed_text)

        # Update metadata
        self._update_processing_metadata(
            document_id=document_id,
            stage=stage,
            processing_hash=processing_hash,
            processing_params=processing_params,
            cache_file_path=cache_file_path,
            text_length=len(processed_text),
            processing_time_ms=processing_time_ms,
        )

        logger.info(
            f"Cached processed text for {document_id}/{stage}: "
            f"{len(processed_text)} chars, hash {processing_hash}"
        )

        return processing_hash, str(cache_file_path)

    def get_processed_text(
        self,
        document_id: str,
        stage: str,
        processing_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Retrieve cached processed text.

        Args:
            document_id: Document identifier
            stage: Processing stage name
            processing_params: If provided, only return text matching these params

        Returns:
            Tuple of (text, metadata) or None if not found/expired
        """
        # Get processing metadata
        metadata_path = self._get_metadata_file_path(document_id)
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read processing metadata for {document_id}: {e}")
            return None

        stage_info = metadata.get("processing_stages", {}).get(stage)
        if not stage_info:
            return None

        # If processing_params provided, check if they match
        if processing_params is not None:
            expected_hash = self._get_processing_hash(processing_params)
            if stage_info.get("processing_hash") != expected_hash:
                logger.debug(
                    f"Processing parameters don't match cached version for {document_id}/{stage}"
                )
                return None

        # Check if cache file exists
        cache_file_path = Path(stage_info["cache_file_path"])
        if not cache_file_path.exists():
            logger.warning(
                f"Cache file missing for {document_id}/{stage}: {cache_file_path}"
            )
            return None

        # Check TTL
        try:
            cached_time = datetime.fromisoformat(stage_info["cached_at"])
            if (
                datetime.now(UTC) - cached_time
            ).total_seconds() > self.config.api_cache_ttl:
                logger.debug(f"Cache expired for {document_id}/{stage}")
                return None
        except Exception as e:
            logger.warning(f"Failed to parse cache time for {document_id}/{stage}: {e}")
            return None

        # Read processed text
        try:
            text = self._read_text_file(cache_file_path)
            logger.info(f"Cache hit for processed text: {document_id}/{stage}")
            return text, stage_info
        except Exception as e:
            logger.error(f"Failed to read cached text for {document_id}/{stage}: {e}")
            return None

    def get_processing_history(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get complete processing history for a document."""
        metadata_path = self._get_metadata_file_path(document_id)
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # Add file existence status for each stage
            for stage, info in metadata.get("processing_stages", {}).items():
                cache_path = Path(info.get("cache_file_path", ""))
                info["file_exists"] = cache_path.exists()
                if cache_path.exists():
                    try:
                        stat = cache_path.stat()
                        info["actual_file_size"] = stat.st_size
                        info["file_modified"] = datetime.fromtimestamp(
                            stat.st_mtime, UTC
                        ).isoformat()
                    except Exception:
                        pass

            return metadata

        except Exception as e:
            logger.error(f"Failed to read processing history for {document_id}: {e}")
            return None

    def get_available_stages(self, document_id: str) -> List[str]:
        """Get list of available processing stages for a document."""
        history = self.get_processing_history(document_id)
        if not history:
            return []

        return list(history.get("processing_stages", {}).keys())

    def invalidate_stage(self, document_id: str, stage: str) -> bool:
        """Invalidate cached text for a specific processing stage."""
        metadata_path = self._get_metadata_file_path(document_id)
        if not metadata_path.exists():
            return False

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read metadata for invalidation: {e}")
            return False

        stage_info = metadata.get("processing_stages", {}).get(stage)
        if not stage_info:
            return False

        # Remove cache file
        cache_path = Path(stage_info["cache_file_path"])
        removed = False
        if cache_path.exists():
            try:
                cache_path.unlink()
                logger.info(f"Removed cached text for {document_id}/{stage}")
                removed = True
            except Exception as e:
                logger.error(f"Failed to remove cache file {cache_path}: {e}")

        # Update metadata
        del metadata["processing_stages"][stage]
        metadata["last_updated"] = datetime.now(UTC).isoformat()

        # Write updated metadata or remove file if no stages left
        if metadata["processing_stages"]:
            try:
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Failed to update metadata after invalidation: {e}")
        else:
            # Remove metadata file if no stages left
            try:
                metadata_path.unlink()
                logger.info(f"Removed processing metadata for {document_id}")
            except Exception as e:
                logger.error(f"Failed to remove metadata file: {e}")

        return removed

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache usage statistics by processing stage."""
        stats = {
            "total_documents": 0,
            "total_stages": 0,
            "total_size_mb": 0,
            "by_stage": {},
            "cache_root": str(self.processed_text_dir),
        }

        try:
            # Count metadata files to get document count
            metadata_files = list(
                self.processing_metadata_dir.rglob("*_processing.json")
            )
            stats["total_documents"] = len(metadata_files)

            # Analyze each stage directory
            for stage_dir in self.processed_text_dir.iterdir():
                if stage_dir.is_dir():
                    stage_name = stage_dir.name
                    stage_stats = {"files": 0, "size_mb": 0}

                    for file_path in stage_dir.rglob("*.txt*"):
                        if file_path.is_file():
                            stage_stats["files"] += 1
                            stage_stats["size_mb"] += file_path.stat().st_size / (
                                1024 * 1024
                            )

                    stats["by_stage"][stage_name] = stage_stats
                    stats["total_stages"] += stage_stats["files"]
                    stats["total_size_mb"] += stage_stats["size_mb"]

        except Exception as e:
            logger.error(f"Failed to calculate cache statistics: {e}")

        return stats


# Global processed text cache instance
_processed_text_cache: Optional[ProcessedTextCache] = None


def get_processed_text_cache(
    config: Optional[CacheConfig] = None,
) -> ProcessedTextCache:
    """Get global processed text cache instance."""
    global _processed_text_cache

    if _processed_text_cache is None:
        _processed_text_cache = ProcessedTextCache(config)

    return _processed_text_cache
