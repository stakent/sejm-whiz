"""Document content cache for HTML/PDF files and processed text."""

import hashlib
import logging
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple
import json
import gzip

from .config import CacheConfig, get_cache_config

logger = logging.getLogger(__name__)


class DocumentContentCache:
    """Manages caching of document content including HTML, PDF, and processed text."""

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or get_cache_config()
        self.config.ensure_directories()

        # Create subdirectories for different content types
        self.html_dir = self.config.processed_data_dir / "html"
        self.pdf_dir = self.config.processed_data_dir / "pdf"
        self.text_dir = self.config.processed_data_dir / "text"
        self.metadata_dir = self.config.metadata_dir / "documents"

        # Ensure subdirectories exist
        for dir_path in [self.html_dir, self.pdf_dir, self.text_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info("Document content cache initialized")

    def _get_content_hash(self, content: Union[str, bytes]) -> str:
        """Generate hash for content."""
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def _get_cache_file_path(
        self, content_type: str, document_id: str, file_hash: str
    ) -> Path:
        """Get cache file path for document content."""
        if content_type == "html":
            base_dir = self.html_dir
            extension = ".html.gz" if self.config.compress_responses else ".html"
        elif content_type == "pdf":
            base_dir = self.pdf_dir
            extension = ".pdf"  # PDFs are already compressed
        elif content_type == "text":
            base_dir = self.text_dir
            extension = ".txt.gz" if self.config.compress_responses else ".txt"
        else:
            raise ValueError(f"Unknown content type: {content_type}")

        # Use first 2 chars of document_id for subdirectory organization
        subdir = document_id[:2] if len(document_id) >= 2 else "misc"
        cache_dir = base_dir / subdir
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Filename includes document_id and content hash for uniqueness
        filename = f"{document_id}_{file_hash[:8]}{extension}"
        return cache_dir / filename

    def _get_metadata_file_path(self, document_id: str) -> Path:
        """Get metadata file path for document."""
        subdir = document_id[:2] if len(document_id) >= 2 else "misc"
        metadata_dir = self.metadata_dir / subdir
        metadata_dir.mkdir(parents=True, exist_ok=True)
        return metadata_dir / f"{document_id}_metadata.json"

    def _write_content_file(
        self, file_path: Path, content: Union[str, bytes], content_type: str
    ) -> None:
        """Write content to cache file with appropriate handling."""
        try:
            if content_type == "pdf":
                # PDF files are binary and already compressed
                with open(file_path, "wb") as f:
                    if isinstance(content, str):
                        content = content.encode("utf-8")
                    f.write(content)
            elif self.config.compress_responses and content_type in ["html", "text"]:
                # Compress text-based content
                with gzip.open(
                    file_path,
                    "wt",
                    encoding="utf-8",
                    compresslevel=self.config.compression_level,
                ) as f:
                    if isinstance(content, bytes):
                        content = content.decode("utf-8")
                    f.write(content)
            else:
                # Write uncompressed text content
                mode = "w" if isinstance(content, str) else "wb"
                encoding = "utf-8" if isinstance(content, str) else None
                with open(file_path, mode, encoding=encoding) as f:
                    f.write(content)

            logger.debug(f"Cached {content_type} content to {file_path}")

        except Exception as e:
            logger.error(f"Failed to write content file {file_path}: {e}")
            raise

    def _read_content_file(
        self, file_path: Path, content_type: str
    ) -> Union[str, bytes]:
        """Read content from cache file with appropriate handling."""
        try:
            if content_type == "pdf":
                # PDF files are binary
                with open(file_path, "rb") as f:
                    return f.read()
            elif file_path.suffix == ".gz":
                # Compressed text content
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    return f.read()
            else:
                # Uncompressed text content
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()

        except Exception as e:
            logger.error(f"Failed to read content file {file_path}: {e}")
            raise

    def _write_metadata(self, document_id: str, metadata: Dict[str, Any]) -> None:
        """Write document metadata to cache."""
        metadata_path = self._get_metadata_file_path(document_id)

        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.debug(f"Cached metadata for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to write metadata for {document_id}: {e}")
            raise

    def _read_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Read document metadata from cache."""
        metadata_path = self._get_metadata_file_path(document_id)

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read metadata for {document_id}: {e}")
            return None

    def cache_document_content(
        self,
        document_id: str,
        content: Union[str, bytes],
        content_type: str,
        source_url: Optional[str] = None,
        mime_type: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        """
        Cache document content (HTML, PDF, or processed text).

        Returns:
            Tuple of (content_hash, cache_file_path)
        """
        # Validate content type
        valid_types = ["html", "pdf", "text"]
        if content_type not in valid_types:
            raise ValueError(
                f"Content type must be one of {valid_types}, got: {content_type}"
            )

        # Generate content hash
        content_hash = self._get_content_hash(content)

        # Get cache file path
        file_path = self._get_cache_file_path(content_type, document_id, content_hash)

        # Write content to cache
        self._write_content_file(file_path, content, content_type)

        # Prepare and write metadata
        metadata = {
            "document_id": document_id,
            "content_type": content_type,
            "content_hash": content_hash,
            "source_url": source_url,
            "mime_type": mime_type,
            "file_size": len(content) if isinstance(content, (str, bytes)) else 0,
            "cached_at": datetime.now(UTC).isoformat(),
            "cache_file_path": str(file_path),
            **(additional_metadata or {}),
        }

        # Update existing metadata or create new
        existing_metadata = self._read_metadata(document_id) or {"content_versions": {}}
        existing_metadata["content_versions"][content_type] = metadata
        existing_metadata["last_updated"] = datetime.now(UTC).isoformat()

        self._write_metadata(document_id, existing_metadata)

        logger.info(
            f"Cached {content_type} content for document {document_id}: {content_hash[:8]}..."
        )
        return content_hash, str(file_path)

    def get_cached_document_content(
        self, document_id: str, content_type: str
    ) -> Optional[Tuple[Union[str, bytes], Dict[str, Any]]]:
        """
        Retrieve cached document content.

        Returns:
            Tuple of (content, metadata) or None if not found
        """
        # Get metadata
        metadata = self._read_metadata(document_id)
        if not metadata:
            return None

        content_info = metadata.get("content_versions", {}).get(content_type)
        if not content_info:
            return None

        # Check if cache file exists
        cache_file_path = Path(content_info["cache_file_path"])
        if not cache_file_path.exists():
            logger.warning(
                f"Cache file missing for {document_id}/{content_type}: {cache_file_path}"
            )
            return None

        # Check TTL
        try:
            cached_time = datetime.fromisoformat(content_info["cached_at"])
            if (
                datetime.now(UTC) - cached_time
            ).total_seconds() > self.config.api_cache_ttl:
                logger.debug(f"Cache expired for {document_id}/{content_type}")
                return None
        except Exception as e:
            logger.warning(f"Failed to parse cache time for {document_id}: {e}")
            return None

        # Read content
        try:
            content = self._read_content_file(cache_file_path, content_type)
            logger.info(f"Cache hit for {content_type} content: {document_id}")
            return content, content_info
        except Exception as e:
            logger.error(
                f"Failed to read cached content for {document_id}/{content_type}: {e}"
            )
            return None

    def cache_html_document(
        self, document_id: str, html_content: str, source_url: Optional[str] = None
    ) -> Tuple[str, str]:
        """Cache HTML document content."""
        return self.cache_document_content(
            document_id=document_id,
            content=html_content,
            content_type="html",
            source_url=source_url,
            mime_type="text/html",
        )

    def cache_pdf_document(
        self, document_id: str, pdf_content: bytes, source_url: Optional[str] = None
    ) -> Tuple[str, str]:
        """Cache PDF document content."""
        return self.cache_document_content(
            document_id=document_id,
            content=pdf_content,
            content_type="pdf",
            source_url=source_url,
            mime_type="application/pdf",
        )

    def cache_processed_text(
        self,
        document_id: str,
        processed_text: str,
        processing_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        """Cache processed/extracted text from documents."""
        return self.cache_document_content(
            document_id=document_id,
            content=processed_text,
            content_type="text",
            mime_type="text/plain",
            additional_metadata={"processing_info": processing_info or {}},
        )

    def get_document_cache_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive cache information for a document."""
        metadata = self._read_metadata(document_id)
        if not metadata:
            return None

        # Add file existence status
        for content_type, info in metadata.get("content_versions", {}).items():
            cache_path = Path(info.get("cache_file_path", ""))
            info["file_exists"] = cache_path.exists()
            if cache_path.exists():
                try:
                    stat = cache_path.stat()
                    info["file_size_on_disk"] = stat.st_size
                    info["file_modified"] = datetime.fromtimestamp(
                        stat.st_mtime, UTC
                    ).isoformat()
                except Exception:
                    pass

        return metadata

    def invalidate_document_cache(
        self, document_id: str, content_type: Optional[str] = None
    ) -> bool:
        """
        Invalidate cached content for a document.

        Args:
            document_id: Document identifier
            content_type: Specific content type to invalidate, or None for all types

        Returns:
            True if any files were removed
        """
        metadata = self._read_metadata(document_id)
        if not metadata:
            return False

        removed = False
        content_versions = metadata.get("content_versions", {})

        if content_type:
            # Invalidate specific content type
            if content_type in content_versions:
                cache_path = Path(content_versions[content_type]["cache_file_path"])
                if cache_path.exists():
                    try:
                        cache_path.unlink()
                        logger.info(
                            f"Removed cached {content_type} for document {document_id}"
                        )
                        removed = True
                    except Exception as e:
                        logger.error(f"Failed to remove cache file {cache_path}: {e}")

                # Remove from metadata
                del content_versions[content_type]
        else:
            # Invalidate all content types
            for ct, info in list(content_versions.items()):
                cache_path = Path(info["cache_file_path"])
                if cache_path.exists():
                    try:
                        cache_path.unlink()
                        logger.info(f"Removed cached {ct} for document {document_id}")
                        removed = True
                    except Exception as e:
                        logger.error(f"Failed to remove cache file {cache_path}: {e}")

            content_versions.clear()

        # Update metadata
        if content_versions:
            metadata["content_versions"] = content_versions
            metadata["last_updated"] = datetime.now(UTC).isoformat()
            self._write_metadata(document_id, metadata)
        else:
            # Remove metadata file if no content versions left
            metadata_path = self._get_metadata_file_path(document_id)
            if metadata_path.exists():
                try:
                    metadata_path.unlink()
                    logger.info(f"Removed metadata for document {document_id}")
                except Exception as e:
                    logger.error(f"Failed to remove metadata file {metadata_path}: {e}")

        return removed


# Global document cache instance
_document_cache: Optional[DocumentContentCache] = None


def get_document_cache(config: Optional[CacheConfig] = None) -> DocumentContentCache:
    """Get global document content cache instance."""
    global _document_cache

    if _document_cache is None:
        _document_cache = DocumentContentCache(config)

    return _document_cache
