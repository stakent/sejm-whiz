"""Cache management and cleanup utilities."""

import logging
from datetime import datetime, UTC, timedelta
from typing import Any, Dict, List, Optional, Set
import json

from .config import CacheConfig, get_cache_config
from .manager import get_cache_manager
from .document_cache import get_document_cache
from .processed_text_cache import get_processed_text_cache

logger = logging.getLogger(__name__)


class CacheMaintenanceManager:
    """Manages cache maintenance operations including cleanup, optimization, and reporting."""

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or get_cache_config()
        self.cache_manager = get_cache_manager(config)
        self.document_cache = get_document_cache(config)
        self.text_cache = get_processed_text_cache(config)

        logger.info("Cache maintenance manager initialized")

    async def run_full_maintenance(self, dry_run: bool = False) -> Dict[str, Any]:
        """Run comprehensive cache maintenance."""
        logger.info(f"Starting full cache maintenance (dry_run={dry_run})")

        results = {
            "maintenance_started_at": datetime.now(UTC).isoformat(),
            "dry_run": dry_run,
            "operations": {},
        }

        try:
            # 1. Clean up expired cache files
            logger.info("Cleaning up expired cache files")
            cleanup_result = await self.cleanup_expired_files(dry_run)
            results["operations"]["expired_cleanup"] = cleanup_result

            # 2. Remove orphaned files
            logger.info("Removing orphaned cache files")
            orphan_result = await self.remove_orphaned_files(dry_run)
            results["operations"]["orphan_cleanup"] = orphan_result

            # 3. Optimize cache structure
            logger.info("Optimizing cache structure")
            optimize_result = await self.optimize_cache_structure(dry_run)
            results["operations"]["optimization"] = optimize_result

            # 4. Generate maintenance report
            logger.info("Generating maintenance report")
            report_result = await self.generate_maintenance_report()
            results["operations"]["reporting"] = report_result

            results["maintenance_completed_at"] = datetime.now(UTC).isoformat()
            results["success"] = True

            logger.info(f"Cache maintenance completed successfully: {results}")

        except Exception as e:
            logger.error(f"Cache maintenance failed: {e}")
            results["error"] = str(e)
            results["success"] = False
            results["maintenance_failed_at"] = datetime.now(UTC).isoformat()

        return results

    async def cleanup_expired_files(self, dry_run: bool = False) -> Dict[str, Any]:
        """Clean up expired cache files across all cache types."""
        result = {
            "files_processed": 0,
            "files_removed": 0,
            "size_freed_mb": 0.0,
            "errors": 0,
            "by_cache_type": {},
        }

        cutoff_time = datetime.now(UTC) - timedelta(days=self.config.max_age_days)

        # Clean up different cache types
        cache_types = [
            ("api_responses", self.config.api_responses_dir),
            ("processed_data", self.config.processed_data_dir),
            ("embeddings", self.config.embeddings_dir),
            ("metadata", self.config.metadata_dir),
        ]

        for cache_type, base_dir in cache_types:
            type_result = {
                "files_processed": 0,
                "files_removed": 0,
                "size_freed_mb": 0.0,
                "errors": 0,
            }

            try:
                if base_dir.exists():
                    for file_path in base_dir.rglob("*"):
                        if file_path.is_file():
                            type_result["files_processed"] += 1

                            try:
                                file_time = datetime.fromtimestamp(
                                    file_path.stat().st_mtime, UTC
                                )

                                if file_time < cutoff_time:
                                    file_size = file_path.stat().st_size

                                    if not dry_run:
                                        file_path.unlink()
                                        logger.debug(
                                            f"Removed expired file: {file_path}"
                                        )
                                    else:
                                        logger.debug(
                                            f"Would remove expired file: {file_path}"
                                        )

                                    type_result["files_removed"] += 1
                                    type_result["size_freed_mb"] += file_size / (
                                        1024 * 1024
                                    )

                            except Exception as e:
                                logger.warning(
                                    f"Failed to process file {file_path}: {e}"
                                )
                                type_result["errors"] += 1

            except Exception as e:
                logger.error(f"Failed to cleanup cache type {cache_type}: {e}")
                type_result["errors"] += 1

            result["by_cache_type"][cache_type] = type_result
            result["files_processed"] += type_result["files_processed"]
            result["files_removed"] += type_result["files_removed"]
            result["size_freed_mb"] += type_result["size_freed_mb"]
            result["errors"] += type_result["errors"]

        return result

    async def remove_orphaned_files(self, dry_run: bool = False) -> Dict[str, Any]:
        """Remove orphaned cache files that have no corresponding metadata."""
        result = {"orphaned_files": 0, "orphaned_size_mb": 0.0, "errors": 0}

        try:
            # Get all metadata files to build a set of valid document IDs
            valid_document_ids: Set[str] = set()

            # Collect document IDs from document cache metadata
            doc_metadata_dir = self.config.metadata_dir / "documents"
            if doc_metadata_dir.exists():
                for metadata_file in doc_metadata_dir.rglob("*_metadata.json"):
                    try:
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                        valid_document_ids.add(metadata.get("document_id", ""))
                    except Exception as e:
                        logger.warning(
                            f"Failed to read metadata file {metadata_file}: {e}"
                        )

            # Collect document IDs from processed text metadata
            text_metadata_dir = self.config.metadata_dir / "processing"
            if text_metadata_dir.exists():
                for metadata_file in text_metadata_dir.rglob("*_processing.json"):
                    try:
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                        valid_document_ids.add(metadata.get("document_id", ""))
                    except Exception as e:
                        logger.warning(
                            f"Failed to read processing metadata file {metadata_file}: {e}"
                        )

            # Check for orphaned files in processed data directory
            processed_data_dirs = [
                self.config.processed_data_dir / "html",
                self.config.processed_data_dir / "pdf",
                self.config.processed_data_dir / "text",
                self.config.processed_data_dir / "processed_text",
            ]

            for data_dir in processed_data_dirs:
                if data_dir.exists():
                    for file_path in data_dir.rglob("*"):
                        if file_path.is_file():
                            try:
                                # Extract document ID from filename
                                filename = file_path.stem
                                if filename.endswith(".txt") or filename.endswith(
                                    ".html"
                                ):
                                    filename = filename.rsplit(".", 1)[
                                        0
                                    ]  # Remove .txt/.html

                                # Get document ID (before first underscore or hash)
                                doc_id = filename.split("_")[0]

                                if doc_id not in valid_document_ids:
                                    file_size = file_path.stat().st_size

                                    if not dry_run:
                                        file_path.unlink()
                                        logger.debug(
                                            f"Removed orphaned file: {file_path}"
                                        )
                                    else:
                                        logger.debug(
                                            f"Would remove orphaned file: {file_path}"
                                        )

                                    result["orphaned_files"] += 1
                                    result["orphaned_size_mb"] += file_size / (
                                        1024 * 1024
                                    )

                            except Exception as e:
                                logger.warning(
                                    f"Failed to process potential orphan {file_path}: {e}"
                                )
                                result["errors"] += 1

        except Exception as e:
            logger.error(f"Failed to remove orphaned files: {e}")
            result["errors"] += 1

        return result

    async def optimize_cache_structure(self, dry_run: bool = False) -> Dict[str, Any]:
        """Optimize cache directory structure and remove empty directories."""
        result = {"empty_dirs_removed": 0, "dirs_reorganized": 0, "errors": 0}

        try:
            # Remove empty directories throughout the cache hierarchy
            cache_roots = [
                self.config.api_responses_dir,
                self.config.processed_data_dir,
                self.config.embeddings_dir,
                self.config.metadata_dir,
            ]

            for root_dir in cache_roots:
                if root_dir.exists():
                    # Get all directories in reverse order (deepest first)
                    all_dirs = sorted(
                        [d for d in root_dir.rglob("*") if d.is_dir()],
                        key=lambda x: len(str(x)),
                        reverse=True,
                    )

                    for dir_path in all_dirs:
                        try:
                            # Check if directory is empty
                            if not any(dir_path.iterdir()):
                                if not dry_run:
                                    dir_path.rmdir()
                                    logger.debug(f"Removed empty directory: {dir_path}")
                                else:
                                    logger.debug(
                                        f"Would remove empty directory: {dir_path}"
                                    )

                                result["empty_dirs_removed"] += 1

                        except Exception as e:
                            logger.warning(
                                f"Failed to remove empty directory {dir_path}: {e}"
                            )
                            result["errors"] += 1

        except Exception as e:
            logger.error(f"Failed to optimize cache structure: {e}")
            result["errors"] += 1

        return result

    async def generate_maintenance_report(self) -> Dict[str, Any]:
        """Generate comprehensive cache maintenance report."""
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "cache_statistics": {},
            "health_metrics": {},
            "recommendations": [],
        }

        try:
            # Get cache statistics from all components
            api_stats = self.cache_manager.get_cache_stats()
            text_stats = self.text_cache.get_cache_statistics()

            report["cache_statistics"] = {
                "api_cache": api_stats,
                "processed_text_cache": text_stats,
                "total_size_mb": api_stats.get("total_size_mb", 0)
                + text_stats.get("total_size_mb", 0),
                "total_files": api_stats.get("total_files", 0)
                + text_stats.get("total_files", 0),
            }

            # Calculate health metrics
            total_size_mb = report["cache_statistics"]["total_size_mb"]
            max_size_mb = self.config.max_cache_size_mb

            report["health_metrics"] = {
                "cache_utilization_percent": (total_size_mb / max_size_mb * 100)
                if max_size_mb > 0
                else 0,
                "size_limit_mb": max_size_mb,
                "current_size_mb": total_size_mb,
                "available_space_mb": max(0, max_size_mb - total_size_mb),
            }

            # Generate recommendations
            utilization = report["health_metrics"]["cache_utilization_percent"]

            if utilization > 90:
                report["recommendations"].append(
                    {
                        "priority": "high",
                        "category": "storage",
                        "message": "Cache utilization is very high (>90%). Consider increasing cache size limit or running cleanup.",
                    }
                )
            elif utilization > 75:
                report["recommendations"].append(
                    {
                        "priority": "medium",
                        "category": "storage",
                        "message": "Cache utilization is high (>75%). Monitor storage usage closely.",
                    }
                )

            if api_stats.get("total_files", 0) > 10000:
                report["recommendations"].append(
                    {
                        "priority": "medium",
                        "category": "performance",
                        "message": "Large number of cache files detected. Consider optimization for better performance.",
                    }
                )

        except Exception as e:
            logger.error(f"Failed to generate maintenance report: {e}")
            report["error"] = str(e)

        return report

    def get_cache_health_status(self) -> Dict[str, Any]:
        """Get current cache health status."""
        try:
            # Get basic statistics
            api_stats = self.cache_manager.get_cache_stats()
            text_stats = self.text_cache.get_cache_statistics()

            total_size_mb = api_stats.get("total_size_mb", 0) + text_stats.get(
                "total_size_mb", 0
            )
            total_files = api_stats.get("total_files", 0) + text_stats.get(
                "total_files", 0
            )

            # Calculate health score (0-100)
            utilization = (
                (total_size_mb / self.config.max_cache_size_mb * 100)
                if self.config.max_cache_size_mb > 0
                else 0
            )

            if utilization <= 50:
                health_score = 100
                status = "excellent"
            elif utilization <= 75:
                health_score = 80
                status = "good"
            elif utilization <= 90:
                health_score = 60
                status = "warning"
            else:
                health_score = 30
                status = "critical"

            return {
                "status": status,
                "health_score": health_score,
                "utilization_percent": utilization,
                "total_size_mb": total_size_mb,
                "total_files": total_files,
                "cache_limit_mb": self.config.max_cache_size_mb,
                "last_checked": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get cache health status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "last_checked": datetime.now(UTC).isoformat(),
            }

    async def invalidate_document_caches(
        self, document_ids: List[str], dry_run: bool = False
    ) -> Dict[str, Any]:
        """Invalidate all caches for specific documents."""
        result = {
            "documents_processed": len(document_ids),
            "caches_invalidated": 0,
            "errors": 0,
            "details": {},
        }

        for document_id in document_ids:
            doc_result = {"document_cache": False, "text_cache": False, "errors": []}

            try:
                # Invalidate document cache
                if not dry_run:
                    if self.document_cache.invalidate_document_cache(document_id):
                        doc_result["document_cache"] = True
                        result["caches_invalidated"] += 1
                else:
                    # Check if document cache exists
                    if self.document_cache.get_document_cache_info(document_id):
                        doc_result["document_cache"] = True

                # Invalidate processed text cache stages
                available_stages = self.text_cache.get_available_stages(document_id)
                if available_stages:
                    for stage in available_stages:
                        if not dry_run:
                            if self.text_cache.invalidate_stage(document_id, stage):
                                doc_result["text_cache"] = True
                                result["caches_invalidated"] += 1
                        else:
                            doc_result["text_cache"] = True

            except Exception as e:
                logger.error(
                    f"Failed to invalidate caches for document {document_id}: {e}"
                )
                doc_result["errors"].append(str(e))
                result["errors"] += 1

            result["details"][document_id] = doc_result

        return result


# Global cache maintenance manager
_maintenance_manager: Optional[CacheMaintenanceManager] = None


def get_maintenance_manager(
    config: Optional[CacheConfig] = None,
) -> CacheMaintenanceManager:
    """Get global cache maintenance manager instance."""
    global _maintenance_manager

    if _maintenance_manager is None:
        _maintenance_manager = CacheMaintenanceManager(config)

    return _maintenance_manager
