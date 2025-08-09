"""Simple retry queue for failed document processing."""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class RetryItem:
    """Retry queue item with metadata."""

    document_id: str
    failure_reason: str
    retry_count: int
    queued_at: str
    last_attempt: Optional[str] = None
    max_retries: int = 3
    source_hint: Optional[str] = None  # "sejm", "eli", or None for auto

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetryItem":
        """Create RetryItem from dictionary."""
        return cls(**data)

    def should_retry(self) -> bool:
        """Check if item should be retried based on retry count."""
        return self.retry_count < self.max_retries

    def is_expired(self, max_age_hours: int = 24) -> bool:
        """Check if retry item has expired."""
        try:
            queued_time = datetime.fromisoformat(self.queued_at)
            age = datetime.now() - queued_time
            return age.total_seconds() > (max_age_hours * 3600)
        except Exception:
            # If we can't parse the date, consider it expired
            return True


class SimpleRetryQueue:
    """Simple retry queue for failed document processing."""

    def __init__(self, redis_client, max_queue_size: int = 1000):
        """Initialize retry queue.

        Args:
            redis_client: Redis client instance
            max_queue_size: Maximum number of items to keep in queue
        """
        self.redis = redis_client
        self.retry_key = "document_retry_queue"
        self.failed_key = "document_failed_permanent"  # For permanently failed items
        self.stats_key = "retry_queue_stats"
        self.max_queue_size = max_queue_size

        logger.info(
            f"SimpleRetryQueue initialized with max_queue_size={max_queue_size}"
        )

    async def add_for_retry(
        self,
        document_id: str,
        failure_reason: str,
        source_hint: Optional[str] = None,
        max_retries: int = 3,
    ) -> bool:
        """Add document to retry queue.

        Args:
            document_id: Document ID to retry
            failure_reason: Reason for failure
            source_hint: Hint for which API to try ("sejm", "eli", or None)
            max_retries: Maximum retry attempts

        Returns:
            True if added successfully, False if queue is full
        """
        try:
            # Check queue size
            queue_size = await self.redis.llen(self.retry_key)
            if queue_size >= self.max_queue_size:
                logger.warning(
                    f"Retry queue full ({queue_size}/{self.max_queue_size}), cannot add {document_id}"
                )
                return False

            # Check if document is already in queue
            existing_items = await self._get_all_retry_items()
            for item in existing_items:
                if item.document_id == document_id:
                    logger.info(
                        f"Document {document_id} already in retry queue, updating retry count"
                    )
                    # Remove the old item and add updated one
                    await self._remove_retry_item(item)
                    item.retry_count += 1
                    item.last_attempt = datetime.now().isoformat()
                    item.failure_reason = failure_reason  # Update with latest reason

                    if item.should_retry():
                        await self._add_retry_item(item)
                        logger.info(
                            f"Updated retry item for {document_id} (attempt {item.retry_count}/{item.max_retries})"
                        )
                        return True
                    else:
                        await self._move_to_permanent_failed(item)
                        logger.info(
                            f"Document {document_id} exceeded max retries, moved to permanent failed"
                        )
                        return False

            # Create new retry item
            retry_item = RetryItem(
                document_id=document_id,
                failure_reason=failure_reason,
                retry_count=0,
                queued_at=datetime.now().isoformat(),
                max_retries=max_retries,
                source_hint=source_hint,
            )

            await self._add_retry_item(retry_item)
            await self._update_stats("items_added", 1)

            logger.info(
                f"Added {document_id} to retry queue (reason: {failure_reason})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to add {document_id} to retry queue: {e}")
            return False

    async def get_next_retry_batch(self, limit: int = 10) -> List[RetryItem]:
        """Get next batch of documents to retry.

        Args:
            limit: Maximum number of items to return

        Returns:
            List of RetryItem objects ready for retry
        """
        try:
            batch = []

            # Get items from queue (non-destructive)
            raw_items = await self.redis.lrange(self.retry_key, 0, limit - 1)

            for raw_item in raw_items:
                try:
                    item_data = json.loads(raw_item)
                    retry_item = RetryItem.from_dict(item_data)

                    # Check if item should be retried and not expired
                    if retry_item.should_retry() and not retry_item.is_expired():
                        batch.append(retry_item)

                        # Remove from queue (we'll re-add if retry fails)
                        await self.redis.lrem(self.retry_key, 1, raw_item)
                    elif retry_item.is_expired():
                        # Remove expired items
                        await self.redis.lrem(self.retry_key, 1, raw_item)
                        await self._move_to_permanent_failed(retry_item)
                        logger.info(
                            f"Removed expired retry item: {retry_item.document_id}"
                        )
                    elif not retry_item.should_retry():
                        # Remove items that exceeded max retries
                        await self.redis.lrem(self.retry_key, 1, raw_item)
                        await self._move_to_permanent_failed(retry_item)
                        logger.info(
                            f"Moved max-retry exceeded item to permanent failed: {retry_item.document_id}"
                        )

                except Exception as e:
                    logger.error(f"Failed to parse retry item: {e}")
                    # Remove malformed items
                    await self.redis.lrem(self.retry_key, 1, raw_item)
                    continue

                if len(batch) >= limit:
                    break

            logger.info(f"Retrieved {len(batch)} items for retry batch")
            return batch

        except Exception as e:
            logger.error(f"Failed to get retry batch: {e}")
            return []

    async def mark_retry_success(self, document_id: str) -> bool:
        """Mark a document as successfully processed after retry.

        Args:
            document_id: Document ID that was successfully processed

        Returns:
            True if marked successfully
        """
        try:
            await self._update_stats("retries_successful", 1)
            logger.info(f"Marked retry success for {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to mark retry success for {document_id}: {e}")
            return False

    async def mark_retry_failed(self, document_id: str, failure_reason: str) -> bool:
        """Mark a retry attempt as failed and re-queue if retries remain.

        Args:
            document_id: Document ID that failed retry
            failure_reason: Reason for retry failure

        Returns:
            True if handled successfully
        """
        # This will be handled by add_for_retry which increments retry count
        return await self.add_for_retry(document_id, failure_reason)

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status and statistics.

        Returns:
            Dictionary with queue statistics
        """
        try:
            retry_count = await self.redis.llen(self.retry_key)
            failed_count = await self.redis.llen(self.failed_key)

            # Get basic stats
            stats_data = await self.redis.get(self.stats_key)
            stats = json.loads(stats_data) if stats_data else {}

            # Get queue health metrics
            items = await self._get_all_retry_items()
            expired_count = sum(1 for item in items if item.is_expired())
            max_retries_count = sum(1 for item in items if not item.should_retry())

            status = {
                "queue_size": retry_count,
                "permanent_failed_count": failed_count,
                "expired_items": expired_count,
                "max_retries_exceeded": max_retries_count,
                "max_queue_size": self.max_queue_size,
                "queue_utilization": retry_count / self.max_queue_size
                if self.max_queue_size > 0
                else 0,
                "stats": stats,
                "timestamp": datetime.now().isoformat(),
            }

            return status

        except Exception as e:
            logger.error(f"Failed to get queue status: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    async def cleanup_expired_items(self, max_age_hours: int = 24) -> Dict[str, int]:
        """Clean up expired items from the retry queue.

        Args:
            max_age_hours: Maximum age in hours before items are considered expired

        Returns:
            Dictionary with cleanup statistics
        """
        try:
            items = await self._get_all_retry_items()
            expired_items = [item for item in items if item.is_expired(max_age_hours)]

            cleanup_stats = {
                "expired_removed": 0,
                "max_retries_moved": 0,
                "total_processed": 0,
            }

            for item in expired_items:
                try:
                    # Remove from retry queue
                    await self._remove_retry_item(item)

                    if item.is_expired(max_age_hours):
                        cleanup_stats["expired_removed"] += 1
                    elif not item.should_retry():
                        await self._move_to_permanent_failed(item)
                        cleanup_stats["max_retries_moved"] += 1

                    cleanup_stats["total_processed"] += 1

                except Exception as e:
                    logger.error(f"Failed to cleanup item {item.document_id}: {e}")
                    continue

            await self._update_stats("cleanup_operations", 1)
            logger.info(f"Cleanup completed: {cleanup_stats}")

            return cleanup_stats

        except Exception as e:
            logger.error(f"Failed to cleanup expired items: {e}")
            return {"error": str(e)}

    async def _add_retry_item(self, item: RetryItem) -> None:
        """Add retry item to queue."""
        item_json = json.dumps(item.to_dict())
        await self.redis.lpush(self.retry_key, item_json)

    async def _remove_retry_item(self, item: RetryItem) -> None:
        """Remove retry item from queue."""
        item_json = json.dumps(item.to_dict())
        await self.redis.lrem(self.retry_key, 1, item_json)

    async def _get_all_retry_items(self) -> List[RetryItem]:
        """Get all retry items from queue (for internal use)."""
        raw_items = await self.redis.lrange(self.retry_key, 0, -1)
        items = []

        for raw_item in raw_items:
            try:
                item_data = json.loads(raw_item)
                item = RetryItem.from_dict(item_data)
                items.append(item)
            except Exception as e:
                logger.error(f"Failed to parse retry item: {e}")
                continue

        return items

    async def _move_to_permanent_failed(self, item: RetryItem) -> None:
        """Move item to permanent failed list."""
        failed_data = {
            **item.to_dict(),
            "moved_to_failed_at": datetime.now().isoformat(),
            "final_failure_reason": item.failure_reason,
        }
        failed_json = json.dumps(failed_data)
        await self.redis.lpush(self.failed_key, failed_json)
        await self._update_stats("items_permanently_failed", 1)

    async def _update_stats(self, stat_name: str, increment: int = 1) -> None:
        """Update queue statistics."""
        try:
            stats_data = await self.redis.get(self.stats_key)
            stats = json.loads(stats_data) if stats_data else {}

            stats[stat_name] = stats.get(stat_name, 0) + increment
            stats["last_updated"] = datetime.now().isoformat()

            await self.redis.set(self.stats_key, json.dumps(stats))
        except Exception as e:
            logger.error(f"Failed to update stats: {e}")
