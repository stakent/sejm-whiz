"""Redis-based job queue for background processing tasks."""

import json
import uuid
import time
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from .connection import get_redis_client
from .config import get_redis_config, RedisConfig

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Job:
    """Job definition for queue processing."""
    id: str
    task_name: str
    args: List[Any]
    kwargs: Dict[str, Any]
    priority: JobPriority
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, (JobStatus, JobPriority)):
                data[key] = value.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        """Create job from dictionary."""
        # Convert ISO strings back to datetime objects
        for key in ["created_at", "started_at", "completed_at"]:
            if data.get(key):
                data[key] = datetime.fromisoformat(data[key])
        
        # Convert enum values
        if "status" in data:
            data["status"] = JobStatus(data["status"])
        if "priority" in data:
            data["priority"] = JobPriority(data["priority"])
        
        return cls(**data)


class RedisJobQueue:
    """Redis-based job queue with priority support and result storage."""
    
    def __init__(self, config: Optional[RedisConfig] = None):
        self.config = config or get_redis_config()
        self._client = get_redis_client(config)
        
        # Queue names by priority
        self.queue_names = {
            JobPriority.CRITICAL: f"{self.config.job_queue_name}:critical",
            JobPriority.HIGH: f"{self.config.job_queue_name}:high", 
            JobPriority.NORMAL: f"{self.config.job_queue_name}:normal",
            JobPriority.LOW: f"{self.config.job_queue_name}:low"
        }
        
        # Keys for job storage
        self.job_key_prefix = f"{self.config.job_queue_name}:job:"
        self.result_key_prefix = f"{self.config.job_queue_name}:result:"
        self.worker_key_prefix = f"{self.config.job_queue_name}:worker:"
    
    def enqueue(self, 
                task_name: str,
                args: List[Any] = None,
                kwargs: Dict[str, Any] = None,
                priority: JobPriority = JobPriority.NORMAL,
                max_retries: int = 3) -> str:
        """Enqueue a job for processing."""
        
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            task_name=task_name,
            args=args or [],
            kwargs=kwargs or {},
            priority=priority,
            status=JobStatus.PENDING,
            created_at=datetime.utcnow(),
            max_retries=max_retries
        )
        
        try:
            # Store job data
            job_key = f"{self.job_key_prefix}{job_id}"
            self._client.setex(job_key, self.config.result_ttl, json.dumps(job.to_dict()))
            
            # Add to priority queue
            queue_name = self.queue_names[priority]
            self._client.lpush(queue_name, job_id)
            
            logger.info(f"Enqueued job {job_id} with task '{task_name}' and priority {priority.value}")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to enqueue job: {e}")
            raise
    
    def dequeue(self, timeout: int = 10) -> Optional[Job]:
        """Dequeue next job by priority (blocking operation)."""
        try:
            # Check queues in priority order
            queue_names = [
                self.queue_names[JobPriority.CRITICAL],
                self.queue_names[JobPriority.HIGH],
                self.queue_names[JobPriority.NORMAL],
                self.queue_names[JobPriority.LOW]
            ]
            
            # Blocking right pop from multiple queues
            result = self._client.brpop(queue_names, timeout)
            if not result:
                return None
            
            queue_name, job_id = result
            
            # Get job data
            job_data = self.get_job(job_id)
            if not job_data:
                logger.warning(f"Job {job_id} not found in storage")
                return None
            
            # Update job status
            job_data.status = JobStatus.RUNNING
            job_data.started_at = datetime.utcnow()
            self._update_job(job_data)
            
            logger.info(f"Dequeued job {job_id} from queue {queue_name}")
            return job_data
            
        except Exception as e:
            logger.error(f"Failed to dequeue job: {e}")
            return None
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        try:
            job_key = f"{self.job_key_prefix}{job_id}"
            job_data = self._client.get(job_key)
            
            if not job_data:
                return None
            
            return Job.from_dict(json.loads(job_data))
            
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            return None
    
    def _update_job(self, job: Job) -> bool:
        """Update job data in storage."""
        try:
            job_key = f"{self.job_key_prefix}{job.id}"
            self._client.setex(job_key, self.config.result_ttl, json.dumps(job.to_dict()))
            return True
        except Exception as e:
            logger.error(f"Failed to update job {job.id}: {e}")
            return False
    
    def complete_job(self, job_id: str, result: Any = None) -> bool:
        """Mark job as completed and store result."""
        try:
            job = self.get_job(job_id)
            if not job:
                logger.warning(f"Job {job_id} not found")
                return False
            
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.result = result
            
            # Update job data
            self._update_job(job)
            
            # Store result separately for easy access
            if result is not None:
                result_key = f"{self.result_key_prefix}{job_id}"
                self._client.setex(result_key, self.config.result_ttl, json.dumps(result))
            
            logger.info(f"Job {job_id} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete job {job_id}: {e}")
            return False
    
    def fail_job(self, job_id: str, error: str, retry: bool = True) -> bool:
        """Mark job as failed and optionally retry."""
        try:
            job = self.get_job(job_id)
            if not job:
                logger.warning(f"Job {job_id} not found")
                return False
            
            job.retry_count += 1
            job.error = error
            
            # Retry if under limit
            if retry and job.retry_count <= job.max_retries:
                job.status = JobStatus.PENDING
                job.started_at = None
                
                # Re-enqueue with same priority
                queue_name = self.queue_names[job.priority]
                self._client.lpush(queue_name, job_id)
                
                logger.info(f"Job {job_id} failed, retrying (attempt {job.retry_count}/{job.max_retries})")
            else:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.utcnow()
                logger.error(f"Job {job_id} failed permanently: {error}")
            
            self._update_job(job)
            return True
            
        except Exception as e:
            logger.error(f"Failed to fail job {job_id}: {e}")
            return False
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job."""
        try:
            job = self.get_job(job_id)
            if not job:
                logger.warning(f"Job {job_id} not found")
                return False
            
            if job.status != JobStatus.PENDING:
                logger.warning(f"Job {job_id} cannot be cancelled (status: {job.status.value})")
                return False
            
            # Remove from all queues
            for queue_name in self.queue_names.values():
                self._client.lrem(queue_name, 0, job_id)
            
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            self._update_job(job)
            
            logger.info(f"Job {job_id} cancelled")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    def get_job_result(self, job_id: str) -> Any:
        """Get job result by ID."""
        try:
            result_key = f"{self.result_key_prefix}{job_id}"
            result_data = self._client.get(result_key)
            
            if result_data:
                return json.loads(result_data)
            
            # Fall back to job data
            job = self.get_job(job_id)
            return job.result if job else None
            
        except Exception as e:
            logger.error(f"Failed to get result for job {job_id}: {e}")
            return None
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        try:
            stats = {}
            
            for priority, queue_name in self.queue_names.items():
                stats[priority.name.lower()] = self._client.llen(queue_name)
            
            # Get job counts by status
            job_keys = self._client.keys(f"{self.job_key_prefix}*")
            status_counts = {status.value: 0 for status in JobStatus}
            
            for job_key in job_keys:
                job_data = self._client.get(job_key)
                if job_data:
                    job = json.loads(job_data)
                    status = job.get("status", "unknown")
                    if status in status_counts:
                        status_counts[status] += 1
            
            stats["job_status"] = status_counts
            stats["total_jobs"] = sum(status_counts.values())
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {}
    
    def clear_completed_jobs(self, older_than_hours: int = 24) -> int:
        """Clear completed jobs older than specified hours."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
            cleared_count = 0
            
            job_keys = self._client.keys(f"{self.job_key_prefix}*")
            
            for job_key in job_keys:
                job_data = self._client.get(job_key)
                if job_data:
                    job = json.loads(job_data)
                    
                    # Check if job is completed and old enough
                    if (job.get("status") in [JobStatus.COMPLETED.value, JobStatus.FAILED.value] and
                        job.get("completed_at")):
                        
                        completed_at = datetime.fromisoformat(job["completed_at"])
                        if completed_at < cutoff_time:
                            # Delete job and result
                            job_id = job["id"]
                            self._client.delete(job_key)
                            self._client.delete(f"{self.result_key_prefix}{job_id}")
                            cleared_count += 1
            
            logger.info(f"Cleared {cleared_count} completed jobs older than {older_than_hours} hours")
            return cleared_count
            
        except Exception as e:
            logger.error(f"Failed to clear completed jobs: {e}")
            return 0


# Global job queue instance
_redis_queue: Optional[RedisJobQueue] = None


def get_redis_queue(config: Optional[RedisConfig] = None) -> RedisJobQueue:
    """Get global Redis job queue instance."""
    global _redis_queue
    
    if _redis_queue is None:
        _redis_queue = RedisJobQueue(config)
    
    return _redis_queue