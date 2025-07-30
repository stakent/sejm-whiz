"""Efficient batch processing for embedding generation and similarity calculations."""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Callable, Union, Coroutine
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from collections import defaultdict
import multiprocessing as mp

from .herbert_encoder import HerBERTEncoder, get_herbert_encoder
from .bag_embeddings import BagEmbeddingsGenerator, get_bag_embeddings_generator, BagEmbeddingResult
from .similarity import SimilarityCalculator, get_similarity_calculator, SimilarityResult
from .config import EmbeddingConfig, get_embedding_config

logger = logging.getLogger(__name__)


@dataclass
class BatchJob:
    """Represents a batch processing job."""
    job_id: str
    job_type: str
    data: List[Any]
    config: Dict[str, Any]
    created_at: float
    priority: int = 0
    

@dataclass
class BatchResult:
    """Result of batch processing."""
    job_id: str
    job_type: str
    success_count: int
    error_count: int
    processing_time: float
    results: List[Any]
    errors: List[str]
    metadata: Dict[str, Any]


class BatchProcessor:
    """Efficient batch processor for embedding operations."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize batch processor."""
        self.config = config or get_embedding_config()
        self.encoder = get_herbert_encoder(config)
        self.bag_generator = get_bag_embeddings_generator(config)
        self.similarity_calculator = get_similarity_calculator()
        
        # Processing settings
        self.max_workers = self.config.max_workers
        self.batch_size = self.config.batch_size
        
        # Job queue and results
        self.job_queue = asyncio.Queue()
        self.results_cache = {}
        self.active_jobs = {}
        
        # Performance tracking
        self.processing_stats = defaultdict(list)
        
    async def process_embedding_batch(self, 
                                    texts: List[str],
                                    job_id: str = None,
                                    use_threading: bool = True) -> BatchResult:
        """
        Process a batch of texts for embedding generation.
        
        Args:
            texts: List of texts to process
            job_id: Optional job identifier
            use_threading: Whether to use threading for parallelization
            
        Returns:
            BatchResult with embeddings and metadata
        """
        job_id = job_id or f"embedding_batch_{int(time.time())}"
        logger.info(f"Processing embedding batch {job_id} with {len(texts)} texts")
        
        start_time = time.time()
        results = []
        errors = []
        
        try:
            if use_threading and len(texts) > self.batch_size:
                # Process in parallel batches
                results = await self._process_embeddings_parallel(texts)
            else:
                # Process sequentially
                results = self.encoder.encode_with_metadata(texts)
            
            success_count = len([r for r in results if hasattr(r, 'embedding')])
            error_count = len(texts) - success_count
            
        except Exception as e:
            logger.error(f"Batch processing failed for job {job_id}: {e}")
            errors.append(str(e))
            success_count = 0
            error_count = len(texts)
        
        processing_time = time.time() - start_time
        
        # Track performance
        self.processing_stats['embedding_batch'].append({
            'job_id': job_id,
            'text_count': len(texts),
            'processing_time': processing_time,
            'success_rate': success_count / len(texts) if texts else 0
        })
        
        batch_result = BatchResult(
            job_id=job_id,
            job_type='embedding_batch',
            success_count=success_count,
            error_count=error_count,
            processing_time=processing_time,
            results=results,
            errors=errors,
            metadata={
                'texts_processed': len(texts),
                'avg_processing_time': processing_time / len(texts) if texts else 0,
                'throughput': len(texts) / processing_time if processing_time > 0 else 0
            }
        )
        
        logger.info(f"Completed batch {job_id}: {success_count}/{len(texts)} successful in {processing_time:.2f}s")
        return batch_result
    
    async def process_bag_embeddings_batch(self,
                                         documents: List[Dict[str, str]],
                                         job_id: str = None,
                                         use_tf_idf: bool = True,
                                         use_legal_weighting: bool = True) -> BatchResult:
        """
        Process a batch of documents for bag of embeddings generation.
        
        Args:
            documents: List of documents with 'text' and optional 'title'
            job_id: Optional job identifier
            use_tf_idf: Whether to use TF-IDF weighting
            use_legal_weighting: Whether to use legal term weighting
            
        Returns:
            BatchResult with bag embeddings
        """
        job_id = job_id or f"bag_embeddings_batch_{int(time.time())}"
        logger.info(f"Processing bag embeddings batch {job_id} with {len(documents)} documents")
        
        start_time = time.time()
        errors = []
        
        try:
            results = self.bag_generator.generate_bag_embeddings_batch(
                documents, use_tf_idf, use_legal_weighting
            )
            
            success_count = len([r for r in results if r.vocabulary_size > 0])
            error_count = len(documents) - success_count
            
        except Exception as e:
            logger.error(f"Bag embeddings batch processing failed for job {job_id}: {e}")
            errors.append(str(e))
            results = []
            success_count = 0
            error_count = len(documents)
        
        processing_time = time.time() - start_time
        
        # Track performance
        self.processing_stats['bag_embeddings_batch'].append({
            'job_id': job_id,
            'document_count': len(documents),
            'processing_time': processing_time,
            'success_rate': success_count / len(documents) if documents else 0
        })
        
        batch_result = BatchResult(
            job_id=job_id,
            job_type='bag_embeddings_batch',
            success_count=success_count,
            error_count=error_count,
            processing_time=processing_time,
            results=results,
            errors=errors,
            metadata={
                'documents_processed': len(documents),
                'avg_processing_time': processing_time / len(documents) if documents else 0,
                'use_tf_idf': use_tf_idf,
                'use_legal_weighting': use_legal_weighting
            }
        )
        
        logger.info(f"Completed bag embeddings batch {job_id}: {success_count}/{len(documents)} successful")
        return batch_result
    
    async def process_similarity_batch(self,
                                     query_embeddings: List[np.ndarray],
                                     candidate_embeddings: List[np.ndarray],
                                     job_id: str = None,
                                     method: str = 'cosine',
                                     top_k: int = 10) -> BatchResult:
        """
        Process a batch of similarity calculations.
        
        Args:
            query_embeddings: List of query embeddings
            candidate_embeddings: List of candidate embeddings
            job_id: Optional job identifier
            method: Similarity calculation method
            top_k: Number of top results per query
            
        Returns:
            BatchResult with similarity results
        """
        job_id = job_id or f"similarity_batch_{int(time.time())}"
        logger.info(f"Processing similarity batch {job_id} with {len(query_embeddings)} queries")
        
        start_time = time.time()
        results = []
        errors = []
        
        try:
            for i, query_emb in enumerate(query_embeddings):
                try:
                    similar_results = self.similarity_calculator.find_most_similar(
                        query_emb, candidate_embeddings, method=method, top_k=top_k
                    )
                    results.append(similar_results)
                except Exception as e:
                    logger.error(f"Similarity calculation failed for query {i}: {e}")
                    errors.append(f"Query {i}: {str(e)}")
                    results.append([])
            
            success_count = len([r for r in results if r])
            error_count = len(query_embeddings) - success_count
            
        except Exception as e:
            logger.error(f"Similarity batch processing failed for job {job_id}: {e}")
            errors.append(str(e))
            success_count = 0
            error_count = len(query_embeddings)
        
        processing_time = time.time() - start_time
        
        batch_result = BatchResult(
            job_id=job_id,
            job_type='similarity_batch',
            success_count=success_count,
            error_count=error_count,
            processing_time=processing_time,
            results=results,
            errors=errors,
            metadata={
                'queries_processed': len(query_embeddings),
                'candidates_count': len(candidate_embeddings),
                'method': method,
                'top_k': top_k,
                'avg_query_time': processing_time / len(query_embeddings) if query_embeddings else 0
            }
        )
        
        logger.info(f"Completed similarity batch {job_id}: {success_count}/{len(query_embeddings)} successful")
        return batch_result
    
    async def _process_embeddings_parallel(self, texts: List[str]) -> List[Any]:
        """Process embeddings in parallel using threading."""
        
        def process_chunk(text_chunk):
            """Process a chunk of texts."""
            return self.encoder.encode_with_metadata(text_chunk)
        
        # Split texts into chunks
        chunks = [
            texts[i:i + self.batch_size] 
            for i in range(0, len(texts), self.batch_size)
        ]
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            
            # Collect results
            all_results = []
            for future in futures:
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}")
                    # Add empty results for failed chunk
                    all_results.extend([None] * len(chunks[0]))
        
        return all_results
    
    async def enqueue_job(self, job: BatchJob) -> str:
        """Enqueue a batch job for processing."""
        await self.job_queue.put(job)
        self.active_jobs[job.job_id] = job
        logger.info(f"Enqueued job {job.job_id} of type {job.job_type}")
        return job.job_id
    
    async def process_job_queue(self, max_concurrent_jobs: int = 3) -> None:
        """Process jobs from the queue with concurrency control."""
        
        semaphore = asyncio.Semaphore(max_concurrent_jobs)
        
        async def process_single_job():
            async with semaphore:
                try:
                    job = await asyncio.wait_for(self.job_queue.get(), timeout=1.0)
                    logger.info(f"Processing job {job.job_id}")
                    
                    # Process based on job type
                    if job.job_type == 'embedding_batch':
                        result = await self.process_embedding_batch(
                            job.data, job.job_id, **job.config
                        )
                    elif job.job_type == 'bag_embeddings_batch':
                        result = await self.process_bag_embeddings_batch(
                            job.data, job.job_id, **job.config
                        )
                    elif job.job_type == 'similarity_batch':
                        result = await self.process_similarity_batch(
                            job.data['queries'], job.data['candidates'], 
                            job.job_id, **job.config
                        )
                    else:
                        logger.error(f"Unknown job type: {job.job_type}")
                        return
                    
                    # Store result
                    self.results_cache[job.job_id] = result
                    
                    # Remove from active jobs
                    if job.job_id in self.active_jobs:
                        del self.active_jobs[job.job_id]
                    
                    self.job_queue.task_done()
                    logger.info(f"Completed job {job.job_id}")
                    
                except asyncio.TimeoutError:
                    # No jobs in queue
                    pass
                except Exception as e:
                    logger.error(f"Job processing error: {e}")
        
        # Process jobs continuously
        while True:
            await process_single_job()
            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
    
    def get_job_result(self, job_id: str) -> Optional[BatchResult]:
        """Get result for a completed job."""
        return self.results_cache.get(job_id)
    
    def get_job_status(self, job_id: str) -> str:
        """Get status of a job."""
        if job_id in self.results_cache:
            return 'completed'
        elif job_id in self.active_jobs:
            return 'processing'
        else:
            return 'not_found'
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = {}
        
        for job_type, job_stats in self.processing_stats.items():
            if job_stats:
                processing_times = [s['processing_time'] for s in job_stats]
                success_rates = [s['success_rate'] for s in job_stats]
                
                stats[job_type] = {
                    'total_jobs': len(job_stats),
                    'avg_processing_time': np.mean(processing_times),
                    'avg_success_rate': np.mean(success_rates),
                    'total_processing_time': sum(processing_times)
                }
        
        stats['queue_size'] = self.job_queue.qsize()
        stats['active_jobs'] = len(self.active_jobs)
        stats['cached_results'] = len(self.results_cache)
        
        return stats
    
    def clear_cache(self, max_age: float = 3600) -> int:
        """Clear old results from cache."""
        current_time = time.time()
        cleared_count = 0
        
        to_remove = []
        for job_id, result in self.results_cache.items():
            # Check if result is old (would need timestamp in BatchResult)
            if hasattr(result, 'created_at'):
                if current_time - result.created_at > max_age:
                    to_remove.append(job_id)
        
        for job_id in to_remove:
            del self.results_cache[job_id]
            cleared_count += 1
        
        logger.info(f"Cleared {cleared_count} old results from cache")
        return cleared_count
    
    async def shutdown(self) -> None:
        """Shutdown the batch processor gracefully."""
        logger.info("Shutting down batch processor...")
        
        # Wait for remaining jobs to complete
        await self.job_queue.join()
        
        # Clear caches
        self.results_cache.clear()
        self.active_jobs.clear()
        
        logger.info("Batch processor shutdown complete")


# Global batch processor
_batch_processor: Optional[BatchProcessor] = None


def get_batch_processor(config: Optional[EmbeddingConfig] = None) -> BatchProcessor:
    """Get global batch processor instance."""
    global _batch_processor
    
    if _batch_processor is None:
        _batch_processor = BatchProcessor(config)
    
    return _batch_processor


# Convenience functions for common batch operations
async def batch_encode_texts(texts: List[str], 
                           config: Optional[EmbeddingConfig] = None) -> BatchResult:
    """Batch encode texts with embeddings."""
    processor = get_batch_processor(config)
    return await processor.process_embedding_batch(texts)


async def batch_generate_bag_embeddings(documents: List[Dict[str, str]],
                                      config: Optional[EmbeddingConfig] = None) -> BatchResult:
    """Batch generate bag of embeddings for documents."""
    processor = get_batch_processor(config)
    return await processor.process_bag_embeddings_batch(documents)


async def batch_calculate_similarities(query_embeddings: List[np.ndarray],
                                     candidate_embeddings: List[np.ndarray],
                                     config: Optional[EmbeddingConfig] = None) -> BatchResult:
    """Batch calculate similarities between queries and candidates."""
    processor = get_batch_processor(config)
    return await processor.process_similarity_batch(query_embeddings, candidate_embeddings)