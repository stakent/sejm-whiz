"""Tests for batch processing functionality."""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import time

from sejm_whiz.embeddings.batch_processor import (
    BatchProcessor, get_batch_processor,
    BatchJob, BatchResult,
    batch_encode_texts, batch_generate_bag_embeddings, batch_calculate_similarities
)
from sejm_whiz.embeddings.config import EmbeddingConfig


class TestBatchProcessor:
    """Test batch processor functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock embedding configuration."""
        return EmbeddingConfig(
            batch_size=2,
            max_workers=2,
            embedding_dim=768
        )
    
    @pytest.fixture
    def mock_processor(self, mock_config):
        """Mock batch processor with dependencies."""
        with patch('sejm_whiz.embeddings.batch_processor.get_herbert_encoder') as mock_encoder, \
             patch('sejm_whiz.embeddings.batch_processor.get_bag_embeddings_generator') as mock_bag_gen, \
             patch('sejm_whiz.embeddings.batch_processor.get_similarity_calculator') as mock_sim_calc:
            
            # Mock encoder
            mock_encoder_instance = Mock()
            mock_encoder_instance.encode_with_metadata.return_value = [
                Mock(embedding=np.random.randn(768)) for _ in range(2)
            ]
            mock_encoder.return_value = mock_encoder_instance
            
            # Mock bag generator
            mock_bag_gen_instance = Mock()
            mock_bag_gen_instance.generate_bag_embeddings_batch.return_value = [
                Mock(vocabulary_size=10) for _ in range(2)
            ]
            mock_bag_gen.return_value = mock_bag_gen_instance
            
            # Mock similarity calculator
            mock_sim_calc_instance = Mock()
            mock_sim_calc_instance.find_most_similar.return_value = [(0, 0.9, 'label')]
            mock_sim_calc.return_value = mock_sim_calc_instance
            
            processor = BatchProcessor(mock_config)
            return processor
    
    def test_initialization(self, mock_config):
        """Test processor initialization."""
        with patch('sejm_whiz.embeddings.batch_processor.get_herbert_encoder'), \
             patch('sejm_whiz.embeddings.batch_processor.get_bag_embeddings_generator'), \
             patch('sejm_whiz.embeddings.batch_processor.get_similarity_calculator'):
            
            processor = BatchProcessor(mock_config)
            
            assert processor.config == mock_config
            assert processor.max_workers == 2
            assert processor.batch_size == 2
            assert processor.encoder is not None
            assert processor.bag_generator is not None
            assert processor.similarity_calculator is not None
    
    @pytest.mark.asyncio
    async def test_process_embedding_batch_simple(self, mock_processor):
        """Test simple embedding batch processing."""
        texts = ["Ustawa o podatku", "Rozporządzenie VAT"]
        
        result = await mock_processor.process_embedding_batch(texts, use_threading=False)
        
        assert isinstance(result, BatchResult)
        assert result.job_type == 'embedding_batch'
        assert result.success_count > 0
        assert result.processing_time > 0
        assert len(result.results) > 0
        assert 'texts_processed' in result.metadata
        
        # Check that encoder was called
        mock_processor.encoder.encode_with_metadata.assert_called_once_with(texts)
    
    @pytest.mark.asyncio
    async def test_process_embedding_batch_with_threading(self, mock_processor):
        """Test embedding batch processing with threading."""
        texts = ["Text 1", "Text 2", "Text 3", "Text 4"]  # More than batch_size
        
        with patch.object(mock_processor, '_process_embeddings_parallel') as mock_parallel:
            mock_parallel.return_value = [Mock(embedding=np.random.randn(768)) for _ in range(4)]
            
            result = await mock_processor.process_embedding_batch(texts, use_threading=True)
            
            assert result.success_count == 4
            mock_parallel.assert_called_once_with(texts)
    
    @pytest.mark.asyncio
    async def test_process_embedding_batch_with_error(self, mock_processor):
        """Test embedding batch processing with error."""
        texts = ["Text 1", "Text 2"]
        
        # Mock encoder to raise exception
        mock_processor.encoder.encode_with_metadata.side_effect = ValueError("Mock error")
        
        result = await mock_processor.process_embedding_batch(texts)
        
        assert result.success_count == 0
        assert result.error_count == 2
        assert len(result.errors) > 0
        assert "Mock error" in result.errors[0]
    
    @pytest.mark.asyncio
    async def test_process_bag_embeddings_batch(self, mock_processor):
        """Test bag embeddings batch processing."""
        documents = [
            {"text": "Ustawa o podatku", "title": "Podatek"},
            {"text": "Rozporządzenie VAT", "title": "VAT"}
        ]
        
        result = await mock_processor.process_bag_embeddings_batch(documents)
        
        assert isinstance(result, BatchResult)
        assert result.job_type == 'bag_embeddings_batch'
        assert result.success_count > 0
        assert 'documents_processed' in result.metadata
        assert 'use_tf_idf' in result.metadata
        assert 'use_legal_weighting' in result.metadata
        
        # Check that bag generator was called
        mock_processor.bag_generator.generate_bag_embeddings_batch.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_bag_embeddings_batch_with_error(self, mock_processor):
        """Test bag embeddings batch with error."""
        documents = [{"text": "Test text"}]
        
        # Mock bag generator to raise exception
        mock_processor.bag_generator.generate_bag_embeddings_batch.side_effect = ValueError("Mock error")
        
        result = await mock_processor.process_bag_embeddings_batch(documents)
        
        assert result.success_count == 0
        assert result.error_count == 1
        assert len(result.errors) > 0
    
    @pytest.mark.asyncio
    async def test_process_similarity_batch(self, mock_processor):
        """Test similarity batch processing."""
        query_embeddings = [np.random.randn(768) for _ in range(2)]
        candidate_embeddings = [np.random.randn(768) for _ in range(5)]
        
        result = await mock_processor.process_similarity_batch(
            query_embeddings, candidate_embeddings, top_k=3
        )
        
        assert isinstance(result, BatchResult)
        assert result.job_type == 'similarity_batch'
        assert result.success_count > 0
        assert 'queries_processed' in result.metadata
        assert 'candidates_count' in result.metadata
        assert 'method' in result.metadata
        assert 'top_k' in result.metadata
        
        # Should have called find_most_similar for each query
        assert mock_processor.similarity_calculator.find_most_similar.call_count == 2
    
    @pytest.mark.asyncio
    async def test_process_similarity_batch_with_error(self, mock_processor):
        """Test similarity batch with partial errors."""
        query_embeddings = [np.random.randn(768) for _ in range(2)]
        candidate_embeddings = [np.random.randn(768) for _ in range(3)]
        
        # Mock similarity calculator to raise exception for second query
        def side_effect(*args, **kwargs):
            if side_effect.call_count == 1:
                return [(0, 0.9, 'label')]
            else:
                raise ValueError("Mock error")
        side_effect.call_count = 0
        
        def mock_find_most_similar(*args, **kwargs):
            side_effect.call_count += 1
            return side_effect(*args, **kwargs)
        
        mock_processor.similarity_calculator.find_most_similar.side_effect = mock_find_most_similar
        
        result = await mock_processor.process_similarity_batch(
            query_embeddings, candidate_embeddings
        )
        
        assert result.success_count == 1
        assert result.error_count == 1
        assert len(result.errors) > 0
        assert len(result.results) == 2  # Should have results for both queries
        assert result.results[1] == []  # Second query should have empty result
    
    @pytest.mark.asyncio
    async def test_process_embeddings_parallel(self, mock_processor):
        """Test parallel embedding processing."""
        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
        
        # Mock the encoder for each chunk
        def mock_encode_chunk(chunk):
            return [Mock(embedding=np.random.randn(768)) for _ in chunk]
        
        mock_processor.encoder.encode_with_metadata.side_effect = mock_encode_chunk
        
        results = await mock_processor._process_embeddings_parallel(texts)
        
        assert len(results) == 5
        # Should have been called multiple times (once per chunk)
        assert mock_processor.encoder.encode_with_metadata.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_enqueue_job(self, mock_processor):
        """Test job enqueueing."""
        job = BatchJob(
            job_id="test_job_1",
            job_type="embedding_batch",
            data=["text1", "text2"],
            config={},
            created_at=time.time()
        )
        
        job_id = await mock_processor.enqueue_job(job)
        
        assert job_id == "test_job_1"
        assert job.job_id in mock_processor.active_jobs
        assert mock_processor.job_queue.qsize() == 1
    
    def test_get_job_result_not_found(self, mock_processor):
        """Test getting result for non-existent job."""
        result = mock_processor.get_job_result("non_existent_job")
        assert result is None
    
    def test_get_job_result_found(self, mock_processor):
        """Test getting result for existing job."""
        job_id = "test_job"
        batch_result = BatchResult(
            job_id=job_id,
            job_type="test",
            success_count=1,
            error_count=0,
            processing_time=1.0,
            results=[],
            errors=[],
            metadata={}
        )
        mock_processor.results_cache[job_id] = batch_result
        
        result = mock_processor.get_job_result(job_id)
        assert result == batch_result
    
    def test_get_job_status(self, mock_processor):
        """Test getting job status."""
        # Test not found
        assert mock_processor.get_job_status("not_found") == 'not_found'
        
        # Test processing
        job = BatchJob("processing_job", "test", [], {}, time.time())
        mock_processor.active_jobs["processing_job"] = job
        assert mock_processor.get_job_status("processing_job") == 'processing'
        
        # Test completed
        result = BatchResult("completed_job", "test", 1, 0, 1.0, [], [], {})
        mock_processor.results_cache["completed_job"] = result
        assert mock_processor.get_job_status("completed_job") == 'completed'
    
    def test_get_processing_stats(self, mock_processor):
        """Test getting processing statistics."""
        # Add some mock stats
        mock_processor.processing_stats['embedding_batch'].append({
            'job_id': 'job1',
            'processing_time': 1.0,
            'success_rate': 1.0
        })
        mock_processor.processing_stats['embedding_batch'].append({
            'job_id': 'job2',
            'processing_time': 2.0,
            'success_rate': 0.8
        })
        
        stats = mock_processor.get_processing_stats()
        
        assert 'embedding_batch' in stats
        assert 'queue_size' in stats
        assert 'active_jobs' in stats
        assert 'cached_results' in stats
        
        embedding_stats = stats['embedding_batch']
        assert embedding_stats['total_jobs'] == 2
        assert embedding_stats['avg_processing_time'] == 1.5
        assert embedding_stats['avg_success_rate'] == 0.9
    
    def test_clear_cache(self, mock_processor):
        """Test cache clearing."""
        import time
        current_time = time.time()
        
        # Add some mock results with proper timestamps
        old_result = Mock()
        old_result.created_at = current_time - 7200  # 2 hours old (should be cleared)
        
        recent_result = Mock()
        recent_result.created_at = current_time - 1800  # 30 minutes old (should not be cleared)
        
        no_timestamp_result = Mock(spec=[])  # Mock without created_at attribute
        
        mock_processor.results_cache["old_job"] = old_result
        mock_processor.results_cache["recent_job"] = recent_result  
        mock_processor.results_cache["no_timestamp_job"] = no_timestamp_result
        
        # Clear cache with default max_age of 3600 seconds (1 hour)
        cleared_count = mock_processor.clear_cache()
        
        # Should clear 1 result (the old one), keep 2 (recent and no timestamp)
        assert cleared_count == 1
        assert len(mock_processor.results_cache) == 2
        assert "old_job" not in mock_processor.results_cache
        assert "recent_job" in mock_processor.results_cache
        assert "no_timestamp_job" in mock_processor.results_cache
    
    @pytest.mark.asyncio
    async def test_shutdown(self, mock_processor):
        """Test processor shutdown."""
        # Add some mock data
        mock_processor.results_cache["job1"] = Mock()
        mock_processor.active_jobs["job1"] = Mock()
        
        await mock_processor.shutdown()
        
        assert len(mock_processor.results_cache) == 0
        assert len(mock_processor.active_jobs) == 0


def test_get_batch_processor():
    """Test global processor instance."""
    with patch('sejm_whiz.embeddings.batch_processor.BatchProcessor') as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        
        processor1 = get_batch_processor()
        processor2 = get_batch_processor()
        
        # Should return same instance (singleton pattern)
        assert processor1 == processor2
        mock_class.assert_called_once()


@pytest.mark.asyncio
async def test_batch_encode_texts_convenience():
    """Test batch encode texts convenience function."""
    texts = ["text1", "text2"]
    
    with patch('sejm_whiz.embeddings.batch_processor.get_batch_processor') as mock_get_processor:
        mock_processor = Mock()
        mock_processor.process_embedding_batch = AsyncMock(return_value=Mock())
        mock_get_processor.return_value = mock_processor
        
        result = await batch_encode_texts(texts)
        
        mock_processor.process_embedding_batch.assert_called_once_with(texts)
        assert result is not None


@pytest.mark.asyncio
async def test_batch_generate_bag_embeddings_convenience():
    """Test batch generate bag embeddings convenience function."""
    documents = [{"text": "test"}]
    
    with patch('sejm_whiz.embeddings.batch_processor.get_batch_processor') as mock_get_processor:
        mock_processor = Mock()
        mock_processor.process_bag_embeddings_batch = AsyncMock(return_value=Mock())
        mock_get_processor.return_value = mock_processor
        
        result = await batch_generate_bag_embeddings(documents)
        
        mock_processor.process_bag_embeddings_batch.assert_called_once_with(documents)
        assert result is not None


@pytest.mark.asyncio
async def test_batch_calculate_similarities_convenience():
    """Test batch calculate similarities convenience function."""
    queries = [np.random.randn(768)]
    candidates = [np.random.randn(768)]
    
    with patch('sejm_whiz.embeddings.batch_processor.get_batch_processor') as mock_get_processor:
        mock_processor = Mock()
        mock_processor.process_similarity_batch = AsyncMock(return_value=Mock())
        mock_get_processor.return_value = mock_processor
        
        result = await batch_calculate_similarities(queries, candidates)
        
        mock_processor.process_similarity_batch.assert_called_once_with(queries, candidates)
        assert result is not None


class TestBatchJob:
    """Test BatchJob data class."""
    
    def test_creation(self):
        """Test job creation."""
        job = BatchJob(
            job_id="test_job",
            job_type="embedding_batch",
            data=["text1", "text2"],
            config={"use_threading": True},
            created_at=1234567890.0,
            priority=5
        )
        
        assert job.job_id == "test_job"
        assert job.job_type == "embedding_batch"
        assert job.data == ["text1", "text2"]
        assert job.config["use_threading"] == True
        assert job.created_at == 1234567890.0
        assert job.priority == 5


class TestBatchResult:
    """Test BatchResult data class."""
    
    def test_creation(self):
        """Test result creation."""
        result = BatchResult(
            job_id="test_job",
            job_type="embedding_batch",
            success_count=5,
            error_count=1,
            processing_time=2.5,
            results=["result1", "result2"],
            errors=["error1"],
            metadata={"total": 6}
        )
        
        assert result.job_id == "test_job"
        assert result.job_type == "embedding_batch"
        assert result.success_count == 5
        assert result.error_count == 1
        assert result.processing_time == 2.5
        assert result.results == ["result1", "result2"]
        assert result.errors == ["error1"]
        assert result.metadata["total"] == 6