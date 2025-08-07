#!/usr/bin/env python3
"""
Test Persistent Disc Cache System
Test the cache system functionality for API responses and processed data.
"""

import asyncio
import logging
import sys
import os
import json
import time
from datetime import datetime, UTC
from pathlib import Path

# Set up environment for baremetal deployment
os.environ["DEPLOYMENT_ENV"] = "baremetal"

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "components"))
sys.path.insert(0, str(project_root / "bases"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cache_test.log')
    ]
)

logger = logging.getLogger(__name__)


async def test_cache_configuration():
    """Test cache configuration and directory setup."""
    logger.info("🔧 Testing cache configuration...")
    
    try:
        from sejm_whiz.cache.config import get_cache_config
        from sejm_whiz.cache.manager import get_cache_manager
        
        config = get_cache_config()
        logger.info(f"✅ Cache root: {config.cache_root}")
        logger.info(f"✅ Sejm API dir: {config.sejm_api_dir}")
        logger.info(f"✅ ELI API dir: {config.eli_api_dir}")
        logger.info(f"✅ TTL: {config.api_cache_ttl}s")
        logger.info(f"✅ Compression: {config.compress_responses}")
        
        # Test directory creation
        config.ensure_directories()
        logger.info("✅ Cache directories ensured")
        
        # Initialize cache manager
        cache = get_cache_manager()
        logger.info("✅ Cache manager initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Cache configuration test failed: {e}")
        return False


async def test_api_response_caching():
    """Test API response caching functionality."""
    logger.info("📡 Testing API response caching...")
    
    try:
        from sejm_whiz.cache.manager import get_cache_manager
        
        cache = get_cache_manager()
        
        # Test data
        test_endpoint = "/api/sejm/proceedings"
        test_params = {"term": 10, "limit": 50}
        test_response = {
            "proceedings": [
                {"id": 1, "title": "Test proceeding 1", "date": "2024-01-15"},
                {"id": 2, "title": "Test proceeding 2", "date": "2024-01-16"}
            ],
            "total": 2,
            "fetched_at": datetime.now(UTC).isoformat()
        }
        
        # Test caching
        logger.info("📥 Caching test API response...")
        cache_key = cache.cache_api_response("sejm", test_endpoint, test_params, test_response)
        logger.info(f"✅ Cached with key: {cache_key[:12]}...")
        
        # Test retrieval
        logger.info("📤 Retrieving cached API response...")
        cached_data = cache.get_cached_api_response("sejm", test_endpoint, test_params)
        
        if cached_data and cached_data == test_response:
            logger.info("✅ API response caching successful")
            logger.info(f"✅ Retrieved {len(cached_data['proceedings'])} proceedings")
            return True
        else:
            logger.error("❌ API response caching failed: data mismatch")
            return False
            
    except Exception as e:
        logger.error(f"❌ API response caching test failed: {e}")
        return False


async def test_processed_data_caching():
    """Test processed data caching functionality."""
    logger.info("⚙️ Testing processed data caching...")
    
    try:
        from sejm_whiz.cache.manager import get_cache_manager
        
        cache = get_cache_manager()
        
        # Test processed text data
        test_data_type = "legal_text_analysis"
        test_identifier = "document_12345"
        test_processed_data = {
            "extracted_text": "Artykuł 1. Niniejsza ustawa określa zasady...",
            "entities": ["ustawa", "artykuł", "zasady"],
            "legal_references": ["Art. 1", "ust. 2"],
            "processing_date": datetime.now(UTC).isoformat(),
            "confidence_score": 0.95
        }
        
        # Test caching
        logger.info("📥 Caching processed data...")
        cache_key = cache.cache_processed_data(test_data_type, test_identifier, test_processed_data)
        logger.info(f"✅ Cached processed data: {cache_key}")
        
        # Test retrieval
        logger.info("📤 Retrieving cached processed data...")
        cached_data = cache.get_cached_processed_data(test_data_type, test_identifier)
        
        if cached_data and cached_data == test_processed_data:
            logger.info("✅ Processed data caching successful")
            logger.info(f"✅ Retrieved confidence score: {cached_data['confidence_score']}")
            return True
        else:
            logger.error("❌ Processed data caching failed: data mismatch")
            return False
            
    except Exception as e:
        logger.error(f"❌ Processed data caching test failed: {e}")
        return False


async def test_embeddings_caching():
    """Test embeddings caching functionality."""
    logger.info("🧠 Testing embeddings caching...")
    
    try:
        from sejm_whiz.cache.manager import get_cache_manager
        
        cache = get_cache_manager()
        
        # Test embeddings data (simulated)
        test_doc_id = "legal_doc_67890"
        test_embeddings = [0.1, 0.2, 0.3, 0.4, 0.5] * 150  # 768-dimensional vector simulation
        
        # Test caching
        logger.info("📥 Caching document embeddings...")
        cache_key = cache.cache_embeddings(test_doc_id, test_embeddings)
        logger.info(f"✅ Cached embeddings: {cache_key}")
        
        # Test retrieval
        logger.info("📤 Retrieving cached embeddings...")
        cached_embeddings = cache.get_cached_embeddings(test_doc_id)
        
        if cached_embeddings and cached_embeddings == test_embeddings:
            logger.info("✅ Embeddings caching successful")
            logger.info(f"✅ Retrieved {len(cached_embeddings)} dimensional embeddings")
            return True
        else:
            logger.error("❌ Embeddings caching failed: data mismatch")
            return False
            
    except Exception as e:
        logger.error(f"❌ Embeddings caching test failed: {e}")
        return False


async def test_cache_performance():
    """Test cache performance and timing."""
    logger.info("⚡ Testing cache performance...")
    
    try:
        from sejm_whiz.cache.manager import get_cache_manager
        
        cache = get_cache_manager()
        
        # Performance test data
        large_response = {
            "documents": [{"id": i, "title": f"Document {i}", "content": "A" * 1000} 
                         for i in range(100)],
            "metadata": {"total": 100, "page": 1, "per_page": 100}
        }
        
        endpoint = "/api/performance/test"
        params = {"test": "performance", "size": "large"}
        
        # Time the caching operation
        start_time = time.time()
        cache.cache_api_response("sejm", endpoint, params, large_response)
        cache_time = time.time() - start_time
        
        # Time the retrieval operation
        start_time = time.time()
        retrieved_data = cache.get_cached_api_response("sejm", endpoint, params)
        retrieval_time = time.time() - start_time
        
        logger.info(f"✅ Cache write time: {cache_time:.4f}s")
        logger.info(f"✅ Cache read time: {retrieval_time:.4f}s")
        logger.info(f"✅ Retrieved {len(retrieved_data['documents'])} documents")
        
        return cache_time < 1.0 and retrieval_time < 0.5  # Performance thresholds
        
    except Exception as e:
        logger.error(f"❌ Cache performance test failed: {e}")
        return False


async def test_cache_stats_and_cleanup():
    """Test cache statistics and cleanup functionality."""
    logger.info("📊 Testing cache stats and cleanup...")
    
    try:
        from sejm_whiz.cache.manager import get_cache_manager
        
        cache = get_cache_manager()
        
        # Get cache statistics
        stats = cache.get_cache_stats()
        logger.info("📈 Cache Statistics:")
        logger.info(f"   Total files: {stats['total_files']}")
        logger.info(f"   Total size: {stats['total_size_mb']:.2f} MB")
        
        for cache_type, type_stats in stats['by_type'].items():
            if type_stats['files'] > 0:
                logger.info(f"   {cache_type}: {type_stats['files']} files, {type_stats['size_mb']:.2f} MB")
        
        # Test cleanup (won't actually remove anything in normal test due to recent creation)
        cleanup_stats = cache.cleanup_expired_cache()
        logger.info(f"🧹 Cleanup results: {cleanup_stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Cache stats/cleanup test failed: {e}")
        return False


async def main():
    """Main test execution."""
    logger.info("🚀 Starting Cache System Tests")
    logger.info("🎯 Testing persistent disc cache for API responses and processed data")
    logger.info("=" * 70)
    
    start_time = datetime.now(UTC)
    tests = [
        ("Configuration", test_cache_configuration),
        ("API Response Caching", test_api_response_caching),
        ("Processed Data Caching", test_processed_data_caching),
        ("Embeddings Caching", test_embeddings_caching),
        ("Performance", test_cache_performance),
        ("Stats & Cleanup", test_cache_stats_and_cleanup),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"TEST: {test_name.upper()}")
        logger.info(f"{'=' * 50}")
        
        try:
            result = await test_func()
            if result:
                logger.info(f"✅ {test_name} test PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} test FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"💥 {test_name} test CRASHED: {e}")
            failed += 1
    
    # Final summary
    end_time = datetime.now(UTC)
    duration = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "=" * 70)
    logger.info("🎉 CACHE SYSTEM TESTS COMPLETED")
    logger.info("=" * 70)
    logger.info(f"⏱️  Total time: {duration:.2f} seconds")
    logger.info(f"✅ Tests passed: {passed}")
    logger.info(f"❌ Tests failed: {failed}")
    logger.info(f"📈 Success rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        logger.info("🎯 ALL TESTS PASSED - Cache system is ready for production!")
        logger.info("💾 Persistent disc cache will speed up pipeline iterations")
        logger.info("🚀 API endpoints will be protected from excessive requests")
    else:
        logger.warning(f"⚠️ {failed} test(s) failed - please review and fix issues")


if __name__ == "__main__":
    asyncio.run(main())