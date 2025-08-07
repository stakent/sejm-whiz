#!/usr/bin/env python3
"""
Cache Performance Test - January 2025 Documents
Test cache system by fetching January 2025 documents twice and comparing timings.
"""

import asyncio
import logging
import sys
import os
import time
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Any

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
        logging.FileHandler('cache_performance_test.log')
    ]
)

logger = logging.getLogger(__name__)


class MockSejmApiClient:
    """Mock Sejm API client that simulates network delays."""
    
    def __init__(self, delay_seconds: float = 1.0):
        self.delay = delay_seconds
        self.call_count = 0
        logger.info(f"Mock Sejm API client initialized with {delay_seconds}s delay")
    
    async def get_proceedings(self, **params) -> Dict[str, Any]:
        """Simulate fetching proceedings with network delay."""
        self.call_count += 1
        logger.info(f"🌐 API Call #{self.call_count}: Fetching proceedings with params: {params}")
        
        # Simulate network delay
        await asyncio.sleep(self.delay)
        
        # Generate mock data based on parameters
        term = params.get('term', 10)
        since_date = params.get('since', '2025-01-01')
        limit = params.get('limit', 50)
        
        # Simulate realistic January 2025 proceedings
        mock_proceedings = []
        for i in range(min(limit, 15)):  # Max 15 proceedings for January 2025
            mock_proceedings.append({
                "id": f"2025_01_{i+1:02d}",
                "term": term,
                "meeting_num": i + 1,
                "date": f"2025-01-{(i*2)+1:02d}",
                "title": f"Posiedzenie Sejmu RP z dnia {(i*2)+1} stycznia 2025 r.",
                "agenda": [
                    f"Punkt {j+1}: Projekt ustawy nr {100+i+j}" 
                    for j in range(3)
                ],
                "status": "completed",
                "documents_count": 15 + i,
                "fetch_timestamp": datetime.now(UTC).isoformat()
            })
        
        result = {
            "proceedings": mock_proceedings,
            "total": len(mock_proceedings),
            "params": params,
            "fetched_at": datetime.now(UTC).isoformat(),
            "api_call_number": self.call_count
        }
        
        logger.info(f"✅ API returned {len(mock_proceedings)} proceedings")
        return result
    
    async def get_documents_for_date(self, date: str, **params) -> Dict[str, Any]:
        """Simulate fetching documents for a specific date."""
        self.call_count += 1
        logger.info(f"🌐 API Call #{self.call_count}: Fetching documents for {date}")
        
        # Simulate network delay
        await asyncio.sleep(self.delay)
        
        # Generate mock documents
        mock_documents = []
        for i in range(5):  # 5 documents per date
            mock_documents.append({
                "id": f"doc_{date}_{i+1}",
                "title": f"Dokument {i+1} z dnia {date}",
                "type": "ustawa" if i % 2 == 0 else "rozporządzenie",
                "date": date,
                "content_preview": f"Treść dokumentu {i+1} z dnia {date}...",
                "size_kb": 50 + i * 10,
                "fetch_timestamp": datetime.now(UTC).isoformat()
            })
        
        result = {
            "documents": mock_documents,
            "date": date,
            "total": len(mock_documents),
            "api_call_number": self.call_count,
            "fetched_at": datetime.now(UTC).isoformat()
        }
        
        logger.info(f"✅ API returned {len(mock_documents)} documents for {date}")
        return result


async def test_cache_timing_performance():
    """Test cache performance by fetching January 2025 documents twice."""
    logger.info("🚀 Starting Cache Performance Test - January 2025 Documents")
    logger.info("=" * 70)
    
    try:
        from sejm_whiz.cache.manager import get_cache_manager
        from sejm_whiz.cache.integration import CachedSejmApiClient
        
        # Initialize components
        cache = get_cache_manager()
        original_client = MockSejmApiClient(delay_seconds=1.5)  # 1.5s delay to simulate real API
        cached_client = CachedSejmApiClient(original_client, cache)
        
        # Clear any existing cache for clean test
        initial_stats = cache.get_cache_stats()
        logger.info(f"📊 Initial cache stats: {initial_stats['total_files']} files, {initial_stats['total_size_mb']:.2f} MB")
        
        # Test parameters for January 2025
        test_params = [
            {"term": 10, "since": "2025-01-01", "until": "2025-01-31", "limit": 20},
            {"date": "2025-01-15"},
            {"date": "2025-01-20"},
            {"term": 10, "meeting_num": 1},
        ]
        
        # Performance tracking
        performance_results = {
            "first_run": {"total_time": 0, "api_calls": 0, "cache_misses": 0},
            "second_run": {"total_time": 0, "api_calls": 0, "cache_hits": 0}
        }
        
        # FIRST RUN - Populate cache
        logger.info("\n" + "=" * 50)
        logger.info("🔄 FIRST RUN - POPULATING CACHE")
        logger.info("=" * 50)
        
        first_run_start = time.time()
        first_run_data = []
        
        # Reset API call counter
        original_client.call_count = 0
        
        for i, params in enumerate(test_params):
            logger.info(f"\n📡 First run - Request {i+1}/4")
            
            if "date" in params:
                # Document fetch
                start_time = time.time()
                result = await cached_client.get_documents_for_date(**params)
                end_time = time.time()
                
                logger.info(f"⏱️  Request {i+1} time: {(end_time - start_time):.3f}s")
                logger.info(f"📄 Retrieved {len(result['documents'])} documents")
                first_run_data.append(result)
            else:
                # Proceedings fetch
                start_time = time.time()
                result = await cached_client.get_proceedings(**params)
                end_time = time.time()
                
                logger.info(f"⏱️  Request {i+1} time: {(end_time - start_time):.3f}s")
                logger.info(f"📋 Retrieved {len(result['proceedings'])} proceedings")
                first_run_data.append(result)
        
        first_run_end = time.time()
        performance_results["first_run"]["total_time"] = first_run_end - first_run_start
        performance_results["first_run"]["api_calls"] = original_client.call_count
        performance_results["first_run"]["cache_misses"] = original_client.call_count
        
        logger.info(f"\n✅ First run completed in {performance_results['first_run']['total_time']:.2f}s")
        logger.info(f"🌐 Total API calls made: {original_client.call_count}")
        
        # Check cache after first run
        cache_stats_after_first = cache.get_cache_stats()
        logger.info(f"📊 Cache after first run: {cache_stats_after_first['total_files']} files")
        
        # Wait a moment to ensure clear separation
        await asyncio.sleep(1)
        
        # SECOND RUN - Use cache
        logger.info("\n" + "=" * 50)
        logger.info("⚡ SECOND RUN - USING CACHE")
        logger.info("=" * 50)
        
        second_run_start = time.time()
        second_run_data = []
        
        # Reset API call counter for second run
        api_calls_before_second = original_client.call_count
        
        for i, params in enumerate(test_params):
            logger.info(f"\n💾 Second run - Request {i+1}/4")
            
            if "date" in params:
                # Document fetch
                start_time = time.time()
                result = await cached_client.get_documents_for_date(**params)
                end_time = time.time()
                
                logger.info(f"⚡ Request {i+1} time: {(end_time - start_time):.3f}s")
                logger.info(f"📄 Retrieved {len(result['documents'])} documents (cached)")
                second_run_data.append(result)
            else:
                # Proceedings fetch
                start_time = time.time()
                result = await cached_client.get_proceedings(**params)
                end_time = time.time()
                
                logger.info(f"⚡ Request {i+1} time: {(end_time - start_time):.3f}s")
                logger.info(f"📋 Retrieved {len(result['proceedings'])} proceedings (cached)")
                second_run_data.append(result)
        
        second_run_end = time.time()
        performance_results["second_run"]["total_time"] = second_run_end - second_run_start
        performance_results["second_run"]["api_calls"] = original_client.call_count - api_calls_before_second
        performance_results["second_run"]["cache_hits"] = len(test_params) - performance_results["second_run"]["api_calls"]
        
        logger.info(f"\n🚀 Second run completed in {performance_results['second_run']['total_time']:.2f}s")
        logger.info(f"🌐 Additional API calls made: {performance_results['second_run']['api_calls']}")
        
        # Final cache statistics
        final_cache_stats = cache.get_cache_stats()
        logger.info(f"📊 Final cache stats: {final_cache_stats['total_files']} files, {final_cache_stats['total_size_mb']:.3f} MB")
        
        # Verify data consistency
        logger.info("\n" + "=" * 50)
        logger.info("🔍 DATA CONSISTENCY CHECK")
        logger.info("=" * 50)
        
        data_consistent = True
        for i, (first_data, second_data) in enumerate(zip(first_run_data, second_run_data)):
            if first_data == second_data:
                logger.info(f"✅ Request {i+1}: Data consistent between runs")
            else:
                logger.error(f"❌ Request {i+1}: Data inconsistent!")
                data_consistent = False
        
        # Performance analysis
        logger.info("\n" + "=" * 70)
        logger.info("📈 PERFORMANCE ANALYSIS")
        logger.info("=" * 70)
        
        speedup = performance_results["first_run"]["total_time"] / performance_results["second_run"]["total_time"]
        time_saved = performance_results["first_run"]["total_time"] - performance_results["second_run"]["total_time"]
        api_calls_saved = performance_results["first_run"]["api_calls"] - performance_results["second_run"]["api_calls"]
        
        logger.info(f"🏃 First run (cache miss): {performance_results['first_run']['total_time']:.3f}s")
        logger.info(f"⚡ Second run (cache hit):  {performance_results['second_run']['total_time']:.3f}s")
        logger.info(f"🚀 Speedup factor: {speedup:.1f}x faster")
        logger.info(f"⏰ Time saved: {time_saved:.3f}s ({time_saved*1000:.0f}ms)")
        logger.info(f"🌐 API calls saved: {api_calls_saved}/{performance_results['first_run']['api_calls']}")
        logger.info(f"💾 Cache efficiency: {(api_calls_saved/performance_results['first_run']['api_calls'])*100:.1f}% hits")
        
        # Generate performance report
        report = {
            "test_timestamp": datetime.now(UTC).isoformat(),
            "test_parameters": test_params,
            "performance_results": performance_results,
            "cache_statistics": {
                "initial": initial_stats,
                "final": final_cache_stats,
                "files_created": final_cache_stats['total_files'] - initial_stats['total_files'],
                "storage_used_mb": final_cache_stats['total_size_mb'] - initial_stats['total_size_mb']
            },
            "performance_metrics": {
                "speedup_factor": speedup,
                "time_saved_seconds": time_saved,
                "api_calls_saved": api_calls_saved,
                "cache_hit_rate": (api_calls_saved/performance_results['first_run']['api_calls'])*100,
                "data_consistency": data_consistent
            },
            "conclusion": "SUCCESS" if data_consistent and speedup > 1 else "FAILED"
        }
        
        return report
        
    except Exception as e:
        logger.error(f"❌ Cache performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "conclusion": "FAILED"}


async def main():
    """Main test execution."""
    start_time = datetime.now(UTC)
    
    logger.info("🎯 Testing cache system with January 2025 document fetching")
    logger.info("📡 Simulating realistic API delays and network traffic")
    
    # Run the performance test
    report = await test_cache_timing_performance()
    
    end_time = datetime.now(UTC)
    total_duration = (end_time - start_time).total_seconds()
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("🎉 CACHE PERFORMANCE TEST COMPLETED")
    logger.info("=" * 70)
    logger.info(f"⏱️  Total test duration: {total_duration:.2f}s")
    
    if report.get("conclusion") == "SUCCESS":
        logger.info("🎯 ✅ CACHE SYSTEM WORKING PERFECTLY!")
        logger.info(f"🚀 Achieved {report['performance_metrics']['speedup_factor']:.1f}x speedup")
        logger.info(f"💾 {report['performance_metrics']['cache_hit_rate']:.1f}% cache hit rate")
        logger.info(f"🌐 Saved {report['performance_metrics']['api_calls_saved']} API calls")
        logger.info("📈 Pipeline iterations will be significantly faster!")
    else:
        logger.error("❌ Cache system test failed - please review issues")
    
    # Save detailed report
    with open('cache_performance_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    logger.info("📄 Detailed report saved to cache_performance_report.json")
    
    return report


if __name__ == "__main__":
    asyncio.run(main())