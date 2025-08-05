#!/usr/bin/env python3
"""
Simplified Full Pipeline Benchmark: GPU vs CPU Performance Analysis
Focuses on the core processing components without complex initialization.
"""

import time
import logging
import os
import asyncio
from datetime import datetime
from typing import Dict, List, Any

# Set up environment for p7
os.environ.setdefault("DATABASE_HOST", "p7")
os.environ.setdefault("DATABASE_PORT", "5432")
os.environ.setdefault("DATABASE_NAME", "sejm_whiz")
os.environ.setdefault("DATABASE_USER", "sejm_whiz_user")
os.environ.setdefault("DATABASE_PASSWORD", "sejm_whiz_password")
os.environ.setdefault("DATABASE_SSL_MODE", "disable")
os.environ.setdefault("REDIS_HOST", "p7")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("DEPLOYMENT_ENV", "p7")

# Import core components
from sejm_whiz.sejm_api.client import SejmApiClient
from sejm_whiz.eli_api.client import EliApiClient
from sejm_whiz.text_processing.cleaner import TextCleaner
from sejm_whiz.text_processing.legal_parser import PolishLegalParser
from sejm_whiz.embeddings.herbert_embedder import HerBERTEmbedder
from sejm_whiz.embeddings.config import EmbeddingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimplePipelineBenchmark:
    """Simplified pipeline benchmark focusing on core processing components."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.results = {}

        # Set device environment variable
        os.environ["EMBEDDING_DEVICE"] = device

        # Initialize basic components
        self.text_cleaner = TextCleaner()
        self.legal_parser = PolishLegalParser()
        self.embedder = None

    def initialize_embedder(self):
        """Initialize embedding component."""
        logger.info(f"Initializing embedder for {self.device.upper()}...")
        config = EmbeddingConfig()
        self.embedder = HerBERTEmbedder(config=config)
        logger.info(f"✅ Embedder initialized for {self.device.upper()}")

    async def benchmark_api_fetching(self, num_items: int = 10) -> Dict[str, Any]:
        """Benchmark API data fetching (Sejm + ELI)."""
        logger.info(f"Benchmarking API fetching ({num_items} items)...")

        start_time = time.time()

        # Fetch Sejm proceedings
        sejm_start = time.time()
        sejm_items = []
        try:
            async with SejmApiClient() as client:
                proceedings = await client.get_recent_proceedings(
                    limit=min(num_items, 5)
                )
                sejm_items = proceedings
        except Exception as e:
            logger.warning(f"Sejm API error: {e}")
        sejm_time = time.time() - sejm_start

        # Fetch ELI documents
        eli_start = time.time()
        eli_items = []
        try:
            async with EliApiClient() as client:
                search_result = await client.search_documents(limit=num_items)
                eli_items = search_result.documents
        except Exception as e:
            logger.warning(f"ELI API error: {e}")
        eli_time = time.time() - eli_start

        total_time = time.time() - start_time
        total_items = len(sejm_items) + len(eli_items)

        result = {
            "total_duration": total_time,
            "total_items": total_items,
            "items_per_second": total_items / total_time if total_time > 0 else 0,
            "sejm_time": sejm_time,
            "sejm_items": len(sejm_items),
            "eli_time": eli_time,
            "eli_items": len(eli_items),
        }

        logger.info(
            f"API Fetching: {total_items} items in {total_time:.2f}s = {result['items_per_second']:.2f} items/sec"
        )
        return result

    def benchmark_text_processing(self, texts: List[str]) -> Dict[str, Any]:
        """Benchmark text cleaning and legal parsing."""
        logger.info(f"Benchmarking text processing ({len(texts)} texts)...")

        start_time = time.time()

        # Process texts
        processed_texts = []
        for text in texts:
            # Clean text
            cleaned = self.text_cleaner.clean_text(text)

            # Parse legal structure (for validation, use cleaned text for embedding)
            self.legal_parser.parse_document_structure(cleaned)
            processed_texts.append(cleaned)  # Use cleaned text for embedding

        end_time = time.time()
        duration = end_time - start_time

        result = {
            "duration": duration,
            "texts_count": len(texts),
            "texts_per_second": len(texts) / duration if duration > 0 else 0,
            "avg_time_per_text": duration / len(texts) if texts else 0,
        }

        logger.info(
            f"Text Processing: {len(texts)} texts in {duration:.2f}s = {result['texts_per_second']:.2f} texts/sec"
        )
        return result, processed_texts

    def benchmark_embedding_generation(self, texts: List[str]) -> Dict[str, Any]:
        """Benchmark embedding generation."""
        logger.info(f"Benchmarking embedding generation ({len(texts)} texts)...")

        # Initialize embedder if not done
        if self.embedder is None:
            self.initialize_embedder()

        start_time = time.time()

        # Generate embeddings
        embeddings = self.embedder.embed_texts(texts)

        end_time = time.time()
        duration = end_time - start_time

        result = {
            "duration": duration,
            "texts_count": len(texts),
            "texts_per_second": len(texts) / duration if duration > 0 else 0,
            "avg_time_per_text": duration / len(texts) if texts else 0,
            "embedding_dimension": embeddings[0].embedding.shape[0]
            if embeddings
            else 0,
        }

        logger.info(
            f"Embedding Generation: {len(texts)} texts in {duration:.2f}s = {result['texts_per_second']:.2f} texts/sec"
        )
        return result

    async def run_full_pipeline_benchmark(
        self, num_documents: int = 15
    ) -> Dict[str, Any]:
        """Run end-to-end pipeline benchmark."""
        logger.info(f"=== STARTING {self.device.upper()} FULL PIPELINE BENCHMARK ===")

        total_start = time.time()

        try:
            # Step 1: API Fetching
            api_result = await self.benchmark_api_fetching(num_documents)

            # Step 2: Prepare test texts (using realistic legal texts)
            test_texts = [
                "Ustawa z dnia 26 stycznia 1982 r. - Karta Nauczyciela wprowadza kompleksowe regulacje dotyczące praw i obowiązków nauczycieli w Polsce.",
                "Rozporządzenie Ministra Edukacji i Nauki w sprawie podstawy programowej wychowania przedszkolnego oraz podstawy programowej kształcenia ogólnego.",
                "Kodeks pracy określa podstawowe prawa i obowiązki pracowników oraz pracodawców, regulując stosunki prawne w zakresie zatrudnienia.",
                "Ustawa o podatku dochodowym od osób fizycznych wprowadza nowe regulacje dotyczące rozliczeń podatkowych dla różnych grup społecznych.",
                "Sejm Rzeczypospolitej Polskiej rozpatruje projekt ustawy o zmianie ustawy o systemie oświaty oraz niektórych innych ustaw.",
                "Konstytucja Rzeczypospolitej Polskiej z dnia 2 kwietnia 1997 r. jest najwyższym prawem Rzeczypospolitej Polskiej.",
                "Ustawa o ochronie danych osobowych implementuje przepisy rozporządzenia RODO w polskim systemie prawnym.",
                "Kodeks postępowania administracyjnego reguluje postępowanie przed organami administracji publicznej.",
                "Rozporządzenie Parlamentu Europejskiego i Rady w sprawie ochrony osób fizycznych w związku z przetwarzaniem danych osobowych.",
                "Ustawa o samorządzie gminnym określa ustrój i zadania gminy jako podstawowej jednostki samorządu terytorialnego.",
                "Prawo budowlane reguluje działalność obejmującą projektowanie, budowę, utrzymanie i rozbiórkę obiektów budowlanych.",
                "Ustawa o planowaniu i zagospodarowaniu przestrzennym określa zasady kształtowania polityki przestrzennej przez jednostki samorządu.",
                "Kodeks karny określa czyny zabronione pod groźbą kary oraz kary i środki karne za te czyny.",
                "Ustawa o postępowaniu w sprawach nieletnich reguluje postępowanie wobec nieletnich, którzy popełnili czyn karalny.",
                "Rozporządzenie w sprawie warunków technicznych, jakim powinny odpowiadać budynki i ich usytuowanie.",
            ] * 2  # 30 texts total

            # Limit to requested number
            test_texts = test_texts[
                : num_documents * 2
            ]  # 2 texts per document on average

            # Step 3: Text Processing
            processing_result, processed_texts = self.benchmark_text_processing(
                test_texts
            )

            # Step 4: Embedding Generation
            embedding_result = self.benchmark_embedding_generation(processed_texts)

            total_time = time.time() - total_start

            # Calculate overall metrics
            total_texts_processed = len(processed_texts)
            overall_throughput = (
                total_texts_processed / total_time if total_time > 0 else 0
            )

            result = {
                "device": self.device,
                "timestamp": datetime.now().isoformat(),
                "total_duration": total_time,
                "total_texts_processed": total_texts_processed,
                "overall_throughput": overall_throughput,
                "benchmarks": {
                    "api_fetching": api_result,
                    "text_processing": processing_result,
                    "embedding_generation": embedding_result,
                },
                "performance_breakdown": {
                    "api_time_percent": (
                        api_result["total_duration"] / total_time * 100
                    )
                    if total_time > 0
                    else 0,
                    "processing_time_percent": (
                        processing_result["duration"] / total_time * 100
                    )
                    if total_time > 0
                    else 0,
                    "embedding_time_percent": (
                        embedding_result["duration"] / total_time * 100
                    )
                    if total_time > 0
                    else 0,
                },
                "summary": {
                    "benchmark_completed": True,
                    "total_pipeline_time": total_time,
                    "total_throughput": overall_throughput,
                },
            }

            logger.info(f"✅ {self.device.upper()} pipeline benchmark completed")
            logger.info(
                f"Total: {total_texts_processed} texts in {total_time:.2f}s = {overall_throughput:.2f} texts/sec"
            )

            return result

        except Exception as e:
            logger.error(f"❌ {self.device.upper()} benchmark failed: {e}")
            return {
                "device": self.device,
                "error": str(e),
                "summary": {"benchmark_completed": False, "error": str(e)},
            }


async def run_comparative_benchmark():
    """Run comparative benchmark between GPU and CPU."""
    logger.info("=== FULL PIPELINE COMPARATIVE BENCHMARK: GPU vs CPU ===")

    results = {
        "benchmark_info": {
            "timestamp": datetime.now().isoformat(),
            "benchmark_type": "full_pipeline_comparative",
            "description": "Sejm-Whiz processing pipeline performance comparison (GPU vs CPU)",
        },
        "results": {},
    }

    # Run GPU benchmark
    logger.info("Starting GPU benchmark...")
    try:
        gpu_benchmark = SimplePipelineBenchmark("cuda")
        gpu_results = await gpu_benchmark.run_full_pipeline_benchmark(20)
        results["results"]["gpu"] = gpu_results
        logger.info("✅ GPU benchmark completed")
    except Exception as e:
        logger.error(f"❌ GPU benchmark failed: {e}")
        results["results"]["gpu"] = {
            "device": "cuda",
            "error": str(e),
            "summary": {"benchmark_completed": False, "error": str(e)},
        }

    # Run CPU benchmark
    logger.info("Starting CPU benchmark...")
    try:
        cpu_benchmark = SimplePipelineBenchmark("cpu")
        cpu_results = await cpu_benchmark.run_full_pipeline_benchmark(20)
        results["results"]["cpu"] = cpu_results
        logger.info("✅ CPU benchmark completed")
    except Exception as e:
        logger.error(f"❌ CPU benchmark failed: {e}")
        results["results"]["cpu"] = {
            "device": "cpu",
            "error": str(e),
            "summary": {"benchmark_completed": False, "error": str(e)},
        }

    # Calculate comparison metrics
    if results["results"].get("gpu", {}).get("summary", {}).get(
        "benchmark_completed"
    ) and results["results"].get("cpu", {}).get("summary", {}).get(
        "benchmark_completed"
    ):
        gpu_throughput = results["results"]["gpu"]["overall_throughput"]
        cpu_throughput = results["results"]["cpu"]["overall_throughput"]
        gpu_total_time = results["results"]["gpu"]["total_duration"]
        cpu_total_time = results["results"]["cpu"]["total_duration"]

        # Extract component timings
        gpu_embedding_time = results["results"]["gpu"]["benchmarks"][
            "embedding_generation"
        ]["duration"]
        cpu_embedding_time = results["results"]["cpu"]["benchmarks"][
            "embedding_generation"
        ]["duration"]

        gpu_processing_time = results["results"]["gpu"]["benchmarks"][
            "text_processing"
        ]["duration"]
        cpu_processing_time = results["results"]["cpu"]["benchmarks"][
            "text_processing"
        ]["duration"]

        speedup_overall = gpu_throughput / cpu_throughput if cpu_throughput > 0 else 0
        speedup_embedding = (
            cpu_embedding_time / gpu_embedding_time if gpu_embedding_time > 0 else 0
        )

        results["comparison"] = {
            "overall_performance": {
                "gpu_throughput": gpu_throughput,
                "cpu_throughput": cpu_throughput,
                "gpu_speedup": speedup_overall,
                "gpu_total_time": gpu_total_time,
                "cpu_total_time": cpu_total_time,
            },
            "embedding_performance": {
                "gpu_embedding_time": gpu_embedding_time,
                "cpu_embedding_time": cpu_embedding_time,
                "gpu_embedding_speedup": speedup_embedding,
            },
            "text_processing_performance": {
                "gpu_processing_time": gpu_processing_time,
                "cpu_processing_time": cpu_processing_time,
            },
            "recommendation": "GPU" if speedup_overall > 1.2 else "CPU",
            "cost_efficiency": "CPU" if speedup_overall < 2.0 else "GPU",
        }

        logger.info("=== COMPARATIVE BENCHMARK RESULTS ===")
        logger.info(f"GPU Overall Throughput: {gpu_throughput:.2f} texts/sec")
        logger.info(f"CPU Overall Throughput: {cpu_throughput:.2f} texts/sec")
        logger.info(f"GPU Overall Speedup: {speedup_overall:.2f}x")
        logger.info(f"GPU Embedding Speedup: {speedup_embedding:.2f}x")
        logger.info(
            f"Performance Recommendation: {results['comparison']['recommendation']}"
        )
        logger.info(
            f"Cost Efficiency Recommendation: {results['comparison']['cost_efficiency']}"
        )

    return results


if __name__ == "__main__":
    results = asyncio.run(run_comparative_benchmark())

    # Save results to file
    import json

    timestamp = int(time.time())
    filename = f"full_pipeline_benchmark_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"✅ Full pipeline benchmark completed. Results saved to {filename}")

    # Print summary
    if "comparison" in results:
        print("\n" + "=" * 60)
        print("FULL PIPELINE BENCHMARK SUMMARY")
        print("=" * 60)
        comp = results["comparison"]
        print(
            f"GPU Overall Throughput: {comp['overall_performance']['gpu_throughput']:.2f} texts/sec"
        )
        print(
            f"CPU Overall Throughput: {comp['overall_performance']['cpu_throughput']:.2f} texts/sec"
        )
        print(f"GPU Overall Speedup: {comp['overall_performance']['gpu_speedup']:.2f}x")
        print(
            f"GPU Embedding Speedup: {comp['embedding_performance']['gpu_embedding_speedup']:.2f}x"
        )
        print(f"Performance Recommendation: {comp['recommendation']}")
        print(f"Cost Efficiency: {comp['cost_efficiency']}")
        print("=" * 60)
