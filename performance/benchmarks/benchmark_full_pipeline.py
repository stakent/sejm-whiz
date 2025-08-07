#!/usr/bin/env python3
"""
Full Pipeline Benchmark: GPU vs CPU Performance Analysis
Measures complete Sejm-Whiz processing workflow including:
- Sejm API calls
- ELI API calls
- Text processing
- Embedding generation
- Database storage
"""

import time
import logging
import os
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict, List, Any

# Set up environment before imports
os.environ.setdefault("DATABASE_HOST", "p7")
os.environ.setdefault("DATABASE_PORT", "5432")
os.environ.setdefault("DATABASE_NAME", "sejm_whiz")
os.environ.setdefault("DATABASE_USER", "sejm_whiz_user")
os.environ.setdefault("DATABASE_PASSWORD", "sejm_whiz_password")
os.environ.setdefault("DATABASE_SSL_MODE", "disable")
os.environ.setdefault("REDIS_HOST", "p7")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("DEPLOYMENT_ENV", "p7")

# Import components
from sejm_whiz.sejm_api.client import SejmApiClient
from sejm_whiz.eli_api.client import EliApiClient
from sejm_whiz.text_processing.cleaner import TextCleaner
from sejm_whiz.text_processing.legal_parser import PolishLegalParser
from sejm_whiz.embeddings.herbert_embedder import HerBERTEmbedder
from sejm_whiz.embeddings.config import EmbeddingConfig
from sejm_whiz.vector_db.operations import VectorDBOperations
from sejm_whiz.database.operations import DocumentOperations
from sejm_whiz.document_ingestion.ingestion_pipeline import DocumentIngestionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PipelineBenchmark:
    """Full pipeline benchmark for GPU vs CPU performance comparison."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.results = {}

        # Set device environment variable
        os.environ["EMBEDDING_DEVICE"] = device

        # Initialize components
        self.sejm_client = None
        self.eli_client = None
        self.text_cleaner = TextCleaner()
        self.legal_parser = PolishLegalParser()
        self.embedder = None
        self.vector_db = None
        self.doc_ops = None
        self.pipeline = None

    async def initialize_components(self):
        """Initialize all pipeline components."""
        logger.info(f"Initializing components for {self.device.upper()} benchmark...")

        # Initialize API clients
        self.sejm_client = SejmApiClient()
        self.eli_client = EliApiClient()

        # Initialize embedding components
        config = EmbeddingConfig()
        self.embedder = HerBERTEmbedder(config=config)

        # Initialize database components
        self.vector_db = VectorDBOperations()
        from sejm_whiz.database import get_database_config
        db_config = get_database_config()
        self.doc_ops = DocumentOperations(db_config)

        # Initialize pipeline
        self.pipeline = DocumentIngestionPipeline()

        logger.info(f"✅ All components initialized for {self.device.upper()}")

    async def cleanup_components(self):
        """Clean up resources."""
        if self.sejm_client:
            await self.sejm_client.close()
        if self.eli_client:
            await self.eli_client.close()
        if self.vector_db:
            await self.vector_db.close()
        if self.doc_ops:
            await self.doc_ops.close()

    @asynccontextmanager
    async def benchmark_context(self):
        """Context manager for benchmark setup and cleanup."""
        await self.initialize_components()
        try:
            yield self
        finally:
            await self.cleanup_components()

    async def benchmark_sejm_api_calls(
        self, num_proceedings: int = 5
    ) -> Dict[str, Any]:
        """Benchmark Sejm API calls for fetching proceedings."""
        logger.info(
            f"Benchmarking Sejm API calls (fetching {num_proceedings} proceedings)..."
        )

        start_time = time.time()

        # Fetch recent proceedings
        async with self.sejm_client as client:
            proceedings = await client.get_recent_proceedings(limit=num_proceedings)

        end_time = time.time()
        duration = end_time - start_time

        result = {
            "duration": duration,
            "proceedings_count": len(proceedings),
            "proceedings_per_second": len(proceedings) / duration
            if duration > 0
            else 0,
            "avg_time_per_proceeding": duration / len(proceedings)
            if proceedings
            else 0,
        }

        logger.info(
            f"Sejm API: {len(proceedings)} proceedings in {duration:.2f}s = {result['proceedings_per_second']:.2f} proceedings/sec"
        )
        return result

    async def benchmark_eli_api_calls(self, num_documents: int = 10) -> Dict[str, Any]:
        """Benchmark ELI API calls for fetching legal documents."""
        logger.info(
            f"Benchmarking ELI API calls (fetching {num_documents} documents)..."
        )

        start_time = time.time()

        # Search for documents
        async with self.eli_client as client:
            search_result = await client.search_documents(limit=num_documents)
            documents = search_result.documents

        end_time = time.time()
        duration = end_time - start_time

        result = {
            "duration": duration,
            "documents_count": len(documents),
            "documents_per_second": len(documents) / duration if duration > 0 else 0,
            "avg_time_per_document": duration / len(documents) if documents else 0,
        }

        logger.info(
            f"ELI API: {len(documents)} documents in {duration:.2f}s = {result['documents_per_second']:.2f} documents/sec"
        )
        return result

    async def benchmark_text_processing(self, texts: List[str]) -> Dict[str, Any]:
        """Benchmark text cleaning and legal parsing."""
        logger.info(f"Benchmarking text processing ({len(texts)} texts)...")

        start_time = time.time()

        # Clean and parse texts
        processed_texts = []
        for text in texts:
            # Clean text
            cleaned = self.text_cleaner.clean_text(text)

            # Parse legal structure
            parsed = self.legal_parser.parse_document_structure(cleaned)
            processed_texts.append(parsed)

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
        return result

    async def benchmark_embedding_generation(self, texts: List[str]) -> Dict[str, Any]:
        """Benchmark embedding generation."""
        logger.info(f"Benchmarking embedding generation ({len(texts)} texts)...")

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

    async def benchmark_database_storage(
        self, num_operations: int = 20
    ) -> Dict[str, Any]:
        """Benchmark database storage operations."""
        logger.info(f"Benchmarking database storage ({num_operations} operations)...")

        start_time = time.time()

        # Create test data
        test_documents = []
        test_embeddings = []

        for i in range(num_operations):
            # Create test document
            doc_data = {
                "eli_id": f"test/benchmark/{self.device}/{int(time.time())}/{i}",
                "title": f"Test Document {i} for {self.device.upper()} Benchmark",
                "content": f"This is test content for document {i} in the {self.device} benchmark. "
                * 10,
                "document_type": "test",
                "publisher": "benchmark",
                "published_date": datetime.now(),
            }

            # Store document
            doc_id = await self.doc_ops.store_document(doc_data)
            test_documents.append(doc_id)

            # Generate and store embedding
            text = doc_data["content"]
            embeddings = self.embedder.embed_texts([text])

            if embeddings:
                embedding_data = {
                    "document_id": doc_id,
                    "embedding": embeddings[0].embedding.tolist(),
                    "model_name": "herbert-base-cased",
                    "embedding_type": "document",
                }
                await self.vector_db.store_embedding(embedding_data)
                test_embeddings.append(embedding_data)

        end_time = time.time()
        duration = end_time - start_time

        # Cleanup test data
        for doc_id in test_documents:
            try:
                await self.doc_ops.delete_document(doc_id)
            except Exception as e:
                logger.warning(f"Failed to cleanup test document {doc_id}: {e}")

        result = {
            "duration": duration,
            "operations_count": num_operations,
            "operations_per_second": num_operations / duration if duration > 0 else 0,
            "avg_time_per_operation": duration / num_operations
            if num_operations > 0
            else 0,
        }

        logger.info(
            f"Database Storage: {num_operations} operations in {duration:.2f}s = {result['operations_per_second']:.2f} ops/sec"
        )
        return result

    async def benchmark_full_pipeline(self, num_documents: int = 10) -> Dict[str, Any]:
        """Benchmark the complete end-to-end pipeline."""
        logger.info(f"Benchmarking full pipeline ({num_documents} documents)...")

        start_time = time.time()

        # Step 1: Fetch Sejm proceedings
        proceedings_start = time.time()
        async with self.sejm_client as client:
            proceedings = await client.get_recent_proceedings(
                limit=min(num_documents, 5)
            )
        proceedings_time = time.time() - proceedings_start

        # Step 2: Fetch ELI documents
        eli_start = time.time()
        async with self.eli_client as client:
            search_result = await client.search_documents(limit=num_documents)
            documents = search_result.documents[:num_documents]
        eli_time = time.time() - eli_start

        # Step 3: Process texts
        processing_start = time.time()
        all_texts = []

        # Extract texts from proceedings
        for proceeding in proceedings:
            if hasattr(proceeding, "title") and proceeding.title:
                all_texts.append(proceeding.title)

        # Extract texts from documents
        for doc in documents:
            if hasattr(doc, "title") and doc.title:
                all_texts.append(doc.title)
            if hasattr(doc, "content") and doc.content:
                # Truncate content for benchmark
                content_sample = (
                    doc.content[:500] if len(doc.content) > 500 else doc.content
                )
                all_texts.append(content_sample)

        # Limit texts for benchmark
        all_texts = all_texts[: num_documents * 2]  # 2 texts per document on average

        # Clean and parse texts
        processed_texts = []
        for text in all_texts:
            cleaned = self.text_cleaner.clean_text(text)
            parsed = self.legal_parser.parse_document_structure(cleaned)
            # Extract text content from parsed structure
            if isinstance(parsed, dict):
                processed_texts.append(
                    cleaned
                )  # Use cleaned text since parsed is a structure dict
            else:
                processed_texts.append(str(parsed))

        processing_time = time.time() - processing_start

        # Step 4: Generate embeddings
        embedding_start = time.time()
        embeddings = self.embedder.embed_texts(processed_texts)
        embedding_time = time.time() - embedding_start

        # Step 5: Store in database (sample only for benchmark)
        storage_start = time.time()
        stored_count = 0

        for i, (text, embedding) in enumerate(
            zip(processed_texts[:5], embeddings[:5])
        ):  # Store only first 5
            try:
                # Store document
                doc_data = {
                    "eli_id": f"benchmark/{self.device}/{int(time.time())}/{i}",
                    "title": f"Benchmark Document {i}",
                    "content": text,
                    "document_type": "benchmark",
                    "publisher": "test",
                    "published_date": datetime.now(),
                }

                doc_id = await self.doc_ops.store_document(doc_data)

                # Store embedding
                embedding_data = {
                    "document_id": doc_id,
                    "embedding": embedding.embedding.tolist(),
                    "model_name": "herbert-base-cased",
                    "embedding_type": "document",
                }
                await self.vector_db.store_embedding(embedding_data)
                stored_count += 1

                # Cleanup immediately
                await self.doc_ops.delete_document(doc_id)

            except Exception as e:
                logger.warning(f"Failed to store benchmark document {i}: {e}")

        storage_time = time.time() - storage_start

        end_time = time.time()
        total_duration = end_time - start_time

        result = {
            "total_duration": total_duration,
            "documents_processed": len(all_texts),
            "documents_per_second": len(all_texts) / total_duration
            if total_duration > 0
            else 0,
            "breakdown": {
                "sejm_api_time": proceedings_time,
                "eli_api_time": eli_time,
                "text_processing_time": processing_time,
                "embedding_generation_time": embedding_time,
                "database_storage_time": storage_time,
            },
            "performance_metrics": {
                "sejm_proceedings_fetched": len(proceedings),
                "eli_documents_fetched": len(documents),
                "texts_processed": len(processed_texts),
                "embeddings_generated": len(embeddings),
                "documents_stored": stored_count,
            },
        }

        logger.info(
            f"Full Pipeline: {len(all_texts)} texts in {total_duration:.2f}s = {result['documents_per_second']:.2f} docs/sec"
        )
        return result

    async def run_benchmark_suite(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        logger.info(f"=== STARTING {self.device.upper()} BENCHMARK SUITE ===")

        async with self.benchmark_context():
            results = {
                "device": self.device,
                "timestamp": datetime.now().isoformat(),
                "benchmarks": {},
            }

            try:
                # Benchmark individual components
                logger.info("Running component benchmarks...")

                # Test texts for component benchmarks
                test_texts = [
                    "Ustawa o podatku dochodowym od osób fizycznych wprowadza nowe regulacje dotyczące rozliczeń podatkowych.",
                    "Sejm Rzeczypospolitej Polskiej rozpatruje projekt ustawy o zmianie ustawy o podatku dochodowym.",
                    "Rozporządzenie Ministra Finansów w sprawie sposobu prowadzenia księgi przychodów i rozchodów.",
                    "Kodeks pracy określa podstawowe prawa i obowiązki pracowników oraz pracodawców.",
                    "Ustawa o ochronie danych osobowych implementuje przepisy RODO w polskim systemie prawnym.",
                ] * 4  # 20 texts total

                # Component benchmarks
                results["benchmarks"]["sejm_api"] = await self.benchmark_sejm_api_calls(
                    5
                )
                results["benchmarks"]["eli_api"] = await self.benchmark_eli_api_calls(
                    10
                )
                results["benchmarks"][
                    "text_processing"
                ] = await self.benchmark_text_processing(test_texts)
                results["benchmarks"][
                    "embedding_generation"
                ] = await self.benchmark_embedding_generation(test_texts)
                results["benchmarks"][
                    "database_storage"
                ] = await self.benchmark_database_storage(10)

                # Full pipeline benchmark
                logger.info("Running full pipeline benchmark...")
                results["benchmarks"][
                    "full_pipeline"
                ] = await self.benchmark_full_pipeline(15)

                # Calculate summary metrics
                total_pipeline_time = results["benchmarks"]["full_pipeline"][
                    "total_duration"
                ]
                total_throughput = results["benchmarks"]["full_pipeline"][
                    "documents_per_second"
                ]

                results["summary"] = {
                    "total_pipeline_time": total_pipeline_time,
                    "total_throughput": total_throughput,
                    "device_performance": self.device.upper(),
                    "benchmark_completed": True,
                }

                logger.info(
                    f"✅ {self.device.upper()} benchmark completed successfully"
                )

            except Exception as e:
                logger.error(f"❌ {self.device.upper()} benchmark failed: {e}")
                results["error"] = str(e)
                results["summary"] = {"benchmark_completed": False, "error": str(e)}

        return results


async def run_full_pipeline_benchmarks():
    """Run full pipeline benchmarks for both GPU and CPU."""
    logger.info("=== FULL PIPELINE BENCHMARK: GPU vs CPU ===")

    all_results = {
        "benchmark_info": {
            "timestamp": datetime.now().isoformat(),
            "benchmark_type": "full_pipeline",
            "description": "Complete Sejm-Whiz processing pipeline performance comparison",
        },
        "results": {},
    }

    # Test GPU first
    logger.info("Starting GPU benchmark...")
    try:
        gpu_benchmark = PipelineBenchmark("cuda")
        gpu_results = await gpu_benchmark.run_benchmark_suite()
        all_results["results"]["gpu"] = gpu_results
        logger.info("✅ GPU benchmark completed")
    except Exception as e:
        logger.error(f"❌ GPU benchmark failed: {e}")
        all_results["results"]["gpu"] = {
            "device": "cuda",
            "error": str(e),
            "benchmark_completed": False,
        }

    # Test CPU
    logger.info("Starting CPU benchmark...")
    try:
        cpu_benchmark = PipelineBenchmark("cpu")
        cpu_results = await cpu_benchmark.run_benchmark_suite()
        all_results["results"]["cpu"] = cpu_results
        logger.info("✅ CPU benchmark completed")
    except Exception as e:
        logger.error(f"❌ CPU benchmark failed: {e}")
        all_results["results"]["cpu"] = {
            "device": "cpu",
            "error": str(e),
            "benchmark_completed": False,
        }

    # Calculate comparison metrics
    if (
        "gpu" in all_results["results"]
        and "cpu" in all_results["results"]
        and all_results["results"]["gpu"].get("summary", {}).get("benchmark_completed")
        and all_results["results"]["cpu"].get("summary", {}).get("benchmark_completed")
    ):
        gpu_throughput = all_results["results"]["gpu"]["summary"]["total_throughput"]
        cpu_throughput = all_results["results"]["cpu"]["summary"]["total_throughput"]
        gpu_total_time = all_results["results"]["gpu"]["summary"]["total_pipeline_time"]
        cpu_total_time = all_results["results"]["cpu"]["summary"]["total_pipeline_time"]

        speedup = gpu_throughput / cpu_throughput if cpu_throughput > 0 else 0
        time_ratio = cpu_total_time / gpu_total_time if gpu_total_time > 0 else 0

        all_results["comparison"] = {
            "gpu_throughput": gpu_throughput,
            "cpu_throughput": cpu_throughput,
            "gpu_speedup": speedup,
            "time_ratio": time_ratio,
            "gpu_total_time": gpu_total_time,
            "cpu_total_time": cpu_total_time,
            "recommendation": "GPU" if speedup > 1.2 else "CPU",
        }

        logger.info("=== BENCHMARK COMPARISON SUMMARY ===")
        logger.info(f"GPU Throughput: {gpu_throughput:.2f} docs/sec")
        logger.info(f"CPU Throughput: {cpu_throughput:.2f} docs/sec")
        logger.info(f"GPU Speedup: {speedup:.2f}x")
        logger.info(f"Recommendation: {all_results['comparison']['recommendation']}")

    return all_results


if __name__ == "__main__":
    results = asyncio.run(run_full_pipeline_benchmarks())

    # Save results to file
    import json

    with open(f"full_pipeline_benchmark_results_{int(time.time())}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Full pipeline benchmark completed. Results saved to file.")
