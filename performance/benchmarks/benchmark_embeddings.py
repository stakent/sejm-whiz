#!/usr/bin/env python3
import time
import logging
import os
from sejm_whiz.embeddings.herbert_embedder import HerBERTEmbedder
from sejm_whiz.embeddings.config import EmbeddingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_embeddings():
    # Test data - various sizes
    test_texts = {
        "small": ["Ustawa o podatkach"] * 10,  # 10 short texts
        "medium": [
            "Ustawa o podatku dochodowym od osób fizycznych wprowadza nowe regulacje dotyczące rozliczeń podatkowych"
        ]
        * 50,  # 50 medium texts
        "large": [
            "Ustawa o podatku dochodowym od osób fizycznych wprowadza nowe regulacje dotyczące rozliczeń podatkowych dla przedsiębiorców i osób prowadzących działalność gospodarczą w Polsce"
        ]
        * 100,  # 100 longer texts
        "xlarge": [
            "Sejm Rzeczypospolitej Polskiej rozpatruje projekt ustawy o zmianie ustawy o podatku dochodowym od osób fizycznych oraz niektórych innych ustaw, który wprowadza istotne zmiany w systemie podatkowym"
        ]
        * 500,  # 500 long texts
    }

    results = {}

    # Test GPU first
    logger.info("=== GPU BENCHMARK ===")
    try:
        # Set CUDA device
        os.environ["EMBEDDING_DEVICE"] = "cuda"
        config_gpu = EmbeddingConfig()
        embedder_gpu = HerBERTEmbedder(config=config_gpu)

        for size, texts in test_texts.items():
            logger.info(f"Testing {size} batch with {len(texts)} texts...")

            start_time = time.time()
            embeddings = embedder_gpu.embed_texts(texts)
            end_time = time.time()

            duration = end_time - start_time
            texts_per_second = len(texts) / duration

            results[f"gpu_{size}"] = {
                "texts": len(texts),
                "duration": duration,
                "texts_per_second": texts_per_second,
                "embedding_shape": embeddings[0].embedding.shape
                if embeddings
                else None,
            }

            logger.info(
                f"GPU {size}: {len(texts)} texts in {duration:.2f}s = {texts_per_second:.2f} texts/sec"
            )

    except Exception as e:
        logger.error(f"GPU benchmark failed: {e}")
        results["gpu_error"] = str(e)

    # Test CPU
    logger.info("=== CPU BENCHMARK ===")
    try:
        # Set CPU device
        os.environ["EMBEDDING_DEVICE"] = "cpu"
        config_cpu = EmbeddingConfig()
        embedder_cpu = HerBERTEmbedder(config=config_cpu)

        # For CPU, test smaller batches to avoid excessive wait times
        cpu_test_texts = {
            "small": test_texts["small"],
            "medium": test_texts["medium"][:25],  # Reduce to 25 for CPU
            "large": test_texts["large"][:50],  # Reduce to 50 for CPU
        }

        for size, texts in cpu_test_texts.items():
            logger.info(f"Testing {size} batch with {len(texts)} texts...")

            start_time = time.time()
            embeddings = embedder_cpu.embed_texts(texts)
            end_time = time.time()

            duration = end_time - start_time
            texts_per_second = len(texts) / duration

            results[f"cpu_{size}"] = {
                "texts": len(texts),
                "duration": duration,
                "texts_per_second": texts_per_second,
                "embedding_shape": embeddings[0].embedding.shape
                if embeddings
                else None,
            }

            logger.info(
                f"CPU {size}: {len(texts)} texts in {duration:.2f}s = {texts_per_second:.2f} texts/sec"
            )

    except Exception as e:
        logger.error(f"CPU benchmark failed: {e}")
        results["cpu_error"] = str(e)

    # Print summary
    logger.info("=== PERFORMANCE SUMMARY ===")
    for key, result in results.items():
        if "error" not in key:
            logger.info(f"{key}: {result['texts_per_second']:.2f} texts/second")

    # Calculate GPU speedup
    if "gpu_medium" in results and "cpu_medium" in results:
        speedup = (
            results["gpu_medium"]["texts_per_second"]
            / results["cpu_medium"]["texts_per_second"]
        )
        logger.info(f"GPU is {speedup:.1f}x faster than CPU for medium batches")

    return results


if __name__ == "__main__":
    benchmark_embeddings()
