#!/usr/bin/env python3
"""Test script for semantic search functionality with p7 database."""

import asyncio
import sys
from pathlib import Path

# Add project components to path
sys.path.insert(0, str(Path(__file__).parent / "components"))
sys.path.insert(0, str(Path(__file__).parent / "bases"))

from sejm_whiz.semantic_search import get_semantic_search_service
from sejm_whiz.database.operations import get_db_session
from sejm_whiz.database.models import LegalDocument


async def test_search_functionality():
    """Test semantic search with real data from p7."""

    print("🔍 Testing Semantic Search with Real Polish Parliamentary Data")
    print("=" * 60)

    try:
        # Check database contents first
        with get_db_session() as session:
            doc_count = session.query(LegalDocument).count()
            print(f"📊 Database contains {doc_count} documents")

            if doc_count == 0:
                print(
                    "❌ No documents found in database. Please run the data ingestion first."
                )
                return

        # Initialize search service
        print("\n🚀 Initializing semantic search service...")
        search_service = get_semantic_search_service()

        # Test queries in Polish
        test_queries = [
            "konstytucja",
            "prawo pracy",
            "podatki",
            "edukacja",
            "ochrona środowiska",
            "ustawa budżetowa",
            "procedura głosowania",
            "komisja sejmowa",
        ]

        print("\n🔍 Testing search queries...")

        for query in test_queries:
            print(f"\n--- Searching for: '{query}' ---")

            try:
                # Perform search directly (skip query processing for now)
                print(f"Searching for: {query}")

                results = search_service.search_documents(
                    query=query,
                    limit=3,
                    similarity_threshold=0.1,  # Lower threshold for more results
                    include_cross_register=True,
                )

                print(f"Found {len(results)} results:")

                for i, result in enumerate(results, 1):
                    print(f"\n  {i}. {result.document.title[:80]}...")
                    print(f"     Type: {result.document.document_type}")
                    print(f"     Similarity: {result.similarity_score:.3f}")

                    if result.search_metadata.get("cross_register_matches"):
                        print(
                            f"     Cross-register matches: {len(result.search_metadata['cross_register_matches'])}"
                        )

                if not results:
                    print("  No results found (similarity threshold may be too high)")

            except Exception as e:
                print(f"  ❌ Search failed: {e}")

        # Test statistics
        print("\n📈 Search System Statistics:")
        try:
            stats = search_service.get_search_statistics()
            print(
                f"  Documents indexed: {stats.get('indexing', {}).get('total_indexed', 'Unknown')}"
            )
            print(
                f"  Embedding model: {stats.get('search_engine', {}).get('embedding_model', 'Unknown')}"
            )
            print(
                f"  Ranking strategy: {stats.get('ranking', {}).get('strategy', 'Unknown')}"
            )
        except Exception as e:
            print(f"  ❌ Could not get statistics: {e}")

        print("\n✅ Search functionality test completed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_search_functionality())
