#!/usr/bin/env python3
"""
Demo script to show Semantic Search functionality
"""

import requests
import sys


def test_search_endpoint(base_url: str = "http://localhost:8000") -> None:
    """Test the semantic search endpoint with sample queries."""

    print("🔍 Semantic Search Demo")
    print("=" * 50)

    # Test queries in Polish (legal domain)
    test_queries = [
        "konstytucja",  # constitution
        "kodeks karny",  # criminal code
        "prawo pracy",  # labor law
        "ustawa o podatkach",  # tax law
        "rozporządzenie ministra",  # ministerial regulation
    ]

    for query in test_queries:
        print(f"\n🔎 Searching for: '{query}'")
        print("-" * 30)

        try:
            # Test GET endpoint
            response = requests.get(
                f"{base_url}/api/v1/search",
                params={"q": str(query), "limit": str(3), "threshold": str(0.3)},
                timeout=10,
            )

            if response.status_code == 200:
                results = response.json()
                print(f"✅ Found {results.get('total_results', 0)} results")
                print(
                    f"⏱️ Processing time: {results.get('processing_time_ms', 0):.1f}ms"
                )

                for i, result in enumerate(results.get("results", []), 1):
                    print(f"\n  {i}. {result.get('title', 'No title')[:60]}...")
                    print(f"     Type: {result.get('document_type', 'unknown')}")
                    print(f"     Similarity: {result.get('similarity_score', 0):.3f}")
                    print(f"     Content: {result.get('content', '')[:100]}...")

            elif response.status_code == 503:
                print("⚠️  Search service unavailable (components not loaded)")
                print(
                    "    This means the ML components are not available in the current deployment"
                )

            elif response.status_code == 404:
                print("❌ Search endpoint not found")
                print("    The API server might not have the search endpoint deployed")

            else:
                print(f"❌ Request failed with status {response.status_code}")
                print(f"   Response: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")
            continue

    # Test POST endpoint
    print("\n🔎 Testing POST endpoint with structured request")
    print("-" * 50)

    try:
        post_data = {
            "query": "ustawa o ochronie danych osobowych",  # GDPR law
            "limit": 2,
            "threshold": 0.4,
            "document_type": "ustawa",  # law type filter
        }

        response = requests.post(
            f"{base_url}/api/v1/search",
            json=post_data,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        if response.status_code == 200:
            results = response.json()
            print("✅ POST request successful")
            print(f"📊 Query: {results.get('query')}")
            print(f"📈 Results: {results.get('total_results')}")
            print(f"⏱️ Time: {results.get('processing_time_ms'):.1f}ms")
        else:
            print(f"❌ POST failed: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ POST connection error: {e}")


def test_health_endpoint(base_url: str = "http://localhost:8000") -> bool:
    """Test if the API server is running."""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API Server is healthy")
            print(f"   Status: {health_data.get('status')}")
            print(f"   Version: {health_data.get('version')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API server: {e}")
        return False


def main():
    """Main demo function."""
    # Default to deployed instance on p7
    base_url = "http://p7:8001"

    if len(sys.argv) > 1:
        base_url = sys.argv[1]

    print(f"🚀 Testing Semantic Search API at: {base_url}")
    print("=" * 60)

    # Test health first
    if not test_health_endpoint(base_url):
        print("\n⚠️  Trying localhost as fallback...")
        base_url = "http://localhost:8000"
        if not test_health_endpoint(base_url):
            print("\n❌ No API server available. Please start the server first.")
            print("\nTo start the server:")
            print("  cd projects/api_server")
            print("  PYTHONPATH='../../components:../../bases' uv run python main.py")
            return

    # Run search tests
    test_search_endpoint(base_url)

    print("\n" + "=" * 60)
    print("🎯 Demo complete!")
    print("\nThe semantic search endpoint provides:")
    print("  • GET /api/v1/search?q=<query>&limit=<n>&threshold=<score>")
    print("  • POST /api/v1/search with JSON body")
    print("  • Semantic similarity using HerBERT embeddings")
    print("  • Filtering by document type and similarity threshold")
    print("  • Processing time measurement")


if __name__ == "__main__":
    main()
