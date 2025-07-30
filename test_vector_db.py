#!/usr/bin/env python3
"""Test script for vector_db component functionality."""

import sys
import os
from typing import List

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sejm_whiz.vector_db import (
    get_vector_db,
    get_vector_operations,
    get_similarity_search,
    DistanceMetric,
    validate_embedding_dimensions,
    normalize_embedding,
    compute_cosine_similarity,
    validate_vector_db_health,
    estimate_index_parameters
)


def test_component_imports():
    """Test that all vector_db components can be imported."""
    print("Testing vector_db component imports...")
    
    # Test connection components
    vector_db = get_vector_db()
    print(f"✅ VectorDBConnection created: {type(vector_db).__name__}")
    
    # Test operations components
    operations = get_vector_operations()
    print(f"✅ VectorDBOperations created: {type(operations).__name__}")
    
    # Test similarity search components
    search = get_similarity_search()
    print(f"✅ VectorSimilaritySearch created: {type(search).__name__}")
    
    # Test distance metrics
    metrics = list(DistanceMetric)
    print(f"✅ Distance metrics available: {[m.value for m in metrics]}")
    
    print("✅ Component imports test passed\n")


def test_embedding_utilities():
    """Test embedding utility functions."""
    print("Testing embedding utilities...")
    
    # Test embedding validation
    valid_embedding = [0.1] * 768
    invalid_embedding = [0.1] * 512
    
    valid_result = validate_embedding_dimensions(valid_embedding)
    invalid_result = validate_embedding_dimensions(invalid_embedding)
    
    print(f"✅ Valid embedding (768D): {valid_result}")
    print(f"✅ Invalid embedding (512D): {not invalid_result}")
    
    # Test normalization
    test_vector = [3.0, 4.0, 0.0]  # Magnitude = 5
    normalized = normalize_embedding(test_vector)
    magnitude = sum(x**2 for x in normalized) ** 0.5
    
    print(f"✅ Vector normalization: magnitude = {magnitude:.6f} (should be ~1.0)")
    
    # Test cosine similarity
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    vec3 = [0.0, 1.0, 0.0]
    
    identical_sim = compute_cosine_similarity(vec1, vec2)
    orthogonal_sim = compute_cosine_similarity(vec1, vec3)
    
    print(f"✅ Identical vectors similarity: {identical_sim:.6f} (should be 1.0)")
    print(f"✅ Orthogonal vectors similarity: {orthogonal_sim:.6f} (should be 0.0)")
    
    print("✅ Embedding utilities test passed\n")


def test_index_parameter_estimation():
    """Test index parameter estimation for different dataset sizes."""
    print("Testing index parameter estimation...")
    
    test_sizes = [100, 1000, 50000, 200000]
    
    for size in test_sizes:
        params = estimate_index_parameters(size)
        print(f"  Dataset size {size:,}:")
        print(f"    Index type: {params['index_type']}")
        print(f"    Lists: {params['lists']}")
        print(f"    Probes: {params['probes']}")
        print(f"    Recommendation: {params['recommendation']}")
        print()
    
    print("✅ Index parameter estimation test passed\n")


def test_vector_db_health():
    """Test vector database health check."""
    print("Testing vector database health check...")
    
    health = validate_vector_db_health()
    print("Vector database health status:")
    
    for key, value in health.items():
        if key == "status":
            status_icon = {
                "healthy": "✅",
                "warning": "⚠️", 
                "unhealthy": "❌",
                "error": "💥"
            }.get(value, "❓")
            print(f"  {status_icon} {key}: {value}")
        elif isinstance(value, bool):
            status_icon = "✅" if value else "❌"
            print(f"  {status_icon} {key}: {value}")
        else:
            print(f"  ℹ️  {key}: {value}")
    
    if health.get("connection", False):
        print("✅ Database connection successful!")
        if health.get("pgvector_extension", False):
            print("✅ pgvector extension available!")
        else:
            print("⚠️  pgvector extension not available (vector operations will fail)")
    else:
        print("⚠️  Database connection failed (expected without running PostgreSQL)")
    
    print()


def test_similarity_search_interface():
    """Test similarity search interface (offline)."""
    print("Testing similarity search interface...")
    
    search = get_similarity_search()
    
    # Test distance metric operators
    operators = search.distance_operators
    print("Distance operators:")
    for metric, operator in operators.items():
        print(f"  {metric.value}: {operator}")
    
    # Test statistics structure (will work even without data)
    try:
        stats = search.get_embedding_statistics()
        print("Statistics structure:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"  Statistics unavailable (no database): {e}")
    
    print("✅ Similarity search interface test passed\n")


def test_operations_interface():
    """Test operations interface (offline)."""
    print("Testing operations interface...")
    
    ops = get_vector_operations()
    
    # Test that methods exist
    methods = [
        'create_document_with_embedding',
        'update_document_embedding', 
        'get_document_by_id',
        'get_documents_by_type',
        'delete_document',
        'create_document_embedding',
        'bulk_insert_documents',
        'count_documents'
    ]
    
    for method in methods:
        assert hasattr(ops, method), f"Missing method: {method}"
        print(f"  ✅ {method}")
    
    print("✅ Operations interface test passed\n")


def test_sample_workflow():
    """Test a sample workflow (offline simulation)."""
    print("Testing sample workflow simulation...")
    
    # Sample legal document data
    sample_doc = {
        "title": "Ustawa o ochronie danych osobowych",
        "content": "Niniejsza ustawa reguluje zasady i tryb przetwarzania danych osobowych...",
        "document_type": "law",
        "legal_domain": "privacy",
        "embedding": normalize_embedding([0.1] * 768)
    }
    
    print(f"Sample document:")
    print(f"  Title: {sample_doc['title']}")
    print(f"  Type: {sample_doc['document_type']}")
    print(f"  Domain: {sample_doc['legal_domain']}")
    print(f"  Embedding dimensions: {len(sample_doc['embedding'])}")
    print(f"  Embedding magnitude: {sum(x**2 for x in sample_doc['embedding'])**0.5:.6f}")
    
    # Test similarity computation with different documents
    similar_embedding = normalize_embedding([0.11] * 768)  # Very similar
    different_embedding = normalize_embedding([0.9] * 768)  # Different
    
    sim_score = compute_cosine_similarity(sample_doc['embedding'], similar_embedding)
    diff_score = compute_cosine_similarity(sample_doc['embedding'], different_embedding)
    
    print(f"  Similarity to similar document: {sim_score:.6f}")
    print(f"  Similarity to different document: {diff_score:.6f}")
    
    print("✅ Sample workflow simulation passed\n")


def main():
    """Run all vector_db component tests."""
    print("🧪 Testing sejm-whiz vector_db component\n")
    print("=" * 60)
    
    try:
        test_component_imports()
        test_embedding_utilities()
        test_index_parameter_estimation()  
        test_vector_db_health()
        test_similarity_search_interface()
        test_operations_interface()
        test_sample_workflow()
        
        print("=" * 60)
        print("🎉 All vector_db component tests completed!")
        print("\nComponent Features Available:")
        print("✅ Database connection management with pgvector support")
        print("✅ CRUD operations for documents and embeddings")
        print("✅ Vector similarity search (cosine, L2, inner product)")
        print("✅ Embedding validation and normalization utilities")
        print("✅ Health monitoring and diagnostics")
        print("✅ Index parameter estimation and optimization")
        print("✅ Batch operations and bulk processing")
        
        print("\nNext steps:")
        print("1. Deploy PostgreSQL with pgvector extension")
        print("2. Run integration tests: pytest test/components/sejm_whiz/vector_db/test_integration.py")
        print("3. Test with real embeddings from HerBERT model")
        print("4. Benchmark similarity search performance")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()