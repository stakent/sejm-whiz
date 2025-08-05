#!/usr/bin/env python3
"""Simple search test to debug issues."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "components"))
sys.path.insert(0, str(Path(__file__).parent / "bases"))

from sejm_whiz.database.operations import get_db_session
from sejm_whiz.database.models import LegalDocument, DocumentEmbedding
from sejm_whiz.vector_db import VectorDBOperations
from sejm_whiz.embeddings import BagEmbeddingsGenerator


def main():
    print("🔍 Simple Search Debug Test")
    print("=" * 40)

    with get_db_session() as session:
        # Check documents
        docs = session.query(LegalDocument).limit(5).all()
        print(f"📄 Found {len(docs)} sample documents:")
        for doc in docs:
            print(f"  - ID: {doc.id}")
            print(f"    Title: {doc.title[:60]}...")
            print(f"    Type: {doc.document_type}")
            print(f"    Content length: {len(doc.content)} chars")
            print()

        # Check embeddings
        embeddings = session.query(DocumentEmbedding).limit(5).all()
        print(f"🧠 Found {len(embeddings)} sample embeddings:")
        for emb in embeddings:
            print(f"  - Document ID: {emb.document_id}")
            print(f"    Model: {emb.model_name}")
            print(f"    Embedding dimension: {len(emb.embedding)}")
            print(f"    Token count: {emb.token_count}")
            print()

    # Test vector search directly
    print("🔍 Testing vector search directly...")
    try:
        vector_db = VectorDBOperations()

        # Generate embedding for test query
        generator = BagEmbeddingsGenerator()
        test_query = "posiedzenie sejmu"
        print(f"Generating embedding for query: '{test_query}'")

        query_embedding = generator.generate_bag_embedding(test_query)
        query_vector = query_embedding.document_embedding

        print(f"Query embedding dimension: {len(query_vector)}")

        # Search for similar documents
        results = vector_db.find_similar_documents(
            query_embedding=query_vector.tolist(),
            limit=5,
            similarity_threshold=0.0,  # Very low threshold
        )

        print(f"Found {len(results)} results with direct vector search:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. Document ID: {result['document_id']}")
            print(f"     Similarity: {result['similarity_score']:.4f}")
            print()

    except Exception as e:
        print(f"❌ Vector search test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
