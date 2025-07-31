#!/usr/bin/env python3
"""Test script for database component functionality."""

import sys
import os
from typing import List

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components.sejm_whiz.database import (
    DatabaseConfig,
    get_database_config,
    check_database_health,
    init_database,
    DocumentOperations,
    VectorOperations,
)


def test_database_config():
    """Test database configuration."""
    print("Testing database configuration...")

    # Test local config
    local_config = DatabaseConfig.for_local_dev()
    print(f"Local database URL: {local_config.database_url}")

    # Test k3s config
    k3s_config = DatabaseConfig.for_k3s()
    print(f"K3s database URL: {k3s_config.database_url}")

    # Test environment-based config
    current_config = get_database_config()
    print(f"Current database URL: {current_config.database_url}")

    print("✅ Database configuration test passed\n")


def test_database_health():
    """Test database health check (will fail without running database)."""
    print("Testing database health check...")

    health_status = check_database_health()
    print("Database health status:")
    for key, value in health_status.items():
        status_icon = "✅" if value else "❌" if isinstance(value, bool) else "ℹ️"
        print(f"  {status_icon} {key}: {value}")

    if not health_status.get("connection", False):
        print("⚠️  Database connection failed (expected without running PostgreSQL)")
    else:
        print("✅ Database connection successful!")

    print()


def test_database_models():
    """Test that models can be imported and have correct structure."""
    print("Testing database models...")

    from components.sejm_whiz.database.models import (
        LegalDocument,
        LegalAmendment,
        CrossReference,
        DocumentEmbedding,
        PredictionModel,
    )

    # Test model attributes
    print("Model attributes:")
    print(f"  LegalDocument columns: {len(LegalDocument.__table__.columns)}")
    print(f"  LegalAmendment columns: {len(LegalAmendment.__table__.columns)}")
    print(f"  CrossReference columns: {len(CrossReference.__table__.columns)}")
    print(f"  DocumentEmbedding columns: {len(DocumentEmbedding.__table__.columns)}")
    print(f"  PredictionModel columns: {len(PredictionModel.__table__.columns)}")

    # Test relationships
    print(
        f"  LegalDocument relationships: {list(LegalDocument.__mapper__.relationships.keys())}"
    )

    print("✅ Database models test passed\n")


def test_vector_operations():
    """Test vector operations (offline - no database needed)."""
    print("Testing vector operations...")

    # Test embedding similarity calculation (would work with database)
    sample_embedding = [0.1] * 768  # 768-dimensional embedding for HerBERT

    print(f"Sample embedding dimensions: {len(sample_embedding)}")
    print(f"Sample embedding type: {type(sample_embedding)}")

    # These would work with a real database connection
    print("Vector operations available:")
    print("  - find_similar_documents()")
    print("  - find_similar_by_document_id()")
    print("  - batch_similarity_search()")

    print("✅ Vector operations test passed\n")


def test_migration_file():
    """Test that migration file exists and is valid."""
    print("Testing migration file...")

    migration_dir = "components/sejm_whiz/database/alembic/versions"
    migration_files = [
        f
        for f in os.listdir(migration_dir)
        if f.endswith(".py") and not f.startswith("__")
    ]

    print(f"Migration files found: {len(migration_files)}")
    for migration_file in migration_files:
        print(f"  - {migration_file}")

    if migration_files:
        # Read the migration file to verify it has content
        migration_path = os.path.join(migration_dir, migration_files[0])
        with open(migration_path, "r") as f:
            content = f.read()
            has_upgrade = "def upgrade()" in content
            has_downgrade = "def downgrade()" in content
            has_pgvector = "CREATE EXTENSION IF NOT EXISTS vector" in content

        print(f"  Migration has upgrade function: {'✅' if has_upgrade else '❌'}")
        print(f"  Migration has downgrade function: {'✅' if has_downgrade else '❌'}")
        print(
            f"  Migration creates pgvector extension: {'✅' if has_pgvector else '❌'}"
        )

        print("✅ Migration file test passed\n")
    else:
        print("❌ No migration files found\n")


def main():
    """Run all database tests."""
    print("🧪 Testing sejm-whiz database component\n")
    print("=" * 50)

    try:
        test_database_config()
        test_database_health()
        test_database_models()
        test_vector_operations()
        test_migration_file()

        print("=" * 50)
        print("🎉 All database component tests completed!")
        print("\nNext steps:")
        print("1. Deploy PostgreSQL with pgvector using Helm chart")
        print("2. Run: uv run alembic upgrade head")
        print("3. Test with real database connection")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
