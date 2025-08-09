"""refactor_schema_naming_to_sejm_whiz_namespace

Revision ID: 94ff641a7af5
Revises: fb191867ebe8
Create Date: 2025-08-08 18:30:06.556421

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "94ff641a7af5"
down_revision: Union[str, Sequence[str], None] = "fb191867ebe8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: Rename tables to follow sejm_whiz namespace convention."""

    # Rename tables to follow sejm_whiz namespace
    # Order is important due to foreign key constraints - start with dependent tables

    # 1. Rename dependent tables first (those with foreign keys)
    op.rename_table("legal_amendments", "sejm_whiz_amendments")
    op.rename_table("cross_references", "sejm_whiz_cross_references")
    op.rename_table("document_embeddings", "sejm_whiz_document_embeddings")
    op.rename_table("prediction_models", "sejm_whiz_prediction_models")

    # 2. Rename main table last (referenced by foreign keys)
    op.rename_table("legal_documents", "sejm_whiz_documents")

    # Update foreign key references in renamed tables
    # Note: PostgreSQL automatically updates FK constraint names when tables are renamed,
    # but we need to update any explicit references in our code

    # Update index names to match new table names
    # legal_documents indexes
    op.execute(
        "ALTER INDEX idx_legal_documents_type RENAME TO idx_sejm_whiz_documents_type"
    )
    op.execute(
        "ALTER INDEX idx_legal_documents_domain RENAME TO idx_sejm_whiz_documents_domain"
    )
    op.execute(
        "ALTER INDEX idx_legal_documents_published RENAME TO idx_sejm_whiz_documents_published"
    )
    op.execute(
        "ALTER INDEX idx_legal_documents_embedding RENAME TO idx_sejm_whiz_documents_embedding"
    )

    # legal_amendments indexes
    op.execute(
        "ALTER INDEX idx_amendments_document RENAME TO idx_sejm_whiz_amendments_document"
    )
    op.execute(
        "ALTER INDEX idx_amendments_omnibus RENAME TO idx_sejm_whiz_amendments_omnibus"
    )
    op.execute(
        "ALTER INDEX idx_amendments_effective RENAME TO idx_sejm_whiz_amendments_effective"
    )

    # cross_references indexes
    op.execute(
        "ALTER INDEX idx_cross_refs_source RENAME TO idx_sejm_whiz_cross_refs_source"
    )
    op.execute(
        "ALTER INDEX idx_cross_refs_target RENAME TO idx_sejm_whiz_cross_refs_target"
    )
    op.execute(
        "ALTER INDEX idx_cross_refs_type RENAME TO idx_sejm_whiz_cross_refs_type"
    )

    # document_embeddings indexes
    op.execute(
        "ALTER INDEX idx_embeddings_document RENAME TO idx_sejm_whiz_embeddings_document"
    )
    op.execute(
        "ALTER INDEX idx_embeddings_model RENAME TO idx_sejm_whiz_embeddings_model"
    )
    op.execute(
        "ALTER INDEX idx_embeddings_vector RENAME TO idx_sejm_whiz_embeddings_vector"
    )

    # prediction_models indexes
    op.execute("ALTER INDEX idx_models_active RENAME TO idx_sejm_whiz_models_active")
    op.execute("ALTER INDEX idx_models_type RENAME TO idx_sejm_whiz_models_type")
    op.execute("ALTER INDEX idx_models_env RENAME TO idx_sejm_whiz_models_env")


def downgrade() -> None:
    """Downgrade schema: Revert table names to original generic naming."""

    # Reverse the upgrade operations
    # Rename tables back to original names

    # 1. Rename main table first
    op.rename_table("sejm_whiz_documents", "legal_documents")

    # 2. Rename dependent tables
    op.rename_table("sejm_whiz_amendments", "legal_amendments")
    op.rename_table("sejm_whiz_cross_references", "cross_references")
    op.rename_table("sejm_whiz_document_embeddings", "document_embeddings")
    op.rename_table("sejm_whiz_prediction_models", "prediction_models")

    # Revert index names
    # sejm_whiz_documents indexes
    op.execute(
        "ALTER INDEX idx_sejm_whiz_documents_type RENAME TO idx_legal_documents_type"
    )
    op.execute(
        "ALTER INDEX idx_sejm_whiz_documents_domain RENAME TO idx_legal_documents_domain"
    )
    op.execute(
        "ALTER INDEX idx_sejm_whiz_documents_published RENAME TO idx_legal_documents_published"
    )
    op.execute(
        "ALTER INDEX idx_sejm_whiz_documents_embedding RENAME TO idx_legal_documents_embedding"
    )

    # sejm_whiz_amendments indexes
    op.execute(
        "ALTER INDEX idx_sejm_whiz_amendments_document RENAME TO idx_amendments_document"
    )
    op.execute(
        "ALTER INDEX idx_sejm_whiz_amendments_omnibus RENAME TO idx_amendments_omnibus"
    )
    op.execute(
        "ALTER INDEX idx_sejm_whiz_amendments_effective RENAME TO idx_amendments_effective"
    )

    # sejm_whiz_cross_references indexes
    op.execute(
        "ALTER INDEX idx_sejm_whiz_cross_refs_source RENAME TO idx_cross_refs_source"
    )
    op.execute(
        "ALTER INDEX idx_sejm_whiz_cross_refs_target RENAME TO idx_cross_refs_target"
    )
    op.execute(
        "ALTER INDEX idx_sejm_whiz_cross_refs_type RENAME TO idx_cross_refs_type"
    )

    # sejm_whiz_document_embeddings indexes
    op.execute(
        "ALTER INDEX idx_sejm_whiz_embeddings_document RENAME TO idx_embeddings_document"
    )
    op.execute(
        "ALTER INDEX idx_sejm_whiz_embeddings_model RENAME TO idx_embeddings_model"
    )
    op.execute(
        "ALTER INDEX idx_sejm_whiz_embeddings_vector RENAME TO idx_embeddings_vector"
    )

    # sejm_whiz_prediction_models indexes
    op.execute("ALTER INDEX idx_sejm_whiz_models_active RENAME TO idx_models_active")
    op.execute("ALTER INDEX idx_sejm_whiz_models_type RENAME TO idx_models_type")
    op.execute("ALTER INDEX idx_sejm_whiz_models_env RENAME TO idx_models_env")
