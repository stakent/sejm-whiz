"""Initial migration: legal documents and embeddings

Revision ID: fb191867ebe8
Revises:
Create Date: 2025-07-29 08:26:58.953746

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = "fb191867ebe8"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # Create legal_documents table
    op.create_table(
        "legal_documents",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("document_type", sa.String(100), nullable=False),
        sa.Column("source_url", sa.String(500)),
        sa.Column("eli_identifier", sa.String(200), unique=True),
        sa.Column("embedding", Vector(768)),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()")),
        sa.Column("published_at", sa.DateTime()),
        sa.Column("legal_act_type", sa.String(100)),
        sa.Column("legal_domain", sa.String(100)),
        sa.Column("is_amendment", sa.Boolean(), default=False),
        sa.Column("affects_multiple_acts", sa.Boolean(), default=False),
    )

    # Create legal_amendments table
    op.create_table(
        "legal_amendments",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("legal_documents.id"),
            nullable=False,
        ),
        sa.Column("amendment_type", sa.String(100)),
        sa.Column("affected_article", sa.String(200)),
        sa.Column("affected_paragraph", sa.String(200)),
        sa.Column("amendment_text", sa.Text()),
        sa.Column("affects_multiple_acts", sa.Boolean(), default=False),
        sa.Column("omnibus_bill_id", sa.String(200)),
        sa.Column("impact_score", sa.Integer(), default=0),
        sa.Column("complexity_level", sa.String(50)),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()")),
        sa.Column("effective_date", sa.DateTime()),
    )

    # Create cross_references table
    op.create_table(
        "cross_references",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "source_document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("legal_documents.id"),
            nullable=False,
        ),
        sa.Column(
            "target_document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("legal_documents.id"),
            nullable=False,
        ),
        sa.Column("reference_type", sa.String(100)),
        sa.Column("reference_text", sa.Text()),
        sa.Column("source_article", sa.String(200)),
        sa.Column("target_article", sa.String(200)),
        sa.Column("similarity_score", sa.Integer()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()")),
    )

    # Create document_embeddings table
    op.create_table(
        "document_embeddings",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("legal_documents.id"),
            nullable=False,
        ),
        sa.Column("embedding", Vector(768)),
        sa.Column("model_name", sa.String(100)),
        sa.Column("model_version", sa.String(50)),
        sa.Column("embedding_method", sa.String(100)),
        sa.Column("token_count", sa.Integer()),
        sa.Column("chunk_size", sa.Integer()),
        sa.Column("preprocessing_version", sa.String(50)),
        sa.Column("confidence_score", sa.Integer()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()")),
    )

    # Create prediction_models table
    op.create_table(
        "prediction_models",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("model_name", sa.String(200), nullable=False),
        sa.Column("model_type", sa.String(100)),
        sa.Column("model_version", sa.String(50)),
        sa.Column("accuracy_score", sa.Integer()),
        sa.Column("precision_score", sa.Integer()),
        sa.Column("recall_score", sa.Integer()),
        sa.Column("f1_score", sa.Integer()),
        sa.Column("training_data_size", sa.Integer()),
        sa.Column("training_date", sa.DateTime()),
        sa.Column("hyperparameters", postgresql.JSON()),
        sa.Column("is_active", sa.Boolean(), default=False),
        sa.Column("deployment_env", sa.String(50)),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()")),
    )

    # Create indexes
    op.create_index("idx_legal_documents_type", "legal_documents", ["document_type"])
    op.create_index("idx_legal_documents_domain", "legal_documents", ["legal_domain"])
    op.create_index(
        "idx_legal_documents_published", "legal_documents", ["published_at"]
    )
    op.create_index(
        "idx_legal_documents_embedding",
        "legal_documents",
        ["embedding"],
        postgresql_using="ivfflat",
    )

    op.create_index("idx_amendments_document", "legal_amendments", ["document_id"])
    op.create_index("idx_amendments_omnibus", "legal_amendments", ["omnibus_bill_id"])
    op.create_index("idx_amendments_effective", "legal_amendments", ["effective_date"])

    op.create_index("idx_cross_refs_source", "cross_references", ["source_document_id"])
    op.create_index("idx_cross_refs_target", "cross_references", ["target_document_id"])
    op.create_index("idx_cross_refs_type", "cross_references", ["reference_type"])

    op.create_index("idx_embeddings_document", "document_embeddings", ["document_id"])
    op.create_index(
        "idx_embeddings_model", "document_embeddings", ["model_name", "model_version"]
    )
    op.create_index(
        "idx_embeddings_vector",
        "document_embeddings",
        ["embedding"],
        postgresql_using="ivfflat",
    )

    op.create_index("idx_models_active", "prediction_models", ["is_active"])
    op.create_index("idx_models_type", "prediction_models", ["model_type"])
    op.create_index("idx_models_env", "prediction_models", ["deployment_env"])


def downgrade() -> None:
    """Downgrade schema."""
    # Drop tables in reverse order due to foreign key constraints
    op.drop_table("prediction_models")
    op.drop_table("document_embeddings")
    op.drop_table("cross_references")
    op.drop_table("legal_amendments")
    op.drop_table("legal_documents")

    # Note: We don't drop extensions as they might be used by other applications
