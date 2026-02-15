"""Add multi-model embeddings support.

Revision ID: 002
Revises: 001
Create Date: 2026-02-15

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add ollama_url to projects
    op.add_column(
        "projects",
        sa.Column("ollama_url", sa.String(512), server_default="http://ollama:11434"),
    )

    # Create project_models table
    op.create_table(
        "project_models",
        sa.Column(
            "project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("model_name", sa.String(100), primary_key=True),
        sa.Column("enabled", sa.Boolean, server_default="true"),
    )

    # Create embedding_status table
    op.create_table(
        "embedding_status",
        sa.Column(
            "chunk_id",
            sa.String(36),
            sa.ForeignKey("chunks.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("model_name", sa.String(100), primary_key=True),
        sa.Column("status", sa.String(50), server_default="pending"),
        sa.Column("error_message", sa.Text, nullable=True),
    )

    # Create embeddings tables for each supported model
    # nomic-embed-text: 768 dimensions
    op.create_table(
        "embeddings_nomic_768",
        sa.Column(
            "chunk_id",
            sa.String(36),
            sa.ForeignKey("chunks.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("embedding", Vector(768), nullable=False),
    )
    op.execute(
        "CREATE INDEX ix_embeddings_nomic_768 ON embeddings_nomic_768 "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    # bge-m3: 1024 dimensions
    op.create_table(
        "embeddings_bge_m3_1024",
        sa.Column(
            "chunk_id",
            sa.String(36),
            sa.ForeignKey("chunks.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("embedding", Vector(1024), nullable=False),
    )
    op.execute(
        "CREATE INDEX ix_embeddings_bge_m3_1024 ON embeddings_bge_m3_1024 "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    # mxbai-embed-large: 1024 dimensions
    op.create_table(
        "embeddings_mxbai_1024",
        sa.Column(
            "chunk_id",
            sa.String(36),
            sa.ForeignKey("chunks.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("embedding", Vector(1024), nullable=False),
    )
    op.execute(
        "CREATE INDEX ix_embeddings_mxbai_1024 ON embeddings_mxbai_1024 "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.drop_table("embeddings_mxbai_1024")
    op.drop_table("embeddings_bge_m3_1024")
    op.drop_table("embeddings_nomic_768")
    op.drop_table("embedding_status")
    op.drop_table("project_models")
    op.drop_column("projects", "ollama_url")
