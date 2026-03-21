"""mlops registry tables

Revision ID: a1b2c3d4e5f6
Revises: 760373e4bc3c
Create Date: 2026-03-21 00:00:00.000000

Renames:
  - training_contexts -> training_summaries
  - agent_store -> agent_checkpoints

Creates:
  - datasets
  - models
  - model_versions
  - run_dataset_links
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "760373e4bc3c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -- Rename existing tables (preserves data) --
    op.rename_table("training_contexts", "training_summaries")
    op.rename_table("agent_store", "agent_checkpoints")

    # -- New: datasets --
    op.create_table(
        "datasets",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("source_type", sa.String(64), nullable=False),
        sa.Column(
            "properties",
            JSONB(astext_type=sa.Text()).with_variant(sa.Text(), "sqlite"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # -- New: models --
    op.create_table(
        "models",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column(
            "properties",
            JSONB(astext_type=sa.Text()).with_variant(sa.Text(), "sqlite"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_models_name"),
    )

    # -- New: model_versions --
    op.create_table(
        "model_versions",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("model_id", UUID(as_uuid=True), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("training_run_id", sa.String(64), nullable=True),
        sa.Column("storage_key", sa.Text(), nullable=True),
        sa.Column(
            "metrics",
            JSONB(astext_type=sa.Text()).with_variant(sa.Text(), "sqlite"),
            nullable=True,
        ),
        sa.Column("is_current", sa.Boolean(), default=False, nullable=False),
        sa.Column(
            "properties",
            JSONB(astext_type=sa.Text()).with_variant(sa.Text(), "sqlite"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["training_run_id"], ["training_jobs.id"], ondelete="SET NULL"
        ),
        sa.UniqueConstraint("model_id", "version", name="uq_model_version"),
    )
    op.create_index(
        "ix_model_versions_is_current", "model_versions", ["is_current"]
    )

    # -- New: run_dataset_links --
    op.create_table(
        "run_dataset_links",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("training_run_id", sa.String(64), nullable=False),
        sa.Column("dataset_id", UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["training_run_id"], ["training_jobs.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"], ["datasets.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint(
            "training_run_id", "dataset_id", "role", name="uq_run_dataset_role"
        ),
    )


def downgrade() -> None:
    op.drop_table("run_dataset_links")
    op.drop_index("ix_model_versions_is_current", table_name="model_versions")
    op.drop_table("model_versions")
    op.drop_table("models")
    op.drop_table("datasets")

    op.rename_table("training_summaries", "training_contexts")
    op.rename_table("agent_checkpoints", "agent_store")
