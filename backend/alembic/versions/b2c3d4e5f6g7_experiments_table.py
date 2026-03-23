"""experiments table + training_jobs.experiment_id FK

Revision ID: b2c3d4e5f6g7
Revises: a1b2c3d4e5f6
Create Date: 2026-03-22 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "b2c3d4e5f6g7"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "experiments",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("goal", sa.Text(), nullable=True),
        sa.Column("status", sa.String(32), server_default="created", nullable=False),
        sa.Column("chat_thread_id", sa.String(64), nullable=False),
        sa.Column(
            "chat_history",
            JSONB(astext_type=sa.Text()).with_variant(sa.Text(), "sqlite"),
            nullable=True,
        ),
        sa.Column(
            "training_state",
            JSONB(astext_type=sa.Text()).with_variant(sa.Text(), "sqlite"),
            nullable=True,
        ),
        sa.Column(
            "training_context",
            JSONB(astext_type=sa.Text()).with_variant(sa.Text(), "sqlite"),
            nullable=True,
        ),
        sa.Column(
            "linked_datasets",
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
        sa.UniqueConstraint("chat_thread_id", name="uq_experiment_chat_thread"),
    )
    op.create_index("ix_experiments_status", "experiments", ["status"])

    op.add_column(
        "training_jobs",
        sa.Column("experiment_id", sa.String(64), nullable=True),
    )
    op.create_foreign_key(
        "fk_training_jobs_experiment",
        "training_jobs",
        "experiments",
        ["experiment_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_training_jobs_experiment_id", "training_jobs", ["experiment_id"])


def downgrade() -> None:
    op.drop_index("ix_training_jobs_experiment_id", table_name="training_jobs")
    op.drop_constraint("fk_training_jobs_experiment", "training_jobs", type_="foreignkey")
    op.drop_column("training_jobs", "experiment_id")
    op.drop_index("ix_experiments_status", table_name="experiments")
    op.drop_table("experiments")
