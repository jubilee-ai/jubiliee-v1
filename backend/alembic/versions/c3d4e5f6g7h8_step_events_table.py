"""step_events table for observability

Revision ID: c3d4e5f6g7h8
Revises: b2c3d4e5f6g7
Create Date: 2026-03-29 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "c3d4e5f6g7h8"
down_revision: Union[str, Sequence[str], None] = "b2c3d4e5f6g7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "step_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("experiment_id", sa.String(64), nullable=True),
        sa.Column("training_job_id", sa.String(64), nullable=True),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("node", sa.String(128), nullable=True),
        sa.Column(
            "payload",
            JSONB(astext_type=sa.Text()).with_variant(sa.Text(), "sqlite"),
            nullable=True,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["experiment_id"], ["experiments.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["training_job_id"], ["training_jobs.id"], ondelete="CASCADE"
        ),
    )
    op.create_index("ix_step_events_experiment_id", "step_events", ["experiment_id"])
    op.create_index("ix_step_events_training_job_id", "step_events", ["training_job_id"])
    op.create_index("ix_step_events_event_type", "step_events", ["event_type"])


def downgrade() -> None:
    op.drop_index("ix_step_events_event_type", table_name="step_events")
    op.drop_index("ix_step_events_training_job_id", table_name="step_events")
    op.drop_index("ix_step_events_experiment_id", table_name="step_events")
    op.drop_table("step_events")
