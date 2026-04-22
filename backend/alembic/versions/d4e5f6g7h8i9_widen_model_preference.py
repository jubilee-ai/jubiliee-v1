"""Widen training_jobs.model_preference to TEXT.

Revision ID: d4e5f6g7h8i9
Revises: c3d4e5f6g7h8

The orchestrator's free-form preferences string ("Regression model for pricing;
optimize for low MAE/RMSE; …") routinely exceeds the original 128-char cap,
which made plan-approval inserts crash with
``StringDataRightTruncation`` and prevented training from starting.
``TEXT`` matches the sibling ``goal`` and ``error`` columns.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "d4e5f6g7h8i9"
down_revision: Union[str, Sequence[str], None] = "c3d4e5f6g7h8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "training_jobs",
        "model_preference",
        existing_type=sa.String(length=128),
        type_=sa.Text(),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "training_jobs",
        "model_preference",
        existing_type=sa.Text(),
        type_=sa.String(length=128),
        existing_nullable=True,
    )
