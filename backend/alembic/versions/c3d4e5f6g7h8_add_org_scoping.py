"""Add org_id / created_by for multi-tenant isolation.

Revision ID: c3d4e5f6g7h8
Revises: b2c3d4e5f6g7
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "c3d4e5f6g7h8"
down_revision: Union[str, Sequence[str], None] = "b2c3d4e5f6g7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("experiments", sa.Column("org_id", sa.String(64), nullable=True))
    op.add_column("experiments", sa.Column("created_by", sa.String(64), nullable=True))
    op.create_index("ix_experiments_org_id", "experiments", ["org_id"])

    op.add_column("training_jobs", sa.Column("org_id", sa.String(64), nullable=True))
    op.create_index("ix_training_jobs_org_id", "training_jobs", ["org_id"])

    op.add_column("datasets", sa.Column("org_id", sa.String(64), nullable=True))
    op.create_index("ix_datasets_org_id", "datasets", ["org_id"])


def downgrade() -> None:
    op.drop_index("ix_datasets_org_id", table_name="datasets")
    op.drop_column("datasets", "org_id")

    op.drop_index("ix_training_jobs_org_id", table_name="training_jobs")
    op.drop_column("training_jobs", "org_id")

    op.drop_index("ix_experiments_org_id", table_name="experiments")
    op.drop_column("experiments", "created_by")
    op.drop_column("experiments", "org_id")
