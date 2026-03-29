"""organizations, users, and ownership columns

Revision ID: d4e5f6g7h8i9
Revises: c3d4e5f6g7h8
Create Date: 2026-03-29 00:00:00.000000
"""
from typing import Sequence, Union

import uuid
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = "d4e5f6g7h8i9"
down_revision: Union[str, Sequence[str], None] = "c3d4e5f6g7h8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

DEFAULT_ORG_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


def upgrade() -> None:
    op.create_table(
        "organizations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("clerk_id", sa.String(256), unique=True, nullable=False),
        sa.Column("email", sa.String(320), nullable=True),
        sa.Column("name", sa.String(256), nullable=True),
        sa.Column("organization_id", UUID(as_uuid=True), nullable=False),
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
        sa.ForeignKeyConstraint(
            ["organization_id"], ["organizations.id"], ondelete="CASCADE"
        ),
    )
    op.create_index("ix_users_clerk_id", "users", ["clerk_id"], unique=True)
    op.create_index("ix_users_organization_id", "users", ["organization_id"])

    # -- Ownership columns on experiments --
    op.add_column(
        "experiments",
        sa.Column("user_id", UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "experiments",
        sa.Column("shared_with_org", sa.Boolean(), server_default="false", nullable=False),
    )
    op.create_foreign_key(
        "fk_experiments_user_id", "experiments", "users",
        ["user_id"], ["id"], ondelete="SET NULL",
    )
    op.create_index("ix_experiments_user_id", "experiments", ["user_id"])

    # -- Ownership columns on datasets --
    op.add_column(
        "datasets",
        sa.Column("user_id", UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "datasets",
        sa.Column("shared_with_org", sa.Boolean(), server_default="false", nullable=False),
    )
    op.create_foreign_key(
        "fk_datasets_user_id", "datasets", "users",
        ["user_id"], ["id"], ondelete="SET NULL",
    )
    op.create_index("ix_datasets_user_id", "datasets", ["user_id"])

    # -- Ownership columns on models --
    op.add_column(
        "models",
        sa.Column("user_id", UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "models",
        sa.Column("shared_with_org", sa.Boolean(), server_default="false", nullable=False),
    )
    op.create_foreign_key(
        "fk_models_user_id", "models", "users",
        ["user_id"], ["id"], ondelete="SET NULL",
    )
    op.create_index("ix_models_user_id", "models", ["user_id"])

    # Seed the default organization
    orgs = sa.table(
        "organizations",
        sa.column("id", UUID(as_uuid=True)),
        sa.column("name", sa.String),
    )
    op.bulk_insert(orgs, [{"id": DEFAULT_ORG_ID, "name": "Default Organization"}])


def downgrade() -> None:
    op.drop_index("ix_models_user_id", table_name="models")
    op.drop_constraint("fk_models_user_id", "models", type_="foreignkey")
    op.drop_column("models", "shared_with_org")
    op.drop_column("models", "user_id")

    op.drop_index("ix_datasets_user_id", table_name="datasets")
    op.drop_constraint("fk_datasets_user_id", "datasets", type_="foreignkey")
    op.drop_column("datasets", "shared_with_org")
    op.drop_column("datasets", "user_id")

    op.drop_index("ix_experiments_user_id", table_name="experiments")
    op.drop_constraint("fk_experiments_user_id", "experiments", type_="foreignkey")
    op.drop_column("experiments", "shared_with_org")
    op.drop_column("experiments", "user_id")

    op.drop_index("ix_users_organization_id", table_name="users")
    op.drop_index("ix_users_clerk_id", table_name="users")
    op.drop_table("users")
    op.drop_table("organizations")
