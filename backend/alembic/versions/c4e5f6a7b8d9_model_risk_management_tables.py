"""model risk management (MRM) demo tables

Revision ID: c4e5f6a7b8d9
Revises: b2c3d4e5f6g7
Create Date: 2026-04-11 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "c4e5f6a7b8d9"
down_revision: Union[str, Sequence[str], None] = "b2c3d4e5f6g7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_jsonb = lambda: JSONB(astext_type=sa.Text()).with_variant(sa.Text(), "sqlite")


def upgrade() -> None:
    op.create_table(
        "model_risk_profiles",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("model_id", UUID(as_uuid=True), nullable=False),
        sa.Column("business_owner", sa.String(256), nullable=False),
        sa.Column("business_line", sa.String(256), nullable=False),
        sa.Column("risk_tier", sa.String(64), nullable=False),
        sa.Column("governance_profile", sa.String(64), nullable=False),
        sa.Column("model_origin", sa.String(32), nullable=False),
        sa.Column("business_purpose", sa.Text(), nullable=False),
        sa.Column("approved_use", sa.Text(), nullable=False),
        sa.Column("prohibited_use", sa.Text(), nullable=False),
        sa.Column("limitations", sa.Text(), nullable=True),
        sa.Column("board_visible", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("current_lifecycle_stage", sa.String(256), nullable=True),
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
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_id", name="uq_model_risk_profiles_model_id"),
    )
    op.create_index("ix_model_risk_profiles_model_id", "model_risk_profiles", ["model_id"])

    op.create_table(
        "model_lifecycle_stages",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("model_id", UUID(as_uuid=True), nullable=False),
        sa.Column("stage_name", sa.String(256), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("owner_role", sa.String(128), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("result_payload", _jsonb(), nullable=True),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "model_id", "stage_name", name="uq_mrm_lifecycle_model_stage"
        ),
    )
    op.create_index("ix_model_lifecycle_stages_model_id", "model_lifecycle_stages", ["model_id"])

    op.create_table(
        "model_artifacts",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("model_id", UUID(as_uuid=True), nullable=False),
        sa.Column("stage_name", sa.String(256), nullable=False),
        sa.Column("artifact_type", sa.String(128), nullable=False),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("storage_key", sa.Text(), nullable=True),
        sa.Column("external_url", sa.Text(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("uploaded_by", sa.String(256), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_artifacts_model_id", "model_artifacts", ["model_id"])

    op.create_table(
        "model_approvals",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("model_id", UUID(as_uuid=True), nullable=False),
        sa.Column("stage_name", sa.String(256), nullable=False),
        sa.Column("approval_role", sa.String(128), nullable=False),
        sa.Column("approver_name", sa.String(256), nullable=False),
        sa.Column("decision", sa.String(32), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_approvals_model_id", "model_approvals", ["model_id"])

    op.create_table(
        "model_monitoring_snapshots",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("model_id", UUID(as_uuid=True), nullable=False),
        sa.Column("model_version_id", UUID(as_uuid=True), nullable=True),
        sa.Column(
            "captured_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("psi_score", sa.Float(), nullable=True),
        sa.Column("backtest_score", sa.Float(), nullable=True),
        sa.Column("metrics_payload", _jsonb(), nullable=True),
        sa.Column("top_csi_features", _jsonb(), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["model_version_id"], ["model_versions.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_model_monitoring_snapshots_model_id", "model_monitoring_snapshots", ["model_id"]
    )
    op.create_index(
        "ix_model_monitoring_snapshots_model_version_id",
        "model_monitoring_snapshots",
        ["model_version_id"],
    )

    op.create_table(
        "model_review_schedules",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("model_id", UUID(as_uuid=True), nullable=False),
        sa.Column("review_type", sa.String(128), nullable=False),
        sa.Column("cadence", sa.String(64), nullable=False),
        sa.Column("next_due_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(64), nullable=False),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_model_review_schedules_model_id", "model_review_schedules", ["model_id"]
    )

    _run_mrm_seed()


def _run_mrm_seed() -> None:
    from backend.mrm.seed import seed_mrm_demo_data_if_needed

    bind = op.get_bind()
    from sqlalchemy.orm import Session

    session = Session(bind=bind)
    try:
        seed_mrm_demo_data_if_needed(session)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def downgrade() -> None:
    op.drop_index("ix_model_review_schedules_model_id", table_name="model_review_schedules")
    op.drop_table("model_review_schedules")
    op.drop_index(
        "ix_model_monitoring_snapshots_model_version_id",
        table_name="model_monitoring_snapshots",
    )
    op.drop_index(
        "ix_model_monitoring_snapshots_model_id", table_name="model_monitoring_snapshots"
    )
    op.drop_table("model_monitoring_snapshots")
    op.drop_index("ix_model_approvals_model_id", table_name="model_approvals")
    op.drop_table("model_approvals")
    op.drop_index("ix_model_artifacts_model_id", table_name="model_artifacts")
    op.drop_table("model_artifacts")
    op.drop_index("ix_model_lifecycle_stages_model_id", table_name="model_lifecycle_stages")
    op.drop_table("model_lifecycle_stages")
    op.drop_index("ix_model_risk_profiles_model_id", table_name="model_risk_profiles")
    op.drop_table("model_risk_profiles")
