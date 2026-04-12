"""Idempotent demo seed for MRM tables (migration + init_db)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy.orm import Session

from backend.shared.models import (
    Model,
    ModelApproval,
    ModelArtifact,
    ModelLifecycleStage,
    ModelMonitoringSnapshot,
    ModelReviewSchedule,
    ModelRiskProfile,
    ModelVersion,
)

MRM_STAGE_NAMES: tuple[str, ...] = (
    "Model Inventory & Risk Tiering",
    "Conceptual Soundness & Design",
    "Data Integrity & Governance",
    "Development & Testing",
    "Independent Validation",
    "Implementation & Use",
    "Ongoing Monitoring & Backtesting",
    "Annual Review & Revalidation",
)

CECL_NAME = "CECL Reserve Forecast Model"
COMMERCIAL_NAME = "Commercial Credit Risk Model"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _get_or_create_model(session: Session, name: str) -> Model:
    row = session.query(Model).filter_by(name=name).first()
    if row is not None:
        return row
    row = Model(name=name, properties={})
    session.add(row)
    session.flush()
    return row


def _ensure_current_version(session: Session, model: Model) -> ModelVersion:
    cur = (
        session.query(ModelVersion)
        .filter_by(model_id=model.id, is_current=True)
        .first()
    )
    if cur is not None:
        return cur
    session.query(ModelVersion).filter_by(model_id=model.id).update(
        {ModelVersion.is_current: False},
        synchronize_session=False,
    )
    mv = ModelVersion(
        model_id=model.id,
        version=1,
        training_run_id=None,
        storage_key=None,
        metrics={},
        is_current=True,
        properties={},
    )
    session.add(mv)
    session.flush()
    return mv


def _seed_cecl(session: Session, model: Model, version: ModelVersion) -> None:
    if session.query(ModelRiskProfile).filter_by(model_id=model.id).first():
        return

    t0 = _utcnow() - timedelta(days=120)
    profile = ModelRiskProfile(
        model_id=model.id,
        business_owner="Taylor Kim",
        business_line="Allowance & Credit Risk",
        risk_tier="moderate",
        governance_profile="small_bank",
        model_origin="vendor",
        business_purpose=(
            "Estimate lifetime expected credit losses under CECL for commercial and retail portfolios."
        ),
        approved_use=(
            "Quarterly reserve calculation, management reporting, and regulatory submissions."
        ),
        prohibited_use=(
            "Loan pricing decisions, individual credit approvals, and marketing eligibility."
        ),
        limitations=(
            "Macro scenarios are refreshed quarterly; does not reflect idiosyncratic obligor events in real time."
        ),
        board_visible=True,
        current_lifecycle_stage=MRM_STAGE_NAMES[-1],
    )
    session.add(profile)

    for i, stage_name in enumerate(MRM_STAGE_NAMES):
        started = t0 + timedelta(days=i * 10)
        completed = started + timedelta(days=7)
        session.add(
            ModelLifecycleStage(
                model_id=model.id,
                stage_name=stage_name,
                status="complete",
                owner_role="Model Owner" if i < 4 else "Model Risk",
                started_at=started,
                completed_at=completed,
                summary=f"{stage_name} completed with no material findings.",
                result_payload={"outcome": "pass", "severity": "low"},
            )
        )

    artifacts: list[dict[str, Any]] = [
        {
            "stage_name": MRM_STAGE_NAMES[3],
            "artifact_type": "documentation",
            "title": "Model Development Memo",
            "summary": "Development methodology, assumptions, and challenger benchmarks.",
            "storage_key": None,
            "external_url": None,
            "uploaded_by": "Taylor Kim",
        },
        {
            "stage_name": MRM_STAGE_NAMES[4],
            "artifact_type": "validation_report",
            "title": "Independent Validation Report",
            "summary": "SR 11-7 style validation; no material weaknesses.",
            "storage_key": None,
            "external_url": "https://example.com/validation/cecl-2025",
            "uploaded_by": "Jordan Lee",
        },
        {
            "stage_name": MRM_STAGE_NAMES[6],
            "artifact_type": "monitoring",
            "title": "Q1 Monitoring Packet",
            "summary": "Population stability and backtesting within thresholds.",
            "storage_key": "mrm/cecl/q1-monitoring.json",
            "external_url": None,
            "uploaded_by": "Monitoring Ops",
        },
    ]
    for a in artifacts:
        session.add(
            ModelArtifact(
                model_id=model.id,
                stage_name=a["stage_name"],
                artifact_type=a["artifact_type"],
                title=a["title"],
                storage_key=a["storage_key"],
                external_url=a["external_url"],
                summary=a["summary"],
                uploaded_by=a["uploaded_by"],
            )
        )

    approvals = [
        (
            MRM_STAGE_NAMES[4],
            "Independent Validation",
            "Jordan Lee",
            "approved",
            "Validation complete; model fit for intended use.",
        ),
        (
            MRM_STAGE_NAMES[5],
            "Implementation Sign-off",
            "Sam Rivera",
            "approved",
            "Implementation matches validated specification.",
        ),
    ]
    for stage_name, role, approver, decision, comment in approvals:
        session.add(
            ModelApproval(
                model_id=model.id,
                stage_name=stage_name,
                approval_role=role,
                approver_name=approver,
                decision=decision,
                comment=comment,
                decided_at=_utcnow() - timedelta(days=40),
            )
        )

    top_csi = [
        {"feature": "delinquency_rate_90d", "contribution": 0.18},
        {"feature": "utilization_ratio", "contribution": 0.12},
    ]
    session.add(
        ModelMonitoringSnapshot(
            model_id=model.id,
            model_version_id=version.id,
            captured_at=_utcnow() - timedelta(days=45),
            status="healthy",
            psi_score=0.04,
            backtest_score=0.92,
            metrics_payload={"drift_events": 0, "alerts": 0},
            top_csi_features=top_csi,
            note="Within tolerance; no governance actions.",
        )
    )
    session.add(
        ModelMonitoringSnapshot(
            model_id=model.id,
            model_version_id=version.id,
            captured_at=_utcnow() - timedelta(days=10),
            status="healthy",
            psi_score=0.03,
            backtest_score=0.94,
            metrics_payload={"drift_events": 0, "alerts": 0},
            top_csi_features=top_csi,
            note="Latest monitoring window healthy.",
        )
    )

    session.add(
        ModelReviewSchedule(
            model_id=model.id,
            review_type="Annual model review",
            cadence="annual",
            next_due_at=_utcnow() + timedelta(days=200),
            last_completed_at=_utcnow() - timedelta(days=165),
            status="scheduled",
        )
    )


def _seed_commercial(session: Session, model: Model, version: ModelVersion) -> None:
    if session.query(ModelRiskProfile).filter_by(model_id=model.id).first():
        return

    t0 = _utcnow() - timedelta(days=200)
    profile = ModelRiskProfile(
        model_id=model.id,
        business_owner="Morgan Patel",
        business_line="Commercial Banking",
        risk_tier="high",
        governance_profile="standard",
        model_origin="internal",
        business_purpose=(
            "Rank-order commercial obligor default risk for portfolio management and stress testing inputs."
        ),
        approved_use=(
            "Portfolio monitoring, risk appetite reporting, and CECL overlays where aligned with policy."
        ),
        prohibited_use=(
            "Primary underwriting decisioning without human review; consumer lending decisions."
        ),
        limitations=(
            "Reduced accuracy for newly onboarded industries with sparse training history."
        ),
        board_visible=True,
        current_lifecycle_stage=MRM_STAGE_NAMES[6],
    )
    session.add(profile)

    statuses = [
        "complete",
        "complete",
        "complete",
        "complete",
        "complete",
        "in_progress",
        "not_started",
        "blocked",
    ]
    for i, stage_name in enumerate(MRM_STAGE_NAMES):
        started = t0 + timedelta(days=i * 12)
        st = statuses[i]
        completed = None
        if st == "complete":
            completed = started + timedelta(days=9)
        session.add(
            ModelLifecycleStage(
                model_id=model.id,
                stage_name=stage_name,
                status=st,
                owner_role="Model Development" if i < 4 else "Model Risk",
                started_at=started,
                completed_at=completed,
                summary=(
                    None
                    if st == "not_started"
                    else (
                        "Monitoring variance elevated; action plan in flight."
                        if st == "in_progress"
                        else (
                            "Dependency on upstream data feed blocked progress."
                            if st == "blocked"
                            else f"{stage_name} completed."
                        )
                    )
                ),
                result_payload=(
                    {"outcome": "watch", "severity": "medium"}
                    if st == "in_progress"
                    else ({"outcome": "blocked", "severity": "high"} if st == "blocked" else {"outcome": "pass"})
                ),
            )
        )

    artifacts: list[dict[str, Any]] = [
        {
            "stage_name": MRM_STAGE_NAMES[2],
            "artifact_type": "data_dictionary",
            "title": "Commercial Credit Feature Dictionary",
            "summary": "Definitions, transforms, and lineage for model inputs.",
            "storage_key": "mrm/commercial/feature-dictionary.json",
            "external_url": None,
            "uploaded_by": "Data Governance",
        },
        {
            "stage_name": MRM_STAGE_NAMES[4],
            "artifact_type": "validation_report",
            "title": "Validation Summary",
            "summary": "Validation complete; conditions for use documented.",
            "storage_key": None,
            "external_url": "https://example.com/validation/commercial-2025",
            "uploaded_by": "Model Validation",
        },
        {
            "stage_name": MRM_STAGE_NAMES[6],
            "artifact_type": "monitoring",
            "title": "Monitoring Escalation Memo",
            "summary": "PSI drift and backtest variance triggered watch status.",
            "storage_key": None,
            "external_url": None,
            "uploaded_by": "Morgan Patel",
        },
    ]
    for a in artifacts:
        session.add(
            ModelArtifact(
                model_id=model.id,
                stage_name=a["stage_name"],
                artifact_type=a["artifact_type"],
                title=a["title"],
                storage_key=a["storage_key"],
                external_url=a["external_url"],
                summary=a["summary"],
                uploaded_by=a["uploaded_by"],
            )
        )

    session.add(
        ModelApproval(
            model_id=model.id,
            stage_name=MRM_STAGE_NAMES[4],
            approval_role="Independent Validation",
            approver_name="Riley Chen",
            decision="approved",
            comment="Validation objectives met; monitoring thresholds defined.",
            decided_at=_utcnow() - timedelta(days=60),
        )
    )

    top_csi = [
        {"feature": "debt_service_coverage", "contribution": 0.21},
        {"feature": "industry_stress_index", "contribution": 0.15},
    ]
    session.add(
        ModelMonitoringSnapshot(
            model_id=model.id,
            model_version_id=version.id,
            captured_at=_utcnow() - timedelta(days=35),
            status="watch",
            psi_score=0.14,
            backtest_score=0.71,
            metrics_payload={"drift_events": 2, "alerts": 1},
            top_csi_features=top_csi,
            note="Drift detected in commercial real estate cohort.",
        )
    )
    session.add(
        ModelMonitoringSnapshot(
            model_id=model.id,
            model_version_id=version.id,
            captured_at=_utcnow() - timedelta(days=5),
            status="watch",
            psi_score=0.16,
            backtest_score=0.69,
            metrics_payload={"drift_events": 3, "alerts": 2},
            top_csi_features=top_csi,
            note="Continued monitoring; remediation plan underway.",
        )
    )

    session.add(
        ModelReviewSchedule(
            model_id=model.id,
            review_type="Monitoring review",
            cadence="quarterly",
            next_due_at=_utcnow() - timedelta(days=12),
            last_completed_at=_utcnow() - timedelta(days=103),
            status="overdue",
        )
    )


def seed_mrm_demo_data_if_needed(session: Session) -> None:
    """Insert two synthetic models and MRM graph when profiles are absent."""
    cecl_model = _get_or_create_model(session, CECL_NAME)
    comm_model = _get_or_create_model(session, COMMERCIAL_NAME)
    cecl_v = _ensure_current_version(session, cecl_model)
    comm_v = _ensure_current_version(session, comm_model)
    _seed_cecl(session, cecl_model, cecl_v)
    _seed_commercial(session, comm_model, comm_v)
