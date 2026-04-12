"""MRM API orchestration and ORM → schema mapping."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from backend.mrm import repository as mrm_repo
from backend.mrm.schemas import (
    ApprovalCreateBody,
    DashboardSummaryOut,
    ModelApprovalOut,
    ModelArtifactOut,
    ModelDetailOut,
    ModelLifecycleStageOut,
    ModelListItemOut,
    ModelMonitoringSnapshotOut,
    ModelReviewScheduleOut,
    ModelRiskProfileOut,
    ProfilePatchBody,
)
from backend.shared.models import ModelApproval, ModelRiskProfile


def _profile_out(p: ModelRiskProfile) -> ModelRiskProfileOut:
    return ModelRiskProfileOut(
        id=str(p.id),
        model_id=str(p.model_id),
        business_owner=p.business_owner,
        business_line=p.business_line,
        risk_tier=p.risk_tier,
        governance_profile=p.governance_profile,
        model_origin=p.model_origin,
        business_purpose=p.business_purpose,
        approved_use=p.approved_use,
        prohibited_use=p.prohibited_use,
        limitations=p.limitations,
        board_visible=p.board_visible,
        current_lifecycle_stage=p.current_lifecycle_stage,
        created_at=p.created_at,
        updated_at=p.updated_at,
    )


def _snapshot_out(s: Any) -> ModelMonitoringSnapshotOut:
    top = s.top_csi_features
    if top is not None and not isinstance(top, list):
        top = None
    return ModelMonitoringSnapshotOut(
        id=str(s.id),
        model_id=str(s.model_id),
        model_version_id=str(s.model_version_id) if s.model_version_id else None,
        captured_at=s.captured_at,
        status=s.status,
        psi_score=s.psi_score,
        backtest_score=s.backtest_score,
        metrics_payload=s.metrics_payload,
        top_csi_features=top,
        note=s.note,
    )


def _schedule_out(s: Any) -> ModelReviewScheduleOut:
    return ModelReviewScheduleOut(
        id=str(s.id),
        model_id=str(s.model_id),
        review_type=s.review_type,
        cadence=s.cadence,
        next_due_at=s.next_due_at,
        last_completed_at=s.last_completed_at,
        status=s.status,
    )


def _stage_out(s: Any) -> ModelLifecycleStageOut:
    return ModelLifecycleStageOut(
        id=str(s.id),
        model_id=str(s.model_id),
        stage_name=s.stage_name,
        status=s.status,
        owner_role=s.owner_role,
        started_at=s.started_at,
        completed_at=s.completed_at,
        summary=s.summary,
        result_payload=s.result_payload,
    )


def _artifact_out(a: Any) -> ModelArtifactOut:
    return ModelArtifactOut(
        id=str(a.id),
        model_id=str(a.model_id),
        stage_name=a.stage_name,
        artifact_type=a.artifact_type,
        title=a.title,
        storage_key=a.storage_key,
        external_url=a.external_url,
        summary=a.summary,
        uploaded_by=a.uploaded_by,
        created_at=a.created_at,
    )


def _approval_out(a: ModelApproval) -> ModelApprovalOut:
    return ModelApprovalOut(
        id=str(a.id),
        model_id=str(a.model_id),
        stage_name=a.stage_name,
        approval_role=a.approval_role,
        approver_name=a.approver_name,
        decision=a.decision,
        comment=a.comment,
        decided_at=a.decided_at,
    )


def list_models(session: Session) -> list[ModelListItemOut]:
    rows = mrm_repo.list_models_with_mrm(session)
    out: list[ModelListItemOut] = []
    for row in rows:
        m = row["model"]
        prof = row["profile"]
        snap = row["latest_monitoring"]
        rev = row["next_review"]
        out.append(
            ModelListItemOut(
                model_id=str(m.id),
                model_name=m.name,
                profile=_profile_out(prof) if prof else None,
                latest_monitoring=_snapshot_out(snap) if snap else None,
                next_review=_schedule_out(rev) if rev else None,
            )
        )
    return out


def get_model_detail(session: Session, model_name: str) -> ModelDetailOut | None:
    bundle = mrm_repo.get_model_detail_bundle(session, model_name)
    if bundle is None:
        return None
    m = bundle["model"]
    prof = bundle["profile"]
    return ModelDetailOut(
        model_id=str(m.id),
        model_name=m.name,
        profile=_profile_out(prof) if prof else None,
        lifecycle_stages=[_stage_out(s) for s in bundle["lifecycle_stages"]],
        artifacts=[_artifact_out(a) for a in bundle["artifacts"]],
        approvals=[_approval_out(a) for a in bundle["approvals"]],
        monitoring_snapshots=[_snapshot_out(s) for s in bundle["monitoring_snapshots"]],
        review_schedules=[_schedule_out(s) for s in bundle["review_schedules"]],
    )


def get_dashboard(session: Session, portal: str) -> DashboardSummaryOut:
    d = mrm_repo.aggregate_dashboard(session, portal)
    return DashboardSummaryOut(**d)


def add_approval(
    session: Session, model_name: str, body: ApprovalCreateBody
) -> ModelApprovalOut | None:
    model = mrm_repo.get_model_by_name(session, model_name)
    if model is None:
        return None
    row = mrm_repo.insert_model_approval(
        session,
        model_id=model.id,
        stage_name=body.stage_name,
        approval_role=body.approval_role,
        approver_name=body.approver_name,
        decision=body.decision,
        comment=body.comment,
    )
    return _approval_out(row)


def patch_profile(
    session: Session, model_name: str, body: ProfilePatchBody
) -> ModelRiskProfileOut | None:
    model = mrm_repo.get_model_by_name(session, model_name)
    if model is None:
        return None
    prof = (
        session.query(ModelRiskProfile)
        .filter(ModelRiskProfile.model_id == model.id)
        .first()
    )
    if prof is None:
        return None
    data = body.model_dump(exclude_unset=True)
    mrm_repo.patch_model_risk_profile(session, prof, data)
    return _profile_out(prof)
