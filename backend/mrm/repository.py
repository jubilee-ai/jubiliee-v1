"""Persistence helpers for MRM (query/aggregate)."""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from backend.shared.models import (
    Model,
    ModelApproval,
    ModelArtifact,
    ModelLifecycleStage,
    ModelMonitoringSnapshot,
    ModelReviewSchedule,
    ModelRiskProfile,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _latest_snapshots_by_model(
    session: Session, model_ids: list[uuid.UUID]
) -> dict[uuid.UUID, ModelMonitoringSnapshot]:
    if not model_ids:
        return {}
    sub = (
        session.query(
            ModelMonitoringSnapshot.model_id,
            func.max(ModelMonitoringSnapshot.captured_at).label("mx"),
        )
        .filter(ModelMonitoringSnapshot.model_id.in_(model_ids))
        .group_by(ModelMonitoringSnapshot.model_id)
        .subquery()
    )
    rows = (
        session.query(ModelMonitoringSnapshot)
        .join(
            sub,
            and_(
                ModelMonitoringSnapshot.model_id == sub.c.model_id,
                ModelMonitoringSnapshot.captured_at == sub.c.mx,
            ),
        )
        .all()
    )
    return {r.model_id: r for r in rows}


def _next_review_by_model(
    session: Session, model_ids: list[uuid.UUID]
) -> dict[uuid.UUID, ModelReviewSchedule | None]:
    if not model_ids:
        return {}
    rows = (
        session.query(ModelReviewSchedule)
        .filter(ModelReviewSchedule.model_id.in_(model_ids))
        .all()
    )
    by_mid: dict[uuid.UUID, list[ModelReviewSchedule]] = defaultdict(list)
    for r in rows:
        by_mid[r.model_id].append(r)
    out: dict[uuid.UUID, ModelReviewSchedule | None] = {}
    for mid in model_ids:
        lst = by_mid.get(mid, [])
        if not lst:
            out[mid] = None
            continue
        with_due = [x for x in lst if x.next_due_at is not None]
        if with_due:
            out[mid] = min(with_due, key=lambda x: x.next_due_at or _utcnow())
        else:
            out[mid] = lst[0]
    return out


def list_models_with_mrm(
    session: Session,
) -> list[dict[str, Any]]:
    models = session.query(Model).order_by(Model.name).all()
    ids = [m.id for m in models]
    profiles = {
        p.model_id: p
        for p in session.query(ModelRiskProfile)
        .filter(ModelRiskProfile.model_id.in_(ids))
        .all()
    }
    snaps = _latest_snapshots_by_model(session, ids)
    reviews = _next_review_by_model(session, ids)
    result: list[dict[str, Any]] = []
    for m in models:
        result.append(
            {
                "model": m,
                "profile": profiles.get(m.id),
                "latest_monitoring": snaps.get(m.id),
                "next_review": reviews.get(m.id),
            }
        )
    return result


def get_model_by_name(session: Session, model_name: str) -> Model | None:
    return session.query(Model).filter(Model.name == model_name).first()


def get_model_detail_bundle(
    session: Session, model_name: str
) -> dict[str, Any] | None:
    model = get_model_by_name(session, model_name)
    if model is None:
        return None
    profile = (
        session.query(ModelRiskProfile)
        .filter(ModelRiskProfile.model_id == model.id)
        .first()
    )
    stages = (
        session.query(ModelLifecycleStage)
        .filter(ModelLifecycleStage.model_id == model.id)
        .order_by(ModelLifecycleStage.started_at.asc())
        .all()
    )
    artifacts = (
        session.query(ModelArtifact)
        .filter(ModelArtifact.model_id == model.id)
        .order_by(ModelArtifact.created_at.asc())
        .all()
    )
    approvals = (
        session.query(ModelApproval)
        .filter(ModelApproval.model_id == model.id)
        .order_by(ModelApproval.decided_at.desc())
        .all()
    )
    monitoring = (
        session.query(ModelMonitoringSnapshot)
        .filter(ModelMonitoringSnapshot.model_id == model.id)
        .order_by(ModelMonitoringSnapshot.captured_at.desc())
        .limit(12)
        .all()
    )
    schedules = (
        session.query(ModelReviewSchedule)
        .filter(ModelReviewSchedule.model_id == model.id)
        .order_by(ModelReviewSchedule.next_due_at.asc())
        .all()
    )
    return {
        "model": model,
        "profile": profile,
        "lifecycle_stages": stages,
        "artifacts": artifacts,
        "approvals": approvals,
        "monitoring_snapshots": monitoring,
        "review_schedules": schedules,
    }


def insert_model_approval(
    session: Session,
    *,
    model_id: uuid.UUID,
    stage_name: str,
    approval_role: str,
    approver_name: str,
    decision: str,
    comment: str | None,
) -> ModelApproval:
    decided_at = None if decision == "pending" else _utcnow()
    row = ModelApproval(
        model_id=model_id,
        stage_name=stage_name,
        approval_role=approval_role,
        approver_name=approver_name,
        decision=decision,
        comment=comment,
        decided_at=decided_at,
    )
    session.add(row)
    session.flush()
    return row


def patch_model_risk_profile(
    session: Session,
    profile: ModelRiskProfile,
    updates: dict[str, Any],
) -> ModelRiskProfile:
    allowed = {
        "business_owner",
        "business_line",
        "risk_tier",
        "governance_profile",
        "model_origin",
        "business_purpose",
        "approved_use",
        "prohibited_use",
        "limitations",
        "board_visible",
        "current_lifecycle_stage",
    }
    for k, v in updates.items():
        if k in allowed:
            setattr(profile, k, v)
    session.flush()
    return profile


def aggregate_dashboard(session: Session, portal: str) -> dict[str, Any]:
    models = session.query(Model).all()
    profiles = session.query(ModelRiskProfile).all()
    prof_by_mid = {p.model_id: p for p in profiles}
    schedules = session.query(ModelReviewSchedule).all()
    stages = session.query(ModelLifecycleStage).all()

    model_id_to_name = {m.id: m.name for m in models}
    now = _utcnow()

    overdue_model_ids: set[uuid.UUID] = set()
    for s in schedules:
        if s.next_due_at is not None and s.next_due_at < now:
            overdue_model_ids.add(s.model_id)
    overdue = len(overdue_model_ids)

    high_risk: list[str] = []
    board_vis: list[str] = []
    watch_breach: list[str] = []
    blocked: list[str] = []
    in_prog_val: list[str] = []

    latest_snap = _latest_snapshots_by_model(session, [m.id for m in models])
    for m in models:
        p = prof_by_mid.get(m.id)
        if p is None:
            continue
        if (p.risk_tier or "").lower() == "high":
            high_risk.append(m.name)
        if p.board_visible:
            board_vis.append(m.name)

    for m in models:
        snap = latest_snap.get(m.id)
        if snap and snap.status in ("watch", "breach"):
            watch_breach.append(m.name)

    for st in stages:
        if st.status == "blocked":
            nm = model_id_to_name.get(st.model_id)
            if nm and nm not in blocked:
                blocked.append(nm)
        if (
            st.stage_name == "Independent Validation"
            and st.status == "in_progress"
        ):
            nm = model_id_to_name.get(st.model_id)
            if nm and nm not in in_prog_val:
                in_prog_val.append(nm)

    total = len(models)
    with_profile = len(profiles)

    out: dict[str, Any] = {
        "portal": portal,
        "total_models": total,
        "with_profile": with_profile,
        "overdue_reviews": overdue,
        "high_risk_models": sorted(high_risk),
        "board_visible_models": sorted(board_vis),
        "watch_or_breach_models": sorted(watch_breach),
        "blocked_lifecycle_models": sorted(blocked),
        "in_progress_validation_models": sorted(in_prog_val),
    }

    if portal == "owner":
        out["board_visible_models"] = []
        out["in_progress_validation_models"] = []
    elif portal == "validation":
        out["board_visible_models"] = []
    elif portal == "board":
        out["high_risk_models"] = sorted(
            [n for n in high_risk if n in set(board_vis) or n in set(watch_breach)]
        )
        out["in_progress_validation_models"] = []
    return out
