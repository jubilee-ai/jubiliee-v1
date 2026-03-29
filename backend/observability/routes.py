"""Observability API — timeline, step drill-down, cross-experiment summaries."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func as sa_func, or_, and_, select, case, Integer

from backend.shared.auth import CurrentUser, get_current_user
from backend.shared.database import get_db_session
from backend.shared.models import (
    Experiment,
    ModelVersion,
    StepEvent,
    TrainingJob,
    User,
)

router = APIRouter()


def _row_to_dict(row: StepEvent) -> dict[str, Any]:
    return {
        "id": row.id,
        "experiment_id": row.experiment_id,
        "training_job_id": row.training_job_id,
        "event_type": row.event_type,
        "node": row.node,
        "payload": row.payload,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
        "duration_ms": row.duration_ms,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.get("/api/experiments/{experiment_id}/timeline")
def get_experiment_timeline(
    experiment_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Ordered step events for an experiment, grouped by node."""
    with get_db_session() as session:
        exp = session.query(Experiment).filter_by(id=experiment_id).first()
        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")

        events = (
            session.query(StepEvent)
            .filter_by(experiment_id=experiment_id)
            .order_by(StepEvent.created_at.asc())
            .all()
        )

        by_node: dict[str, list[dict]] = defaultdict(list)
        flat: list[dict] = []
        for ev in events:
            d = _row_to_dict(ev)
            flat.append(d)
            if d["node"]:
                by_node[d["node"]].append(d)

        return {"experiment_id": experiment_id, "events": flat, "by_node": dict(by_node)}


@router.get("/api/experiments/{experiment_id}/steps/{node}")
def get_step_events(
    experiment_id: str,
    node: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    """All events for a specific pipeline step within an experiment."""
    with get_db_session() as session:
        events = (
            session.query(StepEvent)
            .filter_by(experiment_id=experiment_id, node=node)
            .order_by(StepEvent.created_at.asc())
            .all()
        )
        return {"experiment_id": experiment_id, "node": node, "events": [_row_to_dict(e) for e in events]}


@router.get("/api/observability/summary")
def get_observability_summary(
    current_user: CurrentUser = Depends(get_current_user),
):
    """Aggregated stats across the user's visible experiments."""
    with get_db_session() as session:
        org_user_ids = select(User.id).where(
            User.organization_id == current_user.organization_id
        ).scalar_subquery()
        experiments = (
            session.query(Experiment)
            .filter(
                or_(
                    Experiment.user_id == current_user.id,
                    and_(Experiment.shared_with_org.is_(True), Experiment.user_id.in_(org_user_ids)),
                    Experiment.user_id.is_(None),
                )
            )
            .order_by(Experiment.updated_at.desc())
            .limit(50)
            .all()
        )

        rows = []
        for exp in experiments:
            job = (
                session.query(TrainingJob)
                .filter_by(experiment_id=exp.id)
                .order_by(TrainingJob.started_at.desc())
                .first()
            )

            step_count = (
                session.query(sa_func.count(StepEvent.id))
                .filter_by(experiment_id=exp.id, event_type="step.complete")
                .scalar()
            ) or 0

            total_duration = (
                session.query(sa_func.sum(StepEvent.duration_ms))
                .filter_by(experiment_id=exp.id)
                .filter(StepEvent.duration_ms.isnot(None))
                .scalar()
            )

            metrics: dict[str, Any] = {}
            if job:
                mv = (
                    session.query(ModelVersion)
                    .filter_by(training_run_id=job.id, is_current=True)
                    .first()
                )
                if mv and mv.metrics:
                    metrics = mv.metrics

            rows.append({
                "experiment_id": exp.id,
                "name": exp.name,
                "status": exp.status,
                "goal": exp.goal,
                "created_at": exp.created_at.isoformat() if exp.created_at else None,
                "updated_at": exp.updated_at.isoformat() if exp.updated_at else None,
                "steps_completed": step_count,
                "total_duration_ms": total_duration,
                "job_status": job.status if job else None,
                "metrics": metrics,
            })

        per_step_avg = (
            session.query(
                StepEvent.node,
                sa_func.avg(StepEvent.duration_ms).label("avg_ms"),
                sa_func.count(StepEvent.id).label("count"),
            )
            .filter(StepEvent.event_type == "step.complete", StepEvent.duration_ms.isnot(None))
            .group_by(StepEvent.node)
            .all()
        )

        step_averages = {
            row.node: {"avg_duration_ms": round(row.avg_ms) if row.avg_ms else None, "count": row.count}
            for row in per_step_avg
        }

        return {"experiments": rows, "step_averages": step_averages}
