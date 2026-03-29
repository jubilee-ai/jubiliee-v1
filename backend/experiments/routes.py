import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.catalog.repository import _dataset_to_dict
from backend.experiments.schemas import (
    CreateExperimentRequest,
    ExperimentDetail,
    ExperimentSummary,
    UpdateExperimentRequest,
)
from backend.shared.database import get_db_session
from backend.shared.models import (
    Dataset,
    Model,
    ModelVersion,
    RunDatasetLink,
    TrainingJob,
)
from backend.training import repository

router = APIRouter()


class SaveMessagesRequest(BaseModel):
    messages: list[dict]


@router.post("/api/experiments", response_model=ExperimentSummary)
def create_experiment(request: CreateExperimentRequest):
    experiment_id = f"exp-{uuid.uuid4().hex[:8]}"
    chat_thread_id = f"chat-{experiment_id}"
    name = request.name or f"Experiment {experiment_id[-8:]}"
    result = repository.create_experiment(
        experiment_id=experiment_id,
        name=name,
        chat_thread_id=chat_thread_id,
        linked_datasets=request.linked_datasets,
    )
    return result


@router.get("/api/experiments", response_model=list[ExperimentSummary])
def list_experiments():
    return repository.list_experiments()


@router.get("/api/experiments/{experiment_id}", response_model=ExperimentDetail)
def get_experiment(experiment_id: str):
    result = repository.get_experiment(experiment_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return result


@router.patch("/api/experiments/{experiment_id}")
def update_experiment(experiment_id: str, request: UpdateExperimentRequest):
    updates = request.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    ok = repository.update_experiment(experiment_id, updates)
    if not ok:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"status": "ok"}


@router.delete("/api/experiments/{experiment_id}")
def delete_experiment(experiment_id: str):
    ok = repository.delete_experiment(experiment_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"status": "ok"}


@router.put("/api/experiments/{experiment_id}/messages")
def save_messages(experiment_id: str, request: SaveMessagesRequest):
    exp = repository.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    repository.save_experiment_chat_history(experiment_id, request.messages)
    return {"status": "ok", "count": len(request.messages)}


@router.get("/api/experiments/{experiment_id}/artifacts")
def get_experiment_artifacts(experiment_id: str):
    with get_db_session() as session:
        run_ids = [
            r.id
            for r in session.query(TrainingJob.id)
            .filter_by(experiment_id=experiment_id)
            .all()
        ]
        if not run_ids:
            return {"datasets": [], "models": []}

        dataset_ids = [
            link.dataset_id
            for link in session.query(RunDatasetLink.dataset_id)
            .filter(RunDatasetLink.training_run_id.in_(run_ids))
            .distinct()
            .all()
        ]
        datasets = []
        if dataset_ids:
            rows = session.query(Dataset).filter(Dataset.id.in_(dataset_ids)).all()
            datasets = [_dataset_to_dict(r) for r in rows]

        versions = (
            session.query(ModelVersion)
            .filter(ModelVersion.training_run_id.in_(run_ids))
            .order_by(ModelVersion.created_at.desc())
            .all()
        )
        models = []
        for v in versions:
            model = session.query(Model).filter_by(id=v.model_id).first()
            v_props = v.properties or {}
            m_props = model.properties or {} if model else {}
            models.append({
                "model_name": model.name if model else str(v.model_id),
                "model_type": m_props.get("model_type", ""),
                "version": v.version,
                "metrics": v.metrics or {},
                "storage_key": v.storage_key,
                "is_current": v.is_current,
                "created_at": v.created_at.isoformat() if v.created_at else "",
            })

        return {"datasets": datasets, "models": models}
