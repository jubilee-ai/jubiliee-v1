import uuid
from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel

from backend.catalog.repository import _dataset_to_dict
from backend.experiments.schemas import (
    AsyncTrainRequest,
    CreateExperimentRequest,
    ExperimentDetail,
    ExperimentSummary,
    UpdateExperimentRequest,
)
from backend.shared.auth import ClerkUser, require_org
from backend.shared.database import get_db_session
from backend.shared.models import (
    Dataset,
    Model,
    ModelVersion,
    RunDatasetLink,
    TrainingJob,
)
from backend.training import repository
from backend.training import service as training_service

router = APIRouter()


class SaveMessagesRequest(BaseModel):
    messages: list[dict]


@router.post("/api/experiments", response_model=ExperimentSummary)
def create_experiment(
    request: CreateExperimentRequest,
    user: ClerkUser = Depends(require_org),
):
    experiment_id = f"exp-{uuid.uuid4().hex[:8]}"
    chat_thread_id = f"chat-{experiment_id}"
    name = request.name or f"Experiment {experiment_id[-8:]}"
    result = repository.create_experiment(
        experiment_id=experiment_id,
        name=name,
        chat_thread_id=chat_thread_id,
        linked_datasets=request.linked_datasets,
        org_id=user.org_id,
        created_by=user.user_id,
    )
    return result


@router.get("/api/experiments", response_model=list[ExperimentSummary])
def list_experiments(user: ClerkUser = Depends(require_org)):
    return repository.list_experiments(org_id=user.org_id)


@router.get("/api/experiments/{experiment_id}", response_model=ExperimentDetail)
def get_experiment(
    experiment_id: str,
    user: ClerkUser = Depends(require_org),
):
    result = repository.get_experiment(experiment_id, org_id=user.org_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return result


@router.patch("/api/experiments/{experiment_id}")
def update_experiment(
    experiment_id: str,
    request: UpdateExperimentRequest,
    user: ClerkUser = Depends(require_org),
):
    updates = request.model_dump(exclude_none=True)
    merge_patch = updates.pop("training_state_merge", None)
    did_something = False
    if merge_patch is not None:
        if not repository.merge_experiment_training_state(
            experiment_id, merge_patch, org_id=user.org_id
        ):
            raise HTTPException(status_code=404, detail="Experiment not found")
        did_something = True
    if updates:
        if not repository.update_experiment(
            experiment_id, updates, org_id=user.org_id
        ):
            raise HTTPException(status_code=404, detail="Experiment not found")
        did_something = True
    if not did_something:
        raise HTTPException(status_code=400, detail="No fields to update")
    return {"status": "ok"}


@router.post("/api/experiments/{experiment_id}/async-train")
def start_async_training(
    experiment_id: str,
    request: AsyncTrainRequest = Body(default_factory=AsyncTrainRequest),
    user: ClerkUser = Depends(require_org),
):
    if not repository.get_experiment(experiment_id, org_id=user.org_id):
        raise HTTPException(status_code=404, detail="Experiment not found")
    body = request
    try:
        training_service.start_experiment_async_training(
            experiment_id,
            model_pref=body.user_model_preference,
            conversation=body.conversation,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Experiment not found")
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"status": "ok", "experiment_id": experiment_id}


@router.delete("/api/experiments/{experiment_id}")
def delete_experiment(
    experiment_id: str,
    user: ClerkUser = Depends(require_org),
):
    ok = repository.delete_experiment(experiment_id, org_id=user.org_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"status": "ok"}


@router.put("/api/experiments/{experiment_id}/messages")
def save_messages(
    experiment_id: str,
    request: SaveMessagesRequest,
    user: ClerkUser = Depends(require_org),
):
    exp = repository.get_experiment(experiment_id, org_id=user.org_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    repository.save_experiment_chat_history(
        experiment_id, request.messages, org_id=user.org_id,
    )
    return {"status": "ok", "count": len(request.messages)}


@router.get("/api/experiments/{experiment_id}/artifacts")
def get_experiment_artifacts(
    experiment_id: str,
    user: ClerkUser = Depends(require_org),
):
    if not repository.get_experiment(experiment_id, org_id=user.org_id):
        raise HTTPException(status_code=404, detail="Experiment not found")
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
