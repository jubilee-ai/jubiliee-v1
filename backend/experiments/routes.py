import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.experiments.schemas import (
    CreateExperimentRequest,
    ExperimentDetail,
    ExperimentSummary,
    UpdateExperimentRequest,
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
