from typing import Optional

from pydantic import BaseModel, Field


class CreateExperimentRequest(BaseModel):
    name: Optional[str] = None
    linked_datasets: Optional[list[str]] = None


class UpdateExperimentRequest(BaseModel):
    name: Optional[str] = None
    goal: Optional[str] = None
    status: Optional[str] = None
    linked_datasets: Optional[list[str]] = None
    training_state_merge: Optional[dict] = Field(
        default=None,
        description="Merged into experiments.training_state (top-level keys).",
    )


class AsyncTrainRequest(BaseModel):
    user_model_preference: Optional[str] = None
    conversation: Optional[list[dict]] = None


class ExperimentSummary(BaseModel):
    id: str
    name: str
    goal: Optional[str] = None
    status: str
    chat_thread_id: str
    linked_datasets: Optional[list] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_message: Optional[str] = None
    lab_mode: Optional[str] = None
    task_status: Optional[str] = None


class ExperimentDetail(ExperimentSummary):
    chat_history: list = []
    training_state: Optional[dict] = None
    training_context: Optional[dict] = None
