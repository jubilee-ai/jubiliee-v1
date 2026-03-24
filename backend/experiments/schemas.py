from typing import Optional

from pydantic import BaseModel


class CreateExperimentRequest(BaseModel):
    name: Optional[str] = None
    linked_datasets: Optional[list[str]] = None


class UpdateExperimentRequest(BaseModel):
    name: Optional[str] = None
    goal: Optional[str] = None
    status: Optional[str] = None
    linked_datasets: Optional[list[str]] = None


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


class ExperimentDetail(ExperimentSummary):
    chat_history: list = []
    training_state: Optional[dict] = None
    training_context: Optional[dict] = None
