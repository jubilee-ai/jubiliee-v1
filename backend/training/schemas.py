from typing import Optional

from pydantic import BaseModel


class TrainRequest(BaseModel):
    goal: str
    linked_datasets: Optional[list[str]] = None
    user_model_preference: Optional[str] = None
    experiment_id: Optional[str] = None
    hitl: bool = True


class ResumeRequest(BaseModel):
    thread_id: str
    approved: bool = True
    feedback: Optional[str] = None


class TrainResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    current_step: Optional[str] = None
    state: Optional[dict] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
