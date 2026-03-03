from collections.abc import Generator
from typing import Protocol, Optional

from backend.training.schemas import JobStatus, TrainRequest, TrainResponse


class TrainingServiceInterface(Protocol):
    def start_training(self, request: TrainRequest) -> TrainResponse: ...
    def get_training_status(self, job_id: str) -> JobStatus: ...
    def train_sync(self, request: TrainRequest) -> dict[str, object]: ...
    def cancel_training(self, job_id: str) -> dict[str, str]: ...
    def generate_simple_sse_events(
        self,
        goal: str,
        linked_datasets: Optional[list[str]],
        model_pref: Optional[str],
        hitl: bool = True,
        thread_id: Optional[str] = None,
    ) -> Generator[str, None, None]: ...
    def generate_simple_resume_sse_events(
        self,
        thread_id: str,
        approved: bool,
        feedback: Optional[str],
    ) -> Generator[str, None, None]: ...


class TrainingRepositoryInterface(Protocol):
    def thread_exists(self, thread_id: str) -> bool: ...
