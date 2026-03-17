import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.training.interfaces import (
    TrainingRepositoryInterface,
    TrainingServiceInterface,
)
from backend.training import repository as training_repository
from backend.training import service as training_service
from backend.training.schemas import JobStatus, ResumeRequest, TrainRequest, TrainResponse

router = APIRouter()


def get_training_service() -> TrainingServiceInterface:
    return training_service


def get_training_repository() -> TrainingRepositoryInterface:
    return training_repository


@router.post("/api/train", response_model=TrainResponse)
async def start_training(
    request: TrainRequest,
    service: TrainingServiceInterface = Depends(get_training_service),
):
    return service.start_training(request)


@router.get("/api/train/{job_id}", response_model=JobStatus)
async def get_training_status(
    job_id: str,
    service: TrainingServiceInterface = Depends(get_training_service),
):
    return service.get_training_status(job_id)


@router.post("/api/train-sync")
async def train_sync(
    request: TrainRequest,
    service: TrainingServiceInterface = Depends(get_training_service),
):
    try:
        return service.train_sync(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/train/{job_id}")
async def cancel_training(
    job_id: str,
    service: TrainingServiceInterface = Depends(get_training_service),
):
    return service.cancel_training(job_id)


@router.post("/api/train-stream")
async def train_stream(
    request: TrainRequest,
    service: TrainingServiceInterface = Depends(get_training_service),
):
    generator = service.generate_simple_sse_events(
        request.goal,
        request.linked_datasets,
        request.user_model_preference,
        hitl=request.hitl,
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/api/train-graph")
async def train_graph_stream(
    request: TrainRequest,
    service: TrainingServiceInterface = Depends(get_training_service),
):
    """Start training via the agentic graph (planner + executor + evaluator)."""
    generator = service.generate_graph_sse_events(
        request.goal,
        request.linked_datasets,
        request.user_model_preference,
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/api/train-graph-resume")
async def train_graph_resume(
    request: ResumeRequest,
    service: TrainingServiceInterface = Depends(get_training_service),
    repository: TrainingRepositoryInterface = Depends(get_training_repository),
):
    """Resume the agentic graph after a HITL interrupt."""
    if not repository.thread_exists(request.thread_id):
        return StreamingResponse(
            iter(
                [
                    f"data: {json.dumps({'type': 'error', 'error': 'Thread not found', 'thread_id': request.thread_id})}\n\n"
                ]
            ),
            media_type="text/event-stream",
        )

    generator = service.generate_graph_resume_sse_events(
        request.thread_id,
        request.approved,
        request.feedback,
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/api/train-resume")
async def train_resume(
    request: ResumeRequest,
    service: TrainingServiceInterface = Depends(get_training_service),
    repository: TrainingRepositoryInterface = Depends(get_training_repository),
):
    if not repository.thread_exists(request.thread_id):
        return StreamingResponse(
            iter(
                [
                    f"data: {json.dumps({'type': 'error', 'error': 'Thread not found', 'thread_id': request.thread_id})}\n\n"
                ]
            ),
            media_type="text/event-stream",
        )

    generator = service.generate_simple_resume_sse_events(
        request.thread_id,
        request.approved,
        request.feedback,
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
