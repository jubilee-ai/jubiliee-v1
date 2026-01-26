"""
FastAPI Backend for Jubilee Training Agent
Exposes the training agent functionality to the frontend UI
"""

import json
import sys
import uuid
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add data-tools to path for registry
DATA_TOOLS_DIR = PROJECT_ROOT / "tools" / "data-tools"
sys.path.insert(0, str(DATA_TOOLS_DIR))

# Import the training agent
from agents.training.agent import (
    invoke_training_agent, 
    stream_training_agent_with_updates,
    TrainingAgentState
)

# Import dataset utilities
from utils import register_dataset, get_registered_dataset

# Import dataset catalog
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATASETS_CATALOG_PATH = DATASETS_DIR / "catalog.json"
MODELS_REGISTRY_PATH = PROJECT_ROOT / "trained_models" / "registry.json"


# ============================================================================
# In-memory storage for training jobs
# ============================================================================
training_jobs: dict[str, dict[str, Any]] = {}


# ============================================================================
# Pydantic Models
# ============================================================================

class TrainRequest(BaseModel):
    goal: str
    linked_datasets: Optional[list[str]] = None
    user_model_preference: Optional[str] = None


class TrainResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "running", "completed", "error"
    progress: int  # 0-100
    current_step: Optional[str] = None
    state: Optional[dict] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    print("🚀 Jubilee Training Agent API starting...")
    yield
    print("👋 Jubilee Training Agent API shutting down...")


app = FastAPI(
    title="Jubilee Training Agent API",
    description="API for the ML Training Agent",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Helper Functions
# ============================================================================

def serialize_state(state: TrainingAgentState) -> dict:
    """Convert TrainingAgentState to JSON-serializable dict"""
    # The state is already a TypedDict, so we can convert directly
    # but we need to handle any non-serializable types
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    return make_serializable(dict(state))


def load_and_register_dataset(file_path: str) -> Optional[str]:
    """
    Load a dataset from file and register it in the registry.
    
    Args:
        file_path: Path relative to datasets/ folder (e.g., "csv/Loan_default.csv")
        
    Returns:
        The registered dataset reference, or None if failed
    """
    try:
        # Handle SQL files by trying to find the CSV equivalent
        original_path = file_path
        if file_path.endswith(".sql"):
            # Try to find the CSV version
            # sql/insurance.sql -> csv/insurance.csv
            # sql/loan_default.sql -> csv/Loan_default.csv (case might vary)
            base_name = Path(file_path).stem  # e.g., "insurance" or "loan_default"
            
            # Try common CSV paths
            csv_candidates = [
                f"csv/{base_name}.csv",
                f"csv/{base_name.title()}.csv",
                f"csv/{base_name.replace('_', ' ').title().replace(' ', '_')}.csv",
                f"csv/{base_name.capitalize()}.csv",
            ]
            
            found_csv = None
            for candidate in csv_candidates:
                candidate_path = DATASETS_DIR / candidate
                if candidate_path.exists():
                    found_csv = candidate
                    break
            
            if found_csv:
                print(f"[DEBUG] Converting SQL path to CSV: {file_path} -> {found_csv}")
                file_path = found_csv
            else:
                # Try to list the csv directory and find a match
                csv_dir = DATASETS_DIR / "csv"
                if csv_dir.exists():
                    for csv_file in csv_dir.glob("*.csv"):
                        if base_name.lower() in csv_file.stem.lower():
                            file_path = f"csv/{csv_file.name}"
                            print(f"[DEBUG] Found matching CSV: {file_path}")
                            break
        
        # Build full path
        full_path = DATASETS_DIR / file_path
        
        if not full_path.exists():
            print(f"[DEBUG] Dataset file not found: {full_path}")
            print(f"[DEBUG] Original path was: {original_path}")
            return None
        
        # Generate a reference name from the file
        ref_name = file_path.replace("/", "_").replace(".csv", "").replace(".parquet", "").replace(".sql", "")
        
        # Check if already registered
        existing = get_registered_dataset(ref_name)
        if existing is not None:
            print(f"[DEBUG] Dataset '{ref_name}' already registered with {len(existing)} rows")
            return ref_name
        
        # Load the dataset
        if file_path.endswith(".csv"):
            df = pd.read_csv(full_path)
        elif file_path.endswith(".parquet"):
            df = pd.read_parquet(full_path)
        else:
            print(f"[DEBUG] Unsupported file format: {file_path}")
            return None
        
        # Register the dataset
        register_dataset(ref_name, df)
        print(f"[DEBUG] Registered dataset '{ref_name}' with {len(df)} rows and {len(df.columns)} columns")
        
        return ref_name
        
    except Exception as e:
        print(f"[DEBUG] Error loading dataset {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_training_sync(job_id: str, goal: str, linked_datasets: Optional[list[str]], model_pref: Optional[str]):
    """Run the training agent synchronously (called in background)"""
    try:
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["current_step"] = "data_collection"
        
        print(f"[DEBUG] Starting training job {job_id}")
        print(f"[DEBUG] Goal: {goal}")
        print(f"[DEBUG] Linked datasets input: {linked_datasets}")
        print(f"[DEBUG] Model preference: {model_pref}")
        
        # Pre-register linked datasets so the agent can find them
        registered_refs = []
        if linked_datasets:
            for dataset_path in linked_datasets:
                print(f"[DEBUG] Attempting to load dataset: {dataset_path}")
                ref = load_and_register_dataset(dataset_path)
                if ref:
                    registered_refs.append(ref)
                    print(f"[DEBUG] SUCCESS - Pre-registered dataset: {dataset_path} -> {ref}")
                else:
                    print(f"[DEBUG] FAILED - Could not load dataset: {dataset_path}")
        
        print(f"[DEBUG] Registered refs to pass to agent: {registered_refs}")
        
        # Update step
        training_jobs[job_id]["current_step"] = "select_model"
        
        # Run the actual training agent with the registered references
        # IMPORTANT: Always pass the list, even if empty, so agent knows datasets were intended
        final_linked_datasets = registered_refs if registered_refs else linked_datasets
        print(f"[DEBUG] Calling invoke_training_agent with linked_datasets={final_linked_datasets}")
        
        result = invoke_training_agent(
            goal=goal,
            linked_datasets=final_linked_datasets,
            user_model_preference=model_pref,
        )
        
        # Update job with results
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["progress"] = 100
        training_jobs[job_id]["current_step"] = "completed"
        training_jobs[job_id]["state"] = serialize_state(result)
        training_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        training_jobs[job_id]["status"] = "error"
        training_jobs[job_id]["error"] = str(e)
        training_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        print(f"Training error for job {job_id}: {e}")


# ============================================================================
# API Routes
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "Jubilee Training Agent API"}


@app.get("/api/datasets")
async def get_datasets():
    """Get available datasets from catalog (only CSV/parquet files that can be loaded)"""
    try:
        if DATASETS_CATALOG_PATH.exists():
            with open(DATASETS_CATALOG_PATH) as f:
                catalog = json.load(f)
            
            datasets = catalog.get("datasets", [])
            
            # Filter to only include CSV and parquet files (not SQL or Python loaders)
            loadable_datasets = [
                ds for ds in datasets
                if ds.get("format", "").upper() in ("CSV", "PARQUET")
                or ds.get("file", "").endswith(".csv")
                or ds.get("file", "").endswith(".parquet")
            ]
            
            return loadable_datasets
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
async def get_models():
    """Get available model types"""
    return [
        {"id": "logistic_regression", "name": "Logistic Regression", "description": "Binary/multiclass classification, interpretable"},
        {"id": "random_forest", "name": "Random Forest", "description": "Classification/regression, feature importance"},
        {"id": "xgboost", "name": "XGBoost", "description": "High-performance tabular data"},
        {"id": "glm", "name": "GLM", "description": "Poisson/Gamma/Tweedie regression"},
        {"id": "survival_analysis", "name": "Survival Analysis", "description": "Time-to-event prediction with censoring"},
    ]


@app.get("/api/trained-models")
async def get_trained_models():
    """Get previously trained models from registry"""
    try:
        if MODELS_REGISTRY_PATH.exists():
            with open(MODELS_REGISTRY_PATH) as f:
                registry = json.load(f)
            return registry.get("models", {})
        return {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train", response_model=TrainResponse)
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Start a new training job.
    Returns immediately with a job_id that can be used to poll for status.
    """
    job_id = str(uuid.uuid4())
    
    # Initialize job
    training_jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "current_step": None,
        "state": None,
        "error": None,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "goal": request.goal,
        "linked_datasets": request.linked_datasets,
        "model_preference": request.user_model_preference,
    }
    
    # Run training in background thread (not async, since the agent is synchronous)
    import threading
    thread = threading.Thread(
        target=run_training_sync,
        args=(job_id, request.goal, request.linked_datasets, request.user_model_preference),
        daemon=True,
    )
    thread.start()
    
    return TrainResponse(
        job_id=job_id,
        status="pending",
        message="Training job started. Poll /api/train/{job_id} for status.",
    )


@app.get("/api/train/{job_id}", response_model=JobStatus)
async def get_training_status(job_id: str):
    """Get the status of a training job"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = training_jobs[job_id]
    
    # Estimate progress based on step (since we can't get intermediate state from the agent)
    step_progress = {
        "select_model": 10,
        "data_collection": 20,
        "cleaning": 35,
        "label_split_definition": 45,
        "feature_selection_specification": 55,
        "feature_engineering_executor": 70,
        "human_confirmation": 80,
        "training": 90,
        "generate_report": 95,
        "completed": 100,
    }
    
    progress = step_progress.get(job.get("current_step", ""), 0)
    if job["status"] == "completed":
        progress = 100
    elif job["status"] == "error":
        progress = 0
    
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=progress,
        current_step=job.get("current_step"),
        state=job.get("state"),
        error=job.get("error"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
    )


@app.post("/api/train-sync")
async def train_sync(request: TrainRequest):
    """
    Run training synchronously (blocks until complete).
    Use this for simpler integrations, but it may timeout for long runs.
    """
    try:
        # Pre-register linked datasets
        registered_refs = []
        if request.linked_datasets:
            for dataset_path in request.linked_datasets:
                ref = load_and_register_dataset(dataset_path)
                if ref:
                    registered_refs.append(ref)
        
        result = invoke_training_agent(
            goal=request.goal,
            linked_datasets=registered_refs if registered_refs else None,
            user_model_preference=request.user_model_preference,
        )
        return {
            "status": "completed",
            "state": serialize_state(result),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/train/{job_id}")
async def cancel_training(job_id: str):
    """Cancel/delete a training job"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Note: This doesn't actually stop a running thread, just removes from tracking
    del training_jobs[job_id]
    return {"status": "deleted", "job_id": job_id}


# ============================================================================
# SSE Streaming Endpoint
# ============================================================================

def generate_sse_events(goal: str, linked_datasets: Optional[list[str]], model_pref: Optional[str]):
    """
    Generator that yields SSE-formatted events from the training agent.
    """
    # Pre-register linked datasets
    registered_refs = []
    if linked_datasets:
        for dataset_path in linked_datasets:
            ref = load_and_register_dataset(dataset_path)
            if ref:
                registered_refs.append(ref)
                yield f"data: {json.dumps({'type': 'dataset_loaded', 'dataset': dataset_path, 'ref': ref})}\n\n"
    
    final_linked = registered_refs if registered_refs else linked_datasets
    
    try:
        for update in stream_training_agent_with_updates(
            goal=goal,
            linked_datasets=final_linked,
            user_model_preference=model_pref,
        ):
            # Serialize the update, handling non-serializable values
            serialized_update = serialize_state(update)
            yield f"data: {json.dumps(serialized_update)}\n\n"
            
    except Exception as e:
        error_event = {
            "type": "error",
            "error": str(e),
        }
        yield f"data: {json.dumps(error_event)}\n\n"


@app.post("/api/train-stream")
async def train_stream(request: TrainRequest):
    """
    Stream training progress via Server-Sent Events (SSE).
    
    Returns a stream of JSON events:
    - {type: "started", progress: 0}
    - {type: "node_complete", node: "select_model", progress: 10, state: {...}, summary: {...}}
    - {type: "node_complete", node: "data_collection", progress: 20, ...}
    - ...
    - {type: "completed", progress: 100}
    
    Use EventSource in JavaScript to consume this stream.
    """
    return StreamingResponse(
        generate_sse_events(
            request.goal,
            request.linked_datasets,
            request.user_model_preference,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


# ============================================================================
# Run with: uvicorn app:app --reload
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
