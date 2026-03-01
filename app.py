"""
FastAPI Backend for Jubilee Training Agent
Exposes the training agent functionality to the frontend UI
"""

import json
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add data-tools to path for registry
DATA_TOOLS_DIR = PROJECT_ROOT / "tools" / "data-tools"
sys.path.insert(0, str(DATA_TOOLS_DIR))

# Import dataset utilities
from utils import get_registered_dataset, register_dataset

# Import training agent utilities (simple agent only)
from agents.training.core.state import TrainingAgentState
from agents.training.utils.streaming import build_node_update, extract_interrupt_info

# Import the main orchestrator agent (use importlib to avoid collision with
# agents/data-retrieval/agent.py which is also on sys.path)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("orchestrator_agent_mod", PROJECT_ROOT / "agent.py")
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
orchestrator_agent = _mod.agent

# Import dataset catalog
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATASETS_CATALOG_PATH = DATASETS_DIR / "catalog.json"
MODELS_REGISTRY_PATH = PROJECT_ROOT / "trained_models" / "registry.json"


# ============================================================================
# In-memory storage for training jobs
# ============================================================================
training_jobs: dict[str, dict[str, Any]] = {}

# Storage for simple agent instances (keyed by thread_id)
simple_agent_store: dict[str, dict[str, Any]] = {}

# Last completed training context, injected into the next chat so the
# orchestrator knows which model was "just trained".
_last_training_context: dict[str, Any] = {}
# Chat threads that have already received the training context injection.
_chat_threads_with_context: set[str] = set()

TOOL_TO_STEP = {
    "tool_data_collection": "data_collection",
    "tool_select_model": "select_model",
    "tool_cleaning": "cleaning",
    "tool_label_split_definition": "label_split_definition",
    "tool_feature_selection_specification": "feature_selection_specification",
    "tool_feature_engineering_executor": "feature_engineering_executor",
    "tool_training_approval": "training_approval",
    "tool_training": "training",
    "tool_generate_report": "generate_report",
}


# ============================================================================
# Pydantic Models
# ============================================================================

class TrainRequest(BaseModel):
    goal: str
    linked_datasets: Optional[list[str]] = None
    user_model_preference: Optional[str] = None
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
    status: str  # "pending", "running", "completed", "error"
    progress: int  # 0-100
    current_step: Optional[str] = None
    state: Optional[dict] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    training_context: Optional[str] = None


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
    import math
    import numpy as np

    def make_serializable(obj):
        if obj is None:
            return None
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        elif isinstance(obj, np.ndarray):
            return [make_serializable(x) for x in obj.tolist()]
        elif isinstance(obj, float):
            return None if (math.isnan(obj) or math.isinf(obj)) else obj
        elif isinstance(obj, (str, int, bool)):
            return obj
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
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
    """Run the simple training agent synchronously (called in background)."""
    from agents.training.agent_simple import invoke_simple_training_agent

    try:
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["current_step"] = "data_collection"

        registered_refs = []
        if linked_datasets:
            for dataset_path in linked_datasets:
                ref = load_and_register_dataset(dataset_path)
                if ref:
                    registered_refs.append(ref)

        training_jobs[job_id]["current_step"] = "select_model"

        final_linked_datasets = registered_refs if registered_refs else linked_datasets

        result = invoke_simple_training_agent(
            goal=goal,
            linked_datasets=final_linked_datasets,
            user_model_preference=model_pref,
        )

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
        {"id": "naive_bayes", "name": "Naive Bayes", "description": "Fast probabilistic classifier, great baseline"},
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
        "training": 85,
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
    from agents.training.agent_simple import invoke_simple_training_agent

    try:
        registered_refs = []
        if request.linked_datasets:
            for dataset_path in request.linked_datasets:
                ref = load_and_register_dataset(dataset_path)
                if ref:
                    registered_refs.append(ref)

        result = invoke_simple_training_agent(
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

def _extract_simple_interrupt(interrupt_data: list, thread_id: str | None = None) -> dict:
    """Parse interrupt data from manual interrupt() calls in agent_simple tools.

    Each tool calls interrupt({node, summary, message}) after completing its
    work, so the value is a simple dict we pass through to the UI.
    Also stores all interrupt IDs for proper multi-interrupt resume.
    """
    default = {"node": "unknown", "summary": "Step pending approval"}

    if not interrupt_data:
        return default

    # Collect all interrupt IDs for resume
    interrupt_ids = []
    for item in interrupt_data:
        iid = getattr(item, "id", None)
        if iid:
            interrupt_ids.append(iid)

    if thread_id and thread_id in simple_agent_store:
        simple_agent_store[thread_id]["interrupt_ids"] = interrupt_ids

    # Find the first interrupt with our custom {node, summary} format
    for item in interrupt_data:
        val = item.value if hasattr(item, "value") else item
        if isinstance(val, dict) and "node" in val:
            return {
                "node": val.get("node", "unknown"),
                "summary": val.get("summary", ""),
                "message": val.get("message", "Approve to continue, or provide feedback to redo."),
            }

    return default


def _should_skip_tool_message(content: str) -> bool:
    """Return True if a tool message should not be emitted as a node_complete event."""
    if not isinstance(content, str):
        return False
    return content.startswith("REJECTED") or content.startswith("SKIP:")


def generate_simple_sse_events(
    goal: str,
    linked_datasets: Optional[list[str]],
    model_pref: Optional[str],
    hitl: bool = True,
    thread_id: Optional[str] = None,
):
    """SSE generator for the simple deep-agent pipeline."""
    from agents.training.agent_simple import create_simple_training_agent
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    thread_id = thread_id or f"simple-{uuid.uuid4().hex[:8]}"

    registered_refs = []
    if linked_datasets:
        for dataset_path in linked_datasets:
            ref = load_and_register_dataset(dataset_path)
            if ref:
                registered_refs.append(ref)
                yield f"data: {json.dumps({'type': 'dataset_loaded', 'dataset': dataset_path, 'ref': ref, 'thread_id': thread_id})}\n\n"

    final_linked = registered_refs if registered_refs else linked_datasets

    checkpointer = MemorySaver()
    agent, shared_state = create_simple_training_agent(
        goal=goal,
        linked_datasets=final_linked,
        user_model_preference=model_pref,
        hitl=hitl,
        checkpointer=checkpointer,
    )

    simple_agent_store[thread_id] = {
        "agent": agent,
        "state": shared_state,
        "checkpointer": checkpointer,
    }

    config = {"configurable": {"thread_id": thread_id}}
    emitted_steps: set[str] = set()
    simple_agent_store[thread_id]["emitted_steps"] = emitted_steps

    yield f"data: {json.dumps({'type': 'started', 'node': 'init', 'progress': 0, 'message': 'Simple agent started', 'thread_id': thread_id})}\n\n"

    try:
        for event in agent.stream(
            {"messages": [{"role": "user", "content": goal}]},
            config=config,
            stream_mode="updates",
        ):
            if "__interrupt__" in event:
                info = _extract_simple_interrupt(event["__interrupt__"], thread_id)
                interrupt_event = {
                    "type": "interrupt",
                    "thread_id": thread_id,
                    **info,
                    "state_snapshot": serialize_state(shared_state),
                }
                yield f"data: {json.dumps(serialize_state(interrupt_event))}\n\n"
                return

            if "tools" in event:
                tool_msgs = event["tools"].get("messages", [])
                if tool_msgs:
                    content = getattr(tool_msgs[0], "content", "")
                    if _should_skip_tool_message(content):
                        continue
                    tool_name = getattr(tool_msgs[0], "name", "unknown")
                    step_name = TOOL_TO_STEP.get(tool_name, tool_name)
                    if step_name in emitted_steps:
                        continue
                    emitted_steps.add(step_name)
                    update = build_node_update(step_name, dict(shared_state))
                    update["thread_id"] = thread_id
                    yield f"data: {json.dumps(serialize_state(update))}\n\n"

        # Capture training results so the chat agent can reference them
        _save_training_context(shared_state)

        yield f"data: {json.dumps({'type': 'completed', 'node': 'end', 'progress': 100, 'message': 'Training completed', 'thread_id': thread_id})}\n\n"

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'error': str(e), 'thread_id': thread_id})}\n\n"


def generate_simple_resume_sse_events(
    thread_id: str,
    approved: bool,
    feedback: Optional[str],
):
    """SSE generator for resuming the simple agent after an interrupt."""
    from langgraph.types import Command

    store = simple_agent_store.get(thread_id)
    if not store:
        yield f"data: {json.dumps({'type': 'error', 'error': 'Simple agent thread not found', 'thread_id': thread_id})}\n\n"
        return

    agent = store["agent"]
    shared_state = store["state"]
    config = {"configurable": {"thread_id": thread_id}}
    emitted_steps: set[str] = store.get("emitted_steps", set())

    single_decision = {"approved": approved}
    if not approved:
        single_decision["feedback"] = feedback or "Please redo this step."

    # Query actual pending interrupts from the checkpoint state
    try:
        state_snapshot = agent.get_state(config)
        interrupt_ids = [
            intr.id
            for task in (state_snapshot.tasks or [])
            for intr in (task.interrupts or [])
        ]
    except Exception:
        interrupt_ids = store.get("interrupt_ids", [])

    if len(interrupt_ids) > 1:
        resume_value = {iid: single_decision for iid in interrupt_ids}
    else:
        resume_value = single_decision

    try:
        for event in agent.stream(
            Command(resume=resume_value),
            config=config,
            stream_mode="updates",
        ):
            if "__interrupt__" in event:
                info = _extract_simple_interrupt(event["__interrupt__"], thread_id)
                interrupt_event = {
                    "type": "interrupt",
                    "thread_id": thread_id,
                    **info,
                    "state_snapshot": serialize_state(shared_state),
                }
                yield f"data: {json.dumps(serialize_state(interrupt_event))}\n\n"
                return

            if "tools" in event:
                tool_msgs = event["tools"].get("messages", [])
                if tool_msgs:
                    content = getattr(tool_msgs[0], "content", "")
                    if _should_skip_tool_message(content):
                        continue
                    tool_name = getattr(tool_msgs[0], "name", "unknown")
                    step_name = TOOL_TO_STEP.get(tool_name, tool_name)
                    if step_name in emitted_steps:
                        continue
                    emitted_steps.add(step_name)
                    store["emitted_steps"] = emitted_steps
                    update = build_node_update(step_name, dict(shared_state))
                    update["thread_id"] = thread_id
                    yield f"data: {json.dumps(serialize_state(update))}\n\n"

        _save_training_context(shared_state)

        yield f"data: {json.dumps({'type': 'completed', 'node': 'end', 'progress': 100, 'message': 'Training completed', 'thread_id': thread_id})}\n\n"

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'error': str(e), 'thread_id': thread_id})}\n\n"


@app.post("/api/train-stream")
async def train_stream(request: TrainRequest):
    """
    Stream training progress via Server-Sent Events (SSE).

    Uses the simple deep-agent pipeline. Set hitl=false to run without
    human-in-the-loop interrupts.
    When an interrupt is received, call /api/train-resume to continue.
    """
    generator = generate_simple_sse_events(
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


@app.post("/api/train-resume")
async def train_resume(request: ResumeRequest):
    """
    Resume training after an interrupt (human-in-the-loop).

    Streams the next step's progress until the next interrupt or completion.
    """
    if request.thread_id not in simple_agent_store:
        return StreamingResponse(
            iter([f"data: {json.dumps({'type': 'error', 'error': 'Thread not found', 'thread_id': request.thread_id})}\n\n"]),
            media_type="text/event-stream",
        )

    generator = generate_simple_resume_sse_events(
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


# ============================================================================
# Chat Endpoint (Orchestrator Agent)
# ============================================================================


def _save_training_context(shared_state: dict):
    """Persist a summary of the completed training so the chat agent can
    reference it when the user asks follow-up questions."""
    metrics = shared_state.get("training_metrics", {})
    label_def = shared_state.get("label_definition") or {}
    _last_training_context.clear()
    _last_training_context.update({
        "model_name": metrics.get("model_name") or shared_state.get("model_weights_path"),
        "model_type": metrics.get("model_type") or shared_state.get("selected_model"),
        "goal": shared_state.get("goal"),
        "target_column": label_def.get("target_column"),
        "val_accuracy": metrics.get("val_accuracy"),
        "val_roc_auc": metrics.get("val_roc_auc"),
        "test_accuracy": metrics.get("test_accuracy"),
        "test_roc_auc": metrics.get("test_roc_auc"),
        "val_r2": metrics.get("val_r2"),
        "test_r2": metrics.get("test_r2"),
        "test_rmse": metrics.get("test_rmse"),
        "test_mae": metrics.get("test_mae"),
        "summary": metrics.get("summary"),
        "num_iterations": metrics.get("num_iterations"),
        "report_path": shared_state.get("report_path"),
    })
    _chat_threads_with_context.clear()


def _build_training_context_message(ctx: dict) -> str:
    """Build a human-readable summary of the last training run."""
    lines = [
        "[SYSTEM CONTEXT — a model was just trained via the training pipeline]",
        f"  Model name  : {ctx.get('model_name', 'unknown')}",
        f"  Model type  : {ctx.get('model_type', 'unknown')}",
        f"  Goal        : {ctx.get('goal', 'N/A')}",
        f"  Target col  : {ctx.get('target_column', 'N/A')}",
    ]
    metric_lines = []
    for key, label in [
        ("test_accuracy", "Test Accuracy"),
        ("test_roc_auc", "Test ROC-AUC"),
        ("val_accuracy", "Val Accuracy"),
        ("val_roc_auc", "Val ROC-AUC"),
        ("val_r2", "Val R²"),
        ("test_r2", "Test R²"),
        ("test_rmse", "Test RMSE"),
        ("test_mae", "Test MAE"),
    ]:
        v = ctx.get(key)
        if v is not None:
            metric_lines.append(f"  {label:16s}: {v:.4f}" if isinstance(v, float) else f"  {label:16s}: {v}")
    if metric_lines:
        lines.append("  Metrics:")
        lines.extend(metric_lines)
    if ctx.get("num_iterations"):
        lines.append(f"  Iterations  : {ctx['num_iterations']}")
    if ctx.get("report_path"):
        lines.append(f"  Report      : {ctx['report_path']}")
    lines.append("[END CONTEXT — answer the user's question using this information]")
    return "\n".join(lines)


def _emit_training_step_events(thread_id: str):
    """Yield training step SSE events from the last orchestrator training run.

    After the orchestrator's ``train_model`` tool finishes, the final
    training state is cached in ``_mod._last_training_state``.  We use
    ``build_node_update`` to reconstruct per-step events so the frontend
    can update the progress panel and chat.
    """
    from agents.training.core.state import STEP_ORDER

    state = getattr(_mod, "_last_training_state", None)
    if not state:
        return

    yield f"data: {json.dumps({'type': 'training_started', 'thread_id': thread_id})}\n\n"

    for step_name in STEP_ORDER:
        try:
            update = build_node_update(step_name, state)
            update["thread_id"] = thread_id
            serialized = serialize_state(update)
            yield f"data: {json.dumps(serialized)}\n\n"
        except Exception:
            pass

    yield f"data: {json.dumps({'type': 'training_completed', 'thread_id': thread_id})}\n\n"


def _generate_chat_sse(thread_id: str, message: str, training_context: Optional[str] = None):
    """SSE generator that streams the orchestrator agent's response.

    The orchestrator's checkpointer (MemorySaver in agent.py) keeps full
    message history per thread, so we only pass the newest user message.

    When the orchestrator invokes ``train_model``, we emit rich per-step
    training events so the frontend can update its progress panel.
    """
    config = {"configurable": {"thread_id": thread_id}}

    # Inject training context so the orchestrator knows about the most
    # recent model trained via the training pipeline.
    context_block = training_context or ""
    if not context_block and _last_training_context and thread_id not in _chat_threads_with_context:
        context_block = _build_training_context_message(_last_training_context)

    if context_block:
        _chat_threads_with_context.add(thread_id)
        augmented_message = f"{context_block}\n\nUser message: {message}"
    else:
        augmented_message = message

    agent_input = {"messages": [{"role": "user", "content": augmented_message}]}

    yield f"data: {json.dumps({'type': 'start', 'thread_id': thread_id})}\n\n"

    try:
        for chunk in orchestrator_agent.stream(
            agent_input,
            config=config,
            stream_mode="updates",
        ):
            if "model" in chunk:
                msgs = chunk["model"].get("messages", [])
                for msg in msgs:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            yield f"data: {json.dumps({'type': 'tool_call', 'tool': tc['name'], 'args': serialize_state(tc.get('args', {})), 'thread_id': thread_id})}\n\n"
                    raw_content = getattr(msg, "content", "")
                    if isinstance(raw_content, list):
                        content = "".join(
                            block.get("text", "") if isinstance(block, dict) else str(block)
                            for block in raw_content
                        )
                    else:
                        content = raw_content or ""
                    if content:
                        yield f"data: {json.dumps({'type': 'token', 'content': content, 'thread_id': thread_id})}\n\n"

            if "tools" in chunk:
                tool_msgs = chunk["tools"].get("messages", [])
                for tm in tool_msgs:
                    name = getattr(tm, "name", "unknown")
                    snippet = getattr(tm, "content", "")
                    if len(snippet) > 500:
                        snippet = snippet[:500] + "…"
                    yield f"data: {json.dumps({'type': 'tool_result', 'tool': name, 'result': snippet, 'thread_id': thread_id})}\n\n"

                    if name == "train_model":
                        yield from _emit_training_step_events(thread_id)

        yield f"data: {json.dumps({'type': 'end', 'thread_id': thread_id})}\n\n"

    except Exception as exc:
        yield f"data: {json.dumps({'type': 'error', 'error': str(exc), 'thread_id': thread_id})}\n\n"


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Stream a chat response from the orchestrator agent via SSE.

    The orchestrator decides whether to answer directly, run analysis,
    train a model, or make predictions — then streams tokens back.

    Training with full step-by-step events still uses /api/train-stream.
    """
    thread_id = request.thread_id or f"chat-{uuid.uuid4().hex[:8]}"

    return StreamingResponse(
        _generate_chat_sse(thread_id, request.message, request.training_context),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# Run with: uvicorn app:app --reload
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
