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

# Import the training agent
from agents.training.agent import (TrainingAgentState, invoke_training_agent,
                                   resume_training_agent,
                                   stream_training_agent_with_updates)
from agents.training.utils.streaming import build_node_update, extract_interrupt_info

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
    simple: bool = False
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

def generate_sse_events(goal: str, linked_datasets: Optional[list[str]], model_pref: Optional[str], thread_id: Optional[str] = None):
    """
    Generator that yields SSE-formatted events from the training agent.
    With HITL, will yield interrupt events that pause for user approval.
    """
    import uuid

    # Generate thread_id if not provided
    if not thread_id:
        thread_id = f"training-{uuid.uuid4().hex[:8]}"
    
    # Pre-register linked datasets
    registered_refs = []
    if linked_datasets:
        for dataset_path in linked_datasets:
            ref = load_and_register_dataset(dataset_path)
            if ref:
                registered_refs.append(ref)
                yield f"data: {json.dumps({'type': 'dataset_loaded', 'dataset': dataset_path, 'ref': ref, 'thread_id': thread_id})}\n\n"
    
    final_linked = registered_refs if registered_refs else linked_datasets
    
    try:
        for update in stream_training_agent_with_updates(
            goal=goal,
            linked_datasets=final_linked,
            user_model_preference=model_pref,
            thread_id=thread_id,
        ):
            # Add thread_id to all updates
            update["thread_id"] = thread_id
            
            # Serialize the update, handling non-serializable values
            serialized_update = serialize_state(update)
            yield f"data: {json.dumps(serialized_update)}\n\n"
            
            # If this is an interrupt event, stop streaming (frontend will resume)
            if update.get("type") == "interrupt":
                return
            
    except Exception as e:
        error_event = {
            "type": "error",
            "error": str(e),
            "thread_id": thread_id,
        }
        yield f"data: {json.dumps(error_event)}\n\n"


def generate_resume_sse_events(thread_id: str, approved: bool, feedback: Optional[str]):
    """
    Generator that yields SSE-formatted events when resuming after an interrupt.
    """
    try:
        # Build the decision object
        if approved:
            decision = {"approved": True}
        else:
            decision = {"approved": False, "feedback": feedback or "Please redo this step."}
        
        # Resume the agent - this returns a generator
        from agents.training.agent import \
            stream_resume_training_agent_with_updates
        
        for update in stream_resume_training_agent_with_updates(
            decision=decision,
            thread_id=thread_id,
        ):
            # Add thread_id to all updates
            update["thread_id"] = thread_id
            
            # Serialize the update
            serialized_update = serialize_state(update)
            yield f"data: {json.dumps(serialized_update)}\n\n"
            
            # If this is an interrupt event, stop streaming (frontend will resume again)
            if update.get("type") == "interrupt":
                return
            
    except Exception as e:
        error_event = {
            "type": "error",
            "error": str(e),
            "thread_id": thread_id,
        }
        yield f"data: {json.dumps(error_event)}\n\n"


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

        yield f"data: {json.dumps({'type': 'completed', 'node': 'end', 'progress': 100, 'message': 'Training completed', 'thread_id': thread_id})}\n\n"

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'error': str(e), 'thread_id': thread_id})}\n\n"


@app.post("/api/train-stream")
async def train_stream(request: TrainRequest):
    """
    Stream training progress via Server-Sent Events (SSE).
    
    With HITL enabled, returns events until an interrupt is hit:
    - {type: "started", progress: 0, thread_id: "xxx"}
    - {type: "node_complete", node: "select_model", ...}
    - {type: "interrupt", node: "select_model", summary: "...", thread_id: "xxx"}
    
    Set simple=true to use the deep-agent pipeline instead of the graph agent.
    Set hitl=false to run without human-in-the-loop interrupts.
    When interrupt is received, call /api/train-resume to continue.
    """
    if request.simple:
        generator = generate_simple_sse_events(
            request.goal,
            request.linked_datasets,
            request.user_model_preference,
            hitl=request.hitl,
        )
    else:
        generator = generate_sse_events(
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


@app.post("/api/train-resume")
async def train_resume(request: ResumeRequest):
    """
    Resume training after an interrupt (human-in-the-loop).
    
    Streams the next step's progress until the next interrupt or completion.
    Automatically detects whether the thread belongs to the simple or complex agent.
    """
    if request.thread_id in simple_agent_store:
        generator = generate_simple_resume_sse_events(
            request.thread_id,
            request.approved,
            request.feedback,
        )
    else:
        generator = generate_resume_sse_events(
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
# Run with: uvicorn app:app --reload
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
