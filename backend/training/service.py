import json
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import HTTPException

from agents.training.core.state import TrainingAgentState
from agents.training.utils.streaming import build_node_update
from backend.shared.serialization import serialize_state
from backend.shared.settings import get_settings
from backend.shared.state import TOOL_TO_STEP
from backend.training import repository
from backend.training.schemas import JobStatus, TrainRequest, TrainResponse
from utils import get_registered_dataset, register_dataset


def load_and_register_dataset(file_path: str) -> Optional[str]:
    settings = get_settings()
    datasets_dir = settings.datasets_dir
    try:
        original_path = file_path
        if file_path.endswith(".sql"):
            base_name = Path(file_path).stem
            csv_candidates = [
                f"csv/{base_name}.csv",
                f"csv/{base_name.title()}.csv",
                f"csv/{base_name.replace('_', ' ').title().replace(' ', '_')}.csv",
                f"csv/{base_name.capitalize()}.csv",
            ]
            found_csv = None
            for candidate in csv_candidates:
                candidate_path = datasets_dir / candidate
                if candidate_path.exists():
                    found_csv = candidate
                    break

            if found_csv:
                file_path = found_csv
            else:
                csv_dir = datasets_dir / "csv"
                if csv_dir.exists():
                    for csv_file in csv_dir.glob("*.csv"):
                        if base_name.lower() in csv_file.stem.lower():
                            file_path = f"csv/{csv_file.name}"
                            break

        full_path = datasets_dir / file_path
        if not full_path.exists():
            print(f"[DEBUG] Dataset file not found: {full_path}")
            print(f"[DEBUG] Original path was: {original_path}")
            return None

        ref_name = (
            file_path.replace("/", "_")
            .replace(".csv", "")
            .replace(".parquet", "")
            .replace(".sql", "")
        )
        existing = get_registered_dataset(ref_name)
        if existing is not None:
            return ref_name

        if file_path.endswith(".csv"):
            df = pd.read_csv(full_path)
        elif file_path.endswith(".parquet"):
            df = pd.read_parquet(full_path)
        else:
            return None

        register_dataset(ref_name, df)
        return ref_name
    except Exception as e:
        print(f"[DEBUG] Error loading dataset {file_path}: {e}")
        traceback.print_exc()
        return None


def run_training_sync(
    job_id: str,
    goal: str,
    linked_datasets: Optional[list[str]],
    model_pref: Optional[str],
) -> None:
    from agents.training.agent_simple import invoke_simple_training_agent

    try:
        repository.update_training_status(
            job_id, {"status": "running", "current_step": "data_collection"}
        )

        registered_refs = []
        if linked_datasets:
            for dataset_path in linked_datasets:
                ref = load_and_register_dataset(dataset_path)
                if ref:
                    registered_refs.append(ref)

        repository.update_training_status(job_id, {"current_step": "select_model"})
        final_linked_datasets = registered_refs if registered_refs else linked_datasets

        result = invoke_simple_training_agent(
            goal=goal,
            linked_datasets=final_linked_datasets,
            user_model_preference=model_pref,
        )

        repository.save_run_dataset_links(job_id, result)
        repository.update_training_status(
            job_id,
            {
                "status": "completed",
                "progress": 100,
                "current_step": "completed",
                "state": serialize_state(result),
                "completed_at": datetime.now().isoformat(),
            },
        )
    except Exception as e:
        repository.update_training_status(
            job_id,
            {
                "status": "error",
                "error": str(e),
                "completed_at": datetime.now().isoformat(),
            },
        )
        print(f"Training error for job {job_id}: {e}")


def _extract_simple_interrupt(interrupt_data: list, thread_id: str | None = None) -> dict:
    default = {"node": "unknown", "summary": "Step pending approval"}
    if not interrupt_data:
        return default

    interrupt_ids = []
    for item in interrupt_data:
        iid = getattr(item, "id", None)
        if iid:
            interrupt_ids.append(iid)
    if thread_id:
        repository.save_interrupt_ids(thread_id, interrupt_ids)

    for item in interrupt_data:
        val = item.value if hasattr(item, "value") else item
        if isinstance(val, dict) and "node" in val:
            result = {
                "node": val.get("node", "unknown"),
                "summary": val.get("summary", ""),
                "message": val.get(
                    "message", "Approve to continue, or provide feedback to redo."
                ),
            }
            snap = val.get("state_snapshot")
            if snap:
                if snap.get("plan"):
                    result["plan"] = snap["plan"]
                if snap.get("plan_strategy"):
                    result["plan_strategy"] = snap["plan_strategy"]
                if snap.get("plan_index") is not None:
                    result["plan_index"] = snap["plan_index"]
            return result
    return default


def _should_skip_tool_message(content: str) -> bool:
    return isinstance(content, str) and (
        content.startswith("REJECTED") or content.startswith("SKIP:")
    )


def generate_simple_sse_events(
    goal: str,
    linked_datasets: Optional[list[str]],
    model_pref: Optional[str],
    hitl: bool = True,
    thread_id: Optional[str] = None,
):
    from agents.training.agent_simple import create_simple_training_agent
    from langgraph.checkpoint.memory import MemorySaver

    thread_id = thread_id or f"simple-{uuid.uuid4().hex[:8]}"

    registered_refs = []
    if linked_datasets:
        for dataset_path in linked_datasets:
            ref = load_and_register_dataset(dataset_path)
            if ref:
                registered_refs.append(ref)
                payload = {
                    "type": "dataset_loaded",
                    "dataset": dataset_path,
                    "ref": ref,
                    "thread_id": thread_id,
                }
                yield f"data: {json.dumps(payload)}\n\n"

    final_linked = registered_refs if registered_refs else linked_datasets
    checkpointer = MemorySaver()
    agent, shared_state = create_simple_training_agent(
        goal=goal,
        linked_datasets=final_linked,
        user_model_preference=model_pref,
        hitl=hitl,
        checkpointer=checkpointer,
    )
    repository.put_simple_agent_store(
        thread_id, {"agent": agent, "state": shared_state, "checkpointer": checkpointer}
    )
    config = {"configurable": {"thread_id": thread_id}}
    emitted_steps: set[str] = set()
    repository.put_simple_agent_store(
        thread_id,
        {
            "agent": agent,
            "state": shared_state,
            "checkpointer": checkpointer,
            "emitted_steps": emitted_steps,
        },
    )

    started_payload = {
        "type": "started",
        "node": "init",
        "progress": 0,
        "message": "Simple agent started",
        "thread_id": thread_id,
    }
    yield f"data: {json.dumps(started_payload)}\n\n"

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

        repository.save_training_context(shared_state)
        repository.save_run_dataset_links(thread_id, shared_state)
        completed_payload = {
            "type": "completed",
            "node": "end",
            "progress": 100,
            "message": "Training completed",
            "thread_id": thread_id,
        }
        yield f"data: {json.dumps(completed_payload)}\n\n"
    except Exception as e:
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'error': str(e), 'thread_id': thread_id})}\n\n"


def generate_simple_resume_sse_events(
    thread_id: str,
    approved: bool,
    feedback: Optional[str],
):
    from langgraph.types import Command

    store = repository.get_simple_agent_store(thread_id)
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

    try:
        state_snapshot = agent.get_state(config)
        interrupt_ids = [
            intr.id
            for task in (state_snapshot.tasks or [])
            for intr in (task.interrupts or [])
        ]
    except Exception:
        interrupt_ids = store.get("interrupt_ids", [])

    resume_value = (
        {iid: single_decision for iid in interrupt_ids}
        if len(interrupt_ids) > 1
        else single_decision
    )

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

        repository.save_training_context(shared_state)
        repository.save_run_dataset_links(thread_id, shared_state)
        completed_payload = {
            "type": "completed",
            "node": "end",
            "progress": 100,
            "message": "Training completed",
            "thread_id": thread_id,
        }
        yield f"data: {json.dumps(completed_payload)}\n\n"
    except Exception as e:
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'error': str(e), 'thread_id': thread_id})}\n\n"


def generate_graph_sse_events(
    goal: str,
    linked_datasets: Optional[list[str]],
    model_pref: Optional[str],
    thread_id: Optional[str] = None,
):
    """SSE generator using the agentic graph (planner + executor + evaluator)."""
    from agents.training.core.graph import create_training_agent
    from agents.training.core.state import create_initial_state
    from langgraph.checkpoint.memory import MemorySaver

    thread_id = thread_id or f"graph-{uuid.uuid4().hex[:8]}"

    registered_refs = []
    if linked_datasets:
        for dataset_path in linked_datasets:
            ref = load_and_register_dataset(dataset_path)
            if ref:
                registered_refs.append(ref)
                yield f"data: {json.dumps({'type': 'dataset_loaded', 'dataset': dataset_path, 'ref': ref, 'thread_id': thread_id})}\n\n"

    final_linked = registered_refs if registered_refs else linked_datasets
    checkpointer = MemorySaver()
    agent = create_training_agent(checkpointer=checkpointer)
    initial_state = create_initial_state(
        goal=goal,
        linked_datasets=final_linked,
        user_model_preference=model_pref,
    )

    repository.put_simple_agent_store(
        thread_id, {"agent": agent, "checkpointer": checkpointer, "mode": "graph"}
    )

    config = {"configurable": {"thread_id": thread_id}}
    yield f"data: {json.dumps({'type': 'started', 'node': 'planner', 'progress': 0, 'message': 'Agentic pipeline started', 'thread_id': thread_id})}\n\n"

    try:
        for event in agent.stream(initial_state, config=config, stream_mode="updates"):
            if "__interrupt__" in event:
                info = _extract_simple_interrupt(event["__interrupt__"], thread_id)
                interrupt_event = {"type": "interrupt", "thread_id": thread_id, **info}
                yield f"data: {json.dumps(serialize_state(interrupt_event))}\n\n"
                return

            for node_name, node_output in event.items():
                if node_name.startswith("__"):
                    continue
                progress_payload = {
                    "type": "node_update",
                    "node": node_name,
                    "thread_id": thread_id,
                }
                if isinstance(node_output, dict):
                    if node_output.get("plan"):
                        progress_payload["plan"] = node_output["plan"]
                    if node_output.get("current_step"):
                        progress_payload["current_step"] = node_output["current_step"]
                    if node_name == "planner":
                        plan = node_output.get("plan")
                        if plan and isinstance(plan, list):
                            progress_payload["plan_steps"] = [
                                {
                                    "name": getattr(s, "step_name", s.get("step_name", "")) if isinstance(s, dict) else getattr(s, "step_name", str(s)),
                                    "rationale": getattr(s, "rationale", s.get("rationale", "")) if isinstance(s, dict) else getattr(s, "rationale", ""),
                                }
                                for s in plan
                            ]
                    if node_name == "evaluator":
                        for key in ("evaluator_decision", "evaluator_rationale"):
                            if node_output.get(key):
                                progress_payload[key] = node_output[key]
                yield f"data: {json.dumps(serialize_state(progress_payload))}\n\n"

        yield f"data: {json.dumps({'type': 'completed', 'node': 'end', 'progress': 100, 'message': 'Pipeline completed', 'thread_id': thread_id})}\n\n"
    except Exception as e:
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'error': str(e), 'thread_id': thread_id})}\n\n"


def generate_graph_resume_sse_events(
    thread_id: str,
    approved: bool,
    feedback: Optional[str],
):
    """Resume the agentic graph after a HITL interrupt."""
    from langgraph.types import Command

    store = repository.get_simple_agent_store(thread_id)
    if not store:
        yield f"data: {json.dumps({'type': 'error', 'error': 'Graph thread not found', 'thread_id': thread_id})}\n\n"
        return

    agent = store["agent"]
    config = {"configurable": {"thread_id": thread_id}}

    resume_value = {"approved": approved}
    if not approved:
        resume_value["feedback"] = feedback or "Please redo this step."

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
        resume_value = {iid: resume_value for iid in interrupt_ids}

    try:
        for event in agent.stream(Command(resume=resume_value), config=config, stream_mode="updates"):
            if "__interrupt__" in event:
                info = _extract_simple_interrupt(event["__interrupt__"], thread_id)
                interrupt_event = {"type": "interrupt", "thread_id": thread_id, **info}
                yield f"data: {json.dumps(serialize_state(interrupt_event))}\n\n"
                return

            for node_name, node_output in event.items():
                if node_name.startswith("__"):
                    continue
                progress_payload = {
                    "type": "node_update",
                    "node": node_name,
                    "thread_id": thread_id,
                }
                if isinstance(node_output, dict):
                    if node_output.get("plan"):
                        progress_payload["plan"] = node_output["plan"]
                    if node_output.get("current_step"):
                        progress_payload["current_step"] = node_output["current_step"]
                    if node_name == "planner":
                        plan = node_output.get("plan")
                        if plan and isinstance(plan, list):
                            progress_payload["plan_steps"] = [
                                {
                                    "name": getattr(s, "step_name", s.get("step_name", "")) if isinstance(s, dict) else getattr(s, "step_name", str(s)),
                                    "rationale": getattr(s, "rationale", s.get("rationale", "")) if isinstance(s, dict) else getattr(s, "rationale", ""),
                                }
                                for s in plan
                            ]
                    if node_name == "evaluator":
                        for key in ("evaluator_decision", "evaluator_rationale"):
                            if node_output.get(key):
                                progress_payload[key] = node_output[key]
                yield f"data: {json.dumps(serialize_state(progress_payload))}\n\n"

        yield f"data: {json.dumps({'type': 'completed', 'node': 'end', 'progress': 100, 'message': 'Pipeline completed', 'thread_id': thread_id})}\n\n"
    except Exception as e:
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'error': str(e), 'thread_id': thread_id})}\n\n"


def _celery_available() -> bool:
    """Check if Celery + Redis are reachable."""
    try:
        import redis as _redis
        url = __import__("os").environ.get("REDIS_URL", "redis://localhost:6379/0")
        r = _redis.from_url(url, socket_connect_timeout=1)
        r.ping()
        return True
    except Exception:
        return False


def start_training(request: TrainRequest) -> TrainResponse:
    job_id = str(uuid.uuid4())
    repository.start_training(
        job_id,
        {
            "goal": request.goal,
            "linked_datasets": request.linked_datasets,
            "model_preference": request.user_model_preference,
        },
    )

    if _celery_available():
        from backend.training.tasks import run_training_task
        run_training_task.delay(
            job_id=job_id,
            goal=request.goal,
            linked_datasets=request.linked_datasets,
            model_pref=request.user_model_preference,
        )
        return TrainResponse(
            job_id=job_id,
            status="pending",
            message="Training job queued via Celery. Poll /api/train/{job_id} for status.",
        )

    thread = threading.Thread(
        target=run_training_sync,
        args=(
            job_id,
            request.goal,
            request.linked_datasets,
            request.user_model_preference,
        ),
        daemon=True,
    )
    thread.start()
    return TrainResponse(
        job_id=job_id,
        status="pending",
        message="Training job started (thread). Poll /api/train/{job_id} for status.",
    )


def get_training_status(job_id: str) -> JobStatus:
    job = repository.get_training_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

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


def train_sync(request: TrainRequest) -> dict[str, object]:
    from agents.training.agent_simple import invoke_simple_training_agent

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
    return {"status": "completed", "state": serialize_state(result)}


def cancel_training(job_id: str) -> dict[str, str]:
    deleted = repository.cancel_training(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "deleted", "job_id": job_id}
