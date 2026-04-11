import json
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import HTTPException
from utils import get_registered_dataset, register_dataset

from agents.training.core.graph import ALL_STEP_NAMES
from agents.training.utils.streaming import (build_node_update,
                                             is_unsupervised_passthrough)
from backend.chat.events import (dataset_error, dataset_resolved, error_event,
                                 format_sse, review_required, step_complete,
                                 step_progress, step_skipped, stream_end,
                                 stream_start)
from backend.chat.events import token as token_event
from backend.shared.database import get_db_session
from backend.shared.models import Dataset as DatasetModel
from backend.shared.serialization import serialize_state
from backend.shared.settings import get_settings
from backend.shared.state import TOOL_TO_STEP
from backend.training import repository
from backend.training.schemas import JobStatus, TrainRequest, TrainResponse


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


def _try_register_from_db_row(row) -> Optional[str]:
    """Given a DB Dataset row, register it in-memory from file or R2 and return ref."""
    props = row.properties or {}
    ref_name = row.name

    file_path = props.get("file")
    if file_path:
        loaded = load_and_register_dataset(file_path)
        if loaded:
            return loaded

    storage_key = props.get("storage_key")
    if storage_key:
        try:
            from utils import _dataset_registry, _download_parquet_from_r2
            df = _download_parquet_from_r2(storage_key)
            if df is not None:
                _dataset_registry[ref_name] = df
                return ref_name
        except Exception as e:
            print(f"[resolve] R2 download failed for {ref_name}: {e}")

    if not file_path and not storage_key:
        print(f"[resolve] DB row '{ref_name}' has no file or storage_key — dataset is metadata-only")

    return None


def resolve_linked_dataset(entry: str) -> Optional[str]:
    """Try multiple strategies to resolve a dataset entry to a registered ref.

    Order: in-memory registry → filesystem path → DB name lookup → DB UUID lookup.
    """
    if get_registered_dataset(entry) is not None:
        return entry

    ref = load_and_register_dataset(entry)
    if ref:
        return ref

    def _db_lookup():
        from sqlalchemy import func as sa_func
        with get_db_session() as session:
            row = (
                session.query(DatasetModel)
                .filter(sa_func.lower(DatasetModel.name) == entry.lower())
                .first()
            )
            if row:
                return _try_register_from_db_row(row)

        try:
            entry_uuid = uuid.UUID(entry)
        except (ValueError, AttributeError):
            return None
        with get_db_session() as session:
            row = (
                session.query(DatasetModel)
                .filter(DatasetModel.id == entry_uuid)
                .first()
            )
            if row:
                return _try_register_from_db_row(row)
        return None

    try:
        result = _db_lookup()
        if result:
            return result
    except Exception as e:
        print(f"[resolve] DB lookup failed for '{entry}': {e}")

    print(f"[resolve] Could not resolve dataset '{entry}' — not in memory, filesystem, or DB with data")
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
            for entry in linked_datasets:
                ref = resolve_linked_dataset(entry)
                if ref:
                    registered_refs.append(ref)

        repository.update_training_status(job_id, {"current_step": "select_model"})
        final_linked_datasets = registered_refs or None

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
            if isinstance(snap, dict) and snap:
                result["state_snapshot"] = snap
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


GRAPH_PIPELINE_STEP_NAMES = frozenset(ALL_STEP_NAMES)


def _coerce_str_set(val: object) -> set[str]:
    if val is None:
        return set()
    if isinstance(val, set):
        return {str(x) for x in val}
    if isinstance(val, (list, tuple)):
        return {str(x) for x in val}
    return set()


def _planner_skipped_info(node_output: dict[str, object]) -> tuple[list[str], str]:
    skipped = node_output.get("skipped_steps")
    if isinstance(skipped, list) and skipped:
        r = node_output.get("skip_rationale", "") or ""
        return [str(s) for s in skipped], str(r)
    for t in reversed(node_output.get("audit_trace") or []):
        if t.get("step") == "planner":
            sk = t.get("skipped")
            if isinstance(sk, list) and sk:
                return [str(s) for s in sk], str(node_output.get("skip_rationale", "") or "")
    return [], ""


def _graph_state_snapshot_values(agent, config: dict) -> dict[str, object]:
    try:
        snap = agent.get_state(config)
        vals = getattr(snap, "values", None)
        if isinstance(vals, dict):
            return dict(vals)
    except Exception:
        pass
    return {}


def _pipeline_emit_key(node_name: str, raw: dict[str, object]) -> str:
    """Unique key per pipeline step *execution* (replan / amend / repeat same step name)."""
    hist = raw.get("plan_history")
    h = len(hist) if isinstance(hist, list) else 0
    pi = raw.get("plan_index")
    try:
        pi_i = int(pi) if pi is not None else 0
    except (TypeError, ValueError):
        pi_i = 0
    return f"{h}:{pi_i}:{node_name}"


def _iter_graph_sse_lines(agent, config: dict, thread_id: str, stream_input: object, experiment_id: str | None = None):
    """Yield `data: ...\\n\\n` lines for graph agent.stream(updates + custom)."""
    _boot = repository.get_simple_agent_store(thread_id)
    if _boot is not None:
        _boot.pop("_graph_sse_interrupted", None)

    for chunk in agent.stream(
        stream_input, config=config, stream_mode=["updates", "custom"]
    ):
        if isinstance(chunk, tuple) and len(chunk) == 2:
            mode, event = chunk
        else:
            mode, event = "updates", chunk

        if mode == "custom" and isinstance(event, dict):
            store = repository.get_simple_agent_store(thread_id)
            if store is None:
                continue

            if event.get("type") == "token":
                content = event.get("content", "")
                phase = event.get("phase")
                if content:
                    yield format_sse(
                        token_event(content, experiment_id, phase=phase),
                        experiment_id,
                    )
                continue

            msg = event.get("message")
            if not msg:
                continue
            yield format_sse(step_progress(
                phase=event.get("phase", "graph"),
                message=str(msg),
            ), experiment_id)
            continue

        store = repository.get_simple_agent_store(thread_id)
        if store is None:
            continue

        emitted_steps = _coerce_str_set(store.get("emitted_steps"))
        emitted_skipped_steps = _coerce_str_set(store.get("emitted_skipped_steps"))

        if "__interrupt__" in event:
            info = _extract_simple_interrupt(event["__interrupt__"], thread_id)
            snap_vals = _graph_state_snapshot_values(agent, config)
            intr_snap = info.get("state_snapshot")
            merged_snap: dict[str, object] = dict(snap_vals) if snap_vals else {}
            if isinstance(intr_snap, dict) and intr_snap:
                merged_snap.update(intr_snap)
            evt = review_required(
                node=info.get("node", "unknown"),
                summary=info.get("summary", ""),
                message=info.get("message", "Approve to continue, or provide feedback to redo."),
                state_snapshot=serialize_state(merged_snap) if merged_snap else {},
                review_prompt=f"Review {info.get('node', 'unknown')} output and approve or provide feedback",
            )
            for k in ("plan", "plan_strategy", "plan_index"):
                if k in info:
                    evt[k] = info[k]
            store["emitted_steps"] = emitted_steps
            store["emitted_skipped_steps"] = emitted_skipped_steps
            store["_graph_sse_interrupted"] = True
            yield format_sse(serialize_state(evt), experiment_id)
            return

        for node_name, node_output in event.items():
            if node_name.startswith("__"):
                continue
            raw: dict[str, object] = node_output if isinstance(node_output, dict) else {}

            if node_name in GRAPH_PIPELINE_STEP_NAMES:
                # Node updates can be partial; merge with checkpoint so audit_trace / refs exist for headlines.
                full_vals = _graph_state_snapshot_values(agent, config)
                if full_vals:
                    merged: dict[str, object] = dict(full_vals)
                    merged.update(raw)
                    raw = merged
                step_key = _pipeline_emit_key(node_name, raw)
                if step_key in emitted_steps:
                    continue
                emitted_steps.add(step_key)
                store["emitted_steps"] = emitted_steps

                if is_unsupervised_passthrough(node_name, raw):
                    emitted_skipped_steps.add(node_name)
                    store["emitted_skipped_steps"] = emitted_skipped_steps
                    yield format_sse(step_skipped(
                        node=node_name,
                        headline="Skipped — unsupervised models use cleaned data directly",
                    ), experiment_id)
                else:
                    update = build_node_update(node_name, raw)
                    update["type"] = "step.complete"
                    update["thread_id"] = thread_id
                    update["stream_step_key"] = step_key
                    yield format_sse(serialize_state(update), experiment_id)
            else:
                update = build_node_update(node_name, raw)
                if is_unsupervised_passthrough(node_name, raw):
                    emitted_skipped_steps.add(node_name)
                    store["emitted_skipped_steps"] = emitted_skipped_steps
                    yield format_sse(step_skipped(
                        node=node_name,
                        headline="Skipped — unsupervised models use cleaned data directly",
                    ), experiment_id)
                else:
                    update["type"] = "step.complete"
                    update["thread_id"] = thread_id
                    yield format_sse(serialize_state(update), experiment_id)
                if node_name == "planner":
                    skipped_list, skip_reason = _planner_skipped_info(raw)
                    for sid in skipped_list:
                        if sid in emitted_skipped_steps:
                            continue
                        emitted_skipped_steps.add(sid)
                        store["emitted_skipped_steps"] = emitted_skipped_steps
                        yield format_sse(step_skipped(
                            node=sid,
                            headline=f"Skipped {sid}: {skip_reason}",
                        ), experiment_id)

        store["emitted_steps"] = emitted_steps
        store["emitted_skipped_steps"] = emitted_skipped_steps


def generate_simple_sse_events(
    goal: str,
    linked_datasets: Optional[list[str]],
    model_pref: Optional[str],
    hitl: bool = True,
    thread_id: Optional[str] = None,
):
    from langgraph.checkpoint.memory import MemorySaver

    from agents.training.agent_simple import create_simple_training_agent

    thread_id = thread_id or f"simple-{uuid.uuid4().hex[:8]}"

    repository.ensure_training_job(
        thread_id,
        {
            "goal": goal,
            "linked_datasets": linked_datasets,
            "model_preference": model_pref,
        },
    )

    registered_refs = []
    if linked_datasets:
        for entry in linked_datasets:
            ref = resolve_linked_dataset(entry)
            if ref:
                registered_refs.append(ref)
                payload = {
                    "type": "dataset_loaded",
                    "dataset": entry,
                    "ref": ref,
                    "thread_id": thread_id,
                }
                yield f"data: {json.dumps(payload)}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'dataset_error', 'dataset': entry, 'error': f'Could not resolve dataset: {entry}', 'thread_id': thread_id})}\n\n"

    final_linked = registered_refs or None
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
        for ns, event in agent.stream(
            {"messages": [{"role": "user", "content": goal}]},
            config=config,
            stream_mode="updates",
            subgraphs=True,
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

    repository.ensure_training_job(thread_id, {})

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
        for ns, event in agent.stream(
            Command(resume=resume_value),
            config=config,
            stream_mode="updates",
            subgraphs=True,
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
    experiment_id: Optional[str] = None,
    conversation: Optional[list[dict]] = None,
):
    """SSE generator using the agentic graph (planner + executor + evaluator)."""
    from langgraph.checkpoint.memory import MemorySaver

    from agents.training.core.conversation_context import \
        normalize_conversation_turns
    from agents.training.core.graph import create_training_agent
    from agents.training.core.state import create_initial_state

    thread_id = thread_id or f"graph-{uuid.uuid4().hex[:8]}"
    goal = (goal or "").strip() or "Training run"

    repository.ensure_training_job(
        thread_id,
        {
            "goal": goal,
            "linked_datasets": linked_datasets,
            "model_preference": model_pref,
        },
    )

    if experiment_id:
        exp = repository.get_experiment(experiment_id)
        if exp:
            ts0 = dict(exp.get("training_state") or {})
            if ts0.get("task_status") == "running" or ts0.get("graph_run_status") == "running":
                yield format_sse(
                    error_event(
                        "Another training run is already in progress for this experiment "
                        "(async task or interactive graph)."
                    ),
                    experiment_id,
                )
                yield format_sse(stream_end(experiment_id, pipeline_completed=False), experiment_id)
                return
            ts = dict(ts0)
            ts["graph_thread_id"] = thread_id
            ts["graph_run_status"] = "running"
            ts["graph_run_started_at"] = datetime.now(timezone.utc).isoformat()
            repository.update_experiment(
                experiment_id,
                {
                    "goal": goal,
                    "status": "running",
                    "training_state": ts,
                },
            )

    registered_refs = []
    failed_datasets = []
    if linked_datasets:
        for entry in linked_datasets:
            ref = resolve_linked_dataset(entry)
            if ref:
                registered_refs.append(ref)
                yield format_sse(dataset_resolved(ref=ref, dataset_info={"dataset": entry}), experiment_id)
            else:
                failed_datasets.append(entry)
                yield format_sse(dataset_error(ref=entry, error=f"Could not load dataset '{entry}'. It may be metadata-only with no backing data file."), experiment_id)

    if failed_datasets and not registered_refs:
        yield format_sse(error_event(
            f"Cannot start training: none of the selected datasets could be loaded ({', '.join(failed_datasets)}). "
            "Please select a dataset that has backing data (file or R2 storage)."
        ), experiment_id)
        yield format_sse(stream_end(experiment_id, pipeline_completed=False), experiment_id)
        return

    final_linked = registered_refs or None
    resolved_ds = registered_refs[0] if len(registered_refs) == 1 else None
    checkpointer = MemorySaver()
    agent = create_training_agent(checkpointer=checkpointer)
    conversation_turns = normalize_conversation_turns(
        conversation,
        triggering_message=goal,
    )
    initial_state = create_initial_state(
        goal=goal,
        linked_datasets=final_linked,
        user_model_preference=model_pref,
        resolved_dataset_ref=resolved_ds,
        resolved_model_type=model_pref,
        conversation_history=conversation_turns,
    )

    emitted_steps: set[str] = set()
    emitted_skipped_steps: set[str] = set()
    repository.put_simple_agent_store(
        thread_id,
        {
            "agent": agent,
            "checkpointer": checkpointer,
            "mode": "graph",
            "emitted_steps": emitted_steps,
            "emitted_skipped_steps": emitted_skipped_steps,
            "experiment_id": experiment_id,
        },
    )

    config = {"configurable": {"thread_id": thread_id}}
    yield format_sse(stream_start(experiment_id, training_graph=True), experiment_id)

    try:
        step_events: list[dict[str, object]] = []
        for line in _iter_graph_sse_lines(agent, config, thread_id, initial_state, experiment_id):
            if experiment_id:
                evt = _parse_sse_data_line(line)
                if evt:
                    et = evt.get("type")
                    node = evt.get("node")
                    if isinstance(node, str) and node in ALL_STEP_NAMES:
                        if et == "step.complete":
                            step_events.append(
                                {
                                    "node": node,
                                    "type": "complete",
                                    "at": datetime.now(timezone.utc).isoformat(),
                                }
                            )
                        elif et == "step.skipped":
                            step_events.append(
                                {
                                    "node": node,
                                    "type": "skipped",
                                    "at": datetime.now(timezone.utc).isoformat(),
                                }
                            )
            yield line

        st = repository.get_simple_agent_store(thread_id)
        if st and st.get("_graph_sse_interrupted"):
            st.pop("_graph_sse_interrupted", None)
            return

        final_values = _graph_state_snapshot_values(agent, config)
        if final_values:
            try:
                repository.save_training_context(final_values, experiment_id=experiment_id)
                repository.link_model_to_experiment(
                    repository.extract_model_name_from_training_state(final_values),
                    experiment_id,
                )
                repository.save_run_dataset_links(thread_id, final_values)
            except Exception:
                traceback.print_exc()

        pipeline_error = final_values.get("error") if final_values else None
        if experiment_id:
            serialized = serialize_state(final_values) if final_values else {}
            persisted_state = {
                **serialized,
                "graph_thread_id": thread_id,
                "task_step_events": step_events,
                "task_current_node": step_events[-1]["node"] if step_events else None,
                "task_status": None,
                "task_error": str(pipeline_error) if pipeline_error else None,
                "task_started_at": None,
                "task_completed_at": None,
                "graph_run_status": "failed" if pipeline_error else "completed",
            }
            repository.merge_experiment_training_state(experiment_id, persisted_state)
            repository.update_experiment(
                experiment_id,
                {"goal": goal, "status": "failed" if pipeline_error else "completed"},
            )
        if pipeline_error:
            yield format_sse(error_event(pipeline_error), experiment_id)
        yield format_sse(stream_end(experiment_id, pipeline_completed=not pipeline_error), experiment_id)
    except Exception as e:
        traceback.print_exc()
        if experiment_id:
            repository.merge_experiment_training_state(
                experiment_id,
                {
                    "graph_thread_id": thread_id,
                    "error": str(e),
                    "task_error": str(e),
                    "graph_run_status": "failed",
                },
            )
            repository.update_experiment(experiment_id, {"goal": goal, "status": "failed"})
        yield format_sse(error_event(str(e)), experiment_id)


def generate_graph_resume_sse_events(
    thread_id: str,
    approved: bool,
    feedback: Optional[str],
    experiment_id: Optional[str] = None,
):
    """Resume the agentic graph after a HITL interrupt."""
    from langgraph.types import Command

    store = repository.get_simple_agent_store(thread_id)
    if not store:
        yield format_sse(error_event("Graph thread not found"), experiment_id)
        return

    repository.ensure_training_job(thread_id, {})

    if experiment_id is None:
        experiment_id = store.get("experiment_id")

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

    if store.get("emitted_skipped_steps") is None:
        store["emitted_skipped_steps"] = set()

    try:
        step_events: list[dict[str, object]] = []
        goal = "Training run"
        if experiment_id:
            exp = repository.get_experiment(experiment_id)
            if exp:
                goal = str(exp.get("goal") or goal)
                existing_events = exp.get("training_state", {}).get("task_step_events")
                if isinstance(existing_events, list):
                    step_events = [e for e in existing_events if isinstance(e, dict)]
        for line in _iter_graph_sse_lines(agent, config, thread_id, Command(resume=resume_value), experiment_id):
            if experiment_id:
                evt = _parse_sse_data_line(line)
                if evt:
                    et = evt.get("type")
                    node = evt.get("node")
                    if isinstance(node, str) and node in ALL_STEP_NAMES:
                        if et == "step.complete":
                            step_events.append(
                                {
                                    "node": node,
                                    "type": "complete",
                                    "at": datetime.now(timezone.utc).isoformat(),
                                }
                            )
                        elif et == "step.skipped":
                            step_events.append(
                                {
                                    "node": node,
                                    "type": "skipped",
                                    "at": datetime.now(timezone.utc).isoformat(),
                                }
                            )
            yield line

        st = repository.get_simple_agent_store(thread_id)
        if st and st.get("_graph_sse_interrupted"):
            st.pop("_graph_sse_interrupted", None)
            return

        final_values = _graph_state_snapshot_values(agent, config)
        if final_values:
            try:
                repository.save_training_context(final_values, experiment_id=experiment_id)
                repository.link_model_to_experiment(
                    repository.extract_model_name_from_training_state(final_values),
                    experiment_id,
                )
                repository.save_run_dataset_links(thread_id, final_values)
            except Exception:
                traceback.print_exc()

        pipeline_error = final_values.get("error") if final_values else None
        if experiment_id:
            goal = str((final_values or {}).get("goal") or goal)
            serialized = serialize_state(final_values) if final_values else {}
            persisted_state = {
                **serialized,
                "graph_thread_id": thread_id,
                "task_step_events": step_events,
                "task_current_node": step_events[-1]["node"] if step_events else None,
                "task_status": None,
                "task_error": str(pipeline_error) if pipeline_error else None,
                "task_started_at": None,
                "task_completed_at": None,
                "graph_run_status": "failed" if pipeline_error else "completed",
            }
            repository.merge_experiment_training_state(experiment_id, persisted_state)
            repository.update_experiment(
                experiment_id,
                {"goal": goal, "status": "failed" if pipeline_error else "completed"},
            )
        if pipeline_error:
            yield format_sse(error_event(pipeline_error), experiment_id)
        yield format_sse(stream_end(experiment_id, pipeline_completed=not pipeline_error), experiment_id)
    except Exception as e:
        traceback.print_exc()
        if experiment_id:
            repository.merge_experiment_training_state(
                experiment_id,
                {
                    "graph_thread_id": thread_id,
                    "error": str(e),
                    "task_error": str(e),
                    "graph_run_status": "failed",
                },
            )
            repository.update_experiment(experiment_id, {"goal": goal, "status": "failed"})
        yield format_sse(error_event(str(e)), experiment_id)


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
        "feature_specification_and_engineering": 62,
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
        for entry in request.linked_datasets:
            ref = resolve_linked_dataset(entry)
            if ref:
                registered_refs.append(ref)

    result = invoke_simple_training_agent(
        goal=request.goal,
        linked_datasets=registered_refs or None,
        user_model_preference=request.user_model_preference,
    )
    return {"status": "completed", "state": serialize_state(result)}


def cancel_training(job_id: str) -> dict[str, str]:
    deleted = repository.cancel_training(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "deleted", "job_id": job_id}


def _parse_sse_data_line(line: str) -> Optional[dict[str, object]]:
    raw = line.strip()
    if not raw.startswith("data: "):
        return None
    payload = raw[6:].strip()
    if payload in ("", "[DONE]"):
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _run_experiment_graph_task_worker(
    experiment_id: str,
    goal: str,
    linked_datasets: Optional[list[str]],
    model_pref: Optional[str],
    conversation: Optional[list[dict]],
) -> None:
    """Run the planner graph with HITL auto-approved; persist progress on the experiment row."""
    from langgraph.checkpoint.memory import MemorySaver

    from agents.training.core.conversation_context import \
        normalize_conversation_turns
    from agents.training.core.graph import create_training_agent
    from agents.training.core.state import create_initial_state

    thread_id = f"graph-{uuid.uuid4().hex[:8]}"
    repository.merge_experiment_training_state(
        experiment_id,
        {
            "graph_thread_id": thread_id,
            "task_status": "running",
            "task_step_events": [],
            "task_error": None,
            "lab_mode": "task",
            "task_started_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    repository.update_experiment(experiment_id, {"status": "running"})

    repository.ensure_training_job(
        thread_id,
        {
            "goal": goal,
            "linked_datasets": linked_datasets,
            "model_preference": model_pref,
        },
    )

    try:
        registered_refs: list[str] = []
        failed_datasets: list[str] = []
        if linked_datasets:
            for entry in linked_datasets:
                ref = resolve_linked_dataset(str(entry))
                if ref:
                    registered_refs.append(ref)
                else:
                    failed_datasets.append(str(entry))

        if failed_datasets and not registered_refs:
            raise RuntimeError(
                "Cannot start training: none of the selected datasets could be loaded ("
                + ", ".join(failed_datasets)
                + ")"
            )

        final_linked = registered_refs or None
        resolved_ds = registered_refs[0] if len(registered_refs) == 1 else None
        checkpointer = MemorySaver()
        agent = create_training_agent(checkpointer=checkpointer)
        g = (goal or "").strip() or "Training run"
        conversation_turns = normalize_conversation_turns(
            conversation,
            triggering_message=g,
        )
        initial_state = create_initial_state(
            goal=g,
            linked_datasets=final_linked,
            user_model_preference=model_pref,
            resolved_dataset_ref=resolved_ds,
            resolved_model_type=model_pref,
            conversation_history=conversation_turns,
            hitl_auto_approve=True,
        )

        emitted_steps: set[str] = set()
        emitted_skipped_steps: set[str] = set()
        repository.put_simple_agent_store(
            thread_id,
            {
                "agent": agent,
                "checkpointer": checkpointer,
                "mode": "graph",
                "emitted_steps": emitted_steps,
                "emitted_skipped_steps": emitted_skipped_steps,
                "experiment_id": experiment_id,
            },
        )
        config = {"configurable": {"thread_id": thread_id}}

        step_events: list[dict[str, object]] = []
        for line in _iter_graph_sse_lines(
            agent, config, thread_id, initial_state, experiment_id
        ):
            evt = _parse_sse_data_line(line)
            if not evt:
                continue
            et = evt.get("type")
            if et == "step.complete":
                node = evt.get("node")
                if node:
                    step_events.append(
                        {
                            "node": node,
                            "type": "complete",
                            "at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    repository.merge_experiment_training_state(
                        experiment_id,
                        {
                            "task_step_events": list(step_events),
                            "task_current_node": node,
                        },
                    )
            elif et == "error":
                raise RuntimeError(str(evt.get("error", "Unknown error")))

        st = repository.get_simple_agent_store(thread_id)
        if st and st.get("_graph_sse_interrupted"):
            raise RuntimeError(
                "Pipeline paused for review; async mode requires uninterrupted completion"
            )

        final_values = _graph_state_snapshot_values(agent, config)
        if final_values:
            try:
                repository.save_training_context(final_values, experiment_id=experiment_id)
                repository.link_model_to_experiment(
                    repository.extract_model_name_from_training_state(final_values),
                    experiment_id,
                )
                repository.save_run_dataset_links(thread_id, final_values)
            except Exception:
                traceback.print_exc()

        serialized = serialize_state(final_values) if final_values else {}
        pipeline_error = final_values.get("error") if final_values else None
        ts_complete = {
            **serialized,
            "task_status": "failed" if pipeline_error else "completed",
            "graph_thread_id": thread_id,
            "lab_mode": "task",
            "hitl_auto_approve": False,
            "task_step_events": step_events,
        }
        if pipeline_error:
            ts_complete["task_error"] = str(pipeline_error)
        else:
            ts_complete["task_completed_at"] = datetime.now(timezone.utc).isoformat()

        repository.merge_experiment_training_state(experiment_id, ts_complete)
        repository.update_experiment(
            experiment_id,
            {"status": "failed" if pipeline_error else "completed"},
        )
    except Exception as e:
        traceback.print_exc()
        repository.merge_experiment_training_state(
            experiment_id,
            {
                "task_status": "failed",
                "task_error": str(e),
            },
        )
        repository.update_experiment(experiment_id, {"status": "failed"})


def start_experiment_async_training(
    experiment_id: str,
    model_pref: Optional[str] = None,
    conversation: Optional[list[dict]] = None,
) -> None:
    exp = repository.get_experiment(experiment_id)
    if exp is None:
        raise ValueError("Experiment not found")
    ts = exp.get("training_state") or {}
    if ts.get("task_status") == "running":
        raise RuntimeError("A training task is already running for this experiment")
    if ts.get("graph_run_status") == "running":
        raise RuntimeError(
            "An interactive graph run is already in progress for this experiment"
        )

    goal = (exp.get("goal") or "").strip() or "Training run"
    raw_ld = exp.get("linked_datasets")
    linked: list[str] = []
    if isinstance(raw_ld, list):
        linked = [str(x) for x in raw_ld]

    thread = threading.Thread(
        target=_run_experiment_graph_task_worker,
        args=(experiment_id, goal, linked, model_pref, conversation),
        daemon=True,
    )
    thread.start()
