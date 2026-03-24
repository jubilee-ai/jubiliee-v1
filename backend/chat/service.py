import json
import threading
import uuid
from typing import Optional

from agents.training.utils.streaming import build_node_update
from backend.chat import repository
from backend.chat.intent_router import should_route_to_training_graph
from backend.chat.schemas import ChatRequest
from backend.shared.serialization import serialize_state
from backend.training import repository as training_repo


def _build_training_context_message(ctx: dict) -> str:
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
            metric_lines.append(
                f"  {label:16s}: {v:.4f}" if isinstance(v, float) else f"  {label:16s}: {v}"
            )
    if metric_lines:
        lines.append("  Metrics:")
        lines.extend(metric_lines)
    if ctx.get("num_iterations"):
        lines.append(f"  Iterations  : {ctx['num_iterations']}")
    if ctx.get("report_path"):
        lines.append(f"  Report      : {ctx['report_path']}")
    lines.append("[END CONTEXT — answer the user's question using this information]")
    return "\n".join(lines)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _stable_tool_call_id(tc) -> str:
    """Stable id for deduping tool_call SSE across messages vs updates streams."""
    if isinstance(tc, dict):
        tid = tc.get("id")
        name = tc.get("name") or ""
        raw_args = tc.get("args", {})
    else:
        tid = getattr(tc, "id", None)
        name = getattr(tc, "name", None) or ""
        raw_args = getattr(tc, "args", None)
        if raw_args is None:
            raw_args = {}
    if tid:
        return str(tid)
    args_dict = raw_args if isinstance(raw_args, dict) else serialize_state(raw_args)
    try:
        args_key = json.dumps(serialize_state(args_dict), sort_keys=True)
    except TypeError:
        args_key = str(args_dict)
    return f"{name}:{hash(args_key)}"


def _emit_training_step_events(thread_id: str, experiment_id: Optional[str] = None):
    """Retroactive step events (fallback when progress queue was not used)."""
    from agents.training.core.state import STEP_ORDER

    mod = repository.get_orchestrator_module()
    state = None
    if experiment_id:
        states = getattr(mod, "_training_states", {})
        state = states.get(experiment_id)
    if not state:
        state = getattr(mod, "_last_training_state", None)
    if not state:
        return

    yield _sse({"type": "training_started", "thread_id": thread_id})
    for step_name in STEP_ORDER:
        try:
            update = build_node_update(step_name, state)
            update["thread_id"] = thread_id
            yield _sse(serialize_state(update))
        except Exception:
            pass
    yield _sse({"type": "training_completed", "thread_id": thread_id})


def generate_chat_sse(
    thread_id: str,
    message: str,
    training_context: Optional[str] = None,
    experiment_id: Optional[str] = None,
):
    orchestrator_agent = repository.get_orchestrator_agent()
    config = {"configurable": {"thread_id": thread_id}}

    # Set experiment context on agent module so train_model tool can find it
    if experiment_id:
        mod = repository.get_orchestrator_module()
        thread_exp_map = getattr(mod, "_thread_to_experiment", {})
        thread_exp_map[threading.current_thread().name] = experiment_id

    context_block = training_context or ""
    if not context_block:
        latest_ctx = training_repo.get_latest_training_context(
            experiment_id=experiment_id,
        )
        if latest_ctx:
            context_block = _build_training_context_message(latest_ctx)

    if context_block:
        augmented_message = f"{context_block}\n\nUser message: {message}"
    else:
        augmented_message = message

    agent_input = {"messages": [{"role": "user", "content": augmented_message}]}
    yield _sse({"type": "start", "thread_id": thread_id})

    pending_train_model = False
    seen_tool_call_ids: set[str] = set()
    accumulated_response = []

    try:
        for event in orchestrator_agent.stream(
            agent_input,
            config=config,
            stream_mode=["messages", "updates"],
        ):
            # Dual stream mode yields tuples: (stream_type, payload)
            if isinstance(event, tuple) and len(event) == 2:
                stream_type, payload = event
            else:
                stream_type, payload = "updates", event

            # --- Token-level streaming from "messages" mode ---
            if stream_type == "messages":
                msg_chunk, _metadata = payload if isinstance(payload, tuple) else (payload, {})
                content = getattr(msg_chunk, "content", "")
                if content:
                    accumulated_response.append(content)
                    yield _sse({
                        "type": "token",
                        "content": content,
                        "thread_id": thread_id,
                    })

                tc_chunks = getattr(msg_chunk, "tool_call_chunks", [])
                for tc in tc_chunks:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    if not name:
                        continue
                    tc_id = _stable_tool_call_id(tc)
                    if tc_id in seen_tool_call_ids:
                        continue
                    seen_tool_call_ids.add(tc_id)
                    args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                    yield _sse({
                        "type": "tool_call",
                        "tool": name,
                        "args": serialize_state(args if isinstance(args, dict) else {}),
                        "thread_id": thread_id,
                    })
                    if name == "train_model" and experiment_id:
                        pending_train_model = True

            # --- Node-level updates from "updates" mode ---
            elif stream_type == "updates" and isinstance(payload, dict):
                if "model" in payload:
                    msgs = payload["model"].get("messages", [])
                    for msg in msgs:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tc_name = tc["name"] if isinstance(tc, dict) else getattr(tc, "name", None)
                                if not tc_name:
                                    continue
                                tc_id = _stable_tool_call_id(tc)
                                if tc_id in seen_tool_call_ids:
                                    continue
                                seen_tool_call_ids.add(tc_id)
                                args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                                yield _sse({
                                    "type": "tool_call",
                                    "tool": tc_name,
                                    "args": serialize_state(args if isinstance(args, dict) else {}),
                                    "thread_id": thread_id,
                                })
                                if tc_name == "train_model" and experiment_id:
                                    pending_train_model = True

                if "tools" in payload:
                    tool_msgs = payload["tools"].get("messages", [])
                    for tm in tool_msgs:
                        name = getattr(tm, "name", "unknown")
                        snippet = getattr(tm, "content", "")
                        if len(snippet) > 2000:
                            snippet = snippet[:2000] + "…"
                        yield _sse({
                            "type": "tool_result",
                            "tool": name,
                            "result": snippet,
                            "thread_id": thread_id,
                        })
                        if name == "train_model":
                            yield from _emit_training_step_events(thread_id, experiment_id)
                            pending_train_model = False

        # Persist the exchange to the experiment's chat_history
        if experiment_id and accumulated_response:
            try:
                import time as _t
                now = int(_t.time() * 1000)
                agent_text = "".join(accumulated_response)
                existing = []
                exp_data = training_repo.get_experiment(experiment_id)
                if exp_data and exp_data.get("chat_history"):
                    existing = list(exp_data["chat_history"])
                existing.append({"role": "user", "content": message, "timestamp": now - 1})
                if agent_text.strip():
                    existing.append({"role": "agent", "content": agent_text, "timestamp": now})
                training_repo.save_experiment_chat_history(experiment_id, existing)
            except Exception:
                pass

        yield _sse({"type": "end", "thread_id": thread_id})
    except Exception as exc:
        yield _sse({"type": "error", "error": str(exc), "thread_id": thread_id})


def chat(request: ChatRequest) -> tuple[str, object]:
    """Route unified /api/chat body to graph training or orchestrator SSE."""
    from backend.training import service as training_service

    if request.resume_training:
        rt = request.resume_training
        gen = training_service.generate_graph_resume_sse_events(
            rt.thread_id,
            rt.approved,
            rt.feedback,
        )
        return rt.thread_id, gen

    train_branch = should_route_to_training_graph(request)
    if train_branch:
        goal = (request.message or "").strip()
        if not goal:
            goal = "Training run"
        gen = training_service.generate_graph_sse_events(
            goal,
            request.linked_datasets,
            request.user_model_preference,
            experiment_id=request.experiment_id,
        )
        return "", gen

    if request.experiment_id:
        exp = training_repo.get_experiment(request.experiment_id)
        if exp:
            resolved_thread_id = exp["chat_thread_id"]
        else:
            resolved_thread_id = request.thread_id or f"chat-{uuid.uuid4().hex[:8]}"
    else:
        resolved_thread_id = request.thread_id or f"chat-{uuid.uuid4().hex[:8]}"

    return resolved_thread_id, generate_chat_sse(
        resolved_thread_id,
        request.message,
        request.training_context,
        experiment_id=request.experiment_id,
    )
