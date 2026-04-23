import json
import uuid
from typing import Any, Optional

from backend.chat import repository
from backend.chat.events import (
    format_sse,
    stream_start,
    stream_end,
    token as token_event,
    tool_start,
    tool_end,
    error_event,
    predict_start,
    predict_complete,
    task_plan_proposed as task_plan_proposed_event,
)
from backend.chat.intent_router import should_route_to_training_graph
from backend.chat.schemas import ChatRequest
from backend.shared.serialization import serialize_state
from backend.training import repository as training_repo

_PREDICT_TOOLS = frozenset({"predict_with_model"})


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
    if ctx.get("report_storage_key"):
        lines.append(f"  Report key  : {ctx['report_storage_key']}")
    lines.extend([
        "  This is background context only.",
        "  Do not assume the user's new message is fully answered by this prior run.",
        "  If the new message is short or ambiguous (for example: 'train', 'again', 'run it'),",
        "  ask a clarifying question about what they want to train, change, or do next.",
    ])
    lines.append("[END CONTEXT — use this as background while responding to the user's latest message]")
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


def _orchestrator_user_content(
    message: str,
    conversation: Optional[list[dict[str, Any]]],
    context_block: str,
) -> str:
    """Single user blob so the model sees prior turns (fast-path search skips agent memory)."""
    if not conversation or len(conversation) <= 1:
        if context_block:
            return f"{context_block}\n\nUser message: {message}"
        return message
    parts: list[str] = []
    for t in conversation:
        role = (t.get("role") or "").strip()
        c = (t.get("content") or "").strip()
        if not c:
            continue
        label = "User" if role == "user" else "Assistant"
        parts.append(f"{label}: {c}")
    transcript = "\n\n".join(parts)
    hint = (
        "\n\nReply to the last user message. If it is only a number, map it to the matching "
        "dataset from the assistant's most recent numbered list; do not ask for the dataset again "
        "if the user already chose."
    )
    if context_block:
        return f"{context_block}\n\nChat so far:\n\n{transcript}{hint}"
    return f"Chat so far:\n\n{transcript}{hint}"


def _persist_chat_exchange(
    experiment_id: Optional[str],
    user_message: str,
    agent_text: str,
) -> None:
    if not experiment_id:
        return
    try:
        import time as _t

        now = int(_t.time() * 1000)
        existing = []
        exp_data = training_repo.get_experiment(experiment_id)
        if exp_data and exp_data.get("chat_history"):
            existing = list(exp_data["chat_history"])
        existing.append({"role": "user", "content": user_message, "timestamp": now - 1})
        if agent_text.strip():
            existing.append({"role": "agent", "content": agent_text, "timestamp": now})
        training_repo.save_experiment_chat_history(experiment_id, existing)
    except Exception:
        pass


def generate_direct_dataset_search_sse(
    message: str,
    experiment_id: Optional[str] = None,
):
    orchestrator_mod = repository.get_orchestrator_module()
    tool_args = orchestrator_mod.build_dataset_search_args(message)
    yield format_sse(stream_start(experiment_id), experiment_id)
    yield format_sse(
        tool_start(
            "search_datasets",
            tool_args,
            headline="Running search_datasets...",
            experiment_id=experiment_id,
        ),
        experiment_id,
    )
    try:
        result = orchestrator_mod.search_datasets.invoke(tool_args)
        yield format_sse(tool_end("search_datasets", result, experiment_id), experiment_id)
        visible_reply = result
        if result and not result.startswith("No datasets found"):
            visible_reply += "\n\nReply with the dataset name or its number to continue (or **go** if that single match is what you want)."
        yield format_sse(token_event(visible_reply, experiment_id), experiment_id)
        _persist_chat_exchange(experiment_id, message, visible_reply)
        yield format_sse(stream_end(experiment_id), experiment_id)
    except Exception as exc:
        yield format_sse(error_event(str(exc), experiment_id), experiment_id)


def generate_chat_sse(
    thread_id: str,
    message: str,
    training_context: Optional[str] = None,
    experiment_id: Optional[str] = None,
    conversation: Optional[list[dict[str, Any]]] = None,
):
    orchestrator_agent = repository.get_orchestrator_agent()
    # Snapshot thread when sending full transcript so checkpoint state does not hide prior turns
    # (e.g. fast-path dataset search never wrote to the agent graph).
    _conv_len = len(conversation) if conversation else 0
    effective_thread = (
        f"{thread_id}-orch-{_conv_len}" if _conv_len > 1 else thread_id
    )
    config = {"configurable": {"thread_id": effective_thread}}

    context_block = training_context or ""
    if not context_block:
        latest_ctx = training_repo.get_latest_training_context(
            experiment_id=experiment_id,
        )
        if latest_ctx:
            context_block = _build_training_context_message(latest_ctx)

    augmented_message = _orchestrator_user_content(message, conversation, context_block)

    agent_input = {"messages": [{"role": "user", "content": augmented_message}]}
    yield format_sse(stream_start(experiment_id), experiment_id)

    seen_tool_call_ids: set[str] = set()
    pending_predict_args: dict[str, dict] = {}
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
                    yield format_sse(token_event(content, experiment_id), experiment_id)

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
                    safe_args = serialize_state(args if isinstance(args, dict) else {})
                    if name in _PREDICT_TOOLS:
                        pending_predict_args[name] = safe_args
                        yield format_sse(predict_start(
                            model=safe_args.get("model_name", "unknown"),
                            dataset=safe_args.get("dataset_ref", "unknown"),
                            headline="Running predictions...",
                            experiment_id=experiment_id,
                        ), experiment_id)
                    else:
                        yield format_sse(tool_start(
                            name,
                            safe_args,
                            headline=f"Running {name}...",
                            experiment_id=experiment_id,
                        ), experiment_id)

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
                                args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                                safe_args = serialize_state(args if isinstance(args, dict) else {})
                                if tc_name in _PREDICT_TOOLS:
                                    pending_predict_args[tc_name] = safe_args
                                tc_id = _stable_tool_call_id(tc)
                                if tc_id in seen_tool_call_ids:
                                    continue
                                seen_tool_call_ids.add(tc_id)
                                if tc_name in _PREDICT_TOOLS:
                                    yield format_sse(predict_start(
                                        model=safe_args.get("model_name", "unknown"),
                                        dataset=safe_args.get("dataset_ref", "unknown"),
                                        headline="Running predictions...",
                                        experiment_id=experiment_id,
                                    ), experiment_id)
                                else:
                                    yield format_sse(tool_start(
                                        tc_name,
                                        safe_args,
                                        headline=f"Running {tc_name}...",
                                        experiment_id=experiment_id,
                                    ), experiment_id)

                if "tools" in payload:
                    tool_msgs = payload["tools"].get("messages", [])
                    for tm in tool_msgs:
                        name = getattr(tm, "name", "unknown")
                        snippet = getattr(tm, "content", "")
                        if len(snippet) > 2000:
                            snippet = snippet[:2000] + "…"
                        if name in _PREDICT_TOOLS:
                            stored = pending_predict_args.pop(name, {})
                            try:
                                result_data = json.loads(snippet) if snippet.strip().startswith("{") else {}
                            except (json.JSONDecodeError, TypeError):
                                result_data = {}
                            rows = result_data.get("rows_predicted") or snippet.count("\nRow ")
                            yield format_sse(predict_complete(
                                model=stored.get("model_name", "unknown"),
                                rows_predicted=rows,
                                headline=f"Predicted {rows} rows",
                                result_ref=result_data.get("result_ref"),
                                experiment_id=experiment_id,
                            ), experiment_id)
                        else:
                            yield format_sse(tool_end(name, snippet, experiment_id), experiment_id)

        _persist_chat_exchange(experiment_id, message, "".join(accumulated_response))

        yield format_sse(stream_end(experiment_id), experiment_id)
    except Exception as exc:
        yield format_sse(error_event(str(exc), experiment_id), experiment_id)


def generate_background_intake_sse(
    message: str,
    experiment_id: Optional[str] = None,
    conversation: Optional[list[dict[str, Any]]] = None,
):
    """
    Structured LLM output (``IntakeResponse``): stream ``message`` only, then ``task_plan.proposed``
    when ``plan`` is set — no JSON in the model's visible text.
    """
    from agents.background_intake_agent import run_intake_turn

    yield format_sse(stream_start(experiment_id), experiment_id)

    try:
        response = run_intake_turn(conversation, message)
        text = (response.message or "").strip()
        chunk_size = 200
        for i in range(0, len(text), chunk_size):
            yield format_sse(token_event(text[i : i + chunk_size], experiment_id), experiment_id)

        plan = response.plan
        if plan is not None and plan.goal.strip():
            refs = [str(r).strip() for r in plan.dataset_refs if str(r).strip()]
            raw_labels = [str(x).strip() for x in plan.dataset_labels if str(x).strip()]
            label_final = [
                raw_labels[i] if i < len(raw_labels) else refs[i] for i in range(len(refs))
            ]
            steps = [str(s).strip() for s in plan.recap_steps if str(s).strip()]
            pref_str = plan.preferences.strip() if plan.preferences else None
            yield format_sse(
                task_plan_proposed_event(
                    goal=plan.goal.strip(),
                    dataset_refs=refs,
                    dataset_labels=label_final,
                    preferences=pref_str,
                    recap_steps=steps if steps else None,
                    experiment_id=experiment_id,
                ),
                experiment_id,
            )

        if experiment_id and text:
            try:
                import time as _t

                now = int(_t.time() * 1000)
                existing = []
                exp_data = training_repo.get_experiment(experiment_id)
                if exp_data and exp_data.get("chat_history"):
                    existing = list(exp_data["chat_history"])
                existing.append({"role": "user", "content": message, "timestamp": now - 1})
                existing.append({"role": "agent", "content": text, "timestamp": now})
                training_repo.save_experiment_chat_history(experiment_id, existing)
            except Exception:
                pass

        yield format_sse(stream_end(experiment_id), experiment_id)
    except Exception as exc:
        yield format_sse(error_event(str(exc), experiment_id), experiment_id)


def chat(request: ChatRequest) -> tuple[str, object]:
    """Route unified /api/chat body to graph training or orchestrator SSE."""
    from backend.training import service as training_service

    if request.resume:
        if not request.experiment_id:
            raise ValueError("experiment_id required for resume")
        exp = training_repo.get_experiment(request.experiment_id)
        if not exp or not exp.get("training_state", {}).get("graph_thread_id"):
            raise ValueError("No active training graph found for this experiment")
        graph_thread = exp["training_state"]["graph_thread_id"]
        gen = training_service.generate_graph_resume_sse_events(
            graph_thread, request.resume.approved, request.resume.feedback,
        )
        return graph_thread, gen

<<<<<<< Updated upstream
    train_branch = should_route_to_training_graph(request)
    if train_branch:
        goal = (request.message or "").strip()
        if not goal:
            goal = "Training run"
=======
    # ---- Unified Jubilee stream (chat + optional training pivot) ------------

    if request.chat_thread_id and str(request.chat_thread_id).strip():
        resolved_thread_id = str(request.chat_thread_id).strip()
    elif request.experiment_id:
        exp = training_repo.get_experiment(request.experiment_id, org_id=org_id)
        resolved_thread_id = exp["chat_thread_id"] if exp else f"chat-{uuid.uuid4().hex[:8]}"
    else:
        resolved_thread_id = f"chat-{uuid.uuid4().hex[:8]}"

    attached_block, attach_events = _prepare_attached_datasets_block(
        request.linked_datasets,
    )

    if request.background_intake:
        bg_msg = request.message or ""
        if attached_block:
            bg_msg = attached_block + "\n\n" + bg_msg
        return resolved_thread_id, generate_background_intake_sse(
            bg_msg,
            experiment_id=request.experiment_id,
            conversation=request.conversation,
            org_id=org_id,
        )

    # ---- Plan-approval fast path: bypass Jubilee, go straight to training ---
    # When the UI confirms the plan card it sends ``mode="train"`` plus the
    # already-resolved dataset refs. Routing through the orchestrator here is
    # both unnecessary and unreliable (the LLM sometimes calls
    # ``propose_training_plan`` again, which re-renders the plan card and the
    # user appears stuck in a loop). Hand off to the training sub-agent
    # immediately when we have everything we need.
    if request.mode == "train" and request.linked_datasets:
        refs = [str(r).strip() for r in request.linked_datasets if str(r).strip()]
        goal = (request.message or "").strip() or "Training run"
>>>>>>> Stashed changes
        gen = training_service.generate_graph_sse_events(
            goal,
            request.linked_datasets,
            request.model_preference,
            experiment_id=request.experiment_id,
            conversation=request.conversation,
        )
        return "", gen

    if request.experiment_id:
        exp = training_repo.get_experiment(request.experiment_id)
        resolved_thread_id = exp["chat_thread_id"] if exp else f"chat-{uuid.uuid4().hex[:8]}"
    else:
        resolved_thread_id = f"chat-{uuid.uuid4().hex[:8]}"

    if getattr(request, "background_intake", False):
        return resolved_thread_id, generate_background_intake_sse(
            request.message,
            experiment_id=request.experiment_id,
            conversation=request.conversation,
        )

    orchestrator_mod = repository.get_orchestrator_module()
    if orchestrator_mod.is_dataset_search_request(request.message):
        return resolved_thread_id, generate_direct_dataset_search_sse(
            request.message,
            experiment_id=request.experiment_id,
        )

    training_context = None
    ctx = training_repo.get_latest_training_context(experiment_id=request.experiment_id)
    if ctx:
        training_context = _build_training_context_message(ctx)

    return resolved_thread_id, generate_chat_sse(
        resolved_thread_id,
        request.message,
        training_context,
        experiment_id=request.experiment_id,
        conversation=request.conversation,
    )
