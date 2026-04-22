import json
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from orchestrator_context import (
    reset_attached_datasets_active,
    set_attached_datasets_active,
)

from backend.chat import repository
from backend.chat.events import (
    analysis_chart as analysis_chart_event,
    analysis_result as analysis_result_event,
    dataset_resolved,
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
from backend.chat.schemas import ChatRequest
from backend.shared.serialization import serialize_state
from backend.training import repository as training_repo

_ROOT = Path(__file__).resolve().parents[2]
_DT_TOOLS = _ROOT / "tools" / "data-tools"
if str(_DT_TOOLS) not in sys.path:
    sys.path.insert(0, str(_DT_TOOLS))
from analysis.analysis_sidecar import extract_analysis_json_sidecars  # noqa: E402

_PREDICT_TOOLS = frozenset({"predict_with_model"})

#: Name of the Jubilee tool that hands off to the training sub-agent.
#: The chat generator captures this tool call's args and then pivots the SSE
#: stream into ``generate_graph_sse_events`` on the same HTTP response.
_TRAINING_HANDOFF_TOOL = "run_training_pipeline"


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


def _prepare_attached_datasets_block(
    linked: Optional[list[str]],
) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    """Resolve linked dataset entries, register data, build context for the LLM.

    Returns ``(markdown_block, [(ref, dataset_info), ...])`` for SSE ``dataset.resolved``.
    """
    if not linked:
        return "", []
    from backend.training.service import resolve_linked_dataset
    from utils import get_registered_dataset

    chunks: list[str] = []
    events: list[tuple[str, dict[str, Any]]] = []

    for entry in linked:
        raw = str(entry).strip()
        if not raw:
            continue
        ref = resolve_linked_dataset(raw)
        if not ref:
            chunks.append(f"  - ✗ Could not resolve attached entry `{raw}` — try a catalog name.")
            continue
        df = get_registered_dataset(ref)
        if df is None:
            chunks.append(f"  - ✗ `{ref}` resolved but not loaded into memory.")
            continue

        cols = list(df.columns)
        col_line = ", ".join(f"{c}:{str(df[c].dtype)}" for c in cols[:30])
        if len(cols) > 30:
            col_line += ", …"
        sample = df.head(5)
        sample_md = sample.to_csv(index=False)

        chunks.append(
            f"  - **ref**: `{ref}` ({len(df):,} rows × {len(df.columns)} cols)\n"
            f"    columns: {col_line}\n"
            f"    sample (5 rows):\n{sample_md}"
        )
        info = {
            "rows": len(df),
            "n_columns": len(df.columns),
            "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in cols],
            "sample": sample.to_dict(orient="records"),
        }
        events.append((ref, info))

    if not chunks:
        return "", []

    header = (
        "[ATTACHED DATASETS — already loaded in context; do NOT call search_datasets. "
        "Pass these `dataset_ref` strings to analysis tools.]\n\n"
    )
    return header + "\n\n".join(chunks), events


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
    org_id: Optional[str] = None,
) -> None:
    if not experiment_id:
        return
    try:
        import time as _t

        now = int(_t.time() * 1000)
        existing = []
        exp_data = training_repo.get_experiment(experiment_id, org_id=org_id)
        if exp_data and exp_data.get("chat_history"):
            existing = list(exp_data["chat_history"])
        existing.append({"role": "user", "content": user_message, "timestamp": now - 1})
        if agent_text.strip():
            existing.append({"role": "agent", "content": agent_text, "timestamp": now})
        training_repo.save_experiment_chat_history(
            experiment_id, existing, org_id=org_id,
        )
    except Exception:
        pass


def generate_chat_sse(
    thread_id: str,
    message: str,
    training_context: Optional[str] = None,
    experiment_id: Optional[str] = None,
    conversation: Optional[list[dict[str, Any]]] = None,
    org_id: Optional[str] = None,
    model_preference: Optional[str] = None,
    attached_dataset_events: Optional[list[tuple[str, dict[str, Any]]]] = None,
):
    """Stream one Jubilee turn, pivoting into the training sub-agent if the
    ``run_training_pipeline`` tool is called.

    The pivot reuses the same HTTP SSE response:

    1. Jubilee runs normally — tokens, tool calls, tool results flow to the UI.
    2. If Jubilee calls ``run_training_pipeline`` we capture its args.
    3. Right after Jubilee's turn finishes, we call
       :func:`backend.training.service.generate_graph_sse_events` and yield
       its events on the same stream (it emits its own ``stream.start`` with
       ``training_graph=True`` so the UI knows to switch to the pipeline view).
    4. The training sub-agent's summary is persisted via ``save_training_context``
       and prepended to Jubilee's next turn — that's how the main agent "sees"
       the sub-agent's output for follow-ups.
    """
    orchestrator_agent = repository.get_orchestrator_agent()
    # Snapshot thread when sending full transcript so checkpoint state does not hide prior turns.
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
    for ref, ds_info in attached_dataset_events or []:
        yield format_sse(
            dataset_resolved(ref, dataset_info=ds_info, experiment_id=experiment_id),
            experiment_id,
        )

    set_attached_datasets_active(bool(attached_dataset_events))
    seen_tool_call_ids: set[str] = set()
    pending_predict_args: dict[str, dict] = {}
    accumulated_response: list[str] = []
    training_handoff: Optional[dict[str, Any]] = None

    try:
        for event in orchestrator_agent.stream(
            agent_input,
            config=config,
            stream_mode=["messages", "updates"],
        ):
            if isinstance(event, tuple) and len(event) == 2:
                stream_type, payload = event
            else:
                stream_type, payload = "updates", event

            if stream_type == "messages":
                msg_chunk, _metadata = payload if isinstance(payload, tuple) else (payload, {})
                # Tool-result messages are handled via the "updates" pathway
                # (sidecars → analysis.result events); don't echo them as tokens.
                if getattr(msg_chunk, "type", "") == "tool":
                    continue
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
                        raw_content = getattr(tm, "content", "")
                        snippet = raw_content
                        if len(snippet) > 2000:
                            snippet = snippet[:2000] + "…"
                        if name == _TRAINING_HANDOFF_TOOL:
                            # Don't surface the sentinel payload to the UI; just
                            # capture the handoff args and let the pivot happen
                            # after the agent turn.
                            handoff = _parse_training_handoff(raw_content)
                            if handoff is not None:
                                training_handoff = handoff
                            continue
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
                            orch = repository.get_orchestrator_module()
                            sidechannel = getattr(
                                orch, "SIDECHANNEL_TOOL_NAMES", frozenset(),
                            )
                            cleaned, envelopes = extract_analysis_json_sidecars(
                                raw_content if isinstance(raw_content, str) else str(raw_content or ""),
                            )
                            for env in envelopes:
                                if not isinstance(env, dict):
                                    continue
                                tool_nm = str(env.get("tool") or name)
                                kind = str(env.get("kind") or "unknown")
                                yield format_sse(
                                    analysis_result_event(
                                        tool=tool_nm,
                                        kind=kind,
                                        summary=env.get("summary"),
                                        payload=env.get("payload"),
                                        experiment_id=experiment_id,
                                    ),
                                    experiment_id,
                                )
                                if kind == "chart":
                                    spec = (env.get("payload") or {}).get("spec")
                                    if isinstance(spec, dict):
                                        yield format_sse(
                                            analysis_chart_event(
                                                tool_nm,
                                                spec,
                                                experiment_id=experiment_id,
                                            ),
                                            experiment_id,
                                        )
                            if name in sidechannel and envelopes:
                                continue
                            body = cleaned if envelopes else (
                                raw_content if isinstance(raw_content, str) else str(raw_content or "")
                            )
                            snippet = body
                            if len(snippet) > 2000:
                                snippet = snippet[:2000] + "…"
                            yield format_sse(tool_end(name, snippet, experiment_id), experiment_id)

        _persist_chat_exchange(
            experiment_id, message, "".join(accumulated_response), org_id=org_id,
        )

        if training_handoff is not None:
            # Pivot: hand off the rest of this SSE stream to the training
            # sub-agent. ``generate_graph_sse_events`` emits its own
            # ``stream.start(training_graph=True)`` and ``stream.end``.
            from backend.training import service as training_service

            yield from training_service.generate_graph_sse_events(
                training_handoff.get("goal") or (message or "Training run"),
                training_handoff.get("dataset_refs") or None,
                training_handoff.get("model_preference") or model_preference,
                experiment_id=experiment_id,
                conversation=conversation,
            )
            return

        yield format_sse(stream_end(experiment_id), experiment_id)
    except Exception as exc:
        yield format_sse(error_event(str(exc), experiment_id), experiment_id)
    finally:
        reset_attached_datasets_active()


def _parse_training_handoff(raw: object) -> Optional[dict[str, Any]]:
    """Decode the ``run_training_pipeline`` sentinel tool result.

    Imports lazily so ``backend/chat/service.py`` stays testable without
    pulling in the heavy LangChain chain at module import time.
    """
    try:
        from agent import parse_training_handoff as _parse
    except Exception:
        return None
    return _parse(raw)


def generate_background_intake_sse(
    message: str,
    experiment_id: Optional[str] = None,
    conversation: Optional[list[dict[str, Any]]] = None,
    org_id: Optional[str] = None,
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
                exp_data = training_repo.get_experiment(experiment_id, org_id=org_id)
                if exp_data and exp_data.get("chat_history"):
                    existing = list(exp_data["chat_history"])
                existing.append({"role": "user", "content": message, "timestamp": now - 1})
                existing.append({"role": "agent", "content": text, "timestamp": now})
                training_repo.save_experiment_chat_history(
                    experiment_id, existing, org_id=org_id,
                )
            except Exception:
                pass

        yield format_sse(stream_end(experiment_id), experiment_id)
    except Exception as exc:
        yield format_sse(error_event(str(exc), experiment_id), experiment_id)


def _message_with_training_hint(
    message: str,
    linked: Optional[list[str]],
    model_preference: Optional[str],
) -> str:
    """Inject a directive to run the training pipeline into the user message.

    Used when the UI sets ``mode="train"`` (plan approval / guided run). Jubilee
    sees this and should immediately call ``run_training_pipeline`` with the
    supplied refs. Everything still flows through the same agent path so the
    main agent can observe the sub-agent's output for follow-ups.
    """
    refs = [s.strip() for s in (linked or []) if s and str(s).strip()]
    goal = (message or "").strip() or "Training run"
    parts = [
        "[USER APPROVED TRAINING RUN — please call the run_training_pipeline "
        "tool immediately to kick off the pipeline.]",
        f"  Goal: {goal}",
    ]
    if refs:
        parts.append(f"  Dataset ref(s): {', '.join(refs)}")
    if model_preference:
        parts.append(f"  Model preference: {model_preference}")
    parts.append(
        "Do not ask the user any more questions — pass the refs above as "
        "dataset_refs and write no other reply text in this turn."
    )
    hint = "\n".join(parts) + "\n\n"
    return hint + goal


def chat(
    request: ChatRequest,
    org_id: Optional[str] = None,
) -> tuple[str, object]:
    """Single SSE entry point for ``/api/chat``.

    Routing collapses to two cases:

    - **Resume** (``resume_training`` or legacy ``resume``) — resumes the
      paused training sub-agent after a HITL interrupt.
    - **Everything else** — one Jubilee turn via :func:`generate_chat_sse`.
      Training is launched by Jubilee calling the ``run_training_pipeline``
      tool; the generator pivots the SSE stream into the training sub-agent
      at that point. ``mode="train"`` is an optional hint that injects a
      directive into the user message telling Jubilee to call the training
      tool immediately — useful for the plan-approval UI where the user has
      already confirmed.
    """
    from backend.training import service as training_service

    if request.experiment_id and org_id:
        if not training_repo.get_experiment(request.experiment_id, org_id=org_id):
            raise ValueError("Experiment not found or access denied")

    # ---- Training sub-agent: resume after HITL interrupt --------------------

    if request.resume_training is not None:
        gen = training_service.generate_graph_resume_sse_events(
            request.resume_training.thread_id,
            request.resume_training.approved,
            request.resume_training.feedback,
        )
        return request.resume_training.thread_id, gen

    if request.resume is not None:
        if not request.experiment_id:
            raise ValueError("experiment_id required for resume")
        exp = training_repo.get_experiment(request.experiment_id, org_id=org_id)
        if not exp or not exp.get("training_state", {}).get("graph_thread_id"):
            raise ValueError("No active training graph found for this experiment")
        graph_thread = exp["training_state"]["graph_thread_id"]
        gen = training_service.generate_graph_resume_sse_events(
            graph_thread, request.resume.approved, request.resume.feedback,
        )
        return graph_thread, gen

    # ---- Unified Jubilee stream (chat + optional training pivot) ------------

    if request.experiment_id:
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
        gen = training_service.generate_graph_sse_events(
            goal,
            refs,
            request.model_preference,
            experiment_id=request.experiment_id,
            conversation=request.conversation,
        )
        return resolved_thread_id, gen

    training_context = None
    ctx = training_repo.get_latest_training_context(experiment_id=request.experiment_id)
    if ctx:
        training_context = _build_training_context_message(ctx)

    if request.mode == "train":
        # No refs attached but caller still asked for training: keep the legacy
        # behaviour of nudging Jubilee to call ``run_training_pipeline`` itself.
        user_message = _message_with_training_hint(
            request.message, request.linked_datasets, request.model_preference,
        )
    else:
        user_message = request.message or ""

    if attached_block:
        user_message = attached_block + "\n\n" + user_message

    return resolved_thread_id, generate_chat_sse(
        resolved_thread_id,
        user_message,
        training_context,
        experiment_id=request.experiment_id,
        conversation=request.conversation,
        org_id=org_id,
        model_preference=request.model_preference,
        attached_dataset_events=attach_events or None,
    )
