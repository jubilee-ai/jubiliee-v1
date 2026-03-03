import json
import uuid
from typing import Optional

from agents.training.utils.streaming import build_node_update
from backend.chat import repository
from backend.shared.serialization import serialize_state
from backend.shared.state import chat_threads_with_context, last_training_context


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


def _emit_training_step_events(thread_id: str):
    from agents.training.core.state import STEP_ORDER

    mod = repository.get_orchestrator_module()
    state = getattr(mod, "_last_training_state", None)
    if not state:
        return

    yield f"data: {json.dumps({'type': 'training_started', 'thread_id': thread_id})}\n\n"
    for step_name in STEP_ORDER:
        try:
            update = build_node_update(step_name, state)
            update["thread_id"] = thread_id
            yield f"data: {json.dumps(serialize_state(update))}\n\n"
        except Exception:
            pass
    yield f"data: {json.dumps({'type': 'training_completed', 'thread_id': thread_id})}\n\n"


def generate_chat_sse(
    thread_id: str, message: str, training_context: Optional[str] = None
):
    orchestrator_agent = repository.get_orchestrator_agent()
    config = {"configurable": {"thread_id": thread_id}}

    context_block = training_context or ""
    if (
        not context_block
        and last_training_context
        and thread_id not in chat_threads_with_context
    ):
        context_block = _build_training_context_message(last_training_context)

    if context_block:
        chat_threads_with_context.add(thread_id)
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
                            payload = {
                                "type": "tool_call",
                                "tool": tc["name"],
                                "args": serialize_state(tc.get("args", {})),
                                "thread_id": thread_id,
                            }
                            yield f"data: {json.dumps(payload)}\n\n"
                    raw_content = getattr(msg, "content", "")
                    if isinstance(raw_content, list):
                        content = "".join(
                            block.get("text", "") if isinstance(block, dict) else str(block)
                            for block in raw_content
                        )
                    else:
                        content = raw_content or ""
                    if content:
                        payload = {
                            "type": "token",
                            "content": content,
                            "thread_id": thread_id,
                        }
                        yield f"data: {json.dumps(payload)}\n\n"

            if "tools" in chunk:
                tool_msgs = chunk["tools"].get("messages", [])
                for tm in tool_msgs:
                    name = getattr(tm, "name", "unknown")
                    snippet = getattr(tm, "content", "")
                    if len(snippet) > 500:
                        snippet = snippet[:500] + "…"
                    payload = {
                        "type": "tool_result",
                        "tool": name,
                        "result": snippet,
                        "thread_id": thread_id,
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    if name == "train_model":
                        yield from _emit_training_step_events(thread_id)

        yield f"data: {json.dumps({'type': 'end', 'thread_id': thread_id})}\n\n"
    except Exception as exc:
        yield f"data: {json.dumps({'type': 'error', 'error': str(exc), 'thread_id': thread_id})}\n\n"


def chat(message: str, thread_id: Optional[str], training_context: Optional[str]):
    resolved_thread_id = thread_id or f"chat-{uuid.uuid4().hex[:8]}"
    return resolved_thread_id, generate_chat_sse(
        resolved_thread_id, message, training_context
    )
