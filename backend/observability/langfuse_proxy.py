"""Proxy endpoint to query Langfuse API for per-step LLM telemetry."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from backend.shared.auth import CurrentUser, get_current_user
from backend.shared.settings import get_settings

router = APIRouter()


@router.get("/api/experiments/{experiment_id}/llm-telemetry")
async def get_llm_telemetry(
    experiment_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Query Langfuse for LLM usage data tagged with this experiment."""
    settings = get_settings()

    if not settings.langfuse_enabled:
        return {"configured": False, "traces": [], "total_cost": None, "total_tokens": None}

    try:
        from langfuse import Langfuse

        client = Langfuse(
            secret_key=settings.langfuse_secret_key,
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            host=settings.LANGFUSE_HOST,
        )

        traces = client.fetch_traces(session_id=experiment_id, limit=100)
        items: list[dict[str, Any]] = []
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0

        for trace in traces.data:
            node_tag = trace.tags[0] if trace.tags else None
            cost = getattr(trace, "total_cost", None) or 0
            inp = getattr(trace, "input_tokens", None) or 0
            out = getattr(trace, "output_tokens", None) or 0
            total_cost += cost
            total_input_tokens += inp
            total_output_tokens += out

            items.append({
                "trace_id": trace.id,
                "name": trace.name,
                "node": node_tag,
                "latency_ms": int(trace.latency * 1000) if trace.latency else None,
                "input_tokens": inp,
                "output_tokens": out,
                "total_cost": round(cost, 6) if cost else None,
                "model": getattr(trace, "model", None),
                "created_at": trace.timestamp.isoformat() if trace.timestamp else None,
            })

        return {
            "configured": True,
            "experiment_id": experiment_id,
            "traces": items,
            "total_cost": round(total_cost, 6),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
        }

    except ImportError:
        return {"configured": False, "traces": [], "error": "langfuse package not installed"}
    except Exception as e:
        return {"configured": False, "traces": [], "error": str(e)}
