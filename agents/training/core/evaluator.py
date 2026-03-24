"""
Agentic evaluator node for the ML Training Agent.

After each pipeline step completes, the evaluator reviews the result and
decides how to proceed: continue to the next planned step, amend the
remaining plan, trigger a full replan, or declare the pipeline done.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from agents.training.utils.graph_stream_hooks import (
    GraphTokenStreamHandler,
    emit_graph_stream,
)

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

if TYPE_CHECKING:
    from .state import TrainingAgentState


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------

class AmendedStep(BaseModel):
    step: str = Field(description="Step identifier")
    rationale: str = Field(description="Why this step is in the amended plan")


class EvaluatorDecision(BaseModel):
    decision: Literal["continue", "amend", "replan", "done"] = Field(
        description=(
            "continue = proceed to next step in plan; "
            "amend = modify remaining steps then continue; "
            "replan = go back to planner for a fundamentally different plan; "
            "done = pipeline complete (only after generate_report)"
        )
    )
    reasoning: str = Field(description="Brief explanation of the decision")
    amended_remaining_steps: Optional[list[AmendedStep]] = Field(
        default=None,
        description="New remaining steps (only when decision='amend')",
    )
    replan_reason: Optional[str] = Field(
        default=None,
        description="Why a full replan is needed (only when decision='replan')",
    )


# ---------------------------------------------------------------------------
# Evaluator prompt
# ---------------------------------------------------------------------------

EVALUATOR_PROMPT = """\
You are an ML pipeline evaluator.  A step has just completed. Review the result
and decide what to do next.

## Just-Completed Step

Name: {completed_step}

Summary:
{step_summary}

## Current Plan State

Steps already completed: {completed_steps}
Remaining steps in plan: {remaining_steps}

## Key State Fields

{state_context}

## Decision Options

1. **continue** — the result looks good; proceed to the next step in the plan.
2. **amend** — the result reveals that the remaining plan should change
   (insert, remove, or reorder steps).  Provide the new `amended_remaining_steps`.
3. **replan** — something is fundamentally wrong (wrong model family, unsuitable
   data, training failure) and the planner should produce a brand-new plan.
   Provide `replan_reason`.
4. **done** — the pipeline is complete.  Only valid after `generate_report` has
   run.

## Rules

- **STRONGLY prefer `continue`** when the step succeeded and produced a valid
  result.  Most steps that run without error should just continue.
- Only use **amend** when the result *concretely* indicates the remaining plan
  is wrong (e.g. a step that should have been skipped is still in the plan,
  or a prerequisite was missed).
- **NEVER** re-add a step that just completed successfully.  If `data_collection`
  just succeeded, the amended plan must NOT start with `data_collection` again.
  Similarly for any other step — do not duplicate a step that already produced
  a valid result.
- If the step failed with a recoverable error, prefer **amend** to retry it
  (this is the one case where repeating the same step name is allowed).
- If training metrics are very poor (e.g. accuracy near random, R² < 0.05),
  consider **replan** with a different model family.
- If the completed step is `generate_report` and the report was saved
  successfully, return **done**.
- When amending, only supply the *remaining* steps (not the ones already done).
  All amended steps must be valid step names.
"""

MAX_PLAN_LENGTH = 15
MAX_CONSECUTIVE_AMENDS = 3
MAX_REPLANS = 2
MAX_EVALUATOR_CALLS = 30


def _summarise_completed_step(state: "TrainingAgentState") -> str:
    """Produce a human-readable summary of the last completed step's output."""
    step = state.get("current_step", "")
    parts: list[str] = []

    if step == "data_collection":
        parts.append(f"Dataset: {state.get('collected_dataset_ref', 'N/A')}")
        parts.append(f"Source: {state.get('data_source', 'N/A')}")
    elif step == "select_model":
        parts.append(f"Family: {state.get('selected_model', 'N/A')}")
        parts.append(f"Task type: {state.get('task_type', 'N/A')}")
        parts.append(f"Explanation: {state.get('model_explanation', 'N/A')}")
    elif step == "cleaning":
        parts.append(f"Cleaned ref: {state.get('cleaned_dataset_ref', 'N/A')}")
        parts.append(f"Summary: {state.get('cleaning_summary', 'N/A')}")
    elif step == "label_split_definition":
        ld = state.get("label_definition") or {}
        parts.append(f"Target: {ld.get('target_column', 'N/A')}")
        parts.append(f"Split: {ld.get('split_strategy', 'N/A')}")
    elif step in ("feature_selection_specification", "feature_engineering_executor"):
        parts.append(f"Transformed train: {state.get('transformed_train_ref', 'N/A')}")
        parts.append(f"Validation passed: {state.get('feature_validation_passed', False)}")
    elif step == "training_approval":
        tp = state.get("training_plan") or {}
        parts.append(f"Model type: {tp.get('model_type', 'N/A')}")
        parts.append(f"Strategy: {tp.get('strategy_notes', 'N/A')}")
    elif step == "training":
        tm = state.get("training_metrics") or {}
        parts.append(f"Success: {tm.get('success', 'N/A')}")
        parts.append(f"Iterations: {tm.get('num_iterations', 'N/A')}")
        for k in ("val_accuracy", "val_roc_auc", "val_r2", "silhouette_score", "davies_bouldin"):
            v = tm.get(k)
            if v is not None:
                parts.append(f"{k}: {v}")
        if tm.get("feature_redo_requested"):
            parts.append(f"Feature redo requested: {tm.get('feature_redo_reason', 'N/A')}")
    elif step == "generate_report":
        parts.append(f"Report path: {state.get('report_path', 'N/A')}")

    return "\n".join(parts) if parts else "(no summary available)"


def _state_context(state: "TrainingAgentState") -> str:
    """Lightweight dump of key state values for the evaluator."""
    keys = [
        "goal", "selected_model", "task_type", "collected_dataset_ref",
        "cleaned_dataset_ref", "training_iteration", "feature_redo_iteration",
    ]
    lines = []
    for k in keys:
        v = state.get(k)
        if v is not None:
            lines.append(f"- {k}: {v}")
    err = state.get("error")
    if err:
        lines.append(f"- error: {err}")
    return "\n".join(lines) if lines else "- (minimal state)"


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def evaluator_node(state: "TrainingAgentState") -> "TrainingAgentState":
    """Evaluate the last step and decide the next action."""

    plan = state.get("plan") or []
    plan_index = state.get("plan_index", 0)

    completed_step = plan[plan_index]["step"] if plan_index < len(plan) else state.get("current_step", "")
    completed_names = [s["step"] for s in plan[:plan_index + 1]]
    remaining = plan[plan_index + 1:]
    remaining_names = [s["step"] for s in remaining]

    # Fast-path: if the completed step is generate_report, we're done
    if completed_step == "generate_report":
        return {
            **state,
            "evaluator_decision": "done",
            "plan_index": plan_index + 1,
            "audit_trace": state.get("audit_trace", []) + [{
                "step": "evaluator",
                "completed_step": completed_step,
                "decision": "done",
                "reasoning": "Pipeline finished after generate_report.",
            }],
        }

    # Fast-path: if no remaining steps, done
    if not remaining:
        return {
            **state,
            "evaluator_decision": "done",
            "plan_index": plan_index + 1,
        }

    audit = state.get("audit_trace", [])
    evaluator_calls = sum(1 for e in audit if e.get("step") == "evaluator")
    replan_count = sum(
        1 for e in audit
        if e.get("step") == "evaluator" and e.get("decision") == "replan"
    )

    # Hard safety: abort if we've exhausted evaluator budget
    if evaluator_calls >= MAX_EVALUATOR_CALLS:
        emit_graph_stream({
            "phase": "evaluator",
            "message": f"Safety limit reached ({MAX_EVALUATOR_CALLS} evaluator cycles). Stopping pipeline.",
        })
        return {
            **state,
            "evaluator_decision": "done",
            "plan_index": plan_index + 1,
            "error": f"Pipeline stopped: exceeded {MAX_EVALUATOR_CALLS} evaluator cycles.",
            "audit_trace": audit + [{
                "step": "evaluator",
                "completed_step": completed_step,
                "decision": "done",
                "reasoning": f"Forced stop — {evaluator_calls} evaluator cycles reached.",
            }],
        }

    # Count consecutive amends from audit_trace to enforce the safety cap
    consecutive_amends = 0
    for entry in reversed(audit):
        if entry.get("step") == "evaluator" and entry.get("decision") == "amend":
            consecutive_amends += 1
        else:
            break

    step_summary = _summarise_completed_step(state)
    ctx = _state_context(state)

    budget_warning = ""
    if replan_count >= 1:
        budget_warning = (
            f"\n\n## ⚠️ Budget Warning\n"
            f"This pipeline has already replanned {replan_count} time(s) "
            f"(max allowed: {MAX_REPLANS}). "
            f"**Avoid replanning unless truly necessary.** "
            f"Prefer `continue` or `amend` to move forward."
        )

    prompt = EVALUATOR_PROMPT.format(
        completed_step=completed_step,
        step_summary=step_summary,
        completed_steps=", ".join(completed_names),
        remaining_steps=", ".join(remaining_names) or "(none)",
        state_context=ctx,
    ) + budget_warning

    emit_graph_stream({
        "phase": "evaluator",
        "message": f"Reviewing `{completed_step}` — deciding next move…",
    })
    token_handler = GraphTokenStreamHandler(phase="evaluator")
    llm = init_chat_model(model="gpt-5.4-mini", temperature=0, streaming=True)
    structured_llm = llm.with_structured_output(EvaluatorDecision)
    decision: EvaluatorDecision = structured_llm.invoke(
        prompt, config={"callbacks": [token_handler]}
    )
    reason_snip = (decision.reasoning or "").strip()
    if len(reason_snip) > 120:
        reason_snip = reason_snip[:120] + "…"
    emit_graph_stream({
        "phase": "evaluator",
        "message": f"Evaluator: {decision.decision}" + (f" — {reason_snip}" if reason_snip else ""),
    })

    # Cap replans — force stop if the pipeline keeps failing
    if decision.decision == "replan" and replan_count >= MAX_REPLANS:
        emit_graph_stream({
            "phase": "evaluator",
            "message": f"Replan limit reached ({MAX_REPLANS}). Stopping pipeline and returning control.",
        })
        return {
            **state,
            "evaluator_decision": "done",
            "plan_index": plan_index + 1,
            "error": (
                f"Pipeline stopped after {MAX_REPLANS} replans. "
                f"Last reason: {decision.replan_reason or decision.reasoning}"
            ),
            "audit_trace": audit + [{
                "step": "evaluator",
                "completed_step": completed_step,
                "decision": "done",
                "reasoning": f"Forced stop — {MAX_REPLANS} replans exhausted. {decision.reasoning}",
                "replan_reason": decision.replan_reason,
            }],
        }

    new_plan = list(plan)
    new_index = plan_index + 1

    if decision.decision == "amend" and decision.amended_remaining_steps:
        if consecutive_amends >= MAX_CONSECUTIVE_AMENDS:
            decision.decision = "continue"
            decision.reasoning = (
                f"Forced continue — {MAX_CONSECUTIVE_AMENDS} consecutive amends reached. "
                + (decision.reasoning or "")
            )
        else:
            amended = [{"step": s.step, "rationale": s.rationale} for s in decision.amended_remaining_steps]
            done_set = set(completed_names)
            amended = [s for s in amended if s["step"] not in done_set]
            new_plan = list(plan[:plan_index + 1]) + amended
            if len(new_plan) > MAX_PLAN_LENGTH:
                new_plan = new_plan[:MAX_PLAN_LENGTH]
            new_index = plan_index + 1

    replan_reason = decision.replan_reason or decision.reasoning if decision.decision == "replan" else None

    return {
        **state,
        "plan": new_plan,
        "plan_index": new_index,
        "evaluator_decision": decision.decision if decision.decision != "amend" else "continue",
        "audit_trace": state.get("audit_trace", []) + [{
            "step": "evaluator",
            "completed_step": completed_step,
            "decision": decision.decision,
            "reasoning": decision.reasoning,
            "replan_reason": replan_reason,
        }],
        "plan_history": (
            state.get("plan_history", []) + [{
                "steps": [s for s in plan],
                "strategy": state.get("plan_strategy"),
                "replaced_at": None,
                "replan_reason": replan_reason,
            }]
            if decision.decision == "replan"
            else state.get("plan_history", [])
        ),
    }
