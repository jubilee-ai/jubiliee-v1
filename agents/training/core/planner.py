"""
Agentic planner node for the ML Training Agent.

Generates a dynamic execution plan based on goal, data context, and user
preferences.  The plan is a list of step names with per-step rationale;
the planner can skip, reorder, or annotate steps as it sees fit.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from agents.training.utils.graph_stream_hooks import emit_graph_stream

from .hitl import run_with_hitl

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

if TYPE_CHECKING:
    from .state import TrainingAgentState


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------

class PlanStep(BaseModel):
    step: str = Field(
        description="Step identifier — one of: data_collection, select_model, "
        "cleaning, label_split_definition, feature_selection_specification, "
        "feature_engineering_executor, training_approval, training, generate_report"
    )
    rationale: str = Field(
        description=(
            "Ultra-brief note for this step only: max ~12 words / 90 characters. "
            "No bullet lists or step-by-step prose — the user sees this in a small review card."
        )
    )


class Plan(BaseModel):
    steps: list[PlanStep] = Field(description="Ordered list of steps to execute")
    skipped_steps: list[str] = Field(
        default_factory=list,
        description="Steps intentionally omitted from the plan",
    )
    skip_rationale: str = Field(
        default="",
        description=(
            "If skipped_steps non-empty: one short sentence (~150 chars) why those steps are omitted; "
            "otherwise empty string."
        ),
    )
    strategy: str = Field(
        description=(
            "Very short overall approach for human review: max 2 sentences or ~220 characters total. "
            "No markdown headings, no numbered lists — plain, scannable text only."
        )
    )


# ---------------------------------------------------------------------------
# Valid step names (used for validation)
# ---------------------------------------------------------------------------

def _clamp_review_text(text: str, max_len: int) -> str:
    """Trim text for HITL / compact UI; break on last space when possible."""
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    cut = t[: max_len - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "…"


ALL_STEP_NAMES: set[str] = {
    "data_collection",
    "select_model",
    "cleaning",
    "label_split_definition",
    "feature_selection_specification",
    "feature_engineering_executor",
    "training_approval",
    "training",
    "generate_report",
}


# ---------------------------------------------------------------------------
# Planner prompt
# ---------------------------------------------------------------------------

PLANNER_PROMPT = """\
You are an expert ML pipeline planner.  Given the user's training goal and any
available context, produce an execution plan: an ordered list of pipeline steps
to run, along with steps to skip and why.

## Available Steps

| Step name | What it does |
|---|---|
| data_collection | Load or retrieve the dataset(s) |
| select_model | Choose the model family: supervised, unsupervised, or neural_networks |
| cleaning | Clean and standardise the raw data |
| label_split_definition | Define the target column and create train / val / test splits |
| feature_selection_specification | Analyse data and specify which features to engineer |
| feature_engineering_executor | Execute the feature transformations |
| training_approval | Propose a training configuration (hyperparameters, strategy) |
| training | Train the model and evaluate on validation / test sets |
| generate_report | Produce and save the final training report |

## Dependency Constraints (must be respected)

* `data_collection` must appear before any step that needs the dataset
  (cleaning, label_split_definition, feature_*, training).
* `select_model` must appear before `training_approval` and `training`.
* `label_split_definition` must come after `cleaning` when included.
* `feature_engineering_executor` must come after `feature_selection_specification`.
* `training_approval` must come directly before `training`.
* `generate_report` must be the last step.

## Skipping Guidelines

* For **unsupervised** tasks you MAY skip `label_split_definition` (no target
  column), and optionally `feature_selection_specification` /
  `feature_engineering_executor` if the data already has suitable numeric
  features.
* If the user provides clean, pre-processed data and describes it as such, you
  MAY skip `cleaning`.
* If the user explicitly specifies a model family, `select_model` can still run
  but may be quick.
* NEVER skip `data_collection`, `training`, or `generate_report`.

## Replan Context

{replan_context}

## User Goal

{goal}

## Additional Context

- Linked datasets: {datasets}
- User model preference: {preference}
{state_summary}

## Instructions

Produce a Plan JSON object. Be decisive — include only the steps that are
genuinely needed for this particular goal and dataset.

## Brevity (required — shown in the human approval card)

The user sees `strategy`, each step's `rationale`, and `skip_rationale` in a **compact** UI.
- **strategy**: At most **two short sentences** OR **220 characters** total (whichever is tighter). No headings, no bullets, no pipeline essay.
- **rationale** (each step): **One short phrase** — max **~12 words / 90 characters**. State purpose in a glance (e.g. "Load user CSV" / "Approve hyperparams before fit"). No multi-sentence explanations.
- **skip_rationale**: Only if `skipped_steps` is non-empty: **one sentence**, max **~150 characters**. Otherwise use `""`.
"""


def _build_state_summary(state: "TrainingAgentState") -> str:
    """Build a context string from already-completed state fields."""
    parts: list[str] = []
    if state.get("selected_model"):
        parts.append(f"- Selected model family: {state['selected_model']}")
    if state.get("task_type"):
        parts.append(f"- Task type: {state['task_type']}")
    if state.get("collected_dataset_ref"):
        parts.append(f"- Dataset loaded: {state['collected_dataset_ref']}")
    if state.get("cleaned_dataset_ref"):
        parts.append(f"- Cleaned dataset: {state['cleaned_dataset_ref']}")
    if state.get("training_metrics"):
        metrics = state["training_metrics"]
        parts.append(f"- Previous training result: success={metrics.get('success')}, "
                      f"val_accuracy={metrics.get('val_accuracy')}, "
                      f"silhouette={metrics.get('silhouette_score')}")
    if state.get("feature_redo_reason"):
        parts.append(f"- Feature redo reason: {state['feature_redo_reason']}")
    return "\n".join(parts) if parts else "- (first run — no prior context)"


def _validate_plan(plan: Plan) -> Plan:
    """Enforce hard constraints on the plan.

    Silently fixes issues rather than rejecting the whole plan — the LLM is
    usually close but occasionally forgets a constraint.
    """
    step_names = [s.step for s in plan.steps]

    for s in plan.steps:
        if s.step not in ALL_STEP_NAMES:
            raise ValueError(f"Unknown step in plan: {s.step}")

    required = {"data_collection", "training", "generate_report"}
    for req in required:
        if req not in step_names:
            if req == "data_collection":
                plan.steps.insert(0, PlanStep(step=req, rationale="Auto-inserted: always required"))
            elif req == "training":
                idx = next((i for i, s in enumerate(plan.steps) if s.step == "generate_report"), len(plan.steps))
                plan.steps.insert(idx, PlanStep(step=req, rationale="Auto-inserted: always required"))
            else:
                plan.steps.append(PlanStep(step=req, rationale="Auto-inserted: always required"))

    step_names = [s.step for s in plan.steps]
    if "training_approval" in step_names and "training" in step_names:
        ta_idx = step_names.index("training_approval")
        tr_idx = step_names.index("training")
        if ta_idx > tr_idx:
            plan.steps[ta_idx], plan.steps[tr_idx] = plan.steps[tr_idx], plan.steps[ta_idx]

    if step_names[-1] != "generate_report":
        gr = next(s for s in plan.steps if s.step == "generate_report")
        plan.steps.remove(gr)
        plan.steps.append(gr)

    # Keep planner output short for the HITL card even if the model drifts.
    plan.strategy = _clamp_review_text(plan.strategy, 220)
    plan.skip_rationale = _clamp_review_text(plan.skip_rationale, 150)
    for ps in plan.steps:
        ps.rationale = _clamp_review_text(ps.rationale, 90)

    return plan


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def planner_node(state: "TrainingAgentState") -> "TrainingAgentState":
    """Generate (or regenerate) an execution plan with HITL approval."""

    def do_work(s: "TrainingAgentState", feedback: Optional[str]) -> "TrainingAgentState":
        replan_context = "This is the initial plan."
        prev_plans = s.get("plan_history") or []
        if prev_plans:
            last = prev_plans[-1]
            replan_context = (
                f"This is a REPLAN.  The previous plan was:\n"
                f"{last.get('steps')}\n\n"
                f"Reason for replanning: {last.get('replan_reason', 'evaluator requested replan')}"
            )
        if feedback:
            replan_context += f"\n\nUser feedback on plan: {feedback}"

        prompt = PLANNER_PROMPT.format(
            goal=s.get("goal", ""),
            datasets=s.get("linked_datasets") or "None provided",
            preference=s.get("user_model_preference") or "None",
            state_summary=_build_state_summary(s),
            replan_context=replan_context,
        )

        emit_graph_stream({
            "phase": "planner",
            "message": "Planning: drafting execution steps…",
        })
        llm = init_chat_model(model="gpt-5.4-mini", temperature=0)
        structured_llm = llm.with_structured_output(Plan)
        plan: Plan = structured_llm.invoke(prompt)
        plan = _validate_plan(plan)
        emit_graph_stream({
            "phase": "planner",
            "message": f"Plan ready — {len(plan.steps)} step(s); review when prompted.",
        })

        new_history = list(prev_plans)
        if s.get("plan"):
            new_history.append({
                "steps": [st.model_dump() if hasattr(st, "model_dump") else st for st in (s["plan"] or [])],
                "strategy": s.get("plan_strategy"),
                "replaced_at": datetime.now().isoformat(),
                "replan_reason": replan_context,
            })

        return {
            **s,
            "plan": [step.model_dump() for step in plan.steps],
            "plan_index": 0,
            "plan_strategy": plan.strategy,
            "plan_history": new_history,
            "evaluator_decision": None,
            "current_step": "planner",
            "audit_trace": s.get("audit_trace", []) + [{
                "step": "planner",
                "action": "plan_generated",
                "num_steps": len(plan.steps),
                "skipped": plan.skipped_steps,
                "strategy": plan.strategy,
            }],
        }

    def get_summary(result: "TrainingAgentState") -> str:
        plan_steps = result.get("plan") or []
        strategy = result.get("plan_strategy", "N/A")
        lines = [f"**Plan:** {strategy}"]
        names: list[str] = []
        for step in plan_steps:
            name = step["step"] if isinstance(step, dict) else step.step
            names.append(str(name).replace("_", " "))
        if names:
            lines.append("**Steps:** " + " → ".join(f"`{n}`" for n in names))
        history = result.get("plan_history") or []
        if history:
            lines.append(f"_(Replan #{len(history)})_")
        return "\n".join(lines)

    return run_with_hitl("planner", state, do_work, get_summary)
