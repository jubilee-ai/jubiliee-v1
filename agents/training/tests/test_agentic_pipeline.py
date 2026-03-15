"""
Unit tests for the agentic pipeline redesign: planner, evaluator, dispatcher,
edges, state, and graph construction.

No LLM calls — uses mocks for structured output. Runs quickly.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent.parent.parent
for p in [
    str(ROOT / "tools" / "data-tools"),
    str(ROOT / "tools" / "models-tools" / "training"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        msg = f"  FAIL: {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


# ─── State: new plan fields ─────────────────────────────────────────

print("\n=== State: plan fields ===")

from agents.training.core.state import (
    TrainingAgentState,
    create_initial_state,
    STATE_SNAPSHOT_KEYS,
)

state = create_initial_state("Cluster my customers", linked_datasets=["customers.csv"])

check("plan initialised to None", state["plan"] is None)
check("plan_index initialised to 0", state["plan_index"] == 0)
check("plan_history initialised to empty list", state["plan_history"] == [])
check("plan_strategy initialised to None", state["plan_strategy"] is None)
check("evaluator_decision initialised to None", state["evaluator_decision"] is None)
check("current_step starts at planner", state["current_step"] == "planner")
check("plan in STATE_SNAPSHOT_KEYS", "plan" in STATE_SNAPSHOT_KEYS)
check("plan_index in STATE_SNAPSHOT_KEYS", "plan_index" in STATE_SNAPSHOT_KEYS)
check("plan_strategy in STATE_SNAPSHOT_KEYS", "plan_strategy" in STATE_SNAPSHOT_KEYS)


# ─── Planner: schemas ────────────────────────────────────────────────

print("\n=== Planner: schemas ===")

from agents.training.core.planner import (
    PlanStep,
    Plan,
    ALL_STEP_NAMES,
    _validate_plan,
    _build_state_summary,
)

step = PlanStep(step="data_collection", rationale="Load the dataset")
check("PlanStep creation", step.step == "data_collection")

plan = Plan(
    steps=[
        PlanStep(step="data_collection", rationale="Load data"),
        PlanStep(step="select_model", rationale="Choose family"),
        PlanStep(step="training", rationale="Train"),
        PlanStep(step="generate_report", rationale="Report"),
    ],
    skipped_steps=["cleaning", "label_split_definition"],
    skip_rationale="Pre-cleaned data, unsupervised task",
    strategy="Direct clustering on clean numeric features",
)
check("Plan creation with 4 steps", len(plan.steps) == 4)
check("Plan skipped_steps", len(plan.skipped_steps) == 2)

check("ALL_STEP_NAMES has 9 entries", len(ALL_STEP_NAMES) == 9)
check("data_collection in ALL_STEP_NAMES", "data_collection" in ALL_STEP_NAMES)
check("generate_report in ALL_STEP_NAMES", "generate_report" in ALL_STEP_NAMES)


# ─── Planner: validation ─────────────────────────────────────────────

print("\n=== Planner: _validate_plan ===")

missing_dc = Plan(
    steps=[
        PlanStep(step="select_model", rationale="Choose"),
        PlanStep(step="training", rationale="Train"),
        PlanStep(step="generate_report", rationale="Report"),
    ],
    strategy="Test",
)
validated = _validate_plan(missing_dc)
step_names = [s.step for s in validated.steps]
check("Auto-inserts data_collection when missing", "data_collection" in step_names)
check("data_collection inserted at position 0", step_names[0] == "data_collection")

misplaced_report = Plan(
    steps=[
        PlanStep(step="data_collection", rationale="Load"),
        PlanStep(step="generate_report", rationale="Report"),
        PlanStep(step="training", rationale="Train"),
    ],
    strategy="Test",
)
validated2 = _validate_plan(misplaced_report)
step_names2 = [s.step for s in validated2.steps]
check("generate_report moved to last", step_names2[-1] == "generate_report")

swapped_approval = Plan(
    steps=[
        PlanStep(step="data_collection", rationale="Load"),
        PlanStep(step="select_model", rationale="Choose"),
        PlanStep(step="training", rationale="Train"),
        PlanStep(step="training_approval", rationale="Approve"),
        PlanStep(step="generate_report", rationale="Report"),
    ],
    strategy="Test",
)
validated3 = _validate_plan(swapped_approval)
step_names3 = [s.step for s in validated3.steps]
ta_idx = step_names3.index("training_approval")
tr_idx = step_names3.index("training")
check("training_approval comes before training after fix", ta_idx < tr_idx)

try:
    bad_plan = Plan(
        steps=[PlanStep(step="nonexistent_step", rationale="Bad")],
        strategy="Test",
    )
    _validate_plan(bad_plan)
    check("Rejects unknown step names", False, "Should have raised ValueError")
except ValueError:
    check("Rejects unknown step names", True)


# ─── Planner: _build_state_summary ───────────────────────────────────

print("\n=== Planner: _build_state_summary ===")

empty_state = create_initial_state("Test goal")
summary = _build_state_summary(empty_state)
check("Empty state gives 'first run' summary", "first run" in summary)

rich_state = create_initial_state("Test goal")
rich_state["selected_model"] = "supervised"
rich_state["task_type"] = "classification"
rich_state["collected_dataset_ref"] = "my_dataset"
summary2 = _build_state_summary(rich_state)
check("Rich state includes selected_model", "supervised" in summary2)
check("Rich state includes task_type", "classification" in summary2)
check("Rich state includes dataset", "my_dataset" in summary2)


# ─── Dispatcher ──────────────────────────────────────────────────────

print("\n=== Dispatcher ===")

from agents.training.core.dispatcher import dispatcher_node

dispatch_state = create_initial_state("Test")
dispatch_state["plan"] = [
    {"step": "data_collection", "rationale": "Load data"},
    {"step": "select_model", "rationale": "Choose family"},
    {"step": "training", "rationale": "Train"},
    {"step": "generate_report", "rationale": "Report"},
]
dispatch_state["plan_index"] = 0

result = dispatcher_node(dispatch_state)
check("Dispatcher routes to first step", result["current_step"] == "data_collection")

dispatch_state2 = {**dispatch_state, "plan_index": 1}
result2 = dispatcher_node(dispatch_state2)
check("Dispatcher routes to second step", result2["current_step"] == "select_model")

dispatch_state3 = {**dispatch_state, "plan_index": 4}
result3 = dispatcher_node(dispatch_state3)
check("Dispatcher routes to 'done' when exhausted", result3["current_step"] == "done")

empty_plan_state = create_initial_state("Test")
empty_plan_state["plan"] = []
result4 = dispatcher_node(empty_plan_state)
check("Dispatcher handles empty plan", result4["current_step"] == "done")

none_plan_state = create_initial_state("Test")
none_plan_state["plan"] = None
result5 = dispatcher_node(none_plan_state)
check("Dispatcher handles None plan", result5["current_step"] == "done")


# ─── Edges ───────────────────────────────────────────────────────────

print("\n=== Edges ===")

from agents.training.core.edges import route_to_step, should_continue

edge_state = create_initial_state("Test")
edge_state["current_step"] = "cleaning"
check("route_to_step returns current_step", route_to_step(edge_state) == "cleaning")

edge_state["current_step"] = "done"
check("route_to_step returns 'done'", route_to_step(edge_state) == "done")

edge_state["evaluator_decision"] = "continue"
check("should_continue returns 'continue'", should_continue(edge_state) == "continue")

edge_state["evaluator_decision"] = "replan"
check("should_continue returns 'replan'", should_continue(edge_state) == "replan")

edge_state["evaluator_decision"] = "done"
check("should_continue returns 'done'", should_continue(edge_state) == "done")

edge_state["evaluator_decision"] = "invalid_value"
check("should_continue defaults to 'done' for unknown", should_continue(edge_state) == "done")


# ─── Evaluator: fast paths ──────────────────────────────────────────

print("\n=== Evaluator: fast paths ===")

from agents.training.core.evaluator import evaluator_node, _summarise_completed_step, _state_context

eval_state = create_initial_state("Cluster customers")
eval_state["plan"] = [
    {"step": "data_collection", "rationale": "Load"},
    {"step": "select_model", "rationale": "Choose"},
    {"step": "training", "rationale": "Train"},
    {"step": "generate_report", "rationale": "Report"},
]
eval_state["plan_index"] = 3  # generate_report just completed
eval_state["current_step"] = "generate_report"
eval_state["report_path"] = "/tmp/report.json"

result_eval = evaluator_node(eval_state)
check("Evaluator returns 'done' after generate_report", result_eval["evaluator_decision"] == "done")
check("Evaluator advances plan_index", result_eval["plan_index"] == 4)

eval_state2 = create_initial_state("Test")
eval_state2["plan"] = [{"step": "data_collection", "rationale": "Load"}]
eval_state2["plan_index"] = 0
eval_state2["current_step"] = "data_collection"
eval_state2["collected_dataset_ref"] = "my_data"
result_eval2 = evaluator_node(eval_state2)
check("Evaluator returns 'done' when no remaining steps", result_eval2["evaluator_decision"] == "done")


# ─── Evaluator: _summarise_completed_step ────────────────────────────

print("\n=== Evaluator: _summarise_completed_step ===")

sum_state = create_initial_state("Test")
sum_state["current_step"] = "data_collection"
sum_state["collected_dataset_ref"] = "iris_data"
sum_state["data_source"] = "local"
summary = _summarise_completed_step(sum_state)
check("Summary includes dataset ref", "iris_data" in summary)

sum_state2 = create_initial_state("Test")
sum_state2["current_step"] = "select_model"
sum_state2["selected_model"] = "unsupervised"
sum_state2["task_type"] = "unsupervised"
summary2 = _summarise_completed_step(sum_state2)
check("Summary includes model family", "unsupervised" in summary2)

sum_state3 = create_initial_state("Test")
sum_state3["current_step"] = "training"
sum_state3["training_metrics"] = {"success": True, "num_iterations": 3, "val_accuracy": 0.95}
summary3 = _summarise_completed_step(sum_state3)
check("Summary includes training metrics", "0.95" in summary3)


# ─── Evaluator: _state_context ───────────────────────────────────────

print("\n=== Evaluator: _state_context ===")

ctx_state = create_initial_state("Predict churn")
ctx_state["selected_model"] = "supervised"
ctx_state["task_type"] = "classification"
ctx = _state_context(ctx_state)
check("State context includes goal", "Predict churn" in ctx)
check("State context includes selected_model", "supervised" in ctx)


# ─── Evaluator: LLM path (mocked) ───────────────────────────────────

print("\n=== Evaluator: LLM decision (mocked) ===")

from agents.training.core.evaluator import EvaluatorDecision

mock_decision_continue = EvaluatorDecision(
    decision="continue",
    reasoning="Step succeeded, moving on.",
)

eval_llm_state = create_initial_state("Cluster customers")
eval_llm_state["plan"] = [
    {"step": "data_collection", "rationale": "Load"},
    {"step": "select_model", "rationale": "Choose"},
    {"step": "training", "rationale": "Train"},
    {"step": "generate_report", "rationale": "Report"},
]
eval_llm_state["plan_index"] = 0
eval_llm_state["current_step"] = "data_collection"
eval_llm_state["collected_dataset_ref"] = "customers"

mock_structured = MagicMock()
mock_structured.invoke.return_value = mock_decision_continue

mock_llm = MagicMock()
mock_llm.with_structured_output.return_value = mock_structured

with patch("agents.training.core.evaluator.init_chat_model", return_value=mock_llm):
    result_llm = evaluator_node(eval_llm_state)
    check("LLM continue: decision is 'continue'", result_llm["evaluator_decision"] == "continue")
    check("LLM continue: plan_index advanced", result_llm["plan_index"] == 1)
    check("LLM continue: plan unchanged", len(result_llm["plan"]) == 4)

from agents.training.core.evaluator import AmendedStep
mock_decision_amend = EvaluatorDecision(
    decision="amend",
    reasoning="Data has too many columns, adding feature engineering.",
    amended_remaining_steps=[
        AmendedStep(step="feature_selection_specification", rationale="PCA needed"),
        AmendedStep(step="feature_engineering_executor", rationale="Execute PCA"),
        AmendedStep(step="select_model", rationale="Choose"),
        AmendedStep(step="training", rationale="Train"),
        AmendedStep(step="generate_report", rationale="Report"),
    ],
)
mock_structured2 = MagicMock()
mock_structured2.invoke.return_value = mock_decision_amend
mock_llm2 = MagicMock()
mock_llm2.with_structured_output.return_value = mock_structured2

eval_amend_state = {**eval_llm_state}
with patch("agents.training.core.evaluator.init_chat_model", return_value=mock_llm2):
    result_amend = evaluator_node(eval_amend_state)
    check("LLM amend: decision maps to 'continue'", result_amend["evaluator_decision"] == "continue")
    check("LLM amend: plan was amended", len(result_amend["plan"]) == 6,
          f"got {len(result_amend['plan'])} steps")

mock_decision_replan = EvaluatorDecision(
    decision="replan",
    reasoning="Metrics are terrible.",
    replan_reason="Training accuracy near random, try different model family",
)
mock_structured3 = MagicMock()
mock_structured3.invoke.return_value = mock_decision_replan
mock_llm3 = MagicMock()
mock_llm3.with_structured_output.return_value = mock_structured3

eval_replan_state = {**eval_llm_state}
with patch("agents.training.core.evaluator.init_chat_model", return_value=mock_llm3):
    result_replan = evaluator_node(eval_replan_state)
    check("LLM replan: decision is 'replan'", result_replan["evaluator_decision"] == "replan")
    check("LLM replan: plan_history grows", len(result_replan["plan_history"]) > 0)


# ─── Graph construction (import-safe) ────────────────────────────────

print("\n=== Graph construction (structure) ===")

from agents.training.core.planner import ALL_STEP_NAMES as PLANNER_STEPS

EXPECTED_NODES = {"planner", "dispatcher", "evaluator"} | PLANNER_STEPS
check("Expected graph has 12 nodes", len(EXPECTED_NODES) == 12, f"got {len(EXPECTED_NODES)}")
check("Expected graph includes planner", "planner" in EXPECTED_NODES)
check("Expected graph includes dispatcher", "dispatcher" in EXPECTED_NODES)
check("Expected graph includes evaluator", "evaluator" in EXPECTED_NODES)
check("Expected graph includes all 9 steps", PLANNER_STEPS.issubset(EXPECTED_NODES))

try:
    from agents.training.core.graph import build_training_agent_graph, ALL_STEP_NAMES as GRAPH_STEPS
    graph = build_training_agent_graph()
    node_names = set(graph.nodes.keys())
    check("Graph has planner node", "planner" in node_names)
    check("Graph has dispatcher node", "dispatcher" in node_names)
    check("Graph has evaluator node", "evaluator" in node_names)
    check("Graph has all 9 step nodes", all(s in node_names for s in GRAPH_STEPS),
          f"missing: {set(GRAPH_STEPS) - node_names}")
    check("Graph has 12 nodes total", len(node_names) == 12, f"got {len(node_names)}")
except ImportError as e:
    print(f"  SKIP: Graph import requires full dependency chain ({e})")


# ─── Planner: full node (mocked LLM + mocked HITL) ──────────────────

print("\n=== Planner: full node (mocked) ===")

from agents.training.core.planner import planner_node

mock_plan = Plan(
    steps=[
        PlanStep(step="data_collection", rationale="Load data"),
        PlanStep(step="select_model", rationale="Choose unsupervised"),
        PlanStep(step="cleaning", rationale="Clean data"),
        PlanStep(step="training_approval", rationale="Approve config"),
        PlanStep(step="training", rationale="Train K-means"),
        PlanStep(step="generate_report", rationale="Produce report"),
    ],
    skipped_steps=["label_split_definition", "feature_selection_specification", "feature_engineering_executor"],
    skip_rationale="Unsupervised task with clean numeric data",
    strategy="Direct clustering approach on customer features",
)

mock_plan_structured = MagicMock()
mock_plan_structured.invoke.return_value = mock_plan
mock_plan_llm = MagicMock()
mock_plan_llm.with_structured_output.return_value = mock_plan_structured

planner_state = create_initial_state("Cluster my customers", linked_datasets=["customers.csv"])

with patch("agents.training.core.planner.init_chat_model", return_value=mock_plan_llm), \
     patch("agents.training.core.planner.run_with_hitl") as mock_hitl:
    mock_hitl.side_effect = lambda name, state, work_fn, get_summary_fn: work_fn(state, None)
    result_plan = planner_node(planner_state)
    check("Planner sets plan", result_plan["plan"] is not None)
    check("Planner plan has 6 steps", len(result_plan["plan"]) == 6)
    check("Planner resets plan_index to 0", result_plan["plan_index"] == 0)
    check("Planner sets strategy", result_plan["plan_strategy"] == "Direct clustering approach on customer features")
    check("Planner adds audit trace", any(
        t.get("step") == "planner" for t in result_plan["audit_trace"]
    ))

# Replan scenario
replan_state = {**result_plan}
replan_state["plan_history"] = [{
    "steps": result_plan["plan"],
    "strategy": result_plan["plan_strategy"],
    "replan_reason": "bad metrics",
}]

with patch("agents.training.core.planner.init_chat_model", return_value=mock_plan_llm), \
     patch("agents.training.core.planner.run_with_hitl") as mock_hitl2:
    mock_hitl2.side_effect = lambda name, state, work_fn, get_summary_fn: work_fn(state, None)
    result_replan = planner_node(replan_state)
    check("Replan preserves history", len(result_replan["plan_history"]) >= 1)


# ─── Integration: dispatcher → evaluator → dispatcher loop ──────────

print("\n=== Integration: dispatcher + evaluator loop ===")

plan_steps = [
    {"step": "data_collection", "rationale": "Load"},
    {"step": "select_model", "rationale": "Choose"},
    {"step": "generate_report", "rationale": "Report"},
]

loop_state = create_initial_state("Test integration")
loop_state["plan"] = plan_steps
loop_state["plan_index"] = 0

d1 = dispatcher_node(loop_state)
check("Loop step 1: dispatches to data_collection", d1["current_step"] == "data_collection")

d1["collected_dataset_ref"] = "test_data"
d1["data_source"] = "local"

mock_continue = EvaluatorDecision(decision="continue", reasoning="OK")
mock_s = MagicMock()
mock_s.invoke.return_value = mock_continue
mock_l = MagicMock()
mock_l.with_structured_output.return_value = mock_s

with patch("agents.training.core.evaluator.init_chat_model", return_value=mock_l):
    e1 = evaluator_node(d1)
check("Loop step 1: evaluator continues", e1["evaluator_decision"] == "continue")
check("Loop step 1: plan_index is now 1", e1["plan_index"] == 1)

d2 = dispatcher_node(e1)
check("Loop step 2: dispatches to select_model", d2["current_step"] == "select_model")

d2["selected_model"] = "unsupervised"
d2["task_type"] = "unsupervised"

with patch("agents.training.core.evaluator.init_chat_model", return_value=mock_l):
    e2 = evaluator_node(d2)
check("Loop step 2: evaluator continues", e2["evaluator_decision"] == "continue")
check("Loop step 2: plan_index is now 2", e2["plan_index"] == 2)

d3 = dispatcher_node(e2)
check("Loop step 3: dispatches to generate_report", d3["current_step"] == "generate_report")

d3["report_path"] = "/tmp/report.json"
e3 = evaluator_node(d3)
check("Loop step 3: evaluator says done", e3["evaluator_decision"] == "done")
check("Loop complete: plan_index is now 3", e3["plan_index"] == 3)

d4 = dispatcher_node(e3)
check("After done: dispatcher routes to 'done'", d4["current_step"] == "done")


# ─── Summary ─────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
if failed:
    print("SOME TESTS FAILED")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
