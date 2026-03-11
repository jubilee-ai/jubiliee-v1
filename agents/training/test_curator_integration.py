"""
Integration tests for the Dataset Curator within agent_simple.py.

Validates that use_external_sources properly gates the curator fallback,
that the curator's output flows correctly through the full pipeline state,
and that the pipeline continues normally after curator-sourced data.
"""

import json
import sys
from pathlib import Path
from unittest import mock

import pandas as pd

_ROOT = Path(__file__).parent.parent.parent

# 1. data-tools and data-retrieval must be first (their utils.py / agent.py
#    must win over root-level and agents/training/ modules with the same name)
_dt = str(_ROOT / "tools" / "data-tools")
_dr = str(_ROOT / "agents" / "data-retrieval")
for p in [_dt, _dr]:
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
import utils as _dt_utils  # noqa: F401 — anchor data-tools utils
import agent as _dr_agent  # noqa: F401 — anchor data-retrieval agent

# 2. Training dir (for core.state, steps.*)
_tr = str(_ROOT / "agents" / "training")
if _tr not in sys.path:
    sys.path.insert(0, _tr)

# 3. agents/ dir for dataset_curator package
_ag = str(_ROOT / "agents")
if _ag not in sys.path:
    sys.path.append(_ag)

# 4. Project root LAST (for agents.training package resolution)
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from core.state import create_initial_state
from utils import get_registered_dataset, register_dataset

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results: list[dict] = []


def report(name: str, passed: bool, details: str = ""):
    tag = PASS if passed else FAIL
    results.append({"name": name, "passed": passed})
    print(f"  [{tag}] {name}")
    if details:
        for line in details.strip().split("\n"):
            print(f"         {line}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: use_external_sources flows through create_simple_training_agent
# ─────────────────────────────────────────────────────────────────────────────
print("\n═══ Test 1: Flag propagation through create_simple_training_agent ═══")

from agents.training.agent_simple import create_simple_training_agent

# We can't actually build the full agent without an LLM call, but we can
# inspect that create_initial_state is called with the right params.
with mock.patch("agents.training.agent_simple.create_initial_state", wraps=create_initial_state) as spy:
    with mock.patch("agents.training.agent_simple.init_chat_model"):
        with mock.patch("agents.training.agent_simple.create_agent"):
            try:
                create_simple_training_agent(
                    goal="test",
                    use_external_sources=True,
                    hitl=False,
                )
            except Exception:
                pass
    if spy.called:
        call_kwargs = spy.call_args
        ext_flag = call_kwargs[0][3] if len(call_kwargs[0]) > 3 else call_kwargs[1].get("use_external_sources")
        passed1 = ext_flag == True
        report("use_external_sources=True reaches create_initial_state", passed1,
               f"call_args={call_kwargs}")
    else:
        report("use_external_sources=True reaches create_initial_state", False,
               "create_initial_state was not called")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: State carries use_external_sources and data_source fields
# ─────────────────────────────────────────────────────────────────────────────
print("\n═══ Test 2: State has new fields ═══")

state_a = create_initial_state("goal", use_external_sources=True)
state_b = create_initial_state("goal", use_external_sources=False)
state_c = create_initial_state("goal")

passed2a = state_a["use_external_sources"] is True and state_a["data_source"] is None
passed2b = state_b["use_external_sources"] is False
passed2c = state_c["use_external_sources"] is False  # default

report("State(ext=True):  use_external_sources=True, data_source=None", passed2a)
report("State(ext=False): use_external_sources=False", passed2b)
report("State(default):   use_external_sources=False (default)", passed2c)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: tool_data_collection passes state to _data_collection_impl
# ─────────────────────────────────────────────────────────────────────────────
print("\n═══ Test 3: tool_data_collection passes full state including ext flag ═══")

from steps.data_collection import data_collection as _dc_impl

captured_states = []

def capturing_dc(state):
    captured_states.append(dict(state))
    # Return a minimal valid result
    register_dataset("mock_ref", pd.DataFrame({"a": range(20), "b": range(20)}))
    return {
        **state,
        "collected_dataset_ref": "mock_ref",
        "data_source": "local",
        "audit_trace": [{
            "step": "data_collection",
            "dataset_ref": "mock_ref",
            "rows": 20,
            "columns": ["a", "b"],
        }],
        "explanations": ["Collected mock data"],
        "error": None,
    }

# Build the agent with mocked internals
with mock.patch("agents.training.agent_simple._data_collection_impl", side_effect=capturing_dc), \
     mock.patch("agents.training.agent_simple.init_chat_model") as mock_llm, \
     mock.patch("agents.training.agent_simple.create_agent") as mock_create:

    mock_create.return_value = mock.MagicMock()

    agent, state_ref = create_simple_training_agent(
        goal="predict churn",
        use_external_sources=True,
        hitl=False,
    )

    # Manually call tool_data_collection (it's in the tools list)
    tools = mock_create.call_args[1].get("tools") or mock_create.call_args[0][1] if mock_create.called else []
    dc_tool = None
    for t in tools:
        if hasattr(t, "__name__") and "data_collection" in t.__name__:
            dc_tool = t
            break

    if dc_tool:
        dc_tool()
        if captured_states:
            cap = captured_states[0]
            passed3 = cap.get("use_external_sources") == True
            report(
                "data_collection receives use_external_sources=True in state",
                passed3,
                f"use_external_sources={cap.get('use_external_sources')}, "
                f"goal={cap.get('goal')}",
            )
            # Check state was updated with data_source
            passed3b = state_ref.get("data_source") == "local"
            report(
                "Agent state updated with data_source after collection",
                passed3b,
                f"state.data_source={state_ref.get('data_source')}, "
                f"state.collected_dataset_ref={state_ref.get('collected_dataset_ref')}",
            )
        else:
            report("data_collection receives state", False, "No state captured")
    else:
        report("data_collection tool found in agent tools", False,
               f"Tools: {[getattr(t, '__name__', str(t)) for t in tools]}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: Curator is NOT called when use_external_sources=False and P2 fails
# ─────────────────────────────────────────────────────────────────────────────
print("\n═══ Test 4: Curator gated by use_external_sources=False ═══")

from steps.data_collection import _try_dataset_curator

state4 = create_initial_state("predict something niche", use_external_sources=False)

with mock.patch("steps.data_collection.retrieve_data") as mock_p2, \
     mock.patch("steps.data_collection._try_dataset_curator") as mock_curator:
    mock_p2.return_value = {"error": "Nothing local"}
    result4 = _dc_impl(state4)

passed4 = (
    not mock_curator.called
    and result4.get("collected_dataset_ref") is None
    and result4.get("error") is not None
)
report(
    "Curator NOT called when use_external_sources=False",
    passed4,
    f"curator_called={mock_curator.called}, error={result4.get('error')}",
)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: Curator IS called when use_external_sources=True and P2 fails
# ─────────────────────────────────────────────────────────────────────────────
print("\n═══ Test 5: Curator triggered when use_external_sources=True and P2 fails ═══")

state5 = create_initial_state("predict rare disease outcomes", use_external_sources=True)

with mock.patch("steps.data_collection.retrieve_data") as mock_p2_5, \
     mock.patch("steps.data_collection._try_dataset_curator") as mock_curator5:
    mock_p2_5.return_value = {"error": "Nothing local"}
    mock_curator5.return_value = {
        "collected_dataset_ref": "curator_rare_disease",
        "data_source": "kaggle",
        "audit_trace": [{"step": "data_collection", "action": "used_dataset_curator"}],
        "explanations": ["Curator found data"],
        "error": None,
    }
    result5 = _dc_impl(state5)

passed5 = (
    mock_curator5.called
    and result5.get("collected_dataset_ref") == "curator_rare_disease"
    and result5.get("data_source") == "kaggle"
    and result5.get("error") is None
)
report(
    "Curator IS called and returns data when ext=True and P2 fails",
    passed5,
    f"curator_called={mock_curator5.called}, ref={result5.get('collected_dataset_ref')}, "
    f"source={result5.get('data_source')}",
)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: Curator NOT called when P2 succeeds (even with ext=True)
# ─────────────────────────────────────────────────────────────────────────────
print("\n═══ Test 6: Curator skipped when P2 succeeds ═══")

from agent import DataRetrievalResult

register_dataset("local_success_ds", pd.DataFrame({
    "x": range(50), "y": range(50, 100), "target": [0, 1] * 25,
}))

state6 = create_initial_state("predict target", use_external_sources=True)

with mock.patch("steps.data_collection.retrieve_data") as mock_p2_6, \
     mock.patch("steps.data_collection._try_dataset_curator") as mock_curator6:
    mock_p2_6.return_value = DataRetrievalResult(
        dataset_ref="local_success_ds",
        description="Found locally",
        rows=50,
        columns=["x", "y", "target"],
        source="catalog",
    )
    result6 = _dc_impl(state6)

passed6 = (
    not mock_curator6.called
    and result6.get("collected_dataset_ref") == "local_success_ds"
    and result6.get("data_source") == "local"
)
report(
    "Curator NOT called when P2 succeeds (even with ext=True)",
    passed6,
    f"curator_called={mock_curator6.called}, ref={result6.get('collected_dataset_ref')}, "
    f"source={result6.get('data_source')}",
)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 7: Curator NOT called when P1 succeeds
# ─────────────────────────────────────────────────────────────────────────────
print("\n═══ Test 7: Curator skipped when P1 (linked dataset) succeeds ═══")

register_dataset("linked_good", pd.DataFrame({
    "f1": range(100), "f2": range(100), "label": [0, 1] * 50,
}))

state7 = create_initial_state(
    "predict label",
    linked_datasets=["linked_good"],
    use_external_sources=True,
)

with mock.patch("steps.data_collection.retrieve_data") as mock_p2_7, \
     mock.patch("steps.data_collection._try_dataset_curator") as mock_curator7:
    result7 = _dc_impl(state7)

passed7 = (
    not mock_p2_7.called
    and not mock_curator7.called
    and result7.get("collected_dataset_ref") == "linked_good"
    and result7.get("data_source") == "pre-registered"
)
report(
    "P1 linked dataset: skips both P2 and curator",
    passed7,
    f"p2_called={mock_p2_7.called}, curator_called={mock_curator7.called}, "
    f"ref={result7.get('collected_dataset_ref')}, source={result7.get('data_source')}",
)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 8: Curator failure is graceful — pipeline gets error, not crash
# ─────────────────────────────────────────────────────────────────────────────
print("\n═══ Test 8: Curator failure doesn't crash the pipeline ═══")

state8 = create_initial_state("predict X", use_external_sources=True)

with mock.patch("steps.data_collection.retrieve_data") as mock_p2_8, \
     mock.patch("steps.data_collection._try_dataset_curator") as mock_curator8:
    mock_p2_8.return_value = {"error": "No local data"}
    mock_curator8.return_value = None  # curator failed
    result8 = _dc_impl(state8)

passed8 = (
    result8.get("collected_dataset_ref") is None
    and result8.get("error") is not None
    and result8.get("data_source") is None
)
report(
    "Curator failure: pipeline gets error, no crash",
    passed8,
    f"ref={result8.get('collected_dataset_ref')}, error={result8.get('error')}, "
    f"source={result8.get('data_source')}",
)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 9: data_source tracking for each path
# ─────────────────────────────────────────────────────────────────────────────
print("\n═══ Test 9: data_source values for each path ═══")

# P1 → "pre-registered"  (tested in T7)
# P2 → "local"           (tested in T6)
# P3 kaggle → "kaggle"   (tested in T5)
# P3 hf → "huggingface"
# P3 both → "kaggle+huggingface"

state9_hf = create_initial_state("classify text", use_external_sources=True)
with mock.patch("steps.data_collection.retrieve_data") as m9p2, \
     mock.patch("steps.data_collection._try_dataset_curator") as m9c:
    m9p2.return_value = {"error": "none"}
    m9c.return_value = {
        "collected_dataset_ref": "curator_hf_text",
        "data_source": "huggingface",
        "audit_trace": [], "explanations": [], "error": None,
    }
    r9_hf = _dc_impl(state9_hf)
passed9_hf = r9_hf.get("data_source") == "huggingface"
report("data_source='huggingface' when curator uses HF", passed9_hf,
       f"source={r9_hf.get('data_source')}")

state9_both = create_initial_state("multi source", use_external_sources=True)
with mock.patch("steps.data_collection.retrieve_data") as m9p2b, \
     mock.patch("steps.data_collection._try_dataset_curator") as m9cb:
    m9p2b.return_value = {"error": "none"}
    m9cb.return_value = {
        "collected_dataset_ref": "curator_both",
        "data_source": "kaggle+huggingface",
        "audit_trace": [], "explanations": [], "error": None,
    }
    r9_both = _dc_impl(state9_both)
passed9_both = r9_both.get("data_source") == "kaggle+huggingface"
report("data_source='kaggle+huggingface' for dual-source", passed9_both,
       f"source={r9_both.get('data_source')}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 10: tool_data_collection propagates curator result through agent state
# ─────────────────────────────────────────────────────────────────────────────
print("\n═══ Test 10: tool_data_collection propagates curator fields to agent state ═══")

def curator_dc(st):
    return {
        **st,
        "collected_dataset_ref": "curator_sentiment",
        "data_source": "huggingface",
        "audit_trace": [{
            "step": "data_collection",
            "action": "used_dataset_curator",
            "dataset_ref": "curator_sentiment",
            "rows": 5000,
            "columns": ["text", "label"],
            "source": "huggingface",
            "hf_sources": ["stanfordnlp/sst2"],
        }],
        "explanations": ["Curator found sentiment data from HuggingFace"],
        "current_step": "data_collection",
        "error": None,
    }

with mock.patch("agents.training.agent_simple._data_collection_impl", side_effect=curator_dc), \
     mock.patch("agents.training.agent_simple.init_chat_model") as mock_llm10, \
     mock.patch("agents.training.agent_simple.create_agent") as mock_ca10:

    mock_ca10.return_value = mock.MagicMock()
    agent10, state10 = create_simple_training_agent(
        goal="classify sentiment",
        use_external_sources=True,
        hitl=False,
    )

    tools10 = mock_ca10.call_args[1].get("tools", [])
    dc_tool10 = next((t for t in tools10 if hasattr(t, "__name__") and "data_collection" in t.__name__), None)

    if dc_tool10:
        output10 = dc_tool10()
        passed10a = state10.get("collected_dataset_ref") == "curator_sentiment"
        passed10b = state10.get("data_source") == "huggingface"
        passed10c = "curator_sentiment" in output10 or "unknown" not in output10
        audit10 = state10.get("audit_trace", [])
        passed10d = any(e.get("action") == "used_dataset_curator" for e in audit10)

        report("Agent state: collected_dataset_ref set", passed10a,
               f"ref={state10.get('collected_dataset_ref')}")
        report("Agent state: data_source='huggingface'", passed10b,
               f"source={state10.get('data_source')}")
        report("Tool output mentions dataset", passed10c,
               f"output={output10[:150]}")
        report("Audit trace has curator action", passed10d,
               f"audit_trace={json.dumps(audit10, default=str)[:300]}")
    else:
        report("tool_data_collection found", False, "Not found in tools list")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 11: invoke_simple_training_agent passes use_external_sources
# ─────────────────────────────────────────────────────────────────────────────
print("\n═══ Test 11: invoke_simple_training_agent passes ext flag ═══")

from agents.training.agent_simple import invoke_simple_training_agent
import inspect

sig = inspect.signature(invoke_simple_training_agent)
passed11 = "use_external_sources" in sig.parameters
report("invoke_simple_training_agent has use_external_sources param", passed11,
       f"params={list(sig.parameters.keys())}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 12: _STEP_OUTPUTS includes data_source for proper invalidation
# ─────────────────────────────────────────────────────────────────────────────
print("\n═══ Test 12: Check downstream invalidation considers data_source ═══")

# data_source should ideally be cleared when data_collection is re-run.
# _STEP_OUTPUTS["data_collection"] currently only lists ["collected_dataset_ref"].
# Let's check and flag if data_source isn't included.
with mock.patch("agents.training.agent_simple._data_collection_impl", side_effect=curator_dc), \
     mock.patch("agents.training.agent_simple.init_chat_model"), \
     mock.patch("agents.training.agent_simple.create_agent") as mock_ca12:

    mock_ca12.return_value = mock.MagicMock()
    _, state12 = create_simple_training_agent(
        goal="test invalidation",
        use_external_sources=True,
        hitl=False,
    )

# Check if _STEP_OUTPUTS includes data_source
# We can't directly access _STEP_OUTPUTS from here, but we can check
# the source code to see if it's listed
from agents.training.agent_simple import create_simple_training_agent as _csa
source = inspect.getsource(_csa)
has_data_source_in_outputs = "data_source" in source and '"data_collection": [' in source
if not has_data_source_in_outputs:
    report(
        "RECOMMENDATION: Add 'data_source' to _STEP_OUTPUTS['data_collection']",
        False,
        "data_source not in _STEP_OUTPUTS — it won't be cleared on re-run.\n"
        "This is a minor issue: on redo, the old data_source persists until overwritten.",
    )
else:
    report("data_source in _STEP_OUTPUTS['data_collection']", True)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
total = len(results)
passed_count = sum(1 for r in results if r["passed"])
failed_count = total - passed_count
print(f"  Results: {passed_count}/{total} passed, {failed_count} failed")
if failed_count:
    print("\n  Issues found:")
    for r in results:
        if not r["passed"]:
            print(f"    ✗ {r['name']}")
print("═" * 70)
