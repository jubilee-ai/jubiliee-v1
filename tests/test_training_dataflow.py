"""
Critical data-flow test for the simple training agent pipeline.

Verifies dataset lineage through every step:
  raw → collected → cleaned → split → feature-engineered → training

Strategy:
  We patch `create_agent` to intercept the tool closure references that
  `create_simple_training_agent` builds, then call those closures directly
  in pipeline order.  Each step implementation is mocked to be fast and
  trackable — we verify that every mock receives the CORRECT dataset ref
  and that the shared state dict is updated properly between steps.

Run:  python tests/test_training_dataflow.py
"""

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "data-tools"))

from utils import (
    _dataset_registry,
    get_registered_dataset,
    register_dataset,
)

# ================================================================
# Helpers
# ================================================================

RAW_REF = "test_raw_data"
CLEANED_REF = "test_raw_data_cleaned"

_captured_tools: dict = {}


def _make_raw_df(n: int = 120) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.randint(20, 65, n),
        "income": np.random.normal(55000, 15000, n).round(2),
        "score": np.random.randint(300, 850, n),
        "target": np.random.binomial(1, 0.3, n),
    })


def _make_cleaned_df(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["income"] = df["income"].clip(lower=0)
    df["_was_cleaned"] = True
    return df


def _capturing_create_agent(**kwargs):
    """Intercept create_agent to grab tool closures."""
    for fn in kwargs.get("tools", []):
        _captured_tools[fn.__name__] = fn
    return MagicMock(name="compiled_agent")


def _mock_init_chat_model_factory():
    """Return a mock whose .invoke() returns a plausible JSON training plan."""
    model_mock = MagicMock(name="chat_model")
    model_mock.invoke.return_value = MagicMock(
        content=json.dumps({
            "model_type": "logistic_regression",
            "task_type": "classification",
            "hyperparameters": {"C": 1.0},
            "class_weight": "balanced",
            "max_iterations": 3,
            "strategy_notes": "Test strategy",
            "expected_metrics": "0.80 accuracy",
        })
    )
    return model_mock


# ================================================================
# Mock return values
# ================================================================

def _mock_select_model(state):
    return {"selected_model": "logistic_regression", "model_explanation": "Test selection"}


def _mock_data_collection(state):
    return {
        "collected_dataset_ref": RAW_REF,
        "audit_trace": [{"step": "data_collection", "rows": 120, "columns": ["age", "income", "score", "target"]}],
    }


def _mock_cleaning(dataset_ref, goal, max_iterations, target_col, task_type, selected_model):
    raw = get_registered_dataset(dataset_ref)
    cleaned = _make_cleaned_df(raw)
    register_dataset(CLEANED_REF, cleaned, persist=False, register_sql=False)
    return {
        "cleaned_ref": CLEANED_REF,
        "cleaning_summary": "Clipped incomes, added marker.",
        "transformations": [{"op": "clip_income"}, {"op": "add_marker"}],
    }


def _mock_label_split_def(**kwargs):
    return {
        "target_column": "target",
        "prediction_horizon": None,
        "grain": "row",
        "as_of_cutoff": None,
        "split_strategy": "random",
        "forbidden_columns": [],
    }


def _mock_feature_spec(**kwargs):
    return {
        "feature_spec": {
            "features": [
                {"name": "age", "formula": {"op": "passthrough", "column": "age"}},
                {"name": "income", "formula": {"op": "passthrough", "column": "income"}},
                {"name": "score", "formula": {"op": "passthrough", "column": "score"}},
            ],
        },
        "validation": {"valid": True},
        "key_stats": {},
    }


def _mock_feature_exec(train_ref, val_ref, test_ref, feature_spec, target_column, grain, as_of_cutoff=None):
    t_train_ref = f"{train_ref}_features"
    t_val_ref = f"{val_ref}_features" if val_ref else None
    t_test_ref = f"{test_ref}_features" if test_ref else None

    for src_ref, dst_ref in [(train_ref, t_train_ref), (val_ref, t_val_ref), (test_ref, t_test_ref)]:
        if src_ref and dst_ref:
            src_df = get_registered_dataset(src_ref)
            out = src_df[["age", "income", "score", "target"]].copy()
            out["_feature_engineered"] = True
            register_dataset(dst_ref, out, persist=False, register_sql=False)

    train_df = get_registered_dataset(t_train_ref)
    val_df = get_registered_dataset(t_val_ref) if t_val_ref else None
    test_df = get_registered_dataset(t_test_ref) if t_test_ref else None

    return {
        "train_ref": t_train_ref,
        "val_ref": t_val_ref,
        "test_ref": t_test_ref,
        "features_created": ["age", "income", "score"],
        "errors": [],
        "shapes": {
            "train": list(train_df.shape),
            "val": list(val_df.shape) if val_df is not None else None,
            "test": list(test_df.shape) if test_df is not None else None,
        },
    }


def _mock_training(**kwargs):
    return {
        "success": True,
        "model_name": "test_model_123",
        "model_type": "logistic_regression",
        "val_accuracy": 0.82,
        "val_roc_auc": 0.87,
        "test_accuracy": 0.80,
        "test_roc_auc": 0.85,
        "train_r2": None,
        "val_r2": None,
        "val_rmse": None,
        "val_mae": None,
        "test_r2": None,
        "test_rmse": None,
        "test_mae": None,
        "iterations": [{"model_name": "test_model_123", "success": True}],
        "num_iterations": 1,
        "best_iteration": 0,
        "summary": "Test training run",
        "recommendations": [],
        "feature_redo_requested": False,
        "feature_redo_recommendation": None,
        "feature_redo_reason": None,
    }


# ================================================================
# Shared setup: patches + agent creation + tool capture
# ================================================================

class _PipelineHarness:
    """Holds all patches, captured tools, and shared state."""

    def __init__(self):
        self.state = None
        self.mock_cleaning = None
        self.mock_label_def = None
        self.mock_feature_spec = None
        self.mock_feature_exec = None
        self.mock_training = None

    def setup(self):
        global _captured_tools
        _captured_tools.clear()
        _dataset_registry.clear()

        raw_df = _make_raw_df()
        register_dataset(RAW_REF, raw_df, persist=False, register_sql=False)

        self._patches = [
            patch("agents.training.agent_simple.create_agent", side_effect=_capturing_create_agent),
            patch("agents.training.agent_simple._select_model_impl", side_effect=_mock_select_model),
            patch("agents.training.agent_simple._data_collection_impl", side_effect=_mock_data_collection),
            patch("agents.training.agent_simple._infer_target_column", return_value="target"),
            patch("agents.training.agent_simple.run_cleaning_simple", side_effect=_mock_cleaning),
            patch("agents.training.agent_simple.run_label_split_definition", side_effect=_mock_label_split_def),
            patch("agents.training.agent_simple.run_feature_engineering_simple", side_effect=_mock_feature_spec),
            patch("agents.training.agent_simple.execute_feature_spec_split", side_effect=_mock_feature_exec),
            patch("agents.training.agent_simple._run_training", side_effect=_mock_training),
            patch("agents.training.agent_simple.init_chat_model", return_value=_mock_init_chat_model_factory()),
        ]
        self._mocks = [p.start() for p in self._patches]

        (
            self.mock_create_agent,
            self.mock_select_model,
            self.mock_data_collection,
            self.mock_infer_target,
            self.mock_cleaning,
            self.mock_label_def,
            self.mock_feature_spec,
            self.mock_feature_exec,
            self.mock_training,
            self.mock_init_chat,
        ) = self._mocks

        from agents.training.agent_simple import create_simple_training_agent

        _agent, self.state = create_simple_training_agent(
            goal="Predict loan default risk",
            linked_datasets=[RAW_REF],
            user_model_preference="logistic_regression",
            model=MagicMock(name="main_llm"),
            hitl=False,
        )

    def teardown(self):
        for p in self._patches:
            p.stop()
        _captured_tools.clear()
        _dataset_registry.clear()


# ================================================================
# Tests
# ================================================================

def test_full_pipeline_data_lineage():
    """
    Run every tool in pipeline order and verify the complete dataset chain:
      collected → cleaned → split → transformed → training
    """
    h = _PipelineHarness()
    h.setup()
    try:
        tools = _captured_tools
        state = h.state

        assert len(tools) == 9, f"Expected 9 tools, got {len(tools)}: {list(tools.keys())}"

        # --- Step 1: data_collection ---
        result = tools["tool_data_collection"]()
        assert state["collected_dataset_ref"] == RAW_REF, (
            f"After data_collection, collected_dataset_ref should be '{RAW_REF}', "
            f"got '{state['collected_dataset_ref']}'"
        )
        raw_df = get_registered_dataset(RAW_REF)
        assert raw_df is not None, "Raw dataset must be in registry"
        assert "_was_cleaned" not in raw_df.columns, "Raw dataset must NOT have cleaning marker"
        print(f"  [PASS] data_collection -> collected_dataset_ref = '{RAW_REF}'")

        # --- Step 2: select_model ---
        result = tools["tool_select_model"]()
        assert state["selected_model"] == "logistic_regression"
        print(f"  [PASS] select_model -> selected_model = 'logistic_regression'")

        # --- Step 3: cleaning ---
        result = tools["tool_cleaning"]()

        # CRITICAL: verify cleaning was called with collected_dataset_ref
        h.mock_cleaning.assert_called_once()
        clean_call_kwargs = h.mock_cleaning.call_args
        actual_clean_ref = clean_call_kwargs.kwargs.get("dataset_ref", clean_call_kwargs.args[0] if clean_call_kwargs.args else None)
        assert actual_clean_ref == RAW_REF, (
            f"CRITICAL FAILURE: Cleaning received dataset_ref='{actual_clean_ref}', "
            f"expected '{RAW_REF}' (the collected dataset)"
        )

        assert state["cleaned_dataset_ref"] == CLEANED_REF, (
            f"After cleaning, cleaned_dataset_ref should be '{CLEANED_REF}', "
            f"got '{state['cleaned_dataset_ref']}'"
        )
        assert state["cleaned_dataset_ref"] != state["collected_dataset_ref"], (
            "cleaned_dataset_ref must differ from collected_dataset_ref"
        )

        # Verify cleaned data is actually different from raw
        cleaned_df = get_registered_dataset(CLEANED_REF)
        assert cleaned_df is not None, "Cleaned dataset must be in registry"
        assert "_was_cleaned" in cleaned_df.columns, "Cleaned dataset must have the cleaning marker"
        assert "_was_cleaned" not in raw_df.columns, "Raw dataset must NOT have cleaning marker"
        print(f"  [PASS] cleaning -> received '{RAW_REF}', produced '{CLEANED_REF}'")
        print(f"         Cleaned data is distinguishable from raw (has _was_cleaned column)")

        # --- Step 4: label_split_definition ---
        result = tools["tool_label_split_definition"]()

        # CRITICAL: verify label/split operates on the CLEANED dataset, not raw
        h.mock_label_def.assert_called_once()
        label_call_kwargs = h.mock_label_def.call_args.kwargs
        label_dataset_ref = label_call_kwargs.get("dataset_ref")
        assert label_dataset_ref == CLEANED_REF, (
            f"CRITICAL FAILURE: label_split_definition received dataset_ref='{label_dataset_ref}', "
            f"expected '{CLEANED_REF}' (the cleaned dataset, not the raw data)"
        )

        expected_train_ref = f"{CLEANED_REF}_train"
        expected_val_ref = f"{CLEANED_REF}_val"
        expected_test_ref = f"{CLEANED_REF}_test"

        assert state["train_dataset_ref"] == expected_train_ref
        assert state["val_dataset_ref"] == expected_val_ref
        assert state["test_dataset_ref"] == expected_test_ref

        # Verify the splits are derived from CLEANED data (have the marker)
        train_df = get_registered_dataset(expected_train_ref)
        val_df = get_registered_dataset(expected_val_ref)
        test_df = get_registered_dataset(expected_test_ref)
        assert train_df is not None and len(train_df) > 0
        assert val_df is not None and len(val_df) > 0
        assert test_df is not None and len(test_df) > 0
        assert "_was_cleaned" in train_df.columns, (
            "Train split must contain the cleaning marker — it should be derived from the CLEANED dataset"
        )
        assert "_was_cleaned" in val_df.columns
        assert "_was_cleaned" in test_df.columns

        total_split = len(train_df) + len(val_df) + len(test_df)
        assert total_split == len(cleaned_df), (
            f"Split sizes ({total_split}) must sum to cleaned dataset size ({len(cleaned_df)})"
        )
        print(f"  [PASS] label_split -> splits from '{CLEANED_REF}' (train={len(train_df)}, "
              f"val={len(val_df)}, test={len(test_df)})")
        print(f"         All splits contain the _was_cleaned marker")

        # --- Step 5–6: feature_specification_and_engineering (selection + execution) ---
        result = tools["tool_feature_specification_and_engineering"]()

        h.mock_feature_spec.assert_called_once()
        feat_spec_kwargs = h.mock_feature_spec.call_args.kwargs
        feat_spec_train_ref = feat_spec_kwargs.get("train_ref")
        assert feat_spec_train_ref == expected_train_ref, (
            f"Feature spec received train_ref='{feat_spec_train_ref}', "
            f"expected '{expected_train_ref}'"
        )
        assert state["feature_spec"] is not None
        assert len(state["feature_spec"]["features"]) == 3
        print(f"  [PASS] feature_spec -> received train_ref='{expected_train_ref}'")

        h.mock_feature_exec.assert_called_once()
        feat_exec_kwargs = h.mock_feature_exec.call_args.kwargs
        exec_train = feat_exec_kwargs.get("train_ref")
        exec_val = feat_exec_kwargs.get("val_ref")
        exec_test = feat_exec_kwargs.get("test_ref")
        assert exec_train == expected_train_ref, (
            f"Feature executor received train_ref='{exec_train}', expected '{expected_train_ref}'"
        )
        assert exec_val == expected_val_ref
        assert exec_test == expected_test_ref

        expected_t_train = f"{expected_train_ref}_features"
        expected_t_val = f"{expected_val_ref}_features"
        expected_t_test = f"{expected_test_ref}_features"

        assert state["transformed_train_ref"] == expected_t_train
        assert state["transformed_val_ref"] == expected_t_val
        assert state["transformed_test_ref"] == expected_t_test

        # Verify transformed data exists and has the feature engineering marker
        t_train_df = get_registered_dataset(expected_t_train)
        assert t_train_df is not None
        assert "_feature_engineered" in t_train_df.columns, (
            "Transformed train must have feature engineering marker"
        )
        # Note: _was_cleaned is NOT expected here — feature engineering correctly
        # drops non-feature columns.  The lineage guarantee is through the ref chain:
        #   cleaned → split → feature_exec input (verified via mock args above)
        assert "target" in t_train_df.columns, (
            "Transformed train must keep the target column"
        )
        print(f"  [PASS] feature_engineering_executor -> transformed refs:")
        print(f"         train='{expected_t_train}', val='{expected_t_val}', test='{expected_t_test}'")
        print(f"         Transformed data has _feature_engineered marker and target column")

        # --- Step 7: training_approval ---
        result = tools["tool_training_approval"]()
        assert state.get("training_plan") is not None
        assert state.get("training_plan_approved") is True
        print(f"  [PASS] training_approval -> plan approved")

        # --- Step 8: training ---
        result = tools["tool_training"]()

        h.mock_training.assert_called_once()
        train_kwargs = h.mock_training.call_args.kwargs
        training_train_ref = train_kwargs.get("train_ref")
        training_val_ref = train_kwargs.get("val_ref")
        training_test_ref = train_kwargs.get("test_ref")

        assert training_train_ref == expected_t_train, (
            f"CRITICAL FAILURE: Training received train_ref='{training_train_ref}', "
            f"expected '{expected_t_train}' (the TRANSFORMED dataset, not the cleaned or raw one)"
        )
        assert training_val_ref == expected_t_val, (
            f"CRITICAL FAILURE: Training received val_ref='{training_val_ref}', "
            f"expected '{expected_t_val}'"
        )
        assert training_test_ref == expected_t_test, (
            f"CRITICAL FAILURE: Training received test_ref='{training_test_ref}', "
            f"expected '{expected_t_test}'"
        )

        # Verify training did NOT receive the raw or just-cleaned refs
        assert training_train_ref != RAW_REF, "Training must NOT use the raw dataset"
        assert training_train_ref != CLEANED_REF, "Training must NOT use the cleaned dataset directly"
        assert training_train_ref != expected_train_ref, (
            "Training must NOT use the un-transformed train split"
        )
        print(f"  [PASS] training -> received TRANSFORMED refs (not raw, not cleaned, not un-transformed)")
        print(f"         train='{training_train_ref}', val='{training_val_ref}', test='{training_test_ref}'")

        assert state.get("training_metrics", {}).get("success") is True
        print(f"  [PASS] training -> success=True")

        # --- Step 9: generate_report ---
        result = tools["tool_generate_report"]()
        assert state.get("report_path") is not None
        print(f"  [PASS] generate_report -> report_path set")

        print("\n  === FULL PIPELINE DATA LINEAGE VERIFIED ===")
        print(f"  {RAW_REF}")
        print(f"    -> cleaning -> {CLEANED_REF}")
        print(f"    -> split    -> {expected_train_ref} / {expected_val_ref} / {expected_test_ref}")
        print(f"    -> features -> {expected_t_train} / {expected_t_val} / {expected_t_test}")
        print(f"    -> training (used transformed refs)")

    finally:
        h.teardown()


def test_cleaning_receives_collected_not_raw_when_different():
    """
    Edge case: if data_collection produces a DIFFERENT ref than what was
    originally linked, cleaning must use the collected ref, not the linked one.
    """
    h = _PipelineHarness()

    # Override data_collection to return a different ref
    collected_ref = "collected_from_agent"

    def alt_data_collection(state):
        raw = _make_raw_df()
        register_dataset(collected_ref, raw, persist=False, register_sql=False)
        return {
            "collected_dataset_ref": collected_ref,
            "audit_trace": [{"step": "data_collection"}],
        }

    h.setup()
    try:
        # Re-patch data_collection with alternate behavior
        h.mock_data_collection.side_effect = alt_data_collection

        tools = _captured_tools
        state = h.state

        tools["tool_data_collection"]()
        assert state["collected_dataset_ref"] == collected_ref

        tools["tool_select_model"]()
        tools["tool_cleaning"]()

        h.mock_cleaning.assert_called_once()
        actual_ref = h.mock_cleaning.call_args.kwargs.get(
            "dataset_ref", h.mock_cleaning.call_args.args[0] if h.mock_cleaning.call_args.args else None
        )
        assert actual_ref == collected_ref, (
            f"Cleaning should use collected ref '{collected_ref}', not the original linked dataset. "
            f"Got '{actual_ref}'"
        )
        print(f"  [PASS] Cleaning correctly used collected ref '{collected_ref}', not the linked dataset")
    finally:
        h.teardown()


def test_training_never_sees_raw_or_cleaned_data():
    """
    Verify that training ONLY receives transformed (feature-engineered) refs.
    This is the single most critical invariant of the pipeline.
    """
    h = _PipelineHarness()
    h.setup()
    try:
        tools = _captured_tools
        state = h.state

        # Run full pipeline up to training
        tools["tool_data_collection"]()
        tools["tool_select_model"]()
        tools["tool_cleaning"]()
        tools["tool_label_split_definition"]()
        tools["tool_feature_specification_and_engineering"]()
        tools["tool_training_approval"]()
        tools["tool_training"]()

        train_kwargs = h.mock_training.call_args.kwargs

        # Collect all "bad" refs that training should never use
        forbidden_refs = {
            RAW_REF,
            CLEANED_REF,
            state.get("train_dataset_ref"),
            state.get("val_dataset_ref"),
            state.get("test_dataset_ref"),
        }

        used_refs = {
            train_kwargs.get("train_ref"),
            train_kwargs.get("val_ref"),
            train_kwargs.get("test_ref"),
        }

        overlap = forbidden_refs & used_refs
        assert not overlap, (
            f"CRITICAL: Training used forbidden refs: {overlap}. "
            f"Training must ONLY use transformed (feature-engineered) datasets."
        )

        # Verify each ref ends with "_features" (from feature engineering)
        for key in ["train_ref", "val_ref", "test_ref"]:
            ref = train_kwargs.get(key)
            assert ref is not None, f"Training must receive {key}"
            assert ref.endswith("_features"), (
                f"Training {key}='{ref}' does not end with '_features'. "
                f"It should be the feature-engineered version."
            )

        print(f"  [PASS] Training exclusively used feature-engineered refs")
        print(f"         Forbidden refs: {forbidden_refs}")
        print(f"         Used refs: {used_refs}")
    finally:
        h.teardown()


def test_skip_guards_without_prerequisites():
    """
    Verify that steps return SKIP when prerequisites haven't been run.
    This prevents data from a wrong step being used accidentally.
    """
    h = _PipelineHarness()
    h.setup()
    try:
        tools = _captured_tools
        state = h.state

        # cleaning without data_collection
        result = tools["tool_cleaning"]()
        assert "SKIP" in result, f"Cleaning without data_collection should SKIP, got: {result[:100]}"
        assert "cleaned_dataset_ref" not in state or state["cleaned_dataset_ref"] is None
        print(f"  [PASS] cleaning skips when data_collection hasn't run")

        # label_split without cleaning
        result = tools["tool_label_split_definition"]()
        assert "SKIP" in result, f"label_split without cleaning should SKIP, got: {result[:100]}"
        print(f"  [PASS] label_split_definition skips when cleaning hasn't run")

        # feature pipeline without label_split
        result = tools["tool_feature_specification_and_engineering"]()
        assert "SKIP" in result, f"feature pipeline without label_split should SKIP, got: {result[:100]}"
        print(f"  [PASS] feature_specification_and_engineering skips when label_split hasn't run")

        # training without feature_engineering
        result = tools["tool_training"]()
        assert "SKIP" in result, f"training without feature_eng should SKIP, got: {result[:100]}"
        print(f"  [PASS] training skips when feature_engineering hasn't run")

    finally:
        h.teardown()


def test_downstream_invalidation_on_cleaning_rerun():
    """
    If cleaning is re-run, all downstream state (label_split, features,
    transformed refs, training) must be invalidated. Otherwise stale
    data from a previous cleaning would leak through.
    """
    h = _PipelineHarness()
    h.setup()
    try:
        tools = _captured_tools
        state = h.state

        # Run full pipeline
        tools["tool_data_collection"]()
        tools["tool_select_model"]()
        tools["tool_cleaning"]()
        tools["tool_label_split_definition"]()
        tools["tool_feature_specification_and_engineering"]()
        tools["tool_training_approval"]()
        tools["tool_training"]()

        # Snapshot state before re-running cleaning
        old_train_ref = state.get("train_dataset_ref")
        old_transformed_ref = state.get("transformed_train_ref")
        old_feature_spec = state.get("feature_spec")

        assert old_train_ref is not None
        assert old_transformed_ref is not None
        assert old_feature_spec is not None

        # Force a re-run of cleaning (simulate user rejection)
        h.mock_cleaning.reset_mock()
        new_cleaned_ref = "test_raw_data_cleaned_v2"

        def alt_cleaning(dataset_ref, **kwargs):
            raw = get_registered_dataset(dataset_ref)
            cleaned = _make_cleaned_df(raw)
            cleaned["_v2_marker"] = True
            register_dataset(new_cleaned_ref, cleaned, persist=False, register_sql=False)
            return {
                "cleaned_ref": new_cleaned_ref,
                "cleaning_summary": "Re-cleaned v2",
                "transformations": [{"op": "v2_clean"}],
            }

        h.mock_cleaning.side_effect = alt_cleaning

        # Mark cleaning as needing a redo
        state["_redo_feedback_cleaning"] = "Please try a different cleaning approach"

        tools["tool_cleaning"]()

        assert state["cleaned_dataset_ref"] == new_cleaned_ref, (
            f"After re-cleaning, cleaned_dataset_ref should be '{new_cleaned_ref}', "
            f"got '{state['cleaned_dataset_ref']}'"
        )

        # Downstream state should be invalidated
        assert state.get("label_definition") is None, (
            "label_definition should be invalidated after re-cleaning"
        )
        assert state.get("train_dataset_ref") is None, (
            "train_dataset_ref should be invalidated after re-cleaning"
        )
        assert state.get("transformed_train_ref") is None, (
            "transformed_train_ref should be invalidated after re-cleaning"
        )
        assert state.get("feature_spec") is None, (
            "feature_spec should be invalidated after re-cleaning"
        )
        assert state.get("training_metrics") is None, (
            "training_metrics should be invalidated after re-cleaning"
        )

        print(f"  [PASS] Re-running cleaning invalidated all downstream state")
        print(f"         label_definition: {state.get('label_definition')}")
        print(f"         train_dataset_ref: {state.get('train_dataset_ref')}")
        print(f"         transformed_train_ref: {state.get('transformed_train_ref')}")
        print(f"         feature_spec: {state.get('feature_spec')}")

        # Now re-run downstream steps and verify they use the NEW cleaned dataset
        h.mock_label_def.reset_mock()
        h.mock_feature_spec.reset_mock()
        h.mock_feature_exec.reset_mock()
        h.mock_training.reset_mock()

        tools["tool_label_split_definition"]()
        label_ref = h.mock_label_def.call_args.kwargs.get("dataset_ref")
        assert label_ref == new_cleaned_ref, (
            f"After re-cleaning, label_split should use '{new_cleaned_ref}', got '{label_ref}'"
        )

        new_train_ref = state.get("train_dataset_ref")
        assert new_train_ref == f"{new_cleaned_ref}_train", (
            f"Train split ref should be based on new cleaned ref, got '{new_train_ref}'"
        )
        # Verify the new train split has the v2 marker
        new_train_df = get_registered_dataset(new_train_ref)
        assert "_v2_marker" in new_train_df.columns, (
            "New train split must have the v2 cleaning marker"
        )

        tools["tool_feature_specification_and_engineering"]()
        tools["tool_training_approval"]()
        tools["tool_training"]()

        new_training_ref = h.mock_training.call_args.kwargs.get("train_ref")
        assert new_training_ref.startswith(new_cleaned_ref), (
            f"After re-cleaning, training should use refs derived from '{new_cleaned_ref}', "
            f"got '{new_training_ref}'"
        )
        assert new_training_ref != old_transformed_ref, (
            "After re-cleaning, training must NOT use the old transformed ref"
        )

        print(f"  [PASS] After re-cleaning, full downstream pipeline used new cleaned data")
        print(f"         New training ref: {new_training_ref}")
        print(f"         Old training ref: {old_transformed_ref}")

    finally:
        h.teardown()


def test_split_datasets_contain_cleaned_data_not_raw():
    """
    Verify that the actual DataFrame contents in train/val/test splits
    come from the cleaned dataset, not the raw one.
    """
    h = _PipelineHarness()
    h.setup()
    try:
        tools = _captured_tools

        tools["tool_data_collection"]()
        tools["tool_select_model"]()
        tools["tool_cleaning"]()
        tools["tool_label_split_definition"]()

        state = h.state
        train_ref = state["train_dataset_ref"]
        val_ref = state["val_dataset_ref"]
        test_ref = state["test_dataset_ref"]

        for ref_name, ref in [("train", train_ref), ("val", val_ref), ("test", test_ref)]:
            df = get_registered_dataset(ref)
            assert df is not None, f"{ref_name} split should be registered"

            assert "_was_cleaned" in df.columns, (
                f"{ref_name} split is missing '_was_cleaned' column — "
                f"it was NOT derived from the cleaned dataset!"
            )
            assert df["_was_cleaned"].all(), (
                f"{ref_name} split has some rows where _was_cleaned is False"
            )

            # Also verify no negative incomes (cleaning clipped them)
            assert (df["income"] >= 0).all(), (
                f"{ref_name} split has negative incomes — "
                f"the cleaning transformation was not applied, "
                f"suggesting it was split from raw data, not cleaned data!"
            )

        raw_df = get_registered_dataset(RAW_REF)
        has_negative_raw = (raw_df["income"] < 0).any()
        if has_negative_raw:
            print(f"  [INFO] Raw data has {(raw_df['income'] < 0).sum()} negative income values")
            print(f"         Cleaned/split data has zero -> confirmed cleaning was applied")

        print(f"  [PASS] All splits contain cleaned data (have marker, no negative incomes)")

    finally:
        h.teardown()


def test_feature_engineering_operates_on_cleaned_splits():
    """
    Verify that feature engineering operates on the train/val/test splits
    that were derived from the CLEANED dataset.
    """
    h = _PipelineHarness()
    h.setup()
    try:
        tools = _captured_tools
        state = h.state

        tools["tool_data_collection"]()
        tools["tool_select_model"]()
        tools["tool_cleaning"]()
        tools["tool_label_split_definition"]()
        tools["tool_feature_specification_and_engineering"]()

        exec_kwargs = h.mock_feature_exec.call_args.kwargs

        # Feature executor should receive the SPLIT refs (from cleaned data),
        # not the cleaned dataset itself or the raw dataset
        assert exec_kwargs["train_ref"] != CLEANED_REF, (
            "Feature executor should receive the TRAIN SPLIT, not the full cleaned dataset"
        )
        assert exec_kwargs["train_ref"] != RAW_REF, (
            "Feature executor should receive the TRAIN SPLIT, not the raw dataset"
        )
        assert exec_kwargs["train_ref"] == f"{CLEANED_REF}_train", (
            f"Feature executor train_ref should be '{CLEANED_REF}_train', "
            f"got '{exec_kwargs['train_ref']}'"
        )
        assert exec_kwargs["val_ref"] == f"{CLEANED_REF}_val"
        assert exec_kwargs["test_ref"] == f"{CLEANED_REF}_test"

        # Verify the feature spec was passed correctly
        assert exec_kwargs["feature_spec"] is not None
        assert len(exec_kwargs["feature_spec"]["features"]) == 3
        assert exec_kwargs["target_column"] == "target"

        print(f"  [PASS] Feature engineering executor received correct split refs:")
        print(f"         train='{exec_kwargs['train_ref']}'")
        print(f"         val='{exec_kwargs['val_ref']}'")
        print(f"         test='{exec_kwargs['test_ref']}'")

    finally:
        h.teardown()


# ================================================================
# Runner
# ================================================================

def main():
    tests = [
        ("Full Pipeline Data Lineage", test_full_pipeline_data_lineage),
        ("Cleaning Uses Collected Ref (Edge Case)", test_cleaning_receives_collected_not_raw_when_different),
        ("Training Never Sees Raw/Cleaned Data", test_training_never_sees_raw_or_cleaned_data),
        ("Skip Guards Without Prerequisites", test_skip_guards_without_prerequisites),
        ("Downstream Invalidation on Re-clean", test_downstream_invalidation_on_cleaning_rerun),
        ("Split Datasets Contain Cleaned Data", test_split_datasets_contain_cleaned_data_not_raw),
        ("Feature Engineering Uses Cleaned Splits", test_feature_engineering_operates_on_cleaned_splits),
    ]

    print("\n" + "=" * 70)
    print("TRAINING AGENT CRITICAL DATA-FLOW TESTS")
    print("=" * 70)

    results = {}
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            results[name] = True
        except Exception as e:
            results[name] = False
            print(f"  [FAIL] {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n  {passed}/{total} passed")
    if passed == total:
        print("  All critical data-flow checks passed.")
    else:
        print(f"  {total - passed} test(s) FAILED — data flow may be broken!")
    print("=" * 70)
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
