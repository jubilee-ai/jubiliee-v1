"""
Streaming utilities for the ML Training Agent.
Provides streaming functions for real-time progress updates and HITL support.
"""

import logging
import traceback
import uuid
from typing import Any, Generator

from langgraph.types import Command

from ..core.state import STEP_ORDER, TrainingAgentState, create_initial_state


def _get_training_agent():
    from ..core.graph import create_training_agent
    return create_training_agent()

# Configure logger
logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_or_empty(d: dict, key: str) -> dict:
    """Get dict value or empty dict."""
    return d.get(key, {}) or {}


def _get_audit_entry(node_output: dict, step: str) -> dict:
    """Get audit trace entry for a specific step."""
    return next((t for t in node_output.get("audit_trace", []) if t.get("step") == step), {})


def _format_shape(shapes: dict, key: str) -> str:
    """Format shape tuple as 'rows × cols' string."""
    shape = shapes.get(key)
    return f"{shape[0]} rows × {shape[1]} cols" if shape else "unknown"


def _parse_summary_field(summary: str, field: str) -> str:
    """Extract a field value from cleaning summary text."""
    for line in summary.split("\n"):
        if f"{field}:" in line:
            return line.split(f"{field}:")[1].strip().split()[0]
    return "unknown"


def _gen_thread_id() -> str:
    """Generate a new thread ID."""
    return f"training-{uuid.uuid4().hex[:8]}"


# =============================================================================
# SHARED HELPER FUNCTIONS
# =============================================================================


def calculate_progress(node_name: str) -> int:
    """Calculate progress percentage for a node (0-100)."""
    try:
        return int(((STEP_ORDER.index(node_name) + 1) / len(STEP_ORDER)) * 100)
    except ValueError:
        return 50


def extract_interrupt_info(interrupt_data: list) -> dict[str, Any]:
    """Parse interrupt data into structured dict."""
    default = {"node": "unknown", "summary": "Step completed", "message": "Awaiting human review", "state_snapshot": {}}
    
    if not interrupt_data:
        return default

    obj = interrupt_data[0]
    val = obj.value if hasattr(obj, "value") else obj

    if isinstance(val, dict):
        return {
            "node": val.get("node", "unknown"),
            "summary": val.get("summary", ""),
            "message": val.get("message", "Awaiting approval"),
            "state_snapshot": val.get("state_snapshot", {}),
        }
    return {**default, "summary": str(val)}


def build_node_update(node_name: str, node_output: dict[str, Any]) -> dict[str, Any]:
    """Build summary/details for a node output."""
    update = {"type": "node_complete", "node": node_name, "progress": calculate_progress(node_name), "state": node_output}

    if node_name == "select_model":
        update["summary"] = {"selected_model": node_output.get("selected_model"), "explanation": node_output.get("model_explanation")}
        update["details"] = {
            "title": "Model Selection Complete",
            "description": f"Selected **{node_output.get('selected_model', 'unknown')}** as the optimal model for this task.",
            "reasoning": node_output.get("model_explanation", "No explanation provided."),
        }

    elif node_name == "data_collection":
        audit = _get_audit_entry(node_output, "data_collection")
        cols = audit.get("columns", [])
        update["summary"] = {"dataset": node_output.get("collected_dataset_ref"), "rows": audit.get("rows"), "columns": cols, "source": audit.get("source", "collected")}
        update["details"] = {
            "title": "Data Collection Complete",
            "description": f"Loaded dataset: **{node_output.get('collected_dataset_ref', 'unknown')}**",
            "stats": {"rows": audit.get("rows", "unknown"), "columns": len(cols) if cols else "unknown", "column_names": cols},
        }

    elif node_name in ("cleaning", "cleaning_and_standardization"):
        transforms = node_output.get("cleaning_transformations", [])
        summary_text = node_output.get("cleaning_summary") or ""
        rows = _parse_summary_field(summary_text, "Rows") if summary_text else "unknown"
        cols = _parse_summary_field(summary_text, "Columns") if summary_text else "unknown"
        reason = ""
        for line in summary_text.split("\n"):
            if "Reason:" in line:
                reason = line.split("Reason:")[1].strip()
                break

        update["summary"] = {
            "cleaned_dataset": node_output.get("cleaned_dataset_ref"), "num_transformations": len(transforms),
            "transformations": transforms[:10], "cleaning_message": summary_text, "rows": rows, "columns": cols, "reason": reason,
        }
        update["details"] = {
            "title": "Data Cleaning Complete",
            "description": reason or f"Applied {len(transforms)} transformations to clean the data.",
            "output_dataset": node_output.get("cleaned_dataset_ref"), "all_transformations": transforms,
            "cleaning_summary": summary_text,
            "transformations_applied": [t if isinstance(t, dict) else {"op": str(t)} for t in transforms],
        }

    elif node_name == "label_split_definition":
        ld = _get_or_empty(node_output, "label_definition")
        update["summary"] = {
            "target_column": ld.get("target_column"), "split_strategy": ld.get("split_strategy"), "grain": ld.get("grain"),
            "train_ref": node_output.get("train_dataset_ref"), "val_ref": node_output.get("val_dataset_ref"), "test_ref": node_output.get("test_dataset_ref"),
        }
        update["details"] = {
            "title": "Label & Split Definition Complete",
            "description": f"Target column: **{ld.get('target_column', 'unknown')}** with {ld.get('split_strategy', 'random')} split.",
            "label_definition": {"target": ld.get("target_column"), "strategy": ld.get("split_strategy"), "grain": ld.get("grain"), "forbidden_columns": ld.get("forbidden_columns", [])},
            "datasets": {"train": node_output.get("train_dataset_ref"), "validation": node_output.get("val_dataset_ref"), "test": node_output.get("test_dataset_ref")},
        }

    elif node_name == "feature_selection_specification":
        fs = _get_or_empty(node_output, "feature_spec")
        features = fs.get("features", [])
        trace = node_output.get("analysis_trace", [])
        ks = trace[0].get("key_stats", {}) if trace else {}

        # Copy all key_stats fields to summary
        update["summary"] = {
            "num_features": len(features), "feature_names": [f.get("name") for f in features[:10]],
            **{k: ks.get(k, [] if k != "dataset_overview" and k != "target_analysis" and k != "correlation_matrix" else {})
               for k in ["dataset_overview", "target_analysis", "numeric_summaries", "feature_correlations", "correlation_matrix",
                         "high_correlation_pairs", "leakage_warnings", "feature_health", "distribution_stats",
                         "group_summaries", "concentration_analysis", "categorical_summaries", "schema", "summary_text"]},
        }
        update["details"] = {
            "title": "Feature Selection Complete",
            "description": f"Specified {len(features)} features for the model.",
            "features": [{"name": f.get("name"), "encoding": f.get("encoding"), "formula": str(f.get("formula")) if f.get("formula") else None} for f in features],
            "key_stats": ks, "analysis_trace": trace,
        }

    elif node_name == "feature_engineering_executor":
        audit = _get_audit_entry(node_output, "feature_engineering_executor")
        shapes = audit.get("shapes", {})
        created = audit.get("features_created", [])
        fs = _get_or_empty(node_output, "feature_spec")
        spec_features = fs.get("features", [])
        update["summary"] = {
            "train_ref": node_output.get("transformed_train_ref"), "val_ref": node_output.get("transformed_val_ref"),
            "test_ref": node_output.get("transformed_test_ref"), "validation_passed": node_output.get("feature_validation_passed"),
            "features_created": created, "shapes": shapes,
            "num_spec_features": len(spec_features),
        }
        description = f"Created {len(created)} features"
        if len(created) != len(spec_features):
            description += f" from {len(spec_features)} specifications (one-hot encoding expands categorical features)"
        description += "."
        update["details"] = {
            "title": "Feature Engineering Complete", "description": description,
            "features_created": created, "num_spec_features": len(spec_features),
            "dataset_shapes": {"train": _format_shape(shapes, "train"), "validation": _format_shape(shapes, "val"), "test": _format_shape(shapes, "test")},
            "errors": audit.get("errors", []),
        }

    elif node_name == "training_approval":
        tp = _get_or_empty(node_output, "training_plan")
        hp, ds = tp.get("hyperparameters", {}), tp.get("data_summary", {})
        update["summary"] = {
            "model_type": tp.get("model_type"), "task_type": tp.get("task_type"), "hyperparameters": hp,
            "class_weight": tp.get("class_weight"), "max_iterations": tp.get("max_iterations"),
            "strategy_notes": tp.get("strategy_notes"), "expected_metrics": tp.get("expected_metrics"), "data_summary": ds,
        }
        update["details"] = {
            "title": "Training Configuration Approved",
            "description": f"Training will use **{tp.get('model_type', 'unknown')}** with approved hyperparameters.",
            "training_plan": tp, "hyperparameters": hp, "data_summary": ds,
        }

    elif node_name == "training":
        m = _get_or_empty(node_output, "training_metrics")
        iters = m.get("iterations", [])
        update["summary"] = {
            "success": m.get("success"), "model_name": m.get("model_name"), "model_type": m.get("model_type"), "num_iterations": m.get("num_iterations"),
            "val_accuracy": m.get("val_accuracy"), "val_roc_auc": m.get("val_roc_auc"), "test_accuracy": m.get("test_accuracy"), "test_roc_auc": m.get("test_roc_auc"),
            "val_r2": m.get("val_r2"), "val_rmse": m.get("val_rmse"), "test_r2": m.get("test_r2"), "test_rmse": m.get("test_rmse"), "test_mae": m.get("test_mae"),
        }
        update["details"] = {
            "title": "Training Complete",
            "description": f"Trained **{m.get('model_name', 'model')}** successfully." if m.get("success") else "Training completed with issues.",
            "model": {"name": m.get("model_name"), "type": m.get("model_type"), "path": node_output.get("model_weights_path")},
            "metrics": {
                "validation": {"accuracy": m.get("val_accuracy"), "roc_auc": m.get("val_roc_auc"), "r2": m.get("val_r2"), "rmse": m.get("val_rmse"), "mae": m.get("val_mae")},
                "test": {"accuracy": m.get("test_accuracy"), "roc_auc": m.get("test_roc_auc"), "r2": m.get("test_r2"), "rmse": m.get("test_rmse"), "mae": m.get("test_mae")},
            },
            "iterations": [{"model_name": it.get("model_name"), "tool": it.get("tool"), "success": it.get("success"), "val_r2": it.get("val_r2"), "test_r2": it.get("test_r2")} for it in iters],
            "summary": m.get("summary"), "recommendations": m.get("recommendations"),
        }

    elif node_name == "generate_report":
        update["summary"] = {"report_path": node_output.get("report_path"), "model_path": node_output.get("model_weights_path")}
        update["details"] = {
            "title": "Report Generated",
            "description": f"Final report saved to: **{node_output.get('report_path', 'unknown')}**",
            "report_path": node_output.get("report_path"), "model_weights_path": node_output.get("model_weights_path"),
            "audit_trace_length": len(node_output.get("audit_trace", [])),
        }

    return update


# =============================================================================
# CORE STREAMING HELPERS
# =============================================================================


def _stream_values(agent, input_data, config: dict, thread_id: str) -> Generator[dict[str, Any], None, TrainingAgentState]:
    """Core streaming logic with 'values' mode."""
    final_state = None
    try:
        for event in agent.stream(input_data, config=config, stream_mode="values"):
            if "__interrupt__" in event:
                yield {"type": "interrupt", "interrupt": event["__interrupt__"], "thread_id": thread_id}
                return event
            yield {"type": "state_update", "node": event.get("current_step", "unknown"), "state": event, "thread_id": thread_id}
            final_state = event
    except Exception as e:
        error_msg = f"Error in streaming (values mode): {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        print(f"[STREAMING ERROR] {error_msg}")
        print(traceback.format_exc())
        yield {"type": "error", "error": str(e), "traceback": traceback.format_exc(), "thread_id": thread_id}
    return final_state


def _stream_updates(agent, input_data, config: dict, thread_id: str, include_thread_id: bool = False) -> Generator[dict[str, Any], None, None]:
    """Core streaming logic with 'updates' mode."""
    try:
        for event in agent.stream(input_data, config=config, stream_mode="updates"):
            if "__interrupt__" in event:
                yield {"type": "interrupt", "thread_id": thread_id, **extract_interrupt_info(event["__interrupt__"])}
                return
            for node_name, node_output in event.items():
                update = build_node_update(node_name, node_output)
                if include_thread_id:
                    update["thread_id"] = thread_id
                yield update
    except Exception as e:
        error_msg = f"Error in streaming (updates mode): {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        print(f"[STREAMING ERROR] {error_msg}")
        print(traceback.format_exc())
        yield {"type": "error", "error": str(e), "traceback": traceback.format_exc(), "thread_id": thread_id}


# =============================================================================
# STREAMING FUNCTIONS
# =============================================================================


def stream_training_agent(
    goal: str,
    linked_datasets: list[str] | None = None,
    user_model_preference: str | None = None,
    thread_id: str | None = None,
) -> Generator[dict[str, Any], None, TrainingAgentState]:
    """Stream the training agent with the given inputs (values mode)."""
    thread_id = thread_id or _gen_thread_id()
    try:
        yield from _stream_values(
            _get_training_agent(),
            create_initial_state(goal, linked_datasets, user_model_preference),
            {"configurable": {"thread_id": thread_id}},
            thread_id,
        )
    except Exception as e:
        logger.error(f"Error starting training agent: {e}")
        logger.error(traceback.format_exc())
        print(f"[STREAMING ERROR] Error starting training agent: {e}")
        print(traceback.format_exc())
        yield {"type": "error", "error": str(e), "traceback": traceback.format_exc(), "thread_id": thread_id}


def stream_resume_training_agent(
    decision: Any,
    thread_id: str,
) -> Generator[dict[str, Any], None, TrainingAgentState]:
    """Resume streaming the training agent after an interrupt (values mode)."""
    try:
        yield from _stream_values(
            _get_training_agent(),
            Command(resume=decision),
            {"configurable": {"thread_id": thread_id}},
            thread_id,
        )
    except Exception as e:
        logger.error(f"Error resuming training agent: {e}")
        logger.error(traceback.format_exc())
        print(f"[STREAMING ERROR] Error resuming training agent: {e}")
        print(traceback.format_exc())
        yield {"type": "error", "error": str(e), "traceback": traceback.format_exc(), "thread_id": thread_id}


def stream_training_agent_with_updates(
    goal: str,
    linked_datasets: list[str] | None = None,
    user_model_preference: str | None = None,
    thread_id: str | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Stream the training agent and yield structured updates (updates mode)."""
    thread_id = thread_id or _gen_thread_id()
    
    yield {"type": "started", "node": "init", "progress": 0, "message": "Training agent started", "thread_id": thread_id}
    
    try:
        yield from _stream_updates(
            _get_training_agent(),
            create_initial_state(goal, linked_datasets, user_model_preference),
            {"configurable": {"thread_id": thread_id}},
            thread_id,
        )
        yield {"type": "completed", "node": "end", "progress": 100, "message": "Training completed"}
    except Exception as e:
        logger.error(f"Error in training agent with updates: {e}")
        logger.error(traceback.format_exc())
        print(f"[STREAMING ERROR] Error in training agent with updates: {e}")
        print(traceback.format_exc())
        yield {"type": "error", "error": str(e), "traceback": traceback.format_exc(), "thread_id": thread_id}


def stream_resume_training_agent_with_updates(
    decision: Any,
    thread_id: str,
) -> Generator[dict[str, Any], None, None]:
    """Resume streaming the training agent after an interrupt (updates mode)."""
    try:
        yield from _stream_updates(
            _get_training_agent(),
            Command(resume=decision),
            {"configurable": {"thread_id": thread_id}},
            thread_id,
            include_thread_id=True,
        )
        yield {"type": "completed", "node": "end", "progress": 100, "message": "Training completed", "thread_id": thread_id}
    except Exception as e:
        logger.error(f"Error resuming training agent with updates: {e}")
        logger.error(traceback.format_exc())
        print(f"[STREAMING ERROR] Error resuming training agent with updates: {e}")
        print(traceback.format_exc())
        yield {"type": "error", "error": str(e), "traceback": traceback.format_exc(), "thread_id": thread_id}
