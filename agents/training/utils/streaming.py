"""
Streaming utilities for the ML Training Agent.
Provides streaming functions for real-time progress updates and HITL support.
"""

import logging
import uuid
from typing import Any

from ..core.state import STEP_ORDER

# Configure logger
logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_or_empty(d: dict, key: str) -> dict:
    """Get dict value or empty dict."""
    return d.get(key, {}) or {}


def _humanize_split_strategy(strategy: object) -> str:
    """Short, user-facing copy for train/val/test strategy (not raw enum tokens)."""
    raw = str(strategy).strip().lower() if strategy not in (None, "") else "random"
    if raw == "random":
        return (
            "rows are shuffled, then split into train, validation, and test "
            "(default mix ~70% / 15% / 15%) — appropriate when rows are exchangeable"
        )
    if raw == "time_based":
        return "earlier periods are used for training and later periods for validation and test — preserves time order"
    if raw == "entity_based":
        return "rows are grouped by entity so the same unit never appears in more than one split — reduces leakage"
    return f"custom split strategy ({strategy})"


def _get_audit_entry(node_output: dict, step: str) -> dict:
    """Get the best audit trace row for ``step`` (prefers rows/columns + matching ref).

    When external curator runs after a successful local retrieval, the trace may
    contain multiple ``data_collection`` rows; the last one can be a failure with
    no stats while ``collected_dataset_ref`` still points at the local dataset.
    """
    matches = [t for t in node_output.get("audit_trace", []) if t.get("step") == step]
    if not matches:
        return {}
    ref = node_output.get("collected_dataset_ref")
    if ref:
        for t in reversed(matches):
            if t.get("dataset_ref") == ref and (
                t.get("rows") is not None or t.get("columns")
            ):
                return t
    for t in reversed(matches):
        if t.get("rows") is not None or t.get("columns"):
            return t
    return matches[-1]


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
    if node_name in (
        "feature_selection_specification",
        "feature_engineering_executor",
    ):
        node_name = "feature_specification_and_engineering"
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
            "title": "Model Family Selection Complete",
            "description": f"Selected **{node_output.get('selected_model', 'unknown')}** as the model family for this task.",
            "reasoning": node_output.get("model_explanation", "No explanation provided."),
        }
        _sel = node_output.get("selected_model") or "unknown"
        update["headline"] = f"Selected **{_sel}** — best fit for this task."

    elif node_name == "data_collection":
        audit = _get_audit_entry(node_output, "data_collection")
        cols = audit.get("columns") or []
        rows = audit.get("rows")
        ref = node_output.get("collected_dataset_ref")
        missing_shape = rows in (None, "", "N/A", "?") or not cols
        if missing_shape and ref:
            try:
                from utils import get_registered_dataset

                df = get_registered_dataset(ref)
                if df is not None:
                    rows = len(df)
                    cols = list(df.columns)
            except Exception:
                pass
        if not isinstance(cols, list):
            cols = list(cols) if cols else []

        update["summary"] = {
            "dataset": ref,
            "rows": rows,
            "columns": cols,
            "source": audit.get("source", "collected"),
        }
        ncols = len(cols) if cols else None
        row_label = rows if rows not in (None, "", "N/A", "?") else None
        update["details"] = {
            "title": "Data Collection Complete",
            "description": (
                f"Loaded a dataset with **{row_label or 'unknown'}** rows and **{ncols or 'unknown'}** columns."
            ),
            "stats": {
                "rows": row_label if row_label is not None else "unknown",
                "columns": ncols if ncols is not None else "unknown",
                "column_names": cols,
            },
        }
        if row_label is not None and ncols is not None:
            _rows_fmt = f"{row_label:,}" if isinstance(row_label, int) else str(row_label)
            update["headline"] = f"Loaded **{_rows_fmt} rows** across **{ncols} columns**"
        elif ref:
            short = str(ref).rsplit("/", 1)[-1] or str(ref)
            update["headline"] = f"Loaded dataset **{short}**"
        else:
            update["headline"] = "Dataset ready"

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
        if transforms:
            update["headline"] = f"Cleaned the data — {len(transforms)} transformation(s) applied"
        else:
            update["headline"] = "Cleaned the data — no issues found"

    elif node_name == "label_split_definition":
        ld = _get_or_empty(node_output, "label_definition")
        update["summary"] = {
            "target_column": ld.get("target_column"), "split_strategy": ld.get("split_strategy"), "grain": ld.get("grain"),
            "train_ref": node_output.get("train_dataset_ref"), "val_ref": node_output.get("val_dataset_ref"), "test_ref": node_output.get("test_dataset_ref"),
        }
        col = ld.get("target_column") or "not set"
        split_phrase = _humanize_split_strategy(ld.get("split_strategy"))
        update["details"] = {
            "title": "Label & Split Definition Complete",
            "description": (
                f"The outcome column is **{col}**. {split_phrase[0].upper() + split_phrase[1:] if split_phrase else ''}"
            ),
            "label_definition": {"target": ld.get("target_column"), "strategy": ld.get("split_strategy"), "grain": ld.get("grain"), "forbidden_columns": ld.get("forbidden_columns", [])},
            "datasets": {"train": node_output.get("train_dataset_ref"), "validation": node_output.get("val_dataset_ref"), "test": node_output.get("test_dataset_ref")},
        }
        update["headline"] = f"**Outcome column:** {col} · **Train/val/test:** {split_phrase}"

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
        _leakage = ks.get("leakage_warnings", [])
        _lw = f" — {len(_leakage)} leakage warning(s)" if _leakage else ""
        update["headline"] = f"Selected **{len(features)} features** for modeling{_lw}"

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
        _passed = node_output.get("feature_validation_passed")
        _spec_n = len(spec_features)
        _created_n = len(created)
        update["headline"] = (
            f"Features for training: **{_spec_n}** inputs → **{_created_n}** columns "
            f"({'validation passed' if _passed else 'validation issues'})"
        )

    elif node_name == "feature_specification_and_engineering":
        fs = _get_or_empty(node_output, "feature_spec")
        features = fs.get("features", [])
        trace = node_output.get("analysis_trace", [])
        ks = trace[0].get("key_stats", {}) if trace else {}
        audit = _get_audit_entry(node_output, "feature_engineering_executor")
        shapes = audit.get("shapes", {})
        created = audit.get("features_created", [])
        spec_features = fs.get("features", [])
        exp = _get_or_empty(node_output, "experiment_result")
        rankings = node_output.get("feature_rankings") or {}
        update["summary"] = {
            "num_features": len(features),
            "feature_names": [f.get("name") for f in features[:10]],
            "train_ref": node_output.get("transformed_train_ref"),
            "val_ref": node_output.get("transformed_val_ref"),
            "test_ref": node_output.get("transformed_test_ref"),
            "validation_passed": node_output.get("feature_validation_passed"),
            "features_created": created,
            "shapes": shapes,
            "num_spec_features": len(spec_features),
            "experiment_result": exp if exp else None,
            "feature_rankings": dict(list(rankings.items())[:10]) if rankings else None,
            **{k: ks.get(k, [] if k != "dataset_overview" and k != "target_analysis" and k != "correlation_matrix" else {})
               for k in ["dataset_overview", "target_analysis", "numeric_summaries", "feature_correlations", "correlation_matrix",
                         "high_correlation_pairs", "leakage_warnings", "feature_health", "distribution_stats",
                         "group_summaries", "concentration_analysis", "categorical_summaries", "schema", "summary_text"]},
        }
        exp_details = {}
        if exp:
            exp_details = {
                "best_variant": exp.get("best_variant_name"),
                "best_metric": exp.get("best_metric"),
                "total_variants": exp.get("total_variants"),
                "total_scouts": exp.get("total_scouts"),
                "signal_features": exp.get("signal_features", []),
                "dropped_features": exp.get("dropped_features", []),
                "wall_time_seconds": exp.get("wall_time_seconds"),
                "feature_rankings": dict(list(rankings.items())[:10]) if rankings else {},
            }
        update["details"] = {
            "title": "Features Specified & Engineered",
            "description": f"Specified {len(features)} features; created {len(created)} columns after transforms.",
            "features": [{"name": f.get("name"), "encoding": f.get("encoding"), "formula": str(f.get("formula")) if f.get("formula") else None} for f in features],
            "key_stats": ks, "analysis_trace": trace,
            "features_created": created, "num_spec_features": len(spec_features),
            "dataset_shapes": {"train": _format_shape(shapes, "train"), "validation": _format_shape(shapes, "val"), "test": _format_shape(shapes, "test")},
            "errors": audit.get("errors", []),
            "experiment_grid": exp_details if exp_details else None,
        }
        _passed = node_output.get("feature_validation_passed")
        update["headline"] = (
            f"Selected **{len(features)} features**, built **{len(created)} columns** "
            f"— validation {'passed' if _passed else 'failed'}"
        )

    elif node_name == "evaluate_models":
        comparison = node_output.get("model_comparison", [])
        successful = [r for r in comparison if r.get("success")]
        best = successful[0] if successful else {}
        update["summary"] = {
            "num_models": len(comparison),
            "best_model": best.get("name"),
            "results": comparison,
        }
        update["details"] = {
            "title": "Model Comparison Complete",
            "description": f"Evaluated {len(comparison)} models in parallel.",
            "results": comparison,
        }
        if best:
            metric_parts = []
            for k in ("accuracy", "roc_auc", "r2", "rmse"):
                if best.get(k) is not None:
                    metric_parts.append(f"{k}={best[k]}")
            update["headline"] = f"Best: {best['name']} — {', '.join(metric_parts)}"
        else:
            update["headline"] = "Model comparison complete (no successful models)"

    elif node_name == "feature_experiment_runner":
        exp = node_output.get("experiment_result")
        if not exp:
            update["summary"] = {"skipped": True}
            update["details"] = {
                "title": "Feature experiments",
                "description": "No parallel sweep ran (flow skipped or dataset too small).",
            }
            update["headline"] = "Feature sweep skipped — continuing with your engineered features"
        else:
            bv = exp.get("best_variant_name") or "baseline"
            tv = int(exp.get("total_variants") or 0)
            ts = int(exp.get("total_scouts") or 0)
            bm = exp.get("best_metric")
            update["summary"] = {
                "best_variant_name": bv,
                "best_metric": bm,
                "total_variants": tv,
                "total_scouts": ts,
                "wall_time_seconds": exp.get("wall_time_seconds"),
            }
            grid = node_output.get("experiment_grid_summary") or []
            update["details"] = {
                "title": "Feature experiments",
                "description": (
                    f"Compared **{tv}** feature-set variants using **{ts}** lightweight scout training runs. "
                    f"The pipeline continues with the **{bv}** variant."
                ),
                "experiment_result": exp,
                "experiment_grid_preview": grid[:8] if isinstance(grid, list) else [],
            }
            update["headline"] = (
                f"Feature experiments complete — **{bv}** chosen after **{tv}** feature setups"
            )

    elif node_name == "training_approval":
        tp = _get_or_empty(node_output, "training_plan")
        hp, ds = tp.get("hyperparameters", {}), tp.get("data_summary", {})
        update["summary"] = {
            "model_type": tp.get("model_type"), "task_type": tp.get("task_type"), "hyperparameters": hp,
            "class_weight": tp.get("class_weight"), "max_iterations": tp.get("max_iterations"),
            "strategy_notes": tp.get("strategy_notes"), "expected_metrics": tp.get("expected_metrics"), "data_summary": ds,
        }
        _mtype = tp.get("model_type") or node_output.get("selected_model") or "model"
        _n_hp = len(hp)
        _hp_phrase = (
            "default hyperparameters"
            if _n_hp == 0
            else f"{_n_hp} hyperparameter" + ("s" if _n_hp != 1 else "")
        )
        update["details"] = {
            "title": "Training Configuration Approved",
            "description": (
                f"Training will use **{_mtype}** with approved hyperparameters."
                if _n_hp
                else f"Training will use **{_mtype}** with default hyperparameters."
            ),
            "training_plan": tp, "hyperparameters": hp, "data_summary": ds,
        }
        update["headline"] = f"Ready to train **{_mtype}** with {_hp_phrase}"

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
            "iterations": [
                {
                    "model_name": it.get("model_name"),
                    "tool": it.get("tool"),
                    "success": it.get("success"),
                    "val_accuracy": it.get("val_accuracy"),
                    "val_roc_auc": it.get("val_roc_auc"),
                    "val_r2": it.get("val_r2"),
                    "test_r2": it.get("test_r2"),
                }
                for it in iters
            ],
            "summary": m.get("summary"), "recommendations": m.get("recommendations"),
        }
        _mname = m.get("model_name") or "unknown"
        _t_acc = m.get("test_accuracy")
        _t_auc = m.get("test_roc_auc")
        _t_r2 = m.get("test_r2")
        _t_rmse = m.get("test_rmse")
        if _t_acc is not None or _t_auc is not None:
            _parts = []
            if _t_acc is not None:
                _parts.append(f"**{_t_acc * 100:.1f}% accuracy**")
            if _t_auc is not None:
                _parts.append(f"ROC-AUC {_t_auc:.3f}")
            update["headline"] = f"Trained **{_mname}** — {', '.join(_parts)}"
        elif _t_r2 is not None or _t_rmse is not None:
            _parts = []
            if _t_r2 is not None:
                _parts.append(f"R² {_t_r2:.4f}")
            if _t_rmse is not None:
                _parts.append(f"RMSE {_t_rmse:.0f}")
            update["headline"] = f"Trained **{_mname}** — {', '.join(_parts)}"
        else:
            update["headline"] = f"Trained **{_mname}** successfully"

    elif node_name == "generate_report":
        update["summary"] = {"report_path": node_output.get("report_path"), "model_path": node_output.get("model_weights_path")}
        update["details"] = {
            "title": "Report Generated",
            "description": "Final report generated with detailed metrics, feature importance, and recommendations.",
            "report_path": node_output.get("report_path"), "model_weights_path": node_output.get("model_weights_path"),
            "audit_trace_length": len(node_output.get("audit_trace", [])),
        }
        update["headline"] = "Report saved — view for detailed metrics and recommendations"

    return update
