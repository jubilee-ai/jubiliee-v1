"""
Deterministic H2O-3 AutoML training (no LLM iteration).

Used when ``TRAINING_USE_H2O_ONLY`` is true: one ``H2OAutoML`` run with ``max_models=20``,
register the leader, return the same result shape as ``run_training_agent``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from agents.training.core.task_inference import (
    infer_supervised_task_type_from_target_column,
    infer_task_type,
)
from agents.training.mcp.h2o_server import (
    _H2O_JAVA_SETUP_HINT,
    _h2o_available,
    _java_runtime_check,
    register_h2o_mojo_model,
)
from agents.training.steps.training import (
    TrainingIteration,
    _find_best_iteration,
    _iteration_to_dict,
)
from agents.training.utils.graph_stream_hooks import emit_graph_stream

_DATA_TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools" / "data-tools"
_MODEL_TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools" / "models-tools" / "training"
for _p in (_DATA_TOOLS_DIR, _MODEL_TOOLS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils import get_registered_dataset  # noqa: E402


def _perf_to_supervised_metrics(
    perf: Any, task_type: str, prefix: str = "val",
) -> dict[str, Optional[float]]:
    """Map H2O model_performance metrics to val_* or test_* keys used by the pipeline."""
    keys = {
        "rmse": f"{prefix}_rmse",
        "mae": f"{prefix}_mae",
        "r2": f"{prefix}_r2",
        "auc": f"{prefix}_roc_auc",
        "accuracy": f"{prefix}_accuracy",
    }
    out: dict[str, Optional[float]] = {}
    if perf is None:
        return out
    try:
        if task_type == "regression":
            for attr, key in (("rmse", keys["rmse"]), ("mae", keys["mae"]), ("r2", keys["r2"])):
                if hasattr(perf, attr):
                    try:
                        v = getattr(perf, attr)()
                        if v is not None and v == v:  # not NaN
                            out[key] = float(v)
                    except Exception:
                        pass
        else:
            if hasattr(perf, "auc"):
                try:
                    v = perf.auc()
                    if v is not None and v == v:
                        out[keys["auc"]] = float(v)
                except Exception:
                    pass
            if hasattr(perf, "accuracy"):
                try:
                    acc = perf.accuracy()
                    if isinstance(acc, (list, tuple)) and len(acc) > 0:
                        row = acc[0]
                        if isinstance(row, (list, tuple)) and len(row) > 1:
                            out[keys["accuracy"]] = float(row[1])
                        elif hasattr(row, "__float__"):
                            out[keys["accuracy"]] = float(row)
                    elif acc is not None:
                        out[keys["accuracy"]] = float(acc)
                except Exception:
                    pass
    except Exception:
        pass
    return out


def _leaderboard_row_to_val_metrics(row: dict[str, Any], task_type: str) -> dict[str, Optional[float]]:
    """Best-effort mapping from H2O leaderboard column names to val_* fields."""
    out: dict[str, Optional[float]] = {}
    try:
        if task_type == "regression":
            for col, key in (("rmse", "val_rmse"), ("mae", "val_mae"), ("r2", "val_r2")):
                if col in row and row[col] is not None and pd.notna(row[col]):
                    out[key] = float(row[col])
        else:
            if "auc" in row and row["auc"] is not None and pd.notna(row["auc"]):
                out["val_roc_auc"] = float(row["auc"])
            if "mean_per_class_error" in row and row["mean_per_class_error"] is not None:
                mce = float(row["mean_per_class_error"])
                if mce == mce:
                    out["val_accuracy"] = max(0.0, min(1.0, 1.0 - mce))
    except Exception:
        pass
    return out


def _varimp_to_dict(leader: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        vim = leader.varimp(use_pandas=True)
        if vim is None:
            return out
        if hasattr(vim, "iterrows"):
            for _, r in vim.iterrows():
                name = r.get("variable") if hasattr(r, "get") else r["variable"]
                rel = r.get("relative_importance", r.get("scaled_importance", 0.0))
                if name is not None:
                    out[str(name)] = float(rel) if rel is not None else 0.0
        elif isinstance(vim, dict):
            for k, v in vim.items():
                out[str(k)] = float(v)
    except Exception:
        pass
    return out


def run_h2o_training(
    train_ref: str,
    val_ref: Optional[str],
    test_ref: Optional[str],
    target_column: str,
    goal: str,
    selected_model: str,
    model_name: Optional[str] = None,
    max_models: int = 20,
    max_runtime_secs: int = 0,
    nfolds: int = 3,
    seed: int = 1,
    explicit_task_type: Optional[str] = None,
    max_iterations: int = 4,
    max_continuation_rounds: int = 3,
    estimator_hint: Optional[str] = None,
    experiment_result: Optional[dict[str, Any]] = None,
    feature_rankings: Optional[dict[str, float]] = None,
    training_plan: Optional[dict[str, Any]] = None,
    prior_training_metrics: Optional[dict[str, Any]] = None,
    llm_model: str = "openai:gpt-5.4",
    **kwargs: Any,
) -> dict[str, Any]:
    """Run H2O AutoML once and return the same dict shape as ``run_training_agent``."""
    _ = kwargs
    print(
        f"[h2o_training] enter train_ref={train_ref!r} val_ref={val_ref!r} test_ref={test_ref!r} "
        f"target={target_column!r} selected_model={selected_model!r} max_models={max_models} "
        f"nfolds={nfolds} explicit_task_type={explicit_task_type!r}",
        flush=True,
    )

    train_df = get_registered_dataset(train_ref)
    val_df = get_registered_dataset(val_ref) if val_ref else None
    test_df = get_registered_dataset(test_ref) if test_ref else None
    print(
        f"[h2o_training] data_loaded "
        f"train_rows={(len(train_df) if train_df is not None else None)} "
        f"val_rows={(len(val_df) if val_df is not None else None)} "
        f"test_rows={(len(test_df) if test_df is not None else None)}",
        flush=True,
    )

    if train_df is None:
        print(f"[h2o_training] abort reason=train_df_not_found ref={train_ref!r}", flush=True)
        return {
            "success": False,
            "error": f"Training dataset not found: {train_ref}",
            "model_name": model_name or "",
            "model_type": selected_model,
        }
    for ref, df, label in [(val_ref, val_df, "Validation"), (test_ref, test_df, "Test")]:
        if ref and df is None:
            return {
                "success": False,
                "error": f"{label} dataset not found: {ref}",
                "model_name": model_name or "",
                "model_type": selected_model,
            }

    plan_tt: Optional[str] = None
    tp = training_plan
    if isinstance(tp, dict):
        raw_tt = tp.get("task_type")
        if isinstance(raw_tt, str) and raw_tt:
            plan_tt = raw_tt
    data_tt: Optional[str] = None
    if target_column and target_column in train_df.columns:
        data_tt = infer_supervised_task_type_from_target_column(train_df, target_column)
    base_tt = explicit_task_type or plan_tt or infer_task_type(goal, selected_model=selected_model)
    task_type = base_tt
    if data_tt and base_tt != data_tt:
        if {data_tt, base_tt} == {"classification", "regression"}:
            task_type = data_tt

    print(f"[h2o_training] task_type_resolved task_type={task_type!r} plan_tt={plan_tt!r} data_tt={data_tt!r} base_tt={base_tt!r}", flush=True)

    if task_type == "unsupervised":
        print("[h2o_training] delegating_unsupervised_to_run_training_agent", flush=True)
        from agents.training.steps.training import run_training_agent

        return run_training_agent(
            train_ref=train_ref,
            val_ref=val_ref,
            test_ref=test_ref,
            target_column=target_column,
            selected_model=selected_model,
            goal=goal,
            model_name=model_name,
            explicit_task_type=explicit_task_type,
            training_plan=tp if isinstance(tp, dict) else None,
            prior_training_metrics=prior_training_metrics,
            experiment_result=experiment_result,
            feature_rankings=feature_rankings,
            max_iterations=max_iterations,
            llm_model=llm_model,
            estimator_hint=estimator_hint,
            max_continuation_rounds=max_continuation_rounds,
        )

    if not target_column:
        return {
            "success": False,
            "error": "target_column is required for H2O AutoML training",
            "model_name": model_name or "",
            "model_type": selected_model,
        }

    if not model_name:
        model_name = f"{selected_model}_{int(time.time())}"

    feature_columns = [c for c in train_df.columns if c != target_column]

    emit_graph_stream({"type": "progress", "message": "Running H2O AutoML (deterministic)...", "phase": "training"})
    print(f"[h2o_training] AutoML max_models={max_models} | task={task_type} | target={target_column}")

    ok_pkg, pkg_err = _h2o_available()
    print(f"[h2o_training] h2o_available ok={ok_pkg} err={pkg_err!r}", flush=True)
    if not ok_pkg:
        print(f"[h2o_training] abort reason=h2o_package_unavailable err={pkg_err!r}", flush=True)
        return {
            "success": False,
            "error": pkg_err,
            "model_name": model_name,
            "model_type": "H2OAutoML",
            "task_type": task_type,
            "target_column": target_column,
            "fix": _H2O_JAVA_SETUP_HINT if "java" in pkg_err.lower() else None,
        }

    j_ok, _java_path, j_err = _java_runtime_check()
    print(f"[h2o_training] java_check ok={j_ok} path={_java_path!r} err={j_err!r}", flush=True)
    if not j_ok:
        print(f"[h2o_training] abort reason=java_missing err={j_err!r}", flush=True)
        return {
            "success": False,
            "error": j_err,
            "fix": _H2O_JAVA_SETUP_HINT,
            "model_name": model_name,
            "model_type": "H2OAutoML",
            "task_type": task_type,
            "target_column": target_column,
        }

    import h2o
    from h2o.automl import H2OAutoML

    h2o_started = False
    try:
        print("[h2o_training] h2o.init start", flush=True)
        h2o.init(max_mem_size="4G", nthreads=-1, strict_version_check=False)
        h2o_started = True
        print("[h2o_training] h2o.init done", flush=True)

        print("[h2o_training] importing pandas DataFrames into H2OFrames", flush=True)
        train_hf = h2o.H2OFrame(train_df)
        val_hf = h2o.H2OFrame(val_df) if val_df is not None else None
        test_hf = h2o.H2OFrame(test_df) if test_df is not None else None
        try:
            _train_dim = (getattr(train_hf, "nrow", None), getattr(train_hf, "ncol", None))
        except Exception:
            _train_dim = (None, None)
        print(
            f"[h2o_training] frames_ready train_dim={_train_dim} "
            f"val={(val_hf is not None)} test={(test_hf is not None)}",
            flush=True,
        )

        y = target_column
        if task_type != "regression":
            train_hf[y] = train_hf[y].asfactor()
            if val_hf is not None:
                val_hf[y] = val_hf[y].asfactor()
            if test_hf is not None:
                test_hf[y] = test_hf[y].asfactor()

        x = [c for c in train_hf.columns if c != y]

        aml_kwargs: dict[str, Any] = {
            "max_models": max_models,
            "seed": seed,
            "nfolds": nfolds,
            "sort_metric": "AUTO",
        }
        if max_runtime_secs and max_runtime_secs > 0:
            aml_kwargs["max_runtime_secs"] = max_runtime_secs

        print(f"[h2o_training] building H2OAutoML kwargs={aml_kwargs}", flush=True)
        aml = H2OAutoML(**aml_kwargs)
        train_kw: dict[str, Any] = {"x": x, "y": y, "training_frame": train_hf}
        if val_hf is not None:
            train_kw["validation_frame"] = val_hf
        print(f"[h2o_training] aml.train start x_len={len(x)} y={y!r}", flush=True)
        aml.train(**train_kw)
        print("[h2o_training] aml.train done", flush=True)

        leader = aml.leader
        leader_id = leader.model_id
        algo = getattr(leader, "algo", None) or "H2OAutoML"
        print(f"[h2o_training] leader leader_id={leader_id!r} algo={algo!r}", flush=True)

        val_metrics: dict[str, Optional[float]] = {}
        if val_hf is not None:
            val_metrics = _perf_to_supervised_metrics(
                leader.model_performance(val_hf), task_type, prefix="val"
            )
        if not val_metrics:
            val_metrics = _perf_to_supervised_metrics(
                leader.model_performance(), task_type, prefix="val"
            )

        test_metrics: dict[str, Optional[float]] = {}
        if test_hf is not None:
            test_metrics = _perf_to_supervised_metrics(
                leader.model_performance(test_hf), task_type, prefix="test"
            )

        lb_df = aml.leaderboard.as_data_frame()
        lb_rows = lb_df.head(20).to_dict(orient="records")

        iterations: list[TrainingIteration] = []
        for row in lb_rows:
            mid = str(row.get("model_id", ""))
            if not mid:
                continue
            vm = _leaderboard_row_to_val_metrics(row, task_type)
            if mid == leader_id:
                vm = {**vm, **{k: v for k, v in val_metrics.items() if v is not None}}
            iterations.append(
                TrainingIteration(
                    model_name=mid,
                    tool_used="H2OAutoML",
                    hyperparams={"model_id": mid},
                    success=True,
                    val_accuracy=vm.get("val_accuracy"),
                    val_roc_auc=vm.get("val_roc_auc"),
                    val_r2=vm.get("val_r2"),
                    val_rmse=vm.get("val_rmse"),
                    val_mae=vm.get("val_mae"),
                )
            )

        iterations_dict = [_iteration_to_dict(it) for it in iterations]
        best_iteration = _find_best_iteration(iterations_dict, task_type)

        feat_imp = _varimp_to_dict(leader)
        merged_metrics = {**{k: v for k, v in val_metrics.items() if v is not None}}
        reg = register_h2o_mojo_model(
            leader_id,
            model_name,
            target_column,
            feature_columns,
            metrics=merged_metrics,
        )
        if not reg.get("ok"):
            return {
                "success": False,
                "error": reg.get("error", "register_h2o_mojo_model failed"),
                "model_name": model_name,
                "model_type": str(algo),
                "task_type": task_type,
                "target_column": target_column,
                "train_size": len(train_df),
                "val_size": len(val_df) if val_df is not None else 0,
                "test_size": len(test_df) if test_df is not None else 0,
                "iterations": iterations_dict,
                "num_iterations": len(iterations_dict),
                "best_iteration": best_iteration,
                "summary": f"H2O AutoML ran but registration failed: {reg.get('error')}",
                "feature_importances": feat_imp,
                "feature_redo_requested": False,
                "messages": [],
            }

        summary = (
            f"H2O AutoML trained up to {max_models} models and selected **{algo}** as the leader. "
            f"Primary validation metrics: "
        )
        if task_type == "regression":
            summary += f"R²={merged_metrics.get('val_r2')}, RMSE={merged_metrics.get('val_rmse')}."
        else:
            summary += f"ROC-AUC={merged_metrics.get('val_roc_auc')}, accuracy={merged_metrics.get('val_accuracy')}."

        out: dict[str, Any] = {
            "success": True,
            "model_name": model_name,
            "model_type": str(algo),
            "task_type": task_type,
            "target_column": target_column,
            "train_size": len(train_df),
            "val_size": len(val_df) if val_df is not None else 0,
            "test_size": len(test_df) if test_df is not None else 0,
            "iterations": iterations_dict,
            "num_iterations": len(iterations_dict),
            "best_iteration": best_iteration,
            "summary": summary,
            "feature_importances": feat_imp,
            "feature_redo_requested": False,
            "feature_redo_recommendation": None,
            "feature_redo_reason": None,
            "messages": [],
        }
        for k, v in merged_metrics.items():
            if v is not None:
                out[k] = v
        for k, v in test_metrics.items():
            if v is not None:
                out[k] = v

        print(f"[h2o_training] Leader={leader_id} registered as {model_name}")
        return out

    except Exception as e:
        import traceback as _tb
        msg = str(e)
        print(f"[h2o_training] exception type={type(e).__name__} error={msg[:300]!r}", flush=True)
        _tb.print_exc()
        err_out: dict[str, Any] = {
            "success": False,
            "error": msg,
            "model_name": model_name or "",
            "model_type": "H2OAutoML",
            "task_type": task_type,
            "target_column": target_column,
        }
        if any(s in msg.lower() for s in ("java", "jvm", "jdk")):
            err_out["fix"] = _H2O_JAVA_SETUP_HINT
        return err_out
    finally:
        if h2o_started:
            try:
                print("[h2o_training] shutting down H2O cluster", flush=True)
                h2o.cluster().shutdown(prompt=False)
            except Exception as _se:
                print(f"[h2o_training] shutdown_error {type(_se).__name__}: {_se}", flush=True)
