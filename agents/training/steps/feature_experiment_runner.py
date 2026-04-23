"""
Feature Experiment Runner — parallel scout grid over feature-set variants.

Sits between feature_engineering_executor and training_approval in the
pipeline.  Generates multiple feature-set variants from the base spec,
runs lightweight models (HistGradientBoosting, RandomForest) in parallel
for each variant, ranks features by cross-variant importance, and selects
the best variant for the full LLM-driven training step.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))

from utils import get_registered_dataset, register_dataset

from ..utils.graph_stream_hooks import emit_graph_stream


def _feature_experiment_config() -> dict[str, Any]:
    """Load caps from app settings when available (worker or API process)."""
    try:
        from backend.shared.settings import get_settings

        s = get_settings()
        return {
            "enabled": s.FEATURE_EXPERIMENT_ENABLED,
            "max_workers": s.FEATURE_SCOUT_MAX_WORKERS,
            "max_variants": s.FEATURE_SCOUT_MAX_VARIANTS,
            "rf_n_jobs": s.FEATURE_SCOUT_RF_N_JOBS,
            "skip_above_rows": s.FEATURE_EXPERIMENT_SKIP_ABOVE_ROWS,
        }
    except Exception:
        return {
            "enabled": os.getenv("FEATURE_EXPERIMENT_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            "max_workers": max(1, int(os.getenv("FEATURE_SCOUT_MAX_WORKERS", "2"))),
            "max_variants": max(1, int(os.getenv("FEATURE_SCOUT_MAX_VARIANTS", "4"))),
            "rf_n_jobs": max(1, int(os.getenv("FEATURE_SCOUT_RF_N_JOBS", "1"))),
            "skip_above_rows": int(os.getenv("FEATURE_EXPERIMENT_SKIP_ABOVE_ROWS", "200000")),
        }


def _scout_rf_n_jobs() -> int:
    return int(_feature_experiment_config()["rf_n_jobs"])
from .feature_engineering import _extract_columns_from_formula
from .feature_engineering_executor import execute_feature_spec_split


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FeatureVariant:
    name: str
    feature_spec: dict
    description: str


@dataclass
class ScoutResult:
    variant_name: str
    model_family: str
    metrics: dict[str, float]
    feature_importances: dict[str, float]
    success: bool
    error: Optional[str] = None


@dataclass
class ExperimentResult:
    best_variant_name: str
    best_metric: float
    best_feature_spec: dict
    transformed_train_ref: str
    transformed_val_ref: Optional[str]
    transformed_test_ref: Optional[str]
    experiment_grid: list[dict]
    feature_rankings: dict[str, float]
    dropped_features: list[str]
    signal_features: list[str]
    total_variants: int
    total_scouts: int
    wall_time_seconds: float


# ---------------------------------------------------------------------------
# Variant generation
# ---------------------------------------------------------------------------

def _compute_mutual_info(
    X: pd.DataFrame, y: pd.Series, task_type: str,
) -> dict[str, float]:
    """Mutual information between each feature and the target."""
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    X_numeric = X.select_dtypes(include="number").copy()
    if X_numeric.empty:
        return {}

    X_numeric = X_numeric.fillna(0)

    mi_func = mutual_info_regression if task_type == "regression" else mutual_info_classif
    mi_scores = mi_func(X_numeric, y, random_state=42)
    return dict(sorted(
        zip(X_numeric.columns, mi_scores), key=lambda kv: kv[1], reverse=True,
    ))


def _find_correlated_pairs(
    X: pd.DataFrame, threshold: float = 0.9,
) -> list[tuple[str, str, float]]:
    """Return pairs of features with |correlation| > threshold."""
    X_numeric = X.select_dtypes(include="number")
    if X_numeric.shape[1] < 2:
        return []
    corr = X_numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = []
    for col in upper.columns:
        for idx in upper.index:
            val = upper.at[idx, col]
            if pd.notna(val) and val > threshold:
                pairs.append((idx, col, round(float(val), 4)))
    return pairs


def _group_features_by_op(feature_spec: dict) -> dict[str, list[dict]]:
    """Group features in the spec by their formula op."""
    groups: dict[str, list[dict]] = {}
    for feat in feature_spec.get("features", []):
        op = feat.get("formula", {}).get("op", "unknown")
        groups.setdefault(op, []).append(feat)
    return groups


def generate_feature_variants(
    feature_spec: dict,
    train_df: pd.DataFrame,
    target_column: str,
    task_type: str,
) -> list[FeatureVariant]:
    """Produce a list of feature-set variants from the base spec."""
    features = feature_spec.get("features", [])
    if len(features) < 3:
        return [FeatureVariant("full", feature_spec, "All features from base spec")]

    variants: list[FeatureVariant] = []

    # 1. Full baseline
    variants.append(FeatureVariant("full", feature_spec, "All features from base spec"))

    feature_cols = [c for c in train_df.columns if c != target_column]
    X = train_df[feature_cols]
    y = train_df[target_column]

    # 2. MI top-K (keep ~70% of features by mutual information)
    if task_type != "unsupervised" and len(features) >= 5:
        mi_scores = _compute_mutual_info(X, y, task_type)
        if mi_scores:
            k = max(3, int(len(features) * 0.7))
            top_names = set(list(mi_scores.keys())[:k])
            mi_features = [
                f for f in features
                if (
                    f.get("name") in top_names
                    or _extract_columns_from_formula(f.get("formula", {})) & top_names
                )
            ]
            if len(mi_features) >= 3 and len(mi_features) < len(features):
                variants.append(FeatureVariant(
                    "mi_top_k",
                    {"features": mi_features},
                    f"Top {k} features by mutual information with target",
                ))

    # 3. Decorrelated — drop one from each highly-correlated pair
    corr_pairs = _find_correlated_pairs(X, threshold=0.9)
    if corr_pairs:
        to_drop: set[str] = set()
        for f1, f2, _ in corr_pairs:
            if f1 not in to_drop and f2 not in to_drop:
                to_drop.add(f2)
        if to_drop:
            decorr_features = [
                f for f in features if f.get("name") not in to_drop
            ]
            if len(decorr_features) >= 3 and len(decorr_features) < len(features):
                variants.append(FeatureVariant(
                    "decorrelated",
                    {"features": decorr_features},
                    f"Removed {len(to_drop)} features with |corr| > 0.9",
                ))

    # 4. Ablation by operation group (only for specs with 2+ distinct ops)
    op_groups = _group_features_by_op(feature_spec)
    if len(op_groups) >= 2:
        for op, group_feats in op_groups.items():
            if len(group_feats) >= 2 and len(group_feats) < len(features) - 2:
                drop_names = {f.get("name") for f in group_feats}
                remaining = [f for f in features if f.get("name") not in drop_names]
                if len(remaining) >= 3:
                    variants.append(FeatureVariant(
                        f"ablation_no_{op}",
                        {"features": remaining},
                        f"Ablation: dropped all '{op}' features ({len(group_feats)})",
                    ))

    # Cap total variants (configurable; default lower than historical 6 for memory)
    cfg = _feature_experiment_config()
    max_v = max(1, int(cfg.get("max_variants", 4)))
    if len(variants) > max_v:
        variants = variants[:max_v]

    return variants


# ---------------------------------------------------------------------------
# Scout model training (runs in worker processes)
# ---------------------------------------------------------------------------

def _train_scout(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    task_type: str,
    model_family: str,
) -> dict[str, Any]:
    """Train one scout model and return metrics + feature importances.

    Designed to run inside a process pool — all arguments are plain
    data, no shared state.
    """
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.metrics import (
        accuracy_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

    X_train = X_train.copy()
    X_val = X_val.copy()

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[cat_cols] = oe.fit_transform(X_train[cat_cols])
        X_val[cat_cols] = oe.transform(X_val[cat_cols])

    le = None
    y_tr = y_train.copy()
    y_va = y_val.copy()
    if y_tr.dtype == object or y_tr.dtype.name == "category":
        le = LabelEncoder()
        y_tr = le.fit_transform(y_tr)
        y_va = le.transform(y_va)

    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)

    rf_nj = _scout_rf_n_jobs()
    models = {
        ("classification", "hgb"): HistGradientBoostingClassifier(max_iter=200, max_depth=6, random_state=42),
        ("classification", "rf"): RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=rf_nj
        ),
        ("regression", "hgb"): HistGradientBoostingRegressor(max_iter=200, max_depth=6, random_state=42),
        ("regression", "rf"): RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=rf_nj
        ),
    }

    model = models.get((task_type, model_family))
    if model is None:
        return {"success": False, "error": f"No scout model for {task_type}/{model_family}"}

    model.fit(X_train, y_tr)
    y_pred = model.predict(X_val)

    metrics: dict[str, float] = {}
    if task_type == "regression":
        metrics["r2"] = round(float(r2_score(y_va, y_pred)), 4)
        metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_va, y_pred))), 4)
        metrics["mae"] = round(float(mean_absolute_error(y_va, y_pred)), 4)
    else:
        y_va_arr = np.asarray(y_va)
        y_pred_arr = np.asarray(y_pred)
        metrics["accuracy"] = round(float(accuracy_score(y_va_arr, y_pred_arr)), 4)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val)
            try:
                if y_proba.shape[1] == 2:
                    metrics["roc_auc"] = round(float(roc_auc_score(y_va_arr, y_proba[:, 1])), 4)
                else:
                    metrics["roc_auc"] = round(float(
                        roc_auc_score(y_va_arr, y_proba, multi_class="ovr", average="weighted"),
                    ), 4)
            except (ValueError, TypeError):
                pass

    importances: dict[str, float] = {}
    if hasattr(model, "feature_importances_"):
        raw = model.feature_importances_
        if len(raw) == len(X_train.columns):
            importances = {
                col: round(float(v), 4)
                for col, v in sorted(
                    zip(X_train.columns, raw), key=lambda kv: kv[1], reverse=True,
                )
            }

    return {"success": True, "metrics": metrics, "feature_importances": importances}


def _scout_worker(args: tuple) -> ScoutResult:
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    (
        variant_name, model_family, task_type,
        X_tr_arr, y_tr_arr, X_va_arr, y_va_arr,
        feature_cols,
    ) = args
    try:
        X_train = pd.DataFrame(np.asarray(X_tr_arr), columns=feature_cols)
        y_train = pd.Series(np.asarray(y_tr_arr), name="target")
        X_val = pd.DataFrame(np.asarray(X_va_arr), columns=feature_cols)
        y_val = pd.Series(np.asarray(y_va_arr), name="target")

        result = _train_scout(X_train, y_train, X_val, y_val, task_type, model_family)
        return ScoutResult(
            variant_name=variant_name,
            model_family=model_family,
            metrics=result.get("metrics", {}),
            feature_importances=result.get("feature_importances", {}),
            success=result.get("success", False),
            error=result.get("error"),
        )
    except Exception as exc:
        return ScoutResult(
            variant_name=variant_name,
            model_family=model_family,
            metrics={},
            feature_importances={},
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Primary metric helpers
# ---------------------------------------------------------------------------

def _primary_metric_key(task_type: str) -> str:
    if task_type == "regression":
        return "r2"
    return "roc_auc"


def _primary_metric_value(metrics: dict, task_type: str) -> float:
    key = _primary_metric_key(task_type)
    val = metrics.get(key)
    if val is not None:
        return float(val)
    if task_type == "classification":
        return float(metrics.get("accuracy", -float("inf")))
    return -float("inf")


# ---------------------------------------------------------------------------
# Cross-variant analysis
# ---------------------------------------------------------------------------

def _aggregate_feature_rankings(
    results: list[ScoutResult], task_type: str,
) -> dict[str, float]:
    """Weighted average of feature importances across all successful scouts.

    Weights are the scout's primary metric (higher-performing scouts
    contribute more to the ranking).
    """
    weighted_sums: dict[str, float] = {}
    weight_total: dict[str, float] = {}

    for r in results:
        if not r.success or not r.feature_importances:
            continue
        w = max(_primary_metric_value(r.metrics, task_type), 0.0)
        for feat, imp in r.feature_importances.items():
            weighted_sums[feat] = weighted_sums.get(feat, 0.0) + imp * w
            weight_total[feat] = weight_total.get(feat, 0.0) + w

    rankings: dict[str, float] = {}
    for feat in weighted_sums:
        denom = weight_total[feat]
        rankings[feat] = round(weighted_sums[feat] / denom, 4) if denom > 0 else 0.0

    return dict(sorted(rankings.items(), key=lambda kv: kv[1], reverse=True))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_experiment_grid(
    feature_spec: dict,
    train_ref: str,
    val_ref: Optional[str],
    test_ref: Optional[str],
    target_column: str,
    task_type: str,
    grain: str = "",
    as_of_cutoff: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> ExperimentResult:
    """Run the parallel feature-experiment grid and return the best variant."""
    t0 = time.time()
    cfg = _feature_experiment_config()

    emit_graph_stream({"type": "progress", "message": "Starting feature experiment grid...", "phase": "feature_experiment_runner"})

    train_df = get_registered_dataset(train_ref)
    val_df = get_registered_dataset(val_ref) if val_ref else None

    if train_df is None:
        raise ValueError(f"Training dataset not found: {train_ref}")
    if val_df is None:
        raise ValueError(f"Validation dataset not found: {val_ref}")

    skip_above = int(cfg.get("skip_above_rows") or 0)
    if not cfg.get("enabled", True):
        print("[experiment_runner] Feature experiment grid disabled by config — skipping scouts")
        return ExperimentResult(
            best_variant_name="full",
            best_metric=0.0,
            best_feature_spec=feature_spec,
            transformed_train_ref=f"{train_ref}_features",
            transformed_val_ref=f"{val_ref}_features" if val_ref else None,
            transformed_test_ref=f"{test_ref}_features" if test_ref else None,
            experiment_grid=[],
            feature_rankings={},
            dropped_features=[],
            signal_features=[],
            total_variants=1,
            total_scouts=0,
            wall_time_seconds=round(time.time() - t0, 2),
        )

    try:
        from backend.shared.settings import get_settings as _gs_exp

        if bool(_gs_exp().TRAINING_USE_H2O_ONLY):
            print(
                "[experiment_runner] TRAINING_USE_H2O_ONLY — skipping sklearn scout grid "
                "(HGB/RF); main training uses H2O only."
            )
            return ExperimentResult(
                best_variant_name="full",
                best_metric=0.0,
                best_feature_spec=feature_spec,
                transformed_train_ref=f"{train_ref}_features",
                transformed_val_ref=f"{val_ref}_features" if val_ref else None,
                transformed_test_ref=f"{test_ref}_features" if test_ref else None,
                experiment_grid=[{"skipped": True, "reason": "TRAINING_USE_H2O_ONLY"}],
                feature_rankings={},
                dropped_features=[],
                signal_features=[],
                total_variants=1,
                total_scouts=0,
                wall_time_seconds=round(time.time() - t0, 2),
            )
    except Exception:
        pass

    if skip_above > 0 and len(train_df) > skip_above:
        print(
            f"[experiment_runner] Train rows {len(train_df)} > skip threshold {skip_above} — skipping scout grid"
        )
        return ExperimentResult(
            best_variant_name="full",
            best_metric=0.0,
            best_feature_spec=feature_spec,
            transformed_train_ref=f"{train_ref}_features",
            transformed_val_ref=f"{val_ref}_features" if val_ref else None,
            transformed_test_ref=f"{test_ref}_features" if test_ref else None,
            experiment_grid=[],
            feature_rankings={},
            dropped_features=[],
            signal_features=[],
            total_variants=1,
            total_scouts=0,
            wall_time_seconds=round(time.time() - t0, 2),
        )

    MAX_SCOUT_ROWS = 10_000
    if len(train_df) > MAX_SCOUT_ROWS:
        print(f"[experiment_runner] Sampling train from {len(train_df)} to {MAX_SCOUT_ROWS} rows for scouts")
        train_df = train_df.sample(n=MAX_SCOUT_ROWS, random_state=42)
    if val_df is not None and len(val_df) > MAX_SCOUT_ROWS:
        print(f"[experiment_runner] Sampling val from {len(val_df)} to {MAX_SCOUT_ROWS} rows for scouts")
        val_df = val_df.sample(n=MAX_SCOUT_ROWS, random_state=42)

    print(f"[experiment_runner] Generating feature variants...")
    variants = generate_feature_variants(feature_spec, train_df, target_column, task_type)
    print(f"[experiment_runner] Generated {len(variants)} variants: "
          f"{[v.name for v in variants]}")

    if len(variants) <= 1:
        print("[experiment_runner] Only 1 variant (full) — skipping scout grid")
        return ExperimentResult(
            best_variant_name="full",
            best_metric=0.0,
            best_feature_spec=feature_spec,
            transformed_train_ref=f"{train_ref}_features",
            transformed_val_ref=f"{val_ref}_features" if val_ref else None,
            transformed_test_ref=f"{test_ref}_features" if test_ref else None,
            experiment_grid=[],
            feature_rankings={},
            dropped_features=[],
            signal_features=[],
            total_variants=1,
            total_scouts=0,
            wall_time_seconds=round(time.time() - t0, 2),
        )

    # Execute each variant's feature spec to get transformed datasets
    variant_data: dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]] = {}
    variant_feature_cols: dict[str, list[str]] = {}

    for i, v in enumerate(variants):
        emit_graph_stream({"type": "progress", "message": f"Testing variant {i + 1}/{len(variants)}...", "phase": "feature_experiment_runner"})
        try:
            result = execute_feature_spec_split(
                train_ref=train_ref,
                val_ref=val_ref,
                test_ref=test_ref,
                feature_spec=v.feature_spec,
                target_column=target_column,
                grain=grain,
                as_of_cutoff=as_of_cutoff,
            )
            v_train = get_registered_dataset(result["train_ref"])
            v_val = get_registered_dataset(result["val_ref"]) if result.get("val_ref") else None
            if v_train is not None and v_val is not None:
                if target_column not in v_train.columns or target_column not in v_val.columns:
                    print(
                        f"[experiment_runner] Variant '{v.name}' missing target "
                        f"column '{target_column}', skipping"
                    )
                    continue
                fcols = [c for c in v_train.columns if c != target_column]
                X_tr = v_train[fcols].reset_index(drop=True)
                y_tr = v_train[target_column].reset_index(drop=True)
                X_va = v_val[fcols].reset_index(drop=True)
                y_va = v_val[target_column].reset_index(drop=True)
                variant_data[v.name] = (X_tr, y_tr, X_va, y_va)
                variant_feature_cols[v.name] = fcols
            else:
                print(f"[experiment_runner] Variant '{v.name}' produced empty data, skipping")
        except Exception as exc:
            print(f"[experiment_runner] Variant '{v.name}' feature engineering failed: {exc}")

    if not variant_data:
        print("[experiment_runner] All variants failed feature engineering — falling back to full")
        return ExperimentResult(
            best_variant_name="full",
            best_metric=0.0,
            best_feature_spec=feature_spec,
            transformed_train_ref=f"{train_ref}_features",
            transformed_val_ref=f"{val_ref}_features" if val_ref else None,
            transformed_test_ref=f"{test_ref}_features" if test_ref else None,
            experiment_grid=[],
            feature_rankings={},
            dropped_features=[],
            signal_features=[],
            total_variants=1,
            total_scouts=0,
            wall_time_seconds=round(time.time() - t0, 2),
        )

    # Build scout jobs: each variant x each model family
    model_families = ["hgb", "rf"]
    jobs: list[tuple] = []

    for v_name, (X_tr, y_tr, X_va, y_va) in variant_data.items():
        fcols = variant_feature_cols[v_name]
        for mf in model_families:
            jobs.append((
                v_name, mf, task_type,
                np.ascontiguousarray(X_tr.to_numpy(copy=True)),
                np.asarray(y_tr.to_numpy(copy=True)),
                np.ascontiguousarray(X_va.to_numpy(copy=True)),
                np.asarray(y_va.to_numpy(copy=True)),
                fcols,
            ))

    cap_w = int(cfg.get("max_workers") or 2)
    n_workers = max_workers or min(len(jobs), cap_w, os.cpu_count() or 1)
    n_workers = max(1, n_workers)
    print(f"[experiment_runner] Running {len(jobs)} scout jobs with {n_workers} workers...")

    TOTAL_TIMEOUT = 300
    HEARTBEAT_INTERVAL = 15
    POLL_INTERVAL = 5
    scout_start = time.time()
    last_heartbeat = scout_start
    scout_results: list[ScoutResult] = []
    completed_count = 0
    total_jobs = len(jobs)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        pending = {pool.submit(_scout_worker, job): job for job in jobs}
        while pending:
            elapsed = time.time() - scout_start
            if elapsed > TOTAL_TIMEOUT:
                print(f"[experiment_runner] Total timeout ({TOTAL_TIMEOUT}s) reached, "
                      f"cancelling remaining scouts")
                for f in pending:
                    f.cancel()
                break

            done_batch = {f for f in pending if f.done()}
            if not done_batch:
                now = time.time()
                if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                    emit_graph_stream({
                        "type": "progress",
                        "message": f"Running scouts ({completed_count}/{total_jobs}, {int(elapsed)}s elapsed)…",
                        "phase": "feature_experiment_runner",
                    })
                    last_heartbeat = now
                time.sleep(POLL_INTERVAL)
                continue

            for future in done_batch:
                job_info = pending.pop(future)
                completed_count += 1
                try:
                    sr = future.result(timeout=0)
                    scout_results.append(sr)
                    status = "OK" if sr.success else "FAIL"
                    metric_val = _primary_metric_value(sr.metrics, task_type)
                    print(f"  [{status}] {sr.variant_name} + {sr.model_family}: "
                          f"{_primary_metric_key(task_type)}={metric_val:.4f}")
                except Exception as exc:
                    print(f"  [FAIL] {job_info[0]} + {job_info[1]}: {exc}")

            emit_graph_stream({
                "type": "progress",
                "message": f"Scout {completed_count}/{total_jobs} done…",
                "phase": "feature_experiment_runner",
            })
            last_heartbeat = time.time()

    emit_graph_stream({"type": "progress", "message": "Analyzing experiment results...", "phase": "feature_experiment_runner"})

    # Aggregate and analyze
    successful = [r for r in scout_results if r.success]
    if not successful:
        print("[experiment_runner] All scouts failed — falling back to full variant")
        return ExperimentResult(
            best_variant_name="full",
            best_metric=0.0,
            best_feature_spec=feature_spec,
            transformed_train_ref=f"{train_ref}_features",
            transformed_val_ref=f"{val_ref}_features" if val_ref else None,
            transformed_test_ref=f"{test_ref}_features" if test_ref else None,
            experiment_grid=[{"error": "all scouts failed"}],
            feature_rankings={},
            dropped_features=[],
            signal_features=[],
            total_variants=len(variants),
            total_scouts=len(jobs),
            wall_time_seconds=round(time.time() - t0, 2),
        )

    # Find best variant (by best primary metric across model families)
    best_per_variant: dict[str, float] = {}
    for r in successful:
        val = _primary_metric_value(r.metrics, task_type)
        if r.variant_name not in best_per_variant or val > best_per_variant[r.variant_name]:
            best_per_variant[r.variant_name] = val

    best_variant_name = max(best_per_variant, key=best_per_variant.get)  # type: ignore[arg-type]
    best_metric = best_per_variant[best_variant_name]

    best_spec = feature_spec
    for v in variants:
        if v.name == best_variant_name:
            best_spec = v.feature_spec
            break

    # Compute global feature rankings
    feature_rankings = _aggregate_feature_rankings(successful, task_type)

    signal_features = [f for f, imp in feature_rankings.items() if imp >= 0.02]
    dropped_features = [f for f, imp in feature_rankings.items() if imp < 0.005]

    # Re-execute feature engineering for the winning variant to produce
    # properly registered train/val/test refs
    win_suffix = f"_exp_{best_variant_name}"
    win_train_ref = f"{train_ref}{win_suffix}"
    win_val_ref = f"{val_ref}{win_suffix}" if val_ref else None
    win_test_ref = f"{test_ref}{win_suffix}" if test_ref else None

    try:
        final_result = execute_feature_spec_split(
            train_ref=train_ref,
            val_ref=val_ref,
            test_ref=test_ref,
            feature_spec=best_spec,
            target_column=target_column,
            grain=grain,
            as_of_cutoff=as_of_cutoff,
        )
        win_train_ref = final_result["train_ref"]
        win_val_ref = final_result.get("val_ref")
        win_test_ref = final_result.get("test_ref")
    except Exception as exc:
        print(f"[experiment_runner] Final feature engineering failed, using base: {exc}")
        win_train_ref = f"{train_ref}_features"
        win_val_ref = f"{val_ref}_features" if val_ref else None
        win_test_ref = f"{test_ref}_features" if test_ref else None

    # Build grid summary
    experiment_grid = []
    for r in scout_results:
        experiment_grid.append({
            "variant": r.variant_name,
            "model_family": r.model_family,
            "success": r.success,
            "metrics": r.metrics,
            "error": r.error,
        })
    experiment_grid.sort(
        key=lambda x: _primary_metric_value(x.get("metrics", {}), task_type),
        reverse=True,
    )

    wall_time = round(time.time() - t0, 2)
    print(f"\n[experiment_runner] Done in {wall_time}s")
    print(f"  Best variant: {best_variant_name} "
          f"({_primary_metric_key(task_type)}={best_metric:.4f})")
    print(f"  Signal features ({len(signal_features)}): "
          f"{signal_features[:10]}")
    print(f"  Noise features ({len(dropped_features)}): "
          f"{dropped_features[:10]}")

    return ExperimentResult(
        best_variant_name=best_variant_name,
        best_metric=best_metric,
        best_feature_spec=best_spec,
        transformed_train_ref=win_train_ref,
        transformed_val_ref=win_val_ref,
        transformed_test_ref=win_test_ref,
        experiment_grid=experiment_grid,
        feature_rankings=feature_rankings,
        dropped_features=dropped_features,
        signal_features=signal_features,
        total_variants=len(variants),
        total_scouts=len(jobs),
        wall_time_seconds=wall_time,
    )


__all__ = [
    "run_experiment_grid",
    "generate_feature_variants",
    "ExperimentResult",
    "FeatureVariant",
    "ScoutResult",
]
