"""
H2O-3 tools for the training agent (best-effort; requires Java + h2o package).

Each tool is a small operation the agent composes — no fixed AutoML-only flow.
"""

from __future__ import annotations

import json
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import tool

# In-process handles (same Python worker as training)
_h2o_frames: dict[str, Any] = {}
_automl_handles: dict[str, Any] = {}


def _data_tools_path() -> Path:
    return Path(__file__).resolve().parents[3] / "tools" / "data-tools"


def _ensure_dt_path() -> None:
    p = str(_data_tools_path())
    if p not in sys.path:
        sys.path.insert(0, p)


def _h2o_available() -> tuple[bool, str]:
    try:
        import h2o  # noqa: F401

        return True, ""
    except ImportError:
        return False, "h2o package not installed"
    except Exception as e:
        return False, str(e)


@tool
def h2o_init(max_mem_size: str = "4G", nthreads: int = -1) -> str:
    """Start (or attach to) an H2O cluster. Call once before other h2o_* tools."""
    ok, err = _h2o_available()
    if not ok:
        return json.dumps({"ok": False, "error": err})
    try:
        import h2o

        h2o.init(max_mem_size=max_mem_size, nthreads=nthreads, strict_version_check=False)
        return json.dumps({"ok": True, "cluster": str(h2o.cluster().get_status())})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


@tool
def h2o_shutdown(prompt: bool = True) -> str:
    """Shut down the H2O cluster and clear in-memory frame handles."""
    ok, err = _h2o_available()
    if not ok:
        _h2o_frames.clear()
        _automl_handles.clear()
        return json.dumps({"ok": True, "note": err or "no h2o"})
    try:
        import h2o

        h2o.cluster().shutdown(prompt=prompt)
    except Exception:
        pass
    _h2o_frames.clear()
    _automl_handles.clear()
    return json.dumps({"ok": True})


@tool
def h2o_import_frame(dataset_ref: str) -> str:
    """Load a registered pandas dataset into H2O. Returns frame_id for other tools."""
    ok, err = _h2o_available()
    if not ok:
        return json.dumps({"ok": False, "error": err})
    _ensure_dt_path()
    try:
        from utils import get_registered_dataset

        df = get_registered_dataset(dataset_ref)
        if df is None:
            return json.dumps({"ok": False, "error": f"dataset not found: {dataset_ref}"})
        import h2o

        hf = h2o.H2OFrame(df)
        fid = hf.frame_id
        _h2o_frames[fid] = hf
        return json.dumps({"ok": True, "frame_id": fid, "rows": hf.nrows, "cols": hf.ncols})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


def _get_frame(frame_id: str):
    import h2o

    if frame_id in _h2o_frames:
        return _h2o_frames[frame_id]
    return h2o.get_frame(frame_id)


@tool
def h2o_automl_run(
    frame_id: str,
    target_column: str,
    max_models: int = 10,
    max_runtime_secs: int = 0,
    nfolds: int = 3,
    stopping_metric: str = "AUTO",
    balance_classes: bool = False,
    seed: int = 42,
    project_name: Optional[str] = None,
) -> str:
    """Run H2O AutoML on a frame. Returns automl_id, leader_model_id, and leaderboard preview."""
    ok, err = _h2o_available()
    if not ok:
        return json.dumps({"ok": False, "error": err})
    try:
        import h2o
        from h2o.automl import H2OAutoML

        fr = _get_frame(frame_id)
        if fr is None:
            return json.dumps({"ok": False, "error": f"unknown frame_id: {frame_id}"})
        pname = project_name or f"jubilee_automl_{uuid.uuid4().hex[:8]}"
        aml = H2OAutoML(
            max_models=max_models,
            max_runtime_secs=max_runtime_secs or 3600,
            nfolds=nfolds,
            stopping_metric=stopping_metric,
            balance_classes=balance_classes,
            seed=seed,
            sort_metric="AUTO",
            project_name=pname,
        )
        aml.train(y=target_column, training_frame=fr)
        leader = aml.leader
        aid = getattr(aml, "project_name", None) or pname
        _automl_handles[aid] = aml
        lb = aml.leaderboard.as_data_frame()
        preview = lb.head(15).to_dict(orient="records")
        return json.dumps(
            {
                "ok": True,
                "automl_id": aid,
                "leader_model_id": leader.model_id,
                "leaderboard_preview": preview,
            }
        )
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


@tool
def h2o_leaderboard(automl_id: str) -> str:
    """Return the full leaderboard for a prior AutoML project id."""
    ok, err = _h2o_available()
    if not ok:
        return json.dumps({"ok": False, "error": err})
    aml = _automl_handles.get(automl_id)
    if aml is None:
        return json.dumps({"ok": False, "error": f"unknown automl_id: {automl_id}"})
    try:
        lb = aml.leaderboard.as_data_frame()
        return json.dumps({"ok": True, "leaderboard": lb.to_dict(orient="records")})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


@tool
def h2o_explain(model_id: str, frame_id: str) -> str:
    """Run H2O explain on a model + frame (variable importance, SHAP summary, etc.)."""
    ok, err = _h2o_available()
    if not ok:
        return json.dumps({"ok": False, "error": err})
    try:
        import h2o

        m = h2o.get_model(model_id)
        fr = _get_frame(frame_id)
        ex = h2o.explain(m, fr)
        out = {k: str(v)[:2000] for k, v in (ex or {}).items()} if isinstance(ex, dict) else {"summary": str(ex)[:8000]}
        return json.dumps({"ok": True, "explain": out})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


@tool
def h2o_download_mojo(model_id: str) -> str:
    """Download MOJO for an H2O model to a temp path; returns local filesystem path."""
    ok, err = _h2o_available()
    if not ok:
        return json.dumps({"ok": False, "error": err})
    try:
        import h2o

        m = h2o.get_model(model_id)
        d = tempfile.mkdtemp(prefix="h2o_mojo_")
        path = m.download_mojo(d)
        return json.dumps({"ok": True, "mojo_path": path})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


@tool
def h2o_to_registered_model(model_id: str, model_name: str, target_column: str, feature_columns: list[str]) -> str:
    """Save MOJO + a small sklearn-compatible wrapper, then register via model_storage.register_model."""
    ok, err = _h2o_available()
    if not ok:
        return json.dumps({"ok": False, "error": err})
    try:
        import h2o
        import joblib

        from model_storage import generate_model_path, register_model

        m = h2o.get_model(model_id)
        d = tempfile.mkdtemp(prefix="h2o_reg_")
        mojo_path = m.download_mojo(d)

        class _H2OMojoSklearnAdapter:
            """Minimal predict API for sklearn-like tooling."""

            def __init__(self, mojo_zip: str, target_column: str, feature_columns: list[str]):
                self._mojo = mojo_zip
                self._target = target_column
                self._features = list(feature_columns)

            def predict(self, X):  # noqa: ANN001
                import h2o
                import pandas as pd

                h2o.init(strict_version_check=False)
                mdl = h2o.import_mojo(self._mojo)
                if isinstance(X, pd.DataFrame):
                    hf = h2o.H2OFrame(X[self._features] if all(c in X.columns for c in self._features) else X)
                else:
                    hf = h2o.H2OFrame(X)
                pred = mdl.predict(hf)
                pdf = pred.as_data_frame()
                return pdf.iloc[:, 0].values

        adapter = _H2OMojoSklearnAdapter(mojo_path, target_column, feature_columns)
        out_path = generate_model_path(model_name)
        joblib.dump(adapter, out_path)
        reg = register_model(
            model_name=model_name,
            model_path=out_path,
            model_type=f"h2o_mojo:{model_id}",
            description=f"H2O MOJO wrapper for {model_id}",
            metrics={},
            feature_names=feature_columns,
            target_column=target_column,
            hyperparameters={"h2o_model_id": model_id, "mojo_path": mojo_path},
            training_samples=0,
            classes=[],
        )
        return json.dumps({"ok": True, "registered": reg})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


@tool
def h2o_train_estimator(
    kind: str,
    frame_id: str,
    target_column: str,
    hyperparams_json: str = "{}",
) -> str:
    """Train a single H2O estimator. kind in glm, gbm, xgb, drf, deeplearning, isolationforest, kmeans (no y for kmeans)."""
    ok, err = _h2o_available()
    if not ok:
        return json.dumps({"ok": False, "error": err})
    try:
        import h2o

        fr = _get_frame(frame_id)
        if fr is None:
            return json.dumps({"ok": False, "error": f"unknown frame_id: {frame_id}"})
        params = json.loads(hyperparams_json or "{}")
        k = kind.lower().strip()
        if k == "glm":
            from h2o.estimators import H2OGeneralizedLinearEstimator

            est = H2OGeneralizedLinearEstimator(**params)
            est.train(x=[c for c in fr.columns if c != target_column], y=target_column, training_frame=fr)
        elif k == "gbm":
            from h2o.estimators import H2OGradientBoostingEstimator

            est = H2OGradientBoostingEstimator(**params)
            est.train(x=[c for c in fr.columns if c != target_column], y=target_column, training_frame=fr)
        elif k == "xgb":
            from h2o.estimators import H2OXGBoostEstimator

            est = H2OXGBoostEstimator(**params)
            est.train(x=[c for c in fr.columns if c != target_column], y=target_column, training_frame=fr)
        elif k == "drf":
            from h2o.estimators import H2ORandomForestEstimator

            est = H2ORandomForestEstimator(**params)
            est.train(x=[c for c in fr.columns if c != target_column], y=target_column, training_frame=fr)
        elif k == "deeplearning":
            from h2o.estimators import H2ODeepLearningEstimator

            est = H2ODeepLearningEstimator(**params)
            est.train(x=[c for c in fr.columns if c != target_column], y=target_column, training_frame=fr)
        elif k == "isolationforest":
            from h2o.estimators import H2OIsolationForestEstimator

            est = H2OIsolationForestEstimator(**params)
            est.train(x=[c for c in fr.columns if c != target_column], training_frame=fr)
        elif k == "kmeans":
            from h2o.estimators import H2OKMeansEstimator

            est = H2OKMeansEstimator(**params)
            est.train(x=[c for c in fr.columns], training_frame=fr)
        else:
            return json.dumps({"ok": False, "error": f"unknown kind: {kind}"})
        return json.dumps({"ok": True, "model_id": est.model_id})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


@tool
def h2o_stack_models(
    base_model_ids_json: str,
    metalearner: str = "gbm",
    frame_id: str = "",
    target_column: str = "",
) -> str:
    """Stack existing H2O models into a StackedEnsemble. Pass base model ids as JSON list."""
    ok, err = _h2o_available()
    if not ok:
        return json.dumps({"ok": False, "error": err})
    try:
        import h2o
        from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator

        bases = json.loads(base_model_ids_json or "[]")
        if not bases:
            return json.dumps({"ok": False, "error": "base_model_ids_json must be a non-empty JSON list"})
        models = [h2o.get_model(mid) for mid in bases]
        fr = _get_frame(frame_id) if frame_id else None
        if fr is None:
            return json.dumps({"ok": False, "error": "frame_id required for stacking"})
        se = H2OStackedEnsembleEstimator(
            metalearner_algorithm=metalearner,
            base_models=[m.model_id for m in models],
        )
        xcols = [c for c in fr.columns if c != target_column]
        se.train(x=xcols, y=target_column, training_frame=fr)
        return json.dumps({"ok": True, "model_id": se.model_id})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


def build_h2o_tools():
    """Return LangChain tools for H2O (may fail at runtime if Java missing)."""
    return [
        h2o_init,
        h2o_shutdown,
        h2o_import_frame,
        h2o_automl_run,
        h2o_leaderboard,
        h2o_train_estimator,
        h2o_stack_models,
        h2o_explain,
        h2o_download_mojo,
        h2o_to_registered_model,
    ]
