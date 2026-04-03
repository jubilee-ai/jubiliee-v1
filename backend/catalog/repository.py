import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _trained_model_report_path(model_name: str) -> Path:
    """Same layout as ``GET /api/trained-models/{model_name}/report``."""
    return Path(__file__).resolve().parents[2] / "trained_models" / f"{model_name}_report.json"


def get_datasets(include_derived: bool = False) -> list[dict[str, object]]:
    """Return datasets from Postgres."""
    try:
        return _get_datasets_from_db(include_derived=include_derived)
    except Exception:
        log.exception("Failed to fetch datasets from database")
        return []


def _get_datasets_from_db(include_derived: bool = False) -> list[dict[str, object]]:
    from backend.shared.database import get_db_session
    from backend.shared.models import Dataset

    with get_db_session() as session:
        query = session.query(Dataset).order_by(Dataset.created_at.desc())
        if not include_derived:
            query = query.filter(Dataset.source_type != "derived")
        rows = query.all()
        return [_dataset_to_dict(r) for r in rows]


def _dataset_to_dict(row: Any) -> dict[str, object]:
    props = row.properties or {}
    file_path = props.get("file") or None
    storage_key = props.get("storage_key") or None
    has_data = bool(file_path or storage_key)
    return {
        "id": str(row.id),
        "name": row.name,
        "file": file_path,
        "storage_key": storage_key,
        "trainable": has_data,
        "source_type": row.source_type,
        "format": props.get("format", ""),
        "rows": props.get("row_count"),
        "columns": props.get("columns", []),
        "description": props.get("description", ""),
        "use_case": props.get("use_case", ""),
    }


def get_models() -> list[dict[str, str]]:
    return [
        {
            "id": "logistic_regression",
            "name": "Logistic Regression",
            "description": "Binary/multiclass classification, interpretable",
        },
        {
            "id": "random_forest",
            "name": "Random Forest",
            "description": "Classification/regression, feature importance",
        },
        {
            "id": "xgboost",
            "name": "XGBoost",
            "description": "High-performance tabular data",
        },
        {
            "id": "naive_bayes",
            "name": "Naive Bayes",
            "description": "Fast probabilistic classifier, great baseline",
        },
        {
            "id": "glm",
            "name": "GLM",
            "description": "Poisson/Gamma/Tweedie regression",
        },
        {
            "id": "survival_analysis",
            "name": "Survival Analysis",
            "description": "Time-to-event prediction with censoring",
        },
    ]


def get_trained_models() -> dict[str, object]:
    """Return trained models from Postgres."""
    try:
        return _get_trained_models_from_db()
    except Exception:
        log.exception("Failed to fetch trained models from database")
        return {}


def _get_trained_models_from_db() -> dict[str, object]:
    from backend.shared.database import get_db_session
    from backend.shared.models import Experiment, Model, ModelVersion

    with get_db_session() as session:
        models = session.query(Model).all()
        result = {}
        for model in models:
            version = (
                session.query(ModelVersion)
                .filter(
                    ModelVersion.model_id == model.id,
                    ModelVersion.is_current.is_(True),
                )
                .first()
            )
            if version is None:
                continue
            props = model.properties or {}
            exp_id = props.get("experiment_id") if isinstance(props.get("experiment_id"), str) else None
            experiment_name = None
            if exp_id:
                exp = session.get(Experiment, exp_id)
                if exp is not None:
                    experiment_name = exp.name
                else:
                    snap = props.get("experiment_name")
                    experiment_name = snap if isinstance(snap, str) else None
            v_props = version.properties or {}
            report_path = _trained_model_report_path(model.name)
            result[model.name] = {
                "model_name": model.name,
                "model_type": props.get("model_type", ""),
                "description": props.get("description", ""),
                "metrics": version.metrics or {},
                "feature_names": props.get("feature_names", []),
                "target_column": props.get("target_column", ""),
                "hyperparameters": v_props.get("hyperparameters", {}),
                "training_samples": v_props.get("training_samples", 0),
                "classes": v_props.get("classes", []),
                "created_at": model.created_at.isoformat() if model.created_at else "",
                "updated_at": model.updated_at.isoformat() if model.updated_at else "",
                "version": version.version,
                "experiment_id": exp_id,
                "experiment_name": experiment_name,
                "report_available": report_path.is_file(),
            }
        return result
