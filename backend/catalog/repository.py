import json
from typing import Any

from backend.shared.settings import get_settings


def get_datasets() -> list[dict[str, object]]:
    """Return datasets from Postgres if available, else fall back to catalog.json."""
    try:
        return _get_datasets_from_db()
    except Exception:
        pass

    settings = get_settings()
    if not settings.datasets_catalog_path.exists():
        return []

    with open(settings.datasets_catalog_path) as f:
        catalog = json.load(f)

    datasets = catalog.get("datasets", [])
    return [
        ds
        for ds in datasets
        if ds.get("format", "").upper() in ("CSV", "PARQUET")
        or ds.get("file", "").endswith(".csv")
        or ds.get("file", "").endswith(".parquet")
    ]


def _get_datasets_from_db() -> list[dict[str, object]]:
    from backend.shared.database import get_db_session
    from backend.shared.models import Dataset

    with get_db_session() as session:
        rows = session.query(Dataset).order_by(Dataset.created_at.desc()).all()
        if not rows:
            raise LookupError("no rows")
        return [_dataset_to_dict(r) for r in rows]


def _dataset_to_dict(row: Any) -> dict[str, object]:
    props = row.properties or {}
    file_path = props.get("file") or None
    return {
        "id": str(row.id),
        "name": row.name,
        "file": file_path,
        "trainable": bool(file_path),
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
    """Return trained models from Postgres if available, else fall back to registry.json."""
    try:
        return _get_trained_models_from_db()
    except Exception:
        pass

    settings = get_settings()
    if not settings.models_registry_path.exists():
        return {}

    with open(settings.models_registry_path) as f:
        registry = json.load(f)
    return registry.get("models", {})


def _get_trained_models_from_db() -> dict[str, object]:
    from backend.shared.database import get_db_session
    from backend.shared.models import Model, ModelVersion

    with get_db_session() as session:
        models = session.query(Model).all()
        if not models:
            raise LookupError("no rows")
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
            v_props = version.properties or {}
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
            }
        return result
