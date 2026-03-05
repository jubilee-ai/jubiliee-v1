import json

from backend.shared.settings import get_settings


def get_datasets() -> list[dict[str, object]]:
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
    settings = get_settings()
    if not settings.models_registry_path.exists():
        return {}

    with open(settings.models_registry_path) as f:
        registry = json.load(f)
    return registry.get("models", {})
