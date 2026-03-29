import logging
import uuid as _uuid
from typing import Any, Optional

from sqlalchemy import or_, and_, select

log = logging.getLogger(__name__)


def get_datasets(
    include_derived: bool = False,
    user_id: Optional[_uuid.UUID] = None,
    org_id: Optional[_uuid.UUID] = None,
) -> list[dict[str, object]]:
    """Return datasets from Postgres, scoped to the current user."""
    try:
        return _get_datasets_from_db(
            include_derived=include_derived,
            user_id=user_id,
            org_id=org_id,
        )
    except Exception:
        log.exception("Failed to fetch datasets from database")
        return []


def _get_datasets_from_db(
    include_derived: bool = False,
    user_id: Optional[_uuid.UUID] = None,
    org_id: Optional[_uuid.UUID] = None,
) -> list[dict[str, object]]:
    from backend.shared.database import get_db_session
    from backend.shared.models import (
        Dataset, Experiment, RunDatasetLink, TrainingJob, User,
    )

    with get_db_session() as session:
        query = session.query(Dataset).order_by(Dataset.created_at.desc())
        if not include_derived:
            query = query.filter(Dataset.source_type != "derived")

        if user_id is not None:
            org_user_ids = select(User.id).where(User.organization_id == org_id).scalar_subquery()
            experiment_dataset_ids = (
                select(RunDatasetLink.dataset_id)
                .join(TrainingJob, RunDatasetLink.training_run_id == TrainingJob.id)
                .join(Experiment, TrainingJob.experiment_id == Experiment.id)
                .where(Experiment.user_id == user_id)
                .scalar_subquery()
            )
            query = query.filter(
                or_(
                    Dataset.user_id == user_id,
                    Dataset.id.in_(experiment_dataset_ids),
                    and_(Dataset.shared_with_org.is_(True), Dataset.user_id.in_(org_user_ids)),
                    Dataset.user_id.is_(None),
                )
            )

        rows = query.all()
        return [_dataset_to_dict(r, user_id=user_id) for r in rows]


def _dataset_to_dict(row: Any, user_id: Optional[_uuid.UUID] = None) -> dict[str, object]:
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
        "user_id": str(row.user_id) if row.user_id else None,
        "shared_with_org": row.shared_with_org,
        "is_owner": row.user_id == user_id if user_id else True,
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


def get_trained_models(
    user_id: Optional[_uuid.UUID] = None,
    org_id: Optional[_uuid.UUID] = None,
) -> dict[str, object]:
    """Return trained models from Postgres, scoped to the current user."""
    try:
        return _get_trained_models_from_db(user_id=user_id, org_id=org_id)
    except Exception:
        log.exception("Failed to fetch trained models from database")
        return {}


def _get_trained_models_from_db(
    user_id: Optional[_uuid.UUID] = None,
    org_id: Optional[_uuid.UUID] = None,
) -> dict[str, object]:
    from backend.shared.database import get_db_session
    from backend.shared.models import (
        Experiment, Model, ModelVersion, TrainingJob, User,
    )

    with get_db_session() as session:
        query = session.query(Model)

        if user_id is not None:
            org_user_ids = select(User.id).where(User.organization_id == org_id).scalar_subquery()
            experiment_model_ids = (
                select(Model.id)
                .join(ModelVersion, Model.id == ModelVersion.model_id)
                .join(TrainingJob, ModelVersion.training_run_id == TrainingJob.id)
                .join(Experiment, TrainingJob.experiment_id == Experiment.id)
                .where(Experiment.user_id == user_id)
                .distinct()
                .scalar_subquery()
            )
            query = query.filter(
                or_(
                    Model.user_id == user_id,
                    Model.id.in_(experiment_model_ids),
                    and_(Model.shared_with_org.is_(True), Model.user_id.in_(org_user_ids)),
                    Model.user_id.is_(None),
                )
            )

        models = query.all()
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
                "user_id": str(model.user_id) if model.user_id else None,
                "shared_with_org": model.shared_with_org,
                "is_owner": model.user_id == user_id if user_id else True,
            }
        return result
