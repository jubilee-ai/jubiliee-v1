import json
import re
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from backend.catalog import repository as catalog_repository
from backend.catalog import service as catalog_service
from backend.catalog.interfaces import CatalogServiceInterface
from backend.catalog.repository import _trained_model_report_path
from backend.catalog.service import (DatasetUploadConflictError,
                                     DatasetUploadValidationError)
from backend.shared.artifact_store import get_artifact_store
from backend.shared.auth import ClerkUser, require_org
from backend.shared.training_artifacts import download_json_artifact
from backend.shared.database import get_db_session
from backend.shared.models import Dataset, Model, ModelVersion

router = APIRouter()


class PredictFeaturesBody(BaseModel):
    features: dict[str, Any] = Field(..., description="One row: feature column name → value")


def get_catalog_service() -> CatalogServiceInterface:
    return catalog_service


@router.get("/api/datasets")
async def get_datasets(
    include_derived: bool = False,
    service: CatalogServiceInterface = Depends(get_catalog_service),
    user: ClerkUser = Depends(require_org),
):
    try:
        return service.get_datasets(
            include_derived=include_derived,
            org_id=user.org_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str | None = Form(None),
    description: str | None = Form(None),
    service: CatalogServiceInterface = Depends(get_catalog_service),
    user: ClerkUser = Depends(require_org),
):
    try:
        content = await file.read()
        return service.upload_user_dataset(
            content,
            file.filename,
            name=name,
            description=description,
            org_id=user.org_id,
        )
    except DatasetUploadValidationError as e:
        raise HTTPException(status_code=400, detail=e.detail) from e
    except DatasetUploadConflictError as e:
        raise HTTPException(status_code=409, detail=e.detail) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/api/models")
async def get_models(
    service: CatalogServiceInterface = Depends(get_catalog_service),
    user: ClerkUser = Depends(require_org),
):
    return service.get_models()


@router.get("/api/trained-models")
async def get_trained_models(
    service: CatalogServiceInterface = Depends(get_catalog_service),
    user: ClerkUser = Depends(require_org),
):
    try:
        return service.get_trained_models(org_id=user.org_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/trained-models/{model_name}/report")
async def get_trained_model_report(
    model_name: str,
    user: ClerkUser = Depends(require_org),
):
    visible = catalog_repository.get_trained_models(org_id=user.org_id)
    if model_name not in visible:
        raise HTTPException(status_code=404, detail="Report not found")
    report_path = _trained_model_report_path(model_name)
    if report_path.is_file():
        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail="Invalid report JSON") from e

    safe = re.sub(r"[^\w\-.]", "_", model_name)
    storage_key = f"reports/{safe}_report.json"
    try:
        store = get_artifact_store()
        if store.exists(storage_key):
            return download_json_artifact(storage_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    raise HTTPException(status_code=404, detail="Report not found")


@router.get("/api/trained-models/{model_name}/download")
async def download_model(
    model_name: str,
    user: ClerkUser = Depends(require_org),
):
    visible = catalog_repository.get_trained_models(org_id=user.org_id)
    if model_name not in visible:
        raise HTTPException(status_code=404, detail="Model not found")
    with get_db_session() as session:
        model = session.query(Model).filter_by(name=model_name).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        version = (
            session.query(ModelVersion)
            .filter_by(model_id=model.id, is_current=True)
            .first()
        )
        if not version or not version.storage_key:
            raise HTTPException(status_code=404, detail="No downloadable version found")

        store = get_artifact_store()
        url = store.get_presigned_url(version.storage_key)
        return RedirectResponse(url=url, status_code=307)


@router.post("/api/trained-models/{model_name}/predict")
async def predict_trained_model(
    model_name: str,
    body: PredictFeaturesBody,
    user: ClerkUser = Depends(require_org),
):
    """Run model inference on a single feature row (no LLM)."""
    from model_storage import predict_from_feature_dict

    visible = catalog_repository.get_trained_models(org_id=user.org_id)
    if model_name not in visible:
        raise HTTPException(status_code=404, detail="Model not found")
    try:
        return predict_from_feature_dict(model_name, body.features)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/api/trained-models/{model_name}/input-features")
async def get_trained_model_input_features(
    model_name: str,
    user: ClerkUser = Depends(require_org),
):
    """Raw training columns for the prediction form (resolved from the fitted artifact when possible)."""
    from model_storage import get_predict_input_schema

    visible = catalog_repository.get_trained_models(org_id=user.org_id)
    if model_name not in visible:
        raise HTTPException(status_code=404, detail="Model not found")
    try:
        return get_predict_input_schema(model_name)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/api/datasets/{ref}/preview")
async def preview_dataset(
    ref: str,
    limit: int = 25,
    service: CatalogServiceInterface = Depends(get_catalog_service),
    user: ClerkUser = Depends(require_org),
):
    try:
        return service.get_dataset_preview(ref, limit=limit, org_id=user.org_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/api/datasets/{ref}/download")
async def download_dataset(
    ref: str,
    user: ClerkUser = Depends(require_org),
):
    with get_db_session() as session:
        dataset = (
            session.query(Dataset)
            .filter(Dataset.name == ref, Dataset.org_id == user.org_id)
            .first()
        )
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        storage_key = (dataset.properties or {}).get("storage_key")
        if not storage_key:
            raise HTTPException(status_code=404, detail="No downloadable data for this dataset")

        store = get_artifact_store()
        url = store.get_presigned_url(storage_key)
        return RedirectResponse(url=url, status_code=307)
