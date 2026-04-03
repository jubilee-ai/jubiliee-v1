import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse

from backend.catalog.interfaces import CatalogServiceInterface
from backend.catalog import service as catalog_service
from backend.catalog.repository import _trained_model_report_path
from backend.shared.artifact_store import get_artifact_store
from backend.shared.database import get_db_session
from backend.shared.models import Dataset, Model, ModelVersion

router = APIRouter()


def get_catalog_service() -> CatalogServiceInterface:
    return catalog_service


@router.get("/api/datasets")
async def get_datasets(
    include_derived: bool = False,
    service: CatalogServiceInterface = Depends(get_catalog_service),
):
    try:
        return service.get_datasets(include_derived=include_derived)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/models")
async def get_models(service: CatalogServiceInterface = Depends(get_catalog_service)):
    return service.get_models()


@router.get("/api/trained-models")
async def get_trained_models(
    service: CatalogServiceInterface = Depends(get_catalog_service),
):
    try:
        return service.get_trained_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/trained-models/{model_name}/report")
async def get_trained_model_report(model_name: str):
    report_path = _trained_model_report_path(model_name)
    if not report_path.is_file():
        raise HTTPException(status_code=404, detail="Report not found")
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail="Invalid report JSON") from e


@router.get("/api/trained-models/{model_name}/download")
async def download_model(model_name: str):
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


@router.get("/api/datasets/{ref}/download")
async def download_dataset(ref: str):
    with get_db_session() as session:
        dataset = session.query(Dataset).filter_by(name=ref).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        storage_key = (dataset.properties or {}).get("storage_key")
        if not storage_key:
            raise HTTPException(status_code=404, detail="No downloadable data for this dataset")

        store = get_artifact_store()
        url = store.get_presigned_url(storage_key)
        return RedirectResponse(url=url, status_code=307)
