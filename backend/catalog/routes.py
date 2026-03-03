from fastapi import APIRouter, Depends, HTTPException

from backend.catalog.interfaces import CatalogServiceInterface
from backend.catalog import service as catalog_service

router = APIRouter()


def get_catalog_service() -> CatalogServiceInterface:
    return catalog_service


@router.get("/api/datasets")
async def get_datasets(service: CatalogServiceInterface = Depends(get_catalog_service)):
    try:
        return service.get_datasets()
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
