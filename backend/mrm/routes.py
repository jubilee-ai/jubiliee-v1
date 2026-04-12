"""Model Risk Management (MRM) demo HTTP routes."""

from fastapi import APIRouter, HTTPException

from backend.mrm.schemas import (
    ApprovalCreateBody,
    DashboardSummaryOut,
    ModelApprovalOut,
    ModelDetailOut,
    ModelListItemOut,
    ModelRiskProfileOut,
    ProfilePatchBody,
)
from backend.mrm import service as mrm_service
from backend.shared.database import get_db_session

router = APIRouter()


@router.get("/api/model-risk/models", response_model=list[ModelListItemOut])
async def list_mrm_models():
    with get_db_session() as session:
        return mrm_service.list_models(session)


@router.get(
    "/api/model-risk/models/{model_name}",
    response_model=ModelDetailOut,
)
async def get_mrm_model_detail(model_name: str):
    with get_db_session() as session:
        detail = mrm_service.get_model_detail(session, model_name)
    if detail is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return detail


@router.get(
    "/api/model-risk/dashboards/{portal}",
    response_model=DashboardSummaryOut,
)
async def get_mrm_dashboard(portal: str):
    if portal not in ("owner", "validation", "board"):
        raise HTTPException(
            status_code=404,
            detail="Unknown portal; expected owner, validation, or board",
        )
    with get_db_session() as session:
        return mrm_service.get_dashboard(session, portal)


@router.post(
    "/api/model-risk/models/{model_name}/approvals",
    response_model=ModelApprovalOut,
)
async def create_mrm_approval(model_name: str, body: ApprovalCreateBody):
    with get_db_session() as session:
        row = mrm_service.add_approval(session, model_name, body)
        if row is None:
            raise HTTPException(status_code=404, detail="Model not found")
        return row


@router.patch(
    "/api/model-risk/models/{model_name}/profile",
    response_model=ModelRiskProfileOut,
)
async def patch_mrm_profile(model_name: str, body: ProfilePatchBody):
    with get_db_session() as session:
        prof = mrm_service.patch_profile(session, model_name, body)
        if prof is None:
            raise HTTPException(
                status_code=404,
                detail="Model or risk profile not found",
            )
        return prof
