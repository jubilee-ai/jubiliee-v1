from fastapi import APIRouter, Query

from backend.notifications import service
from backend.notifications.schemas import MarkReadRequest, NotificationOut

router = APIRouter()


@router.get("/api/notifications", response_model=list[NotificationOut])
async def list_notifications(
    limit: int = Query(20, ge=1, le=100),
    unread_only: bool = Query(False),
):
    return service.list_recent(limit=limit, unread_only=unread_only)


@router.post("/api/notifications/read")
async def mark_notifications_read(body: MarkReadRequest):
    count = service.mark_read(body.ids)
    return {"updated": count}
