from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class NotificationOut(BaseModel):
    id: str
    experiment_id: Optional[str] = None
    job_id: Optional[str] = None
    type: str
    title: str
    body: Optional[str] = None
    metadata: Optional[dict] = None
    read: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class MarkReadRequest(BaseModel):
    ids: list[str]
