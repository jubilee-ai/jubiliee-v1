from typing import Optional

from pydantic import BaseModel, Field


class MonitoringReport(BaseModel):
    drift_flags: list[str] = Field(default_factory=list)
    perf_delta: Optional[float] = None
    recommendation: str = ""
    retrain_request_id: Optional[str] = None
