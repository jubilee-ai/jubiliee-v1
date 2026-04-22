from pydantic import BaseModel, Field


class EvaluationReport(BaseModel):
    core_metrics: dict = Field(default_factory=dict)
    calibration: dict | None = None
    fairness_flags: list[str] = Field(default_factory=list)
    artifact_uris: list[str] = Field(default_factory=list)
    recommendation: str = ""
