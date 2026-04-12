"""Pydantic schemas for MRM API responses and bodies."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

PortalName = Literal["owner", "validation", "board"]

Decision = Literal["pending", "approved", "rejected", "acknowledged"]


class ModelRiskProfileOut(BaseModel):
    id: str
    model_id: str
    business_owner: str
    business_line: str
    risk_tier: str
    governance_profile: str
    model_origin: str
    business_purpose: str
    approved_use: str
    prohibited_use: str
    limitations: str | None = None
    board_visible: bool
    current_lifecycle_stage: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": False}


class ModelLifecycleStageOut(BaseModel):
    id: str
    model_id: str
    stage_name: str
    status: str
    owner_role: str | None = None
    started_at: datetime
    completed_at: datetime | None = None
    summary: str | None = None
    result_payload: dict[str, Any] | None = None


class ModelArtifactOut(BaseModel):
    id: str
    model_id: str
    stage_name: str
    artifact_type: str
    title: str
    storage_key: str | None = None
    external_url: str | None = None
    summary: str | None = None
    uploaded_by: str | None = None
    created_at: datetime


class ModelApprovalOut(BaseModel):
    id: str
    model_id: str
    stage_name: str
    approval_role: str
    approver_name: str
    decision: str
    comment: str | None = None
    decided_at: datetime | None = None


class ModelMonitoringSnapshotOut(BaseModel):
    id: str
    model_id: str
    model_version_id: str | None = None
    captured_at: datetime
    status: str
    psi_score: float | None = None
    backtest_score: float | None = None
    metrics_payload: dict[str, Any] | None = None
    top_csi_features: list[dict[str, Any]] | None = None
    note: str | None = None


class ModelReviewScheduleOut(BaseModel):
    id: str
    model_id: str
    review_type: str
    cadence: str
    next_due_at: datetime | None = None
    last_completed_at: datetime | None = None
    status: str


class ModelListItemOut(BaseModel):
    model_id: str
    model_name: str
    profile: ModelRiskProfileOut | None = None
    latest_monitoring: ModelMonitoringSnapshotOut | None = None
    next_review: ModelReviewScheduleOut | None = None


class ModelDetailOut(BaseModel):
    model_id: str
    model_name: str
    profile: ModelRiskProfileOut | None = None
    lifecycle_stages: list[ModelLifecycleStageOut]
    artifacts: list[ModelArtifactOut]
    approvals: list[ModelApprovalOut]
    monitoring_snapshots: list[ModelMonitoringSnapshotOut]
    review_schedules: list[ModelReviewScheduleOut]


class DashboardSummaryOut(BaseModel):
    portal: str
    total_models: int
    with_profile: int
    overdue_reviews: int
    high_risk_models: list[str] = Field(default_factory=list)
    board_visible_models: list[str] = Field(default_factory=list)
    watch_or_breach_models: list[str] = Field(default_factory=list)
    blocked_lifecycle_models: list[str] = Field(default_factory=list)
    in_progress_validation_models: list[str] = Field(default_factory=list)


class ApprovalCreateBody(BaseModel):
    stage_name: str
    approval_role: str
    approver_name: str
    decision: Decision
    comment: str | None = None


class ProfilePatchBody(BaseModel):
    business_owner: str | None = None
    business_line: str | None = None
    risk_tier: str | None = None
    governance_profile: str | None = None
    model_origin: str | None = None
    business_purpose: str | None = None
    approved_use: str | None = None
    prohibited_use: str | None = None
    limitations: str | None = None
    board_visible: bool | None = None
    current_lifecycle_stage: str | None = None
