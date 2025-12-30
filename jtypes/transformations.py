"""Type definitions for data transformations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal, Optional, Union

if TYPE_CHECKING:
    import pandas as pd


def _utc_now() -> str:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# Type alias
DatasetInput = Union["pd.DataFrame", str]


@dataclass
class TransformWarning:
    """Warning generated during transformation."""
    code: str
    message: str
    details: Optional[dict] = None
    
    def to_dict(self) -> dict:
        d = {"code": self.code, "message": self.message}
        if self.details:
            d["details"] = self.details
        return d


@dataclass
class SchemaChange:
    """Describes a schema change from a transformation."""
    change_type: Literal["added", "removed", "modified", "renamed"]
    column: str
    old_dtype: Optional[str] = None
    new_dtype: Optional[str] = None
    renamed_from: Optional[str] = None
    
    def to_dict(self) -> dict:
        d = {"change_type": self.change_type, "column": self.column}
        for k in ("old_dtype", "new_dtype", "renamed_from"):
            if getattr(self, k):
                d[k] = getattr(self, k)
        return d


@dataclass
class TransformAudit:
    """Audit record for a transformation operation."""
    op_name: str
    op_params: dict
    rows_before: int
    rows_after: int
    columns_before: list[str]
    columns_after: list[str]
    schema_changes: list[SchemaChange] = field(default_factory=list)
    warnings: list[TransformWarning] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "op": self.op_name, "params": self.op_params,
            "rows": {"before": self.rows_before, "after": self.rows_after},
            "columns": {"before": self.columns_before, "after": self.columns_after,
                        "count_before": len(self.columns_before), "count_after": len(self.columns_after)},
            "schema_changes": [s.to_dict() for s in self.schema_changes],
            "warnings": [w.to_dict() for w in self.warnings],
            "success": self.success, "error": self.error, "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class TransformResult:
    """Result from a single transformation operation."""
    df: "pd.DataFrame"
    audit: TransformAudit
    
    def to_dict(self) -> dict:
        return {"audit": self.audit.to_dict()}


class BaseTransform(ABC):
    """Abstract base class for transformations."""
    
    op_name: str = "base"
    
    @abstractmethod
    def validate(self, df: "pd.DataFrame") -> list[TransformWarning]:
        """Validate the transform can be applied. Returns warnings."""
        pass
    
    @abstractmethod
    def execute(self, df: "pd.DataFrame") -> TransformResult:
        """Execute the transformation."""
        pass
    
    def get_params(self) -> dict:
        """Get parameters for audit logging."""
        return {}


@dataclass
class LineageStep:
    """Single step in transformation lineage."""
    step_index: int
    op_name: str
    op_params: dict
    input_ref: str
    output_ref: str
    timestamp: str
    
    def to_dict(self) -> dict:
        return {"step": self.step_index, "op": self.op_name, "params": self.op_params,
                "input_ref": self.input_ref, "output_ref": self.output_ref, "timestamp": self.timestamp}


@dataclass
class TransformLineage:
    """Full lineage for a transform pipeline."""
    pipeline_id: str
    source_ref: str
    output_ref: str
    steps: list[LineageStep] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now)
    total_execution_time_ms: float = 0.0
    
    def to_dict(self) -> dict:
        return {"pipeline_id": self.pipeline_id, "source_ref": self.source_ref, "output_ref": self.output_ref,
                "steps": [s.to_dict() for s in self.steps], "created_at": self.created_at,
                "total_execution_time_ms": round(self.total_execution_time_ms, 2)}


@dataclass
class TransformApplyResult:
    """Result from transform_apply pipeline."""
    dataset_ref: str
    success: bool
    ops_applied: int
    ops_total: int
    audits: list[TransformAudit]
    lineage: Optional[TransformLineage] = None
    intermediates: Optional[dict] = None  # dict[str, pd.DataFrame]
    final_schema: list[dict] = field(default_factory=list)
    final_rows: int = 0
    warnings_total: int = 0
    execution_time_ms: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        d = {"dataset_ref": self.dataset_ref, "success": self.success, "ops_applied": self.ops_applied,
             "ops_total": self.ops_total, "audits": [a.to_dict() for a in self.audits],
             "final_schema": self.final_schema, "final_rows": self.final_rows,
             "warnings_total": self.warnings_total, "execution_time_ms": round(self.execution_time_ms, 2)}
        if self.lineage:
            d["lineage"] = self.lineage.to_dict()
        if self.error:
            d["error"] = self.error
        return d

