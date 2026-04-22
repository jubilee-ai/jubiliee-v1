from typing import Optional

from pydantic import BaseModel, Field


class DataAnalysisResult(BaseModel):
    primary_dataset_ref: Optional[str] = Field(None, description="Main collected dataset ref")
    train_ref: Optional[str] = None
    val_ref: Optional[str] = None
    test_ref: Optional[str] = None
    target_column: Optional[str] = None
    task_type_hint: Optional[str] = None
    schema_notes: str = ""
    issues: list[str] = Field(default_factory=list)
    description: Optional[str] = None
