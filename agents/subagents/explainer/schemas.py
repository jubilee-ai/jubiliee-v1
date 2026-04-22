from pydantic import BaseModel, Field


class ExplanationReport(BaseModel):
    summary_markdown: str = ""
    highlight_charts: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
