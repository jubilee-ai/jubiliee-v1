from pydantic import BaseModel, Field


class DeploymentDecision(BaseModel):
    alias: str = "champion"
    endpoint_url: str = ""
    smoke_test_pass: bool = False
    requires_human_approval: bool = True
    reason: str = ""
