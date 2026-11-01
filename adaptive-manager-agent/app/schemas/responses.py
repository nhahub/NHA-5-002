from pydantic import BaseModel

class DispatchCommand(BaseModel):
    command: str
    target_agent: str
    reason: str
    context: str
    status: str = "PENDING"

class RunResponse(BaseModel):
    run_id: int
    student_id: str
    insight: str
    adaptive_model: dict
    metrics: dict
    daily_risk: dict
    risk_score: float
    consecutive_risk_days: int
    predictions: list[dict]
    recommended_block_mins: int
    commands: list[DispatchCommand]
