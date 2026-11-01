from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class EventIn(BaseModel):
    model_config = ConfigDict(extra="allow")
    event_id: str = Field(min_length=1, max_length=100)
    timestamp: datetime
    source: Literal["planning", "assessment", "coach", "tutor"]
    type: str = Field(min_length=1, max_length=60)
    topic: str | None = None
    planned_mins: float | None = Field(default=None, gt=0)
    actual_focus_mins: float | None = Field(default=None, ge=0)
    actual_mins: float | None = Field(default=None, gt=0)
    score_pct: float | None = Field(default=None, ge=0, le=100)
    failed_concepts: list[str] = Field(default_factory=list)
    energy: int | None = Field(default=None, ge=1, le=5)
    procrastination_risk: float | None = Field(default=None, ge=0, le=1)
    day_type: str | None = None
    lecture_hours: float | None = Field(default=None, ge=0)
    exam: str | None = None

class EventBatch(BaseModel):
    events: list[EventIn] = Field(min_length=1)

class ApprovalIn(BaseModel):
    approved: bool
