from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

GoalCategory = Literal["career", "study", "personal"]
MilestoneStatus = Literal["pending", "in_progress", "completed", "blocked"]
RiskLevel = Literal["Low", "Medium", "High"]

class MicroTask(BaseModel):
    title: str
    estimated_mins: int = Field(default=30, ge=5, le=240)

class Milestone(BaseModel):
    id: str
    goal_id: str
    title: str
    status: MilestoneStatus = "pending"
    criteria: str = ""
    due_date: Optional[str] = None
    micro_tasks: list[MicroTask] = Field(default_factory=list)

class Goal(BaseModel):
    goal_id: str
    title: str
    category: GoalCategory = "personal"
    target_date: Optional[str] = None
    why: str = ""
    status: Literal["active", "completed"] = "active"
    strategy_note: str = ""

class Habit(BaseModel):
    name: str
    streak: int = 0
    best_streak: int = 0
    last_completed: Optional[str] = None
    freezes_available: int = 1

class GoalsData(BaseModel):
    career_goals: list[Goal] = Field(default_factory=list)
    study_goals: list[Goal] = Field(default_factory=list)
    personal_goals: list[Goal] = Field(default_factory=list)
    milestones: list[Milestone] = Field(default_factory=list)
    habits: list[Habit] = Field(default_factory=list)

class BurnoutRisk(BaseModel):
    score: float = Field(default=0.2, ge=0, le=1)
    level: RiskLevel = "Low"

class CoachAdaptive(BaseModel):
    motivation_index: float = Field(default=0.6, ge=0, le=1)
    burnout_risk: BurnoutRisk = Field(default_factory=BurnoutRisk)
    burnout_risk_score: RiskLevel = "Low"

class Reflection(BaseModel):
    date: str
    text: str
    mood: float
    energy: float
    friction: list[str] = Field(default_factory=list)
    burnout_signals: list[str] = Field(default_factory=list)

class Achievement(BaseModel):
    key: str
    title: str
    earned_on: str

class CoachState(BaseModel):
    user_id: str
    goals_data: GoalsData = Field(default_factory=GoalsData)
    adaptive_model: CoachAdaptive = Field(default_factory=CoachAdaptive)
    reflections: list[Reflection] = Field(default_factory=list)
    achievements: list[Achievement] = Field(default_factory=list)
    consecutive_low_days: int = 0

    def all_goals(self) -> list[Goal]:
        g = self.goals_data
        return g.career_goals + g.study_goals + g.personal_goals

class MicroTask2(MicroTask):
    pass

class MilestoneDraft(BaseModel):
    title: str
    criteria: str = Field(description="Measurable success criterion (the M in SMART)")
    due_date: Optional[str] = Field(default=None, description="ISO date or null")
    micro_tasks: list[MicroTask]

class GoalDecomposition(BaseModel):
    category: GoalCategory
    why: str = Field(description="One sentence on why this goal matters to the user")
    relevance_tags: list[str] = Field(description="3-5 topics the Tutor can link to this goal")
    milestones: list[MilestoneDraft] = Field(min_length=2, max_length=8)

class ReflectionAnalysis(BaseModel):
    mood: float = Field(ge=-1, le=1)
    energy: float = Field(ge=0, le=1)
    friction: list[str] = Field(default_factory=list)
    avoidance_signals: list[str] = Field(default_factory=list)
    burnout_signals: list[str] = Field(default_factory=list)
    summary: str = ""

class CoachReply(BaseModel):
    message: str
    suggest_strategy_change: bool = False
    strategy_note: str = ""

class AgentEvent(BaseModel):
    event_type: str
    user_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    state_snapshot: Optional[dict[str, Any]] = None

class OutboundEvent(BaseModel):
    target: Literal["PLANNING", "MANAGER", "TUTOR"]
    command: str
    payload: dict[str, Any] = Field(default_factory=dict)

class AgentResult(BaseModel):
    state_updates: dict[str, Any] = Field(default_factory=dict)
    outbound_events: list[OutboundEvent] = Field(default_factory=list)
    user_message: str = ""
    error: Optional[str] = None

