"""Goal & Life Coach Agent module.

Member 4 in the 'Generative AI Personal Life & Study Coach' multi-agent system.
"""

from coach.agent import GoalCoachAgent
from coach.schemas import (
    Achievement,
    AgentEvent,
    AgentResult,
    BurnoutRisk,
    CoachAdaptive,
    CoachReply,
    CoachState,
    Goal,
    GoalCategory,
    GoalDecomposition,
    Habit,
    MicroTask,
    Milestone,
    MilestoneStatus,
    OutboundEvent,
    Reflection,
    ReflectionAnalysis,
    RiskLevel,
)
from coach.storage import JsonStore, MemoryStore, Store

__all__ = [
    "GoalCoachAgent",
    "Store",
    "MemoryStore",
    "JsonStore",
    "CoachState",
    "AgentEvent",
    "AgentResult",
    "OutboundEvent",
    "Goal",
    "GoalCategory",
    "Milestone",
    "MilestoneStatus",
    "MicroTask",
    "Habit",
    "Reflection",
    "Achievement",
    "CoachAdaptive",
    "BurnoutRisk",
    "RiskLevel",
    "GoalDecomposition",
    "ReflectionAnalysis",
    "CoachReply",
]

