"""Goal & Life Coach Router (Accountability, Reflections & Gamification)."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime, timezone

from backend.app.database import db
from backend.app.core_loop import core_orchestrator
from backend.app.agents.goal_coach import goal_coach

router = APIRouter(prefix="/goals", tags=["Goal & Life Coach"])

class GoalCreateIn(BaseModel):
    title: str
    category: Literal["study", "career", "personal"] = "study"
    target_date: Optional[str] = None
    why: str = ""
    milestones: Optional[List[Dict[str, Any]]] = None

class ReflectionIn(BaseModel):
    energy: int = Field(..., ge=1, le=5, description="Energy rating from 1 (exhausted) to 5 (peak)")
    mood: str = Field(..., description="e.g. focused, tired, motivated, anxious, steady")
    reflection: str = Field(..., description="Qualitative self-reflection note")
    obstacles: Optional[str] = ""

class HabitCheckIn(BaseModel):
    habit_name: str

@router.get("/")
def get_goals_and_coaching():
    """Retrieve user goals, SMART milestones, habit streaks, shields, and badges."""
    state = db.get_state()
    model = db.get_adaptive_model()
    return {
        "status": "success",
        "goals": state.get("goals", []),
        "habits": state.get("habits", []),
        "achievements": state.get("achievements", []),
        "reflections": state.get("reflections", []),
        "accountability_metrics": {
            "burnout_risk_score": model.get("burnout_risk_score", "Low"),
            "motivation_index": model.get("motivation_index", 0.75),
            "freezes_summary": sum(h.get("freezes_available", 0) for h in state.get("habits", [])),
        }
    }

@router.post("/")
def create_goal(goal_in: GoalCreateIn):
    """Create a new long-term goal decomposed into milestones."""
    new_goal = goal_coach.create_goal(
        title=goal_in.title,
        category=goal_in.category,
        target_date=goal_in.target_date,
        why=goal_in.why,
        milestones=goal_in.milestones
    )
    goals = db.get_goals()
    goals.append(new_goal)
    db.save_goals(goals)

    # Dispatch event
    db.add_event({
        "source": "Goal Coach Agent",
        "type": "goal_created",
        "payload": {"goal_id": new_goal["goal_id"], "title": new_goal["title"]}
    })

    return {"status": "success", "goal": new_goal}

@router.put("/{goal_id}/milestones/{milestone_id}")
def update_milestone_status(goal_id: str, milestone_id: str, status: str):
    """Update milestone completion status."""
    goals = db.get_goals()
    found = False
    for g in goals:
        if g.get("goal_id") == goal_id:
            for m in g.get("milestones", []):
                if m.get("id") == milestone_id:
                    m["status"] = status
                    found = True
                    break
    if not found:
        raise HTTPException(status_code=404, detail="Milestone or Goal not found.")

    db.save_goals(goals)
    db.add_event({
        "source": "Goal Coach Agent",
        "type": "milestone_updated",
        "payload": {"goal_id": goal_id, "milestone_id": milestone_id, "status": status}
    })
    return {"status": "success", "milestone_id": milestone_id, "new_status": status}

@router.post("/reflections")
def submit_reflection(ref: ReflectionIn):
    """Submit daily reflection check-in, score burnout risk, and trigger closed-loop adaptation."""
    res = core_orchestrator.submit_reflection_and_adapt(
        energy=ref.energy,
        mood=ref.mood,
        reflection_text=ref.reflection,
        obstacles=ref.obstacles or ""
    )
    return {"status": "success", "result": res}

@router.post("/habits/checkin")
def checkin_habit(habit_in: HabitCheckIn):
    """Daily habit check-in with streak freeze shield protection and badge awards."""
    res = core_orchestrator.complete_habit_and_adapt(habit_in.habit_name)
    return res
