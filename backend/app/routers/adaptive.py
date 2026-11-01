"""Adaptive Personal Model, Struggle Predictions & Simulation Router."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from backend.app.database import db
from backend.app.core_loop import core_orchestrator
from backend.app.agents.adaptive_manager import adaptive_manager

router = APIRouter(prefix="/adaptive", tags=["Adaptive Model & Orchestrator"])

class EventIn(BaseModel):
    event_type: str = Field(..., description="e.g. block_completed, block_missed, task_completed, quiz_result, reflection")
    source: str = Field("User Interface", description="Originating agent or UI")
    payload: Dict[str, Any] = Field(default_factory=dict)
    auto_adapt: bool = True

class ModelCalibrateIn(BaseModel):
    effective_focus_duration_mins: Optional[int] = None
    peak_productivity_window: Optional[str] = None
    task_delay_factor: Optional[float] = None
    current_system_mode: Optional[str] = None

class SimulationScenarioIn(BaseModel):
    scenario: str = Field(..., description="e.g. heavy_lecture_stress, failed_quiz_automata, streak_comeback, ideal_morning_focus")

@router.get("/model")
def get_personal_model():
    """Return the user's continuously learned Personal Adaptive Model."""
    model = db.get_adaptive_model()
    user = db.get_user()
    return {
        "status": "success",
        "user_id": user.get("id", "student-001"),
        "adaptive_model": model,
        "usp_summary": "Continuous behavioral model predicting cognitive friction before failure occurs."
    }

@router.post("/model/calibrate")
def calibrate_personal_model(payload: ModelCalibrateIn):
    """Transparent calibration: Allows the user to inspect and adjust their model parameters."""
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No calibration parameters provided.")
    updated = db.update_adaptive_model(updates)
    return {"status": "success", "message": "Personal model calibrated.", "adaptive_model": updated}

@router.get("/predictions")
def get_struggle_predictions():
    """Return active struggle predictions (Task-miss risk, Topic-struggle risk, Overload risk)."""
    preds = db.get_predictions()
    return {"status": "success", "total_predictions": len(preds), "predictions": preds}

@router.get("/events")
def get_behavior_timeline():
    """Return the raw chronological behavior event timeline."""
    events = db.get_events()
    return {"status": "success", "total_events": len(events), "events": events}

@router.post("/events")
def ingest_behavior_event(event: EventIn):
    """Ingest a behavior event and trigger the closed-loop adaptation cycle."""
    res = core_orchestrator.trigger_behavior_event(
        event_type=event.event_type,
        source=event.source,
        payload=event.payload,
        auto_adapt=event.auto_adapt
    )
    return res

@router.post("/simulate")
def run_simulation_scenario(sim: SimulationScenarioIn):
    """Interactive Simulation Lab: Demonstrate live closed-loop adaptation with simulated personas."""
    scenario = sim.scenario.lower()
    now = datetime.now(timezone.utc).isoformat()

    if scenario == "heavy_lecture_stress":
        # Simulate student failing to complete a heavy task after 5 lecture hours
        event_payload = {
            "planned_mins": 60,
            "actual_focus_mins": 15,
            "day_type": "heavy_lecture",
            "reason": "Exhaustion after back-to-back classes"
        }
        res = core_orchestrator.trigger_behavior_event(
            event_type="block_cut_short",
            source="Simulation Lab",
            payload=event_payload,
            auto_adapt=True
        )
        return {
            "scenario": "Heavy Lecture Stress",
            "simulated_action": "Logged 15m focus on a 60m task during a heavy lecture day.",
            "adaptation_outcome": res,
            "explanation": "Adaptive Manager detected lecture fatigue, updated procrastination patterns, and triggered PREEMPTIVE_SCHEDULE_RESTRUCTURE."
        }

    elif scenario == "failed_quiz_automata":
        # Simulate student getting 40% on an Automata quiz
        res = core_orchestrator.trigger_behavior_event(
            event_type="quiz_result",
            source="Assessment Agent",
            payload={
                "topic": "Automata & Formal Languages",
                "score_pct": 40.0,
                "failed_concepts": ["Pumping Lemma proofs", "NFA epsilon transitions"],
                "error_types": {"Conceptual Gap": 3}
            },
            auto_adapt=True
        )
        return {
            "scenario": "Diagnostic Quiz Failure (Automata)",
            "simulated_action": "Submitted 40% diagnostic score in Automata with Pumping Lemma errors.",
            "adaptation_outcome": res,
            "explanation": "Mastery dropped below 65% threshold. Adaptive Manager dispatched AI Tutor for remedial practice and scheduled targeted micro-tasks."
        }

    elif scenario == "streak_comeback":
        # Simulate missing yesterday then completing today (comeback halving rule)
        res = core_orchestrator.complete_habit_and_adapt("Daily Deep Work Block")
        return {
            "scenario": "Habit Streak & Comeback Protection",
            "simulated_action": "Checked in daily habit after a missed day with 1 freeze shield.",
            "adaptation_outcome": res,
            "explanation": "Streak freeze consumed to protect momentum, preventing morale-damaging reset to zero."
        }

    else:
        # Default scenario: High-focus morning session
        res = core_orchestrator.trigger_behavior_event(
            event_type="block_completed",
            source="Planning Agent",
            payload={"planned_mins": 40, "actual_focus_mins": 40, "day_type": "peak_morning"},
            auto_adapt=True
        )
        return {
            "scenario": "Peak Morning Focus Session",
            "simulated_action": "Completed 40-minute focused session during peak window.",
            "adaptation_outcome": res,
            "explanation": "Positive completion recorded; confidence in peak productivity window reinforced."
        }

@router.post("/reset")
def reset_system_data():
    """Reset system state to the default student profile for testing."""
    db.reset_to_default()
    return {"status": "success", "message": "System state reset to default Egyptian DEPI student persona."}
