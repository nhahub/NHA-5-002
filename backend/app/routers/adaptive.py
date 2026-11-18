"""Adaptive Personal Model and behavior events router."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

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

@router.get("/model")
def get_personal_model():
    """Return the user's continuously learned Personal Adaptive Model."""
    model = db.get_adaptive_model()
    user = db.get_user()
    return {
        "status": "success",
        "user_id": user.get("id"),
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

@router.post("/reset")
def reset_system_data():
    """Reset to a blank workspace without sample user data."""
    db.reset_to_clean()
    return {"status": "success", "message": "Workspace reset to a clean state."}
