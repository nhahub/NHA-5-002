"""Personal Planning & Dynamic Calendar Router."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime, timezone
import uuid

from backend.app.database import db
from backend.app.core_loop import core_orchestrator
from backend.app.agents.planner import planner_agent

router = APIRouter(prefix="/planning", tags=["Planning & Dynamic Calendar"])

class TaskCreateIn(BaseModel):
    title: str
    estimated_mins: int = Field(default=30, ge=5, le=360)
    topic_ref: Optional[str] = "General"
    priority_weight: float = Field(default=3.0, ge=1.0, le=5.0)
    cognitive_load: Literal["high", "medium", "low"] = "medium"
    deadline: Optional[str] = None
    scheduled_date: Optional[str] = None
    source: Literal["user", "life_coach", "assessment_remediation"] = "user"

class TaskStatusUpdateIn(BaseModel):
    status: Literal["pending", "scheduled", "completed", "delayed", "missed", "cut_short"]
    actual_mins: Optional[int] = None
    notes: Optional[str] = None

class LectureCreateIn(BaseModel):
    day: str
    date: str
    start: str
    end: str
    title: str
    room: Optional[str] = "Main Hall"
    instructor: Optional[str] = ""

class OptimizeRequestIn(BaseModel):
    target_date: Optional[str] = None

@router.get("/schedule")
def get_full_schedule():
    """Retrieve fixed lectures, study tasks, and current AI-optimized calendar blocks."""
    state = db.get_state()
    return {
        "status": "success",
        "university_schedule": state.get("university_schedule", []),
        "tasks": state.get("tasks", []),
        "scheduled_blocks": state.get("scheduled_blocks", []),
        "adaptive_constraints": {
            "peak_productivity_window": state.get("adaptive_model", {}).get("peak_productivity_window", "09:00-11:00"),
            "effective_focus_duration_mins": state.get("adaptive_model", {}).get("effective_focus_duration_mins", 35),
            "task_delay_factor": state.get("adaptive_model", {}).get("task_delay_factor", 1.2),
            "system_mode": state.get("adaptive_model", {}).get("current_system_mode", "NORMAL"),
        }
    }

@router.post("/optimize")
def run_optimization(req: OptimizeRequestIn):
    """Run 5-step deterministic schedule optimization with adaptive constraints."""
    result = core_orchestrator.run_full_schedule_optimization(req.target_date)
    return {"status": "success", "result": result}

@router.post("/tasks")
def create_task(task_in: TaskCreateIn):
    """Create a new study task and add it to the planner."""
    tasks = db.get_tasks()
    model = db.get_adaptive_model()
    delay_factor = float(model.get("task_delay_factor", 1.0))
    adjusted = round(task_in.estimated_mins * delay_factor)

    new_task = {
        "task_id": f"T{len(tasks) + 101}",
        "title": task_in.title,
        "source": task_in.source,
        "topic_ref": task_in.topic_ref,
        "estimated_mins": task_in.estimated_mins,
        "adjusted_mins": adjusted,
        "deadline": task_in.deadline,
        "priority_weight": task_in.priority_weight,
        "cognitive_load": task_in.cognitive_load,
        "status": "pending",
        "scheduled_date": task_in.scheduled_date or datetime.now().strftime("%Y-%m-%d"),
        "rationale": f"Adjusted by {delay_factor}x estimation factor from {task_in.estimated_mins}m to {adjusted}m."
    }
    tasks.append(new_task)
    db.save_tasks(tasks)

    # Ingest event
    db.add_event({
        "source": "Planning Agent",
        "type": "task_created",
        "payload": {"task_id": new_task["task_id"], "title": new_task["title"]}
    })

    return {"status": "success", "task": new_task}

@router.put("/tasks/{task_id}/status")
def update_task_status(task_id: str, update: TaskStatusUpdateIn):
    """Update task status (e.g. completed, delayed, missed, cut_short) and dispatch behavior event."""
    tasks = db.get_tasks()
    target_task = None
    for t in tasks:
        if t.get("task_id") == task_id:
            target_task = t
            t["status"] = update.status
            if update.actual_mins is not None:
                t["actual_mins"] = update.actual_mins
            break

    if not target_task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")

    db.save_tasks(tasks)

    # Map status to adaptive event type
    event_type_map = {
        "completed": "block_completed",
        "cut_short": "block_cut_short",
        "missed": "block_missed",
        "delayed": "block_cut_short"
    }
    ev_type = event_type_map.get(update.status, "task_status_changed")

    # Closed-loop reaction
    loop_res = core_orchestrator.trigger_behavior_event(
        event_type=ev_type,
        source="Planning Agent",
        payload={
            "task_id": task_id,
            "status": update.status,
            "planned_mins": target_task.get("estimated_mins", 30),
            "actual_focus_mins": update.actual_mins or (target_task.get("estimated_mins", 30) if update.status == "completed" else 15),
            "topic": target_task.get("topic_ref")
        },
        auto_adapt=True
    )

    return {"status": "success", "task": target_task, "closed_loop": loop_res}

@router.post("/tasks/{task_id}/deconstruct")
def deconstruct_task_endpoint(task_id: str):
    """Deconstruct an oversized task into micro-tasks matching the student's attention span."""
    tasks = db.get_tasks()
    target_task = None
    for t in tasks:
        if t.get("task_id") == task_id:
            target_task = t
            break

    if not target_task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")

    model = db.get_adaptive_model()
    focus_mins = int(model.get("effective_focus_duration_mins", 35))
    micros = planner_agent.deconstruct(target_task, focus_duration_mins=focus_mins)

    return {
        "status": "success",
        "parent_task_id": task_id,
        "effective_focus_duration": focus_mins,
        "micro_tasks": micros
    }

@router.post("/lectures")
def add_lecture(lec: LectureCreateIn):
    """Add a new fixed lecture to the university schedule."""
    schedule = db.get_university_schedule()
    new_lec = {
        "id": f"lec-{uuid.uuid4().hex[:6]}",
        "day": lec.day,
        "date": lec.date,
        "start": lec.start,
        "end": lec.end,
        "title": lec.title,
        "room": lec.room,
        "instructor": lec.instructor,
    }
    schedule.append(new_lec)
    db.save_university_schedule(schedule)
    return {"status": "success", "lecture": new_lec}
