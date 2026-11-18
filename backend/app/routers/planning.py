"""Personal Planning & Dynamic Calendar Router."""

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import uuid

from backend.app.database import db
from backend.app.core_loop import core_orchestrator
from backend.app.agents.planner import planner_agent
from backend.app.services.goal_progress import sync_goal_progress_for_task

router = APIRouter(prefix="/planning", tags=["Planning & Dynamic Calendar"])

class TimetableConfirmIn(BaseModel):
    entries: List[Dict[str, str]] = Field(..., min_length=1, max_length=60)

def _next_weekday_date(day: str) -> str:
    weekday_index = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                     "Friday": 4, "Saturday": 5, "Sunday": 6}[day]
    today_cairo = datetime.now(ZoneInfo("Africa/Cairo")).date()
    return (today_cairo + timedelta(days=(weekday_index - today_cairo.weekday()) % 7)).isoformat()

@router.post("/schedule-image/parse")
async def parse_schedule_image(file: UploadFile = File(...)):
    """Read an uploaded weekly schedule image into a draft for user review."""
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="ارفع صورة JPG أو PNG أو WEBP واضحة.")
    image = await file.read(12 * 1024 * 1024 + 1)
    if len(image) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="حجم الصورة يجب ألا يتجاوز 12 ميجابايت.")
    try:
        draft = planner_agent.parse_timetable_image(image, file.content_type)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        import logging
        logging.getLogger("adaptive_coach.planning").exception("Could not parse schedule image")
        raise HTTPException(status_code=502, detail="لم أستطع قراءة الصورة بوضوح. جرّب صورة أوضح أو أدخل المواعيد يدويًا.") from exc
    return {"status": "success", "draft": draft.model_dump(), "requires_confirmation": True}

@router.post("/schedule-image/confirm")
def confirm_schedule_image(payload: TimetableConfirmIn):
    """Persist user-approved image entries as recurring weekly commitments."""
    from app.engine.schedule_image import WEEKDAYS
    schedule = db.get_university_schedule()
    added, duplicates = [], 0
    for raw in payload.entries:
        day_raw = (raw.get("day") or "").strip()
        day = WEEKDAYS.get(day_raw.lower()) or WEEKDAYS.get(day_raw)
        title = (raw.get("title") or "").strip()
        start, end = (raw.get("start") or "").strip(), (raw.get("end") or "").strip()
        if not day or not title or len(title) > 180:
            raise HTTPException(status_code=422, detail="راجع اليوم واسم كل محاضرة.")
        try:
            start_dt, end_dt = datetime.strptime(start, "%H:%M"), datetime.strptime(end, "%H:%M")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="راجع أوقات المحاضرات بصيغة 24 ساعة.") from exc
        if end_dt <= start_dt:
            raise HTTPException(status_code=422, detail="وقت نهاية المحاضرة يجب أن يأتي بعد بدايتها.")
        signature = (day, title.casefold(), start, end)
        if any((item.get("day"), (item.get("title") or "").casefold(), item.get("start"), item.get("end")) == signature
               for item in schedule):
            duplicates += 1
            continue
        item = {
            "id": f"lec-{uuid.uuid4().hex[:10]}", "day": day,
            "date": _next_weekday_date(day), "start": start, "end": end,
            "title": title, "room": (raw.get("room") or "").strip(),
            "instructor": "", "category": "class", "recurring": "weekly",
            "source": "user_confirmed_schedule_image",
        }
        schedule.append(item)
        added.append(item)
    db.save_university_schedule(schedule)
    plan = core_orchestrator.run_full_schedule_optimization(datetime.now(ZoneInfo("Africa/Cairo")).date().isoformat())
    return {"status": "success", "added": added, "duplicate_count": duplicates, "plan": plan}

def _refresh_user_plan(target_date: Optional[str]) -> Dict[str, Any]:
    """Rebuild time blocks after a user input changes their available plan."""
    try:
        today = datetime.now(ZoneInfo("Africa/Cairo")).date()
        requested = datetime.strptime((target_date or today.isoformat())[:10], "%Y-%m-%d").date()
        start = requested if today <= requested <= today + timedelta(days=6) else today
        return core_orchestrator.run_full_schedule_optimization(start.isoformat())
    except Exception as exc:
        import logging
        logging.getLogger("adaptive_coach.planning").exception("Plan refresh failed: %s", exc)
        return {"status": "unavailable", "reason": "plan_refresh_failed"}

class TaskCreateIn(BaseModel):
    title: str
    estimated_mins: int = Field(default=30, ge=5, le=360)
    topic_ref: Optional[str] = "General"
    priority_weight: float = Field(default=3.0, ge=1.0, le=5.0)
    cognitive_load: Literal["high", "medium", "low"] = "medium"
    deadline: Optional[str] = None
    scheduled_date: Optional[str] = None
    source: Literal["user", "life_coach", "assessment_remediation"] = "user"
    category: str = "study"
    goal_id: Optional[str] = None
    parent_milestone_id: Optional[str] = None
    course_id: Optional[str] = None
    lecture_id: Optional[str] = None
    task_kind: Literal["task", "assessment"] = "task"
    notes: str = ""
    recurring: Optional[str] = None

class TaskUpdateIn(BaseModel):
    title: Optional[str] = None
    estimated_mins: Optional[int] = Field(default=None, ge=5, le=360)
    topic_ref: Optional[str] = None
    priority_weight: Optional[float] = Field(default=None, ge=1.0, le=5.0)
    deadline: Optional[str] = None
    scheduled_date: Optional[str] = None
    category: Optional[str] = None
    goal_id: Optional[str] = None
    parent_milestone_id: Optional[str] = None
    course_id: Optional[str] = None
    lecture_id: Optional[str] = None
    task_kind: Optional[Literal["task", "assessment"]] = None
    mastery_target: Optional[float] = Field(default=None, ge=0, le=100)
    notes: Optional[str] = None
    recurring: Optional[str] = None

class TaskStatusUpdateIn(BaseModel):
    status: Literal["pending", "scheduled", "completed", "delayed", "missed", "cut_short"]
    actual_mins: Optional[int] = Field(default=None, ge=1, le=720)
    planned_mins: Optional[int] = Field(default=None, ge=1, le=720)
    notes: Optional[str] = None

class FocusSessionIn(BaseModel):
    actual_mins: int = Field(..., ge=1, le=720)
    planned_mins: int = Field(..., ge=1, le=720)

class LectureCreateIn(BaseModel):
    day: str
    date: str
    start: str
    end: str
    title: str
    room: Optional[str] = "Main Hall"
    instructor: Optional[str] = ""

class CalendarEventIn(BaseModel):
    title: str
    date: str
    start: str
    end: str
    category: Literal["class", "exam", "personal", "work", "appointment", "deadline", "break"] = "personal"
    notes: str = ""
    location: str = ""
    goal_id: Optional[str] = None
    recurring: Optional[Literal["weekly"]] = None

class OptimizeRequestIn(BaseModel):
    target_date: Optional[str] = None

class ReconcileScheduleIn(BaseModel):
    active_task_id: Optional[str] = None

@router.get("/schedule")
def get_full_schedule():
    """Retrieve fixed lectures, study tasks, and current AI-optimized calendar blocks."""
    state = db.get_state()
    return {
        "status": "success",
        "university_schedule": state.get("university_schedule", []),
        "calendar_events": state.get("calendar_events", []),
        "tasks": state.get("tasks", []),
        "scheduled_blocks": state.get("scheduled_blocks", []),
        "adaptive_constraints": {
            "peak_productivity_window": state.get("adaptive_model", {}).get("peak_productivity_window", ""),
            "effective_focus_duration_mins": state.get("adaptive_model", {}).get("effective_focus_duration_mins", 35),
            "task_delay_factor": state.get("adaptive_model", {}).get("task_delay_factor", 1.2),
            "day_start": db.get_user().get("day_start"),
            "day_end": db.get_user().get("day_end"),
            "system_mode": state.get("adaptive_model", {}).get("current_system_mode", "NORMAL"),
        }
    }

@router.post("/reconcile")
def reconcile_schedule(payload: ReconcileScheduleIn):
    """Move expired, unfinished focus blocks into the next available plan window."""
    now = datetime.now(ZoneInfo("Africa/Cairo"))
    today, now_hhmm = now.date().isoformat(), now.strftime("%H:%M")
    tasks = db.get_tasks()
    task_by_id = {task.get("task_id"): task for task in tasks}
    expired_by_task: Dict[str, List[Dict[str, Any]]] = {}
    for block in db.get_state().get("scheduled_blocks", []):
        task_id = block.get("task_id")
        task = task_by_id.get(task_id)
        if (not task or task.get("status") in {"completed", "missed"}
                or task_id == payload.active_task_id):
            continue
        block_date, block_end = (block.get("date") or "")[:10], block.get("end") or "00:00"
        if block_date < today or (block_date == today and block_end <= now_hhmm):
            expired_by_task.setdefault(task_id, []).append(block)
    if not expired_by_task:
        return {"status": "success", "changed": False, "rescheduled_tasks": []}

    for task_id, expired_blocks in expired_by_task.items():
        task = task_by_id[task_id]
        planned_mins = sum(
            max(0, int(block.get("end", "00:00")[:2]) * 60 + int(block.get("end", "00:00")[3:5])
                - int(block.get("start", "00:00")[:2]) * 60 - int(block.get("start", "00:00")[3:5]))
            for block in expired_blocks
        )
        task["last_session_status"] = "missed"
        task["last_missed_at"] = now.isoformat()
        task["rationale"] = "فات موعد الجلسة ولم تُسجل إنجازها؛ أعدت ترتيبها في أقرب وقت متاح."
        core_orchestrator.trigger_behavior_event(
            event_type="block_missed", source="Personal Planning Agent",
            payload={"task_id": task_id, "planned_mins": planned_mins or task.get("estimated_mins", 30),
                     "actual_focus_mins": None, "topic": task.get("topic_ref"),
                     "category": task.get("category", "study"), "day_type": "normal"},
            auto_adapt=False,
        )
    db.save_tasks(tasks)
    plan = core_orchestrator.run_full_schedule_optimization(today)
    return {"status": "success", "changed": True,
            "rescheduled_tasks": list(expired_by_task), "plan": plan}

@router.post("/optimize")
def run_optimization(req: OptimizeRequestIn):
    """Run 5-step deterministic schedule optimization with adaptive constraints."""
    result = core_orchestrator.run_full_schedule_optimization(req.target_date)
    return {"status": "success", "result": result}

@router.post("/tasks")
def create_task(task_in: TaskCreateIn):
    """Create a new study task and add it to the planner."""
    linked_course = next((course for course in db.get_courses()
                          if course.get("id") == task_in.course_id), None) if task_in.course_id else None
    linked_material = next((material for material in (linked_course or {}).get("materials", [])
                            if material.get("id") == task_in.lecture_id), None) if task_in.lecture_id else None
    if task_in.course_id and not linked_course:
        raise HTTPException(status_code=404, detail="المادة المحددة غير موجودة.")
    if task_in.lecture_id and not linked_material:
        raise HTTPException(status_code=404, detail="المحاضرة المحددة غير موجودة داخل المادة.")
    if bool(task_in.course_id) != bool(task_in.lecture_id):
        raise HTTPException(status_code=422, detail="اختَر المادة والمحاضرة معًا لربط المهمة بمحتواك.")
    has_study_materials = any(course.get("materials") for course in db.get_courses())
    if task_in.category == "study" and has_study_materials and not linked_material:
        raise HTTPException(status_code=422, detail="اختَر المادة والمحاضرة المرتبطتين بمهمة الدراسة.")
    if task_in.task_kind == "assessment" and not linked_material:
        raise HTTPException(status_code=422, detail="مهمة الاختبار تحتاج مادة ومحاضرة مرفوعتين.")
    linked_goal = next((goal for goal in db.get_goals() if goal.get("goal_id") == task_in.goal_id), None) if task_in.goal_id else None
    if task_in.goal_id and not linked_goal:
        raise HTTPException(status_code=404, detail="الهدف المرتبط بالمهمة غير موجود.")

    tasks = db.get_tasks()
    model = db.get_adaptive_model()
    delay_factor = float(model.get("task_delay_factor", 1.0))
    adjusted = round(task_in.estimated_mins * delay_factor)
    parent_milestone_id = task_in.parent_milestone_id
    if task_in.goal_id and not parent_milestone_id:
        linked_goal = next((g for g in db.get_goals() if g.get("goal_id") == task_in.goal_id), None)
        next_milestone = next((m for m in (linked_goal or {}).get("milestones", [])
                               if m.get("status") != "completed"), None)
        parent_milestone_id = next_milestone.get("id") if next_milestone else None

    new_task = {
        "task_id": f"T-{uuid.uuid4().hex[:10]}",
        "title": task_in.title,
        "source": task_in.source,
        "topic_ref": task_in.topic_ref,
        "estimated_mins": task_in.estimated_mins,
        "adjusted_mins": adjusted,
        "deadline": task_in.deadline,
        "priority_weight": task_in.priority_weight,
        "cognitive_load": task_in.cognitive_load,
        "status": "pending",
        "scheduled_date": task_in.scheduled_date or None,
        "schedule_locked": bool(task_in.scheduled_date),
        "schedule_mode": "manual" if task_in.scheduled_date else "adaptive",
        "rationale": f"Adjusted by {delay_factor}x estimation factor from {task_in.estimated_mins}m to {adjusted}m.",
        "category": task_in.category,
        "goal_id": task_in.goal_id,
        "mastery_target": linked_goal.get("target_score") if linked_goal else None,
        "parent_milestone_id": parent_milestone_id,
        "course_id": task_in.course_id,
        "lecture_id": task_in.lecture_id,
        "task_kind": task_in.task_kind,
        "notes": task_in.notes,
        "recurring": task_in.recurring
    }
    tasks.append(new_task)
    db.save_tasks(tasks)

    core_orchestrator.trigger_behavior_event("task_created", "Planning Agent", {"task_id": new_task["task_id"]})
    plan = _refresh_user_plan(new_task["scheduled_date"])

    return {"status": "success", "task": new_task, "plan": plan}

@router.put("/tasks/{task_id}")
def edit_task(task_id: str, update: TaskUpdateIn):
    tasks = db.get_tasks()
    task = next((t for t in tasks if t.get("task_id") == task_id), None)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    changes = update.model_dump(exclude_unset=True)
    if changes.get("goal_id") and not any(goal.get("goal_id") == changes["goal_id"] for goal in db.get_goals()):
        raise HTTPException(status_code=404, detail="الهدف المرتبط بالمهمة غير موجود.")
    if "course_id" in changes or "lecture_id" in changes:
        course_id = changes.get("course_id", task.get("course_id"))
        lecture_id = changes.get("lecture_id", task.get("lecture_id"))
        linked_course = next((course for course in db.get_courses() if course.get("id") == course_id), None)
        linked_material = next((material for material in (linked_course or {}).get("materials", [])
                                if material.get("id") == lecture_id), None)
        if bool(course_id) != bool(lecture_id) or (course_id and not linked_material):
            raise HTTPException(status_code=422, detail="اختَر محاضرة صالحة من المادة المرتبطة بالمهمة.")
    if "goal_id" in changes and "parent_milestone_id" not in changes:
        linked_goal = next((goal for goal in db.get_goals() if goal.get("goal_id") == changes["goal_id"]), None)
        next_milestone = next((item for item in (linked_goal or {}).get("milestones", [])
                               if item.get("status") != "completed"), None)
        changes["parent_milestone_id"] = next_milestone.get("id") if next_milestone else None
        changes["mastery_target"] = linked_goal.get("target_score") if linked_goal else None
    if "parent_milestone_id" in changes:
        changes["parent_milestone_id"] = changes["parent_milestone_id"] or None
    if "scheduled_date" in changes:
        changes["scheduled_date"] = changes["scheduled_date"] or None
        changes["schedule_locked"] = bool(changes["scheduled_date"])
        changes["schedule_mode"] = "manual" if changes["scheduled_date"] else "adaptive"
    if "estimated_mins" in changes:
        changes["adjusted_mins"] = round(changes["estimated_mins"] * float(db.get_adaptive_model().get("task_delay_factor", 1.0)))
    task.update(changes)
    db.save_tasks(tasks)
    db.get_state()["scheduled_blocks"] = [b for b in db.get_state().get("scheduled_blocks", []) if b.get("task_id") != task_id]
    db.save()
    plan = _refresh_user_plan(task.get("scheduled_date"))
    return {"status": "success", "task": task, "plan": plan}

@router.delete("/tasks/{task_id}")
def delete_task(task_id: str):
    tasks = db.get_tasks()
    removed = next((t for t in tasks if t.get("task_id") == task_id), None)
    if not removed:
        raise HTTPException(status_code=404, detail="Task not found")
    # Remove generated assessments and decomposed subtasks with their source.
    removed_ids = {task_id}
    while True:
        children = {t.get("task_id") for t in tasks
                    if t.get("source_task_id") in removed_ids
                    or t.get("parent_task_id") in removed_ids}
        if children.issubset(removed_ids):
            break
        removed_ids.update(children)
    remaining = [t for t in tasks if t.get("task_id") not in removed_ids]
    db.save_tasks(remaining)
    state = db.get_state()
    state["scheduled_blocks"] = [b for b in state.get("scheduled_blocks", []) if b.get("task_id") not in removed_ids]
    db.save()
    plan = _refresh_user_plan(removed.get("scheduled_date"))
    return {"status": "success", "plan": plan}

@router.post("/events")
def create_calendar_event(event_in: CalendarEventIn):
    events = db.get_state().setdefault("calendar_events", [])
    event = {"id": f"evt-{uuid.uuid4().hex[:10]}", **event_in.model_dump()}
    events.append(event)
    db.save()
    core_orchestrator.trigger_behavior_event("commitment_created", "Planning Agent", {"event_id": event["id"]})
    plan = _refresh_user_plan(event.get("date"))
    return {"status": "success", "event": event, "plan": plan}

@router.put("/events/{event_id}")
def edit_calendar_event(event_id: str, update: CalendarEventIn):
    events = db.get_state().setdefault("calendar_events", [])
    event = next((e for e in events if e.get("id") == event_id), None)
    if event:
        event.update(update.model_dump())
        db.save()
        plan = _refresh_user_plan(event.get("date"))
        return {"status": "success", "event": event, "plan": plan}
    schedule = db.get_university_schedule()
    event = next((e for e in schedule if e.get("id") == event_id), None)
    if not event:
        raise HTTPException(status_code=404, detail="Calendar event not found")
    event.update({**update.model_dump(), "day": datetime.strptime(update.date, "%Y-%m-%d").strftime("%A"), "room": update.location})
    db.save_university_schedule(schedule)
    plan = _refresh_user_plan(update.date)
    return {"status": "success", "event": event, "plan": plan}

@router.delete("/events/{event_id}")
def delete_calendar_event(event_id: str):
    events = db.get_state().setdefault("calendar_events", [])
    remaining = [e for e in events if e.get("id") != event_id]
    if len(remaining) != len(events):
        removed = next(e for e in events if e.get("id") == event_id)
        db.get_state()["calendar_events"] = remaining
        db.save()
        plan = _refresh_user_plan(removed.get("date"))
        return {"status": "success", "plan": plan}
    schedule = db.get_university_schedule()
    remaining = [e for e in schedule if e.get("id") != event_id]
    if len(remaining) == len(schedule):
        raise HTTPException(status_code=404, detail="Calendar event not found")
    removed = next(e for e in schedule if e.get("id") == event_id)
    db.save_university_schedule(remaining)
    plan = _refresh_user_plan(removed.get("date"))
    return {"status": "success", "plan": plan}

@router.get("/next-action")
def get_next_action(date: Optional[str] = None):
    tasks = [t for t in db.get_tasks() if t.get("status") not in {"completed", "missed"}]
    if not tasks:
        return {"status": "success", "action": None, "reason": "لا توجد مهام بعد. أضف مهمة أو هدفًا ليقترح النظام الخطوة التالية."}
    target_date = date or datetime.now().strftime("%Y-%m-%d")
    def rank(task):
        due = task.get("deadline") or task.get("scheduled_date") or "9999-12-31"
        overdue = due[:10] < target_date
        priority = float(task.get("priority_weight", 3))
        return (0 if overdue else 1, due[:10], -priority, int(task.get("estimated_mins", 30)))
    action = min(tasks, key=rank)
    due = action.get("deadline") or action.get("scheduled_date")
    reason = f"اخترت المهمة حسب الأولوية وموعدها{f' ({due[:10]})' if due else ''}. يمكنك تعديل الخطة في أي وقت."
    return {"status": "success", "action": action, "reason": reason}

@router.put("/tasks/{task_id}/status")
def update_task_status(task_id: str, update: TaskStatusUpdateIn):
    """Update task status (e.g. completed, delayed, missed, cut_short) and dispatch behavior event."""
    tasks = db.get_tasks()
    target_task = None
    previous_status = None
    for t in tasks:
        if t.get("task_id") == task_id:
            target_task = t
            previous_status = t.get("status")
            if update.status == "completed" and t.get("task_kind") == "assessment":
                raise HTTPException(status_code=409, detail="أكمل الاختبار المرتبط بالمهمة لتسجيلها كمنجزة.")
            if update.status in {"cut_short", "delayed"}:
                t["last_session_status"] = update.status
                t["status"] = "pending"
            else:
                t["status"] = update.status
            if update.actual_mins is not None:
                t["actual_mins"] = int(t.get("actual_mins", 0)) + update.actual_mins
                if update.status != "completed":
                    t["remaining_mins"] = max(5, int(t.get("estimated_mins", 30)) - t["actual_mins"])
                else:
                    t["remaining_mins"] = 0
            break

    if not target_task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")

    assessment_task = None
    if (update.status == "completed" and target_task.get("task_kind") != "assessment"
            and not target_task.get("exam_plan")
            and target_task.get("category") == "study"
            and target_task.get("course_id") and target_task.get("lecture_id")):
        already_created = any(t.get("source_task_id") == task_id for t in tasks)
        if not already_created:
            assessment_task = {
                "task_id": f"T-{uuid.uuid4().hex[:10]}",
                "title": f"اختبر فهمك: {target_task.get('topic_ref') or target_task.get('title')}",
                "source": "assessment_remediation", "task_kind": "assessment",
                "source_task_id": task_id,
                "topic_ref": target_task.get("topic_ref"),
                "estimated_mins": 15, "adjusted_mins": 15,
                "deadline": target_task.get("deadline"), "priority_weight": 4.0,
                "cognitive_load": "medium", "status": "pending",
                "scheduled_date": None, "schedule_locked": False, "schedule_mode": "adaptive",
                "category": "study", "goal_id": target_task.get("goal_id"),
                "parent_milestone_id": target_task.get("parent_milestone_id"),
                "course_id": target_task.get("course_id"), "lecture_id": target_task.get("lecture_id"),
                "mastery_target": target_task.get("mastery_target"),
                "rationale": "اختبار قصير من المحاضرة بعد جلسة المذاكرة لقياس الفهم وتحديث خريطة المعرفة.",
            }
            tasks.append(assessment_task)
    db.save_tasks(tasks)
    sync_goal_progress_for_task(target_task, tasks, db)

    next_occurrence = None
    if update.status == "completed" and previous_status != "completed" and target_task.get("recurring"):
        scheduled = datetime.strptime(target_task.get("scheduled_date") or datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d").date()
        cadence = target_task["recurring"]
        if cadence == "daily":
            next_date = scheduled + timedelta(days=1)
        elif cadence == "weekly":
            next_date = scheduled + timedelta(days=7)
        else:
            next_month = (scheduled.replace(day=28) + timedelta(days=4)).replace(day=1)
            last_day = (next_month.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            next_date = next_month.replace(day=min(scheduled.day, last_day.day))
        next_occurrence = dict(target_task)
        next_occurrence.update({
            "task_id": f"T-{uuid.uuid4().hex[:10]}",
            "status": "pending",
            "scheduled_date": next_date.isoformat(),
            "actual_mins": None,
            "rationale": f"تكرار {cadence} للمهمة.",
        })
        tasks.append(next_occurrence)
        db.save_tasks(tasks)

    # Map status to adaptive event type
    event_type_map = {
        "completed": "block_completed",
        "cut_short": "block_cut_short",
        "missed": "block_missed",
        "delayed": "block_cut_short"
    }
    ev_type = event_type_map.get(update.status, "task_status_changed")
    target_date = (target_task.get("scheduled_date") or datetime.now().strftime("%Y-%m-%d"))[:10]
    commitments = db.get_university_schedule() + db.get_state().get("calendar_events", [])
    class_minutes = 0
    for item in commitments:
        item_date = (item.get("date") or "")[:10]
        occurs_today = item_date == target_date
        if item.get("recurring") == "weekly" and item_date:
            try:
                occurs_today = target_date >= item_date and datetime.strptime(target_date, "%Y-%m-%d").weekday() == datetime.strptime(item_date, "%Y-%m-%d").weekday()
            except ValueError:
                occurs_today = False
        if not occurs_today or not item.get("start") or not item.get("end"):
            continue
        if item.get("category", "class") not in {"class", "work"}:
            continue
        try:
            start = datetime.strptime(item["start"], "%H:%M")
            end = datetime.strptime(item["end"], "%H:%M")
            class_minutes += max(0, int((end - start).total_seconds() // 60))
        except ValueError:
            continue

    # Closed-loop reaction
    loop_res = core_orchestrator.trigger_behavior_event(
        event_type=ev_type,
        source="Planning Agent",
        payload={
            "task_id": task_id,
            "status": update.status,
            "planned_mins": update.planned_mins or target_task.get("estimated_mins", 30),
            "actual_focus_mins": update.actual_mins,
            "topic": target_task.get("topic_ref"),
            "course_id": target_task.get("course_id"),
            "lecture_id": target_task.get("lecture_id"),
            "goal_id": target_task.get("goal_id"),
            "category": target_task.get("category", "study"),
            "day_type": "heavy_lecture" if class_minutes >= 240 else "normal"
        },
        auto_adapt=True
    )
    if update.status == "completed" and target_task.get("actual_mins"):
        loop_res = core_orchestrator.trigger_behavior_event(
            event_type="task_completed", source="Personal Planning Agent",
            payload={"task_id": task_id, "topic": target_task.get("topic_ref"),
                     "course_id": target_task.get("course_id"),
                     "lecture_id": target_task.get("lecture_id"),
                     "goal_id": target_task.get("goal_id"),
                     "planned_mins": target_task.get("estimated_mins", 30),
                     "actual_mins": target_task["actual_mins"],
                     "category": target_task.get("category", "study")},
            auto_adapt=False,
        )
    plan = _refresh_user_plan(target_date)

    return {"status": "success", "task": target_task, "assessment_task": assessment_task,
            "next_occurrence": next_occurrence, "closed_loop": loop_res, "plan": plan}

@router.post("/tasks/{task_id}/session")
def record_focus_session(task_id: str, session: FocusSessionIn):
    """Record one completed focus block while keeping its parent task open."""
    task = next((item for item in db.get_tasks() if item.get("task_id") == task_id), None)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.get("status") in {"completed", "missed"}:
        raise HTTPException(status_code=409, detail="المهمة لم تعد مفتوحة لتسجيل جلسة جديدة.")
    task["actual_mins"] = int(task.get("actual_mins", 0)) + session.actual_mins
    task["remaining_mins"] = max(5, int(task.get("estimated_mins", 30)) - task["actual_mins"])
    task["last_session_status"] = "completed"
    db.save_tasks(db.get_tasks())
    loop = core_orchestrator.trigger_behavior_event(
        event_type="block_completed", source="Personal Planning Agent",
        payload={"task_id": task_id, "status": "session_completed",
                 "planned_mins": session.planned_mins, "actual_focus_mins": session.actual_mins,
                 "topic": task.get("topic_ref"), "category": task.get("category", "study"),
                 "day_type": "normal"},
        auto_adapt=False,
    )
    plan = _refresh_user_plan(datetime.now(ZoneInfo("Africa/Cairo")).date().isoformat())
    return {"status": "success", "task": task, "closed_loop": loop, "plan": plan}

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
