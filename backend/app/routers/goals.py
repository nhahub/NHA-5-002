"""Goal & Life Coach Router (Accountability, Reflections & Gamification)."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional
from datetime import date, datetime
from zoneinfo import ZoneInfo
import uuid
from pathlib import Path

from backend.app.database import db
from backend.app.core_loop import core_orchestrator
from backend.app.agents.goal_coach import goal_coach
from backend.app.services.exam_plan import build_exam_plan

router = APIRouter(prefix="/goals", tags=["Goal & Life Coach"])

def _today() -> str:
    return datetime.now(ZoneInfo("Africa/Cairo")).date().isoformat()

class GoalCreateIn(BaseModel):
    title: str
    category: Literal["study", "career", "personal"] = "study"
    target_date: Optional[str] = None
    why: str = ""
    description: str = ""
    milestones: Optional[List[Dict[str, Any]]] = None
    course_id: Optional[str] = None
    lecture_ids: List[str] = Field(default_factory=list)
    target_score: Optional[float] = Field(default=None, ge=0, le=100)
    has_exam: bool = False
    exam_date: Optional[str] = None  # YYYY-MM-DD, required when has_exam is true

class ExamDateIn(BaseModel):
    exam_date: str

class ReflectionIn(BaseModel):
    energy: int = Field(..., ge=1, le=5, description="Energy rating from 1 (exhausted) to 5 (peak)")
    mood: str = Field(..., description="e.g. focused, tired, motivated, anxious, steady")
    reflection: str = Field(..., description="Qualitative self-reflection note")
    obstacles: Optional[str] = ""

class HabitCheckIn(BaseModel):
    habit_name: str

class HabitCreateIn(BaseModel):
    name: str

@router.get("/")
def get_goals_and_coaching():
    """Retrieve user goals, SMART milestones, habit streaks, shields, and badges."""
    state = db.get_state()
    model = db.get_adaptive_model()
    today = datetime.now(ZoneInfo("Africa/Cairo")).date().isoformat()
    tasks = db.get_tasks()
    blocks = state.get("scheduled_blocks", [])
    goals_with_actions = []
    for goal in state.get("goals", []):
        goal_tasks = [t for t in tasks if t.get("goal_id") == goal.get("goal_id")
                      and t.get("status") not in {"completed", "missed"}]
        goal_blocks = [b for b in blocks if b.get("task_id") in {t.get("task_id") for t in goal_tasks}]
        block_by_task = {}
        for block in goal_blocks:
            block_by_task.setdefault(block.get("task_id"), []).append(block)
        for task_blocks in block_by_task.values():
            task_blocks.sort(key=lambda b: ((b.get("date") or ""), (b.get("start") or "")))

        today_tasks = [t for t in goal_tasks
                       if (not t.get("scheduled_date")
                           or (t.get("scheduled_date") or "")[:10] <= today
                           or (t.get("deadline") or "")[:10] <= today)]
        if today_tasks:
            today_tasks.sort(key=lambda t: (
                next((b.get("start", "99:99") for b in block_by_task.get(t.get("task_id"), [])
                      if (b.get("date") or "")[:10] == today), "99:99"),
                (t.get("deadline") or "9999-12-31")[:10],
                -(float(t.get("priority_weight", 3) or 3)),
            ))
            action_task = today_tasks[0]
            action_day = "today"
        elif goal_tasks:
            goal_tasks.sort(key=lambda t: (
                (t.get("scheduled_date") or t.get("deadline") or "9999-12-31")[:10],
                (t.get("deadline") or "9999-12-31")[:10],
                -(float(t.get("priority_weight", 3) or 3)),
            ))
            action_task = goal_tasks[0]
            action_day = "upcoming"
        else:
            action_task = None
            action_day = "needs_task" if any(m.get("status") != "completed" for m in goal.get("milestones", [])) else "complete"
        next_milestone = next((m for m in goal.get("milestones", []) if m.get("status") != "completed"), None)

        action_block = next((b for b in block_by_task.get(action_task.get("task_id"), [])
                             if (b.get("date") or "")[:10] == today), None) if action_task else None
        milestone = next((m for m in goal.get("milestones", [])
                          if m.get("id") == action_task.get("parent_milestone_id")), None) if action_task else None
        action_date = None
        if action_task:
            action_date = action_task.get("scheduled_date") or (action_block or {}).get("date")
        action_plan = {
            "status": action_day,
            "date": action_date,
            "task": action_task,
            "start": action_block.get("start") if action_block else None,
            "duration_mins": action_block.get("duration_mins") if action_block else (action_task.get("estimated_mins") if action_task else None),
            "reason": (f"الخطوة التالية في مرحلة {milestone.get('title')} ضمن هدفك." if milestone else "أقرب مهمة مفتوحة مرتبطة بهذا الهدف." ) if action_task else None,
            "milestone_id": milestone.get("id") if milestone else (next_milestone.get("id") if next_milestone else None),
            "milestone_title": milestone.get("title") if milestone else (next_milestone.get("title") if next_milestone else None),
        }
        goals_with_actions.append({**goal, "action_plan": action_plan})

    today_actions = sorted(
        (g["action_plan"] | {"goal_id": g.get("goal_id"), "goal_title": g.get("title"), "why": g.get("why", "")}
         for g in goals_with_actions if g["action_plan"]["status"] == "today"),
        key=lambda a: (a.get("start") or "99:99", (a.get("task", {}).get("deadline") or "9999-12-31")[:10]),
    )[:5]
    return {
        "status": "success",
        "goals": goals_with_actions,
        "today_actions": today_actions,
        "habits": state.get("habits", []),
        "achievements": state.get("achievements", []),
        "reflections": state.get("reflections", []),
        "accountability_metrics": {
            "burnout_risk_score": model.get("burnout_risk_score", "Low"),
            "motivation_index": model.get("motivation_index", 0.75),
            "freezes_summary": sum(h.get("freezes_available", 0) for h in state.get("habits", [])),
        }
    }


def _parse_exam_date(value: Optional[str]) -> date:
    today = datetime.now(ZoneInfo("Africa/Cairo")).date()
    try:
        exam_date = date.fromisoformat((value or "")[:10])
    except ValueError:
        raise HTTPException(status_code=422, detail="اكتب تاريخ الامتحان بصيغة صحيحة (YYYY-MM-DD).")
    if exam_date <= today:
        raise HTTPException(status_code=422, detail="موعد الامتحان لازم يكون بعد النهاردة.")
    return exam_date


def _exam_plan_tasks(goal: Dict[str, Any], lectures: List[Dict[str, Any]], exam_date: date,
                     milestone_id: Optional[str]) -> Dict[str, Any]:
    model = db.get_adaptive_model()
    today = datetime.now(ZoneInfo("Africa/Cairo")).date()
    return build_exam_plan(
        goal_id=goal["goal_id"], course_id=goal.get("course_id"), lectures=lectures,
        exam_date=exam_date, today=today,
        focus_mins=int(model.get("effective_focus_duration_mins", 35) or 35),
        daily_cap_mins=int(model.get("max_daily_study_mins", 240) or 240),
        milestone_id=milestone_id, mastery_target=goal.get("target_score"),
    )

@router.post("/")
def create_goal(goal_in: GoalCreateIn):
    """Create a new long-term goal decomposed into milestones."""
    linked_course = next((course for course in db.get_courses()
                          if course.get("id") == goal_in.course_id), None) if goal_in.course_id else None
    if goal_in.course_id and not linked_course:
        raise HTTPException(status_code=404, detail="المادة المرتبطة بالهدف غير موجودة.")
    materials_by_id = {material.get("id"): material for material in (linked_course or {}).get("materials", [])}
    selected_lectures = [materials_by_id[material_id] for material_id in goal_in.lecture_ids
                         if material_id in materials_by_id]
    if goal_in.lecture_ids and len(selected_lectures) != len(set(goal_in.lecture_ids)):
        raise HTTPException(status_code=422, detail="بعض المحاضرات المختارة لا تتبع المادة المحددة.")
    if goal_in.lecture_ids and not goal_in.course_id:
        raise HTTPException(status_code=422, detail="اختر المادة قبل ربط المحاضرات بالهدف.")
    if goal_in.course_id and (linked_course or {}).get("materials") and not selected_lectures:
        raise HTTPException(status_code=422, detail="اختر محاضرة واحدة على الأقل لبناء خطة الهدف عليها.")
    exam_date: Optional[date] = None
    if goal_in.has_exam:
        if not goal_in.exam_date:
            raise HTTPException(status_code=422, detail="حدد موعد الامتحان عشان أوزع المذاكرة قبله.")
        if not selected_lectures:
            raise HTTPException(status_code=422, detail="اختر المادة والمحاضرات اللي هتتمتحن فيها عشان أبني خطة الامتحان.")
        exam_date = _parse_exam_date(goal_in.exam_date)
    model = db.get_adaptive_model()
    learned_context = []
    if model.get("focus_sessions_count", 0) >= 3:
        learned_context.append(f"observed focus duration: {model.get('effective_focus_duration_mins')} minutes")
    if model.get("peak_productivity_window"):
        learned_context.append(f"observed productive window: {model['peak_productivity_window']}")
    if model.get("weak_topics"):
        learned_context.append("recent learning gaps: " + ", ".join(model["weak_topics"][:5]))
    if linked_course and selected_lectures:
        lecture_titles = ", ".join(material.get("title", "محاضرة") for material in selected_lectures)
        learned_context.append(f"goal course: {linked_course.get('name')}; user-selected lecture roadmap: {lecture_titles}; build the milestones around understanding and assessing these exact lectures")
    new_goal = goal_coach.create_goal(
        title=goal_in.title,
        category=goal_in.category,
        target_date=exam_date.isoformat() if exam_date else goal_in.target_date,
        why=goal_in.why,
        milestones=goal_in.milestones,
        personal_context="; ".join(learned_context)
    )
    new_goal["description"] = goal_in.description
    new_goal["course_id"] = goal_in.course_id
    new_goal["lecture_ids"] = [material.get("id") for material in selected_lectures]
    new_goal["target_score"] = goal_in.target_score
    new_goal["has_exam"] = bool(exam_date)
    new_goal["exam_date"] = exam_date.isoformat() if exam_date else None
    goals = db.get_goals()
    goals.append(new_goal)
    db.save_goals(goals)

    # Convert the coach's milestone micro-steps into tasks the planner can place.
    tasks = db.get_tasks()
    task_category = {"study": "study", "career": "work", "personal": "personal"}.get(goal_in.category, "personal")
    milestones = new_goal.get("milestones", [])
    for milestone in milestones:
        steps = [] if selected_lectures else (milestone.get("micro_tasks") or [{"title": milestone.get("title", goal_in.title), "estimated_mins": 30}])
        for step in steps:
            if isinstance(step, str):
                step = {"title": step}
            due = milestone.get("due_date") or goal_in.target_date
            tasks.append({
                "task_id": f"T-{uuid.uuid4().hex[:10]}",
                "title": step.get("title") or milestone.get("title") or goal_in.title,
                "source": "life_coach",
                "topic_ref": milestone.get("title", goal_in.title),
                "estimated_mins": max(5, min(360, int(step.get("estimated_mins", 30)))),
                "adjusted_mins": max(5, min(360, int(step.get("estimated_mins", 30)))),
                "deadline": due,
                "priority_weight": 3.0,
                "cognitive_load": "medium",
                "status": "pending",
                "scheduled_date": None,
                "schedule_locked": False,
                "schedule_mode": "adaptive",
                "category": task_category,
                "goal_id": new_goal["goal_id"],
                "parent_milestone_id": milestone.get("id"),
                "rationale": "خطوة ناتجة عن هدفك؛ يمكن تعديل يومها ووقتها.",
            })
    exam_plan_result = None
    if exam_date and selected_lectures:
        first_open = next((m for m in milestones if m.get("status") != "completed"), None) or (milestones[0] if milestones else None)
        exam_plan_result = _exam_plan_tasks(new_goal, selected_lectures, exam_date,
                                            first_open.get("id") if first_open else None)
        tasks.extend(exam_plan_result["tasks"])
        new_goal["exam_plan_summary"] = {
            "days_available": exam_plan_result["days_available"],
            "sessions": len(exam_plan_result["tasks"]),
            "feasible": exam_plan_result["feasible"],
            "warnings": exam_plan_result["warnings"],
        }
        goals = db.get_goals()
        for g in goals:
            if g.get("goal_id") == new_goal["goal_id"]:
                g["exam_plan_summary"] = new_goal["exam_plan_summary"]
        db.save_goals(goals)
    elif selected_lectures:
        available_milestones = [m for m in milestones if m.get("status") != "completed"] or milestones
        for index, material in enumerate(selected_lectures):
            milestone = available_milestones[min(index, len(available_milestones) - 1)] if available_milestones else None
            title = material.get("title") or Path(material.get("filename", "")).stem
            duration = max(25, min(90, int(model.get("effective_focus_duration_mins", 35))))
            tasks.append({
                "task_id": f"T-{uuid.uuid4().hex[:10]}",
                "title": f"ذاكر المحاضرة: {title}",
                "source": "life_coach", "task_kind": "task",
                "topic_ref": title, "estimated_mins": duration, "adjusted_mins": duration,
                "deadline": goal_in.target_date, "priority_weight": 4.0,
                "cognitive_load": "medium", "status": "pending",
                "scheduled_date": None, "schedule_locked": False, "schedule_mode": "adaptive",
                "category": "study", "goal_id": new_goal["goal_id"],
                "parent_milestone_id": milestone.get("id") if milestone else None,
                "course_id": goal_in.course_id, "lecture_id": material.get("id"),
                "mastery_target": goal_in.target_score,
                "rationale": "جزء من خريطة هدفك الدراسية؛ بعد المذاكرة سيظهر اختبار قصير على المحاضرة.",
            })
    db.save_tasks(tasks)

    # Dispatch event
    db.add_event({
        "source": "Goal Coach Agent",
        "type": "goal_created",
        "payload": {"goal_id": new_goal["goal_id"], "title": new_goal["title"],
                    "relevance_tags": new_goal.get("relevance_tags", [])}
    })

    core_orchestrator.trigger_behavior_event("goal_created", "Goal Coach Agent", {"goal_id": new_goal["goal_id"]})
    core_orchestrator.run_full_schedule_optimization(_today())

    return {"status": "success", "goal": new_goal}

@router.put("/{goal_id}/exam")
def update_exam_date(goal_id: str, payload: ExamDateIn):
    """Move the exam date and rebuild the not-yet-done exam sessions."""
    goals = db.get_goals()
    goal = next((g for g in goals if g.get("goal_id") == goal_id), None)
    if not goal or not goal.get("has_exam"):
        raise HTTPException(status_code=404, detail="الهدف ده مش مرتبط بامتحان.")
    exam_date = _parse_exam_date(payload.exam_date)
    course = next((c for c in db.get_courses() if c.get("id") == goal.get("course_id")), None) or {}
    by_id = {m.get("id"): m for m in course.get("materials", [])}
    lectures = [by_id[i] for i in goal.get("lecture_ids", []) if i in by_id]
    if not lectures:
        raise HTTPException(status_code=422, detail="محاضرات الهدف لم تعد موجودة.")
    tasks = db.get_tasks()
    kept = [t for t in tasks if not (t.get("goal_id") == goal_id and t.get("exam_plan")
                                     and t.get("status") not in {"completed", "missed"})]
    first_open = next((m for m in goal.get("milestones", []) if m.get("status") != "completed"), None)
    result = _exam_plan_tasks(goal, lectures, exam_date, first_open.get("id") if first_open else None)
    done_lectures = {(t.get("lecture_id"), t.get("exam_phase")) for t in kept
                     if t.get("goal_id") == goal_id and t.get("exam_plan") and t.get("status") == "completed"}
    new_tasks = [t for t in result["tasks"] if (t.get("lecture_id"), t.get("exam_phase")) not in done_lectures
                 or t.get("exam_phase") in {"review", "review_quiz"}]
    db.save_tasks(kept + new_tasks)
    goal["exam_date"] = exam_date.isoformat()
    goal["target_date"] = exam_date.isoformat()
    goal["exam_plan_summary"] = {"days_available": result["days_available"], "sessions": len(new_tasks),
                                 "feasible": result["feasible"], "warnings": result["warnings"]}
    db.save_goals(goals)
    core_orchestrator.run_full_schedule_optimization(_today())
    return {"status": "success", "goal": goal, "exam_plan": goal["exam_plan_summary"]}

@router.delete("/{goal_id}")
def delete_goal(goal_id: str):
    goals = db.get_goals()
    remaining = [g for g in goals if g.get("goal_id") != goal_id]
    if len(remaining) == len(goals):
        raise HTTPException(status_code=404, detail="Goal not found")
    db.save_goals(remaining)
    tasks = db.get_tasks()
    removed_ids = {t.get("task_id") for t in tasks if t.get("goal_id") == goal_id}
    while True:
        children = {t.get("task_id") for t in tasks
                    if t.get("source_task_id") in removed_ids
                    or t.get("parent_task_id") in removed_ids}
        if children.issubset(removed_ids):
            break
        removed_ids.update(children)
    db.save_tasks([t for t in tasks if t.get("task_id") not in removed_ids])
    db.get_state()["scheduled_blocks"] = [b for b in db.get_state().get("scheduled_blocks", []) if b.get("task_id") not in removed_ids]
    db.save()
    core_orchestrator.run_full_schedule_optimization(_today())
    return {"status": "success"}

@router.post("/habits")
def create_habit(payload: HabitCreateIn):
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Habit name is required")
    habits = db.get_habits()
    if any(h.get("name", "").casefold() == name.casefold() for h in habits):
        raise HTTPException(status_code=409, detail="هذه العادة موجودة بالفعل.")
    habit = {"name": name, "streak": 0, "best_streak": 0, "last_completed": None, "freezes_available": 1}
    habits.append(habit)
    db.save_habits(habits)
    return {"status": "success", "habit": habit}

@router.put("/{goal_id}/milestones/{milestone_id}")
def update_milestone_status(goal_id: str, milestone_id: str, status: str):
    """Update milestone completion status."""
    if status not in {"completed", "pending"}:
        raise HTTPException(status_code=422, detail="Milestone status must be completed or pending.")
    goals = db.get_goals()
    found = False
    for g in goals:
        if g.get("goal_id") == goal_id:
            for m in g.get("milestones", []):
                if m.get("id") == milestone_id:
                    m["status"] = status
                    found = True
                    break
            milestones = g.get("milestones", [])
            g["status"] = "completed" if milestones and all(m.get("status") == "completed" for m in milestones) else "active"
    if not found:
        raise HTTPException(status_code=404, detail="Milestone or Goal not found.")

    db.save_goals(goals)
    tasks = db.get_tasks()
    for task in tasks:
        if task.get("parent_milestone_id") == milestone_id:
            task["status"] = "completed" if status == "completed" else "pending"
    db.save_tasks(tasks)
    core_orchestrator.run_full_schedule_optimization(_today())
    db.add_event({
        "source": "Goal Coach Agent",
        "type": "milestone_updated",
        "payload": {"goal_id": goal_id, "milestone_id": milestone_id, "status": status}
    })
    return {"status": "success", "milestone_id": milestone_id, "new_status": status}

@router.post("/reflections")
def submit_reflection(ref: ReflectionIn):
    """Submit daily reflection check-in, score burnout risk, and trigger closed-loop adaptation."""
    try:
        res = core_orchestrator.submit_reflection_and_adapt(
            energy=ref.energy,
            mood=ref.mood,
            reflection_text=ref.reflection,
            obstacles=ref.obstacles or ""
        )
    except (RuntimeError, ImportError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "success", "result": res}

@router.post("/habits/checkin")
def checkin_habit(habit_in: HabitCheckIn):
    """Daily habit check-in with streak freeze shield protection and badge awards."""
    res = core_orchestrator.complete_habit_and_adapt(habit_in.habit_name)
    return res
