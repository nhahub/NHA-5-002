"""Exam-oriented study plan builder.

Given the lectures a student must be examined on and the exam date, produce
dated tasks that:
  * start today and finish the day BEFORE the exam (never on the exam day),
  * spread each lecture over more than one study session when time allows,
  * add a short quiz after each lecture,
  * reserve the last days for spaced review sessions and review quizzes,
  * respect a daily study-minutes cap (and report when time is not enough).

Pure functions only (no DB access) so it is easy to test.
"""
from __future__ import annotations

import math
import uuid
from datetime import date, timedelta
from typing import Any, Dict, List, Optional


def _round5(minutes: float) -> int:
    return max(5, int(round(minutes / 5.0)) * 5)


def _lecture_minutes(material: Dict[str, Any]) -> int:
    """Rough total study time for a lecture, based on its page count."""
    pages = int(material.get("pages") or 0)
    return max(45, min(150, pages * 5)) if pages else 60


def _task(*, goal_id: str, title: str, topic: str, minutes: int, day: date, deadline: date,
          kind: str, priority: float, course_id: Optional[str], lecture_id: Optional[str],
          milestone_id: Optional[str], mastery_target: Optional[float], exam_date: date,
          phase: str, rationale: str) -> Dict[str, Any]:
    return {
        "task_id": f"T-{uuid.uuid4().hex[:10]}",
        "title": title,
        "source": "life_coach",
        "task_kind": kind,  # "task" | "assessment"
        "topic_ref": topic,
        "estimated_mins": minutes,
        "adjusted_mins": minutes,
        "deadline": deadline.isoformat(),
        "priority_weight": priority,
        "cognitive_load": "medium" if kind == "assessment" or phase == "review" else "high",
        "status": "pending",
        "scheduled_date": day.isoformat(),
        "schedule_locked": True,  # keeps the session on its planned day
        "schedule_mode": "adaptive",
        "category": "study",
        "goal_id": goal_id,
        "parent_milestone_id": milestone_id,
        "course_id": course_id,
        "lecture_id": lecture_id,
        "mastery_target": mastery_target,
        "exam_plan": True,
        "exam_phase": phase,  # "learn" | "quiz" | "review" | "review_quiz"
        "exam_date": exam_date.isoformat(),
        "rationale": rationale,
    }


def build_exam_plan(
    *,
    goal_id: str,
    course_id: Optional[str],
    lectures: List[Dict[str, Any]],
    exam_date: date,
    today: date,
    focus_mins: int = 35,
    daily_cap_mins: int = 240,
    milestone_id: Optional[str] = None,
    mastery_target: Optional[float] = None,
) -> Dict[str, Any]:
    """Return {"tasks": [...], "warnings": [...], "days_available": n, "feasible": bool}."""
    n_days = (exam_date - today).days  # days today .. exam_date-1
    if n_days < 1:
        raise ValueError("exam_date must be after today")
    if not lectures:
        raise ValueError("at least one lecture is required")

    k = len(lectures)
    focus = max(25, min(90, int(focus_mins or 35)))
    deadline = exam_date - timedelta(days=1)
    day_of = lambda idx: today + timedelta(days=max(0, min(n_days - 1, idx)))  # noqa: E731

    # ---- phases ---------------------------------------------------------
    if n_days >= 4:
        review_days = max(1, round(n_days * 0.3))
    elif n_days >= 2:
        review_days = 1
    else:
        review_days = 0
    learn_days = n_days - review_days

    tasks: List[Dict[str, Any]] = []
    last_session_idx: Dict[str, int] = {}

    # ---- learning sessions + quiz after each lecture ----------------------
    for i, material in enumerate(lectures):
        lid = material.get("id")
        title = material.get("title") or material.get("filename") or f"المحاضرة {i + 1}"
        start = (i * learn_days) // k
        end = max(start, ((i + 1) * learn_days) // k - 1)
        window = end - start + 1
        total = _lecture_minutes(material)
        sessions = 1
        if n_days >= 3 and total > focus:
            sessions = min(3, max(2, math.ceil(total / focus)))
        if k > learn_days:  # very tight: keep one session per lecture
            sessions = 1
        per_session = _round5(max(25, min(90, total / sessions)))
        last_idx = start
        for j in range(sessions):
            idx = start + (j * window) // sessions
            last_idx = max(last_idx, idx)
            suffix = f" (جزء {j + 1}/{sessions})" if sessions > 1 else ""
            tasks.append(_task(
                goal_id=goal_id, title=f"ذاكر المحاضرة: {title}{suffix}", topic=title,
                minutes=per_session, day=day_of(idx), deadline=deadline, kind="task", priority=4.0,
                course_id=course_id, lecture_id=lid, milestone_id=milestone_id,
                mastery_target=mastery_target, exam_date=exam_date, phase="learn",
                rationale="جلسة مذاكرة ضمن خطة الامتحان، موزعة على أيام قبل موعد الامتحان.",
            ))
        last_session_idx[lid] = last_idx
        tasks.append(_task(
            goal_id=goal_id, title=f"اختبر فهمك: {title}", topic=title, minutes=15,
            day=day_of(last_idx + 1 if last_idx + 1 < n_days else last_idx), deadline=deadline,
            kind="assessment", priority=4.0, course_id=course_id, lecture_id=lid,
            milestone_id=milestone_id, mastery_target=mastery_target, exam_date=exam_date,
            phase="quiz", rationale="اختبار قصير بعد مذاكرة المحاضرة لقياس الفهم قبل الانتقال لغيرها.",
        ))

    # ---- review phase: spaced review sessions + review quizzes -------------
    if review_days:
        review_start = learn_days
        two_rounds = review_days >= 2 or n_days >= 6
        rounds = 2 if two_rounds else 1
        slots = list(range(review_start, n_days))
        for r in range(rounds):
            for i, material in enumerate(lectures):
                lid = material.get("id")
                title = material.get("title") or material.get("filename") or f"المحاضرة {i + 1}"
                # first round in the first half of the review window, second in the last days
                if rounds == 2:
                    half = max(1, len(slots) // 2)
                    pool = slots[:half] if r == 0 else slots[half:] or slots[-1:]
                else:
                    pool = slots
                idx = pool[(i * len(pool)) // k] if k > len(pool) else pool[i % len(pool)]
                final = (r == rounds - 1)
                tasks.append(_task(
                    goal_id=goal_id, title=f"مراجعة: {title}" + (" (مراجعة أخيرة)" if final and rounds == 2 else ""),
                    topic=title, minutes=_round5(max(20, focus * 0.6)), day=day_of(idx), deadline=deadline,
                    kind="task", priority=5.0, course_id=course_id, lecture_id=lid,
                    milestone_id=milestone_id, mastery_target=mastery_target, exam_date=exam_date,
                    phase="review",
                    rationale="جلسة مراجعة متباعدة لتثبيت المعلومة قبل الامتحان.",
                ))
        if n_days >= 5:
            for i, material in enumerate(lectures):
                lid = material.get("id")
                title = material.get("title") or material.get("filename") or f"المحاضرة {i + 1}"
                pool = slots[len(slots) // 2:] or slots
                idx = pool[(i * len(pool)) // k] if k > len(pool) else pool[i % len(pool)]
                tasks.append(_task(
                    goal_id=goal_id, title=f"اختبار مراجعة: {title}", topic=title, minutes=15,
                    day=day_of(idx), deadline=deadline, kind="assessment", priority=5.0,
                    course_id=course_id, lecture_id=lid, milestone_id=milestone_id,
                    mastery_target=mastery_target, exam_date=exam_date, phase="review_quiz",
                    rationale="اختبار مراجعة قبل الامتحان لاكتشاف نقاط الضعف المتبقية.",
                ))
    elif n_days == 1:
        # Only one day left: add a quick review of every lecture after studying it.
        for i, material in enumerate(lectures):
            title = material.get("title") or material.get("filename") or f"المحاضرة {i + 1}"
            tasks.append(_task(
                goal_id=goal_id, title=f"مراجعة سريعة: {title}", topic=title, minutes=20, day=day_of(0),
                deadline=deadline, kind="task", priority=5.0, course_id=course_id,
                lecture_id=material.get("id"), milestone_id=milestone_id, mastery_target=mastery_target,
                exam_date=exam_date, phase="review", rationale="مراجعة سريعة لأن الوقت المتبقي يوم واحد فقط.",
            ))

    # ---- respect the daily cap -------------------------------------------
    warnings: List[str] = []
    cap = max(30, int(daily_cap_mins or 240))
    load = {d: 0 for d in range(n_days)}
    idx_of = lambda t: (date.fromisoformat(t["scheduled_date"]) - today).days  # noqa: E731
    for t in tasks:
        load[idx_of(t)] += t["estimated_mins"]
    movable_phases = ("review", "review_quiz", "learn")
    for d in range(n_days):
        while load[d] > cap:
            candidates = [t for t in tasks if idx_of(t) == d and t["exam_phase"] in movable_phases]
            # move the lowest-priority, latest-added session first
            candidates.sort(key=lambda t: (t["priority_weight"], -tasks.index(t)))
            moved = False
            for t in candidates:
                # a session may only move within [0, n_days-1]; prefer the nearest day with room
                for step in range(1, n_days):
                    for nd in (d - step, d + step):
                        if 0 <= nd < n_days and load[nd] + t["estimated_mins"] <= cap:
                            load[d] -= t["estimated_mins"]
                            load[nd] += t["estimated_mins"]
                            t["scheduled_date"] = day_of(nd).isoformat()
                            moved = True
                            break
                    if moved:
                        break
                if moved:
                    break
            if not moved:
                break
    over = [d for d in range(n_days) if load[d] > cap]
    feasible = not over
    if over:
        warnings.append(
            f"الوقت المتاح قبل الامتحان ({n_days} يوم) لا يكفي لحمل {cap} دقيقة يوميًا؛ "
            "زوّد ساعات المذاكرة اليومية أو قلّل عدد المحاضرات."
        )
    if n_days <= 2:
        warnings.append("الوقت المتبقي قصير جدًا؛ ركّزت الخطة على الأهم.")

    tasks.sort(key=lambda t: (t["scheduled_date"], t["exam_phase"] != "learn"))
    return {"tasks": tasks, "warnings": warnings, "days_available": n_days, "feasible": feasible}
