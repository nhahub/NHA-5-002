"""Closed-Loop Multi-Agent Orchestrator (The Core USP Engine).
Executes: Collect Behavior -> Understand -> Predict -> Adapt -> Execute -> Measure -> Learn Again
Coordinates:
1. Adaptive Manager Agent (Personal Model & Predictions)
2. Assessment Agent (Diagnostics & Mastery)
3. Personal Planning Agent (Dynamic Scheduling & Deconstruction)
4. Goal & Life Coach Agent (Accountability & Gamification)
5. AI Tutor Agent (Grounded RAG Learning & Remediation)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from backend.app.database import db
from backend.app.agents.adaptive_manager import adaptive_manager
from backend.app.agents.planner import planner_agent
from backend.app.agents.quiz_agent import assessment_agent
from backend.app.agents.goal_coach import goal_coach
from backend.app.services.goal_progress import sync_goal_progress_for_task

logger = logging.getLogger("adaptive_coach.core_loop")

class CoreOrchestrator:
    """Coordinates multi-agent closed feedback loop across all 5 agents."""

    def trigger_behavior_event(
        self,
        event_type: str,
        source: str,
        payload: Dict[str, Any],
        auto_adapt: bool = True
    ) -> Dict[str, Any]:
        """Ingest a student behavior event and execute closed-loop adaptation."""
        # 1. Collect: Record event in database
        event_record = db.add_event({
            "source": source,
            "type": event_type,
            "payload": payload,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        logger.info("Ingested event [%s] from %s: %s", event_type, source, payload)

        # 2. Understand & Predict: Run Adaptive Manager cycle
        events = db.get_events()
        current_model = db.get_adaptive_model()
        tasks = db.get_tasks()
        lectures = db.get_university_schedule() + db.get_state().get("calendar_events", [])

        cycle_result = adaptive_manager.run_adaptive_cycle(
            current_model=current_model,
            events=events,
            upcoming_tasks=tasks,
            university_schedule=lectures
        )

        # Persist updated personal model & predictions
        updated_model = cycle_result["adaptive_model"]
        db.update_adaptive_model(updated_model)

        current_predictions = cycle_result.get("predictions", [])
        active_targets = {(p.get("target"), p.get("target_id")) for p in current_predictions}
        for old in db.get_predictions():
            if old.get("status") == "active" and (old.get("target"), old.get("target_id")) not in active_targets:
                old["status"] = "resolved"
        db.save()
        for pred in current_predictions:
            db.add_prediction(pred)

        # 3. Adapt Strategy: Execute preemptive interventions if auto_adapt is enabled
        execution_log = []
        if auto_adapt:
            for cmd in cycle_result.get("dispatch_commands", []):
                agent = cmd.get("target_agent")
                action = cmd.get("command")
                reason = cmd.get("reason", "")

                if action == "RECOVERY_BOOST":
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    tomorrow_str = (datetime.now() + __import__("datetime").timedelta(days=1)).strftime("%Y-%m-%d")
                    focus_mins = int(updated_model.get("effective_focus_duration_mins", 35))
                    current_tasks = db.get_tasks()
                    moved = []
                    for task in current_tasks:
                        due = (task.get("deadline") or "")[:10]
                        heavy = task.get("cognitive_load") == "high" or int(task.get("estimated_mins", 30)) > focus_mins * 1.4
                        can_defer = not due or due > today_str
                        if task.get("status") == "pending" and (task.get("scheduled_date") or "")[:10] == today_str and heavy and can_defer:
                            task["scheduled_date"] = tomorrow_str
                            task["rationale"] = "أُعيد ترتيب المهمة لليوم التالي بعد تسجيل طاقة منخفضة؛ الموعد النهائي محفوظ."
                            moved.append(task["task_id"])
                    if moved:
                        db.save_tasks(current_tasks)
                    self.run_full_schedule_optimization(today_str)
                    execution_log.append(
                        f"Life Coach: خففت حمل اليوم حسب مراجعتك وأعدت توزيع {len(moved)} مهمة مرنة؛ الخطة الأسبوعية اتحدثت."
                        if moved else "Life Coach: حفظت ملاحظة الطاقة وجدولت الأسبوع من جديد؛ لم تكن هناك مهمة ثقيلة مرنة اليوم."
                    )

                elif action == "PREEMPTIVE_SCHEDULE_RESTRUCTURE":
                    # Find large pending tasks and deconstruct them
                    deconstructed_any = False
                    current_tasks = db.get_tasks()
                    new_task_list = []
                    replaced_parent_ids = set()
                    focus_mins = int(updated_model.get("effective_focus_duration_mins", 35))

                    for t in current_tasks:
                        if (t.get("status") == "pending" and not t.get("actual_mins")
                                and int(t.get("estimated_mins", 30)) > focus_mins * 1.3):
                            # Deconstruct large task into micro-tasks
                            micros = planner_agent.deconstruct(t, focus_duration_mins=focus_mins)
                            if len(micros) > 1:
                                deconstructed_any = True
                                replaced_parent_ids.add(t["task_id"])
                                for i, m in enumerate(micros):
                                    new_task_list.append({
                                        "task_id": f"{t['task_id']}-sub{i+1}",
                                        "parent_milestone_id": t.get("parent_milestone_id"),
                                        "parent_task_id": t.get("task_id"),
                                        "title": m["title"],
                                        "source": t.get("source", "user"),
                                        "topic_ref": t.get("topic_ref"),
                                        "estimated_mins": m["estimated_mins"],
                                        "adjusted_mins": round(m["estimated_mins"] * updated_model.get("task_delay_factor", 1.2)),
                                        "deadline": t.get("deadline"),
                                        "priority_weight": t.get("priority_weight", 3.0),
                                        "cognitive_load": m.get("cognitive_load", "medium"),
                                        "status": "pending",
                                        "scheduled_date": t.get("scheduled_date"),
                                        "category": t.get("category", "study"),
                                        "goal_id": t.get("goal_id"),
                                        "notes": t.get("notes", ""),
                                        "rationale": f"Preemptively split by Adaptive Manager to fit {focus_mins}m attention span."
                                    })
                            else:
                                new_task_list.append(t)
                        else:
                            new_task_list.append(t)

                    if deconstructed_any:
                        db.save_tasks(new_task_list)
                        execution_log.append("Planning Agent: Deconstructed oversized pending tasks into attention-span chunks.")

                    # Re-optimize schedule with peak window placement
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    opt_res = planner_agent.optimize(
                        target_date=today_str,
                        tasks=db.get_tasks(),
                        lectures=db.get_university_schedule() + db.get_state().get("calendar_events", []),
                        adaptive_model=updated_model
                    )
                    preserved = [
                        b for b in db._state.get("scheduled_blocks", [])
                        if b.get("date") != today_str and b.get("task_id") not in replaced_parent_ids
                    ]
                    db._state["scheduled_blocks"] = preserved + opt_res.get("scheduled_blocks", [])
                    db.save()
                    execution_log.append(f"Planning Agent: Optimized calendar with {opt_res.get('total_blocks', 0)} blocks in peak window.")

                elif action == "INITIATE_REMEDIAL_SESSION":
                    tasks = db.get_tasks()
                    weak_topics = updated_model.get("weak_topics", [])
                    added = 0
                    for topic in weak_topics:
                        already_open = any(
                            t.get("source") == "assessment_remediation"
                            and t.get("topic_ref") == topic
                            and t.get("status") not in {"completed", "missed"}
                            for t in tasks
                        )
                        if already_open:
                            continue
                        topic_context = (updated_model.get("topic_context") or {}).get(topic, {})
                        tasks.append({
                            "task_id": f"T-{uuid.uuid4().hex[:10]}",
                            "title": f"مراجعة مركزة: {topic}",
                            "source": "assessment_remediation",
                            "topic_ref": topic,
                            "estimated_mins": int(updated_model.get("effective_focus_duration_mins", 35)),
                            "adjusted_mins": int(updated_model.get("effective_focus_duration_mins", 35)),
                            "deadline": None,
                            "priority_weight": 5.0,
                            "cognitive_load": "medium",
                            "status": "pending",
                            "scheduled_date": datetime.now().strftime("%Y-%m-%d"),
                            "category": "study",
                            "goal_id": None,
                            "course_id": topic_context.get("course_id"),
                            "lecture_id": topic_context.get("lecture_id"),
                            "notes": "تم إنشاء هذه المهمة من نقاط الضعف التي كشفها آخر اختبار.",
                            "rationale": "الأولوية زادت لأن تقييمك الأخير كشف حاجة لمراجعة هذا المفهوم.",
                        })
                        added += 1
                    if added:
                        db.save_tasks(tasks)
                        self.run_full_schedule_optimization(datetime.now().strftime("%Y-%m-%d"))
                    execution_log.append(f"Assessment → Planning: أُضيفت {added} مهام مراجعة مركزة للنقاط التي كشفها الاختبار.")

                elif action == "GENERATE_DIAGNOSTIC_EXAM":
                    execution_log.append(f"Assessment Agent: Queued targeted diagnostic quiz: {reason}.")

                elif action == "TRIGGER_ACCOUNTABILITY_CHECKIN":
                    try:
                        coach_response = goal_coach.process_manager_signal(
                            signal_type=action,
                            reason=reason,
                            details=f"Current system mode: {updated_model.get('current_system_mode', 'NORMAL')}; "
                                    f"motivation index: {updated_model.get('motivation_index', 'not recorded')}; "
                                    f"burnout risk: {updated_model.get('burnout_risk_score', 'unknown')}",
                        )
                        execution_log.append("Goal & Life Coach: " + coach_response["message"])
                        if coach_response.get("suggest_strategy_change") and coach_response.get("strategy_note"):
                            db.add_event({"source": "Goal & Life Coach Agent", "type": "strategy_adjustment_suggested",
                                          "payload": {"reason": reason, "strategy_note": coach_response["strategy_note"]}})
                    except Exception as exc:
                        logger.warning("Goal Coach accountability signal unavailable: %s", type(exc).__name__)
                        execution_log.append("Goal & Life Coach: لم يتوفر رد الوكيل؛ راجع إعدادات Gemini من حالة الوكلاء.")

        return {
            "status": "success",
            "event_ingested": event_record,
            "adaptive_model": updated_model,
            "active_predictions": cycle_result.get("predictions", []),
            "dispatches": cycle_result.get("dispatch_commands", []),
            "executed_interventions": execution_log,
        }

    def run_full_schedule_optimization(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        """Rebuild a rolling seven-day plan using latest user constraints and behavior."""
        now = datetime.now(ZoneInfo("Africa/Cairo"))
        today = now.date()
        date_str = target_date or today.isoformat()
        if date_str[:10] < today.isoformat():
            date_str = today.isoformat()
        current_model = db.get_adaptive_model()
        tasks = db.get_tasks()
        lectures = db.get_university_schedule() + db.get_state().get("calendar_events", [])
        start = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        horizon = [(start + timedelta(days=offset)).isoformat() for offset in range(7)]
        overdue_replanned = False
        for task in tasks:
            scheduled = (task.get("scheduled_date") or "")[:10]
            if task.get("status") not in {"completed", "missed"} and scheduled and scheduled < horizon[0]:
                task["scheduled_date"] = horizon[0]
                if task.get("schedule_locked"):
                    task["schedule_locked"] = False
                    task["schedule_mode"] = "adaptive"
                task["rationale"] = "نقلت المهمة المفتوحة إلى أول يوم في خطتك الأسبوعية؛ موعدها النهائي محفوظ."
                overdue_replanned = True

        # Spread flexible work across the actual seven-day window before the
        # daily planner assigns focus blocks. Explicitly dated user tasks stay
        # fixed; tasks without a chosen day are balanced against deadlines and
        # the user's daily study capacity.
        daily_load = {day: 0 for day in horizon}
        flexible_tasks = []
        for task in tasks:
            if task.get("status") in {"completed", "missed"}:
                continue
            minutes = max(5, int(task.get("adjusted_mins") or task.get("estimated_mins", 30)))
            scheduled = (task.get("scheduled_date") or "")[:10]
            if task.get("schedule_locked") and scheduled:
                # Fixed-day tasks (manual or exam-plan sessions) keep their day;
                # those dated beyond this 7-day window simply wait for their week.
                if scheduled in daily_load:
                    daily_load[scheduled] += minutes
            else:
                flexible_tasks.append(task)
        flexible_tasks.sort(key=lambda task: (
            min((task.get("deadline") or "9999-12-31")[:10], horizon[-1]),
            -float(task.get("priority_weight", 3) or 3),
            -int(task.get("adjusted_mins") or task.get("estimated_mins", 30)),
        ))
        daily_cap = max(30, int(current_model.get("max_daily_study_mins", 240) or 240))
        for task in flexible_tasks:
            earliest = horizon[0]
            due = (task.get("deadline") or "")[:10]
            latest = min(due, horizon[-1]) if due else horizon[-1]
            if latest < earliest:  # already overdue: put it in today's plan
                latest = earliest
            candidates = [day for day in horizon if earliest <= day <= latest] or [earliest]
            minutes = max(5, int(task.get("adjusted_mins") or task.get("estimated_mins", 30)))
            chosen = min(candidates, key=lambda day: (
                daily_load.get(day, 0) + minutes > daily_cap,
                daily_load.get(day, 0),
                day,
            ))
            if (task.get("scheduled_date") or "")[:10] != chosen:
                task["scheduled_date"] = chosen
                task["rationale"] = "وزعت المهمة على يوم متاح قبل موعدها، مع موازنة حمل الأسبوع."
                overdue_replanned = True
            task["schedule_locked"] = False
            task["schedule_mode"] = "adaptive"
            daily_load[chosen] = daily_load.get(chosen, 0) + minutes
        if overdue_replanned:
            db.save_tasks(tasks)
        results = []
        scheduled_task_ids = set()
        rescheduled_task_ids = set()
        changed_task_dates = False
        operating_start = db.get_user().get("day_start") or current_model.get("day_start")
        operating_end = db.get_user().get("day_end") or current_model.get("day_end")
        for d in horizon:
            day_tasks = [
                t for t in tasks
                if t.get("status") not in {"completed", "missed"}
                and t.get("task_id") not in scheduled_task_ids
                and (
                    ((t.get("scheduled_date") or "")[:10] == d)
                    if t.get("schedule_locked")
                    else (not t.get("scheduled_date") or (t.get("scheduled_date") or "")[:10] <= d)
                )
            ]
            busy_blocks = []
            if d == today.isoformat() and operating_start and operating_end:
                start_min = int(operating_start[:2]) * 60 + int(operating_start[3:5])
                end_min = int(operating_end[:2]) * 60 + int(operating_end[3:5])
                now_min = now.hour * 60 + now.minute
                next_quarter = min(((now_min + 14) // 15) * 15, end_min)
                if next_quarter > start_min:
                    busy_blocks.append({"start": operating_start, "end": f"{next_quarter // 60:02d}:{next_quarter % 60:02d}"})
            result = planner_agent.optimize(d, day_tasks, lectures, current_model, busy_blocks=busy_blocks)
            results.append(result)
            placed_today = {block.get("task_id") for block in result.get("scheduled_blocks", [])}
            scheduled_task_ids.update(placed_today)
            for task in day_tasks:
                if (task.get("task_id") in placed_today and not task.get("schedule_locked")
                        and (task.get("scheduled_date") or "")[:10] != d):
                    task["scheduled_date"] = d
                    task["rationale"] = "أعدت توزيعها إلى أول فترة متاحة؛ راعيت مواعيدك ومدة التركيز التي تعلمتها."
                    changed_task_dates = True
                    rescheduled_task_ids.add(task.get("task_id"))
        if changed_task_dates:
            db.save_tasks(tasks)
        blocks = [block for result in results for block in result.get("scheduled_blocks", [])]
        planned_ids = {
            t.get("task_id") for t in tasks
            if (t.get("scheduled_date") or "")[:10] in horizon
        }
        task_by_id = {t.get("task_id"): t for t in tasks}
        preserved = [
            b for b in db._state.get("scheduled_blocks", [])
            if b.get("date") not in horizon
            and b.get("task_id") not in planned_ids
            and task_by_id.get(b.get("task_id"), {}).get("status") not in {"completed", "missed"}
        ]
        db._state["scheduled_blocks"] = preserved + blocks
        db.save()

        # Log optimization event
        db.add_event({
            "source": "Planning Agent",
            "type": "schedule_optimized",
            "payload": {
                "date": date_str,
                "blocks_count": len(blocks),
                "unplaced_count": sum(result.get("total_unscheduled", 0) for result in results)
            }
        })
        horizon_task_ids = {
            t.get("task_id") for t in tasks
            if t.get("status") not in {"completed", "missed"}
            and ((t.get("scheduled_date") or "")[:10] in horizon or not t.get("scheduled_date"))
        }
        unscheduled = sorted(horizon_task_ids - scheduled_task_ids)
        adaptive_projection = self.trigger_behavior_event(
            event_type="plan_updated",
            source="Personal Planning Agent",
            payload={"date": date_str, "blocks_count": len(blocks),
                     "unscheduled_count": len(unscheduled)},
            auto_adapt=False,
        )
        return {
            "target_date": date_str,
            "horizon_days": 7,
            "scheduled_blocks": blocks,
            "unscheduled_tasks": unscheduled,
            "is_feasible": len(unscheduled) == 0,
            "total_blocks": len(blocks),
            "total_unscheduled": len(unscheduled),
            "unscheduled_task_ids": unscheduled,
            "rescheduled_task_count": len(rescheduled_task_ids),
            "adaptive_projection": adaptive_projection.get("active_predictions", []),
        }

    def submit_quiz_and_adapt(
        self,
        quiz_id: str,
        topic: str,
        questions: List[Dict[str, Any]],
        answers: Dict[str, str],
        course_id: Optional[str] = None,
        lecture_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Grade quiz, record assessment, and feed directly into the Adaptive Manager."""
        result = assessment_agent.evaluate_quiz(
            quiz_id=quiz_id,
            course_name=topic,
            questions=questions,
            user_answers=answers
        )
        result.update({"course_id": course_id, "lecture_id": lecture_id})
        concept_totals: Dict[str, List[int]] = {}
        for answer in result.get("detailed_answers", []):
            concept = (answer.get("concept") or "").strip()
            if not concept:
                continue
            counts = concept_totals.setdefault(concept, [0, 0])
            counts[1] += 1
            counts[0] += int(bool(answer.get("is_correct")))
        concept_scores = {
            concept: round(correct / total * 100, 1)
            for concept, (correct, total) in concept_totals.items() if total
        }
        result["concept_scores"] = concept_scores

        required_score = adaptive_manager.mastery_threshold
        completed_assessment_task = None
        goal_requirement_met = None
        if task_id:
            task_list = db.get_tasks()
            completed_assessment_task = next((item for item in task_list if item.get("task_id") == task_id), None)
            if completed_assessment_task:
                required_score = float(completed_assessment_task.get("mastery_target") or required_score)
                goal_requirement_met = float(result.get("score_pct", 0)) >= required_score
                completed_assessment_task["status"] = "completed" if goal_requirement_met else "pending"
                completed_assessment_task["last_score_pct"] = result.get("score_pct")
                completed_assessment_task["last_attempt_at"] = datetime.now(ZoneInfo("Africa/Cairo")).isoformat()
                completed_assessment_task["remaining_mins"] = 0 if goal_requirement_met else completed_assessment_task.get("estimated_mins", 15)
                if not goal_requirement_met:
                    completed_assessment_task["rationale"] = f"تحتاج {required_score:.0f}% لإغلاق خطوة الهدف؛ نتيجتك {float(result.get('score_pct', 0)):.0f}%. راجع النقاط الضعيفة ثم أعد الاختبار."
                db.save_tasks(task_list)
                sync_goal_progress_for_task(completed_assessment_task, task_list, db)

        # Record in assessments collection
        db.add_assessment(result)

        # Check for first quiz badge
        achievements = db.get_achievements()
        has_badge = any(a["key"] == "first_quiz_master" for a in achievements)
        if not has_badge:
            db.add_achievement({
                "key": "first_quiz_master",
                "title": "Diagnostic Explorer",
                "earned_on": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "description": "Completed first comprehensive diagnostic quiz."
            })

        # Close the loop: Dispatch quiz_result event into the Adaptive Manager!
        reported_gaps = list(dict.fromkeys(
            list(result.get("weak_topics", []))
            + [concept for concept, score in concept_scores.items() if score < required_score]
        ))
        result["required_score"] = required_score
        loop_res = self.trigger_behavior_event(
            event_type="quiz_result",
            source="Assessment Agent",
            payload={
                "topic": topic,
                "course_id": course_id,
                "lecture_id": lecture_id,
                "score_pct": result["score_pct"],
                "concept_scores": concept_scores,
                "failed_concepts": reported_gaps,
                "error_types": result["error_types"],
                "score": result["score"],
                "total": result["total"]
            },
            auto_adapt=True
        )
        if completed_assessment_task:
            self.run_full_schedule_optimization(datetime.now(ZoneInfo("Africa/Cairo")).date().isoformat())

        return {
            "evaluation": result,
            "assessment_task": completed_assessment_task,
            "goal_requirement_met": goal_requirement_met,
            "closed_loop_adaptation": loop_res
        }

    def submit_reflection_and_adapt(
        self,
        energy: int,
        mood: str,
        reflection_text: str,
        obstacles: str = ""
    ) -> Dict[str, Any]:
        """Process daily reflection, score burnout, and adapt strategy."""
        history = db.get_reflections()
        coach_res = goal_coach.process_reflection(
            energy=energy,
            mood=mood,
            reflection_text=reflection_text,
            obstacles=obstacles,
            historical_reflections=history
        )

        # Save reflection
        db.add_reflection(coach_res["reflection"])

        # Update model burnout risk & motivation index
        db.update_adaptive_model({
            "burnout_risk_score": coach_res["burnout_risk"],
            "burnout_risk_value": coach_res["burnout_risk_value"],
            "motivation_index": coach_res["motivation_index"],
            "consecutive_low_days": coach_res["consecutive_low_days"],
            "latest_energy_level": energy,
        })

        # Dispatch event to Adaptive Manager
        loop_res = self.trigger_behavior_event(
            event_type="reflection",
            source="Goal Coach Agent",
            payload={
                "energy": energy,
                "mood": mood,
                "mood_score": coach_res["reflection"].get("mood_score"),
                "avoidance_signals": coach_res["reflection"].get("avoidance_signals", []),
                "burnout_signals": coach_res["reflection"].get("burnout_signals", []),
                "burnout_risk": coach_res["burnout_risk"],
                "motivation_index": coach_res["motivation_index"]
            },
            auto_adapt=True
        )

        return {
            "coach_response": coach_res,
            "closed_loop_adaptation": loop_res
        }

    def complete_habit_and_adapt(self, habit_name: str) -> Dict[str, Any]:
        """Check in habit, update streak & shields, award badges, and feed loop."""
        habits = db.get_habits()
        existing_keys = {a["key"] for a in db.get_achievements()}
        updated_list = []
        new_badges = []
        details = {}

        for h in habits:
            if h.get("name") == habit_name:
                res = goal_coach.process_habit_completion(
                    habit_dict=h,
                    existing_badge_keys=existing_keys
                )
                updated_list.append(res["habit"])
                new_badges.extend(res["new_badges"])
                details = res["details"]
            else:
                updated_list.append(h)

        db.save_habits(updated_list)
        for b in new_badges:
            db.add_achievement(b)

        # Feed loop
        loop_res = self.trigger_behavior_event(
            event_type="habit_checked_in",
            source="Goal Coach Agent",
            payload={"habit": habit_name, "streak": details.get("new_streak", 1)}
        )

        return {
            "status": "success",
            "details": details,
            "new_badges": new_badges,
            "closed_loop": loop_res
        }

core_orchestrator = CoreOrchestrator()
