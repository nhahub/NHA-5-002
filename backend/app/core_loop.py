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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.app.database import db
from backend.app.agents.adaptive_manager import adaptive_manager
from backend.app.agents.planner import planner_agent
from backend.app.agents.quiz_agent import assessment_agent
from backend.app.agents.goal_coach import goal_coach
from backend.app.agents.tutor_agent import tutor_agent

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
        lectures = db.get_university_schedule()

        cycle_result = adaptive_manager.run_adaptive_cycle(
            current_model=current_model,
            events=events,
            upcoming_tasks=tasks,
            university_schedule=lectures
        )

        # Persist updated personal model & predictions
        updated_model = cycle_result["adaptive_model"]
        db.update_adaptive_model(updated_model)

        for pred in cycle_result.get("predictions", []):
            db.add_prediction(pred)

        # 3. Adapt Strategy: Execute preemptive interventions if auto_adapt is enabled
        execution_log = []
        if auto_adapt:
            for cmd in cycle_result.get("dispatch_commands", []):
                agent = cmd.get("target_agent")
                action = cmd.get("command")
                reason = cmd.get("reason", "")

                if action == "PREEMPTIVE_SCHEDULE_RESTRUCTURE":
                    # Find large pending tasks and deconstruct them
                    deconstructed_any = False
                    current_tasks = db.get_tasks()
                    new_task_list = []
                    focus_mins = int(updated_model.get("effective_focus_duration_mins", 35))

                    for t in current_tasks:
                        if t.get("status") == "pending" and int(t.get("estimated_mins", 30)) > focus_mins * 1.3:
                            # Deconstruct large task into micro-tasks
                            micros = planner_agent.deconstruct(t, focus_duration_mins=focus_mins)
                            if len(micros) > 1:
                                deconstructed_any = True
                                for i, m in enumerate(micros):
                                    new_task_list.append({
                                        "task_id": f"{t['task_id']}-sub{i+1}",
                                        "parent_milestone_id": t.get("parent_milestone_id"),
                                        "title": m["title"],
                                        "source": "preemptive_adaptation",
                                        "topic_ref": t.get("topic_ref"),
                                        "estimated_mins": m["estimated_mins"],
                                        "adjusted_mins": round(m["estimated_mins"] * updated_model.get("task_delay_factor", 1.2)),
                                        "deadline": t.get("deadline"),
                                        "priority_weight": t.get("priority_weight", 3.0),
                                        "cognitive_load": m.get("cognitive_load", "medium"),
                                        "status": "pending",
                                        "scheduled_date": t.get("scheduled_date"),
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
                        lectures=db.get_university_schedule(),
                        adaptive_model=updated_model
                    )
                    db._state["scheduled_blocks"] = opt_res.get("scheduled_blocks", [])
                    db.save()
                    execution_log.append(f"Planning Agent: Optimized calendar with {opt_res.get('total_blocks', 0)} blocks in peak window.")

                elif action == "INITIATE_REMEDIAL_SESSION":
                    execution_log.append(f"AI Tutor: Prepared remedial concept breakdown for weak areas: {reason}.")

                elif action == "GENERATE_DIAGNOSTIC_EXAM":
                    execution_log.append(f"Assessment Agent: Queued targeted diagnostic quiz: {reason}.")

                elif action == "TRIGGER_ACCOUNTABILITY_CHECKIN":
                    execution_log.append(f"Life Coach: Prepared empathetic recovery check-in: {reason}.")

        return {
            "status": "success",
            "event_ingested": event_record,
            "adaptive_model": updated_model,
            "active_predictions": cycle_result.get("predictions", []),
            "dispatches": cycle_result.get("dispatch_commands", []),
            "executed_interventions": execution_log,
        }

    def run_full_schedule_optimization(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        """Trigger smart schedule optimization using latest adaptive parameters."""
        date_str = target_date or datetime.now().strftime("%Y-%m-%d")
        current_model = db.get_adaptive_model()
        tasks = db.get_tasks()
        lectures = db.get_university_schedule()

        res = planner_agent.optimize(
            target_date=date_str,
            tasks=tasks,
            lectures=lectures,
            adaptive_model=current_model
        )
        db._state["scheduled_blocks"] = res.get("scheduled_blocks", [])
        db.save()

        # Log optimization event
        db.add_event({
            "source": "Planning Agent",
            "type": "schedule_optimized",
            "payload": {
                "date": date_str,
                "blocks_count": res.get("total_blocks", 0),
                "unplaced_count": res.get("total_unplaced", 0)
            }
        })
        return res

    def submit_quiz_and_adapt(
        self,
        quiz_id: str,
        topic: str,
        questions: List[Dict[str, Any]],
        answers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Grade quiz, record assessment, and feed directly into the Adaptive Manager."""
        result = assessment_agent.evaluate_quiz(
            quiz_id=quiz_id,
            topic=topic,
            questions=questions,
            user_answers=answers
        )

        # Record in assessments collection
        db.add_assessment(result.model_dump())

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
        loop_res = self.trigger_behavior_event(
            event_type="quiz_result",
            source="Assessment Agent",
            payload={
                "topic": topic,
                "score_pct": result.score_pct,
                "failed_concepts": result.weak_topics,
                "error_types": result.error_types,
                "score": result.score,
                "total": result.total
            },
            auto_adapt=True
        )

        return {
            "evaluation": result.model_dump(),
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
            "motivation_index": coach_res["motivation_index"]
        })

        # Dispatch event to Adaptive Manager
        loop_res = self.trigger_behavior_event(
            event_type="reflection",
            source="Goal Coach Agent",
            payload={
                "energy": energy,
                "mood": mood,
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
