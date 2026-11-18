"""Goal & Life Coach Agent (Agent 4 of 5).
Wraps Goal-Life-Coach-Agent-main:
- Goal management (Study, Career, Personal)
- SMART Milestone breakdown
- Daily reflections, mood/energy scoring, burnout risk evaluation
- Habit tracking, streak freeze protection, comeback halving, and milestone badges
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure Goal-Life-Coach-Agent-main is on path
BASE_DIR = Path(__file__).resolve().parents[3]
COACH_DIR = BASE_DIR / "Goal-Life-Coach-Agent-main"
if str(COACH_DIR) not in sys.path:
    sys.path.insert(0, str(COACH_DIR))

from coach.schemas import BurnoutRisk, CoachAdaptive, CoachState, Goal, Habit, Milestone
from coach.gamification import update_habit_streak, check_streak_badges
from coach.scoring import update_adaptive_metrics
from coach.llm import analyze_reflection_with_llm, decompose_goal_with_llm, generate_accountability_reply_with_llm, generate_coaching_reply_with_llm, get_llm
from coach.tools import create_goal as _raw_create_goal, send_milestones_to_planner, send_relevance_tags_to_tutor
from backend.app.config import settings

logger = logging.getLogger("adaptive_coach.life_coach")

class GoalCoachAgentWrapper:
    """Wrapper for the Goal & Life Coach agent capabilities."""

    def create_goal(
        self,
        title: str,
        category: str = "study",
        target_date: Optional[str] = None,
        why: str = "",
        milestones: Optional[List[Dict[str, Any]]] = None,
        personal_context: str = "",
    ) -> Dict[str, Any]:
        """Create a new structured goal with initial milestones."""
        from backend.app.database import db
        generated = self._generate_goal_steps(title, category, target_date, why, personal_context) if not milestones else None
        if generated:
            milestones = generated["milestones"]
            why = why or generated.get("why", "")
        _, coach_goal = _raw_create_goal(CoachState(user_id=str(db.get_user().get("id", ""))), title, category, target_date, why)
        goal_id = coach_goal.goal_id
        relevance_tags = generated.get("relevance_tags", []) if generated else []
        m_list = []
        if milestones:
            for i, m in enumerate(milestones):
                count = len(milestones)
                due_date = m.get("due_date")
                if not due_date and target_date:
                    try:
                        start_day = date.today()
                        end_day = max(start_day, date.fromisoformat(target_date[:10]))
                        due_date = (start_day + (end_day - start_day) * ((i + 1) / count)).isoformat()
                    except (ValueError, TypeError):
                        due_date = target_date
                m_list.append({
                    "id": m.get("id") or f"M-{goal_id}-{i+1}",
                    "goal_id": goal_id,
                    "title": m.get("title", f"Milestone {i+1}"),
                    "status": m.get("status", "pending"),
                    "criteria": m.get("criteria", ""),
                    "due_date": due_date,
                    "micro_tasks": m.get("micro_tasks", [])
                })
        else:
            # Default first milestone
            first_step = {
                "study": f"حدد أول جزء دراسي تحتاجه لهدف {title}",
                "career": f"حدد أول خطوة عملية تقربك من {title}",
                "personal": f"اختر خطوة صغيرة تبدأ بها {title}",
            }.get(category, f"حدد خطوة أولى نحو {title}")
            m_list.append({
                "id": f"M-{goal_id}-1",
                "goal_id": goal_id,
                "title": f"الخطوة الأولى نحو {title}",
                "status": "pending",
                "criteria": "خطوة محددة يمكن التحقق من إنجازها.",
                "due_date": target_date,
                "micro_tasks": [
                    {"title": first_step, "estimated_mins": 25}
                ]
            })

        planner_event = send_milestones_to_planner(coach_goal, [
            Milestone(
                id=item["id"], goal_id=goal_id, title=item["title"], status=item["status"],
                criteria=item["criteria"], due_date=item.get("due_date"),
                micro_tasks=item.get("micro_tasks", []),
            ) for item in m_list
        ])
        tutor_event = send_relevance_tags_to_tutor(coach_goal, relevance_tags)
        return {
            "goal_id": goal_id,
            "title": title,
            "category": category,
            "target_date": target_date,
            "why": why,
            "status": "active",
            "strategy_note": "تم بناء الخطوات من هدفك." if generated else "أنشأت خطوة بداية بسيطة؛ ربط مولّد الذكاء الاصطناعي يتيح تفكيكًا أوسع.",
            "milestones": m_list,
            "relevance_tags": relevance_tags,
            "agent_events": [planner_event.model_dump(), tutor_event.model_dump()],
            "decomposition_mode": "llm_generated" if generated else "basic_fallback",
        }

    def _generate_goal_steps(self, title: str, category: str, target_date: Optional[str], why: str, personal_context: str) -> Optional[List[Dict[str, Any]]]:
        try:
            api_key = settings.gemini_api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                return None
            llm = get_llm(api_key=api_key, allow_fake_fallback=False)
            decomposition = decompose_goal_with_llm(
                llm, goal_title=title, target_date=target_date,
                category_hint=category, context=f"{why}\n{personal_context}".strip(),
            )
            raw = [milestone.model_dump() for milestone in decomposition.milestones]
            normalized = []
            for milestone in raw[:5]:
                steps = [
                    {"title": str(step.get("title", "")).strip(),
                     "estimated_mins": max(10, min(120, int(step.get("estimated_mins", 30)))),
                     "cognitive_load": step.get("cognitive_load", "medium") if step.get("cognitive_load") in {"low", "medium", "high"} else "medium"}
                    for step in milestone.get("micro_tasks", [])[:3]
                    if str(step.get("title", "")).strip()
                ]
                if milestone.get("title") and steps:
                    normalized.append({"title": str(milestone["title"]).strip(), "criteria": str(milestone.get("criteria", "")), "micro_tasks": steps})
            return ({"milestones": normalized, "why": decomposition.why,
                     "relevance_tags": decomposition.relevance_tags}
                    if len(normalized) >= 2 else None)
        except Exception as exc:
            logger.warning("Goal decomposition failed; using the basic goal step: %s", exc)
            return None

    def process_habit_completion(
        self,
        habit_dict: Dict[str, Any],
        completion_date: Optional[str] = None,
        existing_badge_keys: Optional[set[str]] = None
    ) -> Dict[str, Any]:
        """Process daily habit check-in with streak, freeze, comeback rule, and badge logic."""
        target_date_str = completion_date or date.today().isoformat()
        h = Habit(
            name=habit_dict.get("name", "Daily Habit"),
            streak=int(habit_dict.get("streak", 0)),
            best_streak=int(habit_dict.get("best_streak", 0)),
            last_completed=habit_dict.get("last_completed"),
            freezes_available=int(habit_dict.get("freezes_available", 1)),
        )

        updated_habit, new_badges, details = update_habit_streak(
            habit=h,
            completion_date_str=target_date_str,
            existing_keys=existing_badge_keys or set(),
        )

        return {
            "habit": {
                "name": updated_habit.name,
                "streak": updated_habit.streak,
                "best_streak": updated_habit.best_streak,
                "last_completed": updated_habit.last_completed,
                "freezes_available": updated_habit.freezes_available,
            },
            "new_badges": [
                {
                    "key": b.key,
                    "title": b.title,
                    "earned_on": b.earned_on,
                    "description": f"Awarded for reaching streak milestone."
                }
                for b in new_badges
            ],
            "details": details,
        }

    def process_manager_signal(self, signal_type: str, reason: str, details: str = "") -> Dict[str, Any]:
        """Run the source Goal Coach's MANAGER_SIGNAL workflow for an adaptive intervention."""
        api_key = settings.gemini_api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Goal & Life Coach needs GOOGLE_API_KEY or GEMINI_API_KEY to respond to adaptive-manager signals.")
        llm = get_llm(api_key=api_key, allow_fake_fallback=False)
        from backend.app.database import db
        model = db.get_adaptive_model()
        level = model.get("burnout_risk_score", "Low")
        adaptive = CoachAdaptive(
            motivation_index=float(model.get("motivation_index", 0.6)),
            burnout_risk=BurnoutRisk(
                score=float(model.get("burnout_risk_value", 0.2)),
                level=level if level in {"Low", "Medium", "High"} else "Low",
            ),
            burnout_risk_score=level if level in {"Low", "Medium", "High"} else "Low",
        )
        state = CoachState(user_id=str(db.get_user().get("id", "student-001")), adaptive_model=adaptive,
                           consecutive_low_days=int(model.get("consecutive_low_days", 0)))
        for item in db.get_goals():
            goal = Goal(
                goal_id=item.get("goal_id", ""), title=item.get("title", ""),
                category=item.get("category", "personal") if item.get("category") in {"career", "study", "personal"} else "personal",
                target_date=item.get("target_date"), why=item.get("why", ""),
                status="active" if item.get("status") != "completed" else "completed",
            )
            getattr(state.goals_data, {"study": "study_goals", "career": "career_goals", "personal": "personal_goals"}[goal.category]).append(goal)
        reply = generate_accountability_reply_with_llm(llm, signal_type, reason, details, state)
        return {"message": reply.message, "strategy_note": reply.strategy_note,
                "suggest_strategy_change": reply.suggest_strategy_change}

    def process_reflection(
        self,
        energy: int,
        mood: str,
        reflection_text: str,
        obstacles: str = "",
        historical_reflections: Optional[List[Dict[str, Any]]] = None,
        recent_blocks: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Run the source Coach Agent's reflection extraction and deterministic scoring."""
        now = datetime.now(timezone.utc).isoformat()
        api_key = settings.gemini_api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Goal & Life Coach needs GOOGLE_API_KEY or GEMINI_API_KEY for its reflection analysis and coaching response.")
        llm = get_llm(api_key=api_key, allow_fake_fallback=False)
        analysis = analyze_reflection_with_llm(
            llm, text=f"Mood: {mood}. Reflection: {reflection_text}. Obstacles: {obstacles}",
            energy_hint=energy / 5.0, date_str=now[:10],
        )
        from backend.app.database import db
        stored_model = db.get_adaptive_model()
        current_level = stored_model.get("burnout_risk_score", "Low")
        prior_burnout = float(stored_model.get("burnout_risk_value", {"Low": 0.2, "Medium": 0.5, "High": 0.8}.get(current_level, 0.2)))
        previous = CoachAdaptive(
            motivation_index=float(stored_model.get("motivation_index", 0.6)),
            burnout_risk=BurnoutRisk(score=prior_burnout, level=current_level if current_level in {"Low", "Medium", "High"} else "Low"),
            burnout_risk_score=current_level if current_level in {"Low", "Medium", "High"} else "Low",
        )
        habits = db.get_habits()
        completion_ratio = (sum(1 for h in habits if h.get("last_completed") == now[:10]) / len(habits)) if habits else 1.0
        adaptive, low_days = update_adaptive_metrics(previous, analysis, completion_ratio, int(stored_model.get("consecutive_low_days", 0)))
        goals = [Goal(goal_id=g.get("goal_id", ""), title=g.get("title", ""), category=g.get("category", "personal") if g.get("category") in {"career", "study", "personal"} else "personal", target_date=g.get("target_date"), why=g.get("why", ""), status="active" if g.get("status") != "completed" else "completed") for g in db.get_goals()]
        coach_state = CoachState(user_id=str(db.get_user().get("id", "student-001")), adaptive_model=adaptive)
        for goal in goals:
            field = {"study": "study_goals", "career": "career_goals", "personal": "personal_goals"}[goal.category]
            getattr(coach_state.goals_data, field).append(goal)
        reply = generate_coaching_reply_with_llm(llm, analysis, coach_state)
        burnout_risk = adaptive.burnout_risk.level

        return {
            "reflection": {
                "id": f"ref-{uuid.uuid4().hex[:6]}",
                "date": now[:10],
                "energy": energy,
                "mood": mood,
                "mood_score": analysis.mood,
                "energy_score": analysis.energy,
                "friction": analysis.friction,
                "avoidance_signals": analysis.avoidance_signals,
                "burnout_signals": analysis.burnout_signals,
                "reflection": reflection_text,
                "obstacles": obstacles,
                "ai_feedback": reply.message,
                "created_at": now
            },
            "burnout_risk": burnout_risk,
            "burnout_risk_value": adaptive.burnout_risk.score,
            "motivation_index": adaptive.motivation_index,
            "consecutive_low_days": low_days,
            "strategy_note": reply.strategy_note if reply.suggest_strategy_change else "",
        }

goal_coach = GoalCoachAgentWrapper()
