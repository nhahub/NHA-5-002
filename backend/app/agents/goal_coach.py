"""Goal & Life Coach Agent (Agent 4 of 5).
Wraps Goal-Life-Coach-Agent-main:
- Goal management (Study, Career, Personal)
- SMART Milestone breakdown
- Daily reflections, mood/energy scoring, burnout risk evaluation
- Habit tracking, streak freeze protection, comeback halving, and milestone badges
"""

from __future__ import annotations

import logging
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

from coach.schemas import Achievement, Habit, Reflection
from coach.gamification import update_habit_streak, check_streak_badges
from coach.scoring import compute_burnout_risk, compute_motivation_index
from coach.tools import create_goal as _raw_create_goal

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
    ) -> Dict[str, Any]:
        """Create a new structured goal with initial milestones."""
        goal_id = f"G-{uuid.uuid4().hex[:6].upper()}"
        m_list = []
        if milestones:
            for i, m in enumerate(milestones):
                m_list.append({
                    "id": m.get("id") or f"M-{goal_id}-{i+1}",
                    "goal_id": goal_id,
                    "title": m.get("title", f"Milestone {i+1}"),
                    "status": m.get("status", "pending"),
                    "criteria": m.get("criteria", ""),
                    "due_date": m.get("due_date", target_date),
                    "micro_tasks": m.get("micro_tasks", [])
                })
        else:
            # Default first milestone
            m_list.append({
                "id": f"M-{goal_id}-1",
                "goal_id": goal_id,
                "title": f"Initial research & setup for {title}",
                "status": "pending",
                "criteria": "Establish baseline study materials and initial diagnostic",
                "due_date": target_date,
                "micro_tasks": [
                    {"title": f"Review lecture notes for {title}", "estimated_mins": 30}
                ]
            })

        return {
            "goal_id": goal_id,
            "title": title,
            "category": category,
            "target_date": target_date,
            "why": why,
            "status": "active",
            "strategy_note": f"Structured by Life Coach for consistent habit progression.",
            "milestones": m_list,
        }

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

    def process_reflection(
        self,
        energy: int,
        mood: str,
        reflection_text: str,
        obstacles: str = "",
        historical_reflections: Optional[List[Dict[str, Any]]] = None,
        recent_blocks: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Evaluate reflection check-in, compute burnout risk and motivation index."""
        now = datetime.now(timezone.utc).isoformat()
        
        # Build reflection objects for scoring
        all_refs = [
            Reflection(
                date=r.get("date", now[:10]),
                energy=int(r.get("energy", 3)),
                mood=r.get("mood", "neutral"),
                reflection=r.get("reflection", ""),
                obstacles=r.get("obstacles", "")
            )
            for r in (historical_reflections or [])
        ]
        current_ref = Reflection(
            date=now[:10],
            energy=energy,
            mood=mood,
            reflection=reflection_text,
            obstacles=obstacles
        )
        all_refs.append(current_ref)

        burnout_risk = compute_burnout_risk(
            reflections=all_refs,
            recent_blocks=recent_blocks or []
        )
        motivation_index = compute_motivation_index(
            reflections=all_refs,
            recent_milestones=[]
        )

        # AI empathetic feedback message
        if energy <= 2 or burnout_risk == "High":
            ai_feedback = (
                "Coach Note: Noticed you are running low on energy today. "
                "The system is switching to RECOVERY_BOOST mode — lowering daily cognitive quotas, "
                "adding extra break buffers, and rescheduling non-urgent heavy tasks."
            )
        elif energy >= 4:
            ai_feedback = (
                "Coach Note: Great energy and high focus today! "
                "Optimal time to tackle challenging conceptual topics like Automata proofs during your peak window."
            )
        else:
            ai_feedback = (
                "Coach Note: Solid steady progress recorded. Remember to stick to your 35-min focus blocks "
                "and take scheduled rest intervals."
            )

        return {
            "reflection": {
                "id": f"ref-{uuid.uuid4().hex[:6]}",
                "date": now[:10],
                "energy": energy,
                "mood": mood,
                "reflection": reflection_text,
                "obstacles": obstacles,
                "ai_feedback": ai_feedback,
                "created_at": now
            },
            "burnout_risk": burnout_risk,
            "motivation_index": round(motivation_index, 2),
        }

goal_coach = GoalCoachAgentWrapper()
