"""Typed, standalone tool functions for the Goal & Life Coach Agent.

Plain Python functions, completely independent, testable in isolation.
Provides:
- create_goal
- update_milestone_status
- log_reflection
- update_habit (handles streak + freeze + comeback recovery)
- award_achievement
- update_goal_strategy
- request_strategy_adjustment (-> outbound to MANAGER)
- send_milestones_to_planner (-> outbound to PLANNING)
- send_relevance_tags_to_tutor (-> outbound to TUTOR)
"""

from __future__ import annotations

import uuid
from typing import Any

from coach.gamification import award_achievement as _award_achievement
from coach.gamification import update_habit_streak
from coach.schemas import (
    Achievement,
    CoachState,
    Goal,
    GoalCategory,
    Habit,
    Milestone,
    MilestoneStatus,
    OutboundEvent,
    Reflection,
)


def create_goal(
    state: CoachState,
    title: str,
    category: GoalCategory = "personal",
    target_date: str | None = None,
    why: str = "",
    strategy_note: str = "",
    goal_id: str | None = None,
) -> tuple[CoachState, Goal]:
    """Create and register a new goal in the user's CoachState.

    Args:
        state: The current CoachState.
        title: Title/name of the goal.
        category: "career", "study", or "personal".
        target_date: Optional target completion date in ISO format.
        why: Core emotional or pragmatic reason for this goal.
        strategy_note: Initial coaching or execution strategy notes.
        goal_id: Optional ID; auto-generated if omitted.

    Returns:
        tuple of (updated CoachState, newly created Goal).
    """
    gid = goal_id or f"goal_{uuid.uuid4().hex[:8]}"
    goal = Goal(
        goal_id=gid,
        title=title,
        category=category,
        target_date=target_date,
        why=why,
        status="active",
        strategy_note=strategy_note,
    )

    if category == "career":
        state.goals_data.career_goals.append(goal)
    elif category == "study":
        state.goals_data.study_goals.append(goal)
    else:
        state.goals_data.personal_goals.append(goal)

    return state, goal


def update_milestone_status(
    state: CoachState,
    milestone_id: str,
    status: MilestoneStatus,
) -> tuple[CoachState, Milestone | None]:
    """Update a milestone's status and automatically check if its parent goal is completed.

    Args:
        state: Current CoachState.
        milestone_id: Unique milestone ID.
        status: New MilestoneStatus ('pending', 'in_progress', 'completed', 'blocked').

    Returns:
        tuple of (updated CoachState, updated Milestone or None).
    """
    target_milestone: Milestone | None = None
    for m in state.goals_data.milestones:
        if m.id == milestone_id:
            m.status = status
            target_milestone = m
            break

    if target_milestone is None:
        return state, None

    # Check parent goal completion if milestone was marked completed
    if status == "completed":
        parent_goal_id = target_milestone.goal_id
        parent_milestones = [
            m for m in state.goals_data.milestones if m.goal_id == parent_goal_id
        ]
        if parent_milestones and all(m.status == "completed" for m in parent_milestones):
            for g in state.all_goals():
                if g.goal_id == parent_goal_id:
                    g.status = "completed"

    return state, target_milestone


def log_reflection(
    state: CoachState,
    date: str,
    text: str,
    mood: float,
    energy: float,
    friction: list[str] | None = None,
    burnout_signals: list[str] | None = None,
) -> tuple[CoachState, Reflection]:
    """Log a user's daily reflection into the state history."""
    reflection = Reflection(
        date=date,
        text=text,
        mood=mood,
        energy=energy,
        friction=friction or [],
        burnout_signals=burnout_signals or [],
    )
    state.reflections.append(reflection)
    return state, reflection


def update_habit(
    state: CoachState,
    habit_name: str,
    completion_date: str,
) -> tuple[CoachState, Habit, list[Achievement], dict[str, Any]]:
    """Update or register a habit completion handling streak, freeze, and comeback recovery.

    Args:
        state: Current CoachState.
        habit_name: Name of the habit.
        completion_date: ISO date string ('YYYY-MM-DD').

    Returns:
        tuple of (updated CoachState, updated Habit, newly earned Achievements, metadata dict).
    """
    habit: Habit | None = None
    for h in state.goals_data.habits:
        if h.name.lower() == habit_name.lower():
            habit = h
            break

    if habit is None:
        habit = Habit(name=habit_name, streak=0, best_streak=0, freezes_available=1)
        state.goals_data.habits.append(habit)

    habit, new_badges, meta = update_habit_streak(
        habit=habit,
        completion_date=completion_date,
        existing_achievements=state.achievements,
    )

    for badge in new_badges:
        state.achievements.append(badge)

    return state, habit, new_badges, meta


def award_achievement(
    state: CoachState,
    key: str,
    title: str,
    earned_on: str,
) -> tuple[CoachState, Achievement | None]:
    """Award an achievement to the user if not already earned."""
    ach = _award_achievement(
        achievements=state.achievements,
        key=key,
        title=title,
        earned_on=earned_on,
    )
    return state, ach


def update_goal_strategy(
    state: CoachState,
    goal_id: str,
    strategy_note: str,
) -> tuple[CoachState, Goal | None]:
    """Update the strategic coaching note for a struggling or active goal."""
    target_goal: Goal | None = None
    for g in state.all_goals():
        if g.goal_id == goal_id:
            g.strategy_note = strategy_note
            target_goal = g
            break
    return state, target_goal


def request_strategy_adjustment(
    state: CoachState,
    goal_id: str,
    reason: str,
    suggested_action: str,
) -> OutboundEvent:
    """Create an outbound event requesting Adaptive Manager coordination for a strategy change."""
    return OutboundEvent(
        target="MANAGER",
        command="REQUEST_STRATEGY_ADJUSTMENT",
        payload={
            "user_id": state.user_id,
            "goal_id": goal_id,
            "reason": reason,
            "suggested_action": suggested_action,
            "motivation_index": state.adaptive_model.motivation_index,
            "burnout_risk": state.adaptive_model.burnout_risk.level,
            "consecutive_low_days": state.consecutive_low_days,
        },
    )


def send_milestones_to_planner(
    goal: Goal,
    milestones: list[Milestone],
) -> OutboundEvent:
    """Create an outbound event routing newly decomposed milestones to Personal Planning Agent."""
    return OutboundEvent(
        target="PLANNING",
        command="RECEIVE_GOAL_MILESTONES",
        payload={
            "goal_id": goal.goal_id,
            "goal_title": goal.title,
            "category": goal.category,
            "target_date": goal.target_date,
            "milestones": [m.model_dump() for m in milestones],
        },
    )


def send_relevance_tags_to_tutor(
    goal: Goal,
    tags: list[str],
) -> OutboundEvent:
    """Create an outbound event routing study relevance tags to Tutor / Generative Learning Agent."""
    return OutboundEvent(
        target="TUTOR",
        command="LINK_GOAL_TOPICS",
        payload={
            "goal_id": goal.goal_id,
            "category": goal.category,
            "goal_title": goal.title,
            "relevance_tags": tags,
        },
    )

