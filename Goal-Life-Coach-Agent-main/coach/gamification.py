"""Gamification, streak tracking, streak freezes, and achievement logic.

Deterministic pure functions with NO LLM dependencies.
Implements the exact streak, freeze, comeback, and badge specification:
- Completed yesterday (or consecutive day) -> streak + 1.
- Missed day + freeze available -> consume freeze, streak continues unbroken (streak + 1).
- Missed day + no freeze -> comeback rule: halve streak (minimum 1), never reset to zero.
- Freeze replenishment: 1 freeze awarded every 7 unbroken days (cap 2 stored).
- Milestone badges awarded at 3, 7, 14, and 30-day streaks.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from coach.schemas import Achievement, Habit

# Streak milestone thresholds and their badge titles
STREAK_BADGES: dict[int, tuple[str, str]] = {
    3: ("streak_3", "3-Day Streak: Consistency Starter"),
    7: ("streak_7", "7-Day Streak: Habit Builder"),
    14: ("streak_14", "14-Day Streak: Momentum Master"),
    30: ("streak_30", "30-Day Streak: Habit Hero"),
}

MAX_FREEZES = 2


def check_streak_badges(
    streak: int,
    earned_on: str,
    existing_keys: set[str] | None = None,
) -> list[Achievement]:
    """Check if the current streak qualifies for any new milestone badges.

    Args:
        streak: The current habit streak.
        earned_on: ISO date string when the achievement was achieved.
        existing_keys: Set of achievement keys already held by the user.

    Returns:
        List of newly unlocked Achievement objects.
    """
    known = existing_keys or set()
    new_achievements: list[Achievement] = []

    for threshold, (key, title) in STREAK_BADGES.items():
        if streak >= threshold and key not in known:
            new_achievements.append(
                Achievement(key=key, title=title, earned_on=earned_on)
            )

    return new_achievements


def award_achievement(
    achievements: list[Achievement],
    key: str,
    title: str,
    earned_on: str,
) -> Achievement | None:
    """Award a specific achievement if not already earned.

    Mutates the achievements list in-place and returns the new Achievement,
    or None if already present.
    """
    if any(a.key == key for a in achievements):
        return None
    new_ach = Achievement(key=key, title=title, earned_on=earned_on)
    achievements.append(new_ach)
    return new_ach


def update_habit_streak(
    habit: Habit,
    completion_date: str,
    existing_achievements: list[Achievement] | None = None,
) -> tuple[Habit, list[Achievement], dict[str, Any]]:
    """Update a habit's streak and freezes deterministically for a given completion date.

    Args:
        habit: The Habit model to update.
        completion_date: ISO date string ('YYYY-MM-DD').
        existing_achievements: Optional list of user achievements to prevent duplicate badges.

    Returns:
        tuple containing:
        - Updated Habit object (mutated copy)
        - List of newly earned Achievement objects
        - Metadata dictionary detailing what happened (e.g., freeze used, comeback applied)
    """
    curr = date.fromisoformat(completion_date)
    existing_keys = {a.key for a in (existing_achievements or [])}
    new_badges: list[Achievement] = []
    meta: dict[str, Any] = {
        "action": "completed",
        "freeze_consumed": False,
        "freeze_awarded": False,
        "comeback_applied": False,
        "previous_streak": habit.streak,
    }

    # Case 1: First time completion
    if habit.last_completed is None:
        habit.streak = 1
        habit.best_streak = max(habit.best_streak, habit.streak)
        habit.last_completed = completion_date
        meta["status"] = "first_completion"
        new_badges = check_streak_badges(habit.streak, completion_date, existing_keys)
        return habit, new_badges, meta

    prev = date.fromisoformat(habit.last_completed)
    day_diff = (curr - prev).days

    # Case 2: Already logged today
    if day_diff == 0:
        meta["status"] = "already_completed_today"
        return habit, [], meta

    # Case 3: Out-of-order past date
    if day_diff < 0:
        meta["status"] = "past_date_ignored"
        return habit, [], meta

    # Case 4: Consecutive day (Done yesterday)
    if day_diff == 1:
        habit.streak += 1
        meta["status"] = "consecutive_day"

        # Award freeze back every 7 unbroken days (capped at 2)
        if habit.streak % 7 == 0 and habit.freezes_available < MAX_FREEZES:
            habit.freezes_available += 1
            meta["freeze_awarded"] = True

    # Case 5: Missed days
    else:
        missed_days = day_diff - 1

        # If freezes can cover the missed days
        if habit.freezes_available >= missed_days:
            habit.freezes_available -= missed_days
            habit.streak += 1
            meta["freeze_consumed"] = True
            meta["status"] = "freeze_consumed_unbroken"

            # Check freeze award on unbroken streak continuation
            if habit.streak % 7 == 0 and habit.freezes_available < MAX_FREEZES:
                habit.freezes_available += 1
                meta["freeze_awarded"] = True

        else:
            # Comeback rule: halve streak (minimum 1), never reset to zero
            habit.streak = max(1, habit.streak // 2)
            meta["comeback_applied"] = True
            meta["status"] = "comeback_recovery"

    habit.best_streak = max(habit.best_streak, habit.streak)
    habit.last_completed = completion_date
    meta["new_streak"] = habit.streak

    new_badges = check_streak_badges(habit.streak, completion_date, existing_keys)
    return habit, new_badges, meta


def calculate_habit_completion_ratio(
    habits: list[Habit],
    target_date: str,
) -> float:
    """Calculate the ratio of habits completed on the specified date.

    If no habits exist, returns 1.0 (neutral, no penalty).
    """
    if not habits:
        return 1.0
    completed = sum(1 for h in habits if h.last_completed == target_date)
    return completed / len(habits)

