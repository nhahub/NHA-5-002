"""Unit tests for gamification, streak, freeze, comeback, and badge logic.

Completely deterministic with zero external dependencies and no LLM calls.
"""

from __future__ import annotations

from coach.gamification import (
    award_achievement,
    calculate_habit_completion_ratio,
    check_streak_badges,
    update_habit_streak,
)
from coach.schemas import Achievement, Habit


def test_first_habit_completion() -> None:
    """A new habit starts at streak=1 on its first log."""
    habit = Habit(name="Deep Work", streak=0, best_streak=0, freezes_available=1)
    updated, badges, meta = update_habit_streak(habit, "2026-10-01")

    assert updated.streak == 1
    assert updated.best_streak == 1
    assert updated.last_completed == "2026-10-01"
    assert updated.freezes_available == 1
    assert meta["status"] == "first_completion"
    assert len(badges) == 0


def test_consecutive_day_streak_increment() -> None:
    """Completing habit on consecutive days increments the streak."""
    habit = Habit(
        name="Study AI",
        streak=1,
        best_streak=1,
        last_completed="2026-10-01",
        freezes_available=1,
    )
    updated, _, meta = update_habit_streak(habit, "2026-10-02")

    assert updated.streak == 2
    assert updated.best_streak == 2
    assert updated.last_completed == "2026-10-02"
    assert meta["status"] == "consecutive_day"


def test_same_day_idempotency() -> None:
    """Logging completion twice on the same day does not duplicate streak increments."""
    habit = Habit(
        name="Exercise",
        streak=4,
        best_streak=4,
        last_completed="2026-10-05",
        freezes_available=1,
    )
    updated, badges, meta = update_habit_streak(habit, "2026-10-05")

    assert updated.streak == 4
    assert meta["status"] == "already_completed_today"
    assert len(badges) == 0


def test_missed_day_with_freeze_available() -> None:
    """Missing a day when a freeze is available consumes the freeze and preserves streak unbroken."""
    habit = Habit(
        name="Reading",
        streak=5,
        best_streak=5,
        last_completed="2026-10-10",
        freezes_available=1,
    )
    # Missed Oct 11, completed on Oct 12 (day_diff = 2)
    updated, _, meta = update_habit_streak(habit, "2026-10-12")

    assert meta["freeze_consumed"] is True
    assert meta["status"] == "freeze_consumed_unbroken"
    assert updated.freezes_available == 0
    assert updated.streak == 6  # Streak continued unbroken!
    assert updated.best_streak == 6
    assert updated.last_completed == "2026-10-12"


def test_missed_day_with_no_freeze_triggers_comeback_rule() -> None:
    """Missing a day without a freeze halves the streak (min 1) and never resets to zero."""
    habit = Habit(
        name="Coding",
        streak=6,
        best_streak=6,
        last_completed="2026-10-10",
        freezes_available=0,
    )
    # Missed Oct 11, completed Oct 12 with 0 freezes
    updated, _, meta = update_habit_streak(habit, "2026-10-12")

    assert meta["comeback_applied"] is True
    assert meta["status"] == "comeback_recovery"
    # Halved: 6 // 2 = 3
    assert updated.streak == 3
    assert updated.best_streak == 6  # Preserves all-time best
    assert updated.last_completed == "2026-10-12"


def test_comeback_rule_minimum_one() -> None:
    """Comeback rule guarantees minimum streak of 1 even for low streak numbers."""
    habit = Habit(
        name="Meditation",
        streak=1,
        best_streak=1,
        last_completed="2026-10-10",
        freezes_available=0,
    )
    updated, _, meta = update_habit_streak(habit, "2026-10-13")

    assert meta["comeback_applied"] is True
    # max(1, 1 // 2) == 1
    assert updated.streak == 1


def test_freeze_replenishment_every_seven_days_and_cap() -> None:
    """Award a freeze back every 7 unbroken days, capped at 2 stored freezes."""
    habit = Habit(
        name="Math",
        streak=6,
        best_streak=6,
        last_completed="2026-10-06",
        freezes_available=0,
    )
    # Day 7 reached
    updated, _, meta = update_habit_streak(habit, "2026-10-07")
    assert updated.streak == 7
    assert updated.freezes_available == 1
    assert meta["freeze_awarded"] is True

    # Check freeze cap at 2
    habit_capped = Habit(
        name="Math",
        streak=13,
        best_streak=13,
        last_completed="2026-10-13",
        freezes_available=2,
    )
    updated_capped, _, _ = update_habit_streak(habit_capped, "2026-10-14")
    assert updated_capped.streak == 14
    assert updated_capped.freezes_available == 2  # Does not exceed MAX_FREEZES (2)


def test_streak_badges_awarded_at_milestones() -> None:
    """Badges are awarded at 3, 7, 14, and 30 day streaks without duplicates."""
    habit = Habit(
        name="Study",
        streak=2,
        best_streak=2,
        last_completed="2026-10-02",
        freezes_available=1,
    )

    # Reaching day 3
    updated, badges_3, _ = update_habit_streak(habit, "2026-10-03")
    assert updated.streak == 3
    assert len(badges_3) == 1
    assert badges_3[0].key == "streak_3"

    # Next consecutive day (day 4) -> no new badge
    updated, badges_4, _ = update_habit_streak(
        updated, "2026-10-04", existing_achievements=badges_3
    )
    assert updated.streak == 4
    assert len(badges_4) == 0


def test_calculate_habit_completion_ratio() -> None:
    """Computes exact proportion of habits logged on a given date."""
    h1 = Habit(name="A", last_completed="2026-10-15")
    h2 = Habit(name="B", last_completed="2026-10-15")
    h3 = Habit(name="C", last_completed="2026-10-14")

    ratio = calculate_habit_completion_ratio([h1, h2, h3], "2026-10-15")
    assert round(ratio, 2) == 0.67

    # Empty list returns 1.0 (no penalty)
    assert calculate_habit_completion_ratio([], "2026-10-15") == 1.0


def test_award_achievement_helper() -> None:
    """Helper awards unique achievements and ignores duplicates."""
    achievements: list[Achievement] = []
    a1 = award_achievement(achievements, "early_bird", "Early Bird", "2026-10-01")
    assert a1 is not None
    assert len(achievements) == 1

    # Attempting duplicate
    a2 = award_achievement(achievements, "early_bird", "Early Bird", "2026-10-02")
    assert a2 is None
    assert len(achievements) == 1

