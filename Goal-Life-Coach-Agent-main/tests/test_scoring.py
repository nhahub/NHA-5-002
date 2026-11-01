"""Unit tests for deterministic motivation_index and burnout_risk scoring.

Validates that mathematical scoring functions are pure, bounded within [0.0, 1.0],
and accurately reflect cognitive and behavioral signals without any LLM hallucination.
"""

from __future__ import annotations

from coach.schemas import CoachAdaptive, ReflectionAnalysis
from coach.scoring import (
    compute_burnout_risk,
    compute_consecutive_low_days,
    compute_motivation_index,
    update_adaptive_metrics,
)


def test_consecutive_low_days_counting() -> None:
    """Low energy, negative mood, or burnout signals correctly advance the low-days counter."""
    # Normal healthy day -> resets to 0
    assert compute_consecutive_low_days(
        previous_low_days=2, mood=0.4, energy=0.7, burnout_signals=[]
    ) == 0

    # Low energy day
    assert compute_consecutive_low_days(
        previous_low_days=0, mood=0.0, energy=0.25, burnout_signals=[]
    ) == 1

    # Negative mood day
    assert compute_consecutive_low_days(
        previous_low_days=1, mood=-0.5, energy=0.6, burnout_signals=[]
    ) == 2

    # Day with burnout signals
    assert compute_consecutive_low_days(
        previous_low_days=2, mood=0.1, energy=0.5, burnout_signals=["Dread"]
    ) == 3


def test_motivation_index_calculation_and_bounds() -> None:
    """Motivation index stays in [0.0, 1.0], rewards progress, and penalizes friction/avoidance."""
    # High performance day
    high_m = compute_motivation_index(
        previous_motivation=0.6,
        mood=0.8,
        energy=0.9,
        completion_ratio=1.0,
        friction_count=0,
        avoidance_count=0,
    )
    assert 0.70 <= high_m <= 1.0

    # Low performance day with friction and avoidance
    low_m = compute_motivation_index(
        previous_motivation=0.6,
        mood=-0.8,
        energy=0.1,
        completion_ratio=0.0,
        friction_count=3,
        avoidance_count=2,
    )
    assert 0.0 <= low_m < 0.40

    # Boundary safety tests
    assert 0.0 <= compute_motivation_index(0.0, -1.0, 0.0, 0.0, 10, 10) <= 1.0
    assert 0.0 <= compute_motivation_index(1.0, 1.0, 1.0, 1.0, 0, 0) <= 1.0


def test_burnout_risk_scoring_and_levels() -> None:
    """Burnout risk correctly scales and maps to Low, Medium, and High levels."""
    # Healthy session -> Low risk (< 0.35)
    healthy_risk = compute_burnout_risk(
        previous_burnout_score=0.2,
        mood=0.6,
        energy=0.8,
        burnout_signals_count=0,
        friction_count=0,
        avoidance_count=0,
        consecutive_low_days=0,
    )
    assert healthy_risk.score < 0.35
    assert healthy_risk.level == "Low"

    # Moderate fatigue session -> Medium risk (0.35 <= score < 0.70)
    medium_risk = compute_burnout_risk(
        previous_burnout_score=0.3,
        mood=-0.2,
        energy=0.3,
        burnout_signals_count=1,
        friction_count=2,
        avoidance_count=1,
        consecutive_low_days=1,
    )
    assert 0.35 <= medium_risk.score < 0.70
    assert medium_risk.level == "Medium"

    # Critical compounding burnout -> High risk (>= 0.70)
    severe_risk = compute_burnout_risk(
        previous_burnout_score=0.7,
        mood=-0.8,
        energy=0.1,
        burnout_signals_count=3,
        friction_count=3,
        avoidance_count=2,
        consecutive_low_days=4,
        completion_ratio=0.2,
    )
    assert severe_risk.score >= 0.70
    assert severe_risk.level == "High"


def test_update_adaptive_metrics_integration() -> None:
    """Validates full update_adaptive_metrics cycle."""
    current_adaptive = CoachAdaptive(
        motivation_index=0.6,
    )
    analysis = ReflectionAnalysis(
        mood=0.7,
        energy=0.8,
        friction=["Noisy environment"],
        avoidance_signals=[],
        burnout_signals=[],
        summary="Productive work session despite background noise.",
    )

    new_adaptive, low_days = update_adaptive_metrics(
        current_adaptive=current_adaptive,
        analysis=analysis,
        completion_ratio=1.0,
        previous_consecutive_low_days=0,
    )

    assert 0.0 <= new_adaptive.motivation_index <= 1.0
    assert 0.0 <= new_adaptive.burnout_risk.score <= 1.0
    assert new_adaptive.burnout_risk_score in ("Low", "Medium", "High")
    assert low_days == 0

