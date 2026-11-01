"""Deterministic scoring logic for motivation_index and burnout_risk.

Pure deterministic Python functions with NO LLM dependencies.
Calculates objective metrics from structured qualitative signals and completion ratios.
"""

from __future__ import annotations

from coach.schemas import BurnoutRisk, CoachAdaptive, ReflectionAnalysis, RiskLevel


def compute_consecutive_low_days(
    previous_low_days: int,
    mood: float,
    energy: float,
    burnout_signals: list[str],
) -> int:
    """Calculate the consecutive low-wellbeing days counter.

    A day counts as a low day if energy is low (<0.35), mood is significantly
    negative (< -0.2), or explicit burnout signals are present.
    """
    is_low = (energy < 0.35) or (mood < -0.2) or (len(burnout_signals) > 0)
    if is_low:
        return previous_low_days + 1
    return 0


def compute_motivation_index(
    previous_motivation: float,
    mood: float,
    energy: float,
    completion_ratio: float = 1.0,
    friction_count: int = 0,
    avoidance_count: int = 0,
) -> float:
    """Compute the updated motivation_index in the range [0.0, 1.0].

    Uses an Exponential Moving Average (EMA) to prevent extreme single-day swings
    while remaining sensitive to persistent shifts.

    Args:
        previous_motivation: Prior motivation index [0.0, 1.0].
        mood: Qualitative mood in [-1.0, 1.0].
        energy: Qualitative energy in [0.0, 1.0].
        completion_ratio: Ratio of completed habits/tasks in [0.0, 1.0].
        friction_count: Number of friction points identified.
        avoidance_count: Number of avoidance signals identified.

    Returns:
        Updated motivation index rounded to 2 decimal places.
    """
    # Normalize mood from [-1.0, 1.0] to [0.0, 1.0]
    norm_mood = max(0.0, min(1.0, (mood + 1.0) / 2.0))
    norm_energy = max(0.0, min(1.0, energy))
    norm_completion = max(0.0, min(1.0, completion_ratio))

    # Base weighted target
    base = (0.35 * norm_mood) + (0.35 * norm_energy) + (0.30 * norm_completion)

    # Friction and avoidance penalties
    penalties = (0.04 * min(friction_count, 4)) + (0.08 * min(avoidance_count, 3))
    raw_target = max(0.0, min(1.0, base - penalties))

    # EMA smoothing: 60% memory, 40% current target
    smoothed = (0.60 * previous_motivation) + (0.40 * raw_target)
    return round(max(0.0, min(1.0, smoothed)), 2)


def compute_burnout_risk(
    previous_burnout_score: float,
    mood: float,
    energy: float,
    burnout_signals_count: int,
    friction_count: int = 0,
    avoidance_count: int = 0,
    consecutive_low_days: int = 0,
    completion_ratio: float = 1.0,
) -> BurnoutRisk:
    """Compute the updated BurnoutRisk (score [0.0, 1.0] and RiskLevel).

    Burnout is driven by:
    - Energy depletion (1.0 - energy)
    - Negative affect / distress (negative mood)
    - Explicit burnout symptoms (exhaustion, numbness, dread)
    - Compounding friction and avoidance
    - Chronic low-wellbeing days
    """
    energy_deficit = max(0.0, 1.0 - energy) * 0.30
    negative_affect = max(0.0, -mood) * 0.20
    symptoms = min(burnout_signals_count, 4) * 0.15
    friction_drag = min(friction_count, 4) * 0.05
    avoidance_drag = min(avoidance_count, 3) * 0.05
    chronic_compounding = min(consecutive_low_days, 5) * 0.08
    incompletion_drag = 0.05 if completion_ratio < 0.5 else 0.0

    raw_burnout = (
        energy_deficit
        + negative_affect
        + symptoms
        + friction_drag
        + avoidance_drag
        + chronic_compounding
        + incompletion_drag
    )
    raw_target = max(0.0, min(1.0, raw_burnout))

    # EMA smoothing: 50% memory, 50% fresh signals
    smoothed_score = (0.50 * previous_burnout_score) + (0.50 * raw_target)
    final_score = round(max(0.0, min(1.0, smoothed_score)), 2)

    # Categorize into RiskLevel
    level: RiskLevel
    if final_score < 0.35:
        level = "Low"
    elif final_score < 0.70:
        level = "Medium"
    else:
        level = "High"

    return BurnoutRisk(score=final_score, level=level)


def update_adaptive_metrics(
    current_adaptive: CoachAdaptive,
    analysis: ReflectionAnalysis,
    completion_ratio: float,
    previous_consecutive_low_days: int,
) -> tuple[CoachAdaptive, int]:
    """Calculate and return new CoachAdaptive model and updated low-days counter.

    Args:
        current_adaptive: Current CoachAdaptive model.
        analysis: Structured ReflectionAnalysis output from LLM.
        completion_ratio: Ratio of completed habits/tasks.
        previous_consecutive_low_days: Count of preceding low days.

    Returns:
        tuple of (new_adaptive_model, new_consecutive_low_days).
    """
    new_low_days = compute_consecutive_low_days(
        previous_low_days=previous_consecutive_low_days,
        mood=analysis.mood,
        energy=analysis.energy,
        burnout_signals=analysis.burnout_signals,
    )

    new_motivation = compute_motivation_index(
        previous_motivation=current_adaptive.motivation_index,
        mood=analysis.mood,
        energy=analysis.energy,
        completion_ratio=completion_ratio,
        friction_count=len(analysis.friction),
        avoidance_count=len(analysis.avoidance_signals),
    )

    new_burnout = compute_burnout_risk(
        previous_burnout_score=current_adaptive.burnout_risk.score,
        mood=analysis.mood,
        energy=analysis.energy,
        burnout_signals_count=len(analysis.burnout_signals),
        friction_count=len(analysis.friction),
        avoidance_count=len(analysis.avoidance_signals),
        consecutive_low_days=new_low_days,
        completion_ratio=completion_ratio,
    )

    new_adaptive = CoachAdaptive(
        motivation_index=new_motivation,
        burnout_risk=new_burnout,
        burnout_risk_score=new_burnout.level,
    )

    return new_adaptive, new_low_days

