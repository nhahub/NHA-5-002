"""Adaptive Manager Agent (Agent 5 of 5 — System Orchestrator & Personal Adaptive Model Engine).
The central orchestrator and PRIMARY INNOVATION (USP) of the project.
Maintains the student's Personal Adaptive Model, monitors behavior deviations across all agents,
runs the Struggle Predictor, and adapts strategies proactively.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from backend.app.config import settings

logger = logging.getLogger("adaptive_coach.manager")

DEFAULT_MODEL = {
    "effective_focus_duration_mins": 35,
    "peak_productivity_window": "09:00-11:00",
    "task_delay_factor": 1.25,
    "learning_rate_velocity": 0.65,
    "avg_task_completion_ratio": 0.75,
    "procrastination_patterns": [
        "Heavy lecture days (> 4 hrs classes)",
        "Complex formal theory topics (Automata & Proofs)",
        "Monolithic study blocks exceeding 50 minutes"
    ],
    "topic_mastery": {
        "Automata & Formal Languages": 58.0,
        "Trend AI & LLM Systems": 88.0
    },
    "weak_topics": [
        "NFA to DFA subset construction",
        "Pumping Lemma proofs"
    ],
    "burnout_risk_score": "Medium",
    "motivation_index": 0.78,
    "current_system_mode": "PREEMPTIVE_RESTRUCTURE",
}

ROUTES = {
    "GENERATE_DIAGNOSTIC_EXAM": "Assessment Agent",
    "INITIATE_REMEDIAL_SESSION": "AI Tutor Agent",
    "PREEMPTIVE_SCHEDULE_RESTRUCTURE": "Personal Planning Agent",
    "TRIGGER_ACCOUNTABILITY_CHECKIN": "Goal & Life Coach Agent",
}


def _shortfall(event: dict) -> float:
    """Compute shortfall ratio between planned and actual focus minutes."""
    if event.get("type") in {"block_missed", "task_missed"}:
        return 1.0
    planned = event.get("planned_mins") or 0
    actual = event.get("actual_focus_mins") or 0
    return max(0.0, min(1.0, 1 - actual / planned)) if planned else 0.0


def observe_events(events: list[dict]) -> dict:
    """Ingests behavior events into unified timeline metrics."""
    sorted_evs = sorted(events, key=lambda e: e.get("timestamp", ""))
    blocks = [e for e in sorted_evs if e.get("type") in {"block_completed", "block_cut_short", "block_missed", "task_completed"}]
    quizzes = [e for e in sorted_evs if e.get("type") == "quiz_result"]
    reflections = [e for e in sorted_evs if e.get("type") == "reflection"]

    focus = [
        e.get("actual_focus_mins")
        for e in blocks
        if e.get("type") != "block_missed" and e.get("actual_focus_mins") is not None
    ]
    delays = [
        e["actual_focus_mins"] / e["planned_mins"]
        for e in blocks
        if e.get("actual_focus_mins") and e.get("planned_mins")
    ]

    scores, failed = defaultdict(list), {}
    for q in quizzes:
        topic = q.get("topic") or (q.get("payload", {}).get("topic"))
        score = q.get("score_pct") or (q.get("payload", {}).get("score_pct"))
        failed_concepts = q.get("failed_concepts") or (q.get("payload", {}).get("failed_concepts")) or []
        if topic and score is not None:
            scores[topic].append(score)
            failed[topic] = failed_concepts

    energies = [r.get("energy") or r.get("payload", {}).get("energy") for r in reflections if r.get("energy") or r.get("payload", {}).get("energy")]

    return {
        "events": sorted_evs,
        "focus_samples": focus,
        "delay_ratios": delays,
        "missed_blocks": sum(e.get("type") == "block_missed" for e in blocks),
        "total_blocks": len(blocks),
        "scores": dict(scores),
        "failed_concepts": failed,
        "avg_energy": round(mean(energies), 2) if energies else None,
        "tutor_completed_topics": list(dict.fromkeys(
            e.get("topic", "") for e in sorted_evs if e.get("type") == "topic_completed" and e.get("topic")
        )),
    }


def analyze_behavior_patterns(events: list[dict]) -> list[str]:
    """Isolate procrastination triggers and friction patterns."""
    by_day = defaultdict(list)
    by_topic = defaultdict(list)
    for e in events:
        t = e.get("type")
        if t not in {"block_completed", "block_cut_short", "block_missed", "task_completed"}:
            continue
        val = _shortfall(e)
        day_type = e.get("day_type") or e.get("payload", {}).get("day_type") or "normal"
        by_day[day_type].append(val)
        topic = e.get("topic") or e.get("payload", {}).get("topic")
        if topic:
            by_topic[topic].append(val)

    patterns = []
    if by_day.get("heavy_lecture") and mean(by_day["heavy_lecture"]) >= 0.35:
        patterns.append("Heavy lecture days (> 4 hrs classes)")

    for topic, values in by_topic.items():
        if len(values) >= 1 and mean(values) >= 0.35:
            patterns.append(f"Difficult topic: {topic}")

    if not patterns:
        patterns = [
            "Heavy lecture days (> 4 hrs classes)",
            "Complex formal theory topics (Automata & Proofs)"
        ]
    return patterns[:5]


def predict_system_risk(events: list[dict]) -> dict:
    """Predict risk levels and consecutive risk trends."""
    days = defaultdict(lambda: {"signals": [], "type": None})
    for e in events:
        ts = e.get("timestamp", "")
        day = ts[:10] if len(ts) >= 10 else datetime.now().strftime("%Y-%m-%d")
        t = e.get("type")
        if t == "reflection" and e.get("energy"):
            energy = e.get("energy") or e.get("payload", {}).get("energy") or 3
            days[day]["signals"].append(max(0.0, (5 - energy) / 4.0))
        elif t in {"block_completed", "block_cut_short", "block_missed"}:
            days[day]["signals"].append(_shortfall(e))
            days[day]["type"] = e.get("day_type")

    daily = {d: round(mean(v["signals"]), 2) for d, v in sorted(days.items()) if v["signals"]}
    consecutive = 0
    if daily:
        observed_dates = sorted(date.fromisoformat(day) for day in daily)
        expected_day = observed_dates[-1]
        for observed_day in reversed(observed_dates):
            if observed_day != expected_day:
                break
            if daily[observed_day.isoformat()] < settings.risk_alert_threshold:
                break
            consecutive += 1
            expected_day = observed_day - timedelta(days=1)

    overall = mean(daily.values()) if daily else 0.0
    return {
        "daily_risk": daily,
        "risk_score": round(list(daily.values())[-1], 2) if daily else 0.45,
        "consecutive_risk_days": consecutive,
        "overall_risk": round(overall, 2)
    }


def adapt_model_parameters(model: dict, metrics: dict, risk: dict) -> tuple[dict, int]:
    """Exponential Moving Average (EMA) update on personal model."""
    m = {**DEFAULT_MODEL, **model}
    a = settings.ema_alpha

    if metrics.get("focus_samples"):
        recent_focus = mean(metrics["focus_samples"])
        m["effective_focus_duration_mins"] = round(a * recent_focus + (1 - a) * m["effective_focus_duration_mins"])

    if metrics.get("delay_ratios"):
        recent_delay = mean(metrics["delay_ratios"])
        m["task_delay_factor"] = round(a * recent_delay + (1 - a) * m["task_delay_factor"], 2)

    # Learning velocity
    if metrics.get("scores"):
        gains = [
            (scores[-1] - scores[0]) / (len(scores) - 1) / 100.0
            for scores in metrics["scores"].values()
            if len(scores) >= 2
        ]
        if gains:
            observed_velocity = max(0.0, min(1.0, 0.5 + mean(gains)))
            m["learning_rate_velocity"] = round(a * observed_velocity + (1 - a) * m["learning_rate_velocity"], 2)

    # Determine System Strategy Mode
    weak = [t for t, scores in metrics.get("scores", {}).items() if scores[-1] < settings.mastery_threshold]
    risk_score = risk.get("risk_score", 0.0)

    if risk.get("consecutive_risk_days", 0) >= settings.alert_consecutive_days or risk_score >= settings.risk_alert_threshold:
        m["current_system_mode"] = "PREEMPTIVE_RESTRUCTURE"
    elif weak:
        m["current_system_mode"] = "REMEDIAL_FOCUS"
    elif risk_score >= 0.40:
        m["current_system_mode"] = "PREEMPTIVE_RESTRUCTURE"
    else:
        m["current_system_mode"] = "NORMAL"

    block = max(15, min(45, round(m["effective_focus_duration_mins"] / 5) * 5))
    return m, block


class AdaptiveManagerWrapper:
    """Orchestrator maintaining the Personal Adaptive Model."""

    def run_adaptive_cycle(
        self,
        current_model: Dict[str, Any],
        events: List[Dict[str, Any]],
        upcoming_tasks: Optional[List[Dict[str, Any]]] = None,
        university_schedule: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Execute closed feedback loop and return adapted model and directives."""
        # 1. Observe
        metrics = observe_events(events)

        # 2. Understand
        patterns = analyze_behavior_patterns(events)

        # 3. Predict
        risk = predict_system_risk(events)

        # 4. Adapt
        updated_model, block_size = adapt_model_parameters(current_model, metrics, risk)
        updated_model["procrastination_patterns"] = patterns

        # Update topic masteries from quiz scores
        if metrics.get("scores"):
            mastery = dict(updated_model.get("topic_mastery", {}))
            for t, scores in metrics["scores"].items():
                mastery[t] = round(scores[-1], 1)
            updated_model["topic_mastery"] = mastery

        if metrics.get("failed_concepts"):
            all_weak = []
            for concepts in metrics["failed_concepts"].values():
                all_weak.extend(concepts)
            if all_weak:
                updated_model["weak_topics"] = list(dict.fromkeys(all_weak))

        # 5. Struggle Predictions (Section 10)
        predictions = self.compute_struggle_predictions(
            adaptive_model=updated_model,
            upcoming_tasks=upcoming_tasks or [],
            university_schedule=university_schedule or [],
            metrics=metrics
        )

        # 6. Dispatch Commands
        commands = []
        for topic in dict.fromkeys(metrics.get("tutor_completed_topics", [])):
            commands.append({
                "command_id": f"cmd-{uuid.uuid4().hex[:6]}",
                "command": "GENERATE_DIAGNOSTIC_EXAM",
                "target_agent": ROUTES["GENERATE_DIAGNOSTIC_EXAM"],
                "reason": f"Tutor completed topic '{topic}'",
                "status": "APPROVED",
                "created_at": datetime.now(timezone.utc).isoformat()
            })

        for topic, scores in metrics.get("scores", {}).items():
            if scores[-1] < settings.mastery_threshold:
                commands.append({
                    "command_id": f"cmd-{uuid.uuid4().hex[:6]}",
                    "command": "INITIATE_REMEDIAL_SESSION",
                    "target_agent": ROUTES["INITIATE_REMEDIAL_SESSION"],
                    "reason": f"Mastery of '{topic}' is {scores[-1]}% (< {settings.mastery_threshold}% threshold)",
                    "status": "APPROVED",
                    "created_at": datetime.now(timezone.utc).isoformat()
                })

        high_risk_preds = [p for p in predictions if p.get("risk_level") == "High"]
        if high_risk_preds:
            updated_model["current_system_mode"] = "PREEMPTIVE_RESTRUCTURE"
            for p in high_risk_preds:
                commands.append({
                    "command_id": f"cmd-{uuid.uuid4().hex[:6]}",
                    "command": "PREEMPTIVE_SCHEDULE_RESTRUCTURE",
                    "target_agent": ROUTES["PREEMPTIVE_SCHEDULE_RESTRUCTURE"],
                    "reason": f"High risk predicted for {p['target_id']}: {p['rationale']}",
                    "status": "APPROVED",
                    "created_at": datetime.now(timezone.utc).isoformat()
                })

        return {
            "metrics": metrics,
            "patterns": patterns,
            "risk_summary": risk,
            "adaptive_model": updated_model,
            "recommended_block_mins": block_size,
            "predictions": predictions,
            "dispatch_commands": commands,
        }

    def compute_struggle_predictions(
        self,
        adaptive_model: Dict[str, Any],
        upcoming_tasks: List[Dict[str, Any]],
        university_schedule: List[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Compute Task-miss, Topic-struggle, and Overload risks."""
        predictions = []
        now_iso = datetime.now(timezone.utc).isoformat()
        delay_factor = float(adaptive_model.get("task_delay_factor", 1.25))
        focus_mins = int(adaptive_model.get("effective_focus_duration_mins", 35))

        # 1. Target 1: Task-miss risk
        for task in upcoming_tasks:
            if task.get("status") in {"completed", "missed"}:
                continue
            est = int(task.get("estimated_mins", 30))
            is_large = (est > focus_mins * 1.2)
            sched_date = task.get("scheduled_date") or ""

            lectures_that_day = [l for l in university_schedule if l.get("date") == sched_date]
            lec_hours = len(lectures_that_day) * 2

            risk_score = 0.20
            reasons = []

            if lec_hours >= 4:
                risk_score += 0.38
                reasons.append(f"Heavy lecture load ({lec_hours}h scheduled on {sched_date})")
            if is_large:
                risk_score += 0.28
                reasons.append(f"Estimated duration ({est}m) exceeds attention threshold ({focus_mins}m)")
            if delay_factor >= 1.25:
                risk_score += 0.14
                reasons.append(f"Historical task delay factor is {delay_factor}x")

            risk_score = min(0.95, round(risk_score, 2))
            level = "High" if risk_score >= 0.65 else ("Medium" if risk_score >= 0.40 else "Low")

            if level in {"High", "Medium"}:
                predictions.append({
                    "id": f"pred-{uuid.uuid4().hex[:6]}",
                    "target": "task_miss_risk",
                    "target_id": task.get("task_id", task.get("title", "Task")),
                    "risk_score": risk_score,
                    "risk_level": level,
                    "rationale": "; ".join(reasons),
                    "preemptive_action": f"Deconstruct task into micro-tasks matching {focus_mins}m focus span and reschedule to peak window.",
                    "status": "active",
                    "created_at": now_iso
                })

        # 2. Target 2: Topic-struggle risk
        mastery_dict = adaptive_model.get("topic_mastery", {})
        weak_topics = adaptive_model.get("weak_topics", [])
        for topic, score in mastery_dict.items():
            if score < settings.mastery_threshold:
                prob = round(min(0.95, (100.0 - score) / 100.0 + 0.15), 2)
                predictions.append({
                    "id": f"pred-{uuid.uuid4().hex[:6]}",
                    "target": "topic_struggle_risk",
                    "target_id": topic,
                    "risk_score": prob,
                    "risk_level": "High" if prob >= 0.65 else "Medium",
                    "rationale": f"Mastery in '{topic}' is {score}% (< {settings.mastery_threshold}%). Unresolved weak concepts: {', '.join(weak_topics[:2]) or 'Basic mechanics'}.",
                    "preemptive_action": "Dispatched AI Tutor for targeted concept breakdown and prepared diagnostic practice.",
                    "status": "active",
                    "created_at": now_iso
                })

        # 3. Target 3: Overload risk
        by_date = {}
        for l in university_schedule:
            d = l.get("date") or "today"
            by_date[d] = by_date.get(d, 0) + 120

        for d, mins in by_date.items():
            if mins >= 240:
                predictions.append({
                    "id": f"pred-{uuid.uuid4().hex[:6]}",
                    "target": "overload_risk",
                    "target_id": f"Date: {d}",
                    "risk_score": 0.70,
                    "risk_level": "High",
                    "rationale": f"Scheduled classes total {mins // 60} hours on {d}, creating high fatigue risk for independent study.",
                    "preemptive_action": "Restrict evening study quota to maximum 45 minutes of light flashcard review.",
                    "status": "active",
                    "created_at": now_iso
                })

        return predictions

adaptive_manager = AdaptiveManagerWrapper()
