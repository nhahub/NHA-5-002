from collections import defaultdict
from datetime import date, timedelta
from statistics import mean
from app.core.config import settings

DEFAULT_MODEL = {
    "effective_focus_duration_mins": 45,
    "procrastination_patterns": [],
    "task_delay_factor": 1.0,
    "learning_rate_velocity": 0.5,
    "current_system_mode": "NORMAL",
}
ROUTES = {
    "GENERATE_DIAGNOSTIC_EXAM": "Assessment Agent",
    "INITIATE_REMEDIAL_SESSION": "AI Tutor Agent",
    "PREEMPTIVE_SCHEDULE_RESTRUCTURE": "Personal Planning Agent",
    "TRIGGER_ACCOUNTABILITY_CHECKIN": "Goal & Life Coach Agent",
}

def _shortfall(event: dict) -> float:
    if event["type"] == "block_missed":
        return 1.0
    planned = event.get("planned_mins") or 0
    actual = event.get("actual_focus_mins") or 0
    return max(0.0, min(1.0, 1 - actual / planned)) if planned else 0.0

def observe(events: list[dict]) -> dict:
    events = sorted(events, key=lambda e: e["timestamp"])
    blocks = [e for e in events if e["type"] in {"block_completed", "block_cut_short", "block_missed"}]
    tasks = [e for e in events if e["type"] == "task_completed"]
    quizzes = [e for e in events if e["type"] == "quiz_result"]
    reflections = [e for e in events if e["type"] == "reflection"]
    focus = [e["actual_focus_mins"] for e in blocks if e["type"] != "block_missed" and e.get("actual_focus_mins") is not None]
    delays = [e["actual_mins"] / e["planned_mins"] for e in tasks if e.get("actual_mins") and e.get("planned_mins")]
    scores, failed = defaultdict(list), {}
    for q in quizzes:
        if q.get("topic") and q.get("score_pct") is not None:
            scores[q["topic"]].append(q["score_pct"])
            failed[q["topic"]] = q.get("failed_concepts") or []
    energies = [r["energy"] for r in reflections if r.get("energy") is not None]
    return {
        "events": events,
        "focus_samples": focus,
        "delay_ratios": delays,
        "missed_blocks": sum(e["type"] == "block_missed" for e in blocks),
        "total_blocks": len(blocks),
        "scores": dict(scores),
        "failed_concepts": failed,
        "avg_energy": round(mean(energies), 2) if energies else None,
        "tutor_completed_topics": list(dict.fromkeys(
            e["topic"] for e in events if e["type"] == "topic_completed" and e.get("topic")
        )),
    }

def analyze_patterns(events: list[dict]) -> list[str]:
    by_day, by_topic = defaultdict(list), defaultdict(list)
    for e in events:
        if e["type"] not in {"block_completed", "block_cut_short", "block_missed"}:
            continue
        value = _shortfall(e)
        by_day[e.get("day_type") or "normal"].append(value)
        if e.get("topic"):
            by_topic[e["topic"]].append(value)
    patterns = []
    if by_day.get("heavy_lecture") and mean(by_day["heavy_lecture"]) >= 0.4:
        patterns.append("Heavy lecture days")
    for topic, values in by_topic.items():
        if len(values) >= 2 and mean(values) >= 0.4:
            patterns.append(f"Difficult topic: {topic}")
    return patterns[:5]

def predict(events: list[dict]) -> dict:
    days = defaultdict(lambda: {"signals": [], "type": None})
    for e in events:
        day = e["timestamp"][:10]
        if e["type"] == "reflection" and e.get("procrastination_risk") is not None:
            days[day]["signals"].append(e["procrastination_risk"])
        elif e["type"] in {"block_completed", "block_cut_short", "block_missed"}:
            days[day]["signals"].append(_shortfall(e))
            days[day]["type"] = e.get("day_type")
    daily = {d: round(mean(v["signals"]), 2) for d, v in sorted(days.items()) if v["signals"]}
    # Consecutive means adjacent calendar days, not merely adjacent observed days.
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
    risk_by_type = defaultdict(list)
    for day, values in days.items():
        if day in daily:
            risk_by_type[values["type"] or "normal"].append(daily[day])
    overall = mean(daily.values()) if daily else 0.0
    predictions = []
    for e in events:
        if e["type"] != "upcoming_load":
            continue
        kind = "heavy_lecture" if (e.get("lecture_hours") or 0) >= settings.heavy_lecture_hours else "light"
        expected = mean(risk_by_type[kind]) if risk_by_type[kind] else overall
        if e.get("exam"):
            expected = min(1.0, expected + 0.1)
        if expected >= settings.risk_alert_threshold:
            predictions.append({"date": e["timestamp"][:10], "expected_risk": round(expected, 2), "reason": f"{e.get('lecture_hours', 0)}h of lectures" + (f" and exam ({e['exam']})" if e.get("exam") else "")})
    return {"daily_risk": daily, "risk_score": round(list(daily.values())[-1], 2) if daily else 0.0, "consecutive_risk_days": consecutive, "predictions": predictions}

def adapt(model: dict, metrics: dict, risk: dict) -> tuple[dict, int]:
    m = {**DEFAULT_MODEL, **model}
    a = settings.ema_alpha
    if metrics["focus_samples"]:
        m["effective_focus_duration_mins"] = round(a * mean(metrics["focus_samples"]) + (1-a) * m["effective_focus_duration_mins"])
    if metrics["delay_ratios"]:
        m["task_delay_factor"] = round(a * mean(metrics["delay_ratios"]) + (1-a) * m["task_delay_factor"], 2)
    # A single score measures current mastery, not learning velocity. Only
    # update velocity when at least two attempts exist for a topic.
    if metrics["scores"]:
        gains = [(scores[-1] - scores[0]) / (len(scores) - 1) / 100
                 for scores in metrics["scores"].values() if len(scores) >= 2]
        if gains:
            # 0.5 is neutral; positive/negative score movement shifts the
            # bounded velocity estimate up/down from that neutral baseline.
            observed_velocity = max(0.0, min(1.0, 0.5 + mean(gains)))
            m["learning_rate_velocity"] = round(
                a * observed_velocity + (1 - a) * m["learning_rate_velocity"], 2
            )
    weak = [t for t, scores in metrics["scores"].items() if scores[-1] < settings.mastery_threshold]
    if risk["predictions"] or risk["consecutive_risk_days"] >= settings.alert_consecutive_days:
        m["current_system_mode"] = "PREEMPTIVE_RESTRUCTURE"
    elif weak:
        m["current_system_mode"] = "REMEDIAL_FOCUS"
    elif risk["risk_score"] >= 0.4:
        m["current_system_mode"] = "MONITOR"
    else:
        m["current_system_mode"] = "NORMAL"
    block = max(15, min(45, round(m["effective_focus_duration_mins"] / 5) * 5))
    return m, block

def create_commands(metrics: dict, risk: dict) -> list[dict]:
    commands = []
    for topic in dict.fromkeys(metrics["tutor_completed_topics"]):
        commands.append(("GENERATE_DIAGNOSTIC_EXAM", f"Tutor completed topic '{topic}'"))
    for topic, scores in metrics["scores"].items():
        if scores[-1] < settings.mastery_threshold and not metrics["failed_concepts"].get(topic):
            commands.append(("GENERATE_DIAGNOSTIC_EXAM", f"Mastery of '{topic}' is {scores[-1]}%"))
    for topic, concepts in metrics["failed_concepts"].items():
        if concepts:
            commands.append(("INITIATE_REMEDIAL_SESSION", f"Failed concepts in '{topic}': {', '.join(concepts)}"))
    if risk["predictions"]:
        dates = ", ".join(p["date"] for p in risk["predictions"])
        commands.append(("PREEMPTIVE_SCHEDULE_RESTRUCTURE", f"High-risk upcoming days: {dates}"))
    if risk["consecutive_risk_days"] >= settings.alert_consecutive_days:
        commands.append(("TRIGGER_ACCOUNTABILITY_CHECKIN", f"Risk exceeded {settings.risk_alert_threshold} for {risk['consecutive_risk_days']} consecutive days"))
    return [{"command": cmd, "target_agent": ROUTES[cmd], "reason": reason, "context": reason, "status": "PENDING"} for cmd, reason in commands]
