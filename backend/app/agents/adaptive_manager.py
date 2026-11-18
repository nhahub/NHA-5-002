"""Adapter that runs the approved Adaptive Manager Agent's native analysis."""
from __future__ import annotations

import importlib
import importlib.util
import sys
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from backend.app.config import settings

ROOT = Path(__file__).resolve().parents[3]
NATIVE_APP = ROOT / "adaptive-manager-agent" / "app"
_PACKAGE = "adaptive_manager_native"
if _PACKAGE not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        _PACKAGE, NATIVE_APP / "__init__.py", submodule_search_locations=[str(NATIVE_APP)]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE] = package
    spec.loader.exec_module(package)
native_analysis = importlib.import_module(f"{_PACKAGE}.services.analysis")

ROUTES = native_analysis.ROUTES


def _flatten_event(event: Dict[str, Any]) -> Dict[str, Any]:
    return {**(event.get("payload") or {}), **{k: v for k, v in event.items() if k != "payload"}}


def _occurs_on(item: dict, target: date) -> bool:
    try:
        base = date.fromisoformat((item.get("date") or "")[:10])
    except ValueError:
        return False
    return target == base or (item.get("recurring") == "weekly" and target >= base and target.weekday() == base.weekday())


def _upcoming_loads(tasks: List[dict], commitments: List[dict], adaptive_model: Optional[Dict[str, Any]] = None) -> list[dict]:
    """Create internal forecast inputs only from the user's actual calendar."""
    cairo_today = datetime.now(ZoneInfo("Africa/Cairo")).date()
    study_capacity_hours = max(0.5, float((adaptive_model or {}).get("max_daily_study_mins", 240) or 240) / 60)
    result = []
    for offset in range(7):
        day = cairo_today + timedelta(days=offset)
        day_items = [item for item in commitments if _occurs_on(item, day)]
        fixed_minutes = 0
        for item in day_items:
            if item.get("category", "class") not in {"class", "work"}:
                continue
            try:
                start = datetime.strptime(item["start"], "%H:%M")
                end = datetime.strptime(item["end"], "%H:%M")
                fixed_minutes += max(0, int((end - start).total_seconds() // 60))
            except (KeyError, ValueError, TypeError):
                continue
        exams = [item.get("title", "") for item in day_items if item.get("category") == "exam"]
        study_minutes = sum(
            max(0, int(task.get("adjusted_mins") or task.get("estimated_mins", 30)))
            for task in tasks
            if task.get("status") not in {"completed", "missed"}
            and (task.get("scheduled_date") or "")[:10] == day.isoformat()
        )
        exams.extend(
            f"موعد تسليم: {task.get('title', '')}"
            for task in tasks
            if task.get("status") not in {"completed", "missed"}
            and (task.get("deadline") or "")[:10] == day.isoformat()
        )
        if not fixed_minutes and not exams and not study_minutes:
            continue
        result.append({
            "type": "upcoming_load", "timestamp": f"{day.isoformat()}T12:00:00+02:00",
            "lecture_hours": fixed_minutes / 60, "study_hours": study_minutes / 60,
            "study_capacity_hours": study_capacity_hours,
            "exam": ", ".join(exams) if exams else None,
        })
    return result


class AdaptiveManagerAdapter:
    """Map the unified app's saved behavior into the source agent's API."""

    @property
    def mastery_threshold(self) -> float:
        """Expose the source agent's passing/mastery threshold to orchestration."""
        return float(native_analysis.settings.mastery_threshold)

    def run_adaptive_cycle(
        self,
        current_model: Dict[str, Any],
        events: List[Dict[str, Any]],
        upcoming_tasks: Optional[List[Dict[str, Any]]] = None,
        university_schedule: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        normalized = [_flatten_event(item) for item in events]
        normalized.sort(key=lambda item: item.get("timestamp", ""))
        all_metrics = native_analysis.observe(normalized)
        now = datetime.now(ZoneInfo("Africa/Cairo"))
        recent_cutoff = now - timedelta(days=30)
        recent_events = []
        for item in normalized:
            try:
                observed_at = datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00"))
                if observed_at.tzinfo is None:
                    observed_at = observed_at.replace(tzinfo=ZoneInfo("Africa/Cairo"))
                if observed_at >= recent_cutoff:
                    recent_events.append(item)
            except (KeyError, TypeError, ValueError):
                continue
        recent_metrics = native_analysis.observe(recent_events)
        patterns = native_analysis.analyze_patterns(recent_events)
        forecasts = _upcoming_loads(upcoming_tasks or [], university_schedule or [], current_model)
        risk = native_analysis.predict(normalized + forecasts)

        # The source manager persists a last processed event id. The unified
        # JSON store uses an append-only event list, so its index is the cursor.
        cursor = max(0, min(int(current_model.get("adaptive_event_cursor", 0)), len(normalized)))
        fresh_events = normalized[cursor:]
        fresh_metrics = native_analysis.observe(fresh_events)
        updated, recommended_block = native_analysis.adapt(
            current_model, fresh_metrics, risk,
            profile_metrics=all_metrics,
            recent_metrics=recent_metrics,
        )
        updated["adaptive_event_cursor"] = len(normalized)
        updated["procrastination_patterns"] = patterns
        updated["focus_sessions_count"] = int(current_model.get("focus_sessions_count", 0)) + len(fresh_metrics.get("focus_samples", []))

        mastery = dict(current_model.get("topic_mastery", {}))
        topic_context = dict(current_model.get("topic_context", {}))
        for topic, scores in all_metrics.get("scores", {}).items():
            if scores:
                mastery[topic] = round(float(scores[-1]), 1)
        for event in normalized:
            context = {key: event.get(key) for key in ("course_id", "lecture_id") if event.get(key)}
            if not context:
                continue
            for topic in (event.get("concept_scores") or {}):
                topic_context[topic] = {**topic_context.get(topic, {}), **context,
                                        "course_name": event.get("topic")}
        updated["topic_mastery"] = mastery
        updated["topic_context"] = topic_context
        weak_topics = []
        for topic, scores in all_metrics.get("scores", {}).items():
            if scores and scores[-1] < native_analysis.settings.mastery_threshold:
                weak_topics.append(topic)
            weak_topics.extend(all_metrics.get("failed_concepts", {}).get(topic, []))
        updated["weak_topics"] = list(dict.fromkeys(weak_topics))

        predictions = [{
            "id": f"pred-{uuid.uuid4().hex[:8]}", "target": "upcoming_load_risk",
            "target_id": prediction["date"], "risk_score": prediction["expected_risk"],
            "risk_level": "High", "rationale": prediction["reason"],
            "preemptive_action": "راجع توزيع مهامك حول المواعيد المسجلة قبل اعتماد الخطة.",
            "status": "active", "created_at": datetime.now(ZoneInfo("Africa/Cairo")).isoformat(),
        } for prediction in risk.get("predictions", [])]
        commands = native_analysis.create_commands(fresh_metrics, risk)
        for command in commands:
            command["command_id"] = f"cmd-{uuid.uuid4().hex[:8]}"
            command["created_at"] = datetime.now(ZoneInfo("Africa/Cairo")).isoformat()

        return {
            "metrics": all_metrics,
            "patterns": patterns,
            "risk_summary": risk,
            "adaptive_model": updated,
            "recommended_block_mins": recommended_block,
            "predictions": predictions,
            "dispatch_commands": commands,
        }


adaptive_manager = AdaptiveManagerAdapter()
