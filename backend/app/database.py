"""Unified Persistent Store for the Adaptive AI Personal Life & Study Coach.
Provides atomic JSON and SQLite persistence with full multi-agent state management.
Supports dynamic Courses, Lectures, Materials, Quizzes, Tasks, and Adaptive Personal Model.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.config import settings

logger = logging.getLogger("adaptive_coach.database")

class Database:
    """Thread-safe persistent state manager."""

    def __init__(self, store_path: Optional[Path] = None):
        self.store_path = store_path or settings.store_file
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.uploads_dir = settings.data_dir / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._state: Dict[str, Any] = {}
        self.load()

    def _default_state(self) -> Dict[str, Any]:
        """Start with a blank workspace and neutral, non-personalized defaults."""
        now = datetime.now(timezone.utc).isoformat()
        
        courses = []

        return {
            "user": {
                "id": f"user-{uuid.uuid4().hex[:12]}",
                "name": "",
                "email": "",
                "university": "",
                "semester": "",
                "major": "",
                "created_at": now,
            },
            "adaptive_model": {
                "effective_focus_duration_mins": 35,
                "focus_sessions_count": 0,
                "latest_energy_level": None,
                "peak_productivity_window": "",
                "task_delay_factor": 1.0,
                "learning_rate_velocity": 0.0,
                "avg_task_completion_ratio": 0.0,
                "procrastination_patterns": [],
                "topic_mastery": {},
                "weak_topics": [],
                "strong_topics": [],
                "burnout_risk_score": "unknown",
                "burnout_risk_value": 0.2,
                "consecutive_low_days": 0,
                "motivation_index": 0.0,
                "current_system_mode": "NORMAL",
                "last_updated": now,
            },
            "courses": courses,
            "university_schedule": [],
            "calendar_events": [],
            "tasks": [],
            "scheduled_blocks": [],
            "goals": [],
            "habits": [],
            "achievements": [],
            "reflections": [],
            "predictions": [],
            "assessments": [],
            "events": []
        }

    def _remove_legacy_demo_entries(self) -> None:
        """Remove only the application's known, preloaded sample records."""
        changed = False
        seeded_goal_ids = {"G-101", "G-102", "G-103"}
        goals_at_load = self._state.get("goals", [])
        user_at_load = self._state.get("user", {})
        legacy_seeded_profile = (
            user_at_load.get("id") == "student-001"
            and (any(g.get("goal_id") in seeded_goal_ids for g in goals_at_load)
                 or user_at_load.get("email") == "ahmed.student@depi.edu.eg")
        )
        if legacy_seeded_profile:
            # This exact account/profile belonged to the bundled fictional
            # student. Ask the current user for their own profile at next load.
            user_at_load.update({
                "id": f"user-{uuid.uuid4().hex[:12]}", "name": "", "email": "",
                "university": "", "semester": "", "major": "", "bio": "", "avatar_url": "",
            })
            changed = True
        events = self._state.get("events", [])
        legacy_demo_bundle = any(event.get("event_id") in {"ev-01", "ev-02", "ev-03"} for event in events)
        if legacy_demo_bundle:
            # These three stable IDs identify the bundled demo history. It must
            # not be used to infer a real user's productivity or mastery.
            self._state["events"] = []
            self._state["predictions"] = []
            self._state["assessments"] = []
            self._state["reflections"] = []
            achievements = self._state.get("achievements", [])
            self._state["achievements"] = [a for a in achievements if a.get("key") not in {"streak_3", "first_quiz_master"}]
            changed = True
        model = self._state.get("adaptive_model", {})
        seeded_patterns = [
            "Heavy lecture days (> 4 hrs classes)",
            "Complex formal theory topics (Automata & Proofs)",
        ]
        if (model.get("procrastination_patterns") == seeded_patterns
                or {"Automata & Formal Languages", "Trend AI & LLM Systems"}.issubset(model.get("topic_mastery", {}))):
            self._state["adaptive_model"] = self._default_state()["adaptive_model"]
            changed = True
        demo_lecture_ids = {"lec-1", "lec-2", "lec-3", "lec-4"}
        demo_task_ids = {"T101", "T102", "T103"}
        demo_course_ids = {"course-automata", "course-trend-ai"}
        courses = self._state.get("courses", [])
        cleaned = [item for item in courses if item.get("id") not in demo_course_ids]
        if len(cleaned) != len(courses):
            self._state["courses"] = cleaned
            changed = True
        schedule = self._state.get("university_schedule", [])
        cleaned = [item for item in schedule if item.get("id") not in demo_lecture_ids]
        if len(cleaned) != len(schedule):
            self._state["university_schedule"] = cleaned
            changed = True
        tasks = self._state.get("tasks", [])
        cleaned = [item for item in tasks if item.get("task_id") not in demo_task_ids]
        if legacy_demo_bundle:
            cleaned = [item for item in cleaned if not (
                item.get("task_id") == "T104" and item.get("title") == "learn"
                and item.get("status") == "completed" and item.get("actual_mins") == 35
            )]
        if len(cleaned) != len(tasks):
            self._state["tasks"] = cleaned
            changed = True
        blocks = self._state.get("scheduled_blocks", [])
        completed_ids = {
            item.get("task_id") for item in self._state.get("tasks", [])
            if item.get("status") in {"completed", "missed"}
        }
        cleaned = [item for item in blocks if item.get("task_id") not in demo_task_ids | completed_ids]
        if len(cleaned) != len(blocks):
            self._state["scheduled_blocks"] = cleaned
            changed = True
        goals = self._state.get("goals", [])
        cleaned = [item for item in goals if item.get("goal_id") not in seeded_goal_ids]
        if len(cleaned) != len(goals):
            self._state["goals"] = cleaned
            changed = True
        habits = self._state.get("habits", [])
        demo_habits = {"Daily Deep Focus Session", "Daily Deep Work Block", "Evening Reflection & Energy Check-in"}
        cleaned = [item for item in habits if item.get("name") not in demo_habits]
        if len(cleaned) != len(habits):
            self._state["habits"] = cleaned
            changed = True
        achievements = self._state.get("achievements", [])
        cleaned = [item for item in achievements if item.get("key") != "streak_3"]
        if len(cleaned) != len(achievements):
            self._state["achievements"] = cleaned
            changed = True
        predictions = self._state.get("predictions", [])
        cleaned = [item for item in predictions if item.get("id") != "pred-001"]
        if len(cleaned) != len(predictions):
            self._state["predictions"] = cleaned
            changed = True
        if "calendar_events" not in self._state:
            self._state["calendar_events"] = []
            changed = True
        if changed:
            self.save()

    def load(self) -> None:
        """Load persisted state, or initialize a blank workspace on first run."""
        with self._lock:
            if not self.store_path.exists():
                self._state = self._default_state()
                self.save()
                return

            try:
                with self.store_path.open("r", encoding="utf-8") as store:
                    loaded = json.load(store)
            except (OSError, json.JSONDecodeError) as exc:
                # Keep the existing file untouched so a transient or malformed
                # store cannot silently erase a user's data.
                raise RuntimeError(
                    f"Could not load application data from {self.store_path}: {exc}"
                ) from exc

            if not isinstance(loaded, dict):
                raise RuntimeError(
                    f"Application data at {self.store_path} must contain a JSON object."
                )

            defaults = self._default_state()
            self._state = {**defaults, **loaded}
            for key, value in defaults.items():
                if key not in loaded:
                    continue
                if key == "adaptive_model" and isinstance(value, dict) and isinstance(loaded[key], dict):
                    self._state[key] = {**value, **loaded[key]}
                elif isinstance(value, list) and not isinstance(loaded[key], list):
                    self._state[key] = value
            self._remove_legacy_demo_entries()

    def save(self) -> None:
        """Persist current state to JSON storage atomically."""
        with self._lock:
            temp_path = self.store_path.with_suffix(".tmp")
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(self._state, f, indent=2, ensure_ascii=False)
                temp_path.replace(self.store_path)
            except Exception as e:
                logger.error("Failed to save state to %s: %s", self.store_path, e)
                if temp_path.exists():
                    temp_path.unlink()

    # --- Course & Material Management ---

    def get_courses(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._state.get("courses", [])

    def create_course(self, name: str, code: str = "", instructor: str = "", description: str = "") -> Dict[str, Any]:
        with self._lock:
            courses = self._state.setdefault("courses", [])
            cid = f"course-{uuid.uuid4().hex[:10]}"
            new_course = {
                "id": cid,
                "name": name,
                "code": code,
                "instructor": instructor,
                "description": description,
                "materials": []
            }
            courses.append(new_course)
            self.save()
            return new_course

    def delete_course(self, course_id: str) -> bool:
        with self._lock:
            courses = self._state.get("courses", [])
            initial_len = len(courses)
            self._state["courses"] = [c for c in courses if c.get("id") != course_id]
            if len(self._state["courses"]) < initial_len:
                self.save()
                return True
            return False

    def add_material_to_course(
        self,
        course_id: str,
        title: str,
        filename: str,
        file_path: str,
        pages: int = 1,
        size_kb: float = 0.0
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            for c in self._state.get("courses", []):
                if c.get("id") == course_id:
                    mat_item = {
                        "id": f"mat-{uuid.uuid4().hex[:10]}",
                        "title": title or filename,
                        "filename": filename,
                        "path": file_path,
                        "pages": pages,
                        "size_kb": size_kb,
                        "uploaded_at": datetime.now(timezone.utc).isoformat()
                    }
                    c.setdefault("materials", []).append(mat_item)
                    self.save()
                    return mat_item
            return None

    def get_material_by_id(self, material_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            for c in self._state.get("courses", []):
                for m in c.get("materials", []):
                    if m.get("id") == material_id:
                        return m
            return None

    # --- Other Entity Accessors ---

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return self._state

    def get_user(self) -> Dict[str, Any]:
        with self._lock:
            return self._state.get("user", {})

    def update_user(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            user = self._state.setdefault("user", {})
            user.update(updates)
            self.save()
            return user

    def get_adaptive_model(self) -> Dict[str, Any]:
        with self._lock:
            return self._state.get("adaptive_model", {})

    def update_adaptive_model(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            model = self._state.setdefault("adaptive_model", {})
            model.update(updates)
            model["last_updated"] = datetime.now(timezone.utc).isoformat()
            self.save()
            return model

    def get_tasks(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._state.get("tasks", [])

    def save_tasks(self, tasks: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._state["tasks"] = tasks
            self.save()

    def get_university_schedule(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._state.get("university_schedule", [])

    def save_university_schedule(self, schedule: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._state["university_schedule"] = schedule
            self.save()

    def get_goals(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._state.get("goals", [])

    def save_goals(self, goals: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._state["goals"] = goals
            self.save()

    def get_habits(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._state.get("habits", [])

    def save_habits(self, habits: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._state["habits"] = habits
            self.save()

    def get_achievements(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._state.get("achievements", [])

    def add_achievement(self, ach: Dict[str, Any]) -> None:
        with self._lock:
            existing = {a["key"] for a in self._state.get("achievements", [])}
            if ach["key"] not in existing:
                self._state.setdefault("achievements", []).append(ach)
                self.save()

    def get_reflections(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._state.get("reflections", [])

    def add_reflection(self, ref: Dict[str, Any]) -> None:
        with self._lock:
            self._state.setdefault("reflections", []).append(ref)
            self.save()

    def get_predictions(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._state.get("predictions", [])

    def add_prediction(self, pred: Dict[str, Any]) -> None:
        with self._lock:
            predictions = self._state.setdefault("predictions", [])
            key = (pred.get("target"), pred.get("target_id"))
            for old in predictions:
                if (old.get("target"), old.get("target_id")) == key and old.get("status") == "active":
                    old["status"] = "updated"
            predictions.insert(0, pred)
            self.save()

    def get_assessments(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._state.get("assessments", [])

    def add_assessment(self, quiz: Dict[str, Any]) -> None:
        with self._lock:
            self._state.setdefault("assessments", []).insert(0, quiz)
            self.save()

    def get_events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._state.get("events", [])

    def add_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            events = self._state.setdefault("events", [])
            new_id = len(events) + 1
            record = {
                "id": new_id,
                "event_id": event.get("event_id") or f"ev-{new_id:04d}",
                "timestamp": event.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                "source": event.get("source", "System"),
                "type": event.get("type", "generic"),
                "payload": event.get("payload", {})
            }
            events.append(record)
            self.save()
            return record

    def clear_all_tasks(self) -> None:
        """Clear all tasks and schedule blocks for a clean slate."""
        with self._lock:
            self._state["tasks"] = []
            self._state["scheduled_blocks"] = []
            self.save()

    def reset_to_clean(self) -> None:
        """Reset state with clean courses and empty schedule/tasks."""
        with self._lock:
            self._state = self._default_state()
            self._state["tasks"] = []
            self._state["scheduled_blocks"] = []
            self._state["university_schedule"] = []
            self._state["assessments"] = []
            self._state["reflections"] = []
            self._state["predictions"] = []
            self.save()

db = Database()
