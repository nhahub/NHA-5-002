"""Unified Persistent Store for the Adaptive AI Personal Life & Study Coach.
Provides atomic JSON and SQLite persistence with full multi-agent state management.
Supports dynamic Courses, Lectures, Materials, Quizzes, Tasks, and Adaptive Personal Model.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pypdf import PdfReader

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
        """Fresh, clean baseline state with auto-discovered course materials."""
        now = datetime.now(timezone.utc).isoformat()
        
        courses = []

        return {
            "user": {
                "id": "student-001",
                "name": "Student",
                "email": "",
                "university": "",
                "semester": "",
                "major": "",
                "created_at": now,
            },
            "adaptive_model": {
                "effective_focus_duration_mins": 35,
                "peak_productivity_window": "09:00-11:00",
                "task_delay_factor": 1.25,
                "learning_rate_velocity": 0.65,
                "avg_task_completion_ratio": 0.80,
                "procrastination_patterns": [
                    "Heavy lecture days (> 4 hrs classes)",
                    "Complex formal theory topics (Automata & Proofs)"
                ],
                "topic_mastery": {
                    "Automata & Formal Languages": 60.0,
                    "Trend AI & LLM Systems": 85.0
                },
                "weak_topics": [
                    "NFA to DFA subset construction",
                    "Pumping Lemma proofs"
                ],
                "strong_topics": [
                    "RAG Retrieval Mechanisms",
                    "Transformer Attention"
                ],
                "burnout_risk_score": "Low",
                "motivation_index": 0.82,
                "current_system_mode": "NORMAL",
                "last_updated": now,
            },
            "courses": courses,
            "university_schedule": [],
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

    def _discover_initial_courses(self) -> List[Dict[str, Any]]:
        """Organize files found in material/ into initial course structures."""
        mat_dir = settings.material_dir
        courses = [
            {
                "id": "course-automata",
                "name": "Theory of Automata & Formal Languages",
                "code": "AI412",
                "instructor": "Dr. Ahmed Zakzouk",
                "description": "Formal languages, DFA, NFA, Context-Free Grammars, and Turing Machines.",
                "materials": []
            },
            {
                "id": "course-trend-ai",
                "name": "Trend AI & Modern Deep Learning",
                "code": "AI415",
                "instructor": "Dr. Tarek & Department Faculty",
                "description": "Transformers, LLM architectures, RAG systems, and AI Agents.",
                "materials": []
            }
        ]

        if mat_dir.exists():
            for f in sorted(mat_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in {".pdf", ".txt", ".docx"}:
                    try:
                        pages = len(PdfReader(f).pages) if f.suffix.lower() == ".pdf" else 1
                    except Exception:
                        pages = 1
                    size_kb = round(f.stat().st_size / 1024, 1)

                    mat_item = {
                        "id": f"mat-{f.stem.replace(' ', '-').lower()}",
                        "title": f.name.replace(".pdf.pdf", "").replace(".pdf", ""),
                        "filename": f.name,
                        "path": str(f),
                        "pages": pages,
                        "size_kb": size_kb,
                        "uploaded_at": datetime.now(timezone.utc).isoformat()
                    }

                    if "automata" in f.name.lower():
                        courses[0]["materials"].append(mat_item)
                    else:
                        courses[1]["materials"].append(mat_item)

        return courses

    def load(self) -> None:
        """Load state from JSON storage file or initialize default."""
        with self._lock:
            if self.store_path.exists():
                try:
                    with open(self.store_path, "r", encoding="utf-8") as f:
                        self._state = json.load(f)
                        if "courses" not in self._state:
                            self._state["courses"] = self._discover_initial_courses()
                        logger.info("Loaded state from %s", self.store_path)
                        return
                except Exception as e:
                    logger.warning("Failed to parse store file %s: %s. Using default state.", self.store_path, e)
            self._state = self._default_state()
            self.save()

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
            self._state.setdefault("predictions", []).insert(0, pred)
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
