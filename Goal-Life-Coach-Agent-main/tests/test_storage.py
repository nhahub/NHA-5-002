"""Unit tests for storage implementations (MemoryStore and JsonStore)."""

from __future__ import annotations

from pathlib import Path

from coach.schemas import CoachState, Goal
from coach.storage import JsonStore, MemoryStore


def test_memory_store_isolation() -> None:
    """MemoryStore deepcopies states so internal mutations don't leak without saving."""
    store = MemoryStore()
    state = store.load("user_1")
    state.goals_data.study_goals.append(Goal(goal_id="g1", title="Study Math"))

    # State not yet saved -> reload should still return fresh default
    reloaded = store.load("user_1")
    assert len(reloaded.goals_data.study_goals) == 0

    # Save and reload -> state persists
    store.save(state)
    persisted = store.load("user_1")
    assert len(persisted.goals_data.study_goals) == 1
    assert persisted.goals_data.study_goals[0].title == "Study Math"


def test_json_store_file_persistence(tmp_path: Path) -> None:
    """JsonStore saves formatted JSON and reloads across fresh instances."""
    store = JsonStore(base_dir=tmp_path)
    state = CoachState(user_id="user_json_1")
    state.consecutive_low_days = 3
    state.goals_data.career_goals.append(
        Goal(goal_id="c1", title="Become Staff Engineer", category="career")
    )
    store.save(state)

    # Check file exists on disk
    expected_file = tmp_path / "user_json_1.json"
    assert expected_file.exists()

    # Load from a new JsonStore instance pointing to the same folder
    new_store = JsonStore(base_dir=tmp_path)
    loaded = new_store.load("user_json_1")

    assert loaded.user_id == "user_json_1"
    assert loaded.consecutive_low_days == 3
    assert len(loaded.goals_data.career_goals) == 1
    assert loaded.goals_data.career_goals[0].title == "Become Staff Engineer"


def test_json_store_fallback_on_corrupted_file(tmp_path: Path) -> None:
    """JsonStore falls back gracefully to a fresh CoachState if JSON file is corrupted."""
    corrupted_file = tmp_path / "corrupted_user.json"
    corrupted_file.write_text("NOT_VALID_JSON{:::}}", encoding="utf-8")

    store = JsonStore(base_dir=tmp_path)
    state = store.load("corrupted_user")
    assert state.user_id == "corrupted_user"
    assert len(state.all_goals()) == 0

