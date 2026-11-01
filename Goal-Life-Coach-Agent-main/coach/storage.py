"""Storage abstraction and implementations for CoachState persistence.

Provides:
- Store (Protocol): Interface decoupling storage from coach business logic.
- MemoryStore: In-memory store useful for unit tests and fast ephemeral runs.
- JsonStore: Free-tier friendly JSON filesystem store for local persistence.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from coach.schemas import CoachState


@runtime_checkable
class Store(Protocol):
    """Abstract storage protocol for coach state."""

    def load(self, user_id: str) -> CoachState:
        """Load coach state for a given user_id.

        If the user does not exist yet, returns a fresh default CoachState.
        """
        ...

    def save(self, state: CoachState) -> None:
        """Persist the given CoachState."""
        ...


class MemoryStore:
    """In-memory implementation of Store protocol.

    Uses deepcopy to prevent unintended side effects on stored states.
    """

    def __init__(self, initial_data: dict[str, CoachState] | None = None) -> None:
        self._data: dict[str, CoachState] = {}
        if initial_data:
            for uid, state in initial_data.items():
                self._data[uid] = state.model_copy(deep=True)

    def load(self, user_id: str) -> CoachState:
        """Retrieve state for user_id or return default CoachState."""
        if user_id in self._data:
            return self._data[user_id].model_copy(deep=True)
        return CoachState(user_id=user_id)

    def save(self, state: CoachState) -> None:
        """Persist state in memory."""
        self._data[state.user_id] = state.model_copy(deep=True)

    def exists(self, user_id: str) -> bool:
        """Check if user state exists in memory."""
        return user_id in self._data


class JsonStore:
    """JSON file-based implementation of Store protocol for local persistence."""

    def __init__(self, base_dir: str | Path = "./data") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, user_id: str) -> Path:
        # Sanitize user_id for filesystem safety
        safe_id = "".join(c for c in user_id if c.isalnum() or c in ("-", "_"))
        if not safe_id:
            safe_id = "default_user"
        return self.base_dir / f"{safe_id}.json"

    def load(self, user_id: str) -> CoachState:
        """Load state from JSON file or return default CoachState if absent."""
        path = self._file_path(user_id)
        if not path.exists():
            return CoachState(user_id=user_id)

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_json = f.read()
            return CoachState.model_validate_json(raw_json)
        except Exception:
            # Fallback to fresh state if corrupted or invalid
            return CoachState(user_id=user_id)

    def save(self, state: CoachState) -> None:
        """Persist state as formatted JSON file atomically."""
        path = self._file_path(state.user_id)
        temp_path = path.with_suffix(".tmp")
        payload = state.model_dump_json(indent=2)
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(payload)
        temp_path.replace(path)

    def exists(self, user_id: str) -> bool:
        """Check if user state file exists."""
        return self._file_path(user_id).exists()

