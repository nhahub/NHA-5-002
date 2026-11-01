"""End-to-end integration tests for GoalCoachAgent workflows using FakeCoachLLM.

Runs completely offline with ZERO network calls, zero API quota dependencies,
and ensures deterministic grading and CI reproducibility.
"""

from __future__ import annotations

from coach.agent import GoalCoachAgent
from coach.llm import FakeCoachLLM
from coach.schemas import (
    AgentEvent,
    CoachReply,
    GoalDecomposition,
    MicroTask,
    MilestoneDraft,
    ReflectionAnalysis,
)
from coach.storage import MemoryStore


def test_goal_creation_flow() -> None:
    """Tests the GOAL_CREATED flow: decomposition, persistence, and outbound routing."""
    store = MemoryStore()
    llm = FakeCoachLLM()
    agent = GoalCoachAgent(store=store, llm=llm)

    event = AgentEvent(
        event_type="GOAL_CREATED",
        user_id="user_depi_01",
        payload={
            "title": "Master LangGraph and Multi-Agent Orchestration",
            "category": "study",
            "target_date": "2026-12-31",
            "why": "To graduate with honors and secure a Senior AI Engineer position",
        },
    )

    result = agent.run(event)

    assert result.error is None
    assert "successfully structured" in result.user_message.lower()
    assert len(result.outbound_events) >= 2

    # Check PLANNING target outbound event
    planning_event = next(
        (e for e in result.outbound_events if e.target == "PLANNING"), None
    )
    assert planning_event is not None
    assert planning_event.command == "RECEIVE_GOAL_MILESTONES"
    assert len(planning_event.payload["milestones"]) >= 2

    # Check TUTOR target outbound event
    tutor_event = next(
        (e for e in result.outbound_events if e.target == "TUTOR"), None
    )
    assert tutor_event is not None
    assert tutor_event.command == "LINK_GOAL_TOPICS"
    assert len(tutor_event.payload["relevance_tags"]) >= 3

    # Check state persistence
    loaded = store.load("user_depi_01")
    assert len(loaded.goals_data.study_goals) == 1
    assert loaded.goals_data.study_goals[0].title == "Master LangGraph and Multi-Agent Orchestration"
    assert len(loaded.goals_data.milestones) >= 2


def test_daily_checkin_flow() -> None:
    """Tests DAILY_CHECKIN: qualitative signal extraction, metric calculation, and coaching reply."""
    store = MemoryStore()
    llm = FakeCoachLLM()
    agent = GoalCoachAgent(store=store, llm=llm)

    # First setup a habit
    init_state = store.load("user_depi_02")
    from coach.tools import update_habit
    update_habit(init_state, "Daily Problem Solving", "2026-10-20")
    store.save(init_state)

    event = AgentEvent(
        event_type="DAILY_CHECKIN",
        user_id="user_depi_02",
        payload={
            "date": "2026-10-21",
            "text": "Completed my study block on graph state machines. Felt very focused and energized!",
            "mood_hint": 0.8,
            "energy_hint": 0.9,
            "completed_habits": ["Daily Problem Solving"],
        },
    )

    result = agent.run(event)

    assert result.error is None
    assert len(result.user_message) > 0
    assert len(result.outbound_events) >= 1

    # Check MANAGER target outbound event
    manager_event = next(
        (e for e in result.outbound_events if e.target == "MANAGER"), None
    )
    assert manager_event is not None
    assert manager_event.command == "ADAPTIVE_METRICS_UPDATED"
    assert "motivation_index" in manager_event.payload
    assert "burnout_risk_score" in manager_event.payload

    # Check updated state
    loaded = store.load("user_depi_02")
    assert len(loaded.reflections) == 1
    assert loaded.reflections[0].mood > 0.0
    # Habit streak should increment from 1 to 2
    habit = next(h for h in loaded.goals_data.habits if h.name == "Daily Problem Solving")
    assert habit.streak == 2


def test_manager_signal_accountability_intervention() -> None:
    """Tests MANAGER_SIGNAL: receiving procrastination trigger and returning supportive coaching."""
    store = MemoryStore()
    llm = FakeCoachLLM()
    agent = GoalCoachAgent(store=store, llm=llm)

    # Pre-populate an active goal
    init_state = store.load("user_depi_03")
    from coach.tools import create_goal
    init_state, goal = create_goal(
        init_state, "Finish Graduation Project Module", category="study"
    )
    store.save(init_state)

    event = AgentEvent(
        event_type="MANAGER_SIGNAL",
        user_id="user_depi_03",
        payload={
            "command": "TRIGGER_ACCOUNTABILITY_CHECKIN",
            "reason": "2 consecutive days of procrastination detected on project tasks",
            "goal_id": goal.goal_id,
            "details": "User opened IDE but completed zero micro-tasks for 48 hours.",
        },
    )

    result = agent.run(event)

    assert result.error is None
    assert len(result.user_message) > 0

    # Check MANAGER outbound event confirming checkin completed
    confirm_event = next(
        (e for e in result.outbound_events if e.target == "MANAGER"), None
    )
    assert confirm_event is not None
    assert confirm_event.command == "ACCOUNTABILITY_CHECKIN_COMPLETED"
    assert "strategy_note" in confirm_event.payload

    # Verify goal strategy note updated
    loaded = store.load("user_depi_03")
    updated_goal = next(g for g in loaded.all_goals() if g.goal_id == goal.goal_id)
    assert len(updated_goal.strategy_note) > 0


def test_error_handling_malformed_event() -> None:
    """Agent gracefully catches malformed input without raising exceptions."""
    store = MemoryStore()
    agent = GoalCoachAgent(store=store, llm=FakeCoachLLM())

    # Missing user_id
    bad_event = AgentEvent(event_type="DAILY_CHECKIN", user_id="")
    result = agent.run(bad_event)
    assert result.error is not None
    assert "Missing required field" in result.error

    # Unknown event_type
    unknown_event = AgentEvent(
        event_type="UNRECOGNIZED_ACTION",
        user_id="user_depi_04",
        payload={"foo": "bar"},
    )
    result2 = agent.run(unknown_event)
    assert result2.error is not None
    assert "Unsupported or unknown event_type" in result2.error

