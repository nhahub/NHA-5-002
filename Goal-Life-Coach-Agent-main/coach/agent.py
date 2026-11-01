"""Main public entrypoint and agent contract for GoalCoachAgent.

Conforms to the agreed multi-agent contract:
- Initialized with a Store protocol instance and optional LLM.
- Exposes run(event: AgentEvent) -> AgentResult.
- Dispatches predictably through LangGraph workflows.
- Handles all exceptions gracefully without raising, returning AgentResult(error=...).
"""

from __future__ import annotations

import logging
from typing import Any

from coach.graph import CoachGraphState, create_coach_graph
from coach.llm import FakeCoachLLM, get_llm
from coach.schemas import AgentEvent, AgentResult, CoachState
from coach.storage import Store

logger = logging.getLogger(__name__)


class GoalCoachAgent:
    """The Goal & Life Coach Agent (Member 4 of Personal Life & Study Coach).

    Answers 'why, and how to keep going':
    - Decomposes goals into SMART milestones and micro-tasks for the Planning Agent.
    - Conducts daily reflection check-ins and tracks motivation / burnout deterministically.
    - Manages streaks, streak freezes, and comeback gamification.
    - Responds to Adaptive Manager signals to deliver accountability interventions.
    """

    def __init__(self, store: Store, llm: Any = None) -> None:
        """Initialize the agent with a backing store and an LLM.

        Args:
            store: Implementation of Store protocol (JsonStore or MemoryStore).
            llm: Optional LangChain ChatModel or FakeCoachLLM. If None, initializes
                 from environment or falls back to FakeCoachLLM if no key is found.
        """
        self.store = store
        if llm is not None:
            self.llm = llm
        else:
            try:
                self.llm = get_llm(allow_fake_fallback=True)
            except Exception:
                self.llm = FakeCoachLLM()

        # Compile the LangGraph pipeline
        self.graph = create_coach_graph(self.llm)

    def run(self, event: AgentEvent) -> AgentResult:
        """Execute the agent loop for an incoming AgentEvent.

        Loads state, routes through LangGraph, persists mutated state,
        and returns an AgentResult. Never raises exceptions on malformed input.

        Args:
            event: The incoming AgentEvent.

        Returns:
            AgentResult containing state updates, outbound events, and user message.
        """
        try:
            # 1. Defensive input validation
            if not isinstance(event, AgentEvent):
                try:
                    event = AgentEvent.model_validate(event)
                except Exception as val_err:
                    return AgentResult(
                        error=f"Invalid event schema: {val_err}",
                        user_message="I received a malformed event request.",
                    )

            if not event.user_id:
                return AgentResult(
                    error="Missing required field 'user_id' in AgentEvent.",
                    user_message="User identification is missing.",
                )

            # 2. Load persistent state via Store interface
            current_state = self.store.load(event.user_id)

            # If an explicit state snapshot was supplied in event, merge if fresh
            if event.state_snapshot and not current_state.all_goals():
                try:
                    current_state = CoachState.model_validate(event.state_snapshot)
                except Exception as snap_err:
                    logger.warning("Could not load state_snapshot: %s", snap_err)

            # 3. Construct initial LangGraph state
            graph_input: CoachGraphState = {
                "event": event,
                "coach_state": current_state,
                "outbound_events": [],
                "user_message": "",
                "error": None,
                "decomposition": None,
                "reflection_analysis": None,
                "coach_reply": None,
                "today": "",
            }

            # 4. Execute LangGraph StateGraph pipeline
            final_graph_state = self.graph.invoke(graph_input)

            # 5. Check if node emitted an error
            err = final_graph_state.get("error")
            if err:
                return AgentResult(
                    error=err,
                    user_message=final_graph_state.get(
                        "user_message", "Failed to process coaching event."
                    ),
                    outbound_events=final_graph_state.get("outbound_events", []),
                )

            # 6. Persist mutated state via Store interface
            updated_state: CoachState = final_graph_state["coach_state"]
            self.store.save(updated_state)

            # 7. Return production AgentResult contract
            return AgentResult(
                state_updates=updated_state.model_dump(),
                outbound_events=final_graph_state.get("outbound_events", []),
                user_message=final_graph_state.get("user_message", ""),
                error=None,
            )

        except Exception as exc:
            logger.exception("Unexpected error inside GoalCoachAgent.run: %s", exc)
            return AgentResult(
                error=f"Internal agent execution error: {str(exc)}",
                user_message="An unexpected error occurred while processing your coaching request.",
            )

