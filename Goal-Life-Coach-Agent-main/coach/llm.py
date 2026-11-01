"""LLM initialization and structured output helpers.

Uses langchain.chat_models.init_chat_model and structured outputs bound to Pydantic models.
Supports Google Gemini (gemini-2.5-flash) free tier via langchain-google-genai,
with configurable models via COACH_MODEL and fallback FakeCoachLLM for offline tests and demos.
"""

from __future__ import annotations

import os
from typing import Any, TypeVar

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from coach.prompts import (
    ACCOUNTABILITY_SYSTEM_PROMPT,
    ACCOUNTABILITY_USER_PROMPT,
    COACH_REPLY_SYSTEM_PROMPT,
    COACH_REPLY_USER_PROMPT,
    GOAL_DECOMPOSITION_SYSTEM_PROMPT,
    GOAL_DECOMPOSITION_USER_PROMPT,
    REFLECTION_ANALYSIS_SYSTEM_PROMPT,
    REFLECTION_ANALYSIS_USER_PROMPT,
)
from coach.schemas import (
    CoachReply,
    CoachState,
    GoalDecomposition,
    MicroTask,
    MilestoneDraft,
    ReflectionAnalysis,
)

load_dotenv()

T = TypeVar("T", bound=BaseModel)


class FakeStructuredRunnable:
    """Runnable returning predefined or dynamically constructed structured outputs."""

    def __init__(self, schema: type[T], canned_data: dict[type[BaseModel], Any] | None = None) -> None:
        self.schema = schema
        self.canned_data = canned_data or {}

    def invoke(self, input_data: Any, **kwargs: Any) -> T:
        """Return a canned or realistic simulated structured output matching the schema."""
        if self.schema in self.canned_data:
            canned = self.canned_data[self.schema]
            if callable(canned):
                return canned(input_data)
            return canned

        # Provide sensible, deterministic defaults based on schema
        if self.schema == GoalDecomposition:
            return GoalDecomposition(
                category="study",
                why="To build foundational mastery in the domain and achieve academic/career success.",
                relevance_tags=["foundations", "theory", "applied-practice", "review"],
                milestones=[
                    MilestoneDraft(
                        title="Phase 1: Foundational Core Concepts",
                        criteria="Complete all core readings and initial concept exercises.",
                        due_date=None,
                        micro_tasks=[
                            MicroTask(title="Review chapter syllabus and core notes", estimated_mins=30),
                            MicroTask(title="Complete foundational concept drills", estimated_mins=45),
                            MicroTask(title="Build initial flashcard deck", estimated_mins=30),
                        ],
                    ),
                    MilestoneDraft(
                        title="Phase 2: Applied Implementation & Practice",
                        criteria="Implement working prototype and pass self-assessment test.",
                        due_date=None,
                        micro_tasks=[
                            MicroTask(title="Build guided practical exercise", estimated_mins=60),
                            MicroTask(title="Refactor and document code/notes", estimated_mins=45),
                        ],
                    ),
                    MilestoneDraft(
                        title="Phase 3: Final Synthesis & Review",
                        criteria="Conduct complete end-to-end review and achieve >85% on practice questions.",
                        due_date=None,
                        micro_tasks=[
                            MicroTask(title="Complete mock assessment", estimated_mins=60),
                            MicroTask(title="Final retrospective and gap review", estimated_mins=30),
                        ],
                    ),
                ],
            )  # type: ignore[return-value]

        if self.schema == ReflectionAnalysis:
            # Inspect user message content (not the system prompt instructions)
            user_text = ""
            if isinstance(input_data, list):
                for msg in input_data:
                    if hasattr(msg, "content") and not isinstance(msg, SystemMessage):
                        user_text += " " + str(msg.content).lower()
            else:
                user_text = str(input_data).lower()

            if any(w in user_text for w in ("exhausted", "burnout", "burned out", "depleted", "struggling", "dread")):
                return ReflectionAnalysis(
                    mood=-0.4,
                    energy=0.2,
                    friction=["High cognitive overload", "Physical exhaustion"],
                    avoidance_signals=["Procrastinated starting main block"],
                    burnout_signals=["Felt depleted", "Struggled with focus"],
                    summary="User is experiencing significant mental fatigue and physical exhaustion.",
                )  # type: ignore[return-value]

            return ReflectionAnalysis(
                mood=0.6,
                energy=0.7,
                friction=["Minor distraction from phone notifications"],
                avoidance_signals=[],
                burnout_signals=[],
                summary="Solid, focused session with steady momentum and clear progress.",
            )  # type: ignore[return-value]

        if self.schema == CoachReply:
            prompt_str = str(input_data).lower()
            if "procrastination" in prompt_str or "accountability" in prompt_str or "stagnation" in prompt_str:
                return CoachReply(
                    message="I noticed things have felt heavy the last couple of days. Let's take a breath - hitting a wall is normal, not a failure. What is the single smallest 5-minute step we can take right now to break inertia?",
                    suggest_strategy_change=True,
                    strategy_note="Break the next milestone into smaller 15-minute micro-steps and reduce daily commitment until momentum returns.",
                )  # type: ignore[return-value]

            if any(w in prompt_str for w in ("exhaustion", "depleted", "energy: 0.2", "mood: -0.", "numb")):
                return CoachReply(
                    message="I hear how heavy today felt. When energy is running on empty, taking a deliberate pause is a strategic recovery, not lost time. What is one kind thing you can do for yourself tonight?",
                    suggest_strategy_change=True,
                    strategy_note="Scale back micro-task block duration by 50% for 48 hours to prioritize nervous system recovery.",
                )  # type: ignore[return-value]

            return CoachReply(
                message="Great effort staying consistent today. Protecting your daily study rhythm is paying off. What is one takeaway you feel proud of today?",
                suggest_strategy_change=False,
                strategy_note="",
            )  # type: ignore[return-value]

        # Generic fallback
        return self.schema()  # type: ignore[return-value]


class FakeCoachLLM:
    """Mock LLM implementation for tests and offline demonstrations.

    Emulates LangChain ChatModel's with_structured_output without requiring
    network requests or an API key.
    """

    def __init__(self, canned_data: dict[type[BaseModel], Any] | None = None) -> None:
        self.canned_data = canned_data or {}

    def with_structured_output(self, schema: type[T], **kwargs: Any) -> FakeStructuredRunnable:
        """Return a runnable bound to the target schema."""
        return FakeStructuredRunnable(schema, self.canned_data)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return "Fake LLM Response"


def get_llm(
    model: str | None = None,
    api_key: str | None = None,
    allow_fake_fallback: bool = False,
) -> BaseChatModel | FakeCoachLLM:
    """Initialize the configured chat model using init_chat_model.

    Defaults to Google Gemini free tier (gemini-2.5-flash) reading from GOOGLE_API_KEY.
    Model name can be overridden via COACH_MODEL env var.
    If no API key is found and allow_fake_fallback is True, returns FakeCoachLLM.
    """
    from langchain.chat_models import init_chat_model

    model_name = model or os.getenv("COACH_MODEL", "gemini-2.5-flash")
    key = api_key or os.getenv("GOOGLE_API_KEY")

    if not key:
        if allow_fake_fallback:
            return FakeCoachLLM()
        raise ValueError(
            "GOOGLE_API_KEY is not set. Please set GOOGLE_API_KEY in your environment "
            "or .env file, or pass allow_fake_fallback=True for offline/mock mode."
        )

    # Initialize model using langchain's init_chat_model with google_genai provider
    return init_chat_model(
        model=model_name,
        model_provider="google_genai",
        api_key=key,
        temperature=0.3,
    )


def decompose_goal_with_llm(
    llm: Any,
    goal_title: str,
    target_date: str | None = None,
    category_hint: str | None = None,
    context: str = "",
) -> GoalDecomposition:
    """Decompose a user goal into structured SMART milestones using the LLM."""
    structured_llm = llm.with_structured_output(GoalDecomposition)
    system_msg = SystemMessage(content=GOAL_DECOMPOSITION_SYSTEM_PROMPT)
    user_msg = HumanMessage(
        content=GOAL_DECOMPOSITION_USER_PROMPT.format(
            goal_title=goal_title,
            target_date=target_date or "Flexible / Not specified",
            category_hint=category_hint or "Infer from goal title",
            context=context or "None provided",
        )
    )
    result = structured_llm.invoke([system_msg, user_msg])
    return result


def analyze_reflection_with_llm(
    llm: Any,
    text: str,
    mood_hint: float | None = None,
    energy_hint: float | None = None,
    completed_habits: list[str] | None = None,
    date_str: str = "",
) -> ReflectionAnalysis:
    """Extract qualitative signals from user reflection text into structured schema."""
    structured_llm = llm.with_structured_output(ReflectionAnalysis)
    system_msg = SystemMessage(content=REFLECTION_ANALYSIS_SYSTEM_PROMPT)
    user_msg = HumanMessage(
        content=REFLECTION_ANALYSIS_USER_PROMPT.format(
            text=text,
            mood_hint=f"{mood_hint:.1f}" if mood_hint is not None else "Not provided",
            energy_hint=f"{energy_hint:.1f}" if energy_hint is not None else "Not provided",
            completed_habits=", ".join(completed_habits) if completed_habits else "None reported",
            date=date_str or "Today",
        )
    )
    result = structured_llm.invoke([system_msg, user_msg])
    return result


def generate_coaching_reply_with_llm(
    llm: Any,
    analysis: ReflectionAnalysis,
    state: CoachState,
    habits_status: str = "",
) -> CoachReply:
    """Generate empathetic daily coaching reply and evaluate strategy change."""
    structured_llm = llm.with_structured_output(CoachReply)
    system_msg = SystemMessage(content=COACH_REPLY_SYSTEM_PROMPT)

    active_goals_summary = ", ".join(
        f"{g.title} ({g.category})" for g in state.all_goals() if g.status == "active"
    ) or "None active"

    user_msg = HumanMessage(
        content=COACH_REPLY_USER_PROMPT.format(
            summary=analysis.summary,
            mood=f"{analysis.mood:.2f}",
            energy=f"{analysis.energy:.2f}",
            friction=", ".join(analysis.friction) if analysis.friction else "None",
            avoidance=", ".join(analysis.avoidance_signals) if analysis.avoidance_signals else "None",
            burnout_signals=", ".join(analysis.burnout_signals) if analysis.burnout_signals else "None",
            motivation_index=f"{state.adaptive_model.motivation_index:.2f}",
            burnout_risk_score=f"{state.adaptive_model.burnout_risk.score:.2f}",
            burnout_risk_level=state.adaptive_model.burnout_risk.level,
            consecutive_low_days=state.consecutive_low_days,
            active_goals=active_goals_summary,
            habits_status=habits_status or "Normal",
        )
    )
    result = structured_llm.invoke([system_msg, user_msg])
    return result


def generate_accountability_reply_with_llm(
    llm: Any,
    signal_type: str,
    reason: str,
    details: str,
    state: CoachState,
) -> CoachReply:
    """Generate accountability check-in reply in response to Manager signals."""
    structured_llm = llm.with_structured_output(CoachReply)
    system_msg = SystemMessage(content=ACCOUNTABILITY_SYSTEM_PROMPT)

    active_goals_summary = ", ".join(
        f"{g.title} ({g.category})" for g in state.all_goals() if g.status == "active"
    ) or "None active"

    user_msg = HumanMessage(
        content=ACCOUNTABILITY_USER_PROMPT.format(
            signal_type=signal_type,
            reason=reason,
            details=details,
            active_goals=active_goals_summary,
            motivation_index=f"{state.adaptive_model.motivation_index:.2f}",
            burnout_risk_level=state.adaptive_model.burnout_risk.level,
            burnout_risk_score=f"{state.adaptive_model.burnout_risk.score:.2f}",
            consecutive_low_days=state.consecutive_low_days,
        )
    )
    result = structured_llm.invoke([system_msg, user_msg])
    return result
