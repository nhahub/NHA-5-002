"""LangGraph StateGraph implementation for the 3 core coach workflows:

1. GOAL_CREATED / GOAL_UPDATED:
   - LLM decomposes goal into 2-8 SMART milestones + micro-tasks + relevance_tags.
   - Saves goal & milestones to CoachState.
   - Emits outbound events to PLANNING (milestones) and TUTOR (study tags).

2. DAILY_CHECKIN:
   - LLM extracts qualitative signals (mood, energy, friction, avoidance, burnout).
   - Deterministic scoring updates motivation_index and burnout_risk.
   - Gamification updates streaks, freezes, and badges.
   - LLM writes coaching message & evaluates strategy changes.
   - Emits outbound update to MANAGER.

3. MANAGER_SIGNAL:
   - Handles triggers (e.g., TRIGGER_ACCOUNTABILITY_CHECKIN).
   - Generates empathetic accountability intervention.
   - Updates goal strategy and reports back to MANAGER.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from coach.gamification import calculate_habit_completion_ratio
from coach.llm import (
    analyze_reflection_with_llm,
    decompose_goal_with_llm,
    generate_accountability_reply_with_llm,
    generate_coaching_reply_with_llm,
)
from coach.schemas import (
    AgentEvent,
    CoachReply,
    CoachState,
    GoalDecomposition,
    Milestone,
    OutboundEvent,
    ReflectionAnalysis,
)
from coach.scoring import update_adaptive_metrics
from coach.tools import (
    create_goal,
    log_reflection,
    send_milestones_to_planner,
    send_relevance_tags_to_tutor,
    update_goal_strategy,
    update_habit,
)


class CoachGraphState(TypedDict, total=False):
    """Internal LangGraph state passed through pipeline nodes."""

    event: AgentEvent
    coach_state: CoachState
    outbound_events: list[OutboundEvent]
    user_message: str
    error: str | None
    decomposition: GoalDecomposition | None
    reflection_analysis: ReflectionAnalysis | None
    coach_reply: CoachReply | None
    today: str


def create_coach_graph(llm: Any) -> Any:
    """Build and compile the LangGraph StateGraph with the provided LLM."""

    # -----------------------------------------------------------------------
    # Flow 1: Goal Creation & Decomposition Nodes
    # -----------------------------------------------------------------------

    def decompose_goal_node(state: CoachGraphState) -> dict[str, Any]:
        """Node 1A: Call LLM to decompose goal into SMART milestones & micro-tasks."""
        payload = state["event"].payload
        title = payload.get("title") or payload.get("goal_title", "Untitled Goal")
        target_date = payload.get("target_date")
        category_hint = payload.get("category")
        context = payload.get("context", "")

        decomposition = decompose_goal_with_llm(
            llm=llm,
            goal_title=title,
            target_date=target_date,
            category_hint=category_hint,
            context=context,
        )
        return {"decomposition": decomposition}

    def persist_goal_node(state: CoachGraphState) -> dict[str, Any]:
        """Node 1B: Save goal and milestones into CoachState and generate outbound events."""
        cstate = state["coach_state"]
        decomp = state["decomposition"]
        payload = state["event"].payload

        title = payload.get("title") or payload.get("goal_title", "Untitled Goal")
        category = decomp.category if decomp else payload.get("category", "personal")
        target_date = payload.get("target_date")
        why = decomp.why if decomp else payload.get("why", "")

        # 1. Create or retrieve goal
        cstate, goal = create_goal(
            state=cstate,
            title=title,
            category=category,
            target_date=target_date,
            why=why,
            goal_id=payload.get("goal_id"),
        )

        # 2. Convert MilestoneDrafts to Milestone models and persist
        created_milestones: list[Milestone] = []
        if decomp and decomp.milestones:
            for idx, draft in enumerate(decomp.milestones, start=1):
                ms_id = f"{goal.goal_id}_ms_{idx}_{uuid.uuid4().hex[:4]}"
                milestone = Milestone(
                    id=ms_id,
                    goal_id=goal.goal_id,
                    title=draft.title,
                    status="pending",
                    criteria=draft.criteria,
                    due_date=draft.due_date or target_date,
                    micro_tasks=draft.micro_tasks,
                )
                cstate.goals_data.milestones.append(milestone)
                created_milestones.append(milestone)

        # 3. Create outbound events
        outbounds: list[OutboundEvent] = list(state.get("outbound_events") or [])

        # -> PLANNING: Milestones for daily scheduling
        outbounds.append(send_milestones_to_planner(goal, created_milestones))

        # -> TUTOR: Relevance tags for study topics linking
        tags = decomp.relevance_tags if decomp else ["learning", "study", "skills"]
        outbounds.append(send_relevance_tags_to_tutor(goal, tags))

        # -> MANAGER: Goal activation notice
        outbounds.append(
            OutboundEvent(
                target="MANAGER",
                command="GOAL_ACTIVATED",
                payload={
                    "user_id": cstate.user_id,
                    "goal_id": goal.goal_id,
                    "title": goal.title,
                    "milestones_count": len(created_milestones),
                },
            )
        )

        # User-facing summary message
        ms_summary = "\n".join(
            f"  {i+1}. {m.title} ({len(m.micro_tasks)} micro-tasks)"
            for i, m in enumerate(created_milestones)
        )
        msg = (
            f"Goal '{goal.title}' successfully structured!\n"
            f"Why this matters: {why}\n"
            f"SMART Milestones created:\n{ms_summary}\n\n"
            f"Sent {len(created_milestones)} milestones to Personal Planner and "
            f"{len(tags)} study topics to Generative Tutor."
        )

        return {
            "coach_state": cstate,
            "outbound_events": outbounds,
            "user_message": msg,
        }

    # -----------------------------------------------------------------------
    # Flow 2: Daily Check-in Nodes
    # -----------------------------------------------------------------------

    def analyze_reflection_node(state: CoachGraphState) -> dict[str, Any]:
        """Node 2A: Use LLM to extract qualitative signals only."""
        payload = state["event"].payload
        text = payload.get("text", "")
        mood_hint = payload.get("mood_hint") or payload.get("mood")
        energy_hint = payload.get("energy_hint") or payload.get("energy")
        completed_habits = payload.get("completed_habits", [])
        today_date = payload.get("date") or datetime.date.today().isoformat()

        analysis = analyze_reflection_with_llm(
            llm=llm,
            text=text,
            mood_hint=float(mood_hint) if mood_hint is not None else None,
            energy_hint=float(energy_hint) if energy_hint is not None else None,
            completed_habits=completed_habits,
            date_str=today_date,
        )
        return {"reflection_analysis": analysis, "today": today_date}

    def update_metrics_node(state: CoachGraphState) -> dict[str, Any]:
        """Node 2B: Pure deterministic scoring and gamification."""
        cstate = state["coach_state"]
        analysis = state["reflection_analysis"]
        payload = state["event"].payload
        today = state.get("today") or datetime.date.today().isoformat()
        assert analysis is not None

        # 1. Update completed habits (gamification with streak, freeze, comeback)
        completed_habit_names = payload.get("completed_habits", [])
        new_badges_total = []
        for h_name in completed_habit_names:
            cstate, _, new_badges, _ = update_habit(
                state=cstate,
                habit_name=h_name,
                completion_date=today,
            )
            new_badges_total.extend(new_badges)

        # 2. Calculate completion ratio
        completion_ratio = calculate_habit_completion_ratio(
            cstate.goals_data.habits, today
        )
        if "milestones_completed_ratio" in payload:
            completion_ratio = (
                completion_ratio + float(payload["milestones_completed_ratio"])
            ) / 2.0

        # 3. Update adaptive model metrics deterministically
        new_adaptive, new_low_days = update_adaptive_metrics(
            current_adaptive=cstate.adaptive_model,
            analysis=analysis,
            completion_ratio=completion_ratio,
            previous_consecutive_low_days=cstate.consecutive_low_days,
        )
        cstate.adaptive_model = new_adaptive
        cstate.consecutive_low_days = new_low_days

        # 4. Log reflection record into CoachState
        cstate, _ = log_reflection(
            state=cstate,
            date=today,
            text=payload.get("text", ""),
            mood=analysis.mood,
            energy=analysis.energy,
            friction=analysis.friction,
            burnout_signals=analysis.burnout_signals,
        )

        return {"coach_state": cstate}

    def generate_coaching_node(state: CoachGraphState) -> dict[str, Any]:
        """Node 2C: Generate empathetic coaching message & check strategy pivot."""
        cstate = state["coach_state"]
        analysis = state["reflection_analysis"]
        assert analysis is not None

        reply = generate_coaching_reply_with_llm(
            llm=llm,
            analysis=analysis,
            state=cstate,
            habits_status=f"Active habits: {len(cstate.goals_data.habits)}",
        )

        outbounds: list[OutboundEvent] = list(state.get("outbound_events") or [])

        # Emit updated adaptive metrics to Adaptive Manager
        outbounds.append(
            OutboundEvent(
                target="MANAGER",
                command="ADAPTIVE_METRICS_UPDATED",
                payload={
                    "user_id": cstate.user_id,
                    "motivation_index": cstate.adaptive_model.motivation_index,
                    "burnout_risk_score": cstate.adaptive_model.burnout_risk.score,
                    "burnout_risk_level": cstate.adaptive_model.burnout_risk.level,
                    "consecutive_low_days": cstate.consecutive_low_days,
                },
            )
        )

        # If LLM flags need for strategy pivot
        if reply.suggest_strategy_change:
            active_goals = [g for g in cstate.all_goals() if g.status == "active"]
            target_goal_id = active_goals[0].goal_id if active_goals else "all_goals"
            if active_goals:
                update_goal_strategy(cstate, target_goal_id, reply.strategy_note)

            outbounds.append(
                OutboundEvent(
                    target="MANAGER",
                    command="REQUEST_STRATEGY_ADJUSTMENT",
                    payload={
                        "user_id": cstate.user_id,
                        "goal_id": target_goal_id,
                        "reason": "Coaching analysis detected high friction/fatigue requiring strategy pivot",
                        "suggested_action": reply.strategy_note,
                        "motivation_index": cstate.adaptive_model.motivation_index,
                        "burnout_risk": cstate.adaptive_model.burnout_risk.level,
                    },
                )
            )

        return {
            "coach_state": cstate,
            "outbound_events": outbounds,
            "coach_reply": reply,
            "user_message": reply.message,
        }

    # -----------------------------------------------------------------------
    # Flow 3: Manager Signal / Accountability Check-in Nodes
    # -----------------------------------------------------------------------

    def handle_manager_signal_node(state: CoachGraphState) -> dict[str, Any]:
        """Node 3: Process inbound Adaptive Manager signal (e.g. TRIGGER_ACCOUNTABILITY_CHECKIN)."""
        cstate = state["coach_state"]
        payload = state["event"].payload

        signal_type = payload.get("command") or payload.get("signal_type", "TRIGGER_ACCOUNTABILITY_CHECKIN")
        reason = payload.get("reason", "Procrastination or avoidance detected by Adaptive Manager")
        details = str(payload.get("details", payload))

        reply = generate_accountability_reply_with_llm(
            llm=llm,
            signal_type=signal_type,
            reason=reason,
            details=details,
            state=cstate,
        )

        # Update strategy note on target goal if provided
        goal_id = payload.get("goal_id")
        if goal_id:
            update_goal_strategy(cstate, goal_id, reply.strategy_note)
        else:
            active_goals = [g for g in cstate.all_goals() if g.status == "active"]
            if active_goals:
                update_goal_strategy(cstate, active_goals[0].goal_id, reply.strategy_note)

        outbounds: list[OutboundEvent] = list(state.get("outbound_events") or [])
        outbounds.append(
            OutboundEvent(
                target="MANAGER",
                command="ACCOUNTABILITY_CHECKIN_COMPLETED",
                payload={
                    "user_id": cstate.user_id,
                    "signal_type": signal_type,
                    "strategy_note": reply.strategy_note,
                    "suggest_strategy_change": reply.suggest_strategy_change,
                },
            )
        )

        return {
            "coach_state": cstate,
            "outbound_events": outbounds,
            "coach_reply": reply,
            "user_message": reply.message,
        }

    def handle_unknown_node(state: CoachGraphState) -> dict[str, Any]:
        """Fallback node for unrecognized event types."""
        ev = state["event"].event_type
        return {
            "error": f"Unsupported or unknown event_type: '{ev}'",
            "user_message": f"I received an unsupported request type: {ev}",
        }

    # -----------------------------------------------------------------------
    # Graph Wiring
    # -----------------------------------------------------------------------
    builder = StateGraph(CoachGraphState)

    # Register nodes
    builder.add_node("decompose_goal", decompose_goal_node)
    builder.add_node("persist_goal", persist_goal_node)
    builder.add_node("analyze_reflection", analyze_reflection_node)
    builder.add_node("update_metrics", update_metrics_node)
    builder.add_node("generate_coaching", generate_coaching_node)
    builder.add_node("handle_manager_signal", handle_manager_signal_node)
    builder.add_node("handle_unknown", handle_unknown_node)

    # Route from START based on event_type
    def route_event(state: CoachGraphState) -> str:
        etype = state["event"].event_type.strip().upper()
        if etype in ("GOAL_CREATED", "GOAL_UPDATED", "CREATE_GOAL"):
            return "decompose_goal"
        if etype in ("DAILY_CHECKIN", "LOG_REFLECTION", "CHECKIN"):
            return "analyze_reflection"
        if etype in ("MANAGER_SIGNAL", "TRIGGER_ACCOUNTABILITY_CHECKIN", "ACCOUNTABILITY_CHECKIN"):
            return "handle_manager_signal"
        return "handle_unknown"

    builder.add_conditional_edges(
        START,
        route_event,
        {
            "decompose_goal": "decompose_goal",
            "analyze_reflection": "analyze_reflection",
            "handle_manager_signal": "handle_manager_signal",
            "handle_unknown": "handle_unknown",
        },
    )

    # Flow 1: Goal decomposition & persistence
    builder.add_edge("decompose_goal", "persist_goal")
    builder.add_edge("persist_goal", END)

    # Flow 2: Daily Checkin pipeline
    builder.add_edge("analyze_reflection", "update_metrics")
    builder.add_edge("update_metrics", "generate_coaching")
    builder.add_edge("generate_coaching", END)

    # Flow 3: Manager Signal
    builder.add_edge("handle_manager_signal", END)
    builder.add_edge("handle_unknown", END)

    return builder.compile()

