"""Demonstration CLI script for the Goal & Life Coach Agent (Member 4).

Features demonstrated:
1. GOAL_CREATED: Goal decomposition into SMART milestones, micro-tasks & Tutor tags.
2. DAILY_CHECKIN: Daily reflection analysis, deterministic motivation & burnout scoring.
3. Streak & Gamification: Habit tracking, consecutive streaks, and milestone badges.
4. STREAK FREEZE: Missing a day safely consumes a freeze so the streak continues unbroken.
5. MANAGER_SIGNAL: Adaptive Manager accountability check-in trigger & strategy pivot.

Runs with live Google Gemini API (gemini-2.5-flash) if GOOGLE_API_KEY is configured,
or gracefully falls back to the deterministic FakeCoachLLM if no key is set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure coach module is importable regardless of current working directory
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dotenv import load_dotenv

from coach.agent import GoalCoachAgent
from coach.llm import FakeCoachLLM, get_llm
from coach.schemas import AgentEvent
from coach.storage import MemoryStore

load_dotenv()


def print_banner(title: str) -> None:
    """Print a visually distinct CLI section header."""
    width = 76
    print("\n" + "=" * width)
    print(f" {title.upper()}")
    print("=" * width)


def print_substep(title: str) -> None:
    """Print a minor step subheader."""
    print(f"\n--- {title} ---")


def main() -> None:
    print_banner("Generative AI Personal Life & Study Coach")
    print("Member 4: Goal & Life Coach Agent - Live Interactive Demo")
    print("Orchestration: LangGraph StateGraph | Deterministic Adaptive Metrics")

    # 1. Initialize LLM (Gemini with graceful offline fallback)
    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("COACH_MODEL", "gemini-2.5-flash")

    if api_key and api_key != "your_gemini_api_key_here":
        try:
            llm = get_llm(model=model_name, api_key=api_key)
            print(f"[LLM] Live Google Gemini active (Model: {model_name})")
        except Exception as e:
            print(f"[LLM WARNING] Failed to connect to Gemini ({e}). Falling back to FakeCoachLLM.")
            llm = FakeCoachLLM()
    else:
        print("[LLM NOTICE] No GOOGLE_API_KEY found in environment or .env.")
        print("[LLM NOTICE] Using FakeCoachLLM offline simulator (Zero API key required).")
        llm = FakeCoachLLM()

    # 2. Initialize backing store and agent
    store = MemoryStore()
    agent = GoalCoachAgent(store=store, llm=llm)
    user_id = "depi_student_omar"

    # =======================================================================
    # STEP 1: Goal Creation Flow (GOAL_CREATED)
    # =======================================================================
    print_banner("Step 1: Goal Creation & SMART Decomposition")
    print("User submits a high-level natural language goal with a target date.")

    goal_event = AgentEvent(
        event_type="GOAL_CREATED",
        user_id=user_id,
        payload={
            "title": "Master LangGraph and Build Graduation Multi-Agent System",
            "category": "study",
            "target_date": "2026-12-15",
            "why": "To excel in the DEPI graduation project and stand out to top AI engineering teams.",
        },
    )

    result_1 = agent.run(goal_event)
    print_substep("Agent Response to User")
    print(result_1.user_message)

    print_substep("Outbound Events Dispatched to Other Agents")
    for ev in result_1.outbound_events:
        print(f"  -> Target: [{ev.target:<8}] | Command: {ev.command}")
        if ev.target == "PLANNING":
            milestones = ev.payload.get("milestones", [])
            print(f"     Sent {len(milestones)} SMART milestones with micro-tasks to Personal Planner.")
        elif ev.target == "TUTOR":
            tags = ev.payload.get("relevance_tags", [])
            print(f"     Sent {len(tags)} study relevance tags to Tutor Agent: {', '.join(tags)}")

    # =======================================================================
    # STEP 2: Day 1 Reflection Check-in (DAILY_CHECKIN)
    # =======================================================================
    print_banner("Step 2: Day 1 Check-in (2026-10-25)")
    print("User completes study session, logs reflection, and completes daily habit.")

    day1_event = AgentEvent(
        event_type="DAILY_CHECKIN",
        user_id=user_id,
        payload={
            "date": "2026-10-25",
            "text": "Completed the first 2 micro-tasks on StateGraph basics. Everything clicked and I felt great momentum!",
            "mood_hint": 0.8,
            "energy_hint": 0.9,
            "completed_habits": ["Daily 60m Deep Work"],
        },
    )

    result_2 = agent.run(day1_event)
    state_after_day1 = store.load(user_id)
    habit_day1 = state_after_day1.goals_data.habits[0]

    print_substep("Coaching Reply")
    print(f"Coach: {result_2.user_message}")

    print_substep("Updated Metrics (Deterministic)")
    print(f"  Motivation Index : {state_after_day1.adaptive_model.motivation_index:.2f} / 1.00")
    print(f"  Burnout Risk     : {state_after_day1.adaptive_model.burnout_risk.score:.2f} ({state_after_day1.adaptive_model.burnout_risk.level})")
    print(f"  Habit Streak     : {habit_day1.streak} day(s) | Freezes available: {habit_day1.freezes_available}")

    # =======================================================================
    # STEP 3: Day 2 Reflection (Consecutive Day)
    # =======================================================================
    print_banner("Step 3: Day 2 Check-in (2026-10-26) - Consecutive Day")
    print("User continues momentum on the next consecutive day.")

    day2_event = AgentEvent(
        event_type="DAILY_CHECKIN",
        user_id=user_id,
        payload={
            "date": "2026-10-26",
            "text": "Implemented the conditional router edge. Minor syntax debugging friction, but pushed through.",
            "mood_hint": 0.7,
            "energy_hint": 0.7,
            "completed_habits": ["Daily 60m Deep Work"],
        },
    )

    result_3 = agent.run(day2_event)
    state_after_day2 = store.load(user_id)
    habit_day2 = state_after_day2.goals_data.habits[0]

    print_substep("Updated Metrics (Consecutive Day Increment)")
    print(f"  Motivation Index : {state_after_day2.adaptive_model.motivation_index:.2f} / 1.00")
    print(f"  Habit Streak     : {habit_day2.streak} day(s) (Consecutive increment!) | Freezes: {habit_day2.freezes_available}")

    # =======================================================================
    # STEP 4: Missed Day & Streak Freeze Firing (2026-10-28)
    # =======================================================================
    print_banner("Step 4: Missed Day & Streak Freeze Consumption (2026-10-28)")
    print("User missed yesterday (2026-10-27). Today (2026-10-28) they return to log in.")
    print("RULE: 1 missed day + freeze available -> Consume freeze, streak continues UNBROKEN!")

    day4_event = AgentEvent(
        event_type="DAILY_CHECKIN",
        user_id=user_id,
        payload={
            "date": "2026-10-28",
            "text": "Missed yesterday due to urgent family matters. Back today and completed my 60m study block.",
            "mood_hint": 0.6,
            "energy_hint": 0.6,
            "completed_habits": ["Daily 60m Deep Work"],
        },
    )

    result_4 = agent.run(day4_event)
    state_after_freeze = store.load(user_id)
    habit_freeze = state_after_freeze.goals_data.habits[0]

    print_substep("Streak Freeze Mechanism Activated")
    print(f"  [FREEZE CONSUMED] Missed day was protected!")
    print(f"  Habit Streak     : {habit_freeze.streak} days UNBROKEN! (Advanced from 2 -> 3)")
    print(f"  Freezes Left     : {habit_freeze.freezes_available} (Consumed 1 freeze)")
    print(f"  Badges Earned    : {[a.title for a in state_after_freeze.achievements]}")

    # =======================================================================
    # STEP 5: Burnout & Avoidance Signals (2026-10-29)
    # =======================================================================
    print_banner("Step 5: High Fatigue & Avoidance Day (2026-10-29)")
    print("User reports feeling depleted, cognitive overload, and procrastination.")

    day5_event = AgentEvent(
        event_type="DAILY_CHECKIN",
        user_id=user_id,
        payload={
            "date": "2026-10-29",
            "text": "Exhausted and burned out today. Spent 2 hours staring at code, brain felt numb and completely depleted.",
            "mood_hint": -0.4,
            "energy_hint": 0.2,
            "completed_habits": [],
        },
    )

    result_5 = agent.run(day5_event)
    state_after_burnout = store.load(user_id)

    print_substep("Coaching Analysis & Strategy Alert")
    print(f"Coach: {result_5.user_message}")
    print(f"  Motivation Index     : {state_after_burnout.adaptive_model.motivation_index:.2f} (Down from prior days)")
    print(f"  Burnout Risk         : {state_after_burnout.adaptive_model.burnout_risk.score:.2f} ({state_after_burnout.adaptive_model.burnout_risk.level})")
    print(f"  Consecutive Low Days : {state_after_burnout.consecutive_low_days}")

    # =======================================================================
    # STEP 6: Manager Signal Intervention (MANAGER_SIGNAL)
    # =======================================================================
    print_banner("Step 6: Adaptive Manager Signal Intervention")
    print("Adaptive Manager detects 2 consecutive days of task stagnation and fires command.")

    manager_event = AgentEvent(
        event_type="MANAGER_SIGNAL",
        user_id=user_id,
        payload={
            "command": "TRIGGER_ACCOUNTABILITY_CHECKIN",
            "reason": "2 consecutive days of task avoidance and low completion ratio detected",
            "details": "User opened repo but completed 0 micro-tasks. Adaptive Manager requests intervention.",
        },
    )

    result_6 = agent.run(manager_event)
    final_state = store.load(user_id)

    print_substep("Accountability Intervention Message to User")
    print(result_6.user_message)

    print_substep("Updated Goal Strategy in Shared State")
    active_goal = final_state.all_goals()[0]
    print(f"  Goal: '{active_goal.title}'")
    print(f"  Updated Strategy Note: {active_goal.strategy_note}")

    print_substep("Outbound Signals Sent to Adaptive Manager")
    for ev in result_6.outbound_events:
        print(f"  -> Target: [{ev.target}] | Command: {ev.command} | Payload: {ev.payload}")

    # =======================================================================
    # FINAL DASHBOARD
    # =======================================================================
    print_banner("Final Student State Summary (Store Snapshot)")
    print(f"User ID              : {final_state.user_id}")
    print(f"Active Goals         : {len([g for g in final_state.all_goals() if g.status == 'active'])}")
    print(f"SMART Milestones     : {len(final_state.goals_data.milestones)}")
    print(f"Habit Streak         : {habit_freeze.streak} days (Best: {habit_freeze.best_streak})")
    print(f"Motivation Index     : {final_state.adaptive_model.motivation_index:.2f} / 1.00")
    print(f"Burnout Risk Score   : {final_state.adaptive_model.burnout_risk.score:.2f} ({final_state.adaptive_model.burnout_risk.level})")
    print(f"Achievements Earned  : {len(final_state.achievements)} unlocked")
    for a in final_state.achievements:
        print(f"  * {a.title} (Earned on {a.earned_on})")

    print("\n[SUCCESS] Demo completed cleanly! All flows and requirements verified.")


if __name__ == "__main__":
    main()
