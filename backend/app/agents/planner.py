"""Personal Planning Agent (Agent 3 of 5).
Wraps personal-planning-agent-main:
- 5-step deterministic schedule optimization
- Attention-span task deconstruction
- Telemetry, velocity slip, and workload density metrics
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure personal-planning-agent-main is on path
BASE_DIR = Path(__file__).resolve().parents[3]
PLANNING_DIR = BASE_DIR / "personal-planning-agent-main"
if str(PLANNING_DIR) not in sys.path:
    sys.path.insert(0, str(PLANNING_DIR))

from app.schema import (
    AdaptiveModelContext,
    FixedLecture,
    ScheduleRequest,
    ScheduleResult,
    ScheduledBlock,
    StudyTask,
)
from app.engine.optimizer import optimize_schedule as _raw_optimize
from app.engine.deconstructor import deconstruct_task as _raw_deconstruct
from app.engine.telemetry import (
    compute_workload_density,
    compute_velocity_slip,
    generate_study_block_invitations,
)

logger = logging.getLogger("adaptive_coach.planner")

class PlannerAgent:
    """Wrapper and coordinator for Personal Planning capabilities."""

    def optimize(
        self,
        target_date: str,
        tasks: List[Dict[str, Any]],
        lectures: List[Dict[str, Any]],
        adaptive_model: Dict[str, Any],
        busy_blocks: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Run the 5-step deterministic schedule optimization."""
        # Map tasks to StudyTask objects
        study_tasks = []
        for t in tasks:
            # Only schedule pending or scheduled tasks
            if t.get("status") in {"completed", "missed"}:
                continue
            study_tasks.append(
                StudyTask(
                    task_id=t.get("task_id", f"T-{len(study_tasks)+1}"),
                    parent_milestone_id=t.get("parent_milestone_id"),
                    title=t.get("title", "Untitled Task"),
                    source=t.get("source", "user"),
                    topic_ref=t.get("topic_ref"),
                    estimated_mins=int(t.get("estimated_mins", 30)),
                    adjusted_mins=t.get("adjusted_mins"),
                    deadline=t.get("deadline"),
                    priority_weight=float(t.get("priority_weight", 3.0)),
                    cognitive_load=t.get("cognitive_load", "medium"),
                    status=t.get("status", "pending"),
                    scheduled_date=t.get("scheduled_date"),
                    scheduled_start=t.get("scheduled_start"),
                    scheduled_end=t.get("scheduled_end"),
                )
            )

        # Map lectures to FixedLecture objects
        fixed_lectures = []
        for l in lectures:
            fixed_lectures.append(
                FixedLecture(
                    id=l.get("id", f"lec-{len(fixed_lectures)+1}"),
                    day=l.get("day", "Monday"),
                    date=l.get("date", target_date),
                    start=l.get("start", "09:00"),
                    end=l.get("end", "11:00"),
                    title=l.get("title", "Lecture"),
                )
            )

        # Build AdaptiveModelContext
        model_ctx = AdaptiveModelContext(
            peak_productivity_window=adaptive_model.get("peak_productivity_window", "09:00-11:00"),
            effective_focus_duration_mins=int(adaptive_model.get("effective_focus_duration_mins", 45)),
            avg_task_completion_ratio=float(adaptive_model.get("avg_task_completion_ratio", 0.75)),
            task_delay_factor=float(adaptive_model.get("task_delay_factor", 1.0)),
            current_system_mode=adaptive_model.get("current_system_mode", "NORMAL"),
        )

        req = ScheduleRequest(
            target_date=target_date,
            tasks=study_tasks,
            university_schedule=fixed_lectures,
            gcal_busy_blocks=busy_blocks or [],
            adaptive_model=model_ctx,
        )

        res: ScheduleResult = _raw_optimize(req)

        blocks = []
        for b in res.scheduled_blocks:
            blocks.append({
                "task_id": b.task_id,
                "title": b.title,
                "date": b.date,
                "start": b.start,
                "end": b.end,
                "part": b.part,
            })

        unscheduled = [t.model_dump() for t in getattr(res, "unscheduled_tasks", [])]

        return {
            "target_date": target_date,
            "scheduled_blocks": blocks,
            "unscheduled_tasks": unscheduled,
            "is_feasible": getattr(res, "is_feasible", True),
            "feasibility_flags": getattr(res, "feasibility_flags", []),
            "total_blocks": len(blocks),
            "total_unscheduled": len(unscheduled),
        }

    def deconstruct(
        self,
        task_data: Dict[str, Any],
        focus_duration_mins: int = 35
    ) -> List[Dict[str, Any]]:
        """Deconstruct a large task into smaller attention-span micro-tasks."""
        st = StudyTask(
            task_id=task_data.get("task_id", "T-SPLIT"),
            title=task_data.get("title", "Complex Study Task"),
            source=task_data.get("source", "user"),
            topic_ref=task_data.get("topic_ref"),
            estimated_mins=int(task_data.get("estimated_mins", 60)),
            priority_weight=float(task_data.get("priority_weight", 3.0)),
            cognitive_load=task_data.get("cognitive_load", "high"),
        )
        res = _raw_deconstruct(st, focus_duration_mins=focus_duration_mins, force_fallback=False)
        return [m.model_dump() for m in res.micro_tasks]

planner_agent = PlannerAgent()
