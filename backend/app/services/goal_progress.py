"""Keep Goal Coach milestones synchronized with the tasks that prove progress."""
from __future__ import annotations

from typing import Any


def sync_goal_progress_for_task(task: dict[str, Any], tasks: list[dict[str, Any]], db) -> None:
    goal_id = task.get("goal_id")
    milestone_id = task.get("parent_milestone_id")
    if not goal_id or not milestone_id:
        return

    linked = [item for item in tasks
              if item.get("goal_id") == goal_id
              and item.get("parent_milestone_id") == milestone_id]
    if not linked:
        return
    milestone_done = all(item.get("status") == "completed" for item in linked)
    goals = db.get_goals()
    changed = False
    for goal in goals:
        if goal.get("goal_id") != goal_id:
            continue
        for milestone in goal.get("milestones", []):
            if milestone.get("id") == milestone_id:
                status = "completed" if milestone_done else "pending"
                if milestone.get("status") != status:
                    milestone["status"] = status
                    changed = True
        milestones = goal.get("milestones", [])
        goal_status = "completed" if milestones and all(
            item.get("status") == "completed" for item in milestones
        ) else "active"
        if goal.get("status") != goal_status:
            goal["status"] = goal_status
            changed = True
    if changed:
        db.save_goals(goals)
