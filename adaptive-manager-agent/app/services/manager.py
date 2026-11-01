from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.tables import StudentModel, EventRecord, RunRecord, DispatchRecord
from app.services.analysis import DEFAULT_MODEL, observe, analyze_patterns, predict, adapt, create_commands
from app.services.llm import generate_behavior_insight


def get_or_create_model(db: Session, student_id: str) -> StudentModel:
    model = db.scalar(select(StudentModel).where(StudentModel.student_id == student_id))
    if model:
        return model
    model = StudentModel(student_id=student_id, **DEFAULT_MODEL)
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def run_manager(db: Session, student_id: str) -> dict:
    stored = db.scalars(
        select(EventRecord)
        .where(EventRecord.student_id == student_id)
        .order_by(EventRecord.id)
    ).all()
    if not stored:
        raise ValueError("No events found for this student. Submit events before running the manager.")

    latest_run = db.scalar(
        select(RunRecord)
        .where(RunRecord.student_id == student_id)
        .order_by(RunRecord.id.desc())
    )
    last_processed_id = (latest_run.result or {}).get("last_processed_event_id", 0) if latest_run else 0
    new_rows = [row for row in stored if row.id > last_processed_id]
    if not new_rows and latest_run:
        # Re-running without new input must not repeatedly apply the same EMA
        # update or create duplicate dispatch commands.
        result = dict(latest_run.result)
        result["run_id"] = latest_run.id
        result["no_new_events"] = True
        return result

    def to_event(row):
        return {
            **row.payload,
            "event_id": row.event_id,
            "timestamp": row.timestamp.isoformat(),
            "source": row.source,
            "type": row.type,
        }

    events = [to_event(row) for row in stored]
    new_events = [to_event(row) for row in new_rows]
    metrics = observe(events)             # Full history for reporting and risk.
    new_metrics = observe(new_events)     # Only new evidence updates the adaptive model.
    patterns = analyze_patterns(events)
    risk = predict(events)
    model_row = get_or_create_model(db, student_id)
    previous = {k: getattr(model_row, k) for k in DEFAULT_MODEL}
    updated, block = adapt(previous, new_metrics, risk)
    updated["procrastination_patterns"] = patterns

    # LLM is optional: the system remains fully functional using deterministic analysis.
    fallback_insight = _build_insight(metrics, patterns, updated, risk)
    insight = generate_behavior_insight(
        {k: v for k, v in metrics.items() if k != "events"},
        patterns,
        updated,
        fallback_insight,
    )
    model_row.effective_focus_duration_mins = updated["effective_focus_duration_mins"]
    model_row.procrastination_patterns = updated["procrastination_patterns"]
    model_row.task_delay_factor = updated["task_delay_factor"]
    model_row.learning_rate_velocity = updated["learning_rate_velocity"]
    model_row.current_system_mode = updated["current_system_mode"]
    db.flush()

    proposed_commands = create_commands(new_metrics, risk)
    existing_dispatches = db.scalars(
        select(DispatchRecord).where(DispatchRecord.student_id == student_id)
    ).all()
    existing_keys = {(item.command, item.reason) for item in existing_dispatches}
    commands = [
        item for item in proposed_commands
        if (item["command"], item["reason"]) not in existing_keys
    ]
    result = {
        "student_id": student_id,
        "insight": insight,
        "adaptive_model": updated,
        "metrics": {k: v for k, v in metrics.items() if k != "events"},
        "daily_risk": risk["daily_risk"],
        "risk_score": risk["risk_score"],
        "consecutive_risk_days": risk["consecutive_risk_days"],
        "predictions": risk["predictions"],
        "recommended_block_mins": block,
        "commands": commands,
        "last_processed_event_id": max(row.id for row in new_rows),
        "no_new_events": False,
    }
    run = RunRecord(student_id=student_id, result=result)
    db.add(run)
    db.flush()
    for command in commands:
        db.add(DispatchRecord(student_id=student_id, run_id=run.id, **command))
    db.commit()
    result["run_id"] = run.id
    return result


def _build_insight(metrics: dict, patterns: list[str], model: dict, risk: dict) -> str:
    parts = []
    if patterns:
        parts.append("Observed recurring friction patterns: " + ", ".join(patterns) + ".")
    if metrics["focus_samples"]:
        parts.append(f"Average observed focus duration is {round(sum(metrics['focus_samples'])/len(metrics['focus_samples']))} minutes.")
    if metrics["avg_energy"] is not None:
        parts.append(f"Average reported energy is {metrics['avg_energy']} out of 5.")
    parts.append(f"Current system mode is {model['current_system_mode']} with latest risk score {risk['risk_score']:.2f}.")
    return " ".join(parts) if parts else "Insufficient behavioral signals for a detailed profile; continue collecting events."


def record_approval(db: Session, student_id: str, command_id: int, approved: bool) -> dict | None:
    command = db.scalar(select(DispatchRecord).where(DispatchRecord.id == command_id, DispatchRecord.student_id == student_id))
    if not command:
        return None
    if command.status != "PENDING":
        return {"id": command.id, "status": command.status, "message": "This command has already been decided."}
    command.status = "APPROVED_PENDING_INTEGRATION" if approved else "REJECTED"
    command.decision_at = datetime.now(timezone.utc)
    db.commit()
    return {"id": command.id, "status": command.status, "message": "Approval recorded. No external agent was called; integration is intentionally out of scope."}
