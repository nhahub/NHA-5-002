from datetime import timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.tables import EventRecord, StudentModel, DispatchRecord, RunRecord
from app.schemas.events import EventBatch, ApprovalIn
from app.services.manager import get_or_create_model, run_manager, record_approval

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok", "service": "adaptive-manager-agent"}

@router.post("/students/{student_id}/events", status_code=201)
def ingest_events(student_id: str, body: EventBatch, db: Session = Depends(get_db)):
    get_or_create_model(db, student_id)
    created, duplicates = 0, 0
    seen_in_batch: set[str] = set()
    for event in body.events:
        data = event.model_dump(mode="json")
        # autoflush is disabled for this session, so also track IDs in this batch.
        if event.event_id in seen_in_batch:
            duplicates += 1
            continue
        seen_in_batch.add(event.event_id)
        exists = db.scalar(select(EventRecord).where(EventRecord.student_id == student_id, EventRecord.event_id == event.event_id))
        if exists:
            duplicates += 1
            continue
        payload = {k: v for k, v in data.items() if k not in {"event_id", "timestamp", "source", "type"}}
        timestamp = event.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        db.add(EventRecord(student_id=student_id, event_id=event.event_id, timestamp=timestamp, source=event.source, type=event.type, payload=payload))
        created += 1
    db.commit()
    return {"student_id": student_id, "created": created, "duplicates_ignored": duplicates}

@router.get("/students/{student_id}/adaptive-model")
def read_model(student_id: str, db: Session = Depends(get_db)):
    model = get_or_create_model(db, student_id)
    return {k: getattr(model, k) for k in ("student_id", "effective_focus_duration_mins", "procrastination_patterns", "task_delay_factor", "learning_rate_velocity", "current_system_mode", "updated_at")}

@router.post("/students/{student_id}/run")
def run_agent(student_id: str, db: Session = Depends(get_db)):
    try:
        return run_manager(db, student_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.get("/students/{student_id}/commands")
def list_commands(student_id: str, db: Session = Depends(get_db)):
    rows = db.scalars(select(DispatchRecord).where(DispatchRecord.student_id == student_id).order_by(DispatchRecord.id.desc()).limit(100)).all()
    return [{"id": r.id, "run_id": r.run_id, "command": r.command, "target_agent": r.target_agent, "reason": r.reason, "context": r.context, "status": r.status, "decision_at": r.decision_at} for r in rows]

@router.post("/students/{student_id}/commands/{command_id}/approval")
def approve_command(student_id: str, command_id: int, body: ApprovalIn, db: Session = Depends(get_db)):
    result = record_approval(db, student_id, command_id, body.approved)
    if result is None:
        raise HTTPException(status_code=404, detail="Command not found")
    return result

@router.get("/students/{student_id}/runs/latest")
def latest_run(student_id: str, db: Session = Depends(get_db)):
    run = db.scalar(select(RunRecord).where(RunRecord.student_id == student_id).order_by(RunRecord.id.desc()))
    if not run:
        raise HTTPException(status_code=404, detail="No runs found")
    result = dict(run.result)
    # The run report stores a point-in-time snapshot; expose current decisions
    # by reconciling each command with its persisted dispatch record.
    dispatches = db.scalars(select(DispatchRecord).where(DispatchRecord.run_id == run.id)).all()
    current_status = {(d.command, d.target_agent, d.reason): d.status for d in dispatches}
    for item in result.get("commands", []):
        key = (item.get("command"), item.get("target_agent"), item.get("reason"))
        if key in current_status:
            item["status"] = current_status[key]
    return {"run_id": run.id, **result}
