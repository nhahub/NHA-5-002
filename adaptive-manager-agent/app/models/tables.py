from datetime import datetime, timezone
from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from app.db.session import Base

class StudentModel(Base):
    __tablename__ = "student_adaptive_models"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    effective_focus_duration_mins: Mapped[int] = mapped_column(Integer, default=45)
    procrastination_patterns: Mapped[list] = mapped_column(JSON, default=list)
    task_delay_factor: Mapped[float] = mapped_column(Float, default=1.0)
    learning_rate_velocity: Mapped[float] = mapped_column(Float, default=0.5)
    current_system_mode: Mapped[str] = mapped_column(String(50), default="NORMAL")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class EventRecord(Base):
    __tablename__ = "events"
    __table_args__ = (UniqueConstraint("student_id", "event_id", name="uq_student_event"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[str] = mapped_column(String(100), index=True)
    event_id: Mapped[str] = mapped_column(String(100), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    source: Mapped[str] = mapped_column(String(50))
    type: Mapped[str] = mapped_column(String(60))
    payload: Mapped[dict] = mapped_column(JSON, default=dict)

class RunRecord(Base):
    __tablename__ = "manager_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[str] = mapped_column(String(100), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    result: Mapped[dict] = mapped_column(JSON)

class DispatchRecord(Base):
    __tablename__ = "dispatch_commands"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[str] = mapped_column(String(100), index=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("manager_runs.id"), index=True)
    command: Mapped[str] = mapped_column(String(80))
    target_agent: Mapped[str] = mapped_column(String(80))
    reason: Mapped[str] = mapped_column(Text)
    context: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(30), default="PENDING")
    decision_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
