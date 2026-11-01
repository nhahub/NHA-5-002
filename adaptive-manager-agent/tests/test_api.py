from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.session import Base, get_db


def make_client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app), engine


def test_duplicate_event_ids_inside_same_batch_are_ignored():
    client, engine = make_client()
    body = {
        "events": [
            {"event_id": "same-id", "timestamp": "2026-10-30T10:00:00Z", "source": "planning", "type": "block_cut_short", "planned_mins": 45, "actual_focus_mins": 15},
            {"event_id": "same-id", "timestamp": "2026-10-30T10:00:00Z", "source": "planning", "type": "block_cut_short", "planned_mins": 45, "actual_focus_mins": 15},
        ]
    }
    response = client.post("/api/v1/students/test-student/events", json=body)
    assert response.status_code == 201
    assert response.json()["created"] == 1
    assert response.json()["duplicates_ignored"] == 1
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)


def test_latest_run_reflects_current_command_decisions():
    client, engine = make_client()
    events = {
        "events": [
            {"event_id": "q1", "timestamp": "2026-10-27T10:00:00Z", "source": "assessment", "type": "quiz_result", "topic": "Math", "score_pct": 30, "failed_concepts": ["Fractions"]},
            {"event_id": "r1", "timestamp": "2026-10-27T20:00:00Z", "source": "coach", "type": "reflection", "procrastination_risk": 0.9, "energy": 1},
            {"event_id": "r2", "timestamp": "2026-10-28T20:00:00Z", "source": "coach", "type": "reflection", "procrastination_risk": 0.9, "energy": 1},
        ]
    }
    assert client.post("/api/v1/students/test-student/events", json=events).status_code == 201
    run = client.post("/api/v1/students/test-student/run")
    assert run.status_code == 200
    commands = client.get("/api/v1/students/test-student/commands").json()
    assert commands
    command_id = commands[0]["id"]
    assert client.post(f"/api/v1/students/test-student/commands/{command_id}/approval", json={"approved": False}).json()["status"] == "REJECTED"
    latest = client.get("/api/v1/students/test-student/runs/latest").json()
    matched = [c for c in latest["commands"] if c["reason"] == commands[0]["reason"]]
    assert matched and matched[0]["status"] == "REJECTED"
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)


def test_repeated_run_without_new_events_does_not_reapply_adaptation():
    client, engine = make_client()
    body = {
        "events": [
            {"event_id": "b1", "timestamp": "2026-10-27T10:00:00Z", "source": "planning", "type": "block_cut_short", "planned_mins": 45, "actual_focus_mins": 15},
            {"event_id": "t1", "timestamp": "2026-10-27T12:00:00Z", "source": "planning", "type": "task_completed", "planned_mins": 40, "actual_mins": 65},
        ]
    }
    assert client.post("/api/v1/students/test-student/events", json=body).status_code == 201
    first = client.post("/api/v1/students/test-student/run").json()
    second = client.post("/api/v1/students/test-student/run").json()
    assert second["no_new_events"] is True
    assert second["run_id"] == first["run_id"]
    assert second["adaptive_model"] == first["adaptive_model"]
    assert len(client.get("/api/v1/students/test-student/commands").json()) == len(first["commands"])
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)
