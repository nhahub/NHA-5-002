from app.services.analysis import DEFAULT_MODEL, analyze_patterns, observe, predict, adapt, create_commands

EVENTS = [
    {"event_id":"1","timestamp":"2026-10-20T09:00:00+00:00","source":"planning","type":"block_cut_short","topic":"Math","day_type":"heavy_lecture","planned_mins":45,"actual_focus_mins":18},
    {"event_id":"2","timestamp":"2026-10-20T14:00:00+00:00","source":"planning","type":"block_missed","topic":"Math","day_type":"heavy_lecture","planned_mins":45},
    {"event_id":"3","timestamp":"2026-10-20T21:00:00+00:00","source":"coach","type":"reflection","energy":2,"procrastination_risk":0.8},
    {"event_id":"4","timestamp":"2026-10-21T21:00:00+00:00","source":"coach","type":"reflection","energy":2,"procrastination_risk":0.9},
    {"event_id":"5","timestamp":"2026-10-21T20:00:00+00:00","source":"assessment","type":"quiz_result","topic":"Automata","score_pct":45,"failed_concepts":["DFA minimization"]},
]

def test_observe_metrics():
    metrics = observe(EVENTS)
    assert metrics["missed_blocks"] == 1
    assert metrics["scores"]["Automata"] == [45]

def test_pattern_detection():
    assert "Heavy lecture days" in analyze_patterns(EVENTS)

def test_prediction_and_commands():
    risk = predict(EVENTS)
    assert risk["consecutive_risk_days"] == 2
    metrics = observe(EVENTS)
    commands = create_commands(metrics, risk)
    assert any(c["command"] == "INITIATE_REMEDIAL_SESSION" for c in commands)
    assert any(c["command"] == "TRIGGER_ACCOUNTABILITY_CHECKIN" for c in commands)

def test_adaptation_bounds_focus_block():
    metrics = observe(EVENTS)
    risk = predict(EVENTS)
    model, block = adapt({}, metrics, risk)
    assert 15 <= block <= 45
    assert model["current_system_mode"] == "PREEMPTIVE_RESTRUCTURE"


def test_single_quiz_score_does_not_claim_learning_velocity():
    metrics = observe([{
        "event_id": "q1", "timestamp": "2026-10-20T10:00:00+00:00",
        "source": "assessment", "type": "quiz_result", "topic": "Math", "score_pct": 45,
    }])
    model, _ = adapt({}, metrics, {"predictions": [], "consecutive_risk_days": 0, "risk_score": 0.0})
    assert model["learning_rate_velocity"] == DEFAULT_MODEL["learning_rate_velocity"]


def test_consecutive_risk_days_break_when_calendar_day_is_missing():
    events = [
        {"event_id": "r1", "timestamp": "2026-10-20T20:00:00+00:00", "source": "coach", "type": "reflection", "procrastination_risk": 0.9},
        {"event_id": "r2", "timestamp": "2026-10-22T20:00:00+00:00", "source": "coach", "type": "reflection", "procrastination_risk": 0.9},
    ]
    assert predict(events)["consecutive_risk_days"] == 1


def test_cut_short_block_is_not_counted_as_fully_missed():
    metrics = observe([{
        "event_id": "b1", "timestamp": "2026-10-20T09:00:00+00:00", "source": "planning",
        "type": "block_cut_short", "planned_mins": 45, "actual_focus_mins": 15,
    }])
    assert metrics["total_blocks"] == 1
    assert metrics["missed_blocks"] == 0
    assert metrics["focus_samples"] == [15]
