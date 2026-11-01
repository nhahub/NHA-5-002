# Adaptive Manager Agent

Standalone implementation of the Adaptive Manager Agent. It observes student events, analyzes behavioral patterns, predicts risk, updates a persistent adaptive model, and creates dispatch commands for the four teammate-owned agents. **It does not call or implement those agents.** Approvals are recorded as `APPROVED_PENDING_INTEGRATION` until a future integration adapter is supplied.

## 1. Requirements
- Python 3.11+
- pip
- SQLite by default; PostgreSQL can be used by changing `DATABASE_URL`.

## 2. Setup (Windows PowerShell)
```powershell
cd adaptive-manager-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```
Open `http://127.0.0.1:8000/docs` for Swagger UI.

## 3. Workflow
1. Ingest events from planning, assessment, coach, and tutor sources.
2. Run the manager: Observe -> Analyze -> Predict -> Adapt -> Orchestrate.
3. Inspect the updated adaptive model and generated commands.
4. Approve or reject commands. Approval is persisted, but no external agent is invoked.

## 4. API endpoints
| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/students/{student_id}/events` | Ingest one or more events |
| GET | `/api/v1/students/{student_id}/adaptive-model` | Read current profile |
| POST | `/api/v1/students/{student_id}/run` | Execute one analysis cycle |
| GET | `/api/v1/students/{student_id}/commands` | List dispatch commands |
| POST | `/api/v1/students/{student_id}/commands/{command_id}/approval` | Approve/reject a command |
| GET | `/api/v1/students/{student_id}/runs/latest` | Read latest report |

## 5. Example requests
In Swagger, first call `POST /api/v1/students/student-001/events` with:
```json
{
  "events": [
    {"event_id":"e1","timestamp":"2026-10-19T09:00:00+02:00","source":"planning","type":"block_cut_short","topic":"Math","day_type":"heavy_lecture","planned_mins":45,"actual_focus_mins":18},
    {"event_id":"e2","timestamp":"2026-10-19T14:00:00+02:00","source":"planning","type":"block_missed","topic":"Math","day_type":"heavy_lecture","planned_mins":45},
    {"event_id":"e3","timestamp":"2026-10-19T21:00:00+02:00","source":"coach","type":"reflection","energy":2,"procrastination_risk":0.8},
    {"event_id":"e4","timestamp":"2026-10-20T21:00:00+02:00","source":"coach","type":"reflection","energy":2,"procrastination_risk":0.9},
    {"event_id":"e5","timestamp":"2026-10-20T20:00:00+02:00","source":"assessment","type":"quiz_result","topic":"Automata","score_pct":45,"failed_concepts":["DFA minimization"]},
    {"event_id":"e6","timestamp":"2026-10-22T00:00:00+02:00","source":"planning","type":"upcoming_load","lecture_hours":6,"exam":"Automata"}
  ]
}
```
Then call `POST /api/v1/students/student-001/run`.

## 6. Tests
```powershell
python -m pytest -v
```
The automated suite covers metric extraction, pattern detection, risk streaks, adaptation, same-batch duplicate event IDs, and command decision status consistency.

## 7. Important design notes
- The core analysis, risk scoring, adaptation, and command routing are deterministic rule-based logic; they do not require an LLM/API key. The optional LLM only rewrites the short natural-language insight and does not make decisions.
- If `LLM_API_KEY` is configured, the service optionally enriches the behavioral insight through the OpenAI-compatible endpoint configured in `.env` (Groq defaults are provided). If the provider is unavailable, the deterministic fallback is used.
- `effective_focus_duration_mins` is an exponentially smoothed estimate (EMA), not the raw mean of the latest focus samples. This avoids overreacting to one session; `recommended_block_mins` is separately bounded to 15–45 minutes.
- `missed_blocks` counts explicit `block_missed` events only. A `block_cut_short` event contributes its focus duration and shortfall risk, but is not counted as fully missed.
- `learning_rate_velocity` is updated only when there are at least two quiz attempts for a topic. A single score is evidence of current performance, not evidence of a learning trend.
- Consecutive high-risk days require adjacent calendar dates; a missing calendar day breaks the streak.
- Each event is used for adaptation once. Re-running the manager without newly ingested events returns the previous run instead of applying the same EMA update again or duplicating commands.
- The latest-run endpoint reconciles command statuses with persisted decisions, while the original run record remains a point-in-time analysis snapshot.
- Event IDs are idempotent per student; duplicate events are ignored.
- `adaptive_model` is stored in the database, not only in process memory.
- Dispatch commands follow the specification routing matrix. They are recorded and await integration with teammates' agents.
- Thresholds are configurable in `.env`.

## 8. Delivery scope
The standalone implementation and its automated tests are included. Live integration with teammate-owned agents is intentionally not included; approval records a decision and returns `APPROVED_PENDING_INTEGRATION`. See `DELIVERY_SUMMARY.md` for the processing cycle, verified behavior, API list, and submission notes.

## 9. Current event vocabulary
Planning: `block_completed`, `block_cut_short`, `block_missed`, `task_completed`, `upcoming_load`.
Assessment: `quiz_result`.
Coach: `reflection`.
Tutor: `topic_completed`.
