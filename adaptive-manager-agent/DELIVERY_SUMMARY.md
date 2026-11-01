# Adaptive Manager Agent — Delivery Summary

## Project scope
This repository contains the standalone Adaptive Manager Agent. It observes student activity events, updates a persistent student-specific adaptive model, estimates academic/behavioral risk, predicts upcoming risk, and creates dispatch commands for other agents.

The implementation is intentionally independent: it does **not** implement or invoke the Tutor, Assessment, Planning, or Goal & Life Coach agents. Commands are recorded for later integration. An approved command is marked `APPROVED_PENDING_INTEGRATION`; approval does not mean the target agent was executed.

## Processing cycle
1. **Observe:** ingest events from supported sources (planning, assessment, coach, tutor).
2. **Understand:** aggregate focus, task completion, quiz, energy, and procrastination signals.
3. **Predict:** calculate risk and identify upcoming high-risk days from available events.
4. **Adapt:** update and persist the student's adaptive model.
5. **Orchestrate:** create deduplicated dispatch commands according to the routing rules.
6. **Review:** expose commands and record an approval/rejection decision.

## Adaptive model
The model currently tracks:
- `effective_focus_duration_mins`
- `procrastination_patterns`
- `task_delay_factor`
- `learning_rate_velocity`
- `current_system_mode`

## Dispatch command routing
| Command | Intended recipient |
|---|---|
| `GENERATE_DIAGNOSTIC_EXAM` | Assessment Agent |
| `INITIATE_REMEDIAL_SESSION` | AI Tutor Agent |
| `PREEMPTIVE_SCHEDULE_RESTRUCTURE` | Personal Planning Agent |
| `TRIGGER_ACCOUNTABILITY_CHECKIN` | Goal & Life Coach Agent |

Only commands relevant to the observed conditions are emitted in a run.

## Verification performed
- Automated test suite: **10 passed**.
- Manual Swagger checks: health endpoint returned HTTP 200; event ingestion accepted 6 events; analysis run generated risk predictions and three relevant commands; repeated run returned `no_new_events: true`; adaptive model was readable from its dedicated endpoint; commands were readable from the commands endpoint; approval endpoint returned HTTP 200 and `APPROVED_PENDING_INTEGRATION`.
- Input validation was also observed to return HTTP 422 for malformed JSON / invalid input.

These checks verify the standalone service and its persistence/decision workflow. They do not verify live communication with teammate-owned agents.

## Run locally (Windows PowerShell)
From the folder containing `requirements.txt`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

Open Swagger at `http://127.0.0.1:8000/docs`.

Run automated tests in a second terminal (or stop the server first):

```powershell
python -m pytest -v
```

## Main API
| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Service health |
| POST | `/api/v1/students/{student_id}/events` | Ingest events |
| POST | `/api/v1/students/{student_id}/run` | Run analysis/adaptation/orchestration |
| GET | `/api/v1/students/{student_id}/adaptive-model` | Read persisted adaptive model |
| GET | `/api/v1/students/{student_id}/commands` | List generated commands |
| POST | `/api/v1/students/{student_id}/commands/{command_id}/approval` | Record approval/rejection |
| GET | `/api/v1/students/{student_id}/runs/latest` | Read latest run report |

## Submission notes
- Keep `.env` local; submit `.env.example`, not secrets or API keys.
- The default deterministic analysis does not require an LLM API key. Optional LLM enrichment depends on provider configuration.
- The project is ready for standalone demonstration. Integration adapters and live calls to the other agents remain future integration work.
