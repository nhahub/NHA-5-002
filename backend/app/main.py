"""Unified FastAPI Application for the Adaptive AI Personal Life & Study Coach.
Integrates all 5 specialized agents into a cohesive closed-loop system and serves the web UI.
"""

import importlib.util
import os
from urllib.request import urlopen
import json
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.config import settings
from backend.app.database import db
from backend.app.agents.learning_agent import learning_agent
from backend.app.routers.adaptive import router as adaptive_router
from backend.app.routers.planning import router as planning_router
from backend.app.routers.goals import router as goals_router
from backend.app.routers.assessment import router as assessment_router
from backend.app.routers.tutor import router as tutor_router

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Adaptive AI Personal Life & Study Coach — A 5-Agent Generative AI System featuring Continuous Personal Behavioral Adaptation."
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(adaptive_router, prefix=settings.api_prefix)
app.include_router(planning_router, prefix=settings.api_prefix)
app.include_router(goals_router, prefix=settings.api_prefix)
app.include_router(assessment_router, prefix=settings.api_prefix)
app.include_router(tutor_router, prefix=settings.api_prefix)

# System Profile & Health Endpoints
@app.get(f"{settings.api_prefix}/health")
def health():
    return {"status": "online", "app_name": settings.app_name, "version": settings.version,
            "agent_status": agent_status()}


def agent_status():
    groq_key = bool(settings.groq_api_key or os.getenv("GROQ_API_KEY"))
    gemini_key = bool(settings.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    learning_llm_ready = groq_key or gemini_key
    google_provider = importlib.util.find_spec("langchain_google_genai") is not None
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    ollama_ready = False
    try:
        with urlopen(ollama_url.rstrip("/") + "/api/tags", timeout=0.5) as response:
            model_names = [item.get("name", "") for item in json.loads(response.read()).get("models", [])]
            ollama_ready = any(name == ollama_model or name.split(":", 1)[0] == ollama_model.split(":", 1)[0] for name in model_names)
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    state = db.get_state()
    material_count = sum(len(course.get("materials", [])) for course in db.get_courses())
    event_count = len(db.get_events())
    task_count = len(db.get_tasks())
    goal_count = len(db.get_goals())
    commitment_count = len(db.get_university_schedule()) + len(state.get("calendar_events", []))
    user_profile = db.get_user()
    planner_needs = []
    if not user_profile.get("name") or not user_profile.get("day_start") or not user_profile.get("day_end"):
        planner_needs.append("أدخل اسمك وحدد الفترة اليومية التي تسمح فيها بالتخطيط.")
    if not task_count and not commitment_count:
        planner_needs.append("ارفع صورة جدولك أو أضف التزاماتك ومهامك حتى يظهر أسبوعك الفعلي.")
    agents = [
        {"id": "adaptive_manager", "name": "Adaptive Manager", "ready": True,
         "mode": "native behavior analysis + schedule interventions", "missing": [],
         "needs": [] if event_count else ["سجّل جلسات تركيز أو اختبارات أو مراجعة يومية ليبني النموذج الشخصي من سلوكك."]},
        {"id": "planner", "name": "Personal Planning", "ready": True,
         "mode": "native schedule optimizer + image timetable reader", "missing": [] if gemini_key and google_provider else [x for x, ok in [("GEMINI_API_KEY (required for timetable image reading)", gemini_key), ("langchain-google-genai (required for image reading; optional for task splitting)", google_provider)] if not ok],
         "needs": planner_needs},
        {"id": "goal_coach", "name": "Goal & Life Coach", "ready": bool(gemini_key and google_provider),
         "mode": "native structured coach prompts", "missing": [x for x, ok in [("GEMINI_API_KEY or GOOGLE_API_KEY", gemini_key), ("langchain-google-genai", google_provider)] if not ok],
         "needs": [] if goal_count else ["أضف هدفًا دراسيًا أو مهنيًا أو شخصيًا ليبني المدرب مراحله."]},
        {"id": "assessment", "name": "Abdelrahman Quiz Agent", "ready": groq_key,
         "mode": "Groq question generation from uploaded material", "missing": [] if groq_key else ["GROQ_API_KEY"],
         "needs": [] if material_count else ["أضف مادة وارفع محاضرة ليولد الاختبار من محتواك."]},
        {"id": "learning_rag", "name": "ai_depi Generative Learning + RAG", "ready": learning_llm_ready,
         "mode": learning_agent.status(), "missing": [] if learning_llm_ready else ["GROQ_API_KEY or GEMINI_API_KEY for grounded generation and evidence validation"],
         "needs": [] if material_count else ["ارفع ملفات المحاضرات التي تريد للـRAG استخدامها."]},
    ]
    return {"agents": agents, "ready_count": sum(bool(a["ready"]) for a in agents), "count": len(agents),
            "missing_configuration": list(dict.fromkeys(item for agent in agents for item in agent["missing"])),
            "missing_inputs": list(dict.fromkeys(item for agent in agents for item in agent["needs"])),
            "ollama_model_ready": ollama_ready, "ollama_model": ollama_model, "ollama_endpoint": ollama_url}


@app.get(f"{settings.api_prefix}/agents/status")
def get_agent_status():
    return agent_status()


@app.on_event("startup")
def sync_uploaded_learning_materials():
    learning_agent.sync_materials()

@app.get(f"{settings.api_prefix}/user/profile")
def get_user_profile():
    return {"status": "success", "user": db.get_user()}

@app.put(f"{settings.api_prefix}/user/profile")
def update_user_profile(updates: dict):
    name = str(updates.get("name") or "").strip()
    if not name or len(name) > 80:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="اكتب الاسم الذي تريد أن يظهر في مساحتك.")
    updates["name"] = name
    start, end = updates.get("day_start"), updates.get("day_end")
    if bool(start) != bool(end):
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="أدخل بداية ونهاية الوقت المتاح معًا.")
    if start and end:
        try:
            if datetime.strptime(start, "%H:%M") >= datetime.strptime(end, "%H:%M"):
                raise ValueError
        except ValueError as exc:
            from fastapi import HTTPException
            raise HTTPException(status_code=422, detail="نهاية الوقت المتاح يجب أن تأتي بعد بدايته.") from exc
    previous = db.get_user().copy()
    updated = db.update_user(updates)
    plan = None
    if ((previous.get("day_start") != updated.get("day_start")
         or previous.get("day_end") != updated.get("day_end"))
            and any(task.get("status") not in {"completed", "missed"} for task in db.get_tasks())):
        from backend.app.core_loop import core_orchestrator
        from zoneinfo import ZoneInfo
        plan = core_orchestrator.run_full_schedule_optimization(
            datetime.now(ZoneInfo("Africa/Cairo")).date().isoformat()
        )
    return {"status": "success", "user": updated, "plan": plan}

# Static Files & Frontend SPA serving
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
