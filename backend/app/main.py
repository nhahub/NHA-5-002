"""Unified FastAPI Application for the Adaptive AI Personal Life & Study Coach.
Integrates all 5 specialized agents into a cohesive closed-loop system and serves the web UI.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.config import settings
from backend.app.database import db
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
    return {
        "status": "online",
        "app_name": settings.app_name,
        "version": settings.version,
        "agents_online": [
            "Agent 1: Generative Learning Model (ai_depi RAG Tutor)",
            "Agent 2: Assessment Agent (Abdelrahman Quiz Engine)",
            "Agent 3: Personal Planning Agent (5-step Deterministic Optimizer)",
            "Agent 4: Goal & Life Coach Agent (Accountability & Gamification)",
            "Agent 5: Adaptive Manager Agent (Personal Behavioral Model Orchestrator)"
        ]
    }

@app.get(f"{settings.api_prefix}/user/profile")
def get_user_profile():
    return {"status": "success", "user": db.get_user()}

@app.put(f"{settings.api_prefix}/user/profile")
def update_user_profile(updates: dict):
    updated = db.update_user(updates)
    return {"status": "success", "user": updated}

# Static Files & Frontend SPA serving
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
