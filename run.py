"""Single-command launcher for the Adaptive AI Personal Life & Study Coach."""

import uvicorn
from backend.app.config import settings

if __name__ == "__main__":
    print("=" * 70)
    print("  ADAPTIVE AI PERSONAL LIFE & STUDY COACH")
    print("  Multi-Agent System & Personal Behavioral Model Platform")
    print("=" * 70)
    print("  [Agent 1] Generative Learning Model (ai_depi RAG Tutor)")
    print("  [Agent 2] Assessment Agent (Abdelrahman Quiz Engine)")
    print("  [Agent 3] Personal Planning Agent (5-step Deterministic Optimizer)")
    print("  [Agent 4] Goal & Life Coach Agent (Accountability & Gamification)")
    print("  [Agent 5] Adaptive Manager Agent (Personal Behavioral Model Orchestrator)")
    print("=" * 70)
    print(f"  Web Dashboard: http://localhost:{settings.port}")
    print(f"  Interactive API Docs: http://localhost:{settings.port}/docs")
    print("=" * 70)
    uvicorn.run("backend.app.main:app", host=settings.host, port=settings.port, reload=False)
