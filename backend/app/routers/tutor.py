"""AI Study Room & Generative Learning Model Router (ai_depi RAG Pipeline)."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional

from backend.app.database import db
from backend.app.agents.tutor_agent import tutor_agent

router = APIRouter(prefix="/tutor", tags=["AI Study Room & RAG Tutor"])

class StudyChatIn(BaseModel):
    query: str
    topic: Optional[str] = None
    filter_doc: Optional[str] = None

class ExplanationIn(BaseModel):
    topic: str
    level: Literal["beginner", "intermediate", "advanced"] = "intermediate"
    doc_name: Optional[str] = None

class FlashcardsIn(BaseModel):
    topic: str = "Automata & Formal Languages"
    count: int = Field(default=5, ge=2, le=10)

class SummaryIn(BaseModel):
    filename: str

@router.get("/materials")
def list_tutor_materials():
    """List lecture materials indexed in the RAG vector store."""
    from backend.app.agents.quiz_agent import assessment_agent
    return {"status": "success", "materials": assessment_agent.list_materials()}

@router.post("/chat")
def study_chat(req: StudyChatIn):
    """Ask a question grounded in the course PDFs; returns answer with document & page citations."""
    model = db.get_adaptive_model()
    weaks = model.get("weak_topics", [])
    user_level = "intermediate"

    res = tutor_agent.study_chat(
        query=req.query,
        topic=req.topic,
        filter_doc=req.filter_doc,
        user_level=user_level,
        weak_topics=weaks
    )
    return {"status": "success", "result": res}

@router.post("/explain")
def generate_explanation(req: ExplanationIn):
    """Generate in-depth pedagogical explanation addressing learner level and weak concepts."""
    model = db.get_adaptive_model()
    weaks = model.get("weak_topics", [])

    res = tutor_agent.generate_personalized_explanation(
        topic=req.topic,
        user_level=req.level,
        weak_topics=weaks,
        doc_name=req.doc_name
    )
    return {"status": "success", "result": res}

@router.post("/flashcards")
def generate_flashcards(req: FlashcardsIn):
    """Generate active recall flashcards for spaced repetition."""
    cards = tutor_agent.generate_flashcards(topic=req.topic, count=req.count)
    return {"status": "success", "topic": req.topic, "flashcards": cards}

@router.post("/summary")
def generate_summary(req: SummaryIn):
    """Generate structured summary of selected course document."""
    res = tutor_agent.generate_summary(filename=req.filename)
    return {"status": "success", "result": res}
