"""Assessment & Diagnostic Center Router (Abdelrahman Quiz Agent Pipeline)."""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from backend.app.database import db
from backend.app.core_loop import core_orchestrator
from backend.app.agents.quiz_agent import assessment_agent

router = APIRouter(prefix="/assessment", tags=["Assessment & Diagnostics"])

@router.post("/courses")
def create_course(name: str = Form(...)):
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Course name is required")
    return {"status": "success", "course": db.create_course(name)}

@router.post("/courses/{course_id}/materials")
async def upload_course_material(course_id: str, file: UploadFile = File(...)):
    course = next((c for c in db.get_courses() if c.get("id") == course_id), None)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    suffix = __import__("pathlib").Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".txt", ".docx"}:
        raise HTTPException(status_code=400, detail="Upload a PDF, TXT, or DOCX file")
    filename = f"{course_id}-{__import__('uuid').uuid4().hex[:8]}{suffix}"
    target = db.uploads_dir / filename
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    target.write_bytes(content)
    pages = 1
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            pages = len(PdfReader(str(target)).pages)
        except Exception:
            target.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Could not read this PDF")
    material = db.add_material_to_course(course_id, file.filename or filename, filename, str(target), pages, round(len(content)/1024, 1))
    return {"status": "success", "material": material}

class GenerateQuizIn(BaseModel):
    course_name: str = ""
    lecture_title: Optional[str] = None
    material_filename: Optional[str] = None
    num_questions: int = Field(default=15, ge=5, le=25)

class CheckSingleIn(BaseModel):
    question_id: str
    selected_answer: str
    correct_answer: str
    explanation: str

class SubmitQuizIn(BaseModel):
    quiz_id: str
    course_name: str
    questions: List[Dict[str, Any]]
    answers: Dict[str, str]

@router.get("/courses")
def get_quiz_courses():
    """Retrieve courses and their uploaded lecture files for quiz targeting."""
    courses = db.get_courses()
    result = []
    for c in courses:
        result.append({
            "course_id": c.get("id"),
            "course_name": c.get("name"),
            "code": c.get("code"),
            "lectures": [
                {
                    "id": m.get("id"),
                    "title": m.get("title"),
                    "filename": m.get("filename"),
                    "pages": m.get("pages")
                }
                for m in c.get("materials", [])
            ]
        })
    return {"status": "success", "courses": result}

@router.post("/generate")
def generate_quiz(req: GenerateQuizIn):
    """Generate a comprehensive diagnostic quiz targeting a specific course and lecture."""
    model = db.get_adaptive_model()
    weaks = model.get("weak_topics", [])

    quiz = assessment_agent.generate_quiz(
        course_name=req.course_name,
        lecture_title=req.lecture_title,
        material_filename=req.material_filename,
        num_questions=req.num_questions,
        target_weakness=weaks
    )
    if not quiz.get("questions"):
        raise HTTPException(status_code=422, detail="لا توجد مادة كافية لإنشاء اختبار. ارفع ملف محاضرة يحتوي على نص قابل للقراءة.")
    return {"status": "success", "quiz": quiz}

@router.post("/check_single")
def check_single_question(item: CheckSingleIn):
    """Instant check for an individual answer (instant interactive feedback)."""
    is_correct = (item.selected_answer.strip().upper() == item.correct_answer.strip().upper())
    return {
        "status": "success",
        "question_id": item.question_id,
        "is_correct": is_correct,
        "selected_answer": item.selected_answer,
        "correct_answer": item.correct_answer,
        "explanation": item.explanation
    }

@router.post("/submit")
def submit_quiz(submission: SubmitQuizIn):
    """Grade complete quiz, classify error types, isolate weak concepts, and adapt Personal Model."""
    res = core_orchestrator.submit_quiz_and_adapt(
        quiz_id=submission.quiz_id,
        topic=submission.course_name,
        questions=submission.questions,
        answers=submission.answers
    )
    return {"status": "success", "diagnostic_report": res}

@router.get("/history")
def get_assessment_history():
    """Retrieve history of completed diagnostic quizzes."""
    return {"status": "success", "assessments": db.get_assessments()}
