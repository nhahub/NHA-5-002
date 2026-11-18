"""Study tutor endpoints backed only by the approved ai_depi RAG agent."""
import logging
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from backend.app.database import db
from backend.app.agents.learning_agent import learning_agent
from ai_depi.app.providers import ProviderError
from ai_depi.app.validation import OutputValidationError
from ai_depi.app.schemas import ExplanationRequest, SummaryRequest, FlashcardRequest

router = APIRouter(prefix="/tutor", tags=["AI Study Room & RAG Tutor"])


class StudyChatIn(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    topic: Optional[str] = None
    filter_doc: Optional[str] = None
    history: list[dict[str, str]] = Field(default_factory=list, max_length=8)


class ExplanationIn(BaseModel):
    topic: str = Field(min_length=1, max_length=300)
    level: Literal["beginner", "intermediate", "advanced"] = "intermediate"
    doc_name: Optional[str] = None


class FlashcardsIn(BaseModel):
    topic: Optional[str] = Field(default=None, max_length=300)
    count: int = Field(default=12, ge=5, le=20)
    doc_name: Optional[str] = None


class SummaryIn(BaseModel):
    filename: str


def _profile() -> dict[str, Any]:
    profile = learning_agent.learner_profile(db.get_adaptive_model(), db.get_assessments(), str(db.get_user().get("id", "")))
    profile["recommended_revision_topics"] = list(dict.fromkeys(
        topic for goal in db.get_goals() if goal.get("status") == "active"
        for topic in goal.get("relevance_tags", [])
    ))
    return profile


def _source_dict(source: Any) -> dict[str, Any]:
    row = source.model_dump() if hasattr(source, "model_dump") else dict(source)
    row["page"] = row.get("page_number")
    return row


def _run(operation):
    try:
        result = operation()
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=f"مولّد التعلم غير متاح: {exc}") from exc
    except OutputValidationError as exc:
        raise HTTPException(status_code=502, detail="تعذر التحقق من المحتوى المولد. أعد المحاولة أو راجع إعداد مزودي النموذج.") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"خدمة التعلم غير جاهزة: {type(exc).__name__}") from exc
    return result


def _require_material(filename: str | None = None) -> str | None:
    if filename:
        safe_name = Path(filename).name
        allowed = {
            Path(material.get("path") or material.get("filename", "")).name
            for course in db.get_courses()
            for material in course.get("materials", [])
        }
        if safe_name not in allowed or not (db.uploads_dir / safe_name).is_file():
            raise HTTPException(status_code=404, detail="الملف المحدد غير موجود ضمن موادك المرفوعة.")
        return safe_name
    if not any(db.uploads_dir.glob("*")):
        raise HTTPException(status_code=422, detail="أضف محاضرة إلى إحدى موادك أولًا حتى يعتمد المساعد على محتواك.")
    return None


def _material_details(filename: str) -> tuple[str, str]:
    """Return the user's original lecture title and course for a stored filename."""
    safe_name = Path(filename).name
    for course in db.get_courses():
        for material in course.get("materials", []):
            stored = Path(material.get("path") or material.get("filename", "")).name
            if stored == safe_name:
                title = material.get("title") or Path(safe_name).stem
                for _ in range(3):
                    if Path(title).suffix.lower() in {".pdf", ".docx", ".txt"}:
                        title = Path(title).stem
                return title, course.get("name", "")
    return Path(safe_name).stem, ""


@router.get("/materials/{filename}/images")
def list_material_images(filename: str, pages: Optional[str] = None):
    """Expose embedded figures from the selected uploaded lecture PDF."""
    safe_name = _require_material(filename)
    path = db.uploads_dir / safe_name
    if path.suffix.lower() != ".pdf":
        return {"status": "success", "images": []}
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        selected_pages = {int(value) for value in (pages or "").split(",") if value.isdigit()}
        found = []
        for page_index, page in enumerate(reader.pages, start=1):
            if selected_pages and page_index not in selected_pages:
                continue
            try:
                page_images = page.images
            except Exception:
                continue
            for image_index, image in enumerate(page_images):
                if not image.data or len(image.data) > 6 * 1024 * 1024:
                    continue
                found.append({
                    "page": page_index,
                    "alt": f"شكل من المحاضرة، صفحة {page_index}",
                    "url": f"/api/tutor/materials/{quote(safe_name)}/images/{page_index}/{image_index}",
                })
                if len(found) >= 8:
                    break
            if len(found) >= 8:
                break
        return {"status": "success", "images": found}
    except Exception as exc:
        raise HTTPException(status_code=422, detail="تعذر قراءة الصور المضمنة في المحاضرة.") from exc


@router.get("/materials/{filename}/images/{page_number}/{image_index}")
def get_material_image(filename: str, page_number: int, image_index: int):
    safe_name = _require_material(filename)
    path = db.uploads_dir / safe_name
    if path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="الملف لا يحتوي على صور صفحات.")
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        if page_number < 1 or page_number > len(reader.pages):
            raise HTTPException(status_code=404, detail="رقم الصفحة غير موجود.")
        images = reader.pages[page_number - 1].images
        if image_index < 0 or image_index >= len(images):
            raise HTTPException(status_code=404, detail="الصورة غير موجودة في الصفحة.")
        data = images[image_index].data
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            media_type = "image/png"
        elif data.startswith(b"\xff\xd8\xff"):
            media_type = "image/jpeg"
        elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            media_type = "image/webp"
        else:
            media_type = "application/octet-stream"
        return Response(data, media_type=media_type, headers={
            "Content-Disposition": "inline",
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "private, max-age=3600",
        })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail="تعذر استخراج صورة المحاضرة.") from exc


@router.get("/materials")
def list_tutor_materials():
    materials = [
        {"course_id": course.get("id"), "course_name": course.get("name"),
         "id": material.get("id"), "title": material.get("title"),
         "filename": Path(material.get("path") or material.get("filename", "")).name}
        for course in db.get_courses() for material in course.get("materials", [])
    ]
    return {"status": "success", "materials": materials, "rag": learning_agent.status()}


@router.post("/sync")
def sync_tutor_materials():
    return {"status": "success", "index": learning_agent.sync_materials()}


@router.get("/diagnose")
def diagnose_rag(filename: Optional[str] = None):
    """Shows why the RAG tutor may read nothing: unreadable PDFs, empty index, metadata filter mismatch."""
    return {"status": "success", "diagnosis": learning_agent.diagnose(filename)}


def _direct_chat_response(req: "StudyChatIn", filename: Optional[str], history_text: Optional[str], why: str):
    fallback = learning_agent.direct_answer(req.query, filename, history_text)
    if not fallback.get("found"):
        raise HTTPException(status_code=422, detail=fallback.get("notice") or why)
    return {"status": "success", "result": {
        "answer": fallback["answer"],
        "notice": "الإجابة من قراءة مباشرة للملف لأن البحث الذكي (RAG) لم يرجع نتائج.",
        "sources": fallback["sources"],
        "grounded": True,
        "mode": fallback["mode"],
        "rag_fallback_reason": why,
    }}


@router.post("/chat")
def study_chat(req: StudyChatIn):
    filename = _require_material(req.filter_doc)
    topic = (req.topic or req.query).strip()[:300]
    history = []
    for item in req.history[-8:]:
        role = item.get("role", "")
        content = " ".join(item.get("content", "").split())[:1200]
        if role in {"user", "assistant"} and content:
            history.append(f"{role}: {content}")
    conversation_context = "\n".join(history)[-6000:] or None
    learning_agent.ensure_index()
    try:
        generator = learning_agent.generator(filename)
        response = generator.explain(ExplanationRequest(
            topic=topic,
            question=req.query,
            conversation_context=conversation_context,
            learner_profile=_profile(),
            knowledge_mode="learning_materials_only",
        ))
    except Exception as exc:  # RAG pipeline/provider failure -> read the file directly
        logging.getLogger("adaptive_coach.tutor").exception("RAG chat failed; using direct document reading")
        return _direct_chat_response(req, filename, conversation_context, f"{type(exc).__name__}: {exc}"[:300])
    data = response.model_dump(mode="json")
    if response.grounding.retrieval_status != "sufficient":
        return _direct_chat_response(req, filename, conversation_context,
                                     f"retrieval_status={response.grounding.retrieval_status}")
    return {"status": "success", "result": {
        "answer": response.explanation,
        "notice": response.grounding.notice,
        "sources": [_source_dict(item) for item in response.grounding.sources],
        "grounded": response.grounding.grounding_status == "grounded",
        "mode": ("extractive_fallback" if response.grounding.failure_status == "generation_failed"
                 else "llm_ai_depi" if response.generation else "retrieved_ai_depi"),
        "learning_result": data,
    }}


@router.post("/explain")
def generate_explanation(req: ExplanationIn):
    filename = _require_material(req.doc_name)
    response = _run(lambda: learning_agent.generator(filename).explain(ExplanationRequest(
        topic=req.topic,
        difficulty="easy" if req.level == "beginner" else "hard" if req.level == "advanced" else "medium",
        learner_profile=_profile(),
        knowledge_mode="learning_materials_only",
    )))
    if response.grounding.retrieval_status != "sufficient":
        raise HTTPException(status_code=422, detail=response.grounding.notice or "لم أجد شرحًا كافيًا لهذا الموضوع في الملف المحدد.")
    return {"status": "success", "result": response.model_dump(mode="json")}


@router.post("/flashcards")
def generate_flashcards(req: FlashcardsIn):
    filename = _require_material(req.doc_name)
    if not filename:
        raise HTTPException(status_code=422, detail="اختر مادة ومحاضرة قبل إنشاء بطاقات المراجعة.")
    lecture_title, _ = _material_details(filename)
    topic = (req.topic or lecture_title).strip()[:300]
    if not topic:
        topic = "الموضوعات الأساسية في المحاضرة"
    response = _run(lambda: learning_agent.generator(filename).flashcards(FlashcardRequest(
        topic=topic,
        number_of_cards=req.count,
        top_k=20,
        learner_profile=_profile(),
        knowledge_mode="learning_materials_only",
    )))
    if response.grounding.retrieval_status != "sufficient" or len(response.cards) < req.count:
        raise HTTPException(status_code=422, detail=response.grounding.notice or "لم أجد محتوى كافيًا لتكوين البطاقات المطلوبة.")
    return {"status": "success", "topic": topic,
            "flashcards": [card.model_dump(mode="json") for card in response.cards],
            "grounding": response.grounding.model_dump(mode="json")}


@router.post("/summary")
def generate_summary(req: SummaryIn):
    filename = _require_material(req.filename)
    path = db.uploads_dir / filename
    lecture_title, course_name = _material_details(filename)
    try:
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            pages = PdfReader(path).pages
            document_text = "\n".join(page.extract_text() or "" for page in pages)
        elif path.suffix.lower() == ".docx":
            from docx import Document
            document_text = "\n".join(p.text for p in Document(path).paragraphs)
        else:
            document_text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        raise HTTPException(status_code=422, detail="تعذر قراءة الملف لتلخيصه.") from exc
    learning_agent.sync_materials()
    title = lecture_title
    heading_candidates = []
    for line in document_text.splitlines():
        line = " ".join(line.split()).strip(" •·\t")
        if 4 <= len(line) <= 100 and (line.endswith(":") or len(line.split()) <= 12):
            heading_candidates.append(line.rstrip(":"))
    heading_context = " | ".join(dict.fromkeys(heading_candidates))[:1500]
    query = f"{title}. Full lecture review: {heading_context or 'main ideas, key terms, definitions, examples, relationships, and exam points'}"[:2000]
    response = _run(lambda: learning_agent.generator(filename).summarize(SummaryRequest(
        topic=query,
        top_k=20,
        style="revision-sheet",
        learner_profile=_profile(),
        knowledge_mode="learning_materials_only",
    )))
    if response.grounding.retrieval_status != "sufficient":
        raise HTTPException(status_code=422, detail=response.grounding.notice or "لم أجد نصًا كافيًا لتلخيصه.")
    result = response.model_dump(mode="json")
    result["document"] = {"filename": filename, "title": title, "course_name": course_name,
                          "pages": len(pages) if path.suffix.lower() == ".pdf" else None}
    return {"status": "success", "result": result}
