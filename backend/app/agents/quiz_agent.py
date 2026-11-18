"""Assessment Agent (Member 2 of 5).
Extracted, upgraded, and productionized from Abdelrahman's Quiz Agent notebooks.
Features:
- Course & Lecture specific examination blueprint
- Comprehensive question pools (10 to 15+ questions covering definitions, proofs, state tables, edge cases)
- Dynamic generation from uploaded PDFs / text files
- Instant answer verification with pedagogical explanations
- Diagnostic error classification & weak concepts isolation feeding the Adaptive Personal Model
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from pypdf import PdfReader

from backend.app.config import settings

logger = logging.getLogger("adaptive_coach.assessment")

# Comprehensive question banks by course and lecture
class AssessmentAgent:
    """Production Assessment Agent managing blueprints, generation, and diagnostic evaluation."""

    def __init__(self, material_dir: Optional[Path] = None):
        self.material_dir = material_dir or settings.material_dir

    def generate_quiz(
        self,
        course_name: str,
        lecture_title: Optional[str] = None,
        material_filename: Optional[str] = None,
        num_questions: int = 10,
        target_weakness: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive multi-question quiz for a specific course and lecture."""
        logger.info("Generating quiz for course: '%s', lecture: '%s', target count: %d", course_name, lecture_title, num_questions)

        # Only user supplied lecture material is eligible as quiz content.
        if not material_filename:
            return {"quiz_id": f"quiz-{uuid.uuid4().hex[:8]}", "course_name": course_name,
                    "lecture_title": lecture_title or "Course Quiz", "total_questions": 0,
                    "source": "no_course_material", "questions": []}

        llm_quiz = self._try_llm_generation(course_name, lecture_title, material_filename, num_questions, target_weakness)
        if llm_quiz and len(llm_quiz.get("questions", [])) >= num_questions:
            return llm_quiz

        source_text = self._read_document_text(material_filename)
        if not source_text:
            source = "insufficient_material"
        elif not (settings.groq_api_key or os.environ.get("GROQ_API_KEY")):
            source = "generation_unavailable"
        else:
            source = "generation_incomplete"
        partial_questions = (llm_quiz or {}).get("questions", []) if source == "generation_incomplete" else []
        return {"quiz_id": f"quiz-{uuid.uuid4().hex[:8]}", "course_name": course_name,
                "lecture_title": lecture_title or "Course Quiz", "total_questions": len(partial_questions),
                "source": source, "questions": partial_questions}

    def _read_document_text(self, filename: str) -> str:
        """Read only the selected uploaded document as the generation source."""
        safe_name = Path(filename).name
        from backend.app.database import db
        target = db.uploads_dir / safe_name
        if not target.exists():
            return ""
        try:
            if target.suffix.lower() == ".pdf":
                return "\n".join(page.extract_text() or "" for page in PdfReader(target).pages)[:24000]
            if target.suffix.lower() == ".txt":
                return target.read_text(encoding="utf-8", errors="ignore")[:24000]
            if target.suffix.lower() == ".docx":
                from docx import Document
                return "\n".join(p.text for p in Document(target).paragraphs)[:24000]
        except Exception as exc:
            logger.warning("Could not extract quiz source %s: %s", safe_name, exc)
        return ""

    def _try_llm_generation(
        self,
        course_name: str,
        lecture_title: Optional[str],
        filename: str,
        num_questions: int,
        target_weakness: Optional[List[str]]
    ) -> Optional[Dict[str, Any]]:
        """Generate and validate a complete quiz, retrying only missing questions."""
        api_key = settings.groq_api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            return None

        source_text = self._read_document_text(filename)
        if len(source_text.strip()) < 300:
            logger.warning("Quiz source is too short filename=%s chars=%d", Path(filename).name, len(source_text.strip()))
            return None

        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model="openai/gpt-oss-20b",
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                temperature=0.2,
                max_tokens=min(8000, max(2500, num_questions * 400)),
                model_kwargs={"response_format": {"type": "json_object"}},
            )

            questions: List[Dict[str, Any]] = []
            seen: set[str] = set()
            decoder = json.JSONDecoder()
            for attempt in range(3):
                missing = num_questions - len(questions)
                if missing <= 0:
                    break
                avoid = [item["question"] for item in questions]
                prompt = f"""
                You are an expert university professor creating a comprehensive diagnostic exam.
                Course: {course_name}
                Lecture: {lecture_title or Path(filename).name}
                Generate exactly {missing} NEW and distinct multiple-choice questions in this response.
                Use ONLY the lecture source below. Cover different concepts and include recall, understanding,
                application, and misconception checks where supported by the source. Never invent facts.
                Target weak concepts when relevant: {target_weakness or 'cover the lecture comprehensively'}.
                Do not repeat or paraphrase any of these existing questions: {json.dumps(avoid, ensure_ascii=False)}
                This is completion attempt {attempt + 1} of 3. Return one valid JSON object and no Markdown.
                Its shape must be {{"questions":[{{"id":"q1","topic":"Subtopic","question":"...",
                "options":{{"A":"...","B":"...","C":"...","D":"..."}},"correct_answer":"A",
                "explanation":"Evidence-based explanation","concept":"Concept","cognitive_level":"Conceptual"}}]}}.
                Include exactly {missing} questions. Each question must have four non-empty options, one correct
                option letter A-D, and an explanation supported by the lecture.
                Lecture source:
                <lecture>
                {source_text}
                </lecture>
                """
                response = llm.invoke(prompt)
                content = response.content
                if isinstance(content, list):
                    content = "\n".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                raw_text = str(content or "")
                payload = None
                # Accept JSON wrapped in explanatory text while avoiding greedy regex parsing.
                for pos, char in enumerate(raw_text):
                    if char != "{":
                        continue
                    try:
                        candidate, _ = decoder.raw_decode(raw_text[pos:])
                    except json.JSONDecodeError:
                        continue
                    if isinstance(candidate, dict) and isinstance(candidate.get("questions"), list):
                        payload = candidate
                        break
                if not payload:
                    logger.warning("Quiz generation returned no valid JSON attempt=%d", attempt + 1)
                    continue

                before = len(questions)
                for item in payload["questions"]:
                    if not isinstance(item, dict):
                        continue
                    stem = str(item.get("question") or "").strip()
                    explanation = str(item.get("explanation") or "").strip()
                    options = item.get("options") or {}
                    if isinstance(options, list):
                        options = {chr(65 + i): value for i, value in enumerate(options[:4])}
                    if not isinstance(options, dict):
                        continue
                    normalized_options: Dict[str, str] = {}
                    for key, value in options.items():
                        key_text = str(key).strip().upper()
                        match = re.search(r"([A-D])$", key_text)
                        if match:
                            normalized_options[match.group(1)] = str(value).strip()
                    if len(normalized_options) < 4:
                        continue
                    normalized_options = {letter: normalized_options[letter] for letter in "ABCD"}
                    correct_raw = str(item.get("correct_answer") or "").strip()
                    correct_match = re.match(r"^([A-D])(?:[.)\s]|$)", correct_raw.upper())
                    correct = correct_match.group(1) if correct_match else correct_raw.upper()
                    if correct not in normalized_options:
                        correct = next((letter for letter, value in normalized_options.items()
                                        if value.casefold() == correct_raw.casefold()), "")
                    identity = re.sub(r"[^a-z0-9\u0600-\u06ff]+", "", stem.casefold())
                    if (not stem or not explanation or len(identity) < 8 or not correct
                            or any(not normalized_options[key] for key in "ABCD") or identity in seen):
                        continue
                    seen.add(identity)
                    questions.append({
                        **item,
                        "id": f"q{len(questions) + 1}",
                        "question": stem,
                        "options": normalized_options,
                        "correct_answer": correct,
                        "explanation": explanation,
                        "topic": str(item.get("topic") or lecture_title or course_name),
                        "concept": str(item.get("concept") or item.get("topic") or lecture_title or course_name),
                    })
                    if len(questions) >= num_questions:
                        break
                if len(questions) == before:
                    logger.warning("Quiz generation added no distinct valid questions attempt=%d", attempt + 1)

            if len(questions) < num_questions:
                logger.warning("Quiz generation incomplete requested=%d produced=%d after_retries=3", num_questions, len(questions))
                return {
                    "quiz_id": f"quiz-{uuid.uuid4().hex[:8]}",
                    "course_name": course_name,
                    "lecture_title": lecture_title or Path(filename).name,
                    "total_questions": len(questions),
                    "source": "generation_incomplete",
                    "questions": questions,
                }
            return {
                "quiz_id": f"quiz-{uuid.uuid4().hex[:8]}",
                "course_name": course_name,
                "lecture_title": lecture_title or Path(filename).name,
                "total_questions": num_questions,
                "source": "llm_generated_quiz",
                "questions": questions[:num_questions],
            }
        except Exception as e:
            logger.warning("LLM quiz generation failed: %s", type(e).__name__)
        return None

    def evaluate_quiz(
        self,
        quiz_id: str,
        course_name: str,
        questions: List[Dict[str, Any]],
        user_answers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Evaluate submitted answers, classify error types, isolate weak concepts."""
        score = 0
        total = len(questions)
        weak_topics = []
        error_types = {"Conceptual Gap": 0, "Application Error": 0, "Factual Slip": 0}
        detailed = []

        for q in questions:
            qid = str(q.get("id"))
            ans = user_answers.get(qid, "").strip().upper()
            correct = q.get("correct_answer", "A").strip().upper()
            is_correct = (ans == correct)

            if is_correct:
                score += 1
            else:
                concept = q.get("concept") or q.get("topic") or "General Theory"
                weak_topics.append(concept)
                level = q.get("cognitive_level", "Conceptual")
                if level == "Application":
                    error_types["Application Error"] += 1
                elif level == "Factual":
                    error_types["Factual Slip"] += 1
                else:
                    error_types["Conceptual Gap"] += 1

            detailed.append({
                "id": qid,
                "question": q.get("question"),
                "user_answer": ans,
                "correct_answer": correct,
                "is_correct": is_correct,
                "explanation": q.get("explanation"),
                "concept": q.get("concept")
            })

        pct = round((score / max(1, total)) * 100, 1)
        unique_weak = list(dict.fromkeys(weak_topics))

        summary = (
            f"You scored {score}/{total} ({pct}%). "
            + (f"Identified weak areas: {', '.join(unique_weak)}." if unique_weak else "Outstanding mastery! Zero conceptual mistakes detected.")
        )

        return {
            "quiz_id": quiz_id,
            "course_name": course_name,
            "score": score,
            "total": total,
            "score_pct": pct,
            "weak_topics": unique_weak,
            "error_types": {k: v for k, v in error_types.items() if v > 0},
            "detailed_answers": detailed,
            "feedback": summary
        }

assessment_agent = AssessmentAgent()
