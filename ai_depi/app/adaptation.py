"""Deterministic, inspectable learner adaptation decisions."""

import json
import re

from .domain import LearnerProfile
from .schemas import AdaptationMetadata


def _norm(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _topics(topic: str) -> list[str]:
    return [part.strip() for part in topic.split(",") if part.strip()] or [topic]


def _matches(candidate: str, topics: list[str]) -> bool:
    left = _norm(candidate)
    if not left:
        return False
    return any(left == _norm(topic) or left in _norm(topic) or _norm(topic) in left for topic in topics)


def _relevant_records(value, topics: list[str]):
    """Keep only records explicitly associated with one of the requested topics."""
    found = []

    def visit(node):
        if isinstance(node, dict):
            if any(_matches(str(key), topics) for key in node):
                for key, child in node.items():
                    if _matches(str(key), topics):
                        found.append({str(key): child})
                return
            labels = [str(node.get(key, "")) for key in ("topic", "subject", "concept", "unit")]
            if any(_matches(label, topics) for label in labels):
                found.append(node)
                return
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return found


def _scores(records) -> list[float]:
    scores = []
    keys = {"score", "accuracy", "question_accuracy", "correctness", "percentage", "percent", "ratio"}

    def visit(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if str(key).casefold() in keys and isinstance(value, (int, float)) and not isinstance(value, bool):
                    score = float(value)
                    normalized_key = str(key).casefold()
                    if normalized_key in {"percentage", "percent"} and 0.0 <= score <= 100.0:
                        scores.append(score / 100.0)
                    elif 0.0 <= score <= 1.0:
                        scores.append(score)
                    elif normalized_key == "score":
                        total = node.get("total", node.get("max_score"))
                        if isinstance(total, (int, float)) and not isinstance(total, bool) and total > 0 and score <= total:
                            scores.append(score / float(total))
                else:
                    visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(records)
    return [score for score in scores if 0.0 <= score <= 1.0]


def _compact(value, limit=1000) -> str:
    blocked = {"user_id", "userid", "name", "email", "phone", "token", "secret", "password"}

    def clean(node):
        if isinstance(node, dict):
            return {str(key): clean(item) for key, item in node.items()
                    if str(key).casefold() not in blocked and not any(term in str(key).casefold() for term in ("secret", "token", "password", "email"))}
        if isinstance(node, list):
            return [clean(item) for item in node]
        if isinstance(node, str):
            return node[:300]
        return node

    return json.dumps(clean(value), ensure_ascii=False, sort_keys=True, default=str)[:limit]


def adapt(topic: str, requested: dict, profile: LearnerProfile) -> AdaptationMetadata:
    """Build effective settings plus topic-scoped instructions for the prompt.

    Requested difficulty is preserved. Learner signals move effective difficulty
    by one step at a time and add targeted scaffolding; they never alter factual
    content or the expected answer.
    """
    adapted = dict(requested)
    reasons: list[str] = []
    guidance: list[str] = []
    topics = _topics(topic)
    mastery = profile.mastery_for(topic)

    level = profile.overall_level.strip().casefold()
    if level in {"beginner", "novice", "basic"}:
        guidance.append("Use beginner-friendly vocabulary; define necessary terms using only the evidence and explain one step at a time.")
    elif level in {"advanced", "expert", "proficient"}:
        guidance.append("Use appropriately advanced terminology and concise, deeper relationships; do not restate basic definitions unless needed.")
    else:
        guidance.append("Use intermediate vocabulary and explain important terms briefly when they are needed.")

    weak = [value for value in profile.weak_topics if _matches(value, topics)]
    weak_emphasis_disabled = "weak_topic_emphasis" in requested and not requested["weak_topic_emphasis"]
    strong = [value for value in profile.strong_topics if _matches(value, topics)]
    recommended = [value for value in profile.recommended_revision_topics if _matches(value, topics)]
    if weak:
        if weak_emphasis_disabled:
            guidance.append(f"These requested topics are listed as weak ({', '.join(weak)}), but exam emphasis is disabled; do not use that profile signal to change question distribution.")
        else:
            guidance.append(f"Treat these matching topics as revision priorities: {', '.join(weak)}. Give their key ideas a little more scaffolding when relevant.")
        reasons.append("A requested topic matches the learner's weak-topic list")
    if strong:
        guidance.append(f"The learner already lists these requested topics as strengths: {', '.join(strong)}. Avoid repeating introductory material; focus on the requested new angle.")
        reasons.append("A requested topic matches the learner's strong-topic list")
    if recommended:
        guidance.append(f"Prefer these relevant revision recommendations when suggesting follow-up study: {', '.join(recommended)}.")
        reasons.append("Relevant learner revision recommendations were prioritized")

    if mastery is not None:
        if mastery < 0.5:
            guidance.append(f"Mastery of the requested topic is low ({mastery:.2f}); begin with foundational ideas and show the supported reasoning steps.")
            reasons.append(f"Topic mastery is low ({mastery:.2f}); adding foundational scaffolding")
        elif mastery >= 0.85:
            guidance.append(f"Mastery of the requested topic is high ({mastery:.2f}); avoid basic reteaching and use a more challenging supported application where the task allows it.")
            reasons.append(f"Topic mastery is high ({mastery:.2f}); allowing a more advanced treatment")

    relevant_quizzes = _relevant_records(profile.recent_quiz_results, topics)
    relevant_exams = _relevant_records(profile.recent_exam_results, topics)
    relevant_history = _relevant_records(profile.historical_performance, topics)
    performance_records = {
        "quizzes": relevant_quizzes,
        "exams": relevant_exams,
        "history": relevant_history,
    }
    recent_scores = _scores(performance_records)
    if relevant_quizzes or relevant_exams or relevant_history:
        guidance.append(
            "Use these matching-topic performance records to target practice and avoid repeating mastered points; "
            "treat them as learner signals, never as evidence for lesson facts: " + _compact(performance_records)
        )
        reasons.append("Recent or historical performance for the requested topic informs remediation")
    if recent_scores and min(recent_scores) < 0.5:
        guidance.append("Recent matching-topic scores show a gap; include a clear intermediate step before asking for independent application.")

    if profile.question_accuracy is not None:
        if profile.question_accuracy < 0.5:
            guidance.append(f"Overall question accuracy is low ({profile.question_accuracy:.2f}); add a brief checkable step and explain the key distinction.")
            reasons.append("Low overall question accuracy calls for more scaffolding")
        elif profile.question_accuracy >= 0.85:
            guidance.append(f"Overall question accuracy is high ({profile.question_accuracy:.2f}); reduce routine scaffolding and focus on the requested challenge.")

    requested_difficulty = requested.get("difficulty")
    if requested_difficulty:
        levels = ["easy", "medium", "hard"]
        index = levels.index(requested_difficulty) if requested_difficulty in levels else 1
        shifts = 0
        if mastery is not None:
            shifts += -1 if mastery < 0.5 else (1 if mastery >= 0.85 else 0)
        if profile.question_accuracy is not None and profile.question_accuracy < 0.5:
            shifts -= 1
        difficulty_score = next((value for name, value in profile.difficulty_performance.items()
                                 if name.casefold() == requested_difficulty.casefold()), None)
        if difficulty_score is not None:
            if difficulty_score < 0.5:
                shifts -= 1
                guidance.append(f"Performance at {requested_difficulty} difficulty is low ({difficulty_score:.2f}); retain the requested topic while adding more steps and hints.")
                reasons.append(f"Low performance at {requested_difficulty} difficulty adds scaffolding")
            elif difficulty_score >= 0.85:
                shifts += 1
                guidance.append(f"Performance at {requested_difficulty} difficulty is high ({difficulty_score:.2f}); reduce hints and use a supported stretch question where appropriate.")
                reasons.append(f"Strong performance at {requested_difficulty} difficulty permits less scaffolding")
        effective = levels[max(0, min(len(levels) - 1, index + max(-1, min(1, shifts))))]
        adapted["effective_difficulty"] = effective
        adapted["difficulty"] = effective
        if effective != requested_difficulty:
            reasons.append(f"Requested {requested_difficulty} difficulty is adapted to {effective} for the learner profile")

    if "weak_topic_emphasis" in requested:
        if requested["weak_topic_emphasis"]:
            if weak:
                guidance.append("For this exam, allocate more of the requested questions to matching weak topics, while keeping every answer supported by the evidence.")
            else:
                guidance.append("Weak-topic emphasis is enabled, but none of the requested exam topics matches the learner's weak-topic list; cover only the requested topics.")
        else:
            guidance.append("Weak-topic emphasis is disabled: distribute exam questions across the requested topics without using the learner's weak-topic list to bias selection.")

    misconceptions = _relevant_records(profile.misconception_history, topics)
    if misconceptions:
        guidance.append(
            "Address these matching-topic misconceptions only when the supplied evidence supports a correction; do not assume the learner still holds them: "
            + _compact(misconceptions)
        )
        reasons.append("Relevant misconception history is targeted for evidence-backed correction")

    if guidance:
        adapted["learner_guidance"] = guidance
    if weak and not weak_emphasis_disabled:
        adapted["matching_weak_topics"] = weak
    if strong:
        adapted["matching_strong_topics"] = strong
    if recommended:
        adapted["matching_revision_topics"] = recommended
    return AdaptationMetadata(requested_settings=requested, adapted_settings=adapted, adaptation_reasons=reasons)
