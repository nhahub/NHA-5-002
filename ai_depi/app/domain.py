from dataclasses import dataclass, field
from typing import Any, Protocol

# Internal types shared across retrieval, personalization, and provider adapters.
# They keep service boundaries typed without tying the core pipeline to FastAPI.
@dataclass
class LearnerProfile:
    user_id: str | None = None
    overall_level: str = "intermediate"
    topic_mastery: dict[str, float] = field(default_factory=dict)
    weak_topics: list[str] = field(default_factory=list)
    strong_topics: list[str] = field(default_factory=list)
    recent_quiz_results: list[dict[str, Any]] = field(default_factory=list)
    recent_exam_results: list[dict[str, Any]] = field(default_factory=list)
    historical_performance: dict[str, Any] = field(default_factory=dict)
    question_accuracy: float | None = None
    difficulty_performance: dict[str, float] = field(default_factory=dict)
    misconception_history: list[dict[str, Any]] = field(default_factory=list)
    recommended_revision_topics: list[str] = field(default_factory=list)
    def mastery_for(self, topic: str) -> float | None:
        key = topic.lower()
        for name, value in self.topic_mastery.items():
            if name.lower() == key or name.lower() in key or key in name.lower(): return value
        return None

class LLMProvider(Protocol):
    name: str
    model: str
    def generate(self, system_prompt: str, user_prompt: str, response_schema: dict | None = None, *, temperature: float = 0.2, max_tokens: int | None = None) -> str: ...

class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
class WebSearchProvider(Protocol):
    def search(self, query: str) -> list[dict[str, Any]]: ...

@dataclass
class RetrievedChunk:
    """One retrieved text segment with provenance and its normalized relevance score."""
    content: str
    document_name: str | None = None
    page_number: int | None = None
    chunk_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    relevance_score: float | None = None

@dataclass
class GroundedContext:
    """Bounded prompt context plus separate retrieval and evidence-grounding states."""
    chunks: list[RetrievedChunk]
    text: str
    retrieval_status: str
    grounding_status: str
    retrieval_method: str | None = None
    failure_status: str | None = None
