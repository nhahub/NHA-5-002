from typing import Annotated, Any, Literal
from pydantic import BaseModel, Field, ConfigDict, StrictBool, StrictFloat, StrictStr

# Pydantic models define both the public request/response contract and the
# structured fields checked after LLM generation.
class LearnerProfileIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    user_id: str | None = Field(default=None, max_length=200)
    overall_level: str = Field(default="intermediate", min_length=1, max_length=50)
    topic_mastery: dict[str, Annotated[float, Field(ge=0.0, le=1.0)]] = Field(default_factory=dict)
    weak_topics: list[str] = Field(default_factory=list)
    strong_topics: list[str] = Field(default_factory=list)
    recent_quiz_results: list[dict[str, Any]] = Field(default_factory=list)
    recent_exam_results: list[dict[str, Any]] = Field(default_factory=list)
    historical_performance: dict[str, Any] = Field(default_factory=dict)
    question_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    difficulty_performance: dict[str, Annotated[float, Field(ge=0.0, le=1.0)]] = Field(default_factory=dict)
    misconception_history: list[dict[str, Any]] = Field(default_factory=list)
    recommended_revision_topics: list[str] = Field(default_factory=list)

class LearningSource(BaseModel):
    document_name: str | None = None
    page_number: int | None = None
    chunk_id: str | None = None
    snippet: str | None = None
    relevance_score: float | None = None

class GroundingMetadata(BaseModel):
    source_mode: Literal["retrieved_learning_content", "general_knowledge", "insufficient_context"]
    grounding_status: Literal["grounded", "partially_grounded", "not_grounded"]
    retrieval_status: Literal["sufficient", "insufficient"]
    failure_status: Literal[
        "retrieval_insufficient", "evidence_insufficient", "generation_failed",
        "output_validation_failed", "grounding_validation_failed",
    ] | None = None
    retrieval_method: str | None = None
    sources: list[LearningSource] = Field(default_factory=list)
    notice: str | None = None

class AdaptationMetadata(BaseModel):
    requested_settings: dict[str, Any] = Field(default_factory=dict)
    adapted_settings: dict[str, Any] = Field(default_factory=dict)
    adaptation_reasons: list[str] = Field(default_factory=list)

class GenerationMetadata(BaseModel):
    provider: str
    model: str
    fallback_used: bool = False

class ClaimValidationResult(BaseModel):
    """Internal, strict result from the semantic evidence judge."""
    model_config = ConfigDict(extra="forbid")
    supported: StrictBool
    confidence: Annotated[StrictFloat, Field(ge=0, le=1)]
    reason: StrictStr = Field(min_length=1, max_length=500)

class RetrievalRequest(BaseModel):
    # None preserves the configured production default; an explicit value is
    # the request's final evidence count, independently of candidate-pool sizes.
    top_k: int | None = Field(default=None, ge=1, le=20)

class ExplanationRequest(RetrievalRequest):
    topic: str = Field(min_length=1, max_length=300)
    question: str | None = Field(default=None, max_length=5000)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    explanation_depth: Literal["brief", "standard", "high"] = "standard"
    knowledge_mode: Literal["learning_materials_only", "learning_materials_then_general_knowledge"] = "learning_materials_only"
    learner_profile: LearnerProfileIn = Field(default_factory=LearnerProfileIn)
class ExplanationResponse(BaseModel):
    explanation: str
    key_concepts: list[str]
    examples: list[str]
    important_points: list[str]
    common_mistakes: list[str]
    recommended_revision: list[str]
    grounding: GroundingMetadata
    adaptation: AdaptationMetadata
    generation: GenerationMetadata | None = None

class SummaryRequest(RetrievalRequest):
    topic: str = Field(min_length=1, max_length=300)
    style: Literal["concise", "exam-focused", "detailed", "bullet-point", "revision-sheet"] = "concise"
    knowledge_mode: Literal["learning_materials_only", "learning_materials_then_general_knowledge"] = "learning_materials_only"
    learner_profile: LearnerProfileIn = Field(default_factory=LearnerProfileIn)
class SummaryResponse(BaseModel):
    summary: str
    key_points: list[str]
    grounding: GroundingMetadata
    adaptation: AdaptationMetadata
    generation: GenerationMetadata | None = None

class MCQ(BaseModel):
    question: str
    options: list[str] = Field(min_length=2)
    correct_answer: str
    explanation: str
    difficulty: str
    topic: str
    learning_objective: str
    source: LearningSource | None = None
class MCQRequest(RetrievalRequest):
    topic: str = Field(min_length=1, max_length=300)
    number_of_questions: int = Field(default=5, ge=1, le=50)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    question_style: str = Field(default="conceptual", min_length=1, max_length=100)
    requested_concepts: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(default_factory=list)
    knowledge_mode: Literal["learning_materials_only", "learning_materials_then_general_knowledge"] = "learning_materials_only"
    learner_profile: LearnerProfileIn = Field(default_factory=LearnerProfileIn)
class MCQSet(BaseModel):
    questions: list[MCQ]
    grounding: GroundingMetadata
    adaptation: AdaptationMetadata
    generation: GenerationMetadata | None = None

class Flashcard(BaseModel):
    front: str
    back: str
    topic: str
    difficulty: str
    concept: str
    source: LearningSource | None = None
class FlashcardRequest(RetrievalRequest):
    topic: str = Field(min_length=1, max_length=300)
    number_of_cards: int = Field(default=10, ge=1, le=50)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    knowledge_mode: Literal["learning_materials_only", "learning_materials_then_general_knowledge"] = "learning_materials_only"
    learner_profile: LearnerProfileIn = Field(default_factory=LearnerProfileIn)
class FlashcardSet(BaseModel):
    cards: list[Flashcard]
    grounding: GroundingMetadata
    adaptation: AdaptationMetadata
    generation: GenerationMetadata | None = None

class PracticeExamRequest(RetrievalRequest):
    topics: list[Annotated[str, Field(min_length=1, max_length=300)]] = Field(min_length=1, max_length=50)
    number_of_questions: int = Field(default=10, ge=1, le=100)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    time_limit_minutes: int | None = Field(default=None, ge=1, le=600)
    weak_topic_emphasis: bool = True
    knowledge_mode: Literal["learning_materials_only", "learning_materials_then_general_knowledge"] = "learning_materials_only"
    learner_profile: LearnerProfileIn = Field(default_factory=LearnerProfileIn)
class PracticeExam(BaseModel):
    title: str
    questions: list[MCQ]
    time_limit_minutes: int | None
    grounding: GroundingMetadata
    adaptation: AdaptationMetadata
    generation: GenerationMetadata | None = None

class AnswerEvaluationRequest(RetrievalRequest):
    question: str = Field(min_length=1, max_length=5000)
    student_answer: str = Field(min_length=1, max_length=10000)
    correct_answer: str | None = Field(default=None, max_length=10000)
    topic: str = Field(default="general", min_length=1, max_length=300)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    knowledge_mode: Literal["learning_materials_only", "learning_materials_then_general_knowledge"] = "learning_materials_only"
    learner_profile: LearnerProfileIn = Field(default_factory=LearnerProfileIn)
class AnswerEvaluation(BaseModel):
    correct: bool
    score: float = Field(ge=0, le=1)
    correct_answer: str | None
    explanation: str
    why_answer_is_wrong: str | None
    related_concept: str
    recommended_revision: list[str]
    confidence: float = Field(ge=0, le=1)
    identified_misconception: str | None
    supporting_source_or_evidence: list[LearningSource]
    groundedness: Literal["grounded", "partially_grounded", "not_grounded"]
    topic: str
    difficulty: str
    grounding: GroundingMetadata
    generation: GenerationMetadata | None = None
