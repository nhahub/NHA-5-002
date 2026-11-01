from app.domain import GroundedContext, LearnerProfile, RetrievedChunk
from app.rag import InMemoryRetriever, ContextBuilder
from app.adaptation import adapt
from app.generation import LearningGenerator
from app.schemas import MCQRequest, LearnerProfileIn, ExplanationRequest
from app.prompting import PromptEngine

def test_context_deduplicates_and_limits():
    c = ContextBuilder(20).build([RetrievedChunk("abc", chunk_id="1", relevance_score=.8), RetrievedChunk("abc", chunk_id="2", relevance_score=.7)])
    assert c.text == "abc" and c.retrieval_status == "sufficient"
def test_adaptation_low_and_high_mastery():
    assert adapt("CNN", {"difficulty":"medium"}, LearnerProfile(topic_mastery={"CNN": .31})).adapted_settings["difficulty"] == "easy"
    assert adapt("CNN", {"difficulty":"medium"}, LearnerProfile(topic_mastery={"CNN": .91})).adapted_settings["difficulty"] == "hard"
    summary = adapt("CNN", {"style":"concise"}, LearnerProfile(topic_mastery={"CNN": .31}))
    assert summary.adapted_settings["style"] == "concise"
    assert any("mastery of the requested topic is low" in text.lower() for text in summary.adapted_settings["learner_guidance"])

def test_prompt_includes_recent_performance_fields():
    profile = LearnerProfile(
        recent_quiz_results=[{"CNN": {"score": .4}}],
        recent_exam_results=[{"CNN": {"score": .5}}],
        historical_performance={"CNN": {"attempts": 2}},
        question_accuracy=.4, difficulty_performance={"hard": .2},
    )
    adaptation = adapt("CNN", {"difficulty": "hard", "explanation_depth": "high"}, profile)
    prompt = PromptEngine().build("personalized explanation", "CNN", profile,
        GroundedContext([], "", "insufficient", "not_grounded"), {
            "request": {"difficulty": "hard", "explanation_depth": "high"},
            "adaptation": adaptation.model_dump(),
        })
    assert "matching-topic performance records" in prompt
    assert "quizzes" in prompt and "exams" in prompt and "history" in prompt
    assert "Overall question accuracy is low" in prompt
    assert "Performance at hard difficulty is low" in prompt
def test_grounding_and_insufficient_context():
    grounded = LearningGenerator(InMemoryRetriever([RetrievedChunk("CNN extracts local features", "notes.txt", 1, "x")])).mcq(MCQRequest(topic="CNN", number_of_questions=1))
    missing = LearningGenerator(InMemoryRetriever()).mcq(MCQRequest(topic="Quantum", number_of_questions=1))
    assert grounded.grounding.source_mode == "retrieved_learning_content" and grounded.questions == []
    assert grounded.grounding.retrieval_status == "sufficient"
    assert grounded.grounding.grounding_status == "grounded"
    assert grounded.grounding.failure_status == "generation_failed"
    assert grounded.grounding.sources
    assert missing.grounding.source_mode == "insufficient_context" and missing.questions == []
    assert missing.grounding.retrieval_status == "insufficient"
def test_schema_api_models():
    result = LearningGenerator(InMemoryRetriever()).explain(ExplanationRequest(topic="CNN"))
    assert result.model_dump()["grounding"]["retrieval_status"] == "insufficient"
