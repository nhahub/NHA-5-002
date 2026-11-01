import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.adaptation import adapt
from app.config import Settings
from app.domain import GroundedContext, LearnerProfile, RetrievedChunk
from app.generation import LearningGenerator
from app.prompting import PromptEngine
from app.evidence import task_evidence
from app.rag import ContextBuilder, HybridCrossEncoderRetriever, Retriever, reciprocal_rank_fusion
from app.schemas import AnswerEvaluationRequest, ExplanationRequest, LearnerProfileIn, MCQRequest


PASSAGE = "The Nile inundation supported Egyptian agriculture because it supplied water during the growing season."


class CapturingRetriever(Retriever):
    method = "test"

    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def retrieve(self, query, filters=None, top_k=5):
        self.calls.append((query, filters, top_k))
        return self.chunks[:top_k]


class EvidenceProvider:
    name, model = "groq-test", "parameter-sensitivity"

    def __init__(self, response=None):
        self.response = response or {
            "explanation": PASSAGE,
            "key_concepts": [], "examples": [], "important_points": [],
            "common_mistakes": [], "recommended_revision": [],
        }
        self.prompts = []
        self.max_tokens = []

    def generate(self, system_prompt, user_prompt="", response_schema=None, **kwargs):
        self.prompts.append(user_prompt)
        self.max_tokens.append(kwargs.get("max_tokens"))
        return json.dumps(self.response)


class AcceptingJudge:
    name, model = "ollama", "qwen2.5:7b-instruct-test"

    def generate(self, system_prompt, user_prompt="", response_schema=None, **kwargs):
        return json.dumps({"supported": True, "confidence": .96, "reason": "Supported by the test evidence."})


def source_chunks(count=12):
    return [RetrievedChunk(
        f"{PASSAGE} Section {index} describes this supported relationship.",
        "nile.pdf", index, f"nile-{index}", {"source_filename": "nile.pdf", "page": index}, .95,
    ) for index in range(1, count + 1)]


def _prompt(task, request, profile=None, *, retrieval_status="sufficient"):
    profile = profile or LearnerProfile()
    topic = request.get("topic", "Nile agriculture")
    requested = {key: value for key, value in request.items() if key in {
        "difficulty", "explanation_depth", "style", "question_style", "requested_concepts",
        "weak_topic_emphasis", "time_limit_minutes",
    }}
    adaptation = adapt(topic, requested, profile)
    context = GroundedContext([], "", retrieval_status, "grounded" if retrieval_status == "sufficient" else "not_grounded")
    return PromptEngine().build(task, topic, profile, context, {
        "request": request,
        "adaptation": adaptation.model_dump(),
        "evidence_status": {"retrieval_status": retrieval_status},
    })


def test_difficulty_and_depth_are_independent_behavioral_controls():
    prompts = {}
    for difficulty in ("easy", "medium", "hard"):
        for depth in ("brief", "standard", "high"):
            request = {"topic": "Nile agriculture", "difficulty": difficulty,
                       "explanation_depth": depth, "knowledge_mode": "learning_materials_only"}
            prompts[difficulty, depth] = _prompt("personalized explanation", request)

    assert len(set(prompts.values())) == 9
    assert "effective difficulty after learner adaptation (easy)" in prompts["easy", "brief"]
    assert "one simple reasoning step" in prompts["easy", "brief"]
    assert "synthesis, comparison, or multiple reasoning steps" in prompts["hard", "brief"]
    assert "Give a deeper, organized explanation" in prompts["easy", "high"]
    assert "Keep the explanation compact" in prompts["hard", "brief"]
    assert "never add assumptions or facts" in prompts["hard", "high"]


def test_summary_mcq_and_exam_parameters_change_task_instructions():
    concise = _prompt("grounded summary", {"topic": "Nile agriculture", "style": "concise"})
    detailed = _prompt("grounded summary", {"topic": "Nile agriculture", "style": "detailed"})
    assert "central supported facts" in concise
    assert "each distinct supported idea" in detailed

    recall = _prompt("MCQ set generation", {"topic": "Nile agriculture", "question_style": "recall",
                                             "requested_concepts": ["annual inundation"]})
    application = _prompt("MCQ set generation", {"topic": "Nile agriculture", "question_style": "application",
                                                  "requested_concepts": ["agriculture"]})
    assert "direct questions" in recall and "annual inundation" in recall
    assert "apply a rule or relationship" in application and "agriculture" in application

    weak = _prompt("practice exam generation", {"topic": "Nile agriculture", "difficulty": "medium",
                                                  "weak_topic_emphasis": True, "time_limit_minutes": 20},
                   LearnerProfile(weak_topics=["Nile agriculture"]))
    balanced = _prompt("practice exam generation", {"topic": "Nile agriculture", "difficulty": "medium",
                                                      "weak_topic_emphasis": False, "time_limit_minutes": 60},
                       LearnerProfile(weak_topics=["Nile agriculture"]))
    assert "allocate more of the requested questions" in weak
    assert "without using the learner's weak-topic list" in balanced
    assert "do not use that profile signal to change question distribution" in balanced
    assert "Treat these matching topics as revision priorities" not in balanced
    assert "matching_weak_topics" not in balanced
    assert "20-minute time limit" in weak and "60-minute time limit" in balanced


def test_each_learner_signal_has_topic_scoped_guidance_and_user_id_is_metadata():
    beginner = LearnerProfile(
        user_id="do-not-send-this-id", overall_level="beginner",
        topic_mastery={"Nile agriculture": .2}, weak_topics=["Nile agriculture"],
        recent_quiz_results=[{"topic": "Nile agriculture", "score": .3}],
        recent_exam_results=[{"topic": "Nile agriculture", "accuracy": .4}],
        historical_performance={"Nile agriculture": {"attempts": 4, "accuracy": .4}},
        question_accuracy=.4, difficulty_performance={"hard": .2},
        misconception_history=[{"topic": "Nile agriculture", "misconception": "The Nile flooded every day."}],
        recommended_revision_topics=["Nile agriculture"],
    )
    beginner_prompt = _prompt("personalized explanation", {
        "topic": "Nile agriculture", "difficulty": "hard", "explanation_depth": "standard",
    }, beginner)
    assert "beginner-friendly vocabulary" in beginner_prompt
    assert "Mastery of the requested topic is low" in beginner_prompt
    assert "matching-topic performance records" in beginner_prompt
    assert "Overall question accuracy is low" in beginner_prompt
    assert "Performance at hard difficulty is low" in beginner_prompt
    assert "matching-topic misconceptions" in beginner_prompt
    assert "revision recommendations" in beginner_prompt
    assert "do-not-send-this-id" not in beginner_prompt

    advanced = LearnerProfile(overall_level="advanced", topic_mastery={"Nile agriculture": .95},
                              strong_topics=["Nile agriculture"], question_accuracy=.95,
                              difficulty_performance={"hard": .95})
    advanced_prompt = _prompt("personalized explanation", {
        "topic": "Nile agriculture", "difficulty": "medium", "explanation_depth": "standard",
    }, advanced)
    assert "appropriately advanced terminology" in advanced_prompt
    assert "already lists these requested topics as strengths" in advanced_prompt
    assert "Mastery of the requested topic is high" in advanced_prompt
    assert "Overall question accuracy is high" in advanced_prompt
    assert beginner_prompt != advanced_prompt
    beginner_high = _prompt("personalized explanation", {
        "topic": "Nile agriculture", "difficulty": "hard", "explanation_depth": "high",
    }, beginner)
    advanced_high = _prompt("personalized explanation", {
        "topic": "Nile agriculture", "difficulty": "hard", "explanation_depth": "high",
    }, advanced)
    assert "Give a deeper, organized explanation" in beginner_high
    assert "Give a deeper, organized explanation" in advanced_high
    assert beginner_high != advanced_high


def test_knowledge_mode_controls_grounding_and_general_fallback():
    materials_only = _prompt("personalized explanation", {
        "topic": "quantum teleportation", "knowledge_mode": "learning_materials_only",
    }, retrieval_status="insufficient")
    fallback = _prompt("personalized explanation", {
        "topic": "quantum teleportation", "knowledge_mode": "learning_materials_then_general_knowledge",
    }, retrieval_status="insufficient")
    fallback_with_evidence = _prompt("personalized explanation", {
        "topic": "Nile agriculture", "knowledge_mode": "learning_materials_then_general_knowledge",
    }, retrieval_status="sufficient")
    assert "EVIDENCE-BOUNDED POLICY" in materials_only
    assert "GENERAL-KNOWLEDGE FALLBACK POLICY" in fallback
    assert "EVIDENCE-BOUNDED POLICY" in fallback_with_evidence

    empty_provider = EvidenceProvider({"explanation": "A general explanation of the requested subject.",
        "key_concepts": [], "examples": [], "important_points": [], "common_mistakes": [], "recommended_revision": []})
    generator = LearningGenerator(CapturingRetriever([]), llm=empty_provider,
                                  settings=Settings(_env_file=None, max_retries=0), top_k=5,
                                  claim_judge_llm=AcceptingJudge())
    result = generator.explain(ExplanationRequest(
        topic="quantum teleportation", knowledge_mode="learning_materials_then_general_knowledge"))
    assert result.grounding.source_mode == "general_knowledge"
    assert result.grounding.retrieval_status == "insufficient"
    assert result.generation is not None


@pytest.mark.parametrize("top_k", [3, 5, 10])
def test_swagger_top_k_reaches_retrieval_and_bounds_evidence(top_k):
    retriever = CapturingRetriever(source_chunks())
    provider = EvidenceProvider()
    generator = LearningGenerator(retriever, llm=provider,
                                  settings=Settings(_env_file=None, max_retries=0), top_k=5,
                                  claim_judge_llm=AcceptingJudge())
    result = generator.explain(ExplanationRequest(
        topic="Nile agriculture", question="Why did the Nile inundation matter?", top_k=top_k))
    assert result.grounding.retrieval_status == "sufficient"
    assert retriever.calls[0][2] == top_k
    assert "Topic: Nile agriculture" in retriever.calls[0][0]
    evidence_ids = [item for item in provider.prompts[0].split("[nile-") if "| nile.pdf" in item]
    assert len(evidence_ids) == top_k


class CountAwareMCQProvider:
    name, model = "groq-test", "count-budget"

    def __init__(self): self.counts, self.token_budgets = [], []

    def generate(self, system_prompt, user_prompt="", response_schema=None, **kwargs):
        count = response_schema["properties"]["questions"]["minItems"]
        self.counts.append(count)
        self.token_budgets.append(kwargs["max_tokens"])
        item = {
            "question": "What did the annual inundation support?",
            "options": ["Egyptian agriculture", "An unrelated subject"],
            "correct_answer": "Egyptian agriculture",
            "explanation": PASSAGE,
            "difficulty": "medium", "topic": "Nile agriculture",
            "learning_objective": "Identify the supported effect of inundation",
        }
        return json.dumps({"questions": [item.copy() for _ in range(count)]})


def test_requested_mcq_count_controls_exact_items_and_output_budget():
    provider = CountAwareMCQProvider()
    generator = LearningGenerator(CapturingRetriever(source_chunks(5)), llm=provider,
                                  settings=Settings(_env_file=None, max_retries=0), top_k=5,
                                  min_relevance_score=.1, claim_judge_llm=AcceptingJudge())
    small = generator.mcq(MCQRequest(topic="Nile agriculture", number_of_questions=5))
    larger = generator.mcq(MCQRequest(topic="Nile agriculture", number_of_questions=20))
    assert len(small.questions) == 5 and len(larger.questions) == 20
    assert provider.counts == [5, 20]
    assert provider.token_budgets == [3000, 3600]


def test_context_character_limit_bounds_chunks_actually_sent_to_prompt():
    chunks = [RetrievedChunk("A" * 50, chunk_id="a", relevance_score=.9),
              RetrievedChunk("B" * 50, chunk_id="b", relevance_score=.9)]
    context = ContextBuilder(max_chars=60, min_relevance_score=.1).build(chunks)
    assert len(context.text) <= 60
    assert sum(len(chunk.content) for chunk in context.chunks) <= 60


def test_exam_topic_list_keeps_evidence_for_each_separately_requested_topic():
    nile = RetrievedChunk("The Nile River supports Egyptian agriculture through annual inundation.", chunk_id="nile", relevance_score=.9)
    revolution = RetrievedChunk("The French Revolution began in 1789 and changed French government.", chunk_id="revolution", relevance_score=.9)
    unrelated = RetrievedChunk("Copper conducts heat efficiently because free electrons transfer energy.", chunk_id="copper", relevance_score=.9)
    assert [chunk.chunk_id for chunk in task_evidence("Nile River\nFrench Revolution", [nile, revolution, unrelated], "exam")] == ["nile", "revolution"]


def test_answer_grading_uses_expected_answer_and_gives_partial_credit():
    grade = LearningGenerator._grade_answer
    assert grade("Liberty and equality", "Liberty, equality, and political rights") == (0.67, False)
    assert grade("Liberty, equality, and political rights", "Liberty, equality, and political rights") == (1.0, True)
    assert grade("The Bastille", "Liberty, equality, and political rights") == (0.0, False)
    assert grade("anything", None) is None


class EvaluationProvider:
    name, model = "groq-test", "evaluation-sensitivity"

    def __init__(self): self.prompts = []

    def generate(self, system_prompt, user_prompt="", response_schema=None, **kwargs):
        self.prompts.append(user_prompt)
        return json.dumps({
            "correct": False, "score": 0.0,
            "correct_answer": "provider-invented answer",
            "explanation": "The Declaration established liberty, equality, and political rights.",
            "why_answer_is_wrong": None, "related_concept": "French Revolution",
            "recommended_revision": [], "confidence": .9,
            "identified_misconception": None,
        })


def test_student_answer_and_learner_profile_interact_in_evaluation():
    expected = "Liberty, equality, and political rights"
    evidence = RetrievedChunk(
        "The Declaration established liberty, equality, and political rights.",
        "history.pdf", 1, "declaration", {"source_filename": "history.pdf", "page": 1}, .95,
    )
    retriever = CapturingRetriever([evidence])
    provider = EvaluationProvider()
    generator = LearningGenerator(retriever, llm=provider,
                                  settings=Settings(_env_file=None, max_retries=0), top_k=5,
                                  min_relevance_score=.1, claim_judge_llm=AcceptingJudge())
    request_values = {
        "question": "What principles did the Declaration establish?",
        "correct_answer": expected,
        "topic": "French Revolution",
        "knowledge_mode": "learning_materials_only",
        "learner_profile": {"weak_topics": ["French Revolution"],
                            "recommended_revision_topics": ["French Revolution"]},
    }
    full = generator.evaluate(AnswerEvaluationRequest(**request_values, student_answer=expected))
    partial = generator.evaluate(AnswerEvaluationRequest(**request_values, student_answer="Liberty and equality"))
    wrong = generator.evaluate(AnswerEvaluationRequest(**request_values, student_answer="The Bastille"))
    assert full.correct is True and full.score == 1.0
    assert partial.correct is False and partial.score == 0.67
    assert partial.why_answer_is_wrong != wrong.why_answer_is_wrong
    assert partial.identified_misconception is None
    assert wrong.correct is False and wrong.score == 0.0
    assert wrong.identified_misconception is None
    assert "The Bastille" in provider.prompts[-1]
    assert "revision recommendations" in provider.prompts[-1]
    assert wrong.recommended_revision[0] == "French Revolution"


def test_request_validation_and_runtime_settings_have_safety_bounds():
    with pytest.raises(ValidationError):
        ExplanationRequest(topic="Nile", top_k=0)
    with pytest.raises(ValidationError):
        ExplanationRequest(topic="Nile", difficulty="extreme")
    with pytest.raises(ValidationError):
        LearnerProfileIn(topic_mastery={"Nile": 1.1})
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_chars=100, chunk_overlap_chars=100)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, cross_encoder_device="quantum-device")


def test_configured_input_character_limit_is_enforced_before_endpoint_parsing():
    from app.main import RequestCharacterLimitMiddleware

    api = FastAPI()

    @api.post("/learning/echo")
    def echo(payload: dict):
        return payload

    client = TestClient(RequestCharacterLimitMiddleware(api, max_chars=20))
    assert client.post("/learning/echo", json={"x": "short"}).status_code == 200
    oversized = client.post("/learning/echo", content='{"text":"' + "x" * 30 + '"}',
                            headers={"content-type": "application/json"})
    assert oversized.status_code == 413


def test_rrf_k_changes_scores_and_can_change_rank_order():
    dense = [RetrievedChunk("A", chunk_id="a"), RetrievedChunk("B", chunk_id="b"),
             RetrievedChunk("x", chunk_id="x"), RetrievedChunk("y", chunk_id="y")]
    sparse = [RetrievedChunk("z", chunk_id="z"), RetrievedChunk("B", chunk_id="b"),
              RetrievedChunk("w", chunk_id="w"), RetrievedChunk("A", chunk_id="a")]
    low = reciprocal_rank_fusion(dense, sparse, rrf_k=1, limit=4)
    high = reciprocal_rank_fusion(dense, sparse, rrf_k=60, limit=4)
    assert low[0].chunk_id == "a"
    assert high[0].chunk_id == "b"
    low_scores = {row.chunk_id: row.metadata["hybrid_score"] for row in low}
    high_scores = {row.chunk_id: row.metadata["hybrid_score"] for row in high}
    assert low_scores["a"] != high_scores["a"]


class DenseRows:
    def __init__(self, rows):
        self.rows, self.collection, self.embedding = rows, type("C", (), {"get": lambda self, include: {"ids": [], "documents": [], "metadatas": []}})(), object()

    def retrieve(self, query, filters=None, top_k=5):
        return self.rows[:top_k]


class RecordingScorer:
    def __init__(self, score=0.0): self.score, self.calls = score, []

    def predict(self, pairs, batch_size, show_progress_bar):
        self.calls.append((pairs, batch_size))
        return [self.score] * len(pairs)


@pytest.mark.parametrize("top_k", [3, 5, 10])
def test_hybrid_final_top_k_is_respected_even_above_default_rerank_pool(top_k):
    rows = [RetrievedChunk(f"retrieval query result {index}", "notes", 1, str(index),
                           {"embedding_version": "v1"}, .9) for index in range(12)]
    dense = DenseRows(rows)
    scorer = RecordingScorer()
    settings = Settings(_env_file=None, dense_candidate_k=12, sparse_candidate_k=12,
                         rerank_candidate_k=4, top_k=5, cross_encoder_batch_size=3)
    pipeline = HybridCrossEncoderRetriever(dense, settings, cross_encoder=scorer)
    result = pipeline.retrieve("retrieval query", top_k=top_k)
    assert len(result) == top_k
    assert len(scorer.calls[0][0]) >= top_k
    assert scorer.calls[0][1] == 3


def test_cross_encoder_offset_changes_threshold_eligibility():
    chunk = RetrievedChunk("retrieval query result", "notes", 1, "one", {"embedding_version": "v1"}, .9)
    settings = Settings(_env_file=None, dense_candidate_k=1, sparse_candidate_k=1,
                        rerank_candidate_k=1, cross_encoder_score_offset=5)
    high = HybridCrossEncoderRetriever(DenseRows([chunk]), settings,
                                       cross_encoder=RecordingScorer(score=-4.5)).retrieve("retrieval query")[0]
    settings_low = Settings(_env_file=None, dense_candidate_k=1, sparse_candidate_k=1,
                            rerank_candidate_k=1, cross_encoder_score_offset=3)
    low = HybridCrossEncoderRetriever(DenseRows([chunk]), settings_low,
                                      cross_encoder=RecordingScorer(score=-4.5)).retrieve("retrieval query")[0]
    assert high.relevance_score > low.relevance_score
    assert high.relevance_score > .35 > low.relevance_score
