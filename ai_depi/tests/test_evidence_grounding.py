import json

import pytest

from app.config import Settings
from app.domain import GroundedContext, RetrievedChunk
from app.evidence import source_for_claim, task_evidence
from app.generation import LearningGenerator
from app.rag import Retriever
from app.schemas import (
    AnswerEvaluationRequest, ExplanationRequest, FlashcardRequest, MCQRequest, SummaryRequest,
    PracticeExamRequest,
)


ESTATES = RetrievedChunk(
    "The French Revolution began in 1789. The Estates-General had three estates: clergy, nobility, and the Third Estate.",
    "history.pdf", 1, "estates", {"source_filename": "history.pdf", "source_path": "sample_data/history.pdf"}, .95,
)
BASTILLE = RetrievedChunk(
    "During the French Revolution, revolutionaries stormed the Bastille on 14 July 1789; the event became an important symbol of the Revolution.",
    "history.pdf", 2, "bastille", {"source_filename": "history.pdf", "source_path": "sample_data/history.pdf"}, .9,
)
DECLARATION = RetrievedChunk(
    "The Declaration of the Rights of Man established principles of liberty, equality, and political rights during the French Revolution.",
    "history.pdf", 3, "declaration", {"source_filename": "history.pdf", "source_path": "sample_data/history.pdf"}, .85,
)


class FixedRetriever(Retriever):
    method = "test"
    def __init__(self, chunks): self.chunks = chunks
    def retrieve(self, query, filters=None, top_k=5): return self.chunks[:top_k]


class PayloadProvider:
    name, model = "groq", "test-model"
    def __init__(self, payload): self.payload, self.calls, self.prompts = payload, 0, []
    def generate(self, system_prompt, user_prompt="", response_schema=None, **kwargs):
        self.calls += 1; self.prompts.append(user_prompt)
        return json.dumps(self.payload)


class ScriptedJudge:
    name, model = "ollama", "qwen2.5:7b-instruct-test"

    def __init__(self, decisions=None):
        self.decisions = list(decisions or [])
        self.prompts = []

    def generate(self, system_prompt, user_prompt="", response_schema=None, **kwargs):
        self.prompts.append(user_prompt)
        decision = self.decisions.pop(0) if self.decisions else True
        if isinstance(decision, dict):
            return json.dumps(decision)
        return json.dumps({"supported": decision, "confidence": .96,
                           "reason": "Scripted test judgment."})


def generator(chunks, provider=None, judge=None):
    return LearningGenerator(FixedRetriever(chunks), llm=provider,
                            settings=Settings(_env_file=None, max_retries=0), min_relevance_score=.1,
                            claim_judge_llm=judge or ScriptedJudge())


def explanation_payload(text):
    return {"explanation": text, "key_concepts": [], "examples": [], "important_points": [],
            "common_mistakes": [], "recommended_revision": [], "grounding": {}, "adaptation": {}}


def test_estates_fact_is_sufficient_and_source_contains_actual_support():
    provider = PayloadProvider(explanation_payload("The Estates-General had three estates: clergy, nobility, and the Third Estate."))
    response = generator([ESTATES], provider).explain(ExplanationRequest(topic="Estates-General", question="What were the three estates?"))
    assert response.grounding.retrieval_status == "sufficient"
    assert response.grounding.sources and "clergy, nobility" in response.grounding.sources[0].snippet


@pytest.mark.parametrize("claim", [
    "Louis XVI called the Estates-General because of the financial crisis.",
    "The Declaration rejected the Divine Right of Kings.",
])
def test_unsupported_claim_retries_once_then_fails_closed(claim):
    chunks = [ESTATES, DECLARATION]
    provider = PayloadProvider(explanation_payload(claim))
    response = generator(chunks, provider, ScriptedJudge([False, False])).explain(ExplanationRequest(topic="French Revolution", question="Explain the French Revolution."))
    assert provider.calls == 2
    assert "STRICT RETRY" in provider.prompts[-1]
    assert response.grounding.retrieval_status == "sufficient"
    assert response.grounding.source_mode == "retrieved_learning_content"
    assert response.grounding.grounding_status == "grounded"
    assert response.grounding.failure_status == "grounding_validation_failed"
    assert response.grounding.sources
    assert "could not be generated" in response.explanation
    assert response.generation is None
    assert not response.key_concepts and not response.examples


def _nile_regression_chunks():
    page16 = RetrievedChunk(
        "Ancient Egypt can be thought of as an oasis in the desert of northeastern Africa, "
        "dependent on the annual inundation of the Nile River to support its agricultural population. "
        "The country's chief wealth came from the fertile floodplain of the Nile valley.",
        "Ancient_Egypt_From.pdf", 16, "nile-p16",
        {"source_filename": "Ancient_Egypt_From.pdf", "page": 16}, .9956,
    )
    page17 = RetrievedChunk(
        "Agriculture centred on the cultivation of cereal crops, chiefly emmer wheat and barley. "
        "The fertility of the land and general predictability of the inundation ensured very high "
        "productivity from a single annual crop.",
        "Ancient_Egypt_From.pdf", 17, "nile-p17",
        {"source_filename": "Ancient_Egypt_From.pdf", "page": 17}, .9953,
    )
    page24 = RetrievedChunk(
        "Surveying was used for the regular division of fields after the annual inundation of the Nile.",
        "Ancient_Egypt_From.pdf", 24, "nile-p24",
        {"source_filename": "Ancient_Egypt_From.pdf", "page": 24}, .9865,
    )
    return [page16, page17, page24]


def test_exact_nile_agriculture_why_request_keeps_page16_and_page17_context():
    question = "Why did the annual inundation of the Nile matter to agriculture?"
    chunks = _nile_regression_chunks()
    service = generator(chunks)
    context = service._context(
        "Topic: Ancient Egyptian agriculture\nQuestion: " + question,
        "explain", evidence_query=question, top_k=3,
    )

    assert context.retrieval_status == "sufficient"
    assert [chunk.page_number for chunk in context.chunks] == [16, 17]
    assert "annual inundation" in context.chunks[0].content
    assert "ensured very high productivity" in context.chunks[1].content


def test_ensured_productivity_is_accepted_as_causal_evidence():
    passage = (
        "The fertility of the land and general predictability of the inundation ensured very high "
        "productivity from a single annual crop."
    )
    chunk = RetrievedChunk(passage, "Ancient_Egypt_From.pdf", chunk_id="ensured-productivity", relevance_score=.95)
    query = "Why did the annual inundation of the Nile matter to agriculture?"

    assert task_evidence(query, [chunk], "explain") == [chunk]


def test_semantic_judge_accepts_faithful_nile_paraphrase_and_rejects_military_claim():
    evidence = RetrievedChunk(
        "Ancient Egypt was dependent on the annual inundation of the Nile River to support its agricultural population.",
        "Ancient_Egypt_From.pdf", 16, "nile", {"page": 16}, .95,
    )
    faithful = "The Nile was important because its annual inundation supported the agricultural population."
    unsupported = "The Nile was important because it allowed the Egyptian army to conquer neighboring countries."

    context = GroundedContext([evidence], evidence.content, "sufficient", "grounded")
    service = generator([evidence], judge=ScriptedJudge([True, False]))
    validation = service._validate_claim_batch(
        [faithful, unsupported], "Ancient Egyptian agriculture",
        "Why did the annual inundation matter?", context,
    )

    assert len(service.claim_judge_llm.prompts) == 2
    assert "annual inundation" in service.claim_judge_llm.prompts[0]
    assert validation.decisions[service._claim_key(faithful)].accepted
    assert not validation.decisions[service._claim_key(unsupported)].accepted


def test_supported_nile_explanation_is_grounded_and_cited():
    answer = (
        "The annual inundation of the Nile supported agriculture by providing conditions that allowed "
        "the agricultural population to farm productively."
    )
    provider = PayloadProvider(explanation_payload(answer))
    service = generator(_nile_regression_chunks(), provider)
    response = service.explain(ExplanationRequest(
        topic="Ancient Egyptian agriculture",
        question="Why did the annual inundation of the Nile matter to agriculture?",
        difficulty="easy", explanation_depth="brief", top_k=3,
        learner_profile={"overall_level": "beginner", "topic_mastery": {"Ancient Egyptian agriculture": .25}},
    ))

    assert provider.calls == 1
    assert "one complete sentence copied verbatim" in provider.prompts[0]
    assert response.explanation == answer
    assert response.grounding.retrieval_method == "test"
    assert response.grounding.retrieval_status == "sufficient"
    assert response.grounding.grounding_status == "grounded"
    assert response.grounding.source_mode == "retrieved_learning_content"
    assert any(source.page_number == 16 for source in response.grounding.sources)
    assert response.grounding.failure_status is None
    assert response.generation is not None


def test_repeated_unsupported_nile_explanation_preserves_retrieval_state():
    unsupported = "The Nile was important because it allowed the Egyptian army to conquer neighboring countries."
    provider = PayloadProvider(explanation_payload(unsupported))
    response = generator(_nile_regression_chunks(), provider, ScriptedJudge([False, False])).explain(ExplanationRequest(
        topic="Ancient Egyptian agriculture",
        question="Why did the annual inundation of the Nile matter to agriculture?",
        top_k=3,
    ))

    assert provider.calls == 2
    assert response.explanation == "The uploaded learning materials contain relevant evidence, but a grounded response could not be generated."
    assert response.grounding.retrieval_status == "sufficient"
    assert response.grounding.grounding_status == "grounded"
    assert response.grounding.source_mode == "retrieved_learning_content"
    assert response.grounding.sources
    assert response.grounding.failure_status == "grounding_validation_failed"
    assert "couldn't find enough information" not in response.explanation
    assert response.generation is None


def test_mixed_explanation_keeps_supported_sentence_and_drops_unsupported_one():
    source = RetrievedChunk(
        "Ancient Egyptian civilization depended on the annual inundation of the Nile River to support its agricultural population. "
        "The country's chief wealth came from the fertile fl  oodplain of the Nile valley and delta.",
        "Ancient_Egypt_From.pdf", 16, "egypt-nile",
        {"source_filename": "Ancient_Egypt_From.pdf"}, .9,
    )
    supported = "The Nile River was crucial to ancient Egyptian civilization as it supported its agricultural population through annual inundation."
    unsupported = "The Nile River allowed Egyptian armies to dominate Mediterranean trade."
    provider = PayloadProvider(explanation_payload(f"{supported} {unsupported}"))

    response = generator([source], provider, ScriptedJudge([True, False, True, False])).explain(ExplanationRequest(
        topic="Ancient Egyptian Civilization",
        question="How did the Nile River contribute to the development of ancient Egyptian civilization?",
    ))

    assert provider.calls == 2
    assert response.explanation == supported
    assert response.grounding.retrieval_status == "sufficient"
    assert response.grounding.sources
    assert response.grounding.sources[0].document_name == "Ancient_Egypt_From.pdf"
    assert response.generation is not None


def test_mixed_response_salvages_three_supported_claims_and_drops_two_unsupported_claims():
    source = RetrievedChunk(
        "The annual inundation supported Egypt's agricultural population. "
        "The fertility of the land and predictability of the inundation ensured high productivity from a single annual crop. "
        "Agriculture centered on cultivation of cereal crops, chiefly emmer wheat and barley.",
        "Ancient_Egypt_From.pdf", 17, "egypt-agriculture", {"page": 17}, .95,
    )
    supported = [
        "The annual inundation supported Egypt's agricultural population.",
        "Predictable flooding supported high productivity from a single annual crop.",
        "Emmer wheat and barley were the chief cereal crops.",
    ]
    unsupported = [
        "The annual inundation helped Egyptian armies conquer neighboring countries.",
        "The Nile deposited a thin layer of nutrient-rich silt.",
    ]
    payload = {
        "explanation": supported[0],
        "key_concepts": [supported[1]],
        "examples": [supported[2]],
        "important_points": [unsupported[0]],
        "common_mistakes": [unsupported[1]],
        "recommended_revision": [],
    }
    judge = ScriptedJudge([True, True, True, False, False, True, True, True, False, False])
    provider = PayloadProvider(payload)

    response = generator([source], provider, judge).explain(ExplanationRequest(
        topic="Ancient Egyptian agriculture",
        question="Why did the annual inundation of the Nile matter to agriculture?",
    ))

    assert provider.calls == 2
    assert response.explanation == supported[0]
    assert response.key_concepts == [supported[1]]
    assert response.examples == [supported[2]]
    assert response.important_points == [] and response.common_mistakes == []
    assert "conquer neighboring countries" not in response.model_dump_json()
    assert "nutrient-rich silt" not in response.model_dump_json()
    assert response.grounding.retrieval_status == "sufficient"
    assert response.grounding.failure_status is None


def test_unsupported_revision_claim_is_judged_removed_and_sent_as_retry_feedback():
    source = RetrievedChunk(
        "The annual inundation of the Nile supported Egypt's agricultural population.",
        "Ancient_Egypt_From.pdf", 16, "egypt-nile", {"page": 16}, .95,
    )
    rejected_revision = "The Nile flood deposited nutrient-rich silt on the floodplain."
    payload = explanation_payload(source.content)
    payload["recommended_revision"] = [rejected_revision]
    provider = PayloadProvider(payload)

    response = generator([source], provider, ScriptedJudge([True, False, True, False])).explain(
        ExplanationRequest(topic="Ancient Egyptian agriculture", question="Why did the annual inundation matter?")
    )

    assert provider.calls == 2
    assert "rejected" in provider.prompts[-1]
    assert rejected_revision in provider.prompts[-1]
    assert response.explanation == source.content
    assert response.recommended_revision == []
    assert "nutrient-rich silt" not in response.model_dump_json()
    assert response.grounding.retrieval_status == "sufficient"
    assert response.grounding.sources


def test_nile_military_and_nutrient_silt_additions_are_rejected_by_semantic_decisions():
    source = _nile_regression_chunks()[0]
    claims = [
        "The Nile helped Egyptian armies conquer neighboring countries.",
        "The Nile deposited a thin layer of nutrient-rich silt.",
        "The Nile flood was the main source of water and nutrients for Egyptian farms.",
        "The text describes the annual flood as depositing fertile mud.",
    ]
    context = GroundedContext([source], source.content, "sufficient", "grounded")
    service = generator([source], judge=ScriptedJudge([False, False, False, False]))
    validation = service._validate_claim_batch(
        claims, "Ancient Egyptian agriculture", "Why did the Nile inundation matter?", context,
    )

    assert all(not validation.decisions[service._claim_key(claim)].accepted for claim in claims)


def test_exact_nile_why_query_accepts_page_16_evidence_and_can_ground():
    passage = (
        "Ancient Egyptian civilization developed in northeastern Africa in the 3rd millennium BC. "
        "Ancient Egypt can be thought of as an oasis in the desert of northeastern Africa, "
        "dependent on the annual inundation of the Nile River to support its agricultural population."
    )
    source = RetrievedChunk(passage, "Ancient_Egypt_From.pdf", 16, "egypt-nile",
                           {"source_filename": "Ancient_Egypt_From.pdf", "page": 16}, .704)
    query = "Why was the Nile River important to the development of ancient Egyptian civilization?"
    provider = PayloadProvider(explanation_payload(passage))

    context = generator([source], provider)._context(query, "explain")
    response = generator([source], provider).explain(ExplanationRequest(
        topic="ancient Egyptian civilization", question=query,
    ))

    assert context.retrieval_status == "sufficient"
    assert any(chunk.page_number == 16 and "annual inundation" in chunk.content for chunk in context.chunks)
    assert response.grounding.retrieval_status == "sufficient"
    assert response.grounding.grounding_status == "grounded"
    assert response.grounding.sources[0].document_name == "Ancient_Egypt_From.pdf"


@pytest.mark.parametrize("sentence", [
    "Ancient Egyptian civilization developed along the Nile because its annual floods sustained agriculture.",
    "Ancient Egyptian civilization depended on the annual inundation of the Nile River to support its agricultural population.",
    "The Nile River enabled ancient Egyptian civilization to develop a stable agricultural population.",
    "The Nile River allowed ancient Egyptian civilization to support its agricultural population.",
])
def test_why_evidence_accepts_explicit_and_functional_causal_constructions(sentence):
    source = RetrievedChunk(sentence, "Ancient_Egypt_From.pdf", 16, "egypt-nile", relevance_score=.9)
    query = "Why was the Nile River important to the development of ancient Egyptian civilization?"

    accepted = task_evidence(query, [source], "explain")

    assert len(accepted) == 1


def test_why_evidence_does_not_accept_topical_noncausal_paragraph():
    source = RetrievedChunk(
        "The Nile River was an important feature in the development of ancient Egyptian civilization.",
        "Ancient_Egypt_From.pdf", 16, "egypt-nile", relevance_score=.9,
    )

    assert task_evidence(
        "Why was the Nile River important to the development of ancient Egyptian civilization?",
        [source], "explain",
    ) == []


def test_why_evidence_still_rejects_unrelated_quantum_query():
    source = RetrievedChunk(
        "Ancient Egyptian civilization depended on the annual inundation of the Nile River to support its agricultural population.",
        "Ancient_Egypt_From.pdf", 16, "egypt-nile", relevance_score=.9,
    )
    response = generator([source]).explain(ExplanationRequest(
        topic="quantum teleportation", question="How does quantum teleportation work?",
    ))

    assert response.grounding.source_mode == "insufficient_context"
    assert response.grounding.retrieval_status == "insufficient"
    assert response.grounding.sources == []
    assert response.generation is None


def test_mixed_summary_keeps_supported_points_and_drops_unsupported_claims():
    supported = "The Estates-General had three estates: clergy, nobility, and the Third Estate."
    unsupported = "The king faced a severe financial crisis."
    provider = PayloadProvider({"summary": f"{supported} {unsupported}", "key_points": [supported, unsupported]})

    response = generator([ESTATES], provider, ScriptedJudge([True, False, True, False])).summarize(SummaryRequest(topic="French Revolution"))

    assert provider.calls == 2
    assert response.summary == supported
    assert response.key_points == [supported]
    assert response.grounding.retrieval_status == "sufficient"
    assert response.grounding.sources[0].chunk_id == "estates"
    assert response.generation is not None


def test_source_selection_repairs_pdf_double_space_word_splits():
    source = RetrievedChunk(
        "The country's chief wealth came from the fertile fl  oodplain of the Nile valley and the Nile delta.",
        "Ancient_Egypt_From.pdf", 16, "egypt-nile", relevance_score=.9,
    )

    assert source_for_claim("The fertile floodplain of the Nile valley and the Nile delta.", [source]).chunk_id == "egypt-nile"


def test_causal_question_fails_closed_without_explicit_causal_evidence():
    provider = PayloadProvider(explanation_payload("Louis XVI called the Estates-General."))
    response = generator([ESTATES], provider).explain(ExplanationRequest(topic="Estates-General", question="Why did Louis XVI call the Estates-General?"))
    assert provider.calls == 0
    assert response.grounding.retrieval_status == "insufficient" and response.generation is None


def test_heading_question_list_and_definition_decoy_are_not_evidence():
    heading = RetrievedChunk("Definition of Linear Dependence and Linear Independence", "decoy.pdf", 1, "heading", relevance_score=.99)
    questions = RetrievedChunk("What is linear independence? What are the axioms? Define basis.", "decoy.pdf", 2, "questions", relevance_score=.98)
    decoy = RetrievedChunk("This decoy document does not provide definitions or axioms for linear independence.", "decoy.pdf", 3, "decoy", relevance_score=.97)
    provider = PayloadProvider(explanation_payload("A made-up definition."))
    response = generator([heading, questions, decoy], provider).explain(ExplanationRequest(topic="Linear independence", question="What is the definition of linear independence?"))
    assert provider.calls == 0
    assert response.grounding.retrieval_status == "insufficient" and response.generation is None


def _mcq(question, answer, explanation):
    return {"question": question, "options": ["The Bastille", "The Declaration of the Rights of Man", "Liberty and equality", "None of these"],
            "correct_answer": answer, "explanation": explanation, "difficulty": "medium", "topic": "French Revolution", "learning_objective": "Recall a supported historical fact"}


def test_mcq_items_are_validated_and_get_claim_specific_sources():
    payload = {"questions": [
        _mcq("What did French revolutionaries storm?", "The Bastille", "Revolutionaries stormed the Bastille on 14 July 1789."),
        _mcq("What principles did the Declaration establish?", "Liberty, equality, and political rights", "The Declaration of the Rights of Man established liberty, equality, and political rights."),
    ]}
    provider = PayloadProvider(payload)
    result = generator([ESTATES, BASTILLE, DECLARATION], provider).mcq(MCQRequest(topic="French Revolution", number_of_questions=2))
    assert len(result.questions) == 2
    assert result.questions[0].source.chunk_id == "bastille"
    assert result.questions[1].source.chunk_id == "declaration"
    assert "Bastille" in result.questions[0].source.snippet
    assert {source.chunk_id for source in result.grounding.sources} == {"bastille", "declaration"}


def test_mcq_distractors_may_be_false_without_invalidating_supported_answer():
    payload = {"questions": [_mcq(
        "What did French revolutionaries storm?", "The Bastille",
        "Roman emperors valued the Bastille as a measure of grain.",
    )]}
    payload["questions"][0]["options"] = [
        "The Bastille", "The Great Pyramid of Khufu", "The Yellow River", "Napoleon Bonaparte",
    ]

    provider = PayloadProvider(payload)
    result = generator([BASTILLE], provider, ScriptedJudge([True, False, True, False])).mcq(MCQRequest(
        topic="French Revolution", number_of_questions=1,
    ))

    assert provider.calls == 2
    assert len(result.questions) == 1
    assert result.questions[0].explanation == "The Bastille"
    assert result.questions[0].source.chunk_id == "bastille"
    assert result.grounding.retrieval_status == "sufficient"


def test_mcq_with_unsupported_correct_answer_fails_closed():
    payload = {"questions": [_mcq(
        "What did French revolutionaries storm?", "Napoleon Bonaparte",
        "Napoleon Bonaparte captured the Bastille in 1804.",
    )]}
    provider = PayloadProvider(payload)

    result = generator([BASTILLE], provider, ScriptedJudge([False, False, False, False])).mcq(MCQRequest(
        topic="French Revolution", number_of_questions=1,
    ))

    assert provider.calls == 2
    assert result.questions == []
    assert result.grounding.retrieval_status == "sufficient"
    assert result.grounding.failure_status == "grounding_validation_failed"
    assert result.grounding.sources
    assert result.generation is None


def test_flashcard_answer_source_is_specific():
    provider = PayloadProvider({"cards": [{"front": "What happened at the Bastille?", "back": "Revolutionaries stormed the Bastille on 14 July 1789.", "topic": "French Revolution", "difficulty": "medium", "concept": "Bastille"}]})
    result = generator([BASTILLE, DECLARATION], provider).flashcards(FlashcardRequest(topic="French Revolution", number_of_cards=1))
    assert len(result.cards) == 1
    assert result.cards[0].source.chunk_id == "bastille"


def test_exam_questions_use_matching_evidence_and_keep_time_limit():
    payload = {"title": "French Revolution", "questions": [
        _mcq("What did French revolutionaries storm?", "The Bastille", "Revolutionaries stormed the Bastille on 14 July 1789."),
        _mcq("What principles did the Declaration establish?", "Liberty, equality, and political rights", "The Declaration of the Rights of Man established liberty, equality, and political rights."),
    ]}
    result = generator([BASTILLE, DECLARATION], PayloadProvider(payload)).exam(PracticeExamRequest(topics=["French Revolution"], number_of_questions=2, time_limit_minutes=20))
    assert len(result.questions) == 2 and result.time_limit_minutes == 20
    assert [item.source.chunk_id for item in result.questions] == ["bastille", "declaration"]


def test_evaluate_answer_uses_direct_answer_evidence():
    payload = {"correct": True, "score": .95, "correct_answer": "The Bastille", "explanation": "Revolutionaries stormed the Bastille on 14 July 1789.", "why_answer_is_wrong": None, "related_concept": "Bastille", "recommended_revision": [], "confidence": .95, "identified_misconception": None}
    request = AnswerEvaluationRequest(question="What did French revolutionaries storm?", student_answer="The Bastille", correct_answer="The Bastille", topic="French Revolution")
    result = generator([BASTILLE, DECLARATION], PayloadProvider(payload)).evaluate(request)
    assert result.grounding.retrieval_status == "sufficient"
    assert [source.chunk_id for source in result.supporting_source_or_evidence] == ["bastille"]


def test_evaluate_retrieval_query_uses_topic_question_and_authoritative_answer_only():
    evidence = RetrievedChunk(
        "Copper conducts heat efficiently because its free electrons transfer energy.",
        "materials.pdf", 2, "copper", {"source_filename": "materials.pdf"}, .95,
    )

    class QueryRecorder(FixedRetriever):
        def retrieve(self, query, filters=None, top_k=5):
            self.query = query
            return super().retrieve(query, filters, top_k)

    retriever = QueryRecorder([evidence])
    payload = {"correct": False, "score": 0, "correct_answer": "Copper conducts heat efficiently.",
               "explanation": "Copper conducts heat efficiently because its free electrons transfer energy.", "why_answer_is_wrong": None,
               "related_concept": "Thermal conductivity", "recommended_revision": [],
               "confidence": .9, "identified_misconception": None}
    request = AnswerEvaluationRequest(
        question="What material efficiently transfers thermal energy?",
        student_answer="Wood is the best conductor.",
        correct_answer="Copper conducts heat efficiently.", topic="Thermal conductivity",
    )

    result = LearningGenerator(retriever, llm=PayloadProvider(payload),
                               settings=Settings(_env_file=None, max_retries=0),
                               min_relevance_score=.1,
                               claim_judge_llm=ScriptedJudge()).evaluate(request)

    assert "Thermal conductivity" in retriever.query
    assert request.question in retriever.query
    assert request.correct_answer in retriever.query
    assert request.student_answer not in retriever.query
    assert result.grounding.retrieval_status == "sufficient"
    assert [source.chunk_id for source in result.supporting_source_or_evidence] == ["copper"]


def test_evaluate_answer_recovers_grounded_grade_when_model_rationale_is_unsupported():
    passage = "Copper conducts heat efficiently because its free electrons transfer energy."
    payload = {"correct": False, "score": 0, "correct_answer": "Copper conducts heat efficiently.",
               "explanation": "Wood conducts heat better than copper in every application.",
               "why_answer_is_wrong": "Copper is never used to transfer heat.",
               "related_concept": "Thermal conductivity", "recommended_revision": [],
               "confidence": .9, "identified_misconception": None}
    request = AnswerEvaluationRequest(
        question="What material efficiently transfers thermal energy?",
        student_answer="Wood is the best conductor.",
        correct_answer="Copper conducts heat efficiently.", topic="Thermal conductivity",
    )

    result = generator([RetrievedChunk(passage, "materials.pdf", 2, "copper",
                                       {"source_filename": "materials.pdf"}, .95)],
                       PayloadProvider(payload), ScriptedJudge([True, False, False, False, False])).evaluate(request)

    assert not result.correct and result.score == 0
    assert result.explanation != request.correct_answer
    assert passage in result.explanation
    assert request.student_answer not in result.explanation
    assert not result.why_answer_is_wrong
    assert result.identified_misconception is None
    assert not result.why_answer_is_wrong
    assert result.identified_misconception is None
    assert result.grounding.retrieval_status == "sufficient"
    assert result.grounding.grounding_status == "grounded"
    assert [source.chunk_id for source in result.supporting_source_or_evidence] == ["copper"]


def test_exact_supported_answer_survives_unsupported_generated_rationale():
    payload = {
        "correct": True, "score": 1, "correct_answer": "The Bastille",
        "explanation": "The response is supported by a long historical explanation absent from the source.",
        "why_answer_is_wrong": "The answer is unrelated to an unsupported dynasty.",
        "related_concept": "French Revolution", "recommended_revision": [], "confidence": 1,
        "identified_misconception": None,
    }
    request = AnswerEvaluationRequest(
        question="What did French revolutionaries storm?", student_answer="The Bastille",
        correct_answer="The Bastille", topic="French Revolution",
    )

    result = generator([BASTILLE], PayloadProvider(payload), ScriptedJudge([True, False, False])).evaluate(request)

    assert result.correct and result.score == 1
    assert result.explanation != "The Bastille"
    assert "stormed the Bastille" in result.explanation
    assert result.grounding.retrieval_status == "sufficient"
    assert result.supporting_source_or_evidence[0].chunk_id == "bastille"


def test_exact_match_recovery_respects_request_correct_answer():
    payload = {
        "correct": True, "score": 1, "correct_answer": "The Bastille",
        "explanation": "An unsupported explanation with several made-up historical details.",
        "why_answer_is_wrong": "A made-up explanation.", "related_concept": "French Revolution",
        "recommended_revision": [], "confidence": 1, "identified_misconception": None,
    }
    provider = PayloadProvider(payload)
    request = AnswerEvaluationRequest(
        question="What principles did the Declaration establish?", student_answer="The Bastille",
        correct_answer="Liberty, equality, and political rights", topic="French Revolution",
    )

    result = generator([BASTILLE, DECLARATION], provider,
                       ScriptedJudge([True, False, False, False, False])).evaluate(request)

    assert provider.calls == 2
    assert not result.correct and result.grounding.retrieval_status == "sufficient"
    assert result.correct_answer == request.correct_answer
    assert "Declaration of the Rights of Man" in result.explanation
    assert result.generation is not None


def test_out_of_material_request_does_not_call_provider_and_has_null_generation():
    provider = PayloadProvider(explanation_payload("Some unsupported claim."))
    response = generator([BASTILLE, DECLARATION], provider).explain(ExplanationRequest(topic="French Revolution", question="What was the exact temperature in Paris in 1790?"))
    assert provider.calls == 0
    assert response.grounding.retrieval_status == "insufficient" and response.generation is None


def test_task_packet_only_contains_filtered_chunks():
    title = RetrievedChunk("French Revolution History Learning Material", "history.pdf", 1, "title", relevance_score=.99)
    question_list = RetrievedChunk("What happened at the Bastille? Which estate was represented?", "history.pdf", 2, "questions", relevance_score=.98)
    assert task_evidence("Bastille French Revolution", [title, question_list, BASTILLE], "explain") == [BASTILLE]
    assert source_for_claim("Revolutionaries stormed the Bastille on 14 July 1789.", [BASTILLE]).chunk_id == "bastille"
