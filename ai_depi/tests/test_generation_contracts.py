import json

import pytest

from app.config import Settings
from app.domain import RetrievedChunk
from app.generation import LearningGenerator
from app.providers import _compact_provider_schema, generate_structured
from app.rag import InMemoryRetriever
from app.validation import OutputValidationError
from app.schemas import (
    AnswerEvaluationRequest, AnswerEvaluation, ExplanationRequest, ExplanationResponse,
    FlashcardRequest, FlashcardSet, MCQRequest, MCQSet, PracticeExamRequest, PracticeExam,
)


class SchemaEchoProvider:
    name = "groq"
    model = "test-groq-model"

    def __init__(self):
        self.schemas = []

    def generate(self, system_prompt, user_prompt="", response_schema=None, **kwargs):
        self.schemas.append(response_schema)
        if "correct" in (response_schema or {}).get("properties", {}):
            return json.dumps({
                "correct": False, "score": 0.25,
                "correct_answer": "Provider answer is replaced by the trusted request answer.",
                "explanation": "A vector space has addition and scalar multiplication that obey its defining axioms.",
                "why_answer_is_wrong": None, "related_concept": "Vector spaces",
                "recommended_revision": [], "confidence": .8, "identified_misconception": None,
            })
        def value_for(node):
            if "enum" in node: return node["enum"][0]
            if "anyOf" in node:
                option = next((option for option in node["anyOf"] if option.get("type") == "null"), None)
                return None if option is not None else value_for(node["anyOf"][0])
            kind = node.get("type")
            if kind == "object": return {key: value_for(child) for key, child in node.get("properties", {}).items()}
            if kind == "array": return [value_for(node.get("items", {})) for _ in range(node.get("minItems", 0))]
            if kind == "string": return "A vector space is a set with addition and scalar multiplication that satisfies the vector space axioms."
            if kind == "integer": return 1
            if kind == "number": return 0.75
            if kind == "boolean": return True
            return None
        return json.dumps(value_for(response_schema))


class AcceptingJudge:
    name, model = "ollama", "qwen2.5:7b-instruct-test"

    def generate(self, system_prompt, user_prompt="", response_schema=None, **kwargs):
        return json.dumps({"supported": True, "confidence": .96, "reason": "Supported by the test evidence."})


def make_generator(provider):
    evidence = RetrievedChunk(
        "A vector space is a set with addition and scalar multiplication that satisfies the vector space axioms.",
        "linear_algebra.pdf", 4, "chunk-4", relevance_score=0.9,
    )
    return LearningGenerator(InMemoryRetriever([evidence]), llm=provider,
                             settings=Settings(_env_file=None, max_retries=0), min_relevance_score=0.35,
                             claim_judge_llm=AcceptingJudge())


@pytest.mark.parametrize("kind,expected", [("mcq", 5), ("flashcards", 5), ("exam", 5)])
def test_requested_item_count_is_enforced_and_adaptation_preserved(kind, expected):
    provider = SchemaEchoProvider()
    generator = make_generator(provider)
    profile = {"topic_mastery": {"Vector spaces": 0.2}}
    if kind == "mcq":
        response = generator.mcq(MCQRequest(topic="Vector spaces", number_of_questions=expected, learner_profile=profile))
        items = response.questions
        schema_field = "questions"
    elif kind == "flashcards":
        response = generator.flashcards(FlashcardRequest(topic="Vector spaces", number_of_cards=expected, learner_profile=profile))
        items = response.cards
        schema_field = "cards"
    else:
        response = generator.exam(PracticeExamRequest(topics=["Vector spaces"], number_of_questions=expected, learner_profile=profile))
        items = response.questions
        schema_field = "questions"
    assert len(items) == expected
    assert response.generation.provider == "groq"
    assert all(item.difficulty == "easy" for item in items)
    assert provider.schemas[-1]["properties"][schema_field]["minItems"] == expected
    assert provider.schemas[-1]["properties"][schema_field]["maxItems"] == expected
    assert len(provider.schemas) == 1


def test_evaluate_answer_server_stamps_grounding_and_validates_fields():
    provider = SchemaEchoProvider()
    result = make_generator(provider).evaluate(AnswerEvaluationRequest(
        question="What is a vector space?", student_answer="A set satisfying the vector space axioms.",
        correct_answer="A vector space is a set with addition and scalar multiplication that satisfies the vector space axioms.", topic="Vector spaces",
    ))
    assert isinstance(result, AnswerEvaluation)
    assert result.topic == "Vector spaces"
    assert result.grounding.retrieval_status == "sufficient"
    assert result.supporting_source_or_evidence
    assert result.generation.provider == "groq"


@pytest.mark.parametrize("schema,counts", [
    (MCQSet, {"questions": 5}),
    (FlashcardSet, {"cards": 5}),
    (PracticeExam, {"questions": 5}),
    (AnswerEvaluation, {}),
])
def test_provider_schema_is_compact_and_has_endpoint_constraints(schema, counts):
    remote = _compact_provider_schema(schema, expected_counts=counts)
    properties = remote["properties"]
    assert not {"grounding", "adaptation", "generation", "supporting_source_or_evidence", "groundedness"} & set(properties)
    for field, count in counts.items():
        assert properties[field]["minItems"] == count
        assert properties[field]["maxItems"] == count
    if schema in (MCQSet, PracticeExam):
        assert "source" not in properties["questions"]["items"]["properties"]
    if schema is FlashcardSet:
        assert "source" not in properties["cards"]["items"]["properties"]


def test_heading_only_retrieval_is_insufficient_for_explanation():
    heading = "Definition of Linear Dependence and Linear Independence Linear Independence Linear Algebra page 2"
    provider = SchemaEchoProvider()
    generator = LearningGenerator(InMemoryRetriever([RetrievedChunk(heading, "LA-Ch04-6.pdf", 2, "heading", relevance_score=0.9)]), llm=provider, claim_judge_llm=AcceptingJudge())
    response = generator.explain(ExplanationRequest(topic="Linear independence", question="What is linear independence?"))
    assert response.grounding.retrieval_status == "insufficient"
    assert response.grounding.grounding_status == "not_grounded"
    assert response.grounding.sources == []
    assert not response.key_concepts and not provider.schemas


def test_invalid_requested_count_fails_validation_instead_of_returning_short_set():
    class TooFew:
        name = "groq"
        model = "test-model"
        def generate(self, *args, **kwargs):
            return json.dumps({"questions": [], "grounding": {}, "adaptation": {}})
    with pytest.raises(OutputValidationError, match="exactly 5"):
        generate_structured(TooFew(), "system", "Return five questions", MCQSet, retries=0,
                            expected_counts={"questions": 5},
                            trusted_fields={"grounding": {"source_mode": "insufficient_context", "grounding_status": "not_grounded", "retrieval_status": "insufficient", "sources": []},
                                            "adaptation": {"requested_settings": {}, "adapted_settings": {}, "adaptation_reasons": []}})
