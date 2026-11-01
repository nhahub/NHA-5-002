import json
import logging

from app.config import Settings
from app.domain import GroundedContext, RetrievedChunk
from app.generation import LearningGenerator
from app.providers import FallbackLLMProvider, ProviderError
from app.rag import Retriever
from app.schemas import AnswerEvaluationRequest


class FixedRetriever(Retriever):
    method = "test"

    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query, filters=None, top_k=5):
        return self.chunks[:top_k]


class ScriptedProvider:
    def __init__(self, name, judge=None, malformed_judge=False, evaluation_claim=None, evaluation_payload=None):
        self.name, self.model = name, f"{name}-test-model"
        self.judge = judge or {"supported": True, "confidence": .96,
                               "reason": "The evidence entails the claim."}
        self.malformed_judge = malformed_judge
        self.evaluation_claim = evaluation_claim or "Copper conducts heat efficiently."
        self.evaluation_payload = evaluation_payload
        self.calls = []
        self.claim_prompt = None
        self.claim_prompts = []
        self.system_prompts = []
        self.judge_schema = None

    def generate(self, system_prompt, user_prompt="", response_schema=None, **kwargs):
        properties = set((response_schema or {}).get("properties", {}))
        if {"supported", "confidence", "reason"}.issubset(properties):
            self.calls.append("judge")
            self.system_prompts.append(system_prompt)
            self.claim_prompt = user_prompt
            self.claim_prompts.append(user_prompt)
            self.judge_schema = response_schema
            if self.malformed_judge:
                return "not-json"
            decision = self.judge(json.loads(user_prompt)) if callable(self.judge) else self.judge
            return json.dumps(decision)
        self.calls.append("evaluation")
        if self.evaluation_payload is not None:
            return json.dumps(self.evaluation_payload)
        claim = self.evaluation_claim
        return json.dumps({
            "correct": False,
            "score": 0,
            "correct_answer": claim,
            "explanation": claim,
            "why_answer_is_wrong": None,
            "related_concept": "test concept",
            "recommended_revision": [],
            "confidence": .9,
            "identified_misconception": None,
        })


class FailingProvider:
    def __init__(self, name):
        self.name, self.model, self.calls = name, f"{name}-test-model", 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        raise ProviderError(f"{self.name} unavailable")


def evidence(text="Copper conducts heat efficiently because free electrons transfer energy.", *, page=2):
    return RetrievedChunk(text, "science.pdf", page, f"chunk-{page}",
                          {"source_filename": "science.pdf", "page": page}, .95)


def request(claim="Copper conducts heat efficiently.", question="How does copper conduct heat?",
            topic="Thermal conductivity", student_answer="Wood is the best conductor."):
    return AnswerEvaluationRequest(question=question, student_answer=student_answer,
                                   correct_answer=claim, topic=topic)


def make_generator(chunks, *, generation=None, judge=None, confidence=.80):
    generation = generation or ScriptedProvider("groq")
    judge = judge or ScriptedProvider("ollama")
    generator = LearningGenerator(
        FixedRetriever(chunks), llm=generation,
        settings=Settings(_env_file=None, max_retries=0, claim_judge_min_confidence=confidence),
        min_relevance_score=.1, claim_judge_llm=judge,
    )
    return generator, generation, judge


def test_exact_claim_uses_local_judge_even_when_wording_matches():
    generator, generation, judge = make_generator([evidence()])
    result = generator.evaluate(request())

    assert len(judge.calls) == 3
    assert generation.calls
    assert result.grounding.grounding_status == "grounded"


def test_incorrect_answer_gets_distinct_grounded_tutoring_and_misconception():
    passage = "Copper conducts heat efficiently because free electrons transfer energy."
    student_answer = "Wood is the best conductor."
    claim = "Copper conducts heat efficiently."
    feedback = {
        "correct": False, "score": 0, "correct_answer": claim,
        "explanation": passage,
        "why_answer_is_wrong": f"{student_answer} {passage}",
        "related_concept": "Thermal conductivity", "recommended_revision": [],
        "confidence": .95, "identified_misconception": "Wood rather than copper's free-electron energy transfer.",
    }
    generation = ScriptedProvider("groq", evaluation_payload=feedback)
    generator, generation, judge = make_generator([evidence(passage)], generation=generation)
    result = generator.evaluate(request(claim, "Which material conducts heat efficiently?", "Thermal conductivity"))

    assert not result.correct and result.score == 0
    assert result.why_answer_is_wrong
    assert result.explanation != result.correct_answer
    assert passage.split(" because")[0] in result.explanation
    assert result.identified_misconception
    assert judge.calls
    assert generation.calls
    assert result.grounding.grounding_status == "grounded"


def test_correct_answer_has_full_score_and_no_wrong_answer_feedback():
    claim = "Copper conducts heat efficiently because free electrons transfer energy."
    feedback = {
        "correct": True, "score": 1, "correct_answer": claim,
        "explanation": claim, "why_answer_is_wrong": None,
        "related_concept": "Thermal conductivity", "recommended_revision": [],
        "confidence": .95, "identified_misconception": None,
    }
    generation = ScriptedProvider("groq", evaluation_payload=feedback)
    generator, generation, judge = make_generator([evidence()], generation=generation)
    result = generator.evaluate(request(claim, claim, "Thermal conductivity", student_answer=claim))

    assert result.correct is True and result.score == 1
    assert result.why_answer_is_wrong is None
    assert result.identified_misconception is None
    assert judge.calls


def test_supported_paraphrase_uses_ollama_and_final_evidence():
    passage = ("Ancient Egypt can be thought of as an oasis in the desert of northeastern Africa, "
               "dependent on the annual inundation of the Nile River to support its agricultural population.")
    claim = "The annual flooding of the Nile helped sustain Egypt's agricultural population."
    generator, generation, judge = make_generator(
        [evidence(passage, page=16)],
        generation=ScriptedProvider("groq", evaluation_claim=claim),
    )
    result = generator.evaluate(request(claim, "Why did the river's seasonal flooding matter to the civilization?",
                                        "Ancient Egyptian Agriculture"))

    assert judge.calls and judge.calls[0] == "judge"
    assert judge.name == "ollama"
    assert passage in judge.claim_prompt
    assert "question" not in json.loads(judge.claim_prompt)
    assert "topic" not in json.loads(judge.claim_prompt)
    assert "inspect the claim's material details separately" in judge.system_prompts[0]
    assert "does NOT support the claim 'the flood deposited a layer of nutrient-rich silt'" in judge.system_prompts[0]
    assert "'Fertile floodplain' alone does not entail 'the flood deposited silt'" in judge.system_prompts[0]
    assert "the evidence says" in judge.system_prompts[0]
    assert "Wood is the best conductor" in judge.claim_prompts[0]
    assert generation.calls
    assert result.grounding.retrieval_status == "sufficient"
    assert result.grounding.grounding_status == "grounded"
    assert result.generation is not None
    assert result.supporting_source_or_evidence[0].page_number == 16
    assert set(judge.judge_schema["properties"]) == {"supported", "confidence", "reason"}
    assert judge.judge_schema["properties"]["confidence"]["minimum"] == 0
    assert judge.judge_schema["properties"]["confidence"]["maximum"] == 1


def test_claim_extraction_splits_sentences_after_closing_quotes():
    text = (
        'The source says "productivity came from one annual crop." '
        'This reliable flood allowed farmers to store surpluses. The surplus supported wealth.'
    )

    claims = LearningGenerator._claim_sentences(text)

    assert len(claims) == 3
    assert claims[0].endswith("annual crop.")
    assert claims[1] == "This reliable flood allowed farmers to store surpluses."
    assert claims[2] == "The surplus supported wealth."


def test_explanation_revision_guidance_claims_are_included_in_validation():
    from types import SimpleNamespace
    from app.schemas import ExplanationResponse

    result = SimpleNamespace(
        explanation="The inundation supported agriculture.",
        key_concepts=[], examples=[], important_points=[], common_mistakes=[],
        recommended_revision=[
            "Summarize the cause-and-effect chain: Nile flood → silt deposition → fertile soil → surplus."
        ],
    )

    claims = [sentence for _, _, _, sentence in LearningGenerator._claim_locations(result, ExplanationResponse)]

    assert result.recommended_revision[0] in claims


def test_markdown_fenced_json_is_malformed_and_fails_closed():
    class FencedOllama(ScriptedProvider):
        def generate(self, system_prompt, user_prompt="", response_schema=None, **kwargs):
            properties = set((response_schema or {}).get("properties", {}))
            if {"supported", "confidence", "reason"}.issubset(properties):
                self.calls.append("judge")
                self.judge_schema = response_schema
                return '```json\n{"supported":true,"confidence":0.95,"reason":"Supported by evidence."}\n```'
            return super().generate(system_prompt, user_prompt, response_schema, **kwargs)

    judge = FencedOllama("ollama")
    claim = "Mobile electrons in copper move thermal energy."
    generator, generation, judge = make_generator(
        [evidence()], judge=judge, generation=ScriptedProvider("groq", evaluation_claim=claim))
    result = generator.evaluate(request(claim, "How do electrons transfer heat?", "Copper"))

    assert judge.calls == ["judge"]
    assert len(generation.calls) == 1
    assert result.grounding.retrieval_status == "sufficient"
    assert result.grounding.failure_status == "grounding_validation_failed"
    assert result.grounding.sources


def test_missing_required_reason_fails_closed():
    judge = ScriptedProvider("ollama", {"supported": True, "confidence": .95})
    generator, generation, judge = make_generator([evidence()], judge=judge)
    result = generator.evaluate(request("Mobile electrons in copper move thermal energy.",
                                        "How do electrons transfer heat?", "Copper"))

    assert judge.calls == ["judge"]
    assert len(generation.calls) == 1
    assert result.generation is None


def test_out_of_range_confidence_fails_closed():
    judge = ScriptedProvider("ollama", {"supported": True, "confidence": 85,
                                         "reason": "The evidence supports the claim."})
    generator, generation, judge = make_generator([evidence()], judge=judge)
    result = generator.evaluate(request("Mobile electrons in copper move thermal energy.",
                                        "How do electrons transfer heat?", "Copper"))

    assert judge.calls == ["judge"]
    assert len(generation.calls) == 1
    assert result.generation is None


def test_judge_rejects_coerced_types_and_extra_fields():
    passage = "Copper conducts heat efficiently because free electrons transfer energy."
    invalid_payloads = [
        {"supported": "true", "confidence": .95, "reason": "Supported."},
        {"supported": True, "confidence": "0.95", "reason": "Supported."},
        {"supported": True, "confidence": .95, "reason": "Supported.", "extra": "not allowed"},
    ]
    context = GroundedContext([evidence(passage)], passage, "sufficient", "grounded")

    for payload in invalid_payloads:
        judge = ScriptedProvider("ollama", payload)
        generator, _, _ = make_generator([evidence(passage)], judge=judge)
        decision = generator._semantic_claim_support(
            "Copper conducts heat efficiently.", "Thermal conductivity", context,
            question="How does copper transfer heat?",
        )
        assert decision.provider_failed
        assert not decision.accepted
        assert judge.calls == ["judge"]


def test_semantic_validation_logs_counts_confidence_latency_and_not_claim_text(caplog):
    claim = "Copper conducts heat efficiently."
    passage = "Copper conducts heat efficiently because free electrons transfer energy."
    generator, _, _ = make_generator([evidence(passage)])
    context = GroundedContext([evidence(passage)], passage, "sufficient", "grounded")

    with caplog.at_level(logging.INFO, logger="learning.generation"):
        generator._validate_claim_batch([claim], "Thermal conductivity", "How does copper conduct heat?", context)

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "semantic_claim_validation_started" in logs
    assert "semantic_claim_validation_result" in logs
    assert "provider=ollama" in logs and "model=ollama-test-model" in logs
    assert "claims=1" in logs and "accepted=1" in logs and "rejected=0" in logs
    assert "confidence=0.960" in logs and "validation_latency_ms=" in logs
    assert claim not in logs and passage not in logs


def test_unsupported_expected_answer_fails_closed_after_generation():
    judge = ScriptedProvider("ollama", {"supported": False, "confidence": .98,
                                         "reason": "The evidence does not state that."})
    generator, generation, judge = make_generator([evidence()], judge=judge)
    result = generator.evaluate(request("Copper conducts heat efficiently but cures every disease.",
                                        "What does copper conduct?", "Copper"))

    assert len(judge.calls) >= 2
    assert len(generation.calls) == 2
    assert result.grounding.retrieval_status == "sufficient"
    assert result.grounding.grounding_status == "grounded"
    assert result.grounding.failure_status == "grounding_validation_failed"
    assert result.grounding.sources
    assert result.generation is None


def test_partially_supported_compound_claim_fails_closed():
    judge = ScriptedProvider("ollama", {"supported": False, "confidence": .99,
                                         "reason": "Only the heat-transfer clause is supported."})
    generator, generation, judge = make_generator([evidence()], judge=judge)
    result = generator.evaluate(request("Copper conducts heat efficiently and cures disease.",
                                        "What does copper conduct?", "Copper"))

    assert len(judge.calls) >= 2
    assert len(generation.calls) == 2
    assert result.grounding.retrieval_status == "sufficient"
    assert result.grounding.grounding_status == "grounded"
    assert result.grounding.failure_status == "grounding_validation_failed"
    assert result.grounding.sources
    assert result.generation is None


def test_insufficient_context_skips_semantic_judge():
    generator, generation, judge = make_generator([])
    result = generator.evaluate(request())

    assert judge.calls == []
    assert generation.calls == []
    assert result.grounding.retrieval_status == "insufficient"
    assert result.generation is None


def test_supported_but_low_confidence_semantic_decision_is_rejected():
    judge = ScriptedProvider("ollama", {"supported": True, "confidence": .55,
                                         "reason": "Some wording is uncertain."})
    generator, generation, judge = make_generator([evidence()], judge=judge, confidence=.80)
    result = generator.evaluate(request("Mobile electrons in copper move thermal energy.",
                                        "How do electrons in copper transfer heat?", "Copper"))

    assert len(judge.calls) >= 2
    assert len(generation.calls) == 2
    assert result.grounding.retrieval_status == "sufficient"
    assert result.grounding.failure_status == "grounding_validation_failed"
    assert result.grounding.sources
    assert result.generation is None


def test_ollama_failure_fails_closed_without_groq_attempt():
    judge = FailingProvider("ollama")
    groq = FailingProvider("groq")
    normal_generation = ScriptedProvider("groq")
    generator, _, _ = make_generator([evidence()], generation=normal_generation, judge=judge)
    result = generator.evaluate(request("Mobile electrons in copper move thermal energy.",
                                        "How do electrons in copper transfer heat?", "Copper"))

    assert judge.calls == 1
    assert groq.calls == 0
    assert len(normal_generation.calls) == 1
    assert result.grounding.retrieval_status == "sufficient"
    assert result.grounding.grounding_status == "grounded"
    assert result.grounding.failure_status == "grounding_validation_failed"
    assert result.grounding.sources
    assert result.generation is None


def test_groq_is_never_called_by_claim_judge_when_ollama_unavailable():
    groq = FailingProvider("groq")
    ollama = FailingProvider("ollama")
    normal_generation = ScriptedProvider("groq")
    generator, _, _ = make_generator([evidence()], generation=normal_generation, judge=ollama)
    result = generator.evaluate(request("Mobile electrons in copper move thermal energy.",
                                        "How do electrons in copper transfer heat?", "Copper"))

    assert ollama.calls == 1
    assert groq.calls == 0
    assert len(normal_generation.calls) == 1
    assert result.grounding.retrieval_status == "sufficient"
    assert result.grounding.grounding_status == "grounded"
    assert result.grounding.failure_status == "grounding_validation_failed"
    assert result.grounding.sources
    assert result.generation is None


def test_invalid_ollama_judge_json_fails_closed_without_groq_fallback():
    judge = ScriptedProvider("ollama", malformed_judge=True)
    generation = ScriptedProvider("groq")
    generator, _, judge = make_generator([evidence()], generation=generation, judge=judge)
    result = generator.evaluate(request("Mobile electrons in copper move thermal energy.",
                                        "How do electrons in copper transfer heat?", "Copper"))

    assert judge.calls == ["judge"]
    assert len(generation.calls) == 1
    assert result.grounding.retrieval_status == "sufficient"
    assert result.grounding.failure_status == "grounding_validation_failed"
    assert result.grounding.sources
    assert result.generation is None


def test_misconfigured_groq_provider_is_rejected_before_claim_judge_call():
    groq_judge = ScriptedProvider("groq")
    generation = ScriptedProvider("groq")
    generator, _, _ = make_generator([evidence()], generation=generation, judge=groq_judge)
    result = generator.evaluate(request("Mobile electrons in copper move thermal energy.",
                                        "How do electrons in copper transfer heat?", "Copper"))

    assert groq_judge.calls == []
    assert len(generation.calls) == 1
    assert result.grounding.retrieval_status == "sufficient"
    assert result.grounding.grounding_status == "grounded"
    assert result.grounding.failure_status == "grounding_validation_failed"
    assert result.grounding.sources
    assert result.generation is None


def test_normal_generation_keeps_groq_to_ollama_fallback():
    groq = FailingProvider("groq")
    ollama = ScriptedProvider("ollama", evaluation_claim="Copper conducts heat efficiently.")
    judge = ScriptedProvider("ollama")
    generator, _, _ = make_generator([evidence()], generation=FallbackLLMProvider(groq, ollama), judge=judge)
    # Normal generation fallback remains separate from the Ollama-only judge call.
    result = generator.evaluate(request())

    assert len(judge.calls) == 3
    assert groq.calls >= 1
    assert ollama.calls
    assert result.generation.provider == "ollama"
    assert result.generation.fallback_used is True
    assert result.grounding.grounding_status == "grounded"


def test_nile_paraphrase_regression_uses_ollama_and_is_grounded_with_page_16():
    text = ("Ancient Egyptian civilization depended on the annual inundation of the Nile River "
            "to support its agricultural population.")
    # This intentionally distant paraphrase exercises claim-level semantic validation.
    claim = "The yearly Nile flood sustained Egyptian food production."
    feedback = {
        "correct": False, "score": 0, "correct_answer": claim,
        "explanation": text,
        "why_answer_is_wrong": (
            "The Nile River was mainly important because it provided a route for military expansion. "
            "The annual inundation of the Nile River supported the agricultural population."
        ),
        "related_concept": "Ancient Egyptian Agriculture", "recommended_revision": ["Agriculture"],
        "confidence": .95,
        "identified_misconception": (
            "Military expansion rather than annual inundation supporting the agricultural population."
        ),
    }
    def reject_military_claim(payload):
        supported = "military expansion" not in payload["claim"].casefold()
        return {"supported": supported, "confidence": .96,
                "reason": "The supplied page supports Nile agriculture only."}

    generator, generation, judge = make_generator(
        [evidence(text, page=16)], generation=ScriptedProvider("groq", evaluation_payload=feedback),
        judge=ScriptedProvider("ollama", judge=reject_military_claim))
    req = AnswerEvaluationRequest(
        question="Why was the Nile River important to ancient Egyptian civilization?",
        student_answer="The Nile River was mainly important because it provided a route for military expansion.",
        correct_answer=claim, topic="Ancient Egyptian Agriculture",
    )
    result = generator.evaluate(req)

    assert judge.calls and judge.calls[0] == "judge"
    assert generation.calls
    assert judge.name == "ollama"
    assert result.grounding.retrieval_status == "sufficient"
    assert result.grounding.grounding_status == "grounded"
    assert result.correct is False
    assert result.generation is not None
    assert any(source.page_number == 16 for source in result.supporting_source_or_evidence)
    assert result.explanation != result.correct_answer
    assert result.why_answer_is_wrong and "annual inundation" in result.why_answer_is_wrong
    assert "military expansion" not in result.why_answer_is_wrong.casefold()
    assert "military expansion" not in (result.identified_misconception or "").casefold()
    assert "annual inundation" in result.supporting_source_or_evidence[0].snippet


def test_semantic_judge_gets_student_response_as_attributed_context_for_feedback():
    passage = ("Ancient Egypt depended on the annual inundation of the Nile River "
               "to support its agricultural population.")
    student_answer = "The Nile River was mainly important because it provided a route for military expansion."
    feedback = ("The student said the Nile was important for military expansion, but the evidence "
                "states its importance lay in annual inundation to support the agricultural population.")
    generator, _, judge = make_generator([evidence(passage, page=16)])
    context = GroundedContext([evidence(passage, page=16)], passage, "sufficient", "grounded")
    decision = generator._semantic_claim_support(
        feedback, "Ancient Egyptian agriculture", context,
        question="Why did annual flooding matter?", learner_response=student_answer,
    )

    assert decision.accepted
    assert "learner_response" in judge.claim_prompt
    assert student_answer in judge.claim_prompt


def test_agriculture_cereal_crop_answer_remains_grounded():
    text = "Agriculture centered on cultivation of cereal crops, chiefly emmer wheat and barley."
    claim = "Emmer wheat and barley were the chief cereal crops."
    generator, generation, judge = make_generator(
        [evidence(text, page=17)], generation=ScriptedProvider("groq", evaluation_claim=claim))
    req = AnswerEvaluationRequest(
        question="What crops were chiefly cultivated in ancient Egyptian agriculture?",
        student_answer="Rice and corn were the main crops.",
        correct_answer=claim, topic="Ancient Egyptian Agriculture",
    )
    result = generator.evaluate(req)

    assert judge.calls
    assert generation.calls
    assert result.grounding.grounding_status == "grounded"
    assert result.correct is False
    assert result.generation is not None
    assert any(source.page_number == 17 for source in result.supporting_source_or_evidence)
    assert result.why_answer_is_wrong
    assert result.why_answer_is_wrong != result.explanation
    assert result.explanation != result.correct_answer
    assert result.identified_misconception is None
    assert "expected idea(s)" in result.why_answer_is_wrong
    assert "emmer" in result.why_answer_is_wrong
