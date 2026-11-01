from fastapi.testclient import TestClient
from app.main import app, generator
from app.providers import FallbackLLMProvider, ProviderError
import json


class AcceptingJudge:
    name, model = "ollama", "qwen2.5:7b-instruct-test"

    def __init__(self):
        self.calls, self.prompts = 0, []

    def generate(self, system_prompt, user_prompt="", response_schema=None, **kwargs):
        self.calls += 1
        self.prompts.append(user_prompt)
        return json.dumps({"supported": True, "confidence": .96, "reason": "Scripted API test judgment."})

def test_health_and_mcq():
    c = TestClient(app)
    assert c.get("/health").status_code == 200
    assert c.get("/openapi.json").status_code == 200
    assert c.get("/docs").status_code == 200
    assert c.get("/redoc").status_code == 200
    response = c.post("/learning/mcq", json={"topic":"Convolutional Neural Networks", "number_of_questions":1})
    assert response.status_code == 200
    assert response.json()["grounding"]["source_mode"] == "insufficient_context"

def test_both_provider_failures_return_clean_502(monkeypatch):
    class FailedProvider:
        def __init__(self, name): self.name, self.model = name, f"{name}-test-model"
        def generate(self, *args, **kwargs): raise ProviderError("simulated outage")
    monkeypatch.setattr(generator, "llm", FallbackLLMProvider(FailedProvider("groq"), FailedProvider("ollama")))
    response = TestClient(app).post("/learning/explain", json={"topic":"CNN", "question":"Explain CNN feature extraction.", "knowledge_mode":"learning_materials_then_general_knowledge"})
    assert response.status_code == 502
    assert response.json()["error"] == "llm_provider_unavailable"
    assert response.json()["failure_status"] == "generation_failed"
    assert "simulated outage" in response.json()["detail"]
    assert "Traceback" not in response.text


def test_invalid_model_output_reports_output_validation_failure(monkeypatch):
    import app.main as main

    class InvalidProvider:
        name, model = "invalid-output-test", "invalid-output-test"
        def generate(self, *args, **kwargs): return '{"unexpected": true}'

    monkeypatch.setattr(main.generator, "llm", InvalidProvider())
    response = TestClient(main.app).post("/learning/explain", json={
        "topic": "Ancient Egyptian agriculture",
        "question": "Why did the annual inundation of the Nile matter to agriculture?",
        "top_k": 3,
        "knowledge_mode": "learning_materials_only",
    })

    assert response.status_code == 502
    assert response.json()["error"] == "invalid_llm_output"
    assert response.json()["failure_status"] == "output_validation_failed"
    assert "unexpected" not in response.text


def test_exact_nile_why_endpoint_grounds_page_16_and_quantum_fails_closed(monkeypatch):
    import json
    import app.main as main

    passage = (
        "Ancient Egypt can be thought of as an oasis in the desert of northeastern Africa, "
        "dependent on the annual inundation of the Nile River to support its agricultural population."
    )

    class EvidenceProvider:
        name, model = "offline-regression", "exact-nile-test"
        def generate(self, *args, **kwargs):
            return json.dumps({"explanation": passage, "key_concepts": [], "examples": [],
                               "important_points": [], "common_mistakes": [], "recommended_revision": []})

    monkeypatch.setattr(main.generator, "llm", EvidenceProvider())
    judge = AcceptingJudge()
    monkeypatch.setattr(main.generator, "claim_judge_llm", judge)
    client = TestClient(main.app)
    query = "Why was the Nile River important to the development of ancient Egyptian civilization?"
    nile = client.post("/learning/explain", json={"topic": "Ancient Egyptian civilization", "question": query})

    assert nile.status_code == 200
    nile_body = nile.json()
    assert nile_body["grounding"]["retrieval_status"] == "sufficient"
    assert nile_body["grounding"]["grounding_status"] == "grounded"
    assert nile_body["grounding"]["retrieval_method"] == "hybrid_cross_encoder"
    assert nile_body["grounding"]["sources"][0]["document_name"] == "Ancient_Egypt_From.pdf"
    assert nile_body["grounding"]["sources"][0]["page_number"] == 16
    assert nile_body["generation"] is not None

    hard = client.post("/learning/explain", json={
        "topic": "Ancient Egyptian agriculture",
        "question": "Why did the annual inundation of the Nile matter to agriculture?",
        "difficulty": "hard", "explanation_depth": "high", "top_k": 10,
        "knowledge_mode": "learning_materials_only",
        "learner_profile": {"overall_level": "beginner",
                            "topic_mastery": {"Ancient Egyptian agriculture": .25}},
    })
    assert hard.status_code == 200
    hard_body = hard.json()
    assert hard_body["grounding"]["retrieval_status"] == "sufficient"
    assert hard_body["grounding"]["failure_status"] is None
    assert hard_body["grounding"]["sources"]
    assert hard_body["generation"] is not None
    assert judge.calls >= 2
    assert "annual inundation" in hard_body["explanation"].casefold()

    quantum = client.post("/learning/explain", json={
        "topic": "quantum teleportation", "question": "How does quantum teleportation work?",
    })
    assert quantum.status_code == 200
    quantum_body = quantum.json()
    assert quantum_body["grounding"]["source_mode"] == "insufficient_context"
    assert quantum_body["grounding"]["retrieval_status"] == "insufficient"
    assert quantum_body["grounding"]["sources"] == []
    assert quantum_body["generation"] is None


def test_all_learning_endpoints_supported_unsupported_and_invalid(monkeypatch):
    """Exercise the shared production retrieval path without external LLM calls."""
    import json
    import app.main as main

    fact = "Ancient Egyptian civilization developed in northeastern Africa in the 3rd millennium bc."

    class EvidenceProvider:
        name, model = "benchmark-stub", "offline-api-contract"
        def generate(self, system_prompt, user_prompt="", response_schema=None, **kwargs):
            keys = set((response_schema or {}).get("properties", {}))
            if "explanation" in keys and "key_concepts" in keys:
                result = {"explanation": fact, "key_concepts": [], "examples": [],
                          "important_points": [], "common_mistakes": [], "recommended_revision": []}
            elif "summary" in keys:
                result = {"summary": fact, "key_points": []}
            elif "questions" in keys:
                item = {"question": "Where did Ancient Egyptian civilization develop?", "options": [fact, "Rome", "Athens", "Babylon"],
                        "correct_answer": fact, "explanation": fact, "difficulty": "medium",
                        "topic": "Ancient Egyptian civilization", "learning_objective": "Recall a source fact"}
                result = {"title": "Ancient Egypt"} if "title" in keys else {}
                result["questions"] = [item]
            elif "cards" in keys:
                result = {"cards": [{"front": "Where did Ancient Egyptian civilization develop?", "back": fact,
                                     "topic": "Ancient Egyptian civilization", "difficulty": "medium", "concept": "Origins"}]}
            elif "correct" in keys:
                result = {"correct": True, "score": 1.0, "correct_answer": fact, "explanation": fact,
                          "why_answer_is_wrong": None, "related_concept": "Ancient Egyptian civilization",
                          "recommended_revision": [], "confidence": 1.0, "identified_misconception": None}
            else:
                raise AssertionError(f"Unexpected response schema properties: {keys}")
            return json.dumps(result)

    monkeypatch.setattr(main.generator, "llm", EvidenceProvider())
    monkeypatch.setattr(main.generator, "claim_judge_llm", AcceptingJudge())
    client = TestClient(main.app)
    endpoints = {
        "/learning/explain": {"topic": "Ancient Egyptian civilization", "question": "Where did Ancient Egyptian civilization develop?"},
        "/learning/summarize": {"topic": "Ancient Egyptian civilization"},
        "/learning/mcq": {"topic": "Ancient Egyptian civilization", "number_of_questions": 1},
        "/learning/flashcards": {"topic": "Ancient Egyptian civilization", "number_of_cards": 1},
        "/learning/exam": {"topics": ["Ancient Egyptian civilization"], "number_of_questions": 1},
        "/learning/evaluate-answer": {"question": "Where did Ancient Egyptian civilization develop?", "student_answer": fact,
                                        "correct_answer": fact, "topic": "Ancient Egyptian civilization"},
    }
    assert client.get("/health").status_code == 200
    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json()["document_chunks"] == 661
    for path, payload in endpoints.items():
        supported = client.post(path, json=payload)
        assert supported.status_code == 200, (path, supported.text)
        supported_body = supported.json()
        grounding = supported_body["grounding"]
        assert grounding["retrieval_status"] == "sufficient" and grounding["grounding_status"] == "grounded"
        assert grounding["retrieval_method"] == "hybrid_cross_encoder"
        assert grounding["sources"]
        assert supported_body["generation"]["model"] == "offline-api-contract"

        unsupported = {**payload, "topic": "quantum entanglement in superconducting qubits"}
        if "topics" in unsupported:
            unsupported["topics"] = ["quantum entanglement in superconducting qubits"]
        if "question" in unsupported:
            unsupported["question"] = "Explain quantum entanglement in superconducting qubits."
        if "student_answer" in unsupported:
            unsupported["student_answer"] = "A state description with entanglement."
            unsupported.pop("correct_answer", None)
        rejected = client.post(path, json=unsupported)
        assert rejected.status_code == 200, (path, rejected.text)
        rejected_body = rejected.json()
        assert rejected_body["grounding"]["retrieval_status"] == "insufficient"
        assert rejected_body["grounding"]["grounding_status"] == "not_grounded"
        assert rejected_body["grounding"]["sources"] == []
        assert rejected_body["generation"] is None

        invalid = client.post(path, json={})
        assert invalid.status_code == 422, (path, invalid.text)
