import json
from app.config import Settings
from app.domain import RetrievedChunk
from app.generation import LearningGenerator
from app.providers import OllamaProvider, GroqProvider, generate_structured, ProviderError
from app.rag import InMemoryRetriever
from app.schemas import ExplanationRequest, ExplanationResponse, GroundingMetadata, AdaptationMetadata

class FakeProvider:
    name = "fake"; model = "fake-model"
    def __init__(self, payload): self.payload, self.calls = payload, 0
    def generate(self, system_prompt, user_prompt="", response_schema=None, **kwargs): self.calls += 1; return json.dumps(self.payload)

class AcceptingJudge:
    name, model = "ollama", "qwen2.5:7b-instruct-test"
    def generate(self, system_prompt, user_prompt="", response_schema=None, **kwargs):
        return json.dumps({"supported": True, "confidence": .96, "reason": "Supported by the test evidence."})

def explanation_payload():
    return {"explanation":"CNN extracts local features", "key_concepts":["CNN"], "examples":[], "important_points":[], "common_mistakes":[], "recommended_revision":[], "grounding":{"source_mode":"retrieved_learning_content","grounding_status":"grounded","retrieval_status":"sufficient","sources":[],"notice":None}, "adaptation":{"requested_settings":{},"adapted_settings":{},"adaptation_reasons":[]}}

def test_real_generation_path_uses_schema_aware_provider():
    fake = FakeProvider(explanation_payload())
    gen = LearningGenerator(InMemoryRetriever([RetrievedChunk("CNN extracts local features", "notes", 1, "c1", relevance_score=.8)]), llm=fake, settings=Settings(), claim_judge_llm=AcceptingJudge())
    result = gen.explain(ExplanationRequest(topic="CNN"))
    assert result.explanation == "CNN extracts local features" and fake.calls == 1

def test_invalid_structured_output_retries_and_fails():
    fake = FakeProvider({"bad": True})
    try: generate_structured(fake, "system", "user", ExplanationResponse, retries=1)
    except Exception as exc: assert "did not match" in str(exc)
    else: raise AssertionError("expected validation failure")

def test_ollama_payload_is_structured(monkeypatch):
    seen = {}
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps({"message":{"content":"{}"}}).encode()
    def fake_urlopen(req, timeout): seen.update({"url": req.full_url, "body": json.loads(req.data), "headers": dict(req.header_items())}); return Response()
    monkeypatch.setattr("app.providers.request.urlopen", fake_urlopen)
    OllamaProvider(Settings()).generate("system", "user", {"type":"object"})
    assert seen["url"].endswith("/api/chat") and seen["body"]["format"] == {"type":"object"}

def test_groq_payload_uses_model_supported_json_schema_mode(monkeypatch):
    seen = {}
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps({"choices":[{"message":{"content":"{}"}}]}).encode()
    def fake_urlopen(req, timeout): seen.update({"url": req.full_url, "body": json.loads(req.data), "headers": dict(req.header_items())}); return Response()
    monkeypatch.setattr("app.providers.request.urlopen", fake_urlopen)
    GroqProvider(Settings(groq_api_key="test-key", groq_model="openai/gpt-oss-120b")).generate("system", "user", {"type":"object"}, max_tokens=2000)
    assert seen["body"]["response_format"]["type"] == "json_schema"
    assert seen["body"]["response_format"]["json_schema"]["schema"] == {"type":"object"}
    assert seen["body"]["response_format"]["json_schema"]["strict"] is True
    assert seen["body"]["reasoning_effort"] == "low"
    assert seen["body"]["max_completion_tokens"] == 2000
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["body"]["model"] == "openai/gpt-oss-120b"
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert seen["headers"]["Accept"] == "application/json"
    assert seen["headers"]["User-agent"] == "generative-learning-model/0.1.0"

def test_groq_http_error_reports_safe_json_diagnostic(monkeypatch):
    from io import BytesIO
    from urllib.error import HTTPError
    from app.providers import ProviderError
    def denied(req, timeout):
        raise HTTPError(req.full_url, 403, "Forbidden", {}, BytesIO(b'{"error":{"type":"model_permission_error","code":"model_not_allowed","message":"model blocked at organization level"}}'))
    monkeypatch.setattr("app.providers.request.urlopen", denied)
    provider = GroqProvider(Settings(groq_api_key="gsk-secret-example", groq_model="openai/gpt-oss-120b"))
    try:
        provider.generate("system", "user", {"type":"object"})
    except ProviderError as exc:
        message = str(exc)
        assert "HTTP 403" in message and "model_permission_error" in message
        assert "model blocked at organization level" in message
        assert "gsk-secret-example" not in message and "Bearer" not in message
    else:
        raise AssertionError("expected Groq 403 diagnostic")
