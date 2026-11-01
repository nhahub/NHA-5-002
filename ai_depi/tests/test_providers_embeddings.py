import json

import pytest

from app.config import Settings
from app.providers import (
    FallbackLLMProvider, ProviderError, create_llm_provider,
    generate_structured, _load_sentence_transformer,
)
from app.schemas import ExplanationResponse


class StubProvider:
    def __init__(self, name, model, output="{}", fail=False):
        self.name, self.model, self.output, self.fail, self.calls = name, model, output, fail, 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        if self.fail:
            raise ProviderError(f"{self.name} unavailable")
        return self.output


def test_primary_success_skips_fallback_and_reports_provider():
    primary = StubProvider("groq", "cloud-model")
    fallback = StubProvider("ollama", "local-model")
    provider = FallbackLLMProvider(primary, fallback)
    provider.generate("system", "user")
    assert (primary.calls, fallback.calls) == (1, 0)
    assert provider.call_metadata.provider == "groq"
    assert provider.call_metadata.model == "cloud-model"
    assert not provider.call_metadata.fallback_used


def test_primary_failure_uses_fallback_and_reports_actual_provider():
    # Ensures response metadata describes the provider that actually returned output.
    provider = FallbackLLMProvider(StubProvider("groq", "cloud", fail=True), StubProvider("ollama", "local"))
    provider.generate("system", "user")
    assert provider.call_metadata.provider == "ollama"
    assert provider.call_metadata.model == "local"
    assert provider.call_metadata.fallback_used


def test_fallback_can_use_local_output_budget_without_capping_primary():
    class BudgetProvider(StubProvider):
        def generate(self, *args, **kwargs):
            self.kwargs = kwargs
            return super().generate(*args, **kwargs)
    local = BudgetProvider("ollama", "local")
    provider = FallbackLLMProvider(StubProvider("groq", "cloud", fail=True), local)
    provider.generate("system", "user", max_tokens=1600, fallback_max_tokens=700)
    assert local.kwargs["max_tokens"] == 700
    assert provider.call_metadata.fallback_used


def test_both_providers_fail_without_deterministic_success():
    provider = FallbackLLMProvider(StubProvider("groq", "cloud", fail=True), StubProvider("ollama", "local", fail=True))
    with pytest.raises(ProviderError, match="failed"):
        provider.generate("system", "user")


def test_deterministic_is_only_selected_explicitly():
    settings = Settings(llm_provider="groq", llm_fallback_provider="ollama", _env_file=None)
    assert create_llm_provider(settings).name == "groq"
    explicit = create_llm_provider(Settings(llm_provider="deterministic", _env_file=None))
    assert explicit.name == "deterministic"


def test_structured_response_metadata_matches_successful_fallback():
    payload = {"explanation":"ok", "key_concepts":[], "examples":[], "important_points":[], "common_mistakes":[], "recommended_revision":[],
        "grounding":{"source_mode":"retrieved_learning_content","grounding_status":"grounded","retrieval_status":"sufficient","sources":[]},
        "adaptation":{"requested_settings":{},"adapted_settings":{},"adaptation_reasons":[]}}
    provider = FallbackLLMProvider(StubProvider("groq", "cloud", fail=True), StubProvider("ollama", "local", json.dumps(payload)))
    result = generate_structured(provider, "system", "user", ExplanationResponse)
    assert result.generation.model_dump() == {"provider":"ollama", "model":"local", "fallback_used":True}


def test_sentence_transformer_cached_load_batch_dimension_and_stability(monkeypatch):
    # Protects model reuse, batch encoding, vector dimension, and normalized output setup.
    import numpy as np
    calls = []
    class Model:
        def get_sentence_embedding_dimension(self): return 3
        def encode(self, texts, **kwargs):
            calls.append((list(texts), kwargs))
            return np.array([[len(text), 1, 0] for text in texts], dtype=float)
    _load_sentence_transformer.cache_clear()
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", lambda name: Model())
    from app.providers import SentenceTransformerEmbeddingProvider
    first = SentenceTransformerEmbeddingProvider("test-model")
    second = SentenceTransformerEmbeddingProvider("test-model")
    vectors = first.embed(["a", "long"])
    assert first.dimension == 3
    assert vectors == second.embed(["a", "long"])
    assert len(calls) == 2 and calls[0][0] == ["a", "long"]
    assert calls[0][1]["normalize_embeddings"] is True
    _load_sentence_transformer.cache_clear()
