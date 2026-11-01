import sys, time
from app.config import get_settings
from app.providers import create_llm_provider, ProviderError
from app.generation import LearningGenerator
from app.rag import InMemoryRetriever
from app.schemas import ExplanationRequest

def main():
    settings = get_settings()
    if settings.llm_provider == "deterministic":
        print("LLM_PROVIDER=deterministic; refusing to claim a real provider smoke test")
        return 2
    try:
        provider = create_llm_provider(settings)
        started = time.perf_counter()
        result = LearningGenerator(InMemoryRetriever(), settings.max_context_chars, provider, settings).explain(ExplanationRequest(topic="machine learning", knowledge_mode="learning_materials_then_general_knowledge"))
        print({"provider": result.generation.provider, "model": result.generation.model, "fallback_used": result.generation.fallback_used, "grounding": result.grounding.model_dump(), "latency_ms": round((time.perf_counter()-started)*1000, 1)})
        return 0
    except Exception as exc:
        print(f"LLM smoke test failed: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__": raise SystemExit(main())
