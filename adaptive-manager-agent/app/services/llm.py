import httpx
from app.core.config import settings


def generate_behavior_insight(metrics: dict, patterns: list[str], adaptive_model: dict, fallback: str) -> str:
    """Optional OpenAI-compatible LLM enrichment. Falls back safely when not configured/available."""
    if not settings.llm_api_key:
        return fallback
    prompt = (
        "You are an evidence-based student behavior analyst. Write a concise 2-3 sentence insight. "
        "Use only the supplied aggregate metrics; do not diagnose, speculate, or invent facts. "
        "Mention uncertainty when data is limited.\n"
        f"Patterns: {patterns}\nMetrics: {metrics}\nAdaptive model: {adaptive_model}"
    )
    try:
        response = httpx.post(
            f"{settings.llm_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            json={"model": settings.llm_model, "temperature": 0.2,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=20.0,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        return content or fallback
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        return fallback
