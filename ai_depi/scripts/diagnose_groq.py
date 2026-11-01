"""Make one minimal Groq request and print only safe provider diagnostics."""
import sys

from app.config import get_settings
from app.providers import GroqProvider, ProviderError


def main() -> int:
    settings = get_settings()
    provider = GroqProvider(settings)
    endpoint = "https://api.groq.com/openai/v1/chat/completions"
    print({"provider": provider.name, "endpoint": endpoint, "model": provider.model,
           "api_key_configured": bool(provider.api_key)})
    try:
        response = provider.generate(
            "Return valid JSON only.",
            "Return exactly one JSON object with the boolean field ok set to true.",
            {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False},
            max_tokens=256,
        )
        print({"result": "success", "response_chars": len(response)})
        return 0
    except ProviderError as exc:
        print({"result": "failed", "safe_diagnostic": str(exc)}, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
