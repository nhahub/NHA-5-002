import hashlib, json, math, re, time, logging, threading
from dataclasses import dataclass
from functools import lru_cache
from urllib import request
from urllib.error import HTTPError, URLError
from .config import Settings
from .domain import LLMProvider
from .validation import OutputValidationError

log = logging.getLogger("learning.providers")

# Provider adapters isolate transport details from generation and attach the
# actual successful provider metadata to each structured response.

class ProviderError(RuntimeError): pass

@dataclass(frozen=True)
class ProviderMetadata:
    provider: str
    model: str
    fallback_used: bool = False

class DeterministicProvider:
    name = "deterministic"; model = "offline-test-double"
    def generate(self, system_prompt: str, user_prompt: str = "", response_schema=None, *, temperature: float = 0.2, max_tokens=None) -> str:
        return json.dumps({"text": "Offline provider output. Use Ollama or Groq for model-generated content."})

def _schema_prompt(system_prompt: str, user_prompt: str, response_schema: dict | None) -> str:
    schema = json.dumps(response_schema or {}, separators=(",", ":"))
    return f"{system_prompt}\n\n{user_prompt}\n\nReturn ONLY valid JSON matching this schema:\n{schema}"

class OllamaProvider:
    name = "ollama"
    def __init__(self, settings: Settings): self.base_url, self.model, self.timeout = settings.ollama_base_url.rstrip("/"), settings.ollama_model, settings.request_timeout
    def generate(self, system_prompt: str, user_prompt: str = "", response_schema=None, *, temperature: float = 0.2, max_tokens=None) -> str:
        body = {"model": self.model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "stream": False, "format": response_schema or "json", "options": {"temperature": temperature}}
        if max_tokens: body["options"]["num_predict"] = max_tokens
        try:
            req = request.Request(self.base_url + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read())
                content = payload.get("message", {}).get("content")
                if not isinstance(content, str): raise ProviderError("Ollama returned an invalid response structure")
                return content
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise ProviderError(f"Ollama request failed ({type(exc).__name__})") from exc

class GroqProvider:
    name = "groq"
    def __init__(self, settings: Settings):
        self.api_key, self.model, self.timeout = settings.groq_api_key, settings.groq_model, settings.request_timeout

    def _safe_http_error(self, exc: HTTPError) -> str:
        """Extract only bounded, redacted API error fields; never log response headers."""
        raw = b""
        content_type = "unknown"
        if exc.headers:
            content_type = exc.headers.get_content_type() or "unknown"
        detail = f"HTTP {exc.code}; content_type={content_type}"
        if exc.headers:
            for header in ("server", "cf-error-type", "cf-error-origin"):
                value = exc.headers.get(header)
                if value:
                    detail += f"; {header}={value}"
        try:
            raw = exc.read()
            payload = json.loads(raw.decode("utf-8", errors="replace"))
            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            if isinstance(error, dict):
                fields = [f"{key}={error[key]}" for key in ("type", "code", "message") if error.get(key)]
                if fields:
                    detail += "; " + "; ".join(fields)
                else:
                    fields = [f"{key}={payload[key]}" for key in ("type", "code", "message", "detail") if payload.get(key)]
                    if fields:
                        detail += "; " + "; ".join(fields)
            elif isinstance(error, str) and error:
                detail += f"; message={error}"
        except (ValueError, OSError):
            text_body = raw.decode("utf-8", errors="replace").strip()
            if text_body:
                detail += f"; response={text_body[:300]}"
            else:
                detail += "; response_body=empty"
        if self.api_key:
            detail = detail.replace(self.api_key, "[REDACTED]")
        detail = re.sub(r"(?i)\bBearer\s+[^\s,;]+", "Bearer [REDACTED]", detail)
        detail = re.sub(r"(?i)\b(?:gsk|sk)-[A-Za-z0-9_-]{8,}", "[REDACTED]", detail)
        return detail[:500]

    def generate(self, system_prompt: str, user_prompt: str = "", response_schema=None, *, temperature: float = 0.2, max_tokens=None) -> str:
        if not self.api_key: raise ProviderError("GROQ_API_KEY is not configured")
        response_format = {"type": "json_object"}
        if isinstance(response_schema, dict) and response_schema:
            name = re.sub(r"[^A-Za-z0-9_-]", "_", response_schema.get("title", "learning_response"))[:64] or "learning_response"
            response_format = {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": response_schema}}
        body = {"model": self.model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": temperature, "response_format": response_format}
        if self.model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}:
            body["reasoning_effort"] = "low"
        if max_tokens: body["max_completion_tokens"] = max_tokens
        try:
            req = request.Request("https://api.groq.com/openai/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "generative-learning-model/0.1.0", "Authorization": f"Bearer {self.api_key}"})
            with request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read())
                content = payload["choices"][0]["message"]["content"]
                if not isinstance(content, str): raise ProviderError("Groq returned an invalid response structure")
                return content
        except HTTPError as exc:
            raise ProviderError(f"Groq request failed ({self._safe_http_error(exc)})") from exc
        except (URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Groq request failed ({type(exc).__name__})") from exc

class FallbackLLMProvider:
    """Try the configured primary and then fallback for provider failures only."""
    def __init__(self, primary: LLMProvider, fallback: LLMProvider | None):
        self.primary, self.fallback = primary, fallback
        self._call_state = threading.local()
        self.name, self.model = primary.name, primary.model
        log.info("llm_provider_config primary=%s fallback=%s", primary.name, fallback.name if fallback else "none")

    @property
    def call_metadata(self) -> ProviderMetadata:
        return getattr(self._call_state, "metadata", ProviderMetadata(self.primary.name, self.primary.model, False))

    def generate(self, system_prompt: str, user_prompt: str = "", response_schema=None, *, temperature: float = 0.2, max_tokens=None, fallback_max_tokens=None) -> str:
        # Only transport/provider failures trigger fallback; malformed model output is retried by validation.
        try:
            output = self.primary.generate(system_prompt, user_prompt, response_schema, temperature=temperature, max_tokens=max_tokens)
            self._call_state.metadata = ProviderMetadata(self.primary.name, self.primary.model, False)
            log.info("llm_generation_succeeded provider=%s model=%s fallback_used=false", self.primary.name, self.primary.model)
            return output
        except ProviderError as primary_error:
            log.warning("primary_provider_failed provider=%s failure=%s", self.primary.name, str(primary_error))
            if self.fallback is None:
                self._call_state.metadata = ProviderMetadata(self.primary.name, self.primary.model, False)
                raise ProviderError(f"Primary provider {self.primary.name} failed and no fallback is configured") from primary_error
            log.info("fallback_provider_selected provider=%s", self.fallback.name)
            try:
                output = self.fallback.generate(system_prompt, user_prompt, response_schema, temperature=temperature, max_tokens=fallback_max_tokens or max_tokens)
                self._call_state.metadata = ProviderMetadata(self.fallback.name, self.fallback.model, True)
                log.info("fallback_provider_succeeded provider=%s", self.fallback.name)
                log.info("llm_generation_succeeded provider=%s model=%s fallback_used=true", self.fallback.name, self.fallback.model)
                return output
            except ProviderError as fallback_error:
                self._call_state.metadata = ProviderMetadata(self.fallback.name, self.fallback.model, True)
                log.error("both_providers_failed primary=%s fallback=%s", self.primary.name, self.fallback.name)
                raise ProviderError(f"Primary provider {self.primary.name} failed ({primary_error}); fallback provider {self.fallback.name} failed ({fallback_error})") from fallback_error

def _compact_provider_schema(schema, response_schema=None, expected_counts=None):
    """Return a small provider-facing schema; trusted API metadata is stamped locally."""
    result = response_schema or schema.model_json_schema()
    definitions = result.pop("$defs", {})
    # These fields come from trusted request/context data and are stamped locally after generation.
    server_fields = {"grounding", "adaptation", "generation", "source", "supporting_source_or_evidence", "groundedness", "time_limit_minutes"}

    def transform(node):
        if isinstance(node, dict):
            ref = node.get("$ref")
            if ref and ref.startswith("#/$defs/"):
                definition = definitions.get(ref.rsplit("/", 1)[-1], {})
                return transform(definition)
            result = {}
            for key, value in node.items():
                if key in {"title", "default", "$defs", "required", "additionalProperties"}:
                    continue
                if key == "properties" and isinstance(value, dict):
                    properties = {name: transform(child) for name, child in value.items() if name not in server_fields}
                    result["properties"] = properties
                    result["required"] = list(properties)
                    result["additionalProperties"] = False
                else:
                    result[key] = transform(value) if isinstance(value, (dict, list)) else value
            return result
        if isinstance(node, list):
            return [transform(value) for value in node]
        return node

    result = transform(result)
    for field, count in (expected_counts or {}).items():
        if field in result.get("properties", {}):
            result["properties"][field]["minItems"] = count
            result["properties"][field]["maxItems"] = count
    return result

def _generate_validated(provider, system_prompt, user_prompt, schema, provider_schema, temperature, max_tokens, retries, trusted_fields=None, expected_counts=None, fallback_max_tokens=None, allow_fenced_json=False):
    last = None
    # Provider JSON is not returned until counts and the full Pydantic response schema pass.
    for attempt in range(retries + 1):
        try:
            call_options = {"temperature": temperature, "max_tokens": max_tokens}
            if isinstance(provider, FallbackLLMProvider):
                call_options["fallback_max_tokens"] = fallback_max_tokens
            raw = provider.generate(system_prompt, user_prompt, provider_schema, **call_options)
            if not isinstance(raw, str): raise OutputValidationError("Provider returned non-text output")
            json_text = raw
            if allow_fenced_json:
                stripped = raw.strip()
                if stripped.startswith("```") and stripped.endswith("```"):
                    first_newline = stripped.find("\n")
                    if first_newline >= 0 and stripped[:first_newline].strip().lower() in {"```", "```json"}:
                        json_text = stripped[first_newline + 1:-3].strip()
            value = json.loads(json_text)
            if not isinstance(value, dict): raise OutputValidationError("Provider returned a non-object response")
            value.update(trusted_fields or {})
            for field, expected in (expected_counts or {}).items():
                actual = value.get(field)
                if not isinstance(actual, list) or len(actual) != expected:
                    raise OutputValidationError(f"{field} must contain exactly {expected} items")
            result = schema.model_validate(value)
            if hasattr(result, "generation"):
                metadata = getattr(provider, "call_metadata", ProviderMetadata(provider.name, provider.model, False))
                from .schemas import GenerationMetadata
                result.generation = GenerationMetadata(provider=metadata.provider, model=metadata.model, fallback_used=metadata.fallback_used)
            return result
        except (json.JSONDecodeError, OutputValidationError, ValueError) as exc:
            last = exc
            if attempt < retries:
                user_prompt += "\nCORRECTION: The previous response failed validation. Return one complete JSON object, include every required field, and obey the exact requested array counts. Keep field values concise."
                time.sleep(0.05 * (attempt + 1))
            else:
                if hasattr(exc, "errors"):
                    failures = [{"path": ".".join(map(str, error.get("loc", ()))), "type": error.get("type")} for error in exc.errors()]
                else:
                    failures = {"type": type(exc).__name__, "detail": str(exc)[:120]}
                logging.getLogger("learning.providers").warning("structured_output_rejected schema=%s failures=%s", schema.__name__, failures)
        except ProviderError as exc:
            raise
    raise OutputValidationError(f"Provider output did not match {schema.__name__}: {last}")

def generate_structured(provider: LLMProvider, system_prompt: str, user_prompt: str, schema, *, temperature=0.2, max_tokens=2000, retries=2, expected_counts=None, trusted_fields=None, fallback_max_tokens=None, allow_fenced_json=False):
    provider_schema = _compact_provider_schema(schema, expected_counts=expected_counts)
    return _generate_validated(provider, system_prompt, user_prompt, schema, provider_schema, temperature, max_tokens, retries, trusted_fields, expected_counts, fallback_max_tokens, allow_fenced_json)

def _provider_by_name(name: str, settings: Settings) -> LLMProvider:
    if name == "ollama": return OllamaProvider(settings)
    if name == "groq": return GroqProvider(settings)
    if name == "deterministic": return DeterministicProvider()
    raise ValueError(f"Unsupported LLM provider: {name}")

def create_llm_provider(settings: Settings) -> LLMProvider:
    primary = _provider_by_name(settings.llm_provider, settings)
    fallback_name = settings.llm_fallback_provider
    fallback = _provider_by_name(fallback_name, settings) if fallback_name and fallback_name != settings.llm_provider and settings.llm_provider != "deterministic" else None
    if settings.llm_provider == "deterministic": return primary
    return FallbackLLMProvider(primary, fallback)

@lru_cache(maxsize=2)
def _load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

class SentenceTransformerEmbeddingProvider:
    """Cached local semantic encoder shared by ingestion and retrieval."""
    version_prefix = "sentence-transformers"
    def __init__(self, model_name: str):
        self.provider_name = "sentence_transformers"
        self.model_name = model_name
        self.model = _load_sentence_transformer(model_name)
        dimension_method = getattr(self.model, "get_embedding_dimension", self.model.get_sentence_embedding_dimension)
        self.dimension = int(dimension_method())
        self.version = f"{self.version_prefix}:{model_name}"
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts: return []
        # Unit-length vectors make cosine comparisons consistent across text lengths.
        vectors = self.model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return vectors.tolist()

def create_embedding_provider(settings: Settings):
    if settings.embedding_provider == "sentence_transformers": return SentenceTransformerEmbeddingProvider(settings.embedding_model)
    if settings.embedding_provider == "hash": return HashEmbeddingProvider()
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")

class HashEmbeddingProvider:
    """Small deterministic lexical feature embedding, not a semantic model."""
    dimension = 32
    provider_name = "hash"
    model_name = "hash-token-v1"
    version = "hash-token-v1"
    def embed(self, texts: list[str]) -> list[list[float]]:
        # Hash tokens and adjacent token pairs into fixed-width vectors; this preserves
        # lexical overlap deterministically but does not encode semantic similarity.
        vectors = []
        for text in texts:
            tokens = re.findall(r"[a-z0-9]+", text.lower())
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            vector = [0.0] * self.dimension
            for feature in features:
                digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[index] += sign
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector] if norm else vector)
        return vectors
