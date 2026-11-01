# Models and Technologies

This inventory is based on current source/configuration and saved artifacts. “Configured” means a setting is present; “loaded/used” means code reaches the implementation. No model is trained or fine-tuned in this project.

## 1. Models

| Model/config name | Type and task | Loaded/used by | Input → output | Where it runs and device | Fallback, resources, status |
|---|---|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | Sentence embedding encoder for semantic search | SentenceTransformerEmbeddingProvider in app/providers.py; called by PDF indexing and ChromaRetriever query path | Text batch → normalized float vectors | Local via Sentence Transformers. App does not pass a device parameter; library decides. The dimension is read from loaded model; saved benchmark report records 384. | Cached with LRU cache (up to two model names); encode batch size 32. No fallback to another semantic model. Model files are downloaded/cached by library. No project training code. |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | Cross-Encoder relevance reranker | Lazy loader in app/rag.py, called after RRF | List of (query, passage) pairs → raw relevance logits | Local; max length 512; requested device defaults auto, selecting CUDA if torch reports available and CPU otherwise. Saved latest benchmark reports CPU. | Lazy loaded/cached (up to four model/device combinations), prediction batch size 16, reranks max 30. Old benchmark report recorded about 90.9 MB weights/183.6 MB total cache in one prior environment; current runtime RSS is not measured. No training code. |
| qwen2.5:7b-instruct | General local instruct LLM; normal-generation fallback and Semantic Claim Judge | OllamaProvider from app/providers.py; judge constructed separately by LearningGenerator | System/user messages and optional JSON Schema → message text containing JSON | Local Ollama HTTP service at configured base URL, default http://localhost:11434. Application does not select CPU/GPU; Ollama controls execution. | Fallback for ordinary generation only on primary ProviderError. Judge calls Ollama directly and fails closed; no Groq judge fallback. Config timeout defaults 60 seconds. No fine-tuning. |
| openai/gpt-oss-120b | Hosted instruct/reasoning LLM used for normal educational generation | GroqProvider in app/providers.py through Groq chat-completions API | System/user messages plus strict JSON Schema → model content string | Remote Groq service; model is not loaded in this process. | Default primary, no local weights/resource requirements in this app. Groq API failures invoke configured Ollama fallback. For GPT-OSS IDs, request includes reasoning_effort=low. Inference only here. |
| deterministic / offline-test-double | Non-generative provider stub | DeterministicProvider, only selected when explicitly configured | No meaningful model inference; returns a fixed JSON text | Local process | Not automatic fallback. Normal production generation path recognizes this provider and uses local deterministic branches/empty sets; it is not a substitute LLM. |
| hash-token-v1 | Fixed 32-dimensional token/bigram hash embedding utility | HashEmbeddingProvider, selectable with EMBEDDING_PROVIDER=hash; used by tests | Alphanumeric tokens and adjacent pairs → signed hash features, normalized vector | Local Python | No semantic model or resource download. It is an optional test/utility embedding, not the configured default. |

### Model roles in plain language

- **Embedding model:** turns each passage and query into a vector separately so Chroma can quickly find semantically nearby passages.
- **Cross-Encoder:** takes one query and one passage together, then scores their relationship. It runs only on the bounded RRF candidate set.
- **LLM:** writes the requested learning output as JSON fields.
- **Semantic Claim Judge:** a separate Ollama-only constrained LLM call that checks extracted generated factual claims against the final evidence packet after structured generation. Claims pass only when `supported=true` and confidence ≥0.80; provider and schema failures fail closed.

These roles are different. The embedding score is a retrieval similarity; the Cross-Encoder score is a pairwise relevance logit; an LLM produces content; the judge emits a supported/confidence/reason decision.

## 2. Provider behavior

### Normal generation

Settings defaults and local .env configure Groq primary and Ollama fallback. GroqProvider sends requests to https://api.groq.com/openai/v1/chat/completions with bearer authorization and a 60-second default timeout. The code uses response_format type json_schema and strict=true when it has a provider schema. It sets low reasoning effort for the configured GPT-OSS model IDs.

OllamaProvider posts to /api/chat, stream=false, with the schema dictionary in format when provided. It passes configured temperature and num_predict when token budget is supplied. Timeout defaults to 60 seconds.

FallbackLLMProvider catches ProviderError from the primary only. A successful primary returns fallback_used=false. Primary failure invokes the fallback with fallback-specific token budget when provided; returned metadata names the provider/model that actually generated the output. If fallback fails, provider error reaches the API exception handler.

Malformed JSON, missing fields, schema errors or count errors do not by themselves trigger fallback. Structured generation retries the chain up to three attempts total. Each attempt starts with the primary again. This distinction matters: “fallback” means provider failure, not content validation failure.

### Per-task output budgets

These are code-level call arguments in LearningGenerator._real, not separate environment settings:

| Task/schema | Primary max tokens | Ollama fallback max tokens |
|---|---:|---:|
| MCQ set / practice exam | min(16,000, max(3,000, requested question count × 180)) | min(8,000, max(1,200, requested question count × 120)) |
| Flashcard set | min(12,000, max(2,200, requested card count × 120)) | min(6,000, max(900, requested card count × 80)) |
| Answer evaluation | 1,800 | 900 |
| Explanation brief / standard / high | 800 / 1,400 / 2,400 | 500 / 900 / 1,400 |
| Other/default structured outputs | 2,000 | 1,200 |
| Semantic Claim Judge | 120 | No provider fallback; direct Ollama call |

The configured generation temperature defaults to 0.2; Semantic Claim Judge overrides it to 0.0. Provider request timeout defaults to 60 seconds. MAX_RETRIES defaults to two additional structured-output attempts.

### Semantic Claim Judge

LearningGenerator constructs its judge as an OllamaProvider, separate from the normal generation wrapper. It rejects a judge whose provider name is not ollama. It calls generate_structured directly at temperature 0 and max_tokens 120, allows surrounding fenced JSON to be stripped, then applies the configured minimum confidence of 0.80. Provider errors and exhausted schema retries are caught and treated as a rejected claim. It never uses Groq.

## 3. Technologies actually used

| Technology | What it is | Role here | Why this implementation uses it | Alternative examples |
|---|---|---|---|---|
| Python 3.11+ | Runtime language | All application, tests and scripts | pyproject requires >=3.11 | Other server-side languages; no project decision record. |
| FastAPI | Python HTTP framework | Typed routes, request handling, generated OpenAPI docs, middleware/lifespan | Routes declare Pydantic models and response models | Flask or another ASGI framework. |
| Pydantic 2 | Data validation/model library | Request/response schemas and structured LLM output | Bounds and enumerations become executable contracts | Dataclasses alone or another validation library. |
| pydantic-settings | Settings/environment parsing | Loads defaults, .env and environment overrides; settings object cached | Centralizes provider/model/search configuration | Direct environment parsing. |
| Uvicorn | ASGI server | Runs app.main:app, default port 8000 | Container and module startup use it | Another ASGI server. |
| ChromaDB | Vector database | Persistent local vectors/documents/metadata, cosine HNSW collection | Provides persisted dense retrieval and page metadata | FAISS, Qdrant, pgvector, etc.; repository contains no historical comparison. |
| Sentence Transformers | Model library | Loads SentenceTransformer and CrossEncoder, vector encoding and pairwise scoring | Both models and cached inference are in one integration | Other embedding/reranking stacks; no benchmarked alternative is configured. |
| PyTorch | ML tensor/runtime library, transitive through model stack | Underlies local model inference/device reporting | Required by model dependency stack | Other ML runtimes not configured. |
| pypdf | PDF parsing package | Extracts text page by page | Ingestion code imports PdfReader directly | Other PDF text extractors are not configured. |
| Custom BM25 | In-project lexical ranker | Builds token stats over Chroma chunk snapshot and searches per query | Avoids an external sparse database in this local service; README says snapshot avoids a second persistent index | A search engine or another BM25 library. |
| Custom RRF | In-project fusion function | Combines rank positions from dense and sparse retrieval | Avoids comparing incompatible raw score scales | Weighted fusion/other rank fusion. |
| Ollama | Local model-serving application/API | Qwen fallback and isolated judge | App calls its HTTP /api/chat endpoint | Another local server or hosted provider. |
| Groq API | Hosted LLM service | Primary generation model | Configured provider adapter sends chat-completions requests | Local-only or another hosted API. Historical rationale is not recorded. |
| httpx | HTTP client dependency | Present as declared dependency, notably used by FastAPI testing stack; providers themselves use urllib.request | Dependency declaration | Not used by provider adapter for model calls. |
| python-dotenv | dotenv support | Declared dependency used with pydantic settings | Allows local .env configuration | OS environment only. |
| pytest / pytest-asyncio | Test tooling | Optional dev dependencies, pytest configured to search tests/ | Supports test suite and async tests | Other test frameworks. |
| NumPy | Numeric arrays | Sentence Transformers returns NumPy vectors and one embedding test constructs arrays; vectors are converted to lists before Chroma | Present through the ML dependency stack, but not directly declared in pyproject.toml | Pure Python arrays would be less convenient. |

The custom BM25 tokenizer is ASCII alphanumeric regex after casefolding. It then applies small suffix rules. There is no third-party BM25 package in pyproject.

## 4. Configuration actually present

### Local .env values inspected (secret redacted)

| Setting | Local value |
|---|---|
| LLM_PROVIDER | groq |
| LLM_FALLBACK_PROVIDER | ollama |
| OLLAMA_BASE_URL | http://localhost:11434 |
| OLLAMA_MODEL | qwen2.5:7b-instruct |
| GROQ_API_KEY | [SECRET — NOT DOCUMENTED] |
| GROQ_MODEL | openai/gpt-oss-120b |
| EMBEDDING_PROVIDER | sentence_transformers |
| EMBEDDING_MODEL | sentence-transformers/all-MiniLM-L6-v2 |
| CHROMA_PATH / CHROMA_COLLECTION | ./data/chroma / learning_materials_minilm |
| TOP_K | 5 |
| MAX_CONTEXT_CHARS | 12000 |
| TEMPERATURE | 0.2 |
| MAX_RETRIES | 2 |
| REQUEST_TIMEOUT | 60 |
| MAX_INPUT_CHARS | 20000 |
| CORS_ORIGINS | localhost:3000 and localhost:5173 |
| LOG_LEVEL | INFO |

The local .env does not define all settings shown in .env.example. Missing values use app/config.py defaults, unless process environment overrides them. TOP_K is accepted by the compatibility alias; FINAL_TOP_K is preferred if both aliases are supplied.

### Important application defaults and .env.example values

| Setting | Default / example | Use |
|---|---|---|
| AUTO_INGEST_PDFS | true | Synchronize sample PDFs during app lifespan. |
| SAMPLE_DATA_DIR | sample_data | Recursive PDF directory. |
| RETRIEVAL_STRATEGY | hybrid_cross_encoder | Only allowed Literal value; main constructs hybrid retriever directly. |
| DENSE_CANDIDATE_K | 30 | Chroma returned dense candidates. |
| SPARSE_CANDIDATE_K | 30 | BM25 candidates. |
| RRF_K | 60 | Rank fusion constant. |
| RERANK_CANDIDATE_K | 30 | Fused pool cap. |
| TOP_K / FINAL_TOP_K | 5 | Default final production results; optional request-body `top_k` (1–20) overrides it per learning request. |
| CROSS_ENCODER_MODEL | cross-encoder/ms-marco-MiniLM-L-6-v2 | Reranker model. |
| CROSS_ENCODER_DEVICE | auto | CUDA if available, CPU otherwise. |
| CROSS_ENCODER_BATCH_SIZE | 16 | Pair scoring batch. |
| CROSS_ENCODER_SCORE_OFFSET | 5.0 | Add before sigmoid transform. |
| MIN_RELEVANCE_SCORE | 0.35 | ContextBuilder floor on transformed relevance. |
| CHUNK_CHARS / CHUNK_OVERLAP_CHARS | 1000 / 120 | PDF chunking. |
| CLAIM_JUDGE_MIN_CONFIDENCE | 0.80 | Minimum accepted judge confidence. |
| MAX_RETRIES | 2 | Additional structured-output attempts. |
| MAX_CONTEXT_CHARS | 12000 | ContextBuilder and actual provider evidence-packet character cap. |
| REQUEST_TIMEOUT | 60 sec | Provider HTTP timeout. |
| MAX_INPUT_CHARS | 20000 | Decoded-character limit on `/learning/*` POST bodies; enforced before Pydantic parsing. |
| TEMPERATURE | 0.2 | Normal model generation; judge overrides to 0.0. |
| CORS_ORIGINS | localhost:3000, localhost:5173 | Middleware allowlist. |

Paths are relative to process working directory. Settings are cached with lru_cache. The current local .env contains a Groq credential key; this documentation does not include its value.

## 5. Packaging and container files

pyproject.toml declares FastAPI, Uvicorn, Pydantic, pydantic-settings, httpx, ChromaDB, pypdf, python-dotenv, and Sentence Transformers >=3,<6. Optional dev dependencies are pytest and pytest-asyncio. Python >=3.11. BM25 is in-project code.

Dockerfile uses python:3.11-slim, installs the package, exposes port 8000, and runs Uvicorn. docker-compose defines one learning-api service, publishes 8000, reads .env and mounts ./data at /service/data. .dockerignore excludes .env, data/chroma, .venv and caches; sample_data is copied into the image. Ollama is not in compose; its URL must address a separately running Ollama service from the container.

## 6. Historical choice versus technical rationale

README documents the hybrid-search rationale and RRF rationale. It does not contain a full architecture decision record or show alternatives trialed for each model.

| Component | Verified reason stated by repository | Further rationale (inference, not documented history) |
|---|---|---|
| Dense + BM25 | README: semantic retrieval handles differing wording; BM25 finds exact terms/names. | Complementary retrieval signals can improve candidate coverage. |
| RRF | README: combines rankings without comparing unlike score scales. | Reduces need to calibrate cosine and BM25 magnitudes. |
| Cross-Encoder | README/code: reranks bounded candidates for passage relevance. | Joint query/text interaction can improve ordering but costs inference time. |
| Chroma | Persistent local collection is configured; README notes local corpus workflow. | Keeps vector index local/persistent; no historical comparison documented. |
| MiniLM models | Exact names and use are configured; no training code. | Compact local retrieval inference is plausible, but original rationale is not documented. |
| Groq + Ollama | Provider order is configured and code behavior is explicit. | Hosted primary with local fallback is a practical topology; historical motive is not recorded. |
| Ollama-only judge | Enforced by provider-name check and no fallback. | Keeps this decision path isolated, but why this choice was made is not documented. |

## 7. Things mentioned in project but not implemented as runtime services

- WebSearchProvider is a protocol only; there is no web search call.
- DeterministicProvider is a fixed test/development double, not an automatic recovery model.
- HashEmbeddingProvider is optional test/utility embedding, not the default semantic model.
- No model training/fine-tuning, authenticated account service, stored learner profile, multi-tenant ownership filter, upload endpoint or scheduler.
- OutputValidator is a standalone helper; generation validation is implemented in providers.generate_structured.
- No independent BM25 server/library; BM25 is local custom Python.


