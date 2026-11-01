# Complete Project Technical Report

**Project:** Generative Learning Model  
**Audit date:** 2026-10-06  
**Scope:** source, configuration (secrets withheld), schemas, tests, benchmark scripts and saved results, sample PDFs, README, packaging, and container files. The report describes the implementation found in this project, not a generic RAG architecture. Historical design motives are distinguished from technical inferences.

## 1. What the project does

**Simple explanation:** A learner requests an explanation, summary, quiz, flashcards, practice exam, or feedback on an answer. The service searches learning PDFs, filters passages, asks a language model for typed educational content when appropriate, and returns JSON with grounding and source metadata. In the default materials-only mode, it can decline to generate when evidence is insufficient.

**Technical explanation:** This Python 3.11+ FastAPI service creates a persistent Chroma retriever, a hybrid dense-plus-BM25 retrieval pipeline with Reciprocal Rank Fusion and a Sentence Transformers Cross-Encoder, a Groq-primary/Ollama-fallback LLM provider, and one shared LearningGenerator. Six learning POST routes use the generator. PDF indexing occurs at application startup or through a script; there is no upload or ingestion endpoint.

Regular generation defaults to Groq model ID openai/gpt-oss-120b, then local Ollama model qwen2.5:7b-instruct only after a provider error. After generation, a separate Ollama-only Semantic Claim Judge checks extracted factual claims across evidence-required learning routes. Embedding and reranking models run locally through Sentence Transformers.

## 2. Project structure

~~~text
AI_Personal_Learning/
├── app/
│   ├── __init__.py
│   ├── __main__.py
│   ├── adaptation.py
│   ├── config.py
│   ├── domain.py
│   ├── evidence.py
│   ├── generation.py
│   ├── main.py
│   ├── prompting.py
│   ├── providers.py
│   ├── rag.py
│   ├── schemas.py
│   └── validation.py
├── data/
│   ├── chroma/ (SQLite and vector-index binaries)
│   ├── hybrid_cross_encoder_queries.json
│   ├── hybrid_cross_encoder_results.json
│   ├── retrieval_benchmark_queries.json
│   ├── retrieval_benchmark_report.md
│   └── retrieval_benchmark_results.json
├── sample_data/
│   ├── Ancient_Egypt_From.pdf
│   ├── history_rag_grounding_material.pdf
│   └── rag_grounding_decoy_material.pdf
├── scripts/
│   ├── benchmark_hybrid_cross_encoder.py
│   ├── benchmark_retrieval.py
│   ├── diagnose_groq.py
│   ├── evaluate.py
│   ├── ingest_sample.py
│   └── smoke_test_llm.py
├── tests/ (11 modules, listed in Testing and Benchmarks)
├── generative_learning_model.egg-info/ (generated package metadata)
├── .dockerignore
├── .env (local settings; secret value omitted)
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
├── TestCases.txt (manual API examples)
├── docker-compose.yml
├── pyproject.toml
└── rag.py (empty 0-byte root-level file)
~~~

| File/group | Purpose and relationships |
|---|---|
| app/main.py | Composition root: service construction, startup sync, CORS, middleware, exception handlers, health/readiness, and all eight routes. |
| app/config.py | Pydantic Settings defaults and environment loading; get_settings is cached. |
| app/schemas.py | Request/response contracts and source, grounding, adaptation, generation, claim-judge models. |
| app/domain.py | Internal dataclasses and protocols; WebSearchProvider is only a protocol, with no implementation wired in. |
| app/rag.py | PDF extraction/sync, chunking, Chroma dense search, custom BM25/RRF, Cross-Encoder, context building. |
| app/evidence.py | Substantive text, task evidence filters, citation candidate ranking, source snippets. |
| app/generation.py | Shared API orchestration, query construction, claim validation, retries, fail-closed behavior, trusted metadata. |
| app/prompting.py | Main prompt construction, profile and evidence sections, grounding rules, retry rules. |
| app/adaptation.py | Deterministic difficulty and adaptation-reason rules. |
| app/providers.py | Groq/Ollama/deterministic providers, fallback, structured output, Sentence Transformers and hash embeddings. |
| app/validation.py | OutputValidator helper and OutputValidationError; active generation calls generate_structured instead. |
| app/__main__.py | Uvicorn entrypoint at 0.0.0.0:8000. |
| scripts/ingest_sample.py | Manual call to the same PDF synchronizer. |
| scripts/benchmark_retrieval.py | Older 25-query, one-PDF dense/BM25/RRF/standalone-reranker experiment. |
| scripts/benchmark_hybrid_cross_encoder.py | Current hybrid pipeline benchmark; syncs PDFs before measuring. |
| scripts/evaluate.py | Small InMemoryRetriever wiring check, not a quality benchmark. |
| scripts/smoke_test_llm.py | Real provider request; not run in this audit. |
| scripts/diagnose_groq.py | Minimal authenticated Groq request; not run in this audit. |
| tests/ | API, schema, ingestion, retrieval, evidence, providers, judge, and regression tests. External LLM providers are mocked. |
| data/chroma/ | Persistent local Chroma database. Latest saved hybrid artifact records 661 chunks. |
| data/*benchmark* | Relevance labels and saved results, not live measurements. |
| README.md | Setup, architecture and operating notes; agrees with source that current API uses hybrid retrieval. |
| TestCases.txt | Handwritten example requests and expected relevant/irrelevant behavior, not pytest tests. |
| pyproject.toml | Canonical dependencies, Python/package configuration, pytest settings. |
| .env.example/.env | Example and local settings; GROQ_API_KEY is not disclosed. |
| Dockerfile/docker-compose.yml | One API service; compose reads local .env and mounts data. Ollama is external. |
| root rag.py | Empty and not imported. |
| .venv, .pytest_cache, __pycache__ | Generated runtime/test files, summarized rather than expanded. |

Module map:

~~~text
main → config, providers, rag, generation, schemas
generation → evidence, adaptation, prompting, providers, schemas, rag
rag → config, domain, embedding provider
providers → config, domain, validation
~~~

## 3. Main architecture

~~~mermaid
flowchart TD
    U[User JSON] --> F[FastAPI and request schema]
    F --> E[One of six learning endpoints]
    E --> G[LearningGenerator]
    G --> Q[Build retrieval query]
    Q --> D[Chroma dense search, up to 30]
    Q --> B[Process-local BM25, up to 30]
    D --> R[RRF, keep up to 30]
    B --> R
    R --> C[Cross-Encoder query-passage score]
    C --> K[Final top 5]
    K --> X[Relevance, substantive text, topic, task filters]
    X --> T{Usable evidence context?}
    T -- No in materials-only mode --> Z[Typed insufficient response]
    T -- Yes --> P[Build prompt with final evidence]
    P --> L[Groq generation]
    L -- Provider error --> O[Ollama generation fallback]
    L --> S[Parse JSON and validate schema/counts]
    O --> S
    S -- Invalid --> I[Up to 2 additional attempts]
    I --> S
    S --> G2[Extract factual claims]
    G2 --> J[Local Ollama Semantic Claim Judge]
    J --> C2{All claims supported at confidence ≥ 0.80?}
    C2 -- Yes --> M[Stamp provenance locally]
    C2 -- No or judge failure --> R2[One strict evidence-only retry]
    R2 --> L
    M --> OUT[Response schema and JSON]
~~~

**Simple:** Search, keep useful evidence, generate a typed result, check its factual claims, return JSON.

**Technical:** All learning routes use HybridCrossEncoderRetriever. Retrieval score ranks candidates; it does not certify generated facts. Evidence filtering and claim validation are separate. After structured generation, the local Ollama semantic judge validates each extracted factual claim against final evidence.

## 4. Request lifecycle

| Endpoint | Retrieval query |
|---|---|
| /learning/explain | question if provided, otherwise topic. |
| /learning/summarize | topic. |
| /learning/mcq | topic; requested concepts/question style go to generation prompt, not retrieval. |
| /learning/flashcards | topic. |
| /learning/exam | topics joined by comma and space. |
| /learning/evaluate-answer | Nonempty Topic, Question, and optional Expected answer lines. Student answer is excluded because it may be wrong, then passed to the feedback prompt. |

Public routes do not pass per-user/document metadata filters.

### Retrieval in order

1. pypdf extracts each PDF page separately; empty pages are omitted and original one-based numbers retained.
2. Each page is chunked at 1,000 characters by default with 120 characters of overlap and a nearby-word-boundary adjustment.
3. SHA-256 chunk IDs use normalized source path, page, start offset, and text. Metadata records source path/name, file hash, page, offset, embedding identity/version, and chunk ID.
4. Sentence Transformers embeds chunks and queries. Normalized vectors and metadata are stored in persistent Chroma with cosine space.
5. Dense retrieval requests 30 (or at least explicit request top_k). Chroma may fetch up to min(corpus size, max(top_k × 20, 50)) before incompatible embedding versions are dropped.
6. BM25 searches a process-local snapshot built from compatible Chroma rows; it returns up to 30 (or at least request top_k). It uses lowercase alphanumeric tokens with lightweight suffix normalization, k1=1.5, b=0.75.
7. RRF uses one-based ranks, k=60, chunk-ID tie-breaks and a max 30-row fused pool (expanded to satisfy a larger explicit final top_k). The union may contain up to 60 rows first.
8. Cross-Encoder scores up to the effective rerank pool in batches of 16, then returns configured final top 5 or request top_k.
9. ContextBuilder removes missing/low scores (<0.35), empty/duplicate content, and caps both selected chunk content and joined text at 12,000 characters.
10. Task evidence removes question sentences and known boilerplate; requires four words, three salient terms, declarative punctuation or a recognized verb, topical overlap (40% when >2 query terms, otherwise all), causal wording for why questions, and definition decoy checks for definitions.
11. ContextBuilder runs again. “Sufficient” means at least one chunk remains; it does not prove every claim.

Score meanings:
- Dense score: clamp(1 − cosine distance, 0, 1).
- BM25: lexical relevance, normalized by the best returned BM25 score.
- RRF: sum 1/(60 + rank), normalized by top fused score; uses ranks, not retrieval score magnitudes.
- Cross-Encoder: raw logit, not probability.
- Final relevance_score: sigmoid(raw logit + 5.0), with the sigmoid input clipped to [-60,60]. This is monotonic and compared to the 0.35 context floor.

BM25 contribution:

~~~text
idf(t) × tf(t,d) × (k1 + 1)
-----------------------------------------------
tf(t,d) + k1 × (1 - b + b × |d| / average_document_length)
~~~

idf(t) = ln(1 + (N - df(t) + 0.5)/(df(t) + 0.5)).

RRF:

~~~text
RRF(d) = sum over lists L of 1 / (60 + rank_L(d))
~~~

A chunk absent from a list contributes zero; a chunk in both lists receives both contributions.

### Ingestion and startup

AUTO_INGEST_PDFS defaults true. Startup sync runs in a worker thread. File hashes skip unchanged PDFs; new/changed chunks are embedded/upserted before stale chunks for the same source path are deleted; deleted PDFs under the synchronized directory are removed; a cross-process lock coordinates ingestion. Per-file errors are logged and do not stop other files. Outer sync failure is logged but does not prevent service startup; sparse refresh is attempted in finally. scripts/ingest_sample.py runs the same sync.

The latest hybrid benchmark artifact records 661 chunks, three PDFs discovered, all unchanged/skipped, and zero failures in that run. The older report attributes 657 chunks to Ancient_Egypt_From.pdf and four to the other two sample PDFs.

## 5. Evidence, claim validation, prompts and profile

### Grounding states

Retrieval failure means no candidates. Evidence failure means candidates did not survive score/content/topic/task filters. Claim failure means context exists but a factual claim is unsupported. Generation failure means provider transport or structured-output validation failed. These are separate states: retrieval/evidence insufficiency uses `insufficient_context`; later generation or claim-validation failure preserves sufficient retrieval state, source mode, and local sources and reports a `failure_status`.

In materials-only mode, genuinely insufficient evidence returns a typed response without an LLM call, with empty sources and null generation metadata. A response rejected after sufficient retrieval uses a safe typed failure response; it does not tell the learner the materials lack information. Generated answer and feedback claims are validated after generation; rejection with evidence is recorded as `grounding_validation_failed` while retrieval remains sufficient.

### Local semantic claim validation

After structured output parsing, declarative factual claims are extracted from learner-facing content and checked against the final evidence packet by the local Ollama Semantic Claim Judge (`qwen2.5:7b-instruct`). Claims pass only when strict JSON reports `supported=true` and confidence ≥0.80. Invalid output, unsupported/uncertain claims, or provider failure fail closed. Judge reasons are internal and are never copied into the answer. This replaces lexical claim coverage as the support decision; lexical matching remains only in retrieval/evidence heuristics and citation candidate ranking.

### Semantic Claim Judge

The judge receives one generated claim at a time and the final accepted evidence (source/page/text). Topic and question are omitted to reduce priming plausible details absent from evidence. It is instructed to use only evidence, allow faithful paraphrase, distinguish stated outcomes from unstated mechanisms, verify generated attributions against actual source text, treat source labels as provenance rather than proof, and return JSON only. Its strict schema has `supported` boolean, confidence float in [0,1], and a concise reason, with extra fields forbidden. It uses Ollama directly at temperature 0 with native JSON Schema output. It accepts only supported=true and confidence ≥0.80. Provider failure, malformed output, unsupported claim, invalid provider, no evidence, or low confidence fails closed. No Groq fallback applies to the judge.

Validation runs after retrieval, evidence filtering, final context construction, generation, and parsing; it does not call Ollama for retrieved chunks. Extraction covers factual learner-facing fields across explanation, summary, quiz/exam, flashcards and answer feedback, while skipping intentionally false distractors and interrogative text. Sentence splitting recognizes punctuation before closing quotes. The judge's reason remains internal. Answer evaluation validates its generated feedback; student answers are not used as evidence of factual truth.

### Generation prompt and learner profile

The main system prompt says: “You are a precise educational AI. Follow grounding rules and output JSON only.” PromptEngine adds task, topic, request fields other than profile, grounding state, adapted settings, required counts, learner profile, and evidence packet with chunk/source/page/relevance/text. Retrieved text is marked untrusted; instructions inside it must be ignored.

Materials-only instructions require facts to be traceable to supplied evidence, no gap-filling/general knowledge, omission of unsupported material, and separate support for MCQ/card answer and rationale. General-knowledge mode allows model knowledge only when the request permits it. Strict retry adds stronger evidence-only wording.

Task-specific constraints:
- Brief evidence-bounded explanations are instructed to copy one complete sentence from evidence into `explanation` and leave optional factual arrays empty. Claim validation remains active; standard/high depth still use their own guidance.
- MCQ/exam: exact question count; short evidence-copied answer and full evidence sentence as explanation; prompt asks four options, schema requires only at least two.
- Flashcards: exact count and short front/back.
- Evaluation: distinct roles for correct_answer, why_answer_is_wrong, explanation, identified_misconception; student text is only proof of what learner wrote; facts come from evidence; scores/confidence in [0,1].

Profile data comes from each request, not persistence or a user database. Relevant profile fields are topic-scoped and converted to explicit adaptation guidance; `user_id` is not included in prompts or logs. Requested difficulty remains recorded while mastery/performance signals adjust effective difficulty at most one step and add scaffolding or reduced-hint instructions. See [PARAMETER_BEHAVIOR_AUDIT.md](PARAMETER_BEHAVIOR_AUDIT.md) for each profile field and request parameter.

## 6. Models and technology inventory

| Name | Role in this project |
|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | Local embedding encoder shared by ingestion and query. Code reads dimension dynamically; saved report records 384. Embeddings are normalized. |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | Local relevance reranker; lazy load, max length 512, batch 16, device auto (CUDA when available, else CPU). |
| qwen2.5:7b-instruct | Local Ollama model for normal-generation fallback and direct claim judging. |
| openai/gpt-oss-120b | Configured Groq-hosted normal-generation model; Groq call uses strict JSON Schema and reasoning_effort=low for GPT-OSS. |
| HashEmbeddingProvider | Optional deterministic 32-D lexical token/bigram hash vector utility; not configured default semantic embedding. |
| ChromaDB | Persistent local vectors, documents and metadata; cosine space and embedding-identity checks. |
| BM25 / RRF | Custom app code, no external BM25 dependency. |
| FastAPI/Pydantic/Uvicorn | HTTP routes, request/response contracts, serialization and server. |
| pypdf | Page text extraction from PDFs. |
| Sentence Transformers/PyTorch | Local embedding and Cross-Encoder inference. PyTorch is transitive, not direct in pyproject. |
| Groq API/Ollama | Remote generation API and local model service. |
| pytest/pytest-asyncio | Optional development testing dependencies. |
| httpx / python-dotenv | Direct project dependencies. NumPy is used by Sentence Transformers and imported by an embedding test, but is transitive and not directly declared in pyproject.toml. |

No training or fine-tuning code exists; these models are inference-only in this project. Saved benchmark says Cross-Encoder ran CPU. The embedding adapter passes no explicit device, so Sentence Transformers selects it. RAM/RSS/GPU memory are not measured. Original model-selection history is not recorded; compact local retrieval models plus hosted generation is a technical inference, not documented historical fact.

## 7. API endpoints and schemas

| Method/route | Request → response | Purpose |
|---|---|---|
| GET /health | none → object | Liveness, service and configured provider; no dependency probe. |
| GET /ready | none → object or 503 | Chroma count and collection; does not check providers. |
| POST /learning/explain | ExplanationRequest → ExplanationResponse | Explanation. |
| POST /learning/summarize | SummaryRequest → SummaryResponse | Summary. |
| POST /learning/mcq | MCQRequest → MCQSet | Multiple-choice set. |
| POST /learning/flashcards | FlashcardRequest → FlashcardSet | Flashcards. |
| POST /learning/exam | PracticeExamRequest → PracticeExam | Exam across topics. |
| POST /learning/evaluate-answer | AnswerEvaluationRequest → AnswerEvaluation | Grading and feedback. |

FastAPI also serves /docs, /redoc, and /openapi.json.

Verified schema details:
- Topic fields: required 1–300 characters for explain/summary/mcq/flashcards. Explain question is optional.
- Summary style is an enum: concise, exam-focused, detailed, bullet-point, revision-sheet.
- Knowledge mode: learning_materials_only default or learning_materials_then_general_knowledge.
- MCQ count 1–50 default 5; flashcard count 1–50 default 10; exam requires ≥1 topic, count 1–100 default 10.
- Exam time limit optional 1–600; weak_topic_emphasis defaults true.
- MCQ options minimum 2 and no schema maximum; prompt asks four.
- Evaluation question 1–5,000 chars, student answer 1–10,000; optional correct answer max 10,000.
- Evaluation score/confidence each [0,1].
- Learner mastery, accuracy, and per-difficulty performance values are bounded to [0,1].
- Grounding schema allows grounded/partially_grounded/not_grounded, but runtime emits grounded/not_grounded.
- LearningSource records optional document/page/chunk/snippet/relevance. GenerationMetadata records provider/model/fallback_used.
- ClaimValidationResult forbids extras; supported bool, confidence [0,1], reason length 1–1,000.

MAX_INPUT_CHARS=20000 is enforced on `/learning/*` POST bodies and returns 413 above the configured character cap. Declared request fields have explicit bounds; empty/missing required fields and invalid enum/range values return 422. LearnerProfileIn explicitly ignores extra fields.

### Structured output

Groq sends strict JSON Schema when given a schema. Ollama sends /api/chat with schema-valued format. Provider-facing schema removes server-stamped grounding, adaptation, generation, source/evidence, and time-limit fields. Requested card/question counts are encoded as minItems=maxItems and checked again locally. Code parses JSON object, injects trusted fields, validates exact counts and full Pydantic schema, then stamps the actual provider metadata.

Invalid JSON/schema/count receives up to two additional attempts (three total). It does not itself trigger provider fallback; only ProviderError does. Exhaustion maps to 502 invalid_llm_output. Provider errors from both providers map to 502 llm_provider_unavailable. Other uncaught errors map to 500 internal_error. The OutputValidator helper is not the active generation path.

## 8. Nile example

The following is based on the Semantic Claim Judge regression fixture. It uses fixed page-16 evidence and mocked providers, so it tests the claim path but is not a real LLM transcript.

~~~json
{
  "question": "Why was the Nile River important to ancient Egyptian civilization?",
  "student_answer": "The Nile River was mainly important because it provided a route for military expansion.",
  "correct_answer": "The annual inundation of the Nile River supported the agricultural population by providing fertile conditions for farming.",
  "topic": "Ancient Egyptian Agriculture",
  "difficulty": "medium",
  "knowledge_mode": "learning_materials_only",
  "learner_profile": {}
}
~~~

Test trace: retrieval query includes topic, question and expected answer but excludes student answer; fixed evidence describes annual inundation on page 16; mocked Ollama accepts a faithful paraphrase at confidence ≥0.80 and receives the final evidence packet; mocked normal provider produces feedback; assertions verify grounded status and page-16 source.

Separate saved benchmark record for the similar Nile “Why was the Nile River important to the development…” query used evaluation top 10: first five pages were 10, 17, 16, 10, 150. Page 16 ranked 3 with raw Cross-Encoder score about 3.3925. Total retrieval latency about 725 ms, Cross-Encoder about 669 ms. This is a benchmark record, not necessarily the exact endpoint top five.

Illustrative response shape (not captured model output):

~~~json
{
  "correct": false,
  "score": 0.0,
  "correct_answer": "source-backed expected answer",
  "explanation": "evidence-backed teaching",
  "why_answer_is_wrong": "specific mismatch",
  "related_concept": "Ancient Egyptian Agriculture",
  "recommended_revision": ["Agriculture"],
  "confidence": 0.95,
  "identified_misconception": "Prioritizing military expansion over inundation",
  "supporting_source_or_evidence": [{"document_name":"Ancient_Egypt_From.pdf","page_number":16,"chunk_id":"local id","snippet":"relevant passage","relevance_score":0.7}],
  "groundedness": "grounded",
  "topic": "Ancient Egyptian Agriculture",
  "difficulty": "medium",
  "grounding": {"source_mode":"retrieved_learning_content","grounding_status":"grounded","retrieval_status":"sufficient","retrieval_method":"hybrid_cross_encoder","sources":[],"notice":null},
  "generation": {"provider":"groq","model":"configured model","fallback_used":false}
}
~~~

## 9. Testing and saved evaluation

Test suite command `.venv\Scripts\python.exe -m pytest -q`: **134 passed, 2 dependency deprecation warnings, 40.71 seconds**. Coverage includes API behavior, PDF sync/Chroma metadata, hybrid retrieval, evidence and source checks, semantic judge, structured output, provider fallback/embedding cache, request/response contracts, parameter-sensitivity combinations, and benchmark labels/helpers. Real Groq/Ollama calls are mocked in the suite; some local retrieval integration tests use persistent Chroma and retrieval models. A separate live 23-case semantic benchmark and API checks are summarized in [TESTING_AND_BENCHMARKS.md](TESTING_AND_BENCHMARKS.md). Tests do not establish general model quality.

Latest saved multi-document benchmark:
- 661 chunks; 11 labeled queries (9 positive, 2 negative); benchmark top 10 versus production top 5.
- 30 dense, 30 sparse, RRF k=60, fused/rerank cap 30; CPU Cross-Encoder.
- Positive-query Recall@5 0.611, Hit@5 1.000, MRR 0.778.
- Mean retrieval latency 731.75 ms and p95 1,072.32 ms over all 11 queries.
- Stage means: dense 45.73 ms, sparse 7.08 ms, fusion 0.10 ms, Cross-Encoder 678.64 ms, total 731.66 ms.
- All nine positive contexts passed context sufficiency; neither negative context did, though candidate retrieval returned rows.

These are stored benchmark measurements, not rerun for this audit. The older 25-query single-PDF report labels dense-only “CURRENT” and says Cross-Encoder was not integrated. That narrative conflicts with current main.py and tests, which use HybridCrossEncoderRetriever. The older experiment used different labels, 20+20 candidates, pool 20, top 10, and standalone raw-logit reranking. Its production/winner description is stale, and its metrics are not directly comparable.

Recall@k is relevant retrieved chunks / all labeled relevant chunks. Hit Rate@k is the share with any relevant top-k. Reciprocal Rank is 1 / first relevant rank; MRR averages it. p95 uses the script’s ceiling-index percentile. Current recall/hit metrics average positive queries only; latency uses all records. These small manually labeled sets support engineering comparisons, not statistical proof.

## 10. Errors, decisions, limitations

| Event | Behavior |
|---|---|
| PDF-level parse/embed/Chroma error during sync | Log/count failure, continue other PDFs; old chunks are not deleted until replacement succeeds. |
| Outer startup sync failure | Log and continue startup; attempt sparse refresh in finally. |
| Chroma readiness error | /ready returns 503. |
| Other unhandled request error | HTTP 500 with error name and request ID. |
| Groq provider error | Try Ollama fallback; both fail → HTTP 502. |
| Invalid model JSON/schema/count | Up to 2 retries; exhaustion → HTTP 502; no fallback solely for malformed content. |
| Insufficient context in materials-only mode | HTTP 200 typed insufficient result, no generation, no sources, null generation. |
| Generated claim rejected after retrieval | Safe typed result; `retrieval_status` and sources remain sufficient, `source_mode` remains retrieved learning content, and `failure_status` is `grounding_validation_failed`. |
| Generated claim unsupported | Strict evidence-only retry; salvage coherent individually accepted claims if possible, otherwise grounding-validation failure. Sufficient retrieval status and sources are preserved. |
| Judge failure | Direct Ollama validation fails closed; never calls Groq. Sufficient retrieval metadata is preserved. |

README records hybrid rationale: semantic retrieval for paraphrases plus BM25 for exact wording; RRF combines ranked lists without comparing unlike score scales. It does not preserve historical decision records for most models/database choices. Chroma/local models/Groq choices may have technical rationales, but these are inferences rather than recorded motives.

| Removed | Consequence |
|---|---|
| BM25 | Less exact-word/name recall. |
| Dense retrieval | Semantic paraphrase search disappears. |
| RRF | No policy combines ranked lists. |
| Cross-Encoder | Faster but no pairwise reranking. |
| Evidence filtering | Headings/boilerplate/off-topic/non-causal passages could pass. |
| Context sufficiency | Materials-only no-evidence gate disappears. |
| Deterministic claim check | No cheap lexical-only generated-claim support decision. |
| Semantic judge | More equivalent paraphrases are rejected. |
| Ollama | No local fallback/judge; some cases fail closed. |
| Groq | Generation depends on local Ollama. |
| ChromaDB | Persistent vectors and BM25's corpus source need replacement. |
| Learner profile | Personalization/adaptation disappears; retrieval remains. |
| Pydantic | Less request/output validation. |
| FastAPI | HTTP routing, automatic validation/docs, middleware need replacement. |

Verified current limitations:
- Benchmark sets are small (11 multi-document queries, two negatives; older set 25 queries in one PDF).
- CPU Cross-Encoder dominates saved retrieval latency; LLM latency is not measured.
- Reranker scores are not calibrated probabilities.
- ContextBuilder bounds both its text and chunk contents; the provider evidence packet uses those same bounded chunks.
- Request `top_k` can override configured final top-K for each learning endpoint, up to 20.
- Profile is request-supplied; no authentication, user persistence or public tenant filter. Corpus is shared.
- Semantic Claim Judge validates generated factual claims across learning routes after generation; judgment is probabilistic and fail-closed.
- General-knowledge fallback is permitted only when filtered retrieval is insufficient. It is marked `general_knowledge`/`not_grounded`; sufficient retrieved context remains evidence-validated even in fallback mode.
- Model memory/RSS/GPU usage is not measured; no training/fine-tuning, scheduler, profile history service, or web search is implemented.
- OutputValidator exists but is unused in the active path; partially_grounded is schema-only in current runtime.

## 11. Final architecture summary and project at a glance

~~~text
PDF → page extraction → 1,000-char chunks with 120 overlap → embeddings and provenance
    → persistent Chroma and in-memory BM25

Request → FastAPI/Pydantic → endpoint query → dense 30 + sparse 30 → RRF top 30
        → Cross-Encoder → top 5 → relevance/content/topic/task evidence
        → prompt → Groq, Ollama fallback after provider error
        → JSON/count/schema validation → extract factual claims
        → local Ollama claim judge (threshold 0.80) → retry/salvage → provenance
        → typed JSON or typed insufficient-context response
~~~

| Item | Current implementation |
|---|---|
| Purpose | Learning generation and answer feedback grounded in PDFs. |
| Inputs/outputs | Six learning request shapes → typed JSON with grounding/adaptation/generation fields. |
| Models | MiniLM embedding; MiniLM Cross-Encoder; Groq GPT-OSS; Ollama Qwen fallback/judge. |
| Data/retrieval | Persistent Chroma, BM25 snapshot, RRF, Cross-Encoder, production top 5. |
| Validation | Pydantic/structural checks, evidence filters, post-generation local semantic claim judge, strict retry/salvage. |
| Strengths | Page provenance, reproducible sync, hybrid retrieval, separate evidence checks, fail-closed path, parameter behavior tests, 134 passing tests. |
| Measured bottleneck | CPU Cross-Encoder. |
| Test status | 134 passed; 2 dependency warnings; tests use provider doubles; live Ollama benchmark is separate. |
| Documentation caveat | Legacy report’s dense-only “current” label conflicts with current source. |

## Related documents

- [System architecture](SYSTEM_ARCHITECTURE.md)
- [API and data flow](API_AND_DATA_FLOW.md)
- [Models and technologies](MODELS_AND_TECHNOLOGIES.md)
- [RAG and retrieval](RAG_AND_RETRIEVAL.md)
- [Grounding and validation](GROUNDING_AND_VALIDATION.md)
- [Testing and benchmarks](TESTING_AND_BENCHMARKS.md)
- [Professor presentation guide](PROFESSOR_PRESENTATION_GUIDE.md)
- [Glossary](GLOSSARY.md)

