# Testing and Benchmarks

This document separates code that was tested, saved benchmark measurements, and scripts that were inspected but not executed during the audit.

## 1. Current test run

Ran `.venv\Scripts\python.exe -m pytest -q` on 2026-10-06 after local semantic claim validation and sentence extraction changes.

**Result: 134 passed, 2 dependency deprecation warnings, 40.71 seconds.**

Warnings come from Starlette/FastAPI TestClient's use of the current httpx/AnyIO compatibility layer. They did not fail tests. This is the observed current-suite result, not a historical README count.

No Groq or Ollama live requests were made by this suite. Provider behavior and HTTP payloads use fakes/mocks. Local API/retrieval regressions load the project's Chroma collection and run local retrieval/model paths.

The full test suite uses scripted providers; it makes no live Groq or Ollama calls. Some retrieval tests load Chroma and local retrieval models. Live local Ollama behavior is measured separately below.

### Manual Swagger verification

On 2026-10-06, the running app served `/docs` with HTTP 200. The final verification used the live `/learning/explain` route. The exact easy/brief/top-3 Nile request produced an evidence-grounded page-16 answer. The exact hard/high/top-10 request was rerun after strengthening claim extraction and judge instructions; its result is recorded below. The unsupported quantum-teleportation request returned `insufficient_context`, `retrieval_status: insufficient`, empty sources, and null generation.

In the hard/high audit, the first response made an unsupported silt-deposition claim. This exposed two issues: a closing quote prevented the next sentence from being split, and the user question could prime the judge with historical knowledge. Sentence extraction now splits after closing quotes; the judge receives the topic but not the question and is explicitly told to treat plausible facts as unsupported unless stated. A direct check on the silt claim against pages 16–17 then returned unsupported at confidence 0.60. The final live hard/high rerun outcome is reported after that change. The unsupported-generated-claim and Ollama-outage cases are covered by deterministic endpoint/unit tests that assert sufficient retrieval and sources remain while validation fails closed.

### Local semantic validation benchmark

`scripts/benchmark_semantic_claim_validation.py` ran 23 fixed claim/evidence pairs using the configured local Ollama model at temperature 0, after the final prompt change. It contains 5 clearly supported claims, 5 faithful paraphrases, 5 clearly unsupported claims, and 8 plausible-but-unsupported claims (including Nile silt/nutrient additions and unsupported claims that attribute wording to the evidence).

| Measure | Result |
|---|---:|
| Supported acceptance | 10/10 (100%) |
| Unsupported rejection | 13/13 (100%) |
| False positives | 0 |
| False negatives | 0 |

The expanded set now rejects “The annual Nile flood deposited fertile silt onto the floodplain” against evidence that mentions the fertile floodplain and productivity but not silt deposition. The prior 21-case run accepted that claim, so this targeted prompt change removes that observed false positive in the small benchmark. The 23-case result is not a statistical quality guarantee; the judge confidence threshold remains 0.80.

## 2. Test modules and what they establish

| Test module | Scope and representative assertions |
|---|---|
| test_api.py | Health/OpenAPI/docs pages, MCQ route, both-provider failure mapped to clean 502, invalid structured output classification, exact Nile “why” request finds page-16 source, quantum query fails closed, all six learning routes cover supported/unsupported/invalid bodies. Provider is mocked; retrieval path uses configured app retriever. |
| test_core.py | Context text deduplication/character cap, low/high mastery adaptation, prompt includes profile performance fields, schema behavior. One test expects no provider output for strict-mode quiz generation without LLM. |
| test_evidence_grounding.py | Exact Nile why query retains pages 16 and 17, page-17 `ensured` causal evidence, causal paraphrase and unsupported-army/silt checks, supported and repeated-unsupported generated-answer status/source regressions, substantive/task evidence, topical/cause/definition rejection, citation candidate ranking, retry/retention, source-specific content, and query privacy. |
| test_generation_contracts.py | Exact requested card/question counts, adaptation preserved, server-stamped grounding/evidence, compact provider schemas omit trusted fields, heading-only context rejected, too-few count fails schema validation. |
| test_hybrid_retrieval.py | BM25 exact/morphological matching and empty queries, RRF union/dedup/ties, Cross-Encoder pool/batch/top-k behavior, device selection and lazy model loading contracts. |
| test_llm_generation.py | Structured output call, invalid-output retry, mocked Ollama payload format, mocked Groq JSON Schema request, safe HTTP error redaction. |
| test_providers_embeddings.py | Primary/fallback metadata, fallback token budget, no deterministic automatic rescue, explicit deterministic selection, sentence-transformer caching/batch/dimension/normalization setup via a fake model. |
| test_rag_pdf.py | Generated small PDFs, page metadata, stable ID/idempotency, Chroma retrieval, configured production retriever type/model, relevant and unrelated retrieval gates, Nile evidence regression, sync new/unchanged/modified/deleted PDFs, per-file failure continuation, startup sync. |
| test_retrieval_benchmark.py | Deterministic BM25 and RRF ordering/ties, metric formulas, page labels exist in collection, hybrid labels span documents and saved positive/negative context-gate outcomes. It validates existing artifact; it does not rerun benchmark. |
| test_semantic_claim_judge.py | Strict schema/score limits; unsupported/partial claims, low confidence, malformed output, Ollama failure and wrong provider all fail closed; no Groq judge fallback; normal generation still has Groq→Ollama fallback; Nile and cereal-crop regression; splitting sentences after closing quotes. Providers are scripted. |
| test_parameter_sensitivity.py | Difficulty×depth×profile prompts, styles/concepts, knowledge-mode×retrieval, topic-scoped profile fields, evaluation answer/profile interaction, answer-score sensitivity, top-K 3/5/10, context limit, input cap, RRF k, score offset, and bounds. Uses deterministic provider/retriever doubles. |
| test_validation.py | OutputValidator accepts valid JSON and rejects malformed JSON. This utility is not used in active generation path. |

The 134 pytest cases include parametrized variants. A passing assertion establishes only the behavior and fixture in that test; it is not proof of general model accuracy. The complete parameter map and manual Swagger matrix are in `PARAMETER_BEHAVIOR_AUDIT.md`.

## 3. Test categories

### Unit and contract tests
Custom BM25, RRF, token normalization, text/substantive filters, source selection, profile adaptation, schema shape, count requirements, provider payload construction, fallback metadata and exception mapping are tested with small fixed data or fake providers.

### Local integration/regression
PDF ingestion tests create temporary PDFs and temporary Chroma stores. Additional tests use the configured persistent local Chroma collection to verify the default retriever, embedding model identity, Nile page-16 results, and unsupported-topic gating. They do not make external generation requests.

### API tests
FastAPI TestClient covers endpoint response status and schema, all six learning POST routes, missing/invalid bodies, supported and unsupported behavior, source metadata, request IDs and provider errors. LLM generation is replaced with a fake provider.

### Claim-judge tests
The judge tests use ScriptedProvider/FailingProvider, not a live local Ollama service. They prove orchestration rules and schema decisions against chosen responses; they do not independently establish the Qwen model's judgment quality.

### Evaluation endpoint tests
Answer grading behavior is checked with deterministic test passages and generated payloads. The project does not contain a separate numerical rubric implementation that tests can validate.

## 4. Benchmark scripts and datasets

| Artifact/script | Role |
|---|---|
| scripts/benchmark_hybrid_cross_encoder.py | Current implementation benchmark for multiple sample PDFs; synchronizes corpus, warms pipeline, measures dense/sparse/fusion/Cross-Encoder and checks evidence-context sufficiency. |
| data/hybrid_cross_encoder_queries.json | Fixed multi-document positive/negative queries with expected source filenames and physical PDF pages. |
| data/hybrid_cross_encoder_results.json | Saved run output including per-query ranked chunks, scores, context sufficiency, stage times and summary metrics. |
| scripts/benchmark_retrieval.py | Older experimental comparison of dense-only, local BM25+RRF and standalone Cross-Encoder on the older query set. |
| data/retrieval_benchmark_queries.json | 25 questions and manually chosen relevant Ancient Egypt pages. |
| data/retrieval_benchmark_results.json | Saved per-query rankings, metrics and timings for old experiment variants. |
| data/retrieval_benchmark_report.md | Human-written report for old experiment; architecture narrative now conflicts with current source. |
| scripts/evaluate.py | Small deterministic InMemoryRetriever schema/grounding/adaptation wiring check, not benchmark. |
| TestCases.txt | Manual endpoint request examples/expected outcomes, not executable pytest. |
| scripts/smoke_test_llm.py | Would make a real configured provider request; not run in this audit. |
| scripts/diagnose_groq.py | Would make one real Groq API diagnostic call; not run in this audit. |

The benchmark scripts were inspected; saved results are documented below. The benchmark scripts were not re-executed as part of this audit.

## 5. Latest multi-document hybrid result

Source: data/hybrid_cross_encoder_results.json. The artifact records 661 corpus chunks after a sync of three sample PDFs (all three unchanged, zero failures). It has 11 queries: nine with expected source pages and two negative queries without expected sources. Retrieval metrics below average over positive queries; latency includes all records. Evaluation requests top 10 even though production top_k defaults to five.

| Variant | Recall@5 | Hit Rate@5 | MRR | Avg latency | p95 latency |
|---|---:|---:|---:|---:|---:|
| Dense | 0.528 | 0.778 | 0.703 | 46.89 ms | 52.46 ms |
| Sparse BM25 | 0.481 | 0.889 | 0.704 | 7.06 ms | 9.62 ms |
| Hybrid RRF | 0.574 | 1.000 | 0.739 | 54.07 ms | 62.19 ms |
| Hybrid + Cross-Encoder | 0.611 | 1.000 | 0.778 | 731.75 ms | 1,072.32 ms |

For the Cross-Encoder row, Hit@3 is 1.000; Hit@1 is 0.556. Both negative queries returned retrieval candidates, but both failed the task-evidence context gate. All nine positive contexts passed that gate. The context-gate metric is only actually evaluated for the production Cross-Encoder variant; the script's zero values for other variants come from missing context_sufficient fields defaulting to false, so they are not evidence those variants ran the gate.

Recorded Cross-Encoder stage mean is 678.64 ms, dense 45.73 ms, sparse 7.08 ms, fusion 0.10 ms, total 731.66 ms. Device is CPU. One-time model load is excluded from per-query latency. The output artifact includes a sync result, but not a model-memory/RSS measurement.

### Metric definitions

- **Recall@k:** For each positive query, count relevant chunks among top k, divide by all chunks labeled relevant for that query, then average across positive queries.
- **Hit Rate@k:** Fraction of positive queries with at least one relevant chunk in top k.
- **Reciprocal Rank:** 1 divided by rank of first relevant result, or 0 if absent.
- **MRR:** Mean Reciprocal Rank across positive queries.
- **Average latency:** Arithmetic mean per-query wall time in milliseconds.
- **p95 latency:** Sorted values at index ceil(0.95 × n) − 1. It is a nearest-rank-style percentile implementation.
- **Negative retrieval return rate:** Share of negative queries with any retrieved candidate.
- **Negative sufficient-context rate:** Share of negative queries whose task evidence passed sufficiency. It measures behavior on this small labeled negative set, not a guarantee on unknown questions.

The current benchmark's candidate counts are 30 dense and 30 sparse; RRF k=60; up to 30 fused rows; final top 10 for measurement. Cross-Encoder sorting uses raw score. Score offset + sigmoid creates the relevance interface score used by context gating.

## 6. Older single-document experiment and documentation conflict

The legacy retrieval benchmark includes 25 queries from Ancient_Egypt_From.pdf, 46 query-to-page labels over 34 pages, 152 relevant query/chunk pairs, and three ranker variants. It uses top 10, 20 semantic candidates, 20 lexical candidates, a 20-row fusion pool, RRF k=60, and CPU Cross-Encoder. The human report documents the results:

| Variant in old report | Recall@5 | Hit@5 | MRR | Avg latency |
|---|---:|---:|---:|---:|
| CURRENT (dense-only Chroma in that experiment) | 0.383 | 0.880 | 0.748 | about 23 ms |
| HYBRID (RRF) | 0.371 | 0.920 | 0.709 | about 41 ms |
| HYBRID + CROSS-ENCODER | 0.339 | 0.880 | 0.732 | about 476 ms |

That report says dense-only was the production retriever and Cross-Encoder was not integrated. Current app/main.py constructs HybridCrossEncoderRetriever, README describes hybrid production, and a test asserts this type/model. Therefore the old report's architecture conclusion is stale. The old metrics are still a saved experiment but cannot be treated as a current comparison because labels, corpus scope, candidate sizes, score handling and script differ from the latest hybrid benchmark.

The earlier report describes its quality selection rule and says no challenger qualified on that dataset. The current source's use of hybrid is a later repository state; no historical migration record was found.

## 7. Performance interpretation

**Measured:** For the latest saved 11-query run, Cross-Encoder mean ≈679 ms out of ≈732 ms total. Dense mean ≈46 ms; BM25 ≈7 ms; RRF ≈0.1 ms. This is the dominant measured retrieval stage on that CPU environment.

**Not measured in saved results:** Groq generation latency, Ollama fallback latency, Ollama judge latency, concurrent load, end-to-end HTTP p95, peak RAM/RSS, GPU memory, cold model download time for the latest run.

**Configured behavior:** embedding model and Cross-Encoder load/caching; Cross-Encoder device auto; request timeout 60 seconds; retry count two additional structured-output attempts. No explicit performance optimization task was performed.

## 8. Evaluation script distinctions

scripts/evaluate.py constructs a tiny InMemoryRetriever with one CNN passage, asks for MCQs and returns booleans for schema-valid, grounded and adapted difficulty. Its comment explicitly calls it a wiring check, not retrieval benchmark or quality metric.

scripts/benchmark_hybrid_cross_encoder.py calls sync_pdf_directory and therefore can update data/chroma when run. scripts/ingest_sample.py also mutates the persistent index. Neither was run during this audit. The saved artifacts and current test suite were sufficient to document status.


