# Professor Presentation Guide

This guide gives presentation-length summaries and questions grounded in the current source. It distinguishes documented design rationale from inference.

## 30-second explanation

This project is a FastAPI learning service that answers from indexed course PDFs. It retrieves passages with dense vector search and BM25, combines rankings with RRF, reranks a small candidate set using a Cross-Encoder, and filters evidence before generation. Groq is the normal generation provider with Ollama as fallback. After structured generation, a separate local Ollama Semantic Claim Judge checks generated factual claims against the final evidence packet.

## 1-minute explanation

The application indexes PDF pages as overlapping chunks and stores normalized Sentence Transformers embeddings plus source/page metadata in persistent ChromaDB. At query time, dense retrieval and a custom in-memory BM25 index each return up to 30 passages. Reciprocal Rank Fusion merges their ranks, caps the pool at 30, and the local Cross-Encoder evaluates query/passage pairs. The API returns the top five to evidence filters that reject non-substantive text, poor topical overlap, definition decoys and non-causal passages for “why” questions.

All six learning routes share one generator and typed Pydantic schemas. In materials-only mode, insufficient evidence returns a typed notice. A later generation or claim-validation failure preserves sufficient retrieval state and sources and reports a separate failure status. The alternate mode permits general-knowledge fallback only when filtered evidence is insufficient; sufficient evidence is still strictly validated. Groq generates structured JSON; Ollama is used on provider failure and independently validates claims. Unsupported output gets a strict retry and may fail closed. The complete parameter audit adds task-specific instructions for difficulty, explanation depth, style, retrieval count, and learner context. Current full test result: 134 passed. Saved retrieval and claim-validation benchmarks are small engineering samples, not proof of broad accuracy.

## 3-minute explanation

1. **Ingestion:** pypdf extracts text by physical page. Each page is chunked at 1,000 characters with 120 overlap. Stable IDs, hashes and page/source metadata are stored with normalized MiniLM vectors in local persistent Chroma. Unchanged PDFs are skipped at sync.
2. **Hybrid retrieval:** Chroma dense search returns up to 30 results. Custom BM25 searches a process-local snapshot of the same Chroma corpus for up to 30 lexical candidates. RRF with k=60 sums reciprocal rank contributions and keeps up to 30 unique chunks.
3. **Reranking:** The model cross-encoder/ms-marco-MiniLM-L-6-v2 scores query/passage pairs in batches of 16. It returns the configured final top five by default, or the request's explicit `top_k` (1–20). The score used by ContextBuilder is sigmoid(raw logit + configured offset), which is an interface score, not a calibrated probability.
4. **Evidence:** The service requires score ≥0.35 and filters headings/questions/boilerplate, off-topic chunks, definition decoys and non-causal passages for why-questions. Context sufficiency means one chunk survived; separate claim validation checks actual answer statements.
5. **Generation:** Prompt contains task, request, learner profile, adaptation and accepted evidence. Materials-only mode prohibits unsupported facts and treats evidence as untrusted data, not instructions. Groq is primary; Ollama is the normal provider fallback after ProviderError.
6. **Claim validation:** After structured JSON parsing, declarative factual statements are extracted from learner-facing fields and sent one at a time to the local Ollama judge with the final evidence packet. Topic and question are omitted to reduce priming. Only `supported=true` with confidence ≥0.80 is accepted. Invalid output, low confidence, unsupported claims, and Ollama failure fail closed.
7. **Output:** Pydantic and deterministic structural checks validate fields/counts, then sources are locally stamped after factual validation. Answer evaluation still scores against the authoritative expected answer; the judge validates claims in the generated feedback after generation. Semantic judgments and response style remain probabilistic, and evaluation sets are small.

## 5-minute technical explanation

**Problem and service boundary.** The service provides learning-content generation grounded in PDF material. FastAPI defines six learning routes with request/response Pydantic models. The request body carries learner profile data; this project does not own user accounts, persist a learner profile, or expose authenticated per-document isolation. PDF sync is performed on startup or by a script.

**Index and retrieval.** pypdf extracts each physical page. Chunks default to 1,000 characters with 120 overlap and keep source/page provenance. A shared sentence-transformers/all-MiniLM-L6-v2 encoder writes normalized vectors and encodes queries; Chroma uses a persistent local collection configured for cosine. The app builds BM25 from compatible Chroma text at startup and refreshes it after sync. The BM25 implementation uses lowercase alphanumeric tokens plus small suffix normalization, with k1=1.5/b=0.75.

Dense and sparse lists have separate score scales. RRF avoids direct score calibration by summing 1/(60+rank) with one-based ranks. The union is capped at 30. A lazy-loaded cross-encoder/ms-marco-MiniLM-L-6-v2 evaluates query/passage pairs at batch size 16 and max length 512, then returns five. On the saved multi-document benchmark, CPU Cross-Encoder time averaged about 679 ms of 732 ms total; this is the largest measured retrieval cost.

**Evidence and grounding.** ContextBuilder applies a relevance floor and de-duplicates. Substantive text checks, topical overlap and task rules require answer-like content; a “why” query requires recognized causal/functional wording. Sufficient context is a heuristic boolean, not proof that all facts are supported. Generated factual claims are then judged semantically against the final packet by local Ollama; lexical overlap no longer decides support. Structural checks remain deterministic.

**Semantic claim validation.** Once retrieval, evidence filtering and structured generation finish, each unique extracted factual claim is checked against the final evidence by a separate Ollama-only provider using qwen2.5:7b-instruct. The native strict schema requires `supported`, confidence in [0,1], and a concise reason. Claims pass only at confidence ≥0.80 and `supported=true`. It does not fall back to Groq; errors fail closed. Normal generation still uses Groq then Ollama after provider failures. The Cross-Encoder is retained only as a retrieval reranker.

**Generation and safety boundaries.** Prompt separates task, learner profile and evidence; it says retrieved text is untrusted and only supplied evidence may support facts in materials-only mode. Structured JSON output is checked for object form, required array counts, Pydantic schema and bounded numeric fields. Unsupported claims trigger one stricter retry; if still mixed, only independently validated coherent content may be retained. Otherwise the API returns a grounding-validation failure while retaining sufficient retrieval status and sources. Sources are assigned locally.

**Evaluation status.** The current full test suite passed 134 cases in 40.71 seconds with two dependency deprecation warnings. Test provider interactions are mocked; some local retrieval tests exercise Chroma/models. The saved retrieval benchmark has 11 multi-document queries and two negatives, and the live local semantic claim benchmark has 23 cases (10/10 supported claims accepted; 13/13 unsupported claims rejected); both are useful for engineering regression, not statistical generalization. An older benchmark report calls dense-only the current production retriever, conflicting with current source; the app and tests use hybrid Cross-Encoder retrieval.

## Likely professor questions

### Why use hybrid retrieval?

README's stated rationale is that semantic retrieval can find related wording while BM25 is helpful for exact words, names and rare terms. The code implements both and fuses their ranks. This does not prove hybrid is universally better; the saved small benchmark compares variants under two different protocols.

### Why not use vector search only?

Dense vectors may miss exact rare names or wording. BM25 provides a second candidate path. Conversely, BM25 alone can miss a paraphrase; the project combines them.

### What does RRF contribute?

RRF combines positions rather than raw dense/BM25 scores. A candidate gets 1/(60+rank) from each list where it appears. This avoids treating unlike score magnitudes as directly comparable.

### Why use a Cross-Encoder?

The embedding model encodes query and passage independently for efficient broad search. The Cross-Encoder reads a query/passage pair together and scores their relevance. It only scores up to 30 candidates instead of the whole corpus.

### Why is the Cross-Encoder slower?

It performs model inference for each candidate pair. In the saved CPU run, it averaged 678.6 ms versus 45.7 ms dense retrieval and 7.1 ms BM25. The run excludes model loading; different hardware/load may differ.

### Why an Ollama Semantic Claim Judge?

It runs locally after generation to check claims against retrieved evidence, so claim validation uses the existing Ollama model without an external validation API or extra per-retrieved-chunk calls. The Cross-Encoder scores retrieval relevance and is not an entailment model. The judge has a strict schema, a confidence threshold, and fails closed. Ollama is the chosen architecture for local claim validation; no comparative accuracy study is claimed.

### Why is Groq used for generation?

Groq is the configured primary, with GPT-OSS model ID. Ollama is fallback on provider error. The repository documents configuration and fallback, but does not record a decision study explaining why Groq was historically selected.

### What happens if the model hallucinates?

Materials-only generation has an evidence-only prompt, Pydantic output validation, local semantic checks for extracted factual claims, one stricter retry and coherent supported-content salvage/fail-closed behavior. It reduces risk but cannot prove hallucinations are impossible.

### How do you know an answer is grounded?

The code requires a usable context, checks generated factual claims against accepted evidence using the Ollama Semantic Claim Judge, then selects sources locally. This is a model judgment rather than a formal proof; sufficient context alone still does not establish every detail.

### What is the difference between sufficient and grounded?

Sufficient means at least one chunk remains after search/evidence filtering. Grounding metadata then marks retrieved learning content grounded. Claim validation is a later check for answer-bearing text. A sufficient context alone does not establish every detail.

### How is retrieval evaluated?

Saved scripts use manually labeled source pages and expand them to indexed chunks. They report Recall@k, Hit Rate@k, MRR and latency; the multi-document benchmark also checks evidence sufficiency for positive/negative queries. Labels and query counts are small.

### What is Recall@K?

It measures what fraction of all labeled relevant chunks appear in the first K results. A query may have multiple relevant chunks, so this differs from Hit Rate@K, which only asks whether at least one was found.

### What is MRR?

For each query, find the first relevant result; reciprocal rank is 1/rank. MRR averages that value, rewarding relevant results near the top.

### What does p95 mean?

It is a latency percentile. About 95% of measured queries are at or below that reported time under that benchmark run; with only 11 queries, it is a coarse small-sample statistic, not a service-level guarantee.

### What if retrieval finds the right page but claim validation rejects it?

The Cross-Encoder score establishes retrieval relevance, not claim entailment. Claim validation is a distinct post-generation step: the local Ollama judge evaluates each extracted factual claim against the final packet. Unsupported or uncertain claims trigger a strict retry; coherent accepted statements may be retained, otherwise the result fails closed while preserving sufficient retrieval and sources.

### Why fail closed?

If the evidence-based judge cannot establish support, a grounding-validation failure is safer than returning an unsupported source-backed claim. Sufficient retrieval and sources remain separately reported.

### What is the current performance bottleneck?

In the saved multi-document CPU benchmark, Cross-Encoder reranking is the largest measured retrieval component. LLM generation/judging latency and concurrent load are not measured by that artifact.

### What are the main limitations?

Small manually labeled benchmarks, CPU reranking latency, heuristic retrieval/evidence filters, probabilistic semantic judgments and model adherence to style guidance, general fallback answers marked not grounded, and no authenticated per-user corpus isolation.

## Presentation cautions

- Say “the tested 11-query benchmark reports...” rather than claiming universal retrieval quality.
- Say “the Semantic Claim Judge checks generated claims against the final evidence packet” rather than “the system understands all paraphrases.”
- Mention that models are inference-only here; the repository has no training/fine-tuning loop.
- Distinguish local Qwen fallback/judge from remote Groq normal generation.
- Do not repeat the legacy report’s dense-only production statement; it is inconsistent with current source.

