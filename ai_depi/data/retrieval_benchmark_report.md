# Retrieval Benchmark Report

Date: 2026-10-06  
Project: `AI_Personal_Learning`  
Corpus: current persistent Chroma collection `learning_materials_minilm` (661 chunks)

## Executive result

**Winner: CURRENT RETRIEVER.** It has the best measured MRR and Recall@5, while hybrid retrieval improves Hit@5 by 0.04 but lowers MRR and Recall@5. The cross encoder increases Hit@3 but reduces MRR, Recall@5, and Hit@10, with much higher latency. The current retriever is already the shared production implementation, so no production retrieval code change was needed. All six learning endpoints continue to use the same `LearningGenerator` and its single configured retriever.

## A. Current retriever

- Sentence Transformers `sentence-transformers/all-MiniLM-L6-v2`, 384-dimensional normalized query vectors.
- Persistent ChromaDB at `data/chroma`, collection `learning_materials_minilm`, configured `hnsw:space=cosine`.
- Chroma cosine distance `d` is converted to similarity as `clamp(1 - d, 0, 1)`. Results are ordered by that similarity.
- Production request top-k is 5; the benchmark asks this retriever for 10 so every system can be compared at Recall/Hit@10. Chroma fetches up to `max(top_k * 20, 50)` neighbors (bounded by collection size) before filtering out chunks with an incompatible embedding version.
- The query is sent as written to the embedding model; the application does not add lexical query expansion or custom normalization.
- The configured relevance floor is 0.35. It is applied by `ContextBuilder` after retrieval, not by the Chroma query itself. Production then applies content/evidence checks, deduplicates identical text, and caps the assembled context at 12,000 characters.
- Retrieval ranking has no lexical matcher and no reranker. There is lexical overlap logic downstream for evidence filtering and claim/source validation; the test-only `InMemoryRetriever` also has a simple word-overlap score but is not the production retriever.
- Chunking is page-scoped, 1,000 characters with 120 characters of overlap. This benchmark leaves the collection untouched.

The production `_context` flow retrieves top 5, builds a thresholded context, applies `task_evidence`, then builds the final context again. `substantive_text` drops question-form sentences and known boilerplate, requires at least four words and three salient tokens, and keeps declarative prose/headings only when punctuation or a recognized verb is present. Salient tokens remove stop/generic words and normalize common suffixes. `task_evidence` requires at least 50% query-term overlap when there are more than two salient query terms, otherwise full overlap; definition queries reject chunks that explicitly say they do not provide definitions, and causal queries require causal wording. Context sufficiency means at least one chunk survives the relevance floor and those evidence filters. Claim validation later requires at least 82% salient-token coverage, matching numbers, and explicit causal wording when the claim is causal; unsupported claims trigger a stricter retry and then fail closed. Sources are stamped from locally matched chunks rather than model-authored citation fields.

Strength: fast and highest MRR / Recall@5 on this evaluation. Weakness: misses some relevant chunks, particularly on broad paraphrases and longer page labels.

## B. Hybrid retriever

- Semantic side: same MiniLM embedding model and Chroma collection; top 20.
- Lexical side: deterministic local BM25 over the exact 661 stored chunks; lowercase alphanumeric tokenization, `k1=1.5`, `b=0.75`.
- Fusion: Reciprocal Rank Fusion, `score(d) = Σ 1 / (60 + rank(d))` over the semantic and lexical lists, with one-based ranks. Ties sort by chunk ID. The fused pool is capped at 20, and benchmark output is top 10.
- No dependency or paid service was added. The experimental implementation normalizes RRF scores against the top fused score for compatibility with the existing evidence score interface; benchmark ranking uses RRF order. Current scores are cosine-derived similarities, hybrid scores are normalized RRF values, and BM25's candidate scores are normalized within the lexical result list; these score scales are only meaningful within their respective systems.

Result: Hit@5 improved 0.04; MRR decreased 0.039 and Recall@5 decreased 0.013. Warm average latency increased from 23 ms to 41 ms.

## C. Hybrid + cross encoder

- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`, a compact English Sentence Transformers relevance model.
- Candidate path: 20 semantic + 20 lexical → RRF pool of 20 → cross encoder scores only those 20 → output top 10.
- The cross encoder reports its raw relevance logits as the result scores. It does not rescore the 661-chunk collection.
- This runtime has CPU-only PyTorch (`torch.cuda.is_available() == False`), so reranking ran on CPU. The downloaded model weights occupy about 90.9 MB on disk; total model cache was about 183.6 MB. Runtime RSS was not measured.
- On the final run, model initialization took 5.1 seconds with model files already present. The first run, which also resolved/downloaded the model assets, took 35.6 seconds. Per query, retrieval and fusion averaged 41.6 ms, cross-encoder scoring averaged 434.8 ms, and total retrieval averaged 476 ms.

Result: Hit@3 was the best of the three at 0.88, but MRR, Recall@5, and Hit@10 were below CURRENT. Its warm total latency was about 20.5× CURRENT on this CPU-only runtime.

## D. Benchmark dataset and method

- 25 fixed queries, all from `sample_data/Ancient_Egypt_From.pdf`; this is the dominant source (657/661 chunks). The four chunks from the two other PDFs remained in the searchable corpus for all systems.
- Categories: 8 direct factual, 3 paraphrase, 3 conceptual, 6 topic-based, 4 related-concept, and 1 specific-page query.
- Manually selected page labels came from inspecting the PDF's extracted text before running the retrieval benchmark. There are 46 query-to-page labels across 34 unique physical PDF pages. Chroma page metadata uses the same one-based physical PDF numbering.
- Every indexed Ancient Egypt chunk on a labeled page is treated as relevant to that query. This yields 152 query/chunk relevance pairs (2–15 relevant chunks per query). Labels are explicit in the dataset; the result file expands each page label to actual chunk IDs. No result from any retriever was used to define relevance.
- The same 25 queries, full 661-chunk corpus, page labels, and metric implementation were used for all three systems. Models were warmed before timing; model loading is reported separately.
- This is an engineering benchmark, not a statistically powered research study. Page-level labeling can overcount a chunk if a page contains adjacent unrelated text, and there is no independent second annotator.

Metrics are macro-averaged across queries. For query `q`, `Recall@k(q) = |top_k(q) ∩ relevant(q)| / |relevant(q)|`; `Recall@k` is the mean over queries. `Hit@k(q) = 1` if `top_k(q)` contains any relevant chunk, otherwise 0; Hit Rate@k is its mean. `RR(q) = 1 / rank(first relevant result)`, or 0 for no hit; MRR is mean RR. Average latency is the arithmetic mean of 25 warm per-query timings. Zero-hit count means no relevant chunk in the returned top 10.

## E. Comparison

| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Avg latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CURRENT | 0.164 | 0.289 | **0.383** | 0.456 | **0.680** | 0.760 | 0.880 | **0.960** | **0.748** | **23.3 ms** |
| HYBRID | 0.154 | 0.236 | 0.371 | **0.487** | 0.600 | 0.760 | **0.920** | **0.960** | 0.709 | 41.2 ms |
| HYBRID + CROSS-ENCODER | 0.125 | 0.274 | 0.339 | 0.426 | 0.600 | **0.880** | 0.880 | 0.920 | 0.732 | 476.3 ms |

Each system returns 10 ranked output candidates on average. Internally, CURRENT searches 10 semantic candidates; HYBRID searches 20 semantic + 20 BM25 and fuses at most 20; CROSS-ENCODER uses that same pool of 20 and scores all 20 before returning 10. Zero-hit queries in top 10: CURRENT 1, HYBRID 1, CROSS-ENCODER 2.

## F. Selection rule and winner

Before selecting a winner, define a meaningful quality gain as at least +0.05 absolute on one of the primary measures (MRR, Recall@5, Hit@5), with no other primary measure falling by more than 0.02. Changes smaller than that are treated as effectively tied at this 25-query scale; in a tie, choose the simpler and faster system.

No challenger meets the quality-gain rule. HYBRID's +0.04 Hit@5 is below the threshold and accompanies lower MRR and Recall@5. CROSS-ENCODER improves Hit@3 (a secondary measure) but has lower MRR and Recall@5 and much higher latency. **WINNER: CURRENT.** It remains the production retriever.

## G. Production changes

No production retrieval code was changed. That is the benchmark-selected integration: the current Chroma retriever already feeds the shared `LearningGenerator` used by `/learning/explain`, `/learning/summarize`, `/learning/mcq`, `/learning/flashcards`, `/learning/exam`, and `/learning/evaluate-answer`. Existing evidence filtering, sufficiency checks, grounding, claim validation, source stamping, unsupported-query behavior, and 422 request validation remain in place.

Files added:

- `scripts/benchmark_retrieval.py`: deterministic BM25/RRF/cross-encoder experiments and common metrics.
- `data/retrieval_benchmark_queries.json`: fixed questions, categories, manually chosen relevant pages, and labeling notes.
- `data/retrieval_benchmark_results.json`: per-query chunk IDs, pages, ranks, scores, hit flags, timing, and aggregate metrics for every ranker.
- `tests/test_retrieval_benchmark.py`: BM25, RRF, metric formula, and real-page label coverage tests.
- `data/retrieval_benchmark_report.md`: this report.

`tests/test_api.py` was extended with six-endpoint API contract coverage using an offline test provider; no external generation API calls were made for those tests.

## H. Regression and API verification

- `pytest -q`: **61 passed**, 2 dependency deprecation warnings.
- `python -m compileall app tests`: **passed**.
- Restarted local FastAPI/Uvicorn on `127.0.0.1:8000`; startup PDF sync skipped all three unchanged PDFs and reported zero failures.
- `/health`: HTTP 200, status `ok`.
- `/ready`: HTTP 200, status `ready`, Chroma collection `learning_materials_minilm`, 661 chunks.
- All six learning endpoints were exercised for a supported Ancient Egypt request, an unsupported request, and an invalid empty request. Supported: HTTP 200, sufficient/grounded, sources present, test-provider generation metadata present. Unsupported: HTTP 200, insufficient/not_grounded, no sources, generation null. Invalid: HTTP 422. All of these were verified through FastAPI's in-process test client to avoid making billable generation calls.
- Existing fallback-provider error coverage remains enabled in the suite; the supported API contract test reported `fallback_used=false` because it uses the local test provider.

## I. Limitations

- Small, manually page-labeled dataset; results are useful for this project’s engineering choice but do not establish statistical significance.
- Page-level relevance is approximate at chunk granularity and may include unrelated adjacent chunks on a labeled page.
- Latency is local warm retrieval timing from this Windows/Python environment; cross-encoder measurements used CPU because CUDA was unavailable. Different hardware and loaded-service concurrency can change latency.
- Cross-encoder is English-trained; its scores are raw relevance logits and should not be interpreted as calibrated probabilities.
- Runtime model RSS and GPU memory were not measured. The cross encoder was not integrated into production because it did not meet the selection rule.
