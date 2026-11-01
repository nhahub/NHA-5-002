# RAG and Retrieval

This guide explains the implementation in app/rag.py, its callers in app/generation.py, evidence processing in app/evidence.py, and saved benchmark artifacts.

## 1. RAG from zero

**Simple explanation:** A language model can answer from its training. Retrieval-Augmented Generation first finds passages in the learning PDFs and gives those passages to the model with the request.

**In this project:** PDFs are extracted, split into page-local chunks, embedded and stored in Chroma. At request time, dense vector search and BM25 produce separate ranked lists. RRF combines ranks, a Cross-Encoder reorders a bounded pool, then task-specific evidence filters select context.

~~~mermaid
flowchart TD
    PDF[PDF pages] --> EXTRACT[pypdf text extraction]
    EXTRACT --> CHUNK[Page-local chunks]
    CHUNK --> META[Page and source metadata]
    CHUNK --> EMBED[Embedding model]
    EMBED --> CHROMA[(Persistent Chroma)]
    CHUNK --> SNAPSHOT[BM25 snapshot]
    QUERY[Request query] --> DENSE[Dense retrieval]
    CHROMA --> DENSE
    QUERY --> BM25[BM25 sparse retrieval]
    SNAPSHOT --> BM25
    DENSE --> RRF[RRF fusion]
    BM25 --> RRF
    RRF --> CE[Cross-Encoder reranking]
    CE --> EVIDENCE[Top-k evidence filters]
    EVIDENCE --> CONTEXT[LLM context]
~~~

## 2. Ingestion and chunks

PDF extraction uses pypdf. Pages are independent: chunks do not cross page boundaries, original physical page numbers start at 1, and pages without text are omitted. Defaults are chunk_chars=1000 and overlap_chars=120. The chunker prefers a space near the requested end when it does not make the chunk much shorter. Overlap means the next chunk starts before the previous one ends.

A stable chunk ID is SHA-256 over normalized source path, page, start character and chunk text. Metadata includes source path/name, file hash, page, start offset, chunk ID, and embedding provider/model/version. Embeddings are computed in batches of 32, normalized, then explicitly stored in Chroma.

Directory synchronization:
- recursively sorts PDF paths;
- hashes each PDF and skips unchanged content;
- prepares embeddings/upserts replacement chunks before deleting stale IDs for that exact source path;
- removes chunks for PDFs that disappeared under the synchronized root, without deleting unrelated paths;
- takes a cross-process lock;
- logs/counts per-file errors and continues with other files.

During application lifespan, sync runs via a worker thread if AUTO_INGEST_PDFS is true. After the attempt, BM25 snapshot refresh runs in finally. scripts/ingest_sample.py calls the same synchronizer. There is no HTTP ingestion endpoint.

## 3. Embeddings and dense retrieval

### Embedding model

**Simple:** The embedding model turns each piece of text into a list of numbers so Chroma can find related meanings even if the wording differs.

**Technical:** The configured Sentence Transformers model is sentence-transformers/all-MiniLM-L6-v2. SentenceTransformerEmbeddingProvider loads a cached model, asks it for the vector dimension, encodes normalized vectors with batch size 32, and returns Python lists. Ingestion and query use the same configured model/version. The saved report records 384 dimensions; the runtime reads the value rather than hard-coding it.

The model runs locally. The application does not pass a device argument to the embedding model, so Sentence Transformers selects the device. No fine-tuning path exists.

### Chroma and dense retrieval

The persistent local path defaults to ./data/chroma and collection to learning_materials_minilm. Collection metadata records embedding provider/model/version and hnsw:space=cosine. Initialization raises ValueError if stored identity does not match current settings.

At query time, the query is embedded and sent to Chroma. Production requests dense_candidate_k=30, raised when necessary to satisfy an explicit final top-K. Chroma fetches min(collection_count, max(top_k × 20, 50)) rows to allow embedding-version filtering while retaining enough usable candidates. It returns text, metadata, ID and distance. The code converts cosine distance d to a clamped interface score:

~~~text
similarity = clamp(1 - d, 0, 1)
~~~

This is a vector similarity value, not the probability that a response claim is true.

~~~mermaid
flowchart LR
    QUERY[Query text] --> ENCODER[all-MiniLM-L6-v2]
    ENCODER --> V[Normalized vector]
    V --> DB[(Chroma cosine collection)]
    DB --> ROWS[Text, metadata, distances]
    ROWS --> FILTER[Drop embedding-version mismatch]
    FILTER --> SCORE[Clamp 1 minus distance]
    SCORE --> TOP[Top 30 dense results]
~~~

## 4. BM25 sparse retrieval

**Simple:** BM25 ranks passages by matching words, with stronger weight for informative/rare terms and a document-length adjustment.

**Project-specific implementation:** SparseBM25Index is custom Python in app/rag.py. On construction/refresh it reads compatible chunks from Chroma into a process-local snapshot. Tokens use casefold plus the regular expression [a-z0-9]+. A small normalizer:
- changes words longer than five characters ending ies to y;
- removes ing from words longer than five;
- removes ed from words longer than four;
- removes a final s from words longer than four.

No stop-word list is used in BM25. The index records token counts, document length, average length, and document frequency.

Parameters are k1=1.5 and b=0.75. The term contribution is:

~~~text
idf(t) × tf(t,d) × (k1 + 1)
-----------------------------------------------
tf(t,d) + k1 × (1 - b + b × |d| / average_document_length)
~~~

~~~text
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
~~~

tf is term frequency, df is the number of chunks containing the term, N is corpus size, and |d| is chunk token length. Candidates with positive score are sorted by score descending, chunk ID for ties, then limited to 30. Returned BM25 scores divide by the maximum score in that result list.

The in-memory snapshot refreshes after PDF sync. It is not a persistent second index; all compatible Chroma records are scanned during refresh.

## 5. Hybrid retrieval and RRF

**Why not vector-only?** Semantic search can find related wording but can rank an exact rare name lower.

**Why not BM25-only?** Exact terms are useful, but a PDF and a question can describe the same concept differently.

**What RRF does:** It combines the ranked lists by rank position instead of comparing unrelated score magnitudes.

~~~mermaid
flowchart LR
    DENSE[Dense list, up to 30] --> UNION[Deduplicate by chunk ID]
    SPARSE[BM25 list, up to 30] --> UNION
    UNION --> RRF[Sum reciprocal rank votes, k=60]
    RRF --> SORT[Score descending, ID tie-break]
    SORT --> POOL[Keep up to 30]
~~~

Formula:

~~~text
RRF(d) = Σ over lists L containing d of 1 / (k + rank_L(d))
~~~

Current k is 60 and ranks start at 1. A chunk found in both lists gets both contributions. Distinct union can reach 60; rerank_candidate_k caps it at 30 by default and expands to satisfy a larger final top-K. The fused score is normalized by the highest fused score and stored as hybrid_score. The Cross-Encoder later replaces relevance_score with its own transformed score.

The production API dispatches through the `RETRIEVAL_STRATEGY` factory. The setting currently permits only `hybrid_cross_encoder`, so there is one supported runtime choice.

## 6. Cross-Encoder

**Simple:** Embedding search compares vectors made separately from query and passage. A Cross-Encoder reads the query and one passage together and scores their relationship.

**Technical:** The configured model is cross-encoder/ms-marco-MiniLM-L-6-v2. The retriever builds (query, chunk.content) pairs and calls predict with batch size 16 and no progress bar. Model max_length is 512. It scores no more than 30 rows, sorts descending by raw output, and returns at most five.

The model loads lazily on the first nonempty rerank request. Device auto selects CUDA when torch reports availability, otherwise CPU. If CUDA is requested but unavailable, the code logs and uses CPU. The loader is cached by model/device; a lock avoids duplicate concurrent first loads. Saved benchmark artifacts report CPU execution.

Raw Cross-Encoder logits are not cosine similarity and may be negative. The raw score is stored as cross_encoder_raw_score and cross_encoder_score. Final interface score:

~~~text
calibrated = raw_score + 5.0
relevance_score = 1 / (1 + exp(-clip(calibrated, -60, 60)))
~~~

The default offset is 5.0. The sigmoid is monotonic and does not alter sort order; it is not a calibrated probability. ContextBuilder compares this value with 0.35.

### Worked pair example

Query: “What crops were chiefly cultivated in ancient Egypt?”  
Passage: “Agriculture centered on cultivation of cereal crops, chiefly emmer wheat and barley.”

The embedding model independently encodes query and passage for nearest-neighbor search. The Cross-Encoder receives the query and full passage as one pair and returns a relevance logit. It does not generate an answer.

## 7. Candidate counts and score meanings

| Stage | Current setting | Meaning |
|---|---:|---|
| Dense returned candidates | 30 | Converted cosine score; Chroma fetch may be up to max(600,50), bounded by corpus, before version filtering. |
| Sparse candidates | 30 | BM25 score, normalized within returned list. |
| RRF inputs | Up to 30 + 30 | Rank contributions with k=60. |
| Fused pool | At most 30 | RRF normalized score. |
| Cross-Encoder pairs | At most 30 | Raw relevance logit. |
| Final retriever result | 5 by default; optional request top_k 1–20 | Sort by raw logit; relevance_score is sigmoid(raw+configured offset). |
| ContextBuilder | Up to request/configured final top-K | Require score ≥0.35, unique nonempty text, bounded chunk text and total character cap. |
| Provider evidence packet | Selected chunks | Uses the same bounded chunk text that passed evidence filtering. |

If the corpus is smaller, fewer candidates are available. If BM25 has no match but dense results exist, RRF uses dense-only rows. If both lists are empty, no Cross-Encoder model load is needed.

## 8. Evidence and context are separate from retrieval

Generation applies ContextBuilder once, then task_evidence, then ContextBuilder again. The first pass applies score floor, deduplication and text limit. Evidence processing removes headings/questions/boilerplate and requires substantive, topical prose; a why query additionally requires causal/functional wording. Context sufficiency means at least one chunk remains after these steps. It is not proof that all answer claims are supported.

ContextBuilder now truncates the selected chunk contents themselves to the configured character budget, counting separators as well. Since generation._real builds its evidence packet from those bounded chunks, the actual provider packet is limited too. Final evidence count can be overridden per request with `top_k` (1–20); when that is larger than configured dense/sparse/rerank pools, those effective pools expand to at least the requested count.

## 9. Saved retrieval evaluation

The latest data/hybrid_cross_encoder_results.json records:
- 661 chunks; 11 labeled queries (9 positive, 2 negative);
- top 10 benchmark outputs, while production top_k is 5;
- 30 dense, 30 sparse, RRF k=60, fused/rerank cap 30;
- CPU Cross-Encoder;
- positive-query Recall@5 0.611, Hit@5 1.000, MRR 0.778;
- average total latency 731.75 ms and p95 1,072.32 ms;
- stage average 45.73 ms dense, 7.08 ms sparse, 0.10 ms fusion, 678.64 ms Cross-Encoder;
- all positive contexts passed its gate and both negative contexts failed the gate despite candidate rows.

These are saved results, not rerun during the audit. The legacy data/retrieval_benchmark_report.md uses a different 25-query Ancient-Egypt-only set, 20+20 candidates, pool 20, top 10, standalone raw-logit reranker. It calls dense-only “CURRENT” and says Cross-Encoder was not integrated, contradicted by current app/main.py, README, and tests. Treat that narrative as stale architecture documentation and the results as a separate protocol.

## 10. Trade-offs and component removal

README documents hybrid retrieval because semantic matching and exact lexical matching complement each other. It documents RRF as a way to combine ranks without comparing unlike raw score scales. It does not preserve original alternative-selection experiments for these models/database.

| Component | Trade-off or removal consequence |
|---|---|
| Dense search | Supports paraphrase matching; removing it leaves lexical BM25 only. |
| BM25 | Emphasizes exact names/terms; removing it loses sparse matches. |
| RRF | Rank-based fusion avoids score calibration but discards score margin; removing it means choosing one list or another method. |
| Cross-Encoder | Joint query-passage ordering improves rerank signal but costs CPU latency; removing it is faster and returns RRF order. |
| Chroma | Persistent local vectors/metadata; removing it requires a new durable vector store and BM25 source. |
| Evidence filters | Reduce headings/boilerplate/off-topic/noncausal context; removing them lets more retrieval decoys through. |

## 11. Generated-answer grounding validation

The retrieval pipeline remains dense search + BM25 → Reciprocal Rank Fusion → Cross-Encoder → final top-K → task evidence filters → bounded context. The Cross-Encoder ranks query/passage relevance; its score is not an entailment decision.

After that final evidence packet is sent to the configured generation provider and a structured response is parsed, `LearningGenerator` extracts declarative factual claims from learner-facing fields, including fact-bearing recommended-revision text. It checks those claims one at a time with the local Ollama Semantic Claim Judge (`qwen2.5:7b-instruct` by default). The judge sees only the claim and final evidence packet, omitting topic and question to avoid priming plausible details not present in the material. It checks each material detail (event, mechanism, property, quantity, location and causal link) separately, so a stated outcome cannot stand in for an unstated mechanism or property. It must return strict schema-conforming JSON. The claim passes only when `supported=true` and confidence is at least `0.80`; uncertainty, malformed output, or Ollama failure fails closed. The judge's reason remains internal.

This replaces lexical token coverage as the generated-claim support gate. Lexical overlap still has limited retrieval and citation-ranking uses, but does not decide whether a generated claim follows from evidence. A single strict evidence-only retry is allowed; rejected claims are passed to the generation provider as untrusted diagnostic strings to discourage repetition. Fact-bearing revision guidance is checked and removed along with other rejected text. Validated coherent statements can be retained if retry output is still mixed, otherwise the route returns a typed grounding-validation failure while preserving sufficient retrieval status and sources. Structural checks remain deterministic. See [GROUNDING_AND_VALIDATION.md](GROUNDING_AND_VALIDATION.md) for extraction fields and detailed failure behavior.

