# Glossary

Each entry starts with a beginner explanation, then says what the term means in this project, then gives a small example.

## API

- **Simple:** A set of rules that lets one program ask another program to do something.
- **Here:** The HTTP interface used by a learner app to request learning content and receive JSON.
- **Example:** A client sends a POST request to /learning/explain.

## FastAPI

- **Simple:** A Python tool for building web services.
- **Here:** It defines the eight routes, validates request bodies through Pydantic models, exposes API docs, and serializes declared response models.
- **Example:** FastAPI routes an explain request to the explain handler.

## Endpoint / route

- **Simple:** A named address where a service handles a kind of request.
- **Here:** Examples are GET /health and POST /learning/evaluate-answer.
- **Example:** The exam request goes to /learning/exam.

## JSON

- **Simple:** A text format for sending named values and lists.
- **Here:** The API accepts and returns JSON objects; LLM output is expected to be JSON too.
- **Example:** {"topic":"Ancient Egypt"}.

## Pydantic

- **Simple:** A Python library that checks whether data has the expected shape and values.
- **Here:** Request schemas set types and constraints; response schemas validate API and model output.
- **Example:** number_of_questions must be between 1 and 50 for MCQRequest.

## Schema

- **Simple:** A rulebook describing which fields an object should contain.
- **Here:** ExplanationResponse requires explanation and arrays, plus grounding/adaptation metadata.
- **Example:** AnswerEvaluation requires score from 0 to 1.

## RAG (Retrieval-Augmented Generation)

- **Simple:** Search trusted material first, then let a language model use the found passages.
- **Here:** The service searches indexed PDFs before generating learning responses.
- **Example:** Find the Nile page, then ask the model to explain it.

## PDF ingestion / indexing

- **Simple:** Reading a PDF and preparing its contents so search can find them later.
- **Here:** pypdf extracts page text; the code chunks, embeds and stores it in Chroma.
- **Example:** Page 16 text becomes one or more searchable chunks.

## Chunk

- **Simple:** A manageable piece cut from a larger document.
- **Here:** Default page-local text segment is about 1,000 characters with 120-character overlap.
- **Example:** Two neighboring chunks share a short boundary passage.

## Metadata

- **Simple:** Extra facts that describe some content.
- **Here:** Each chunk can store document name/path, page number, file hash, chunk ID and embedding version.
- **Example:** A source record says Ancient_Egypt_From.pdf, page 16.

## Embedding

- **Simple:** A list of numbers that represents the meaning of text.
- **Here:** all-MiniLM-L6-v2 converts PDF chunks and user queries to normalized vectors for Chroma.
- **Example:** “annual flooding” and “yearly inundation” can map to nearby vectors.

## Vector

- **Simple:** A list of numbers with a direction and length.
- **Here:** The embedding model emits vectors used in nearest-neighbor search.
- **Example:** A passage may have a 384-number vector; code reads dimension from the model, saved report records 384.

## Semantic similarity

- **Simple:** How close two pieces of text are in meaning, even if words differ.
- **Here:** Dense retrieval uses embedding vectors to find related passages.
- **Example:** A query about yearly flooding can find a chunk using the word inundation.

## Dense retrieval

- **Simple:** Search that compares number-vectors representing text.
- **Here:** Chroma finds nearby query vectors in cosine space and returns up to 30 candidates.
- **Example:** Search “Nile and crop growth” can find related language without exact wording.

## Sparse retrieval

- **Simple:** Search that pays attention to which words appear in each passage.
- **Here:** The sparse path is custom BM25 over a process-local snapshot of Chroma chunks.
- **Example:** Searching an exact rare name can work well if that name occurs in the chunk.

## BM25

- **Simple:** A word-matching ranking formula that gives weight to informative words and adjusts for passage length.
- **Here:** Custom implementation uses k1=1.5, b=0.75, term frequency, document frequency and token normalization.
- **Example:** A chunk with “emmer wheat” may rank for a query containing those words.

## Hybrid retrieval

- **Simple:** Use more than one search style and combine their results.
- **Here:** Dense Chroma results plus BM25 results are fused with RRF.
- **Example:** A paraphrase may arrive from dense search while the exact place name arrives from BM25.

## Token / tokenizer

- **Simple:** A small text piece, often a word or word part; a tokenizer splits text into these pieces.
- **Here:** BM25 uses lowercase alphanumeric word tokens and suffix cleanup. Evidence matching uses a separate salient-token function.
- **Example:** “crops” can normalize to “crop” for BM25.

## Term frequency (TF)

- **Simple:** How many times a word appears in a passage.
- **Here:** BM25 includes the count of a query term in the candidate chunk.
- **Example:** “Nile” appearing twice has a term frequency of two.

## Document frequency (DF)

- **Simple:** How many documents contain a word.
- **Here:** BM25 uses chunk-level document frequency to estimate whether a term is common or distinctive.
- **Example:** A rare Pharaoh name has lower DF than “river.”

## IDF

- **Simple:** A weight that makes rarer words more informative.
- **Here:** BM25 computes a smoothed log inverse document frequency from the chunk corpus.
- **Example:** A term in few chunks can contribute more than a term in every chunk.

## RRF (Reciprocal Rank Fusion)

- **Simple:** A voting rule that gives better-ranked search results more points.
- **Here:** Each list contributes 1/(60+one-based rank); same chunk can get a vote from both lists; pool is capped at 30.
- **Example:** A chunk ranked second in dense and first in BM25 receives both contributions.

## Reranking

- **Simple:** Sort already-found passages again using a more detailed check.
- **Here:** The Cross-Encoder reranks the at-most-30 RRF candidates.
- **Example:** A page with closer query wording can move above a passage that only matched one keyword.

## Cross-Encoder

- **Simple:** A model that reads a question and passage together and scores how well they match.
- **Here:** ms-marco-MiniLM-L-6-v2 returns a raw relevance logit for each query/chunk pair.
- **Example:** Pair (“Which crops?”, “Emmer wheat and barley...”) receives one score.

## Bi-encoder

- **Simple:** A model that encodes two texts separately before comparing their numbers.
- **Here:** The embedding model encodes query and chunk separately for Chroma search.
- **Example:** Make one vector for “Nile crops” and another for a PDF paragraph, then compare them.

## Top-K

- **Simple:** The first K results from a ranking.
- **Here:** Production retriever returns top 5 after reranking; benchmark asks top 10.
- **Example:** Top 5 means ranks 1 through 5.

## Candidate pool

- **Simple:** The set of items that are considered before choosing the final few.
- **Here:** Dense and sparse can contribute up to 60 unique chunks; RRF keeps at most 30 for Cross-Encoder.
- **Example:** The reranker sees no more than 30 passages for one query.

## Evidence

- **Simple:** Material that can support an answer.
- **Here:** Accepted retrieved chunk text after score, content, topic and task-specific filtering.
- **Example:** A declarative page-16 sentence about Nile inundation.

## Grounding

- **Simple:** Keeping an answer tied to the source material.
- **Here:** Metadata summarizes accepted retrieval context; claim validation and local source stamping are separate checks.
- **Example:** A response can include an Ancient Egypt page citation.

## Context

- **Simple:** The background information given to a model with a question.
- **Here:** The final selected learning chunks and associated request/profile instructions.
- **Example:** A prompt includes the page passage and asks for a summary.

## Prompt

- **Simple:** The instructions and information sent to a language model.
- **Here:** PromptEngine combines task, request, profile, adaptation, evidence, knowledge-mode constraints and output rules.
- **Example:** “Use only these passages and return JSON.”

## LLM (Large Language Model)

- **Simple:** A model that can produce text based on patterns learned from lots of examples.
- **Here:** Groq-hosted GPT-OSS normally generates educational outputs; local Qwen via Ollama is fallback and judge.
- **Example:** The model writes an explanation from retrieved evidence.

## Inference

- **Simple:** Using a trained model to produce an output.
- **Here:** Models are used for embeddings, reranking, generation and judging; repository has no training code.
- **Example:** Ask Qwen to evaluate one claim.

## Structured output

- **Simple:** Model output forced into an expected data shape.
- **Here:** Providers receive schemas when possible, JSON is parsed, required counts checked, and full Pydantic models validated.
- **Example:** An MCQ response must contain the expected questions array.

## JSON Schema

- **Simple:** A machine-readable description of valid JSON fields/types.
- **Here:** Compact schema goes to Groq/Ollama; server-owned provenance fields are omitted and stamped locally.
- **Example:** Claim judge schema requires supported, confidence and reason.

## Confidence

- **Simple:** A number expressing how sure a model says it is.
- **Here:** Judge confidence must be in [0,1] and at least 0.80; it is model-reported, not calibrated probability.
- **Example:** 0.92 passes the configured threshold if supported is true.

## Fail-closed

- **Simple:** If a safety check is unsure, stop instead of guessing.
- **Here:** Unsupported/uncertain expected answer returns insufficient evaluation and skips normal generation.
- **Example:** Ollama judge unavailable means claim rejection.

## Fallback

- **Simple:** A backup used after a primary option fails.
- **Here:** Ollama generation is tried after Groq raises ProviderError. It is not used merely because Groq output is malformed JSON.
- **Example:** Groq times out, then the normal generation wrapper calls Ollama.

## Provider

- **Simple:** A service or adapter that supplies a model response.
- **Here:** GroqProvider and OllamaProvider implement transport; FallbackLLMProvider manages ordinary generation order.
- **Example:** Generation metadata can report provider=ollama with fallback_used=true.

## ChromaDB

- **Simple:** A database designed to store/search vectors with associated content.
- **Here:** Persistent local store of PDF chunks, embeddings, IDs and page/source metadata.
- **Example:** Chroma returns page text plus its original page number.

## Cosine similarity / cosine distance

- **Simple:** A way to compare the direction of two vectors; closer directions are more similar.
- **Here:** Collection is configured for cosine distance, and app converts distance d to clamp(1-d,0,1).
- **Example:** Smaller distance becomes a larger dense similarity score.

## Relevance score

- **Simple:** A number used to order or gate search results.
- **Here:** Meaning depends on stage: cosine-derived dense score, normalized BM25/RRF values, raw Cross-Encoder logit, or sigmoid(logit+5) final interface score.
- **Example:** Do not read Cross-Encoder score 4.2 as 420% confidence.

## Claim validation

- **Simple:** Check whether an answer statement is supported by the passage.
- **Here:** After structured generation, the local Ollama Semantic Claim Judge checks factual claims against the final evidence packet; claims pass only if supported=true and confidence ≥0.80. Lexical overlap ranks citation candidates but does not decide claim support.
- **Example:** A page says “inundation supported agriculture”; a claim adds an unsupported military cause and should fail.

## Semantic Claim Judge

- **Simple:** A model asked whether a claim means the same supported idea as the passage.
- **Here:** Ollama-only structured judge checks extracted generated claims after retrieval and generation, across evidence-required learning routes. It receives evidence and topic orientation, not the question, and fails closed on unsupported, uncertain, malformed, or unavailable judgments.
- **Example:** It may accept “annual flooding helped farmers” against evidence using “annual inundation.”

## Source / provenance

- **Simple:** Where information came from.
- **Here:** LearningSource includes document, page, chunk ID, snippet and relevance score, assigned from retrieved evidence.
- **Example:** Ancient_Egypt_From.pdf, page 16.

## Latency

- **Simple:** How long an operation takes.
- **Here:** Benchmark reports per-query retrieval time in milliseconds; it does not include normal LLM generation in the current retrieval artifact.
- **Example:** Cross-Encoder stage average was about 679 ms in the saved CPU run.

## p95 latency

- **Simple:** A high-end typical time; about 95 percent of measured items are at or below it in that sample.
- **Here:** Script sorts latencies and chooses index ceil(0.95×n)-1; current sample has only 11 queries.
- **Example:** Saved total retrieval p95 is about 1,072 ms, not a service guarantee.

## Recall@K

- **Simple:** What fraction of all relevant passages were found in the first K results.
- **Here:** For each labeled query, relevant retrieved chunk count divided by all labeled chunks on its expected pages, averaged over positive queries.
- **Example:** If there are four relevant chunks and top five contains two, Recall@5 for that query is 2/4=0.5.

## Hit Rate@K

- **Simple:** How often at least one right passage appears in the first K results.
- **Here:** Mean of a yes/no hit over positive benchmark queries.
- **Example:** One correct chunk in top five is a hit even if other relevant chunks are missing.

## Reciprocal Rank and MRR

- **Simple:** Give more credit when the first right answer appears near the top.
- **Here:** Reciprocal rank is 1/rank of first relevant chunk; MRR is its mean over positive queries.
- **Example:** First relevant result at rank 2 has reciprocal rank 0.5.

## HTTP 422 / 502 / 503

- **Simple:** Standard web-service status numbers describing request or service failure.
- **Here:** 422 for invalid request schema, 502 for LLM provider/structured output failure, 503 when Chroma readiness count fails.
- **Example:** Empty JSON body to an endpoint with required fields returns 422.

## Batch size

- **Simple:** How many items a model processes together.
- **Here:** Embedding encode batch defaults to 32; Cross-Encoder prediction batch is 16.
- **Example:** Reranking 30 passages uses batches of up to 16.

## Temperature

- **Simple:** A setting that changes how varied model outputs may be.
- **Here:** Normal generation default is 0.2; Semantic Claim Judge explicitly uses 0.0.
- **Example:** The judge uses the lower setting when returning supported/confidence/reason.

## Confidence versus score

- **Simple:** Confidence says how sure a model claims to be; score represents a different output measure.
- **Here:** Claim-judge confidence controls acceptance threshold. AnswerEvaluation score is generated grading output constrained to 0–1; neither is a calibrated probability.
- **Example:** A judge confidence of 0.9 does not mean a student has 90% probability of being correct.

