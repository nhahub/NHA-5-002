# Grounding and Validation

The application keeps retrieval sufficiency separate from whether generated facts follow from the retrieved material. Retrieval filters build a final evidence packet. After structured generation, each factual claim is checked against that packet by the local Ollama Semantic Claim Judge.

## Status vocabulary

| Term/status | Meaning |
|---|---|
| Sufficient context | At least one chunk survived ranking, score, substantive-text, topical, task, and context-size filters. It does not prove every generated claim. |
| `retrieval_status` | `sufficient` or `insufficient`; describes evidence availability only. |
| `grounding_status` | Runtime reports `grounded` when useful context exists; it is not a certificate for every claim. |
| `source_mode` | Retrieved material, general-knowledge fallback, or insufficient context. General knowledge is allowed only when the request permits it and filtered evidence is insufficient. |
| `failure_status` | A later generation, output-validation, or grounding-validation failure. It does not rewrite retrieval status or remove retrieved sources. |
| Fail closed | If the semantic judge cannot establish support, the claim is rejected. If no coherent supported response remains, return a safe typed failure. |

Retrieval failure, generation failure, and semantic validation failure are different states. A semantic validation failure after sufficient retrieval keeps `retrieval_status: sufficient`, preserves sources, and uses `failure_status: grounding_validation_failed`. A genuinely insufficient context is the only case returned as `insufficient_context`.

## Evidence filtering before generation

`LearningGenerator._context` keeps the existing pipeline:

1. Retrieve the requested final `top_k` candidates.
2. Apply `ContextBuilder` relevance, de-duplication, and size limits.
3. Run `task_evidence` to require substantive text and query scope.
4. Build the final bounded context and record its retrieval method.

`ContextBuilder` uses the configured relevance floor (default 0.35), drops empty and duplicate chunks, and bounds context text (default 12,000 characters). `task_evidence` retains the existing topical overlap and definition-decoy checks. Why/how questions still need recognized causal or functional evidence in a substantive sentence that also matches the requested topic. These are retrieval-time filters, not the generated-claim entailment decision.

The final evidence packet contains the exact bounded chunk text, source, and page passed to generation and to claim validation. Learner profile fields are not evidence.

## Generation and claim extraction

The normal generator returns structured JSON. Parsing, required fields, Pydantic constraints, numeric ranges, enum values, and requested question/card counts are checked before factual validation. Server-owned grounding, adaptation, generation metadata, and sources are stamped locally.

For evidence-required responses, claim extraction splits answer-bearing text into sentences (including when sentence punctuation precedes a closing quote) and semicolon-separated clauses. Repeated normalized claims are judged once per pass. Extraction covers:

- explanation, key concepts, examples, important points, and common mistakes;
- fact-bearing statements embedded in recommended revision guidance;
- summary and key points;
- MCQ/exam declarative question premises, correct answers, and rationales;
- declarative flashcard fronts and answers;
- evaluation answers, teaching explanations, mismatch feedback, and misconception labels.

Interrogative question text, intentionally false MCQ distractors, metadata labels, and instructional revision topics are not treated as factual assertions. Claim splitting stays sentence-based; it does not build a general NLP parser.

## Local Ollama Semantic Claim Judge

After generation and structured parsing, each extracted claim is checked against the final evidence packet. The judge does not receive the topic or user question: both can prime historically plausible details that the supplied material does not state. The claim and evidence packet provide the validation context. Evaluation feedback may also include the learner's response, explicitly labeled as evidence only of what the learner wrote—not proof of its factual content.

The judge uses the configured local Ollama model, default `qwen2.5:7b-instruct`. It checks the subject/event, action or mechanism, object, property, quantity, location, and causal link separately; every material detail must be stated or clearly entailed. Related evidence is not enough: for example, land fertility and predictable inundation supporting productivity do not establish that a flood deposited a layer of nutrient-rich silt. The judge also verifies claims that attribute words to the evidence and names the missing detail when rejecting a claim. Its internal strict schema is:

```json
{
  "supported": true,
  "confidence": 0.94,
  "reason": "The supplied evidence supports the claim."
}
```

`supported` is a strict boolean, `confidence` is a number in `[0, 1]`, `reason` is a non-empty string up to 500 characters, and extra fields are forbidden. Ollama receives the native JSON Schema. Fenced JSON, malformed JSON, coercible string booleans/numbers, missing fields, and extra fields are rejected; judge output gets no repair retry.

The decision threshold is `SEMANTIC_SUPPORT_THRESHOLD = 0.80` by default (`CLAIM_JUDGE_MIN_CONFIDENCE` can configure it). A claim is accepted only when `supported` is true and confidence is at least the threshold. A false decision is unsupported; a true decision below threshold is uncertain and rejected. Provider, schema, or decoding failure is also closed. The judge's `reason` remains internal and is never copied into learner output.

The judge is a separate direct Ollama provider, outside the normal Groq-to-Ollama generation fallback. It runs only after retrieval, filtering, final-context construction, generation, and structured parsing. Claims are evaluated sequentially; it is not called for each retrieved chunk. If Ollama is unavailable, the generated response is not accepted, but sufficient retrieval and its sources remain intact.

## Retry and salvage

When a claim is rejected, the generator makes one stricter evidence-only generation attempt and includes the rejected claim text as untrusted diagnostic data so the provider can avoid repeating the same unsupported detail. It validates that output again. If the retry is fully supported, it is returned. Otherwise, the application may keep complete, independently accepted claims that remain coherent on their own. It drops rejected statements, fact-bearing revision guidance, dependent sentences whose referents may have been removed, and incomplete quiz items. It can reuse already validated first-attempt claims if the retry supplies no coherent safe response. If no useful answer remains, it fails closed.

The semantic validator never salvages an answer by changing a rejected claim into an unvalidated paraphrase. Citations are assigned only after validation. Lexical overlap in `source_for_claim` ranks citation candidates; it cannot accept or reject a claim.

## Deterministic safeguards retained

The semantic judge replaces lexical claim support as the factuality decision. Deterministic checks remain for:

- strict JSON parsing, complete response schemas, non-empty answer structure, enum/range constraints, and exact requested item counts;
- request validation and provider failure handling;
- source stamping and evidence-bounded citation selection;
- answer-evaluation score/correctness consistency based on the authoritative expected answer.

Retrieval-time text, topic, definition, and causal filters remain as described above. They do not substitute for generated-claim validation. The old 82% salient-token claim-coverage test and its generated-claim causal gate have been removed.

## General-knowledge fallback

In `learning_materials_then_general_knowledge` mode, general knowledge may be used only when filtered evidence is insufficient, and grounding is marked `general_knowledge`/`not_grounded`. When evidence is sufficient, the response remains evidence-bounded and its claims use the same local semantic validation. No claim is described as grounded in uploaded material when the final evidence packet was insufficient.

## Decision flow

~~~mermaid
flowchart TD
    RETRIEVE[Dense + BM25 → RRF → Cross-Encoder] --> FILTER[Evidence filters and final context]
    FILTER --> SUFF{Context sufficient?}
    SUFF -- No, materials-only --> INSUFF[Typed insufficient-context result]
    SUFF -- No, fallback allowed --> GENERAL[General-knowledge generation]
    SUFF -- Yes --> GENERATE[Structured LLM generation]
    GENERATE --> PARSE[Parse and validate response structure]
    PARSE --> CLAIMS[Extract factual sentences/clauses]
    CLAIMS --> JUDGE[Local Ollama semantic judge]
    JUDGE --> ACCEPT{All claims supported at confidence ≥ 0.80?}
    ACCEPT -- Yes --> STAMP[Stamp sources and grounding locally]
    ACCEPT -- No --> RETRY[One strict evidence-only retry]
    RETRY --> RECHECK[Validate retry; salvage coherent approved claims]
    RECHECK --> STAMP
    RECHECK --> FAIL[Safe validation failure; keep sufficient retrieval and sources]
~~~

## Operational logging

`semantic_claim_validation_started` and `semantic_claim_validation_result` record provider/model, claim count, accepted/rejected counts, per-claim decision/confidence, aggregate confidence, latency, and final status. Claim text, evidence packets, credentials, and learner identifiers are not logged by the judge path.

## Limitations

- Context sufficiency is heuristic and can be wrong even when claim validation is careful.
- A language model makes the semantic judgment; reported confidence is not statistically calibrated, and a judge can still accept a related but unsupported mechanism or property.
- Sentence-level splitting does not identify every possible multi-clause proposition.
- The validator checks claims against supplied text only. It cannot establish truth beyond that material.
- General-knowledge fallback with insufficient evidence is explicitly not grounded in the uploaded documents.
