# Endpoint Parameter Behavior Audit

**Audit date:** 2026-10-06  
**Runtime path:** FastAPI request model → `LearningGenerator` → hybrid retrieval/context → adaptation and prompt rules → Groq/Ollama provider → schema/evidence validation → response stamping.

This audit records the implemented contract after the parameter-sensitivity corrections. A parameter's presence in a request or prompt is not counted as proof by itself: the tables identify the concrete consumer and the observable decision it controls. API request data is bounded by `MAX_INPUT_CHARS` (default 20,000 decoded characters) before FastAPI parses it.

## Endpoint inventory

| Method and route | Request contract | Path/query parameters | Notes |
|---|---|---|---|
| `GET /health` | None | None | Returns service status and configured provider name. |
| `GET /ready` | None | None | Reads Chroma collection count; returns 503 if Chroma is unavailable. |
| `POST /learning/explain` | `ExplanationRequest` | None | Generates a grounded explanation. |
| `POST /learning/summarize` | `SummaryRequest` | None | Produces a selected summary style. |
| `POST /learning/mcq` | `MCQRequest` | None | Generates exactly the requested number of questions when valid generation succeeds. |
| `POST /learning/flashcards` | `FlashcardRequest` | None | Generates exactly the requested number of cards when valid generation succeeds. |
| `POST /learning/exam` | `PracticeExamRequest` | None | Generates an exam over the supplied topic list and time budget. |
| `POST /learning/evaluate-answer` | `AnswerEvaluationRequest` | None | Grades the submitted answer against the provided expected answer or an evidence-derived answer. |

No learning route currently has URL query parameters or path parameters. `top_k` below is a JSON-body parameter added for all six learning endpoints; omitting it uses the configured production final top-K.

## Classification key

- **Behavioral:** changing the value changes retrieval, generation policy, output shape, or grading.
- **Contextual:** the value steers adaptation only when it applies to the requested subject/task.
- **Metadata:** identifies or describes a request but does not intentionally change generated lesson behavior.

## Shared request fields

All six learning request models inherit `top_k` and accept `learner_profile`.

| Parameter | Type, default, validation | Classification | Consumer and functional effect | Sensitivity proof |
|---|---|---|---|---|
| `top_k` | Optional integer, default `null` (use configured `FINAL_TOP_K`/`TOP_K`); 1–20 when supplied | Behavioral | `LearningGenerator._context` forwards the effective value to `HybridCrossEncoderRetriever.retrieve`. Dense and sparse candidates and the rerank pool expand as needed to return the requested final count, subject to available evidence and configured corpus/candidate limits. | `test_swagger_top_k_reaches_retrieval_and_bounds_evidence` and `test_hybrid_final_top_k_is_respected_even_above_default_rerank_pool` exercise 3, 5, and 10. |
| `learner_profile` | Object; defaults to an empty `LearnerProfileIn`; unknown nested fields are ignored | Contextual | `adapt()` turns relevant signals into explicit `AdaptationMetadata` and task instructions. Only matching-topic records are included in prompt guidance. | `test_each_learner_signal_has_topic_scoped_guidance_and_user_id_is_metadata`. |

### Nested `learner_profile` fields

| Field | Type, default, validation | Classification | Functional effect and limits |
|---|---|---|---|
| `user_id` | Optional string, default `null`, max 200 characters | Metadata | Not sent to the generation prompt and not logged. It does not alter the response. |
| `overall_level` | String, default `intermediate`, length 1–50 | Contextual | Adds explicit beginner/intermediate/advanced vocabulary and explanation guidance. Recognized beginner aliases: `beginner`, `novice`, `basic`; advanced aliases: `advanced`, `expert`, `proficient`; other values receive intermediate guidance. |
| `topic_mastery` | Map of topic string to float in [0, 1], default `{}` | Contextual | `mastery_for()` finds a matching requested topic. Below 0.50 adds foundational scaffolding and contributes a one-step effective-difficulty reduction; at or above 0.85 avoids basic reteaching and contributes a one-step increase. Requested difficulty remains recorded separately. |
| `weak_topics` | List of strings, default `[]` | Contextual | Matching topics are prioritized in explanation/remediation guidance and recommendations. In exams, `weak_topic_emphasis=true` allocates more items to matching weak topics; `false` explicitly prevents that bias. Unmatched weak topics are ignored. |
| `strong_topics` | List of strings, default `[]` | Contextual | Matching topics receive less introductory reteaching; instructions favor the requested new angle or a supported application. |
| `recent_quiz_results` | List of arbitrary dictionaries, default `[]` | Contextual | Only topic-matching records are summarized for the model. Recognized normalized score/accuracy values below 0.50 add an intermediate-step/scaffolding instruction. `percentage`/`percent` accepts 0–100; other rates are 0–1; point scores normalize only with a positive `total`/`max_score`. Unrelated records do not enter the prompt. |
| `recent_exam_results` | List of arbitrary dictionaries, default `[]` | Contextual | Same topic-scoped behavior as recent quiz records; relevant exam performance helps target remediation. |
| `historical_performance` | Dictionary of arbitrary values, default `{}` | Contextual | Topic-keyed historical records are included only when they match the requested topic; recognized low scores add scaffolding. |
| `question_accuracy` | Optional float in [0, 1], default `null` | Contextual | Below 0.50 adds a checkable reasoning step and contributes one effective-difficulty step downward; at or above 0.85 reduces routine scaffolding. It never changes the factual correctness standard in evaluation. |
| `difficulty_performance` | Map of difficulty string to float in [0, 1], default `{}` | Contextual | The entry matching the requested `difficulty` is read. Below 0.50 adds steps/hints and contributes one effective-difficulty step downward; at or above 0.85 reduces hints and contributes one step upward. |
| `misconception_history` | List of arbitrary dictionaries, default `[]` | Contextual | Topic-matching misconception records are exposed as learner context with an explicit instruction to correct them only when the evidence supports doing so. They are never treated as lesson evidence. |
| `recommended_revision_topics` | List of strings, default `[]` | Contextual | Matching recommendations are prioritized in the response's `recommended_revision` field for explanation/evaluation and included in relevant prompt guidance. Unrelated recommendations are ignored. |

Arbitrary profile result dictionaries have no strict internal schema. The overall request character limit bounds their total size. Prompt records are compacted and common identity/secret fields (`user_id`, name, email, phone, token, secret, password) are removed before they are included.

## Endpoint request parameter matrix

Every field below is read from the Pydantic request object; fields described as prompt guidance influence only pedagogical form. Evidence selection and answer validation remain separate from that stylistic adaptation.

| Endpoint | Parameter | Type, default, validation | Classification | Consumer and intended effect | Tested |
|---|---|---|---|---|---|
| `/learning/explain` | `topic` | String, required, length 1–300 | Behavioral | Included in retrieval query and subject scope; passed to adaptation and prompt. When `question` is also supplied, both topic and question form the retrieval query. | Existing endpoint grounding tests; Swagger `top_k` sensitivity test. |
| `/learning/explain` | `question` | Optional string, default `null`, max 5,000 | Behavioral | Directs the explanation and evidence filter; topic remains in the retrieval query even when question is present. | `test_swagger_top_k_reaches_retrieval_and_bounds_evidence`; question text appears in task instructions. |
| `/learning/explain` | `difficulty` | `easy`, `medium`, or `hard`; default `medium` | Behavioral | Explicitly changes reasoning complexity. Profile signals may shift effective difficulty one level while preserving the requested level and adding profile scaffolding. Facts remain evidence-bounded. | `test_difficulty_and_depth_are_independent_behavioral_controls`. |
| `/learning/explain` | `explanation_depth` | `brief`, `standard`, or `high`; default `standard` | Behavioral | Changes explanatory detail and provider output-token budget (800/1,400/2,400 primary; 500/900/1,400 fallback). High requests deeper supported reasoning, term definitions, and concept connections. | `test_difficulty_and_depth_are_independent_behavioral_controls`; depth budgets in `_real`. |
| `/learning/explain` | `knowledge_mode` | `learning_materials_only` or `learning_materials_then_general_knowledge`; default materials-only | Behavioral | Materials-only fails closed on insufficient evidence. The second mode permits general-knowledge fallback only when evidence is insufficient. If retrieval is sufficient, both modes use evidence-bounded generation and validation. | `test_knowledge_mode_controls_grounding_and_general_fallback`; endpoint grounding tests. |
| `/learning/summarize` | `topic` | String, required, length 1–300 | Behavioral | Retrieval query, subject scope, matching learner profile. | Existing endpoint tests. |
| `/learning/summarize` | `style` | Enum `concise`, `exam-focused`, `detailed`, `bullet-point`, `revision-sheet`; default `concise` | Behavioral | Task instructions change selection and organization of facts. Unsupported details remain excluded in evidence-bounded mode. | `test_summary_mcq_and_exam_parameters_change_task_instructions`. |
| `/learning/summarize` | `knowledge_mode` | Same two-mode enum; default materials-only | Behavioral | Controls fail-closed vs. evidence-insufficient general fallback as above. | Existing grounding tests; shared mode sensitivity test. |
| `/learning/mcq` | `topic` | String, required, length 1–300 | Behavioral | Included with `requested_concepts` in retrieval; used for topical evidence filtering and adaptation. | Existing API endpoint tests. |
| `/learning/mcq` | `number_of_questions` | Integer, default 5, range 1–50 | Behavioral | Exact array size is enforced by structured generation validation; also informs requested coverage. | Existing `test_requested_item_count_is_enforced_and_adaptation_preserved`; count schema enforcement. |
| `/learning/mcq` | `difficulty` | `easy`, `medium`, `hard`; default `medium` | Behavioral | Changes reasoning demand and question complexity; profile may adapt effective level. | Shared difficulty sensitivity test; generation-contract tests. |
| `/learning/mcq` | `question_style` | String, default `conceptual`, length 1–100 | Behavioral | Known styles have explicit recall/conceptual/application/scenario guidance; other strings are treated as custom style requests subject to evidence limits. | `test_summary_mcq_and_exam_parameters_change_task_instructions`. |
| `/learning/mcq` | `requested_concepts` | List of strings, default `[]`; each length 1–200 | Behavioral | Added to retrieval query and prompt; prioritizes distinct requested concepts up to the question count. Unsupported concepts are skipped rather than answered from memory. | `test_summary_mcq_and_exam_parameters_change_task_instructions`. |
| `/learning/mcq` | `knowledge_mode` | Same two-mode enum; default materials-only | Behavioral | Controls retrieval sufficiency, general fallback permission, and strict evidence validation. | Existing grounding tests and shared knowledge-mode sensitivity test. |
| `/learning/flashcards` | `topic` | String, required, length 1–300 | Behavioral | Retrieval query, topic evidence filter, adaptation scope. | Existing endpoint tests. |
| `/learning/flashcards` | `number_of_cards` | Integer, default 10, range 1–50 | Behavioral | Exact card count enforced by structured generation validation. | Existing item-count test. |
| `/learning/flashcards` | `difficulty` | `easy`, `medium`, `hard`; default `medium` | Behavioral | Changes front/back complexity and challenge; profile may adjust effective difficulty. | Existing generation-contract tests plus shared difficulty prompt test. |
| `/learning/flashcards` | `knowledge_mode` | Same two-mode enum; default materials-only | Behavioral | Controls grounding and fallback policy. | Shared knowledge-mode sensitivity test. |
| `/learning/exam` | `topics` | List of 1–50 strings; each length 1–300 | Behavioral | Joined for retrieval; supplied topics bound question coverage and profile matching. | Existing endpoint tests. |
| `/learning/exam` | `number_of_questions` | Integer, default 10, range 1–100 | Behavioral | Exact question array size enforced by structured generation validation. | Existing item-count test. |
| `/learning/exam` | `difficulty` | `easy`, `medium`, `hard`; default `medium` | Behavioral | Changes question reasoning complexity; profile may adjust effective level and scaffolding. | Shared difficulty and learner profile tests. |
| `/learning/exam` | `time_limit_minutes` | Optional integer, default `null`, range 1–600 | Behavioral | Prompt constrains reading/reasoning demand to the allotted time. Response time limit is stamped from the request, not trusted from the model. | `test_summary_mcq_and_exam_parameters_change_task_instructions`. |
| `/learning/exam` | `weak_topic_emphasis` | Boolean, default `true` | Behavioral/contextual | True biases distribution toward matching weak topics; false explicitly requests even coverage without profile bias. Only requested topics can receive questions. | `test_summary_mcq_and_exam_parameters_change_task_instructions`. |
| `/learning/exam` | `knowledge_mode` | Same two-mode enum; default materials-only | Behavioral | Controls grounding and fallback policy. | Shared knowledge-mode sensitivity test and endpoint tests. |
| `/learning/evaluate-answer` | `question` | String, required, length 1–5,000 | Behavioral | Included in retrieval query and task instructions; scopes the expected answer and feedback to this question. | Existing answer-evaluation tests; request/prompt inspection. |
| `/learning/evaluate-answer` | `student_answer` | String, required, length 1–10,000 | Behavioral | Determines the deterministic exact/partial/zero lexical grade when an expected answer exists and drives answer-specific mismatch feedback. It is excluded from retrieval so an incorrect answer cannot steer evidence selection. A missing misconception is left null; the server does not infer a learner's reasoning from a wrong answer alone. | `test_answer_grading_uses_expected_answer_and_gives_partial_credit`; `test_student_answer_and_learner_profile_interact_in_evaluation`; evaluation tests. |
| `/learning/evaluate-answer` | `correct_answer` | Optional string, default `null`, max 10,000 | Behavioral | When supplied it is stamped as the authoritative expected answer and checked against retrieved evidence when evidence is required. When absent, the model derives a target from the evidence and question. | Updated `test_exact_match_recovery_respects_request_correct_answer`; partial/full/zero grade test. |
| `/learning/evaluate-answer` | `topic` | String, default `general`, length 1–300 | Behavioral | Included in retrieval query, response topic, feedback scope, and topic-matched profile adaptation. | Existing evaluation tests. |
| `/learning/evaluate-answer` | `difficulty` | `easy`, `medium`, `hard`; default `medium` | Behavioral for feedback only | Changes feedback sophistication/scaffolding; it does not change the answer grade or expected factual standard. | Prompt and grade independence assertions in parameter tests. |
| `/learning/evaluate-answer` | `knowledge_mode` | Same two-mode enum; default materials-only | Behavioral | Materials-only fails closed on missing evidence or post-generation unsupported claims. General fallback accepts general answers only when retrieval is insufficient; sufficient retrieval still requires evidence validation. | Shared knowledge-mode sensitivity test; semantic claim judge tests. |

### Evaluation score behavior

When `correct_answer` exists, `_grade_answer` compares normalized text first. Otherwise it compares salient content-term precision/recall and returns their F1 score: full lexical match is correct, partial overlap receives partial credit, and no shared content terms receives zero. The score and correctness result do not use `difficulty`. If the evaluator leaves `why_answer_is_wrong` blank, local feedback identifies expected content terms missing from the student's answer. The server does not fabricate `identified_misconception`; that field remains null unless an evaluator supplies a concise, validated misconception. This is deterministic and testable, but token overlap cannot recognize every semantically equivalent paraphrase; see limitations below.

## Retrieval, grounding, and generation settings

These are runtime configuration parameters, not request-body fields unless stated above. There is no client request filter field today.

| Setting | Default and validation | Consumer and functional effect | Sensitivity / validation |
|---|---|---|---|
| `retrieval_strategy` | `hybrid_cross_encoder` (only supported Literal) | Selects the production architecture: dense + BM25 → RRF → Cross-Encoder. | Existing hybrid retrieval tests. |
| `dense_candidate_k` | 30; integer 1–1,000 | Chroma candidate count; raised to at least request final `top_k` when needed. | `test_dense_retriever_is_called_with_configured_candidate_pool`; final top-K tests. |
| `sparse_candidate_k` | 30; integer 1–1,000 | BM25 candidate count; raised to at least request final `top_k` when needed. | Hybrid retrieval tests. |
| `rrf_k` | 60; integer 1–10,000 | Passed directly into each reciprocal-rank contribution `1/(rrf_k + rank)`; changes fusion scores and can change rank order. | `test_rrf_k_changes_scores_and_can_change_rank_order`. |
| `rerank_candidate_k` | 30; integer 1–1,000 | RRF output limit and normal Cross-Encoder pool size; effective pool expands to at least final request `top_k` to avoid silently returning fewer results than requested. | Existing pool/batch test; final top-K sensitivity test. |
| `top_k` / `FINAL_TOP_K` | 5; integer 1–100; `FINAL_TOP_K` wins over compatibility alias `TOP_K` | Default final evidence count. A request body's `top_k` (1–20) overrides it. | Hybrid and request-level top-K tests. |
| `cross_encoder_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Model name passed to the lazy model loader. | Existing model/scorer tests; real model smoke test if dependencies are present. |
| `cross_encoder_device` | `auto` | Device passed to model loader; `auto` selects CUDA when available, CPU otherwise; unavailable requested CUDA falls back to CPU. | `test_device_auto_selects_cpu_and_cuda_fallback`. |
| `cross_encoder_batch_size` | 16; integer 1–256 | Batch size passed to `predict`. | Existing and added recording-scorer tests. |
| `cross_encoder_score_offset` | 5.0; float −20 to 20 | Adds to raw Cross-Encoder logit before sigmoid normalization; affects relevance floor eligibility, not ranking order (ranking uses raw score). | `test_cross_encoder_offset_changes_threshold_eligibility`. |
| `min_relevance_score` | 0.35; float [0, 1] | `ContextBuilder` removes lower-scoring chunks before evidence filtering. | Existing context/retrieval tests. |
| `max_context_chars` | 12,000; integer 1–1,000,000 | Bounds both `GroundedContext.text` and the actual per-chunk evidence packet sent to providers. Separators count toward the bound. | `test_context_character_limit_bounds_chunks_actually_sent_to_prompt`. |
| `chunk_chars` | 1,000; integer 1–100,000 | Ingestion chunk size; larger chunks preserve local context but reduce retrieval granularity. | Existing PDF ingestion tests. |
| `chunk_overlap_chars` | 120; integer 0–99,999 and less than `chunk_chars` | Ingestion overlap; prevents boundary facts from being split from adjacent context. | Settings cross-field validation and PDF tests. |
| BM25 `k1`, `b` | Fixed 1.5 and 0.75 | Used in sparse term-frequency normalization; not environment settings or API fields. | BM25 lexical tests. |
| Chroma fetch expansion | Fixed multiplier 20 and minimum 50 rows | Dense retriever fetches extra records to retain enough results after embedding-version filtering. | Retrieval tests. |
| Cross-Encoder max length | Fixed 512 tokens | Passed when loading the configured Cross-Encoder; not independently configurable. | Loader code and existing real-pair test. |
| `max_input_chars` / `MAX_INPUT_CHARS` | 20,000; integer 1–1,000,000 | ASGI middleware enforces decoded character count on every `/learning/*` POST before Pydantic parsing; oversized input gets HTTP 413. | `test_configured_input_character_limit_is_enforced_before_endpoint_parsing`. |

Other active runtime settings: `llm_provider` (Groq default), `llm_fallback_provider` (Ollama default), provider model names/base URL, embedding provider/model, Chroma path/collection, `temperature` (0–2), `max_retries` (0–5), `request_timeout` (>0–600 seconds), `claim_judge_min_confidence` (0–1), `auto_ingest_pdfs`, `sample_data_dir`, `cors_origins`, and `log_level`. Provider transport settings are consumed in `providers.py`; embedding/model and Chroma identity are consumed in `rag.py`; ingestion settings are consumed by the lifespan hook and ingestion helpers. Normal generation retains Groq primary → Ollama fallback on provider errors. The semantic claim judge remains a separate Ollama-only, fail-closed path.

Generation token budgets (code controls): MCQ/exam primary `min(16,000, max(3,000, question_count × 180))`, fallback `min(8,000, max(1,200, question_count × 120))`; flashcard primary `min(12,000, max(2,200, card_count × 120))`, fallback `min(6,000, max(900, card_count × 80))`; evaluation 1,800/900; explanation brief/standard/high 800/500, 1,400/900, 2,400/1,400; other tasks 2,000/1,200. Provider output is schema-validated, exact requested question/card counts are enforced, and grounding/source metadata is stamped locally.

## Query construction and important interactions

| Endpoint | Retrieval query | Additional behavior |
|---|---|---|
| Explain | `topic` plus optional `question` | Evidence filtering uses the question when supplied, otherwise topic; topic remains a retrieval subject boundary. |
| Summarize | `topic` | `style` changes summary organization after retrieval. |
| MCQ | `topic` plus requested concepts when present | Evidence filter remains topic-scoped; unsupported requested concepts are omitted. |
| Flashcards | `topic` | Difficulty changes challenge, not source facts. |
| Exam | Joined `topics` | Weak-topic bias changes question distribution only when enabled. |
| Evaluate answer | `topic`, `question`, and optional `correct_answer` | `student_answer` is not part of retrieval; it is used in grading and feedback after evidence is selected. |

Interactions verified:

- Requested difficulty and explanation depth are separate instructions; their 3×3 combinations remain distinct.
- Learner profile can change effective difficulty and scaffolding, while the original requested difficulty remains visible and the factual answer is still evidence-controlled.
- `learning_materials_then_general_knowledge` uses general knowledge only if the retrieved/task-filtered context is insufficient; adequate retrieved context invokes the same evidence validation as materials-only mode.
- Request `top_k` expands dense/sparse candidate counts and reranking capacity when configured pools are smaller, then caps the final evidence set at the requested value (or available evidence).
- Evaluation answer grading is unchanged by `difficulty`; changing student answer changes exact/partial/zero grade against the same expected answer.

Grounding lifecycle note: retrieval/evidence insufficiency is reported as `source_mode: insufficient_context` and `retrieval_status: insufficient`. If retrieval is sufficient but generation or later claim validation fails, the response retains `retrieval_status: sufficient`, the retrieved-learning source mode and local sources, and reports the separate `failure_status`; it does not tell the learner that the materials lack the answer. Why-question evidence checks use generic causal/support constructions and require topical overlap in the same substantive sentence.

## Sensitivity tests added or updated

New tests in `tests/test_parameter_sensitivity.py` cover depth × difficulty, summary/MCQ/exam style controls, topic-scoped learner signals, knowledge-mode fallback, request and hybrid top-K values 3/5/10, count-dependent item/token budgets, real evidence-packet character bounds, exact/partial/wrong answer grades, schema/config bounds, input-body enforcement, RRF-k rank/score effects, and Cross-Encoder score-offset threshold effects. Existing tests updated where they asserted previous ineffective behavior: mastery now adapts one step rather than replacing the requested scale with `foundational`/`advanced`; effective item difficulty is preserved in generated MCQ/cards/exams; and a provided expected answer is authoritative even if the model returns another answer.

Related existing coverage includes `tests/test_api.py`, `test_generation_contracts.py`, `test_hybrid_retrieval.py`, `test_evidence_grounding.py`, `test_semantic_claim_judge.py`, `test_validation.py`, and the complete suite.

## Swagger manual comparison matrix

Open `/docs`, choose a route, and run contrasting bodies. For `/learning/explain`, the project corpus includes Nile evidence; use a topic/question supported by the indexed PDF corpus for grounded comparisons. For general fallback, choose a topic absent from the corpus and compare the returned grounding source/status.

### `/learning/explain`

```json
{
  "topic": "Ancient Egyptian agriculture",
  "question": "Why did the annual inundation of the Nile matter to agriculture?",
  "difficulty": "easy",
  "explanation_depth": "brief",
  "top_k": 3,
  "knowledge_mode": "learning_materials_only",
  "learner_profile": {"overall_level": "beginner", "topic_mastery": {"Ancient Egyptian agriculture": 0.25}}
}
```

Compare with `difficulty: "hard"`, `explanation_depth: "high"`, `top_k: 10`, `overall_level: "advanced"`, and `topic_mastery` 0.92. Compare explanation detail and `adaptation`; facts/sources should remain tied to the evidence. Also compare `brief`, `standard`, and `high` independently to isolate depth.

For fallback, repeat an unsupported topic such as `quantum teleportation` with each mode:

```json
{"topic":"quantum teleportation","question":"What is quantum teleportation?","knowledge_mode":"learning_materials_only"}
```

```json
{"topic":"quantum teleportation","question":"What is quantum teleportation?","knowledge_mode":"learning_materials_then_general_knowledge"}
```

The first returns an insufficient-context response without a provider generation; the second may generate a general answer and reports `source_mode: general_knowledge`, `retrieval_status: insufficient`.

### `/learning/summarize`

```json
{"topic":"Ancient Egyptian agriculture","style":"concise","top_k":3}
```

```json
{"topic":"Ancient Egyptian agriculture","style":"detailed","top_k":10}
```

Also try `exam-focused`, `bullet-point`, and `revision-sheet` to compare organization while checking the same source and grounding status.

### `/learning/mcq`

```json
{"topic":"Ancient Egyptian agriculture","number_of_questions":3,"difficulty":"easy","question_style":"recall","requested_concepts":["annual inundation"],"top_k":3}
```

```json
{"topic":"Ancient Egyptian agriculture","number_of_questions":3,"difficulty":"hard","question_style":"application","requested_concepts":["Nile agriculture","annual inundation"],"top_k":10,"learner_profile":{"weak_topics":["Ancient Egyptian agriculture"]}}
```

Compare question reasoning and concept coverage. The number of returned questions should equal `number_of_questions` when valid, grounded items are available.

### `/learning/flashcards`

```json
{"topic":"Ancient Egyptian agriculture","number_of_cards":3,"difficulty":"easy","top_k":3}
```

```json
{"topic":"Ancient Egyptian agriculture","number_of_cards":3,"difficulty":"hard","top_k":10,"learner_profile":{"overall_level":"advanced","topic_mastery":{"Ancient Egyptian agriculture":0.9}}}
```

### `/learning/exam`

```json
{"topics":["Ancient Egyptian agriculture","Nile River"],"number_of_questions":6,"difficulty":"medium","time_limit_minutes":45,"weak_topic_emphasis":true,"learner_profile":{"weak_topics":["Nile River"]},"top_k":5}
```

Compare with `weak_topic_emphasis: false` and a shorter `time_limit_minutes: 15`. Check question distribution, task complexity, exact question count, and echoed time limit.

### `/learning/evaluate-answer`

Keep question, topic, expected answer, mode, and learner profile fixed; change only `student_answer`:

```json
{"question":"What did the annual inundation of the Nile River support?","topic":"Ancient Egyptian agriculture","correct_answer":"The annual inundation of the Nile River supported its agricultural population.","student_answer":"The annual inundation of the Nile River supported its agricultural population.","difficulty":"medium","knowledge_mode":"learning_materials_only"}
```

```json
{"question":"What did the annual inundation of the Nile River support?","topic":"Ancient Egyptian agriculture","correct_answer":"The annual inundation of the Nile River supported its agricultural population.","student_answer":"The annual inundation of the Nile River","difficulty":"medium","knowledge_mode":"learning_materials_only"}
```

```json
{"question":"What did the annual inundation of the Nile River support?","topic":"Ancient Egyptian agriculture","correct_answer":"The annual inundation of the Nile River supported its agricultural population.","student_answer":"The Bastille","difficulty":"medium","knowledge_mode":"learning_materials_only"}
```

Expected lexical grades: full credit, partial credit, and zero. Then keep the correct student answer fixed while changing `difficulty` to `hard`; correctness/score should remain unchanged while feedback guidance becomes more sophisticated. Try an unrelated question with the same expected answer to check that retrieval and feedback stay question-scoped.

## Known limitations

- Prompt-following remains probabilistic. Explicit difficulty/depth/style guidance can influence output but cannot guarantee a fixed sentence count or an identical response on every provider call.
- In evidence-bounded mode, a requested high depth or hard difficulty is intentionally constrained by available evidence. The system must stay concise rather than inventing supporting facts.
- The answer score is deterministic salient-token F1 when an expected answer is supplied. It cannot reliably identify all synonyms or paraphrases with little vocabulary overlap; review such answers or use an expected answer whose key terms express the intended concepts.
- Free-form `question_style` accepts custom text; known named styles have stronger explicit behavior. Arbitrary custom style instructions remain subject to prompt safety and grounding.
- No endpoint accepts user-defined Chroma metadata filters. Filters are supported internally by dense/sparse retrievers but are not exposed in this API contract.
- The included benchmark corpus and hand-labeled query set are small engineering checks, not proof of broad retrieval or pedagogical quality.
