# API and Data Flow

This document records the public HTTP interface and traces how request data moves through the current implementation. Routes and models are defined in app/main.py and app/schemas.py; shared processing is in app/generation.py.

## 1. API endpoint inventory

| Method | Path | Request schema | Response schema | Main behavior |
|---|---|---|---|---|
| GET | /health | None | object | Liveness response; returns status, service name and configured LLM provider. Does not contact Chroma or LLM providers. |
| GET | /ready | None | object or 503 | Calls Chroma collection count; reports collection, chunk count and configured provider. Does not test Groq/Ollama or embedding inference. |
| POST | /learning/explain | ExplanationRequest | ExplanationResponse | Retrieves context and returns explanation/learning fields. |
| POST | /learning/summarize | SummaryRequest | SummaryResponse | Retrieves topic context and returns summary and key points. |
| POST | /learning/mcq | MCQRequest | MCQSet | Retrieves topic context and generates requested count of MCQs. |
| POST | /learning/flashcards | FlashcardRequest | FlashcardSet | Retrieves topic context and generates requested count of cards. |
| POST | /learning/exam | PracticeExamRequest | PracticeExam | Joins topics, retrieves context, and generates a practice exam. |
| POST | /learning/evaluate-answer | AnswerEvaluationRequest | AnswerEvaluation | Retrieves authoritative expected-answer context, validates expected answer, then generates feedback when safe. |

FastAPI also supplies /docs, /redoc, and /openapi.json. There is no PDF upload route, no user profile read/write route, and no separate retrieval route.

## 2. Representative request bodies

All sample payloads are illustrative contract examples based on schemas and TestCases.txt. They omit optional learner-profile detail for readability.

### Explain

~~~json
{
  "topic": "Ancient Egypt",
  "question": "Why was the Nile River important to ancient Egyptian civilization?",
  "difficulty": "medium",
  "explanation_depth": "standard",
  "knowledge_mode": "learning_materials_only",
  "learner_profile": {}
}
~~~

### Summarize

~~~json
{
  "topic": "Ancient Egyptian agriculture",
  "style": "exam-focused",
  "knowledge_mode": "learning_materials_only",
  "learner_profile": {}
}
~~~

### MCQ

~~~json
{
  "topic": "Ancient Egypt",
  "number_of_questions": 5,
  "difficulty": "medium",
  "question_style": "conceptual",
  "requested_concepts": ["Nile River", "agriculture"],
  "knowledge_mode": "learning_materials_only",
  "learner_profile": {}
}
~~~

### Flashcards

~~~json
{
  "topic": "Ancient Egyptian agriculture",
  "number_of_cards": 10,
  "difficulty": "medium",
  "knowledge_mode": "learning_materials_only",
  "learner_profile": {}
}
~~~

### Practice exam

~~~json
{
  "topics": ["Ancient Egypt", "Nile River"],
  "number_of_questions": 10,
  "difficulty": "medium",
  "time_limit_minutes": 15,
  "weak_topic_emphasis": true,
  "knowledge_mode": "learning_materials_only",
  "learner_profile": {}
}
~~~

### Evaluate answer

~~~json
{
  "question": "Why was the Nile River important to ancient Egyptian civilization?",
  "student_answer": "It was mainly a route for military expansion.",
  "correct_answer": "The annual inundation of the Nile supported the agricultural population.",
  "topic": "Ancient Egyptian Agriculture",
  "difficulty": "medium",
  "knowledge_mode": "learning_materials_only",
  "learner_profile": {}
}
~~~

## 3. Request schema details

Pydantic performs request parsing before the endpoint handler runs. Missing required fields, wrong types that cannot be parsed, Literal mismatches, and declared bounds produce HTTP 422 from FastAPI.

### LearnerProfileIn

| Field | Type/default | Runtime use |
|---|---|---|
| user_id | optional string, None, max 200 | Metadata only; not sent to the prompt or logs. |
| overall_level | string, intermediate, length 1–50 | Adds explicit vocabulary and explanation guidance for beginner/intermediate/advanced levels. |
| topic_mastery | dict topic→float, empty, values 0–1 | Matching mastery adds scaffolding or advanced application guidance and may shift effective difficulty one step. |
| weak_topics | list string, empty | Matching topics are prioritized in remediation and, for exams, question distribution when weak-topic emphasis is enabled. |
| strong_topics | list string, empty | Matching topics avoid introductory reteaching and focus on the requested new angle. |
| recent_quiz_results | list arbitrary dictionaries, empty | Matching-topic records are used for remediation; recognized scores below 0.50 add an intermediate step. |
| recent_exam_results | list arbitrary dictionaries, empty | Same topic-scoped remediation use as quiz records. |
| historical_performance | dict string→any, empty | Matching historical records inform remediation; recognized scores below 0.50 add an intermediate step. |
| question_accuracy | optional float, None, range 0–1 | Low accuracy adds scaffolding and shifts effective difficulty down one step; high accuracy reduces routine hints. |
| difficulty_performance | dict difficulty→float, values 0–1 | Low/high performance at the requested difficulty shifts effective difficulty and scaffolding accordingly. |
| misconception_history | list arbitrary dictionaries, empty | Matching misconceptions are addressed only when the retrieved evidence supports a correction. |
| recommended_revision_topics | list string, empty | Matching revision topics are prioritized in relevant follow-up recommendations. |

Extra LearnerProfileIn fields are explicitly ignored. The other request classes inherit Pydantic BaseModel and use its default extra-field behavior.

### Request models

| Schema | Fields, defaults, constraints |
|---|---|
| ExplanationRequest | Shared optional `top_k` 1–20; topic required, length 1–300; optional question max 5,000; difficulty enum easy/medium/hard; explanation_depth enum brief/standard/high; two knowledge modes; empty profile default. |
| SummaryRequest | Shared optional `top_k`; topic required, length 1–300; style enum concise/exam-focused/detailed/bullet-point/revision-sheet; knowledge mode/profile defaults. |
| MCQRequest | Shared optional `top_k`; topic required, length 1–300; number_of_questions default 5, range 1–50; difficulty enum; question_style default conceptual, length 1–100; requested_concepts strings length 1–200; knowledge mode/profile defaults. |
| FlashcardRequest | Shared optional `top_k`; topic required, length 1–300; number_of_cards default 10, range 1–50; difficulty enum; knowledge mode/profile defaults. |
| PracticeExamRequest | Shared optional `top_k`; 1–50 topics, each length 1–300; number_of_questions default 10, range 1–100; difficulty enum; optional time_limit_minutes 1–600; weak_topic_emphasis true; knowledge mode/profile defaults. |
| AnswerEvaluationRequest | Shared optional `top_k`; question required, length 1–5,000; student_answer required, length 1–10,000; correct_answer optional max 10,000; topic default general, length 1–300; difficulty enum; knowledge mode/profile defaults. |

`MAX_INPUT_CHARS=20,000` is enforced by request middleware on `/learning/*` POST bodies before Pydantic parsing; over-limit requests return 413. All six learning request types use this shared limit.

## 4. Response schemas

| Schema | Fields |
|---|---|
| LearningSource | Optional document_name, page_number, chunk_id, snippet, relevance_score. Source values are stamped locally from retrieved chunks. |
| GroundingMetadata | source_mode: retrieved_learning_content/general_knowledge/insufficient_context; grounding_status: grounded/partially_grounded/not_grounded; retrieval_status sufficient/insufficient; optional retrieval_method, sources, notice. Runtime currently emits grounded or not_grounded, not partially_grounded. |
| AdaptationMetadata | requested_settings, adapted_settings, adaptation_reasons. |
| GenerationMetadata | provider, model, fallback_used. Null when there was no generation call on common insufficient-context paths. |
| ExplanationResponse | explanation; key_concepts; examples; important_points; common_mistakes; recommended_revision; grounding; adaptation; optional generation. |
| SummaryResponse | summary; key_points; grounding; adaptation; optional generation. |
| MCQ | question; options (at least 2, no maximum); correct_answer; explanation; difficulty; topic; learning_objective; optional source. |
| MCQSet | questions; grounding; adaptation; optional generation. |
| Flashcard | front; back; topic; difficulty; concept; optional source. |
| FlashcardSet | cards; grounding; adaptation; optional generation. |
| PracticeExam | title; questions; optional time_limit_minutes; grounding; adaptation; optional generation. |
| AnswerEvaluation | correct; score [0,1]; nullable correct_answer; explanation; nullable why_answer_is_wrong; related_concept; recommended_revision; confidence [0,1]; nullable identified_misconception; supporting_source_or_evidence; groundedness enum; topic; difficulty; grounding; optional generation. |
| ClaimValidationResult | Internal strict object: supported bool; confidence float [0,1]; reason length 1–1,000; extra fields forbidden. |

Pydantic validates public response models and provider output schemas. It does not independently decide if the generated score correctly reflects a pedagogical rubric.

## 5. Request lifecycle

~~~mermaid
sequenceDiagram
    actor User
    participant API as FastAPI
    participant Schema as Pydantic
    participant Gen as LearningGenerator
    participant Retrieve as Hybrid retriever
    participant Chroma
    participant BM25
    participant CE as Cross-Encoder
    participant Ev as Evidence/context
    participant Judge as Ollama judge
    participant LLM as Groq generation
    participant Fall as Ollama fallback
    participant Out as Output validation

    User->>API: POST JSON
    API->>Schema: Parse and validate request
    Schema-->>API: Typed request or HTTP 422
    API->>Gen: Call route-specific generator method
    Gen->>Retrieve: Search endpoint-built query
    Retrieve->>Chroma: Dense top 30
    Retrieve->>BM25: Sparse top 30
    Chroma-->>Retrieve: Text, metadata, distance
    BM25-->>Retrieve: Lexically ranked chunks
    Retrieve->>CE: RRF pool, at most 30 query/passage pairs
    CE-->>Retrieve: Raw scores and final top 5
    Retrieve-->>Gen: Ranked chunks
    Gen->>Ev: Score floor, substantive/topic/task filter, context build
    alt No context in materials-only mode
        Ev-->>Gen: Insufficient
        Gen-->>API: Typed empty result; no LLM call
    else Context exists
        opt Evaluate answer with expected answer and deterministic check fails
            Gen->>Judge: Claim + final evidence, Ollama schema JSON
            alt Judge supported and confidence ≥ 0.80
                Judge-->>Gen: Accept expected answer
            else Judge rejects, fails, or malformed
                Judge-->>Gen: Fail closed
                Gen-->>API: Insufficient evaluation, no generation
            end
        end
        Gen->>LLM: Prompt + provider schema
        alt Groq provider error
            LLM-->>Gen: ProviderError
            Gen->>Fall: Retry through configured Ollama fallback
            Fall-->>Out: Text result or provider error
        else Groq returns text
            LLM-->>Out: Text result
        end
        Out->>Out: Parse JSON, counts, full Pydantic schema
        opt Invalid JSON/schema/count
            Out->>LLM: Up to two additional structured attempts
        end
        Out-->>Gen: Typed result or output error
        Gen->>Ev: Validate generated claims and stamp local provenance
        Gen-->>API: Typed result or insufficient result
    end
    API-->>User: JSON response and request ID header
~~~

## 6. Query construction precisely

- Explain: request.topic plus optional request.question; the question scopes evidence filtering when supplied.
- Summary and flashcards: request.topic.
- MCQ: request.topic plus requested_concepts when supplied; evidence filtering remains topic-scoped.
- Exam: join request.topics with comma-space.
- Evaluation: newline join available strings for topic, question, correct_answer, each prefixed by “Topic:”, “Question:”, or “Expected answer:”. The student_answer is intentionally omitted. It is passed later as learner text for answer comparison.
- Learner profile, difficulty, requested style, and time limit do not enter retrieval; they produce explicit task/adaptation guidance after retrieval. `top_k` controls retrieval count. MCQ requested concepts also refine retrieval.
- API endpoints do not pass filters, even though Retriever.retrieve and Chroma can accept metadata filters.

## 7. Evaluation endpoint deep dive

Evaluation input consists of question, student answer, optional authoritative correct answer, topic, difficulty, knowledge mode and learner profile.

1. Build retrieval query from topic, question and correct_answer only. Exclude student answer.
2. Retrieve and filter evidence through dense/BM25 → RRF → Cross-Encoder → ContextBuilder → task_evidence → ContextBuilder.
3. In materials-only mode, insufficient context returns false/score 0, explanation notice, no source, groundedness not_grounded and no generation. In the fallback mode, general generation is allowed only if filtered evidence is insufficient.
4. With sufficient evidence, prompt for AnswerEvaluation structured JSON. The supplied question/expected answer can guide retrieval and grading, but the expected answer is not pre-approved as evidence.
5. The model is asked to keep correct_answer, why_answer_is_wrong, explanation and identified_misconception distinct; prompt says expected answer should match the request and feedback must be evidence-bounded.
6. Local code validates response fields and checks each generated factual claim—including the expected answer and feedback—against the final evidence using the Ollama-only Semantic Claim Judge at confidence ≥ configured 0.80.
7. Unsupported or uncertain claims trigger one stricter evidence-only retry; if still mixed, only independently validated coherent content can be salvaged, otherwise return a grounding-validation failure while preserving sufficient retrieval and sources.
8. The server grades against a supplied authoritative expected answer using its current exact/partial/zero lexical grading rule, then locally stamps grounding and evidence sources.
9. HTTP response follows AnswerEvaluation schema.

Field semantics from the prompt/code:
- correct_answer: concise expected answer.
- why_answer_is_wrong: specific mismatch between student wording and supported answer.
- explanation: teach the concept/relationship using evidence, not a comparison.
- identified_misconception: short conceptual label, not copied passage.
- recommended_revision: revision topic list.
- supporting_source_or_evidence: locally sourced citation object(s), not trusted from LLM output.

When an expected answer exists, local exact/partial/zero lexical grading stamps the final score and correctness, independent of difficulty. Without an expected answer, the evaluator supplies the score and correctness within schema bounds. Missing mismatch feedback is filled with expected content terms absent from the learner answer; the server leaves a missing misconception null rather than guessing the learner's reasoning. Token overlap cannot recognize every semantically equivalent paraphrase.

## 8. Output, response metadata and errors

Provider schema is compacted to omit server-owned grounding, adaptation, generation, provenance and time-limit fields. Requested card/question counts become exact minimum and maximum array lengths. After provider output:
1. Parse JSON, optionally stripping a fenced block for judge calls.
2. Require a JSON object.
3. Inject trusted fields computed from request/context.
4. Check requested array counts.
5. Validate the complete Pydantic response.
6. Stamp actual provider, model and fallback_used when a generation field exists.
7. Apply project-specific claim validation and locally derived sources.
8. FastAPI applies the route response_model and serializes JSON.

Failure behavior:
- Request validation: 422.
- Provider transport/API error: Groq error triggers Ollama normal-generation fallback; both fail → 502 llm_provider_unavailable.
- Malformed model output: up to two additional structured attempts; no fallback solely because JSON/schema is invalid; exhausted attempts → 502 invalid_llm_output.
- Materials-only insufficient context or unsupported claim: generally HTTP 200 with typed insufficient response and null generation.
- Chroma readiness count failure: 503.
- Other unhandled error: 500 internal_error with request ID.
- Semantic judge provider/parse failure: reject expected answer and return insufficient evaluation; never try Groq.

## 9. Exact example status

TestCases.txt provides payload examples for all six POST paths. The API tests exercise supported, unsupported and invalid requests across all six learning endpoints with a mocked provider: supported returns grounded data/sources; unsupported materials-only returns insufficient/not-grounded with empty sources and null generation; empty request returns 422.

The Nile semantic regression uses a fixed page-16 chunk and scripted providers, while test_api also exercises the exact Nile “why” request through the real configured retrieval path with mocked generation and asserts an Ancient_Egypt_From.pdf page-16 source. Do not describe mocked generation as a real Groq completion.

See System Architecture for subsystem diagrams and Models and Technologies for provider details.

