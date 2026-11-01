import asyncio, logging, time, uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse as StarletteJSONResponse
from .config import get_settings
from .providers import create_llm_provider
from .providers import ProviderError
from .validation import OutputValidationError
from .rag import ChromaRetriever, HybridCrossEncoderRetriever, sync_pdf_directory
from .generation import LearningGenerator
from .schemas import *

# Construct shared services once per process; routes pass validated request models
# to the generator, while the lifespan hook synchronizes PDFs before serving traffic.
settings = get_settings(); logging.basicConfig(level=settings.log_level); log = logging.getLogger("learning")
dense_retriever = ChromaRetriever(settings)
retriever_factory = {"hybrid_cross_encoder": HybridCrossEncoderRetriever}[settings.retrieval_strategy]
retriever = retriever_factory(dense_retriever, settings)
provider = create_llm_provider(settings)
generator = LearningGenerator(retriever, settings.max_context_chars, provider, settings, top_k=settings.top_k, min_relevance_score=settings.min_relevance_score)
@asynccontextmanager
async def lifespan(application):
    # Startup ingestion runs off the event loop because PDF parsing and embedding are blocking.
    if settings.auto_ingest_pdfs:
        try:
            await asyncio.to_thread(sync_pdf_directory, settings.sample_data_dir, retriever.collection, retriever.embedding,
                                    chroma_path=settings.chroma_path, chunk_chars=settings.chunk_chars,
                                    overlap_chars=settings.chunk_overlap_chars)
        except Exception:
            log.exception("RAG auto-ingestion could not complete")
        finally:
            await asyncio.to_thread(retriever.refresh_sparse_index)
    else:
        log.info("RAG auto-ingestion disabled by AUTO_INGEST_PDFS")
    yield

app = FastAPI(title="Generative Learning Model", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[x.strip() for x in settings.cors_origins.split(",")], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

class RequestCharacterLimitMiddleware:
    """Bound JSON body characters before FastAPI parses request models."""
    def __init__(self, app, max_chars: int):
        self.app, self.max_chars = app, max_chars

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") != "POST" or not scope.get("path", "").startswith("/learning/"):
            await self.app(scope, receive, send)
            return

        max_bytes = self.max_chars * 4  # UTF-8 uses at most four bytes per Unicode code point.
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        try:
            content_length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            content_length = 0
        if content_length > max_bytes:
            await StarletteJSONResponse(status_code=413, content={"detail": f"Request body exceeds the {self.max_chars}-character limit."})(scope, receive, send)
            return

        body = bytearray()
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > max_bytes:
                await StarletteJSONResponse(status_code=413, content={"detail": f"Request body exceeds the {self.max_chars}-character limit."})(scope, receive, send)
                return
            if not message.get("more_body", False):
                break
        if len(bytes(body).decode("utf-8", errors="replace")) > self.max_chars:
            await StarletteJSONResponse(status_code=413, content={"detail": f"Request body exceeds the {self.max_chars}-character limit."})(scope, receive, send)
            return

        replayed = False
        async def replay_receive():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return {"type": "http.disconnect"}

        await self.app(scope, replay_receive, send)

app.add_middleware(RequestCharacterLimitMiddleware, max_chars=settings.max_input_chars)

@app.exception_handler(ProviderError)
async def provider_error_handler(request: Request, exc: ProviderError):
    log.error("generation_failed endpoint=%s failure=provider", request.url.path)
    return JSONResponse(status_code=502, content={"error": "llm_provider_unavailable", "failure_status": "generation_failed", "detail": str(exc)})

@app.exception_handler(OutputValidationError)
async def output_error_handler(request: Request, exc: OutputValidationError):
    log.error("output_validation_failed endpoint=%s", request.url.path)
    return JSONResponse(status_code=502, content={"error": "invalid_llm_output", "failure_status": "output_validation_failed", "detail": "The configured model returned an invalid structured response."})

@app.middleware("http")
async def request_logging(request: Request, call_next):
    rid = str(uuid.uuid4()); started = time.perf_counter()
    try:
        response = await call_next(request); response.headers["X-Request-ID"] = rid
        log.info("request_id=%s endpoint=%s status=%s latency_ms=%.1f", rid, request.url.path, response.status_code, (time.perf_counter()-started)*1000); return response
    except Exception:
        log.exception("request_id=%s endpoint=%s", rid, request.url.path); return JSONResponse(status_code=500, content={"error": "internal_error", "request_id": rid})

@app.get("/health")
def health(): return {"status": "ok", "service": "generative-learning-model", "provider": settings.llm_provider}
@app.get("/ready")
def ready():
    try:
        count = retriever.collection.count()
        return {"status": "ready", "retriever": "chroma", "collection": settings.chroma_collection, "document_chunks": count, "provider": settings.llm_provider}
    except Exception:
        log.exception("Chroma readiness check failed")
        return JSONResponse(status_code=503, content={"status": "not_ready", "retriever": "chroma"})
@app.post("/learning/explain", response_model=ExplanationResponse)
# FastAPI validates incoming bodies and serializes the generator's typed response.
def explain(request: ExplanationRequest): return generator.explain(request)
@app.post("/learning/summarize", response_model=SummaryResponse)
def summarize(request: SummaryRequest): return generator.summarize(request)
@app.post("/learning/mcq", response_model=MCQSet)
def mcq(request: MCQRequest): return generator.mcq(request)
@app.post("/learning/flashcards", response_model=FlashcardSet)
def flashcards(request: FlashcardRequest): return generator.flashcards(request)
@app.post("/learning/exam", response_model=PracticeExam)
def exam(request: PracticeExamRequest): return generator.exam(request)
@app.post("/learning/evaluate-answer", response_model=AnswerEvaluation)
def evaluate(request: AnswerEvaluationRequest): return generator.evaluate(request)
