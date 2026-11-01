"""PDF ingestion, persistent vector retrieval, and bounded context assembly.

Chunks and their source metadata are stored in Chroma; retrieval returns ranked
chunks for the evidence and generation stages without constructing prompts here.
"""
import re
import hashlib
import logging
import math
import os
import time
import statistics
import threading
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any
from .config import Settings
from .domain import RetrievedChunk, GroundedContext
from .providers import create_embedding_provider

log = logging.getLogger("learning.rag")

class Retriever:
    def retrieve(self, query: str, filters: dict[str, Any] | None = None, top_k: int = 5) -> list[RetrievedChunk]: raise NotImplementedError

class InMemoryRetriever(Retriever):
    def __init__(self, chunks: list[RetrievedChunk] | None = None): self.chunks = chunks or []
    def add(self, chunks: list[RetrievedChunk]): self.chunks.extend(chunks)
    def retrieve(self, query: str, filters=None, top_k=5):
        terms = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        scored = []
        for c in self.chunks:
            if filters and any(filters.get(k) and c.metadata.get(k) != v for k, v in filters.items()): continue
            words = set(re.findall(r"[a-zA-Z0-9]+", c.content.lower()))
            score = len(terms & words) / max(1, len(terms))
            if score > 0: scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [RetrievedChunk(**{**c.__dict__, "relevance_score": score}) for score, c in scored[:top_k]]

class ChromaRetriever(Retriever):
    def __init__(self, settings: Settings, embedding=None):
        import chromadb
        self.embedding = embedding or create_embedding_provider(settings)
        self.client = chromadb.PersistentClient(path=settings.chroma_path)
        provider_name = getattr(self.embedding, "provider_name", settings.embedding_provider)
        model_name = getattr(self.embedding, "model_name", settings.embedding_model)
        # Collection metadata prevents silently querying old vectors with a different encoder.
        identity = {"embedding_provider": provider_name, "embedding_model": model_name,
                    "embedding_version": getattr(self.embedding, "version", "custom"), "hnsw:space": "cosine"}
        self.collection = self.client.get_or_create_collection(settings.chroma_collection, metadata=identity)
        metadata = self.collection.metadata or {}
        for key in ("embedding_provider", "embedding_model", "embedding_version", "hnsw:space"):
            if metadata.get(key) != identity[key]:
                raise ValueError(f"Chroma collection {settings.chroma_collection!r} has incompatible {key}; choose a new collection and re-ingest")
        log.info("embedding_config provider=%s model=%s dimension=%s", provider_name, model_name, getattr(self.embedding, "dimension", "unknown"))
    def retrieve(self, query, filters=None, top_k=5):
        where = filters or None
        if self.collection.count() == 0 or top_k <= 0: return []
        # The configured collection uses cosine distance, so convert distance to similarity.
        # Fetch extra neighbors so version filtering can still leave enough usable results.
        fetch_k = min(self.collection.count(), max(top_k * 20, 50))
        result = self.collection.query(query_embeddings=self.embedding.embed([query]), n_results=fetch_k, where=where, include=["documents", "metadatas", "distances"])
        rows = []
        for i, content in enumerate((result.get("documents") or [[]])[0]):
            md = (result.get("metadatas") or [[]])[0][i] or {}
            if md.get("embedding_version") != getattr(self.embedding, "version", None):
                continue
            distance = (result.get("distances") or [[]])[0][i] if result.get("distances") else None
            similarity = max(0.0, min(1.0, 1.0 - float(distance))) if distance is not None else None
            rows.append(RetrievedChunk(content=content, document_name=md.get("document_name") or md.get("source"), page_number=md.get("page_number"), chunk_id=(result.get("ids") or [[]])[0][i], metadata=md, relevance_score=similarity))
        rows.sort(key=lambda chunk: chunk.relevance_score or 0.0, reverse=True)
        log.info("dense_retrieval query_chars=%d candidates=%d preview=%s", len(query), len(rows),
                 [(row.document_name, row.page_number, round(row.relevance_score or 0, 4)) for row in rows[:5]])
        return rows[:top_k]


_BM25_TOKEN = re.compile(r"[a-z0-9]+", re.I)


def _bm25_tokens(text: str) -> list[str]:
    # A tiny suffix normalizer handles common singular/plural and verb variants without a runtime corpus dependency.
    words = _BM25_TOKEN.findall(text.casefold())
    normalized = []
    for word in words:
        if len(word) > 5 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 5 and word.endswith("ing"):
            word = word[:-3]
        elif len(word) > 4 and word.endswith("ed"):
            word = word[:-2]
        elif len(word) > 4 and word.endswith("s"):
            word = word[:-1]
        normalized.append(word)
    return normalized


class SparseBM25Index:
    """Small in-process BM25 index built from the current Chroma chunk snapshot."""
    def __init__(self, chunks: list[RetrievedChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [_bm25_tokens(chunk.content) for chunk in chunks]
        self.lengths = [len(doc) for doc in self.docs]
        self.avgdl = statistics.mean(self.lengths) if self.lengths else 1.0
        self.df: Counter[str] = Counter()
        for doc in self.docs:
            self.df.update(set(doc))

    def retrieve(self, query: str, limit: int, filters: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        terms = _bm25_tokens(query)
        if not terms or limit <= 0 or not self.chunks:
            return []
        scored = []
        n = len(self.chunks)
        for chunk, doc, length in zip(self.chunks, self.docs, self.lengths):
            if filters and any(chunk.metadata.get(key) != value for key, value in filters.items()):
                continue
            counts = Counter(doc)
            score = 0.0
            for term in terms:
                freq = counts.get(term, 0)
                if freq:
                    idf = math.log(1.0 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                    score += idf * freq * (self.k1 + 1.0) / (freq + self.k1 * (1.0 - self.b + self.b * length / self.avgdl))
            if score > 0:
                scored.append((score, chunk.chunk_id or chunk.content, chunk))
        scored.sort(key=lambda item: (-item[0], item[1]))
        maximum = scored[0][0] if scored else 1.0
        return [RetrievedChunk(**{**chunk.__dict__, "relevance_score": score / maximum})
                for score, _, chunk in scored[:limit]]


def reciprocal_rank_fusion(dense: list[RetrievedChunk], sparse: list[RetrievedChunk], rrf_k: int = 60,
                           limit: int = 60) -> list[RetrievedChunk]:
    """Fuse ranked lists by rank contributions; repeated chunks get both votes."""
    scores: Counter[str] = Counter()
    by_id: dict[str, RetrievedChunk] = {}
    for rows in (dense, sparse):
        for rank, chunk in enumerate(rows, 1):
            key = chunk.chunk_id or hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
            by_id[key] = chunk
            scores[key] += 1.0 / (rrf_k + rank)
    ordered = sorted(scores, key=lambda key: (-scores[key], key))[:max(0, limit)]
    highest = max((scores[key] for key in ordered), default=1.0)
    return [RetrievedChunk(**{**by_id[key].__dict__, "relevance_score": scores[key] / highest,
                              "metadata": {**(by_id[key].metadata or {}), "hybrid_score": scores[key] / highest,
                                           "retrieval_method": "hybrid_cross_encoder"}})
            for key in ordered]


@lru_cache(maxsize=4)
def _load_cross_encoder(model_name: str, device: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name, device=device, max_length=512)


def _cross_encoder_device(requested: str) -> str:
    if requested.casefold() != "auto":
        if requested.casefold().startswith("cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    return requested
            except ImportError:
                pass
            log.warning("CUDA requested for Cross-Encoder but unavailable; using CPU")
            return "cpu"
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class HybridCrossEncoderRetriever(Retriever):
    """Production retrieval: Chroma + BM25, RRF, then bounded Cross-Encoder reranking."""
    method = "hybrid_cross_encoder"

    def __init__(self, dense: ChromaRetriever, settings: Settings, *, cross_encoder=None, cross_encoder_loader=None):
        self.dense, self.settings = dense, settings
        self.cross_encoder = cross_encoder
        self.cross_encoder_loader = cross_encoder_loader or _load_cross_encoder
        self._sparse_index = SparseBM25Index([])
        self.last_timings_ms = {}
        self._model_lock = threading.Lock()
        self.refresh_sparse_index()

    @property
    def collection(self): return self.dense.collection
    @property
    def embedding(self): return self.dense.embedding

    def __getattr__(self, name):
        # Preserve the Chroma access used by ingestion and readiness endpoints.
        if name in {"collection", "embedding"}:
            return getattr(self.dense, name)
        raise AttributeError(name)

    def refresh_sparse_index(self):
        stored = self.collection.get(include=["documents", "metadatas"])
        chunks = []
        for chunk_id, content, metadata in zip(stored.get("ids", []), stored.get("documents", []), stored.get("metadatas", [])):
            metadata = metadata or {}
            if metadata.get("embedding_version") != getattr(self.embedding, "version", None):
                continue
            chunks.append(RetrievedChunk(content=content,
                         document_name=metadata.get("document_name") or metadata.get("source"),
                         page_number=metadata.get("page_number"), chunk_id=chunk_id, metadata=metadata))
        # Replace the snapshot in one assignment so concurrent readers use a consistent index.
        self._sparse_index = SparseBM25Index(chunks)
        log.info("sparse_index_refreshed chunks=%d", len(chunks))
        return len(chunks)

    def _model(self):
        if self.cross_encoder is None:
            with self._model_lock:
                if self.cross_encoder is None:
                    device = _cross_encoder_device(self.settings.cross_encoder_device)
                    started = time.perf_counter()
                    self.cross_encoder = self.cross_encoder_loader(self.settings.cross_encoder_model, device)
                    log.info("cross_encoder_loaded model=%s device=%s load_ms=%.1f", self.settings.cross_encoder_model, device,
                             (time.perf_counter() - started) * 1000)
        return self.cross_encoder

    def retrieve(self, query: str, filters=None, top_k: int | None = None) -> list[RetrievedChunk]:
        final_k = self.settings.top_k if top_k is None else top_k
        if not query.strip() or final_k <= 0:
            return []
        started = time.perf_counter()
        dense_started = time.perf_counter()
        dense_k = max(self.settings.dense_candidate_k, final_k)
        sparse_k = max(self.settings.sparse_candidate_k, final_k)
        dense_rows = self.dense.retrieve(query, filters=filters, top_k=dense_k)
        dense_ms = (time.perf_counter() - dense_started) * 1000
        sparse_started = time.perf_counter()
        sparse_rows = self._sparse_index.retrieve(query, sparse_k, filters)
        sparse_ms = (time.perf_counter() - sparse_started) * 1000
        fusion_started = time.perf_counter()
        # A request may ask for more final results than the configured baseline
        # rerank pool. Expand only enough to satisfy that explicit final top-K.
        rerank_k = max(self.settings.rerank_candidate_k, final_k)
        fused = reciprocal_rank_fusion(dense_rows, sparse_rows, self.settings.rrf_k, rerank_k)
        fusion_ms = (time.perf_counter() - fusion_started) * 1000
        if not fused:
            self.last_timings_ms = {"dense": dense_ms, "sparse": sparse_ms, "fusion": fusion_ms,
                                    "cross_encoder": 0.0, "total": (time.perf_counter() - started) * 1000}
            log.info("retrieval method=%s query_chars=%d dense=%d sparse=%d fused=0 reranked=0 timings_ms=dense:%.1f,sparse:%.1f,fusion:%.1f,total:%.1f",
                     self.method, len(query), len(dense_rows), len(sparse_rows), dense_ms, sparse_ms, fusion_ms,
                     (time.perf_counter() - started) * 1000)
            return []
        rerank_started = time.perf_counter()
        model = self._model()
        scores = model.predict([(query, chunk.content) for chunk in fused],
                               batch_size=self.settings.cross_encoder_batch_size, show_progress_bar=False)
        rerank_ms = (time.perf_counter() - rerank_started) * 1000
        rows = []
        for raw_score, chunk in zip(scores, fused):
            score = float(raw_score)
            # Cross-Encoder logits are not cosine similarities and can be negative
            # for relevant passages; calibrate before applying the existing [0,1] floor.
            calibrated = score + self.settings.cross_encoder_score_offset
            normalized = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, calibrated))))
            rows.append(RetrievedChunk(**{**chunk.__dict__, "relevance_score": normalized,
                         "metadata": {**(chunk.metadata or {}), "cross_encoder_raw_score": score,
                                      "cross_encoder_score": score,
                                      "retrieval_method": self.method}}))
        rows.sort(key=lambda chunk: (-float((chunk.metadata or {}).get("cross_encoder_score", 0.0)), chunk.chunk_id or ""))
        total_ms = (time.perf_counter() - started) * 1000
        self.last_timings_ms = {"dense": dense_ms, "sparse": sparse_ms, "fusion": fusion_ms,
                                "cross_encoder": rerank_ms, "total": total_ms}
        log.info("retrieval method=%s query_chars=%d dense=%d sparse=%d fused=%d reranked=%d final=%d configured_dense=%d configured_sparse=%d configured_rerank=%d effective_rerank=%d timings_ms=dense:%.1f,sparse:%.1f,fusion:%.1f,cross_encoder:%.1f,total:%.1f results=%s",
                 self.method, len(query), len(dense_rows), len(sparse_rows), len(fused), len(rows), min(final_k, len(rows)),
                 self.settings.dense_candidate_k, self.settings.sparse_candidate_k, self.settings.rerank_candidate_k, rerank_k,
                 dense_ms, sparse_ms, fusion_ms, rerank_ms, total_ms,
                 [(c.document_name, c.page_number,
                   round(float((c.metadata or {}).get("cross_encoder_raw_score", 0.0)), 4),
                   round(c.relevance_score or 0, 4)) for c in rows[:final_k]])
        return rows[:final_k]

class ContextBuilder:
    def __init__(self, max_chars=12000, min_relevance_score=0.35):
        self.max_chars, self.min_relevance_score = max_chars, min_relevance_score
    def build(self, chunks: list[RetrievedChunk]) -> GroundedContext:
        # Retrieval relevance is only an initial gate; task-specific evidence checks happen later.
        selected, seen, parts = [], set(), []
        remaining = self.max_chars
        for c in chunks:
            if c.relevance_score is None or c.relevance_score < self.min_relevance_score: continue
            key = c.content.strip()
            if not key or key in seen: continue
            separator = "\n\n" if parts else ""
            allowance = remaining - len(separator)
            if allowance <= 0: break
            bounded = key[:allowance]
            if not bounded: break
            seen.add(key)
            selected.append(RetrievedChunk(**{**c.__dict__, "content": bounded}))
            parts.append(separator + bounded)
            remaining -= len(separator) + len(bounded)
            if len(bounded) < len(key): break
        sufficient = bool(selected)
        return GroundedContext(selected, "".join(parts), "sufficient" if sufficient else "insufficient", "grounded" if sufficient else "not_grounded")

def chunk_text(text: str, chunk_chars: int = 1000, overlap_chars: int = 120) -> list[tuple[int, str]]:
    if chunk_chars <= 0 or overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise ValueError("chunk_chars must be positive and overlap_chars must be in [0, chunk_chars)")
    chunks, start = [], 0
    # Prefer a nearby word boundary when it does not make the chunk much too short.
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + chunk_chars // 2: end = boundary
        chunk = text[start:end].strip()
        if chunk: chunks.append((start, chunk))
        if end >= len(text): break
        start = max(start + 1, end - overlap_chars)
    return chunks

def _stable_chunk_id(source: str, page: int, start: int, content: str) -> str:
    # Stable IDs make re-ingestion an upsert and allow obsolete chunks to be replaced safely.
    raw = f"{source.casefold()}\0{page}\0{start}\0{content}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def _source_key(path: Path) -> str:
    return hashlib.sha256(path.resolve().as_posix().casefold().encode("utf-8")).hexdigest()

def _upsert_page_chunks(path: Path, pages: list[tuple[int, str]], collection, embedding, chunk_chars: int, overlap_chars: int, file_hash: str | None = None) -> int:
    source_key = _source_key(path)
    source_path = path.resolve().as_posix()
    ids, docs, metadatas = [], [], []
    # Keep page and source identity on every chunk so answers can point back to the PDF location.
    for page_number, text in pages:
        for start, content in chunk_text(text, chunk_chars, overlap_chars):
            chunk_id = _stable_chunk_id(source_key, page_number, start, content)
            ids.append(chunk_id); docs.append(content)
            metadatas.append({"source": path.name, "source_key": source_key, "source_path": source_path, "source_filename": path.name, "file_hash": file_hash or "", "document": path.name, "document_name": path.name, "page": page_number, "page_number": page_number, "chunk_id": chunk_id, "start_char": start, "embedding_provider": getattr(embedding, "provider_name", "custom"), "embedding_model": getattr(embedding, "model_name", "custom"), "embedding_version": getattr(embedding, "version", "custom")})
    # Restrict replacement to the stable path identity; same-named files elsewhere remain untouched.
    existing = collection.get(where={"source_key": source_key}, include=["metadatas"])
    keep = set(ids)
    stale = [old_id for old_id, metadata in zip(existing.get("ids", []), existing.get("metadatas", []))
             if old_id not in keep and (metadata or {}).get("source_key") == source_key]
    stale = list(set(stale))
    # Prepare replacement vectors before deleting stale IDs, so failed embedding leaves old content intact.
    vectors = embedding.embed(docs) if ids else []
    if ids: collection.upsert(ids=ids, documents=docs, embeddings=vectors, metadatas=metadatas)
    if stale: collection.delete(ids=stale)
    return len(ids)

def _configured_embedding(embedding):
    return embedding or create_embedding_provider(Settings())

def ingest_text(path: str, collection, embedding=None, chunk_chars=1200, overlap_chars=120):
    file_path = Path(path)
    return _upsert_page_chunks(file_path, [(1, file_path.read_text(encoding="utf-8"))], collection, _configured_embedding(embedding), chunk_chars, overlap_chars)

def ingest_pdf(path: str | Path, collection, embedding=None, chunk_chars=1000, overlap_chars=120) -> dict[str, int]:
    from pypdf import PdfReader
    file_path = Path(path)
    reader = PdfReader(str(file_path))
    # Extraction stays page-scoped; empty pages are omitted from indexing but page numbers remain original.
    pages = [(index + 1, page.extract_text() or "") for index, page in enumerate(reader.pages)]
    nonempty_pages = [(number, text) for number, text in pages if text.strip()]
    file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
    count = _upsert_page_chunks(file_path, nonempty_pages, collection, _configured_embedding(embedding), chunk_chars, overlap_chars, file_hash)
    return {"pages": len(reader.pages), "pages_with_text": len(nonempty_pages), "chunks": count}

@contextmanager
def _ingestion_lock(lock_path: Path, timeout: float = 15.0):
    """Cross-process lock shared by reload workers; deterministic IDs remain the final idempotency guard."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    if handle.tell() == 0:
        handle.write(b"0"); handle.flush()
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (OSError, BlockingIOError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for PDF ingestion lock: {lock_path}")
                time.sleep(0.1)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()

def sync_pdf_directory(directory: str | Path, collection, embedding, *, chroma_path: str | Path, chunk_chars: int = 1000, overlap_chars: int = 120) -> dict[str, int]:
    """Ingest only new/changed PDFs into the caller's existing Chroma collection."""
    root = Path(directory)
    pdfs = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf") if root.exists() else []
    root = root.resolve()
    totals = {"discovered": len(pdfs), "new": 0, "updated": 0, "skipped": 0, "removed": 0, "failed": 0, "chunks": 0}
    log.info("RAG auto-ingestion started directory=%s", root.resolve())
    log.info("Found %d PDF files", len(pdfs))
    with _ingestion_lock(Path(chroma_path) / ".pdf_ingestion.lock"):
        for pdf in pdfs:
            try:
                fingerprint = hashlib.sha256(pdf.read_bytes()).hexdigest()
                source_key = _source_key(pdf)
                existing = collection.get(where={"source_key": source_key}, include=["metadatas"])
                metadata = existing.get("metadatas", [])
                # File hashes avoid repeating extraction and embedding when a PDF is unchanged.
                if metadata and all((item or {}).get("file_hash") == fingerprint for item in metadata):
                    totals["skipped"] += 1
                    log.info("Skipping unchanged PDF: %s", pdf.name)
                    continue
                is_update = bool(existing.get("ids"))
                log.info("%s PDF: %s", "Re-ingesting modified" if is_update else "Ingesting new", pdf.name)
                result = ingest_pdf(pdf, collection, embedding, chunk_chars, overlap_chars)
                totals["updated" if is_update else "new"] += 1
                totals["chunks"] += result["chunks"]
            except Exception:
                totals["failed"] += 1
                log.exception("PDF ingestion failed: %s", pdf)
        # Remove only PDFs previously synchronized from this directory that no longer exist.
        # Other directories/documents sharing the collection remain untouched.
        live_paths = {path.resolve().as_posix().casefold() for path in pdfs}
        stored = collection.get(include=["metadatas"])
        stale_ids = []
        for chunk_id, metadata in zip(stored.get("ids", []), stored.get("metadatas", [])):
            source_path = (metadata or {}).get("source_path")
            if not source_path:
                continue
            try:
                stored_path = Path(source_path).resolve()
                if stored_path.is_relative_to(root) and stored_path.suffix.casefold() == ".pdf" and stored_path.as_posix().casefold() not in live_paths:
                    stale_ids.append(chunk_id)
            except (OSError, ValueError):
                continue
        if stale_ids:
            collection.delete(ids=stale_ids)
            totals["removed"] = len(stale_ids)
    log.info("RAG auto-ingestion completed: %d new, %d updated, %d skipped, %d removed chunks, %d failures; %d chunks", totals["new"], totals["updated"], totals["skipped"], totals["removed"], totals["failed"], totals["chunks"])
    return totals
