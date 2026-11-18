"""Integration adapter for the approved ai_depi Generative Learning Model.

The only corpus indexed here is the current user's uploaded course material.
"""
from __future__ import annotations

import logging
import hashlib
import os
from pathlib import Path
import re
from typing import Any

from ai_depi.app.config import Settings as LearningSettings
from ai_depi.app.generation import LearningGenerator
from ai_depi.app.providers import create_llm_provider, create_embedding_provider
from ai_depi.app.rag import ChromaRetriever, HybridCrossEncoderRetriever, sync_learning_material_directory

from backend.app.config import settings as app_settings

log = logging.getLogger("adaptive_coach.learning_agent")


class DocumentScopedRetriever:
    """Apply an optional uploaded-file scope to both dense and sparse retrieval."""

    def __init__(self, retriever, document_name: str | None = None):
        self.retriever = retriever
        self.document_name = Path(document_name).name if document_name else None
        self.method = getattr(retriever, "method", "hybrid_cross_encoder")

    def retrieve(self, query: str, filters: dict[str, Any] | None = None, top_k: int | None = None):
        selected = dict(filters or {})
        if self.document_name:
            selected["source_filename"] = self.document_name
        return self.retriever.retrieve(query, filters=selected or None, top_k=top_k)

    def retrieve_document_all(self):
        """Return the complete indexed document in original page/chunk order."""
        if not self.document_name:
            return []
        result = self.retriever.collection.get(
            where={"source_filename": self.document_name},
            include=["documents", "metadatas"],
        )
        from ai_depi.app.domain import RetrievedChunk
        chunks = []
        for chunk_id, content, metadata in zip(
            result.get("ids", []), result.get("documents", []), result.get("metadatas", [])
        ):
            metadata = metadata or {}
            if metadata.get("embedding_version") != getattr(self.retriever.embedding, "version", None):
                continue
            chunks.append(RetrievedChunk(
                content=content,
                document_name=metadata.get("document_name") or self.document_name,
                page_number=metadata.get("page") or metadata.get("page_number"),
                chunk_id=chunk_id,
                metadata=metadata,
                relevance_score=1.0,
            ))
        return sorted(chunks, key=lambda row: (
            row.page_number or 0, (row.metadata or {}).get("start_char", 0), row.chunk_id or ""
        ))


_AR_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u0640]")


def _norm_token(tok: str) -> str:
    tok = _AR_DIACRITICS.sub("", tok.casefold())
    tok = tok.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"}))
    if len(tok) > 4 and tok.startswith("ال"):
        tok = tok[2:]
    return tok


def tokenize(text: str) -> list[str]:
    return [t for t in (_norm_token(x) for x in re.findall(r"\w+", text, flags=re.UNICODE)) if len(t) > 1]


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for a PDF/DOCX/TXT file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        return [(i, (page.extract_text() or "")) for i, page in enumerate(PdfReader(path).pages, start=1)]
    if suffix == ".docx":
        from docx import Document
        return [(1, "\n".join(p.text for p in Document(path).paragraphs))]
    return [(1, path.read_text(encoding="utf-8", errors="replace"))]


def split_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    if len(text) <= size:
        return [text] if len(text) > 30 else []
    pieces, start = [], 0
    while start < len(text):
        pieces.append(text[start:start + size])
        start += size - overlap
    return pieces


def rank_passages(query: str, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> list[dict[str, Any]]:
    """Small BM25 ranking (works for Arabic and English, no external model needed)."""
    import math
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return chunks[:6]
    docs = [tokenize(c["content"]) for c in chunks]
    avg = (sum(len(d) for d in docs) / len(docs)) or 1.0
    df = {t: sum(1 for d in docs if t in d) for t in q_tokens}
    n = len(docs)
    scored = []
    for chunk, toks in zip(chunks, docs):
        score = 0.0
        for t in q_tokens:
            tf = toks.count(t)
            if tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(toks) / avg))
        if score > 0:
            scored.append((score, chunk))
    return [c for _, c in sorted(scored, key=lambda x: -x[0])]


class GenerativeLearningAgent:
    """Owns the ai_depi retrieval, evidence-grounding, and generation pipeline."""

    def __init__(self):
        from backend.app.database import db

        self.uploads_dir = db.uploads_dir
        self.chroma_path = app_settings.data_dir / "learning_materials_chroma"
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "hash").strip().casefold()
        embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").strip()
        embedding_key = hashlib.sha256(f"{embedding_provider}:{embedding_model}".encode()).hexdigest()[:10]
        collection_name = f"user_learning_{re.sub(r'[^a-z0-9_]+', '_', embedding_provider)[:30]}_{embedding_key}"
        groq_key = app_settings.groq_api_key or os.getenv("GROQ_API_KEY")
        gemini_key = (app_settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
                      or os.getenv("GOOGLE_API_KEY"))
        gemini_model = (os.getenv("GEMINI_MODEL") or os.getenv("COACH_MODEL")
                        or "gemini-3.8-flash")
        self.settings = LearningSettings(
            llm_provider="groq" if groq_key else ("gemini" if gemini_key else "deterministic"),
            # Keep the grounded tutor available when either configured provider
            # has a temporary outage, quota issue, or unsupported request.
            llm_fallback_provider=("gemini" if groq_key and gemini_key else ""),
            groq_api_key=app_settings.groq_api_key,
            gemini_api_key=gemini_key,
            gemini_model=gemini_model,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            chroma_path=str(self.chroma_path),
            chroma_collection=collection_name,
            sample_data_dir=str(self.uploads_dir),
            auto_ingest_pdfs=False,
        )
        self.embedding = create_embedding_provider(self.settings)
        self.dense = ChromaRetriever(self.settings, embedding=self.embedding)
        self.retriever = HybridCrossEncoderRetriever(self.dense, self.settings)
        self.provider = create_llm_provider(self.settings)
        self.claim_judge = self.provider
        self.last_sync: dict[str, Any] = {"status": "not_synced"}

    def sync_materials(self) -> dict[str, Any]:
        try:
            result = sync_learning_material_directory(
                self.uploads_dir,
                self.retriever.collection,
                self.embedding,
                chroma_path=self.chroma_path,
                chunk_chars=self.settings.chunk_chars,
                overlap_chars=self.settings.chunk_overlap_chars,
            )
            count = self.retriever.refresh_sparse_index()
            self.last_sync = {"status": "ready", "chunks": count, **result}
        except Exception as exc:
            log.exception("Could not synchronize uploaded study materials")
            self.last_sync = {"status": "error", "reason": type(exc).__name__, "message": str(exc)[:300]}
        return dict(self.last_sync)

    def status(self) -> dict[str, Any]:
        semantic_embeddings = self.settings.embedding_provider == "sentence_transformers"
        return {
            **self.last_sync,
            "retriever": "ai_depi hybrid Chroma + BM25",
            "embedding_provider": getattr(self.embedding, "provider_name", self.settings.embedding_provider),
            "embedding_quality": "multilingual semantic vectors active" if semantic_embeddings else "lexical fallback; install sentence-transformers to use dense semantic vectors",
            "reranker": "Cross-Encoder lazy load; uses reciprocal-rank fusion if model/runtime unavailable",
            "llm_provider_configured": bool(self.settings.groq_api_key or self.settings.gemini_api_key),
            "llm_primary": self.settings.llm_provider,
            "llm_fallback": self.settings.llm_fallback_provider or None,
            "answer_validation": f"{self.claim_judge.name} structured claim judge",
            "ollama_url": self.settings.ollama_base_url,
            "corpus_scope": "user uploads only",
        }

    def ensure_index(self) -> dict[str, Any]:
        """Re-sync when nothing has been indexed yet (e.g. startup sync failed or files were added)."""
        has_files = any(self.uploads_dir.glob("*"))
        if has_files and (self.last_sync.get("status") != "ready" or not self.last_sync.get("chunks")):
            return self.sync_materials()
        return dict(self.last_sync)

    def diagnose(self, filename: str | None = None) -> dict[str, Any]:
        """Explain why retrieval might return nothing: files, text extraction, index contents, filters."""
        report: dict[str, Any] = {"sync": dict(self.last_sync), "embedding_provider": self.settings.embedding_provider}
        files = []
        for path in sorted(self.uploads_dir.glob("*")):
            if path.suffix.lower() not in {".pdf", ".docx", ".txt"}:
                continue
            try:
                pages = extract_pages(path)
                chars = sum(len(text) for _, text in pages)
            except Exception as exc:  # noqa: BLE001
                pages, chars = [], -1
                report.setdefault("extract_errors", {})[path.name] = f"{type(exc).__name__}: {exc}"
            files.append({"file": path.name, "pages": len(pages), "text_chars": chars,
                          "readable": chars > 200,
                          "hint": None if chars > 200 else "الملف مفيهوش نص قابل للقراءة (غالبًا PDF صور/ممسوح ضوئيًا) — محتاج OCR."})
        report["files"] = files
        try:
            collection = self.retriever.collection
            report["index_total_chunks"] = collection.count()
            sample = collection.get(limit=3, include=["metadatas"])
            report["metadata_keys_sample"] = sorted({k for m in (sample.get("metadatas") or []) for k in (m or {})})
            if filename:
                name = Path(filename).name
                got = collection.get(where={"source_filename": name}, limit=1)
                report["chunks_for_filename_filter"] = len(got.get("ids", []))
                report["filter_hint"] = (None if report["chunks_for_filename_filter"] else
                                         "الفلتر source_filename مبيلاقيش الملف في الـ index — يا الملف مش متفهرس يا اسم الحقل في الـ metadata مختلف.")
        except Exception as exc:  # noqa: BLE001
            report["index_error"] = f"{type(exc).__name__}: {exc}"
        return report

    def direct_answer(self, query: str, filename: str | None = None, history: str | None = None,
                      top_n: int = 6) -> dict[str, Any]:
        """Fallback that bypasses the vector index: read the uploaded file(s) and answer from the best passages."""
        paths = [self.uploads_dir / Path(filename).name] if filename else \
            [p for p in sorted(self.uploads_dir.glob("*")) if p.suffix.lower() in {".pdf", ".docx", ".txt"}]
        chunks: list[dict[str, Any]] = []
        for path in paths:
            if not path.is_file():
                continue
            for page_no, text in extract_pages(path):
                for piece in split_text(text):
                    chunks.append({"document_name": path.name, "page": page_no, "content": piece})
        if not chunks:
            return {"found": False, "reason": "no_text",
                    "notice": "مقدرتش أستخرج نص من الملف. لو هو PDF صور (ممسوح ضوئيًا) محتاج OCR."}
        ranked = rank_passages(query, chunks)[:top_n]
        if not ranked:
            return {"found": False, "reason": "no_match",
                    "notice": "لم أجد في الملف فقرات مرتبطة بسؤالك."}
        sources = [{**c, "content": c["content"][:400]} for c in ranked]
        context = "\n\n".join(f"[{c['document_name']} - صفحة {c['page']}]\n{c['content']}" for c in ranked)
        answer, mode = None, "extractive_direct"
        key = app_settings.groq_api_key or os.getenv("GROQ_API_KEY")
        if key:
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model=app_settings.groq_model, api_key=key,
                                 base_url="https://api.groq.com/openai/v1", temperature=0.2, max_tokens=1200)
                prompt = (
                    "أنت مساعد دراسة. أجب عن سؤال الطالب بالاعتماد فقط على المقاطع التالية من ملفه. "
                    "اكتب بنفس لغة السؤال، وأشر لرقم الصفحة عند الاستشهاد. إن لم تكفِ المقاطع قل ذلك صراحة ولا تخترع معلومات.\n\n"
                    f"سياق المحادثة السابقة:\n{history or '-'}\n\nالمقاطع:\n{context}\n\nالسؤال: {query}"
                )
                answer = llm.invoke(prompt).content
                mode = "llm_direct_document"
            except Exception as exc:  # noqa: BLE001
                log.warning("Direct-document LLM call failed: %s", exc)
        if not answer:
            answer = "أقرب فقرات من ملفك لسؤالك:\n\n" + "\n\n".join(
                f"(صفحة {c['page']}) {c['content'][:500]}" for c in ranked[:3])
        return {"found": True, "answer": answer, "sources": sources, "mode": mode}

    def generator(self, document_name: str | None = None) -> LearningGenerator:
        scoped = DocumentScopedRetriever(self.retriever, document_name)
        return LearningGenerator(
            scoped,
            max_context_chars=self.settings.max_context_chars,
            llm=self.provider,
            settings=self.settings,
            top_k=self.settings.top_k,
            min_relevance_score=self.settings.min_relevance_score,
            claim_judge_llm=self.claim_judge,
        )

    @staticmethod
    def learner_profile(model: dict[str, Any], assessments: list[dict[str, Any]] | None = None, user_id: str = "") -> dict[str, Any]:
        return {
            "user_id": user_id,
            "topic_mastery": {
                key: max(0.0, min(1.0, float(value) / 100.0))
                for key, value in (model.get("topic_mastery") or {}).items()
            },
            "weak_topics": list(model.get("weak_topics") or []),
            "strong_topics": list(model.get("strong_topics") or []),
            "recent_quiz_results": (assessments or [])[:10],
            "historical_performance": {
                "focus_duration_mins": model.get("effective_focus_duration_mins"),
                "completion_ratio": model.get("avg_task_completion_ratio"),
            },
        }


learning_agent = GenerativeLearningAgent()
