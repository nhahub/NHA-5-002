"""Reproducible local retrieval benchmark; does not change production retrieval."""
from __future__ import annotations

import json
import math
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.domain import RetrievedChunk
from app.rag import ChromaRetriever

ROOT = Path(__file__).resolve().parents[1]
QUERY_PATH = ROOT / "data" / "retrieval_benchmark_queries.json"
OUT_PATH = ROOT / "data" / "retrieval_benchmark_results.json"
TOP_K = 10
SEMANTIC_CANDIDATES = 20
LEXICAL_CANDIDATES = 20
FUSION_POOL = 20
RRF_K = 60
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def rrf(semantic: list[RetrievedChunk], lexical: list[RetrievedChunk], limit: int) -> list[RetrievedChunk]:
    by_id: dict[str, RetrievedChunk] = {}
    scores: Counter[str] = Counter()
    for rows in (semantic, lexical):
        for rank, chunk in enumerate(rows, 1):
            if chunk.chunk_id is None:
                continue
            by_id[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] += 1.0 / (RRF_K + rank)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:limit]
    maximum = max((scores[cid] for cid in ordered), default=1.0)
    return [RetrievedChunk(**{**by_id[cid].__dict__, "relevance_score": scores[cid] / maximum}) for cid in ordered]


class BM25:
    """Deterministic BM25 (k1=1.5, b=0.75) over the exact Chroma chunk corpus."""
    def __init__(self, chunks: list[RetrievedChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [tokens(chunk.content) for chunk in chunks]
        self.lengths = [len(doc) for doc in self.docs]
        self.avgdl = statistics.mean(self.lengths) if self.lengths else 1.0
        self.df: Counter[str] = Counter()
        for doc in self.docs:
            self.df.update(set(doc))

    def retrieve(self, query: str, limit: int) -> list[RetrievedChunk]:
        terms = tokens(query)
        n = len(self.docs)
        scored = []
        for chunk, doc, dl in zip(self.chunks, self.docs, self.lengths):
            tf = Counter(doc)
            score = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                idf = math.log(1.0 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                score += idf * freq * (self.k1 + 1.0) / (freq + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl))
            if score > 0:
                scored.append((score, chunk.chunk_id or "", chunk))
        scored.sort(key=lambda row: (-row[0], row[1]))
        max_score = scored[0][0] if scored else 1.0
        return [RetrievedChunk(**{**chunk.__dict__, "relevance_score": score / max_score}) for score, _, chunk in scored[:limit]]


def ground_truth_ids(pages: list[int], corpus: list[RetrievedChunk]) -> set[str]:
    wanted = set(pages)
    return {c.chunk_id for c in corpus if c.chunk_id and c.page_number in wanted}


def metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    ks = (1, 3, 5, 10)
    metrics: dict[str, Any] = {}
    for k in ks:
        recalls, hits = [], []
        for row in records:
            relevant = set(row["ground_truth_chunk_ids"])
            found = set(row["ranked_chunk_ids"][:k]) & relevant
            recalls.append(len(found) / len(relevant) if relevant else 0.0)
            hits.append(1.0 if found else 0.0)
        metrics[f"recall@{k}"] = sum(recalls) / count if count else 0.0
        metrics[f"hit_rate@{k}"] = sum(hits) / count if count else 0.0
    reciprocal = []
    for row in records:
        relevant = set(row["ground_truth_chunk_ids"])
        rank = next((i for i, chunk_id in enumerate(row["ranked_chunk_ids"], 1) if chunk_id in relevant), None)
        reciprocal.append(1.0 / rank if rank else 0.0)
    metrics["mrr"] = sum(reciprocal) / count if count else 0.0
    metrics["avg_latency_ms"] = statistics.mean(row["latency_ms"] for row in records) if records else 0.0
    metrics["avg_candidate_count"] = statistics.mean(row["candidate_count"] for row in records) if records else 0.0
    metrics["zero_hit_queries"] = sum(not (set(row["ranked_chunk_ids"]) & set(row["ground_truth_chunk_ids"])) for row in records)
    return metrics


def main() -> None:
    dataset = json.loads(QUERY_PATH.read_text(encoding="utf-8"))
    settings = get_settings()
    chroma = ChromaRetriever(settings)
    stored = chroma.collection.get(include=["documents", "metadatas"])
    corpus = []
    for chunk_id, content, metadata in zip(stored["ids"], stored["documents"], stored["metadatas"]):
        metadata = metadata or {}
        if metadata.get("embedding_version") != getattr(chroma.embedding, "version", None):
            continue
        corpus.append(RetrievedChunk(content=content, chunk_id=chunk_id,
                     document_name=metadata.get("document_name") or metadata.get("source"),
                     page_number=metadata.get("page_number"), metadata=metadata))
    corpus.sort(key=lambda c: c.chunk_id or "")
    lexical = BM25(corpus)

    # Warm the semantic encoder and persistent collection outside measured query timings.
    chroma.retrieve("Ancient Egypt Nile", top_k=1)
    variants: dict[str, list[dict[str, Any]]] = {"current": [], "hybrid": [], "hybrid_cross_encoder": []}
    cross_encoder = None
    cross_encoder_load_seconds = None
    for query_row in dataset["queries"]:
        query = query_row["query"]
        truth = ground_truth_ids(query_row["relevant_pages"], corpus)
        if not truth:
            raise ValueError(f"No indexed chunks correspond to ground-truth pages for {query_row['id']}: {query_row['relevant_pages']}")

        started = time.perf_counter()
        current = chroma.retrieve(query, top_k=TOP_K)
        elapsed = (time.perf_counter() - started) * 1000
        variants["current"].append(record(query_row, truth, current[:TOP_K], elapsed,
                                          candidate_stages={"semantic_candidates": len(current)}))

        started = time.perf_counter()
        sem = chroma.retrieve(query, top_k=SEMANTIC_CANDIDATES)
        lex = lexical.retrieve(query, LEXICAL_CANDIDATES)
        fused = rrf(sem, lex, FUSION_POOL)
        elapsed = (time.perf_counter() - started) * 1000
        variants["hybrid"].append(record(query_row, truth, fused[:TOP_K], elapsed,
                                          candidate_stages={"semantic_candidates": len(sem),
                                                            "lexical_candidates": len(lex),
                                                            "fused_candidates": len(fused)}))

        if cross_encoder is None:
            from sentence_transformers import CrossEncoder
            started_load = time.perf_counter()
            cross_encoder = CrossEncoder(CROSS_ENCODER_NAME, device="cpu", max_length=512)
            cross_encoder_load_seconds = time.perf_counter() - started_load
        started = time.perf_counter()
        sem = chroma.retrieve(query, top_k=SEMANTIC_CANDIDATES)
        lex = lexical.retrieve(query, LEXICAL_CANDIDATES)
        pool = rrf(sem, lex, FUSION_POOL)
        retrieval_fusion_ms = (time.perf_counter() - started) * 1000
        rerank_started = time.perf_counter()
        raw_scores = cross_encoder.predict([(query, chunk.content) for chunk in pool], show_progress_bar=False)
        reranked = [RetrievedChunk(**{**chunk.__dict__, "relevance_score": float(score)})
                    for score, chunk in sorted(zip(raw_scores, pool), key=lambda row: (-float(row[0]), row[1].chunk_id or ""))]
        rerank_ms = (time.perf_counter() - rerank_started) * 1000
        elapsed = retrieval_fusion_ms + rerank_ms
        variants["hybrid_cross_encoder"].append(record(
            query_row, truth, reranked[:TOP_K], elapsed,
            candidate_stages={"semantic_candidates": len(sem), "lexical_candidates": len(lex),
                              "fused_candidates": len(pool), "reranked_candidates": len(reranked)},
            stage_latencies_ms={"retrieval_and_fusion": retrieval_fusion_ms, "cross_encoder_reranking": rerank_ms}))

    report = {
        "dataset": dataset,
        "protocol": {"top_k": TOP_K, "semantic_candidates": SEMANTIC_CANDIDATES,
                     "lexical_candidates": LEXICAL_CANDIDATES, "fusion_pool": FUSION_POOL,
                     "fusion": "Reciprocal Rank Fusion", "rrf_k": RRF_K,
                     "bm25": {"k1": 1.5, "b": 0.75, "tokenizer": "lowercase alphanumeric regex"},
                     "cross_encoder": CROSS_ENCODER_NAME, "cross_encoder_device": "cpu",
                     "metric_relevant_unit": "all indexed Ancient_Egypt_From.pdf chunks whose physical PDF page is annotated relevant",
                     "latency": "warm model, per-query retrieval wall time; includes semantic encode/query, lexical/fusion, and rerank as applicable; excludes model loading"},
        "corpus": {"chunks": len(corpus), "source_documents": dict(Counter(c.document_name for c in corpus)),
                   "ground_truth_chunk_counts_by_query": {row["id"]: len(ground_truth_ids(row["relevant_pages"], corpus)) for row in dataset["queries"]}},
        "cross_encoder_load_seconds": cross_encoder_load_seconds,
        "variants": {name: {"metrics": metric_summary(rows), "queries": rows} for name, rows in variants.items()}
    }
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, data in report["variants"].items():
        print(name, json.dumps(data["metrics"], sort_keys=True))
    print("cross_encoder_load_seconds", cross_encoder_load_seconds)
    print("output", OUT_PATH)


def record(query_row: dict[str, Any], truth: set[str], chunks: list[RetrievedChunk], latency_ms: float,
           *, candidate_stages: dict[str, int] | None = None,
           stage_latencies_ms: dict[str, float] | None = None) -> dict[str, Any]:
    return {"query_id": query_row["id"], "query": query_row["query"], "relevant_pages": query_row["relevant_pages"],
            "ground_truth_chunk_ids": sorted(truth), "ranked_chunk_ids": [c.chunk_id for c in chunks],
            "retrieved_pages": [c.page_number for c in chunks], "scores": [c.relevance_score for c in chunks],
            "ranked_results": [{"rank": rank, "chunk_id": c.chunk_id, "page": c.page_number,
                                "score": c.relevance_score, "relevant": c.chunk_id in truth}
                               for rank, c in enumerate(chunks, 1)],
            "relevant_at_1": bool(chunks and chunks[0].chunk_id in truth),
            "relevant_at_3": any(c.chunk_id in truth for c in chunks[:3]),
            "relevant_at_5": any(c.chunk_id in truth for c in chunks[:5]),
            "relevant_at_10": any(c.chunk_id in truth for c in chunks[:10]),
            "latency_ms": latency_ms, "candidate_count": len(chunks),
            "candidate_stages": candidate_stages or {}, "stage_latencies_ms": stage_latencies_ms or {}}


if __name__ == "__main__":
    main()
