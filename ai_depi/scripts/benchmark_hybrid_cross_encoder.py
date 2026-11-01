"""Evaluate the production hybrid retriever on bundled, multi-document labels."""
from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

from app.config import get_settings
from app.evidence import task_evidence
from app.rag import (ChromaRetriever, ContextBuilder, HybridCrossEncoderRetriever, SparseBM25Index,
                     reciprocal_rank_fusion, sync_pdf_directory)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "hybrid_cross_encoder_queries.json"
OUTPUT = ROOT / "data" / "hybrid_cross_encoder_results.json"
EVAL_TOP_K = 10


def expected(chunk, sources):
    metadata = chunk.metadata or {}
    return any(metadata.get("source_filename") == source["source_filename"] and
               chunk.page_number in source["pages"] for source in sources)


def metrics(records):
    positives = [row for row in records if row["expected_count"]]
    result = {}
    for k in (1, 3, 5, 10):
        result[f"recall@{k}"] = statistics.mean(
            (len(set(row["ranked_ids"][:k]) & set(row["relevant_ids"])) / row["expected_count"])
            for row in positives) if positives else 0.0
        result[f"hit_rate@{k}"] = statistics.mean(
            bool(set(row["ranked_ids"][:k]) & set(row["relevant_ids"])) for row in positives) if positives else 0.0
    result["mrr"] = statistics.mean(
        next((1 / rank for rank, hit in enumerate(row["is_relevant"], 1) if hit), 0.0)
        for row in positives) if positives else 0.0
    latencies = [row["latency_ms"] for row in records]
    result["avg_latency_ms"] = statistics.mean(latencies) if latencies else 0.0
    result["p95_latency_ms"] = percentile95(latencies)
    negatives = [row for row in records if not row["expected_count"]]
    result["negative_queries"] = len(negatives)
    result["negative_retrieval_return_rate"] = statistics.mean(bool(row["ranked_ids"]) for row in negatives) if negatives else 0.0
    result["negative_sufficient_context_rate"] = statistics.mean(row.get("context_sufficient", False) for row in negatives) if negatives else 0.0
    return result


def percentile95(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(.95 * len(ordered)) - 1)]


def record(query, sources, chunks, latency_ms):
    labels = [expected(chunk, sources) for chunk in chunks]
    relevant = [chunk.chunk_id for chunk, hit in zip(chunks, labels) if hit]
    expected_ids = sorted({chunk.chunk_id for chunk in chunks if expected(chunk, sources)})
    # The denominator is every indexed chunk on a manually labeled source/page,
    # not just the chunks retrieved by this query.
    return {"query_id": query["id"], "query": query["query"], "expected_count": 0,
            "relevant_ids": relevant, "ranked_ids": [chunk.chunk_id for chunk in chunks],
            "is_relevant": labels, "retrieved": [{"source": chunk.document_name, "page": chunk.page_number,
                         "chunk_id": chunk.chunk_id, "score": chunk.relevance_score,
                         "cross_encoder_score": (chunk.metadata or {}).get("cross_encoder_score")} for chunk in chunks],
            "latency_ms": latency_ms, "_expected_seen": expected_ids}


def main():
    settings = get_settings()
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    dense = ChromaRetriever(settings)
    sync_result = sync_pdf_directory(settings.sample_data_dir, dense.collection, dense.embedding,
                                     chroma_path=settings.chroma_path, chunk_chars=settings.chunk_chars,
                                     overlap_chars=settings.chunk_overlap_chars)
    pipeline = HybridCrossEncoderRetriever(dense, settings)
    stored = dense.collection.get(include=["documents", "metadatas"])
    corpus = []
    for chunk_id, text, metadata in zip(stored["ids"], stored["documents"], stored["metadatas"]):
        metadata = metadata or {}
        if metadata.get("embedding_version") == getattr(dense.embedding, "version", None):
            from app.domain import RetrievedChunk
            corpus.append(RetrievedChunk(text, metadata.get("source_filename"), metadata.get("page_number"),
                                         chunk_id, metadata))
    sparse = SparseBM25Index(corpus)
    all_expected_counts = {}
    for query in dataset["queries"]:
        all_expected_counts[query["id"]] = sum(expected(chunk, query["expected_sources"]) for chunk in corpus)
        if query["expected_sources"] and not all_expected_counts[query["id"]]:
            raise ValueError(f"No labeled chunks found for {query['id']}: {query['expected_sources']}")

    # Exclude one-time model load and warm up the same real retrieval path used by API calls.
    pipeline.retrieve("warm up hybrid cross encoder", top_k=EVAL_TOP_K)
    variants = {name: [] for name in ("dense", "sparse_bm25", "hybrid_rrf", "hybrid_cross_encoder")}
    ranking_changed = 0
    for query in dataset["queries"]:
        text, sources = query["query"], query["expected_sources"]
        dense_start = time.perf_counter()
        dense_rows = dense.retrieve(text, top_k=settings.dense_candidate_k)
        dense_ms = (time.perf_counter() - dense_start) * 1000
        sparse_start = time.perf_counter()
        sparse_rows = pipeline._sparse_index.retrieve(text, settings.sparse_candidate_k)
        sparse_ms = (time.perf_counter() - sparse_start) * 1000
        fusion_start = time.perf_counter()
        fused = reciprocal_rank_fusion(dense_rows, sparse_rows, settings.rrf_k,
                                       settings.rerank_candidate_k)
        fusion_ms = (time.perf_counter() - fusion_start) * 1000

        variants["dense"].append(record(query, sources, dense_rows[:EVAL_TOP_K], dense_ms))
        variants["sparse_bm25"].append(record(query, sources, sparse_rows[:EVAL_TOP_K], sparse_ms))
        variants["hybrid_rrf"].append(record(query, sources, fused[:EVAL_TOP_K], dense_ms + sparse_ms + fusion_ms))

        final_start = time.perf_counter()
        ranked = pipeline.retrieve(text, top_k=EVAL_TOP_K)
        full_ms = (time.perf_counter() - final_start) * 1000
        production_record = record(query, sources, ranked, full_ms)
        production_record["stage_latencies_ms"] = dict(pipeline.last_timings_ms)
        filtered = task_evidence(text, ranked, "explain")
        production_record["context_sufficient"] = ContextBuilder(
            settings.max_context_chars, settings.min_relevance_score).build(filtered).retrieval_status == "sufficient"
        variants["hybrid_cross_encoder"].append(production_record)
        ranking_changed += [chunk.chunk_id for chunk in ranked] != [chunk.chunk_id for chunk in fused[:EVAL_TOP_K]]

    for variant_rows in variants.values():
        for row in variant_rows:
            row["expected_count"] = all_expected_counts[row["query_id"]]
            row.pop("_expected_seen", None)
    stages = {stage: [row["stage_latencies_ms"].get(stage, 0.0)
                      for row in variants["hybrid_cross_encoder"]]
              for stage in ("dense", "sparse", "fusion", "cross_encoder", "total")}
    report = {
        "dataset": dataset,
        "protocol": {"corpus_scope": "all currently indexed bundled sample_data PDFs",
                     "production_top_k": settings.top_k, "evaluation_top_k": EVAL_TOP_K,
                     "dense_candidate_k": settings.dense_candidate_k,
                     "sparse_candidate_k": settings.sparse_candidate_k,
                     "rrf_k": settings.rrf_k, "rerank_candidate_k": settings.rerank_candidate_k,
                     "cross_encoder": settings.cross_encoder_model,
                     "cross_encoder_score_offset": settings.cross_encoder_score_offset,
                     "device": str(pipeline.cross_encoder.device) if hasattr(pipeline.cross_encoder, "device") else settings.cross_encoder_device,
                     "sync": sync_result, "corpus_chunks": len(corpus)},
        "variants": {name: {"metrics": metrics(rows), "queries": rows} for name, rows in variants.items()},
        "hybrid_cross_encoder_latency_ms": {"stages": {stage: {"average": statistics.mean(values) if values else 0.0,
                                                                  "p95": percentile95(values)} for stage, values in stages.items()},
                                             "ranking_changed_vs_rrf_queries": ranking_changed,
                                             "query_count": len(dataset["queries"])},
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, data in report["variants"].items():
        print(name, json.dumps(data["metrics"], sort_keys=True))
    print("hybrid_cross_encoder_latency_ms", json.dumps(report["hybrid_cross_encoder_latency_ms"], sort_keys=True))
    print("output", OUTPUT)


if __name__ == "__main__":
    main()
