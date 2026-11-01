import json
from pathlib import Path

from app.domain import RetrievedChunk
from scripts.benchmark_retrieval import BM25, metric_summary, rrf


def chunk(chunk_id, text, page):
    return RetrievedChunk(text, "Ancient_Egypt_From.pdf", page, chunk_id, {}, 0.8)


def test_bm25_ranks_exact_topic_match_deterministically():
    corpus = [chunk("a", "The Nile supported agriculture in Egypt.", 1),
              chunk("b", "Pyramids were built as royal tombs.", 2),
              chunk("c", "Agriculture along the Nile produced cereal crops.", 3)]
    first = BM25(corpus).retrieve("Nile agriculture", 3)
    second = BM25(corpus).retrieve("Nile agriculture", 3)
    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
    assert first[0].chunk_id in {"a", "c"}
    assert {item.chunk_id for item in first[:2]} == {"a", "c"}


def test_rrf_combines_rankings_and_uses_chunk_id_for_ties():
    semantic = [chunk("shared", "same", 1), chunk("semantic", "only semantic", 2)]
    lexical = [chunk("shared", "same", 1), chunk("lexical", "only lexical", 3)]
    fused = rrf(semantic, lexical, 3)
    assert [item.chunk_id for item in fused] == ["shared", "lexical", "semantic"]
    assert fused[0].relevance_score == 1.0


def test_metrics_use_page_labeled_relevant_chunks_and_first_relevant_rank():
    rows = [
        {"ground_truth_chunk_ids": ["a", "b"], "ranked_chunk_ids": ["x", "b", "a"],
         "latency_ms": 10, "candidate_count": 3},
        {"ground_truth_chunk_ids": ["c"], "ranked_chunk_ids": ["x", "y"],
         "latency_ms": 20, "candidate_count": 2},
    ]
    metrics = metric_summary(rows)
    assert metrics["recall@1"] == 0
    assert metrics["recall@3"] == 0.5
    assert metrics["hit_rate@1"] == 0
    assert metrics["hit_rate@3"] == 0.5
    assert metrics["mrr"] == 0.25
    assert metrics["avg_latency_ms"] == 15
    assert metrics["avg_candidate_count"] == 2.5
    assert metrics["zero_hit_queries"] == 1


def test_benchmark_labels_point_to_real_indexed_ancient_egypt_pages():
    root = Path(__file__).resolve().parents[1]
    dataset = json.loads((root / "data" / "retrieval_benchmark_queries.json").read_text(encoding="utf-8"))
    from app.config import get_settings
    from app.rag import ChromaRetriever

    retriever = ChromaRetriever(get_settings())
    pages = {metadata.get("page_number") for metadata in
             retriever.collection.get(include=["metadatas"])["metadatas"]
             if (metadata or {}).get("document_name") == "Ancient_Egypt_From.pdf"}
    assert len(dataset["queries"]) == 25
    assert all(query["relevant_pages"] and set(query["relevant_pages"]) <= pages
               for query in dataset["queries"])


def test_hybrid_benchmark_labels_cover_multiple_indexed_documents():
    root = Path(__file__).resolve().parents[1]
    dataset = json.loads((root / "data" / "hybrid_cross_encoder_queries.json").read_text(encoding="utf-8"))
    from app.config import get_settings
    from app.rag import ChromaRetriever

    collection = ChromaRetriever(get_settings()).collection
    rows = collection.get(include=["metadatas"])
    available = {(metadata.get("source_filename"), metadata.get("page_number"))
                 for metadata in rows["metadatas"] if metadata}
    positives = [query for query in dataset["queries"] if query["expected_sources"]]
    filenames = {source["source_filename"] for query in positives for source in query["expected_sources"]}
    assert len(dataset["queries"]) >= 10 and len(filenames) >= 2
    for query in positives:
        for source in query["expected_sources"]:
            assert all((source["source_filename"], page) in available for page in source["pages"])


def test_hybrid_cross_encoder_benchmark_confirms_positive_and_negative_context_gates():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "data" / "hybrid_cross_encoder_results.json").read_text(encoding="utf-8"))
    rows = report["variants"]["hybrid_cross_encoder"]["queries"]
    positive = [row for row in rows if row["expected_count"]]
    negative = [row for row in rows if not row["expected_count"]]
    assert len(positive) == 9 and len(negative) == 2
    assert all(row["context_sufficient"] for row in positive)
    assert all(not row["context_sufficient"] for row in negative)
