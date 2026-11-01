from app.config import Settings
from app.domain import RetrievedChunk
from app.rag import (
    HybridCrossEncoderRetriever, SparseBM25Index, _cross_encoder_device, _load_cross_encoder,
    reciprocal_rank_fusion,
)
import pytest


def chunk(chunk_id, text, metadata=None):
    return RetrievedChunk(text, "notes.pdf", 1, chunk_id,
                          {"embedding_version": "v1", **(metadata or {})}, .8)


class Collection:
    def __init__(self, rows): self.rows = rows
    def get(self, include):
        return {"ids": [c.chunk_id for c in self.rows], "documents": [c.content for c in self.rows],
                "metadatas": [c.metadata for c in self.rows]}


class Embedding:
    version = "v1"


class Dense:
    embedding = Embedding()
    def __init__(self, rows): self.rows, self.collection, self.calls = rows, Collection(rows), []
    def retrieve(self, query, filters=None, top_k=5):
        self.calls.append((query, filters, top_k))
        return self.rows[:top_k]


class Scorer:
    def __init__(self, score_by_id=None): self.score_by_id, self.batches = score_by_id or {}, []
    def predict(self, pairs, batch_size, show_progress_bar):
        self.batches.append((pairs, batch_size, show_progress_bar))
        return [self.score_by_id.get(text.split()[0], 1.0) for _, text in pairs]


def service(rows, scorer=None, **overrides):
    defaults = {"_env_file": None, "dense_candidate_k": 4, "sparse_candidate_k": 4,
                "rrf_k": 60, "rerank_candidate_k": 4, "top_k": 3,
                "cross_encoder_batch_size": 2}
    defaults.update(overrides)
    loader = defaults.pop("cross_encoder_loader", None)
    return HybridCrossEncoderRetriever(Dense(rows), Settings(**defaults), cross_encoder=scorer,
                                       cross_encoder_loader=loader)


def test_bm25_exact_and_morphological_match_and_empty_or_unrelated_queries():
    index = SparseBM25Index([chunk("a", "Farmers grow crops beside the river."),
                             chunk("b", "The pyramid contains a royal burial chamber.")])
    assert index.retrieve("farmer grows crop", 2)[0].chunk_id == "a"
    assert index.retrieve("royal burial", 2)[0].chunk_id == "b"
    assert index.retrieve("quantum teleportation", 2) == []
    assert index.retrieve("", 2) == []


def test_rrf_keeps_dense_only_sparse_only_and_overlap_and_deduplicates():
    dense = [chunk("shared", "shared text"), chunk("dense", "dense only")]
    sparse = [chunk("shared", "shared text"), chunk("sparse", "sparse only")]
    result = reciprocal_rank_fusion(dense, sparse, rrf_k=60, limit=5)
    assert {item.chunk_id for item in result} == {"shared", "dense", "sparse"}
    assert result[0].chunk_id == "shared"
    assert len({item.chunk_id for item in result}) == len(result)
    assert result[0].metadata["hybrid_score"] == 1.0


def test_cross_encoder_reranks_bounded_pool_and_batches_with_final_top_k():
    rows = [chunk("low", "low relevance passage"), chunk("high", "high relevance passage"),
            chunk("other", "other topic text")]
    scorer = Scorer({"low": -2.0, "high": 3.0, "other": 0.0})
    pipeline = service(rows, scorer, dense_candidate_k=3, sparse_candidate_k=3,
                       rerank_candidate_k=2, top_k=1)
    result = pipeline.retrieve("query", top_k=1)
    assert len(result) == 1 and result[0].chunk_id == "high"
    assert len(scorer.batches[0][0]) == 2
    assert scorer.batches[0][1] == 2
    assert result[0].metadata["retrieval_method"] == "hybrid_cross_encoder"
    assert "cross_encoder_score" in result[0].metadata


def test_negative_cross_encoder_logit_is_calibrated_for_context_floor():
    pipeline = service([chunk("relevant", "relevant annual river inundation")], Scorer({"relevant": -4.15}))
    result = pipeline.retrieve("annual river cycle", top_k=1)
    assert result[0].relevance_score > .35
    assert result[0].metadata["cross_encoder_raw_score"] == -4.15
    assert result[0].metadata["cross_encoder_score"] == -4.15


def test_dense_retriever_is_called_with_configured_candidate_pool():
    dense = Dense([chunk("dense", "dense result")])
    pipeline = HybridCrossEncoderRetriever(dense, Settings(_env_file=None, dense_candidate_k=7,
            sparse_candidate_k=7, rerank_candidate_k=7, cross_encoder_batch_size=4), cross_encoder=Scorer())
    pipeline.retrieve("a useful query")
    assert dense.calls[0][2] == 7


def test_encoder_is_loaded_once_and_empty_pool_does_not_load_it():
    rows = [chunk("one", "one substantive result")]
    calls = []
    pipeline = service(rows, cross_encoder_loader=lambda name, device: calls.append((name, device)) or Scorer())
    assert pipeline._model() is pipeline._model()
    assert len(calls) == 1

    empty_calls = []
    empty = service([], cross_encoder_loader=lambda *args: empty_calls.append(args))
    assert empty.retrieve("query") == []
    assert empty_calls == []


def test_device_auto_selects_cpu_and_cuda_fallback(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _cross_encoder_device("auto") == "cpu"
    assert _cross_encoder_device("cuda") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _cross_encoder_device("auto") == "cuda"


def test_production_retrieval_configuration_defaults_and_override():
    defaults = Settings(_env_file=None)
    custom = Settings(_env_file=None, dense_candidate_k=12, sparse_candidate_k=14,
                      rrf_k=45, rerank_candidate_k=20, top_k=4,
                      cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert defaults.retrieval_strategy == "hybrid_cross_encoder"
    assert (defaults.dense_candidate_k, defaults.sparse_candidate_k, defaults.rrf_k,
            defaults.rerank_candidate_k, defaults.top_k) == (30, 30, 60, 30, 5)
    assert defaults.cross_encoder_score_offset == 5.0
    assert (custom.dense_candidate_k, custom.sparse_candidate_k, custom.rrf_k,
            custom.rerank_candidate_k, custom.top_k) == (12, 14, 45, 20, 4)


def test_real_cross_encoder_scores_relevant_pair_above_unrelated_pair():
    pytest.importorskip("sentence_transformers")
    model = _load_cross_encoder("cross-encoder/ms-marco-MiniLM-L-6-v2", "cpu")
    scores = model.predict([
        ("What does the Nile's annual inundation support?",
         "The annual inundation of the Nile supports Egypt's agricultural population."),
        ("What does the Nile's annual inundation support?",
         "A linear transformation maps vectors from one vector space to another."),
    ], batch_size=2, show_progress_bar=False)
    assert scores[0] > scores[1]


def test_sparse_snapshot_refresh_drops_deleted_chunks():
    rows = [chunk("one", "specific spectral decomposition")]
    dense = Dense(rows)
    pipeline = HybridCrossEncoderRetriever(dense, Settings(_env_file=None), cross_encoder=Scorer())
    assert pipeline._sparse_index.retrieve("spectral decomposition", 3)
    dense.collection.rows = []
    assert pipeline.refresh_sparse_index() == 0
    assert pipeline._sparse_index.retrieve("spectral decomposition", 3) == []
