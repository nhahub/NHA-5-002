from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from app.config import Settings
from app.providers import HashEmbeddingProvider
from app.rag import ChromaRetriever, HybridCrossEncoderRetriever, ingest_pdf, sync_pdf_directory

def make_text_pdf(path, pages):
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        font = writer._add_object(DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")}))
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as handle: writer.write(handle)

def test_pdf_page_metadata_ids_chroma_retrieval_and_idempotency(tmp_path):
    # Covers the ingestion contract end to end: original page numbers, stable IDs,
    # repeat-ingestion behavior, and retrieval with persisted metadata.
    pdf = tmp_path / "linear_algebra.pdf"
    make_text_pdf(pdf, ["A vector space is a set closed under vector addition and scalar multiplication.", "An eigenvalue describes scaling under a linear transformation."])
    settings = Settings(chroma_path=str(tmp_path / "chroma"), chroma_collection="test_materials", chunk_chars=240, chunk_overlap_chars=20)
    retriever = ChromaRetriever(settings, HashEmbeddingProvider())
    first = ingest_pdf(pdf, retriever.collection, retriever.embedding, chunk_chars=240, overlap_chars=20)
    after_first = retriever.collection.count()
    ids_after_first = set(retriever.collection.get(include=[])['ids'])
    second = ingest_pdf(pdf, retriever.collection, retriever.embedding, chunk_chars=240, overlap_chars=20)
    assert first["pages"] == 2 and first["pages_with_text"] == 2
    assert first["chunks"] == second["chunks"] and retriever.collection.count() == after_first
    assert set(retriever.collection.get(include=[])['ids']) == ids_after_first
    stored = retriever.collection.get(include=["documents", "metadatas"])
    assert set(row["page_number"] for row in stored["metadatas"]) == {1, 2}
    assert len(set(stored["ids"])) == len(stored["ids"])
    found = retriever.retrieve("vector space scalar multiplication", top_k=4)
    assert found and found[0].document_name == "linear_algebra.pdf"
    assert found[0].page_number == 1 and found[0].relevance_score > 0


def test_dense_retriever_handles_empty_collection_and_empty_query(tmp_path):
    settings = Settings(chroma_path=str(tmp_path / "chroma"), chroma_collection="empty_dense_test")
    retriever = ChromaRetriever(settings, HashEmbeddingProvider())
    assert retriever.retrieve("linear transformation", top_k=5) == []
    assert retriever.retrieve("", top_k=5) == []

def test_api_default_retriever_uses_hybrid_cross_encoder_over_chroma():
    from app.main import retriever
    assert isinstance(retriever, HybridCrossEncoderRetriever)
    assert isinstance(retriever.dense, ChromaRetriever)
    assert retriever.method == "hybrid_cross_encoder"
    assert retriever.embedding.provider_name == "sentence_transformers"
    assert retriever.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert retriever.collection.metadata["embedding_model"] == retriever.embedding.model_name

def test_real_semantic_retrieval_and_unrelated_question_gate():
    from app.config import get_settings
    from app.main import retriever
    from app.rag import ContextBuilder
    settings = get_settings()
    relevant = retriever.retrieve("What is linear independence?", top_k=5)
    assert relevant
    from app.evidence import task_evidence
    assert not task_evidence("What is the definition of linear independence?", relevant, "explain")
    unrelated = retriever.retrieve("How do I bake chocolate cake?", top_k=5)
    assert ContextBuilder(min_relevance_score=settings.min_relevance_score).build(unrelated).retrieval_status == "insufficient"


def test_ancient_egypt_nile_question_reaches_grounded_context_without_llm():
    """Regression for PDF line wrapping that previously erased useful evidence."""
    from pathlib import Path
    from app.main import retriever, settings
    from app.generation import LearningGenerator
    from app.rag import sync_pdf_directory

    pdf = Path(settings.sample_data_dir) / "Ancient_Egypt_From.pdf"
    assert pdf.is_file(), "The Ancient Egypt regression fixture must be present"
    sync_pdf_directory(settings.sample_data_dir, retriever.collection, retriever.embedding,
                       chroma_path=settings.chroma_path, chunk_chars=settings.chunk_chars,
                       overlap_chars=settings.chunk_overlap_chars)

    question = "How did the Nile River contribute to the development of ancient Egyptian civilization?"
    generator = LearningGenerator(retriever, settings.max_context_chars, top_k=settings.top_k,
                                  min_relevance_score=settings.min_relevance_score)
    context = generator._context(question, "explain")
    grounding = generator.grounding(context, "learning_materials_only")

    assert context.retrieval_status != "insufficient"
    assert grounding.sources
    assert any(source.document_name == "Ancient_Egypt_From.pdf" for source in grounding.sources)
    assert any("nile" in chunk.content.casefold() and
               any(term in chunk.content.casefold() for term in ("inundation", "agricultural", "floodplain"))
               for chunk in context.chunks)

def test_pdf_directory_sync_skips_unchanged_adds_new_and_replaces_only_modified(tmp_path):
    # Guards synchronization semantics so a changed source replaces its own chunks
    # without deleting unrelated PDFs from the shared collection.
    root = tmp_path / "sample_data"; root.mkdir()
    first_pdf = root / "history.pdf"
    make_text_pdf(first_pdf, ["The French Revolution began in 1789 and transformed France."])
    settings = Settings(chroma_path=str(tmp_path / "chroma"), chroma_collection="learning_materials_minilm")
    retriever = ChromaRetriever(settings, HashEmbeddingProvider())

    first = sync_pdf_directory(root, retriever.collection, retriever.embedding, chroma_path=settings.chroma_path)
    assert first["new"] == 1 and first["chunks"] > 0
    old = retriever.collection.get(where={"source_filename": "history.pdf"}, include=["documents", "metadatas"])
    assert all(row["file_hash"] and row["source_path"] for row in old["metadatas"])
    old_ids = set(old["ids"])

    unchanged = sync_pdf_directory(root, retriever.collection, retriever.embedding, chroma_path=settings.chroma_path)
    assert unchanged["skipped"] == 1 and retriever.collection.count() == len(old_ids)

    second_pdf = root / "french_revolution.pdf"
    make_text_pdf(second_pdf, ["The National Assembly abolished feudal privileges in August 1789."])
    added = sync_pdf_directory(root, retriever.collection, retriever.embedding, chroma_path=settings.chroma_path)
    assert added["new"] == 1 and added["skipped"] == 1
    unrelated = set(retriever.collection.get(where={"source_filename": "french_revolution.pdf"}, include=[])["ids"])

    make_text_pdf(first_pdf, ["The revolution ended with the rise of Napoleon Bonaparte in 1799."])
    updated = sync_pdf_directory(root, retriever.collection, retriever.embedding, chroma_path=settings.chroma_path)
    current = retriever.collection.get(include=["documents", "metadatas"])
    assert updated["updated"] == 1 and updated["skipped"] == 1
    assert not old_ids.intersection(current["ids"])
    assert unrelated.issubset(set(current["ids"]))
    assert any("Napoleon Bonaparte" in text for text in current["documents"])
    assert retriever.collection.name == "learning_materials_minilm"

def test_pdf_sync_empty_directory_and_per_file_failure_continue(tmp_path, caplog):
    root = tmp_path / "sample_data"; root.mkdir()
    settings = Settings(chroma_path=str(tmp_path / "chroma"), chroma_collection="learning_materials_minilm")
    retriever = ChromaRetriever(settings, HashEmbeddingProvider())
    empty = sync_pdf_directory(root, retriever.collection, retriever.embedding, chroma_path=settings.chroma_path)
    assert empty["discovered"] == empty["new"] == empty["failed"] == 0
    (root / "broken.pdf").write_bytes(b"not a PDF")
    make_text_pdf(root / "valid.pdf", ["A valid source document contains useful educational content."])
    result = sync_pdf_directory(root, retriever.collection, retriever.embedding, chroma_path=settings.chroma_path)
    assert result["failed"] == 1 and result["new"] == 1
    assert "PDF ingestion failed" in caplog.text
    assert retriever.collection.count() > 0


def test_pdf_sync_removes_deleted_source_chunks_only_within_synced_directory(tmp_path):
    root = tmp_path / "sample_data"; root.mkdir()
    outside = tmp_path / "other"; outside.mkdir()
    pdf = root / "remove_me.pdf"
    other_pdf = outside / "keep_me.pdf"
    make_text_pdf(pdf, ["The Nile River supported agriculture along the fertile floodplain."])
    make_text_pdf(other_pdf, ["A separate source discusses vector spaces and scalar multiplication."])
    settings = Settings(chroma_path=str(tmp_path / "chroma"), chroma_collection="learning_materials_minilm")
    retriever = ChromaRetriever(settings, HashEmbeddingProvider())
    ingest_pdf(other_pdf, retriever.collection, retriever.embedding)
    sync_pdf_directory(root, retriever.collection, retriever.embedding, chroma_path=settings.chroma_path)
    assert retriever.collection.count() == 2

    pdf.unlink()
    result = sync_pdf_directory(root, retriever.collection, retriever.embedding, chroma_path=settings.chroma_path)

    assert result["removed"] == 1
    remaining = retriever.collection.get(include=["metadatas"])
    assert len(remaining["ids"]) == 1
    assert remaining["metadatas"][0]["source_filename"] == "keep_me.pdf"

def test_api_startup_calls_reusable_pdf_sync(monkeypatch, tmp_path):
    import app.main as main
    seen = {}
    def sync(directory, collection, embedding, **kwargs):
        seen.update(directory=directory, collection=collection.name, kwargs=kwargs)
        return {"discovered": 0, "new": 0, "updated": 0, "skipped": 0, "failed": 0, "chunks": 0}
    monkeypatch.setattr(main, "sync_pdf_directory", sync)
    monkeypatch.setattr(main.settings, "sample_data_dir", str(tmp_path / "sample_data"))
    monkeypatch.setattr(main.settings, "auto_ingest_pdfs", True)
    from fastapi.testclient import TestClient
    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 200
    assert seen["directory"] == str(tmp_path / "sample_data")
    assert seen["collection"] == main.settings.chroma_collection
