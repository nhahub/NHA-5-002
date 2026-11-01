from app.config import get_settings
from app.rag import ChromaRetriever, sync_pdf_directory

if __name__ == "__main__":
    settings = get_settings(); retriever = ChromaRetriever(settings)
    result = sync_pdf_directory(settings.sample_data_dir, retriever.collection, retriever.embedding,
                                chroma_path=settings.chroma_path, chunk_chars=settings.chunk_chars,
                                overlap_chars=settings.chunk_overlap_chars)
    print(f"Collection: {settings.chroma_collection}; discovered={result['discovered']} new={result['new']} updated={result['updated']} skipped={result['skipped']} chunks={result['chunks']} failures={result['failed']}; collection_chunks={retriever.collection.count()}")
    if result["failed"]:
        raise SystemExit(1)
