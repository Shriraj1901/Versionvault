from pathlib import Path
import chromadb
from document_loader import load_documents
from chunker import chunk_documents
from embedder import embed_chunks

# Always points at the chroma_db folder in the project root,
# no matter which folder you launch Python from.
DB_PATH = str(Path(__file__).resolve().parent.parent / "chroma_db")

client = chromadb.PersistentClient(path=DB_PATH)


def store_chunks(chunks):
    """
    Rebuilds the collection from scratch, then saves each chunk's text,
    embedding, and metadata (version, source_file). Rebuilding means you
    can re-run this any time without duplicate-ID errors.
    """
    try:
        client.delete_collection(name="versiondocs")
    except Exception:
        pass  # collection didn't exist yet, that's fine

    collection = client.get_or_create_collection(name="versiondocs")

    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        embeddings=[chunk["embedding"].tolist() for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {"version": chunk["version"], "source_file": chunk["source_file"]}
            for chunk in chunks
        ],
    )
    return collection


if __name__ == "__main__":
    docs, skipped = load_documents()
    chunks = chunk_documents(docs)
    embedded_chunks = embed_chunks(chunks)

    collection = store_chunks(embedded_chunks)

    print(f"DB path: {DB_PATH}")
    print(f"Stored {collection.count()} chunks in the vector database.")