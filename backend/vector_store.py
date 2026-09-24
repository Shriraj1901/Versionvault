import chromadb
from document_loader import load_documents
from chunker import chunk_documents
from embedder import embed_chunks

# Creates a persistent local database folder called "chroma_db"
# so your embeddings survive between runs, not just in memory.
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(name="versiondocs")


def store_chunks(chunks):
    """
    Saves each chunk's text, embedding, and metadata (version, source_file)
    into the Chroma collection so it can be searched later.
    """
    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        embeddings=[chunk["embedding"].tolist() for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {"version": chunk["version"], "source_file": chunk["source_file"]}
            for chunk in chunks
        ],
    )


if __name__ == "__main__":
    docs, skipped = load_documents()
    chunks = chunk_documents(docs)
    embedded_chunks = embed_chunks(chunks)

    store_chunks(embedded_chunks)

    print(f"Stored {collection.count()} chunks in the vector database.")