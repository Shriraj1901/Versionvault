import os
os.environ["HF_HUB_OFFLINE"] = "1"
from sentence_transformers import SentenceTransformer
from document_loader import load_documents
from chunker import chunk_documents

# This downloads a small, fast embedding model the first time you run it
# (~80MB, one-time download, then cached locally).
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks):
    """
    Converts each chunk's text into a vector embedding.
    Returns the same chunks, each with an added 'embedding' field.
    """
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    return chunks


if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    embedded_chunks = embed_chunks(chunks)

    print(f"\nEmbedded {len(embedded_chunks)} chunks.\n")
    first = embedded_chunks[0]
    print(f"Example chunk: [{first['chunk_id']}]")
    print(f"Text: \"{first['text'][:60]}...\"")
    print(f"Embedding vector length: {len(first['embedding'])}")
    print(f"First 5 numbers of the vector: {first['embedding'][:5]}")