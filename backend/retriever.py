import os
os.environ["HF_HUB_OFFLINE"] = "1"

import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="versiondocs")

model = SentenceTransformer("all-MiniLM-L6-v2")


def search(query, version=None, top_k=3):
    """
    Embeds the user's query and searches the vector DB for the most
    similar chunks. If a version is given, results are filtered to
    ONLY that version — this is what makes answers version-specific
    instead of mixing up v1/v2/v3 content.
    """
    query_embedding = model.encode([query])[0].tolist()

    where_filter = {"version": version} if version else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
    )

    matches = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        matches.append({
            "text": text,
            "version": metadata["version"],
            "source_file": metadata["source_file"],
            "distance": distance,
        })

    return matches


if __name__ == "__main__":
    query = "How do I authenticate?"

    print(f"Query: \"{query}\"  (no version filter)\n")
    for m in search(query):
        print(f"  [{m['version']}/{m['source_file']}] (dist={m['distance']:.3f})  {m['text'][:70]}...")

    print(f"\nQuery: \"{query}\"  (filtered to v3 only)\n")
    for m in search(query, version="v3"):
        print(f"  [{m['version']}/{m['source_file']}] (dist={m['distance']:.3f})  {m['text'][:70]}...")