import os
os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# Always points at the chroma_db folder in the project root,
# no matter which folder you launch Python from.
DB_PATH = str(Path(__file__).resolve().parent.parent / "chroma_db")

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="versiondocs")

print(f"[retriever] DB path: {DB_PATH}")
print(f"[retriever] Chunks in collection: {collection.count()}")

model = SentenceTransformer("all-MiniLM-L6-v2")


def search(query, version=None, top_k=3):
    """
    Embeds the user's query and searches the vector DB for the most
    similar chunks. If a version is given, results are filtered to
    ONLY that version.
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


def list_versions():
    """Returns every version tag present in the collection, e.g. ['v1', 'v2', 'v3']."""
    data = collection.get(include=["metadatas"])
    return sorted({m["version"] for m in data["metadatas"]})


def search_across_versions(query, per_version=2):
    """
    Searches each version separately and merges the results by distance,
    so one version can't crowd the others out of the top results.
    """
    matches = []
    for v in list_versions():
        matches.extend(search(query, version=v, top_k=per_version))
    matches.sort(key=lambda m: m["distance"])
    return matches


if __name__ == "__main__":
    query = "How do I authenticate?"

    print(f"\nQuery: \"{query}\"  (per-version search)\n")
    for m in search_across_versions(query):
        print(f"  [{m['version']}/{m['source_file']}] (dist={m['distance']:.3f})  {m['text'][:70]}...")