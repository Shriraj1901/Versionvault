from document_loader import load_documents
from chunker import chunk_documents


def validate_corpus():
    """
    Runs the full ingestion pipeline (load -> chunk) and reports a
    summary so nothing silently disappears from the corpus. Use this
    any time you add new documents, before trusting the system with
    real questions.
    """
    docs, skipped = load_documents()
    chunks = chunk_documents(docs)

    print("=" * 50)
    print("CORPUS VALIDATION REPORT")
    print("=" * 50)

    print(f"\nDocuments loaded: {len(docs)}")

    by_version = {}
    for doc in docs:
        by_version.setdefault(doc["version"], []).append(doc["source_file"])

    for version, files in sorted(by_version.items()):
        print(f"  {version}: {len(files)} file(s) -> {', '.join(files)}")

    print(f"\nChunks created: {len(chunks)}")
    avg_tokens = sum(c["token_count"] for c in chunks) / len(chunks) if chunks else 0
    print(f"  Average chunk size: {avg_tokens:.1f} tokens")

    empty_docs = [d for d in docs if len(d["text"]) < 20]
    if empty_docs:
        print(f"\n  WARNING: {len(empty_docs)} document(s) have suspiciously little text:")
        for d in empty_docs:
            print(f"    - [{d['version']}] {d['source_file']}")

    if skipped:
        print(f"\nSkipped {len(skipped)} file(s):")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")
    else:
        print("\nNo files skipped.")

    print("\n" + "=" * 50)
    if skipped or empty_docs:
        print("STATUS: Review warnings above before proceeding.")
    else:
        print("STATUS: Corpus looks healthy. Safe to re-index.")
    print("=" * 50)


if __name__ == "__main__":
    validate_corpus()