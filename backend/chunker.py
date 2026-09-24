import tiktoken
from document_loader import load_documents

# Using OpenAI's tokenizer as a good general-purpose approximation —
# it's not perfect for every LLM, but it's consistent and free.
encoding = tiktoken.get_encoding("cl100k_base")

CHUNK_SIZE_TOKENS = 150   # target size per chunk
CHUNK_OVERLAP_TOKENS = 30  # shared context between neighboring chunks


def chunk_documents(documents, chunk_size=CHUNK_SIZE_TOKENS, overlap=CHUNK_OVERLAP_TOKENS):
    """
    Splits each document into overlapping, token-sized chunks instead
    of splitting on blank lines. This keeps chunks a consistent size
    and preserves context across chunk boundaries via overlap, so we
    don't lose meaning when a sentence gets cut mid-thought.
    """
    chunks = []

    for doc in documents:
        tokens = encoding.encode(doc["text"])
        total_tokens = len(tokens)

        if total_tokens == 0:
            continue

        start = 0
        chunk_index = 0

        while start < total_tokens:
            end = min(start + chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = encoding.decode(chunk_tokens).strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": f"{doc['version']}_{doc['source_file']}_{chunk_index}",
                    "text": chunk_text,
                    "source_file": doc["source_file"],
                    "version": doc["version"],
                    "token_count": len(chunk_tokens),
                })
                chunk_index += 1

            if end == total_tokens:
                break

            start = end - overlap  # step forward, but re-include the overlap

    return chunks


if __name__ == "__main__":
    docs, skipped = load_documents()
    chunks = chunk_documents(docs)

    print(f"Created {len(chunks)} chunks from {len(docs)} documents:\n")
    for c in chunks:
        preview = c["text"][:70].replace("\n", " ")
        print(f"  [{c['chunk_id']}] ({c['token_count']} tokens)  ->  \"{preview}...\"")