import os
os.environ["HF_HUB_OFFLINE"] = "1"

from dotenv import load_dotenv
from groq import Groq
from retriever import search

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def build_prompt(query, chunks):
    """
    Injects the retrieved chunks into a prompt, clearly labeled with
    their source and version, and instructs the model to answer ONLY
    from this context — this is what keeps answers grounded instead
    of the model making things up from general knowledge.
    """
    context_blocks = []
    for c in chunks:
        context_blocks.append(
            f"[Source: {c['source_file']}, version: {c['version']}]\n{c['text']}"
        )
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a documentation assistant. Answer the question using
ONLY the context below. Do not use any outside knowledge.

Context:
{context}

Question: {query}

Answer clearly and concisely. At the end, cite which source file(s)
you used."""

    return prompt


def ask(query, version=None):
    chunks = search(query, version=version, top_k=3)

    # Guardrail: if nothing relevant was retrieved, or the best match
    # is too weak (high distance = low similarity), refuse instead
    # of letting the LLM guess from general knowledge.
    if not chunks or chunks[0]["distance"] > 1.5:
        return "I don't have enough information in the documentation to answer that confidently."

    prompt = build_prompt(query, chunks)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    query = "How do I authenticate?"

    print("=== Answer for v1 ===")
    print(ask(query, version="v1"))

    print("\n=== Answer for v3 ===")
    print(ask(query, version="v3"))

    print("\n=== Answer for an unrelated question ===")
    print(ask("What is the capital of France?", version="v3"))