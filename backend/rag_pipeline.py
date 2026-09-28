import os
os.environ["HF_HUB_OFFLINE"] = "1"

from dotenv import load_dotenv
from groq import Groq
from retriever import search

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

GOOD_MATCH_THRESHOLD = 1.5   # distance below this = confident match
WEAK_MATCH_THRESHOLD = 1.8   # distance above this = too weak, don't even retry


def build_prompt(query, chunks, note=None):
    context_blocks = []
    for c in chunks:
        context_blocks.append(
            f"[Source: {c['source_file']}, version: {c['version']}]\n{c['text']}"
        )
    context = "\n\n".join(context_blocks)

    extra = f"\n\nNote to reader: {note}" if note else ""

    prompt = f"""You are a documentation assistant. Answer the question using
ONLY the context below. Do not use any outside knowledge.

Context:
{context}

Question: {query}

Answer clearly and concisely. At the end, cite which source file(s)
you used.{extra}"""

    return prompt


def _call_llm(prompt):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def ask(query, version=None):
    """
    Agentic retrieval: tries a scoped search first. If the results are
    weak, it autonomously decides to broaden the search (drop the
    version filter) before falling back to a refusal. This mimics an
    agent reasoning about whether its first attempt was good enough.
    """
    attempt_log = []

    # --- Attempt 1: scoped to the requested version (if given) ---
    chunks = search(query, version=version, top_k=3)
    best_distance = chunks[0]["distance"] if chunks else float("inf")
    attempt_log.append(f"Attempt 1 (version={version or 'any'}): best_distance={best_distance:.3f}")

    if chunks and best_distance <= GOOD_MATCH_THRESHOLD:
        prompt = build_prompt(query, chunks)
        answer = _call_llm(prompt)
        return {"answer": answer, "log": attempt_log, "broadened": False}

    # --- Attempt 2: agent decides to broaden — drop version filter ---
    if version:
        chunks_broad = search(query, version=None, top_k=3)
        best_distance_broad = chunks_broad[0]["distance"] if chunks_broad else float("inf")
        attempt_log.append(f"Attempt 2 (broadened, no version filter): best_distance={best_distance_broad:.3f}")

        if chunks_broad and best_distance_broad <= WEAK_MATCH_THRESHOLD:
            note = (
                f"No strong match was found specifically for version '{version}', "
                f"so results from other versions are shown instead. Mention this "
                f"clearly in your answer and specify which version each fact applies to."
            )
            prompt = build_prompt(query, chunks_broad, note=note)
            answer = _call_llm(prompt)
            return {"answer": answer, "log": attempt_log, "broadened": True}

    # --- Final: nothing good enough found ---
    attempt_log.append("No sufficiently relevant content found. Refusing.")
    return {
        "answer": "I don't have enough information in the documentation to answer that confidently.",
        "log": attempt_log,
        "broadened": False,
    }


if __name__ == "__main__":
    print("=== v1 auth (should match directly) ===")
    result = ask("How do I authenticate?", version="v1")
    print(result["answer"])
    print("\nReasoning log:")
    for line in result["log"]:
        print(" ", line)

    print("\n=== A version that doesn't exist (should broaden) ===")
    result = ask("How do I authenticate?", version="v99")
    print(result["answer"])
    print("\nReasoning log:")
    for line in result["log"]:
        print(" ", line)

    print("\n=== Totally unrelated question (should refuse) ===")
    result = ask("What is the capital of France?", version="v3")
    print(result["answer"])
    print("\nReasoning log:")
    for line in result["log"]:
        print(" ", line)