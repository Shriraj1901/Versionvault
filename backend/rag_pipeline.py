import os
os.environ["HF_HUB_OFFLINE"] = "1"

from dotenv import load_dotenv
from groq import Groq
from retriever import search, search_across_versions

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Coarse filter only: catches clearly unrelated questions (measured 1.85+).
# The LLM's NOT_IN_DOCS check is the fine judge for near-domain questions.
# Re-measure with evaluate.py whenever the documents change.
REFUSE_THRESHOLD = 1.8

REFUSAL_MESSAGE = "I don't have enough information in the documentation to answer that confidently."
NOT_IN_DOCS = "NOT_IN_DOCS"


def build_prompt(query, chunks, note=None):
    context_blocks = []
    for c in chunks:
        context_blocks.append(
            f"[Source: {c['source_file']}, version: {c['version']}]\n{c['text']}"
        )
    context = "\n\n".join(context_blocks)

    extra = f"\n\nNote: {note}" if note else ""

    return f"""You are a documentation assistant. Answer the question using ONLY the context below.

Rules:
- Use only facts stated in the context. Do not add outside knowledge, and do not add labels or claims that are not in the context (for example, do not call a version "current" or "legacy" unless the context says so).
- If the context does not contain the answer, reply with exactly: {NOT_IN_DOCS}
- Otherwise, answer clearly and concisely, and cite the source file and version for each fact.

Context:
{context}

Question: {query}{extra}"""


def _call_llm(prompt):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def _finalize(answer_text, log, broadened):
    """Second guardrail: if the LLM says the context doesn't answer it, refuse."""
    if NOT_IN_DOCS in answer_text[:40]:
        log.append("LLM judged the retrieved context insufficient. Refusing.")
        return {"answer": REFUSAL_MESSAGE, "log": log, "broadened": broadened}
    return {"answer": answer_text, "log": log, "broadened": broadened}


def ask(query, version=None):
    """
    Agentic retrieval with two guardrails:
    1. Distance check: clearly unrelated questions are refused before calling the LLM.
    2. LLM check: the model answers NOT_IN_DOCS if the context lacks the answer.
    If the scoped search is weak, the agent broadens (searches every version)
    once before giving up. With no version selected, each version is searched
    separately so one version can't crowd the others out of the results.
    """
    log = []

    # Attempt 1: scoped to the requested version, or across all versions
    if version:
        chunks = search(query, version=version, top_k=3)
    else:
        chunks = search_across_versions(query, per_version=2)

    best = chunks[0]["distance"] if chunks else float("inf")
    log.append(f"Attempt 1 (version={version or 'any'}): best_distance={best:.3f}")

    if chunks and best <= REFUSE_THRESHOLD:
        answer = _call_llm(build_prompt(query, chunks))
        return _finalize(answer, log, broadened=False)

    # Attempt 2: broaden by searching every version
    if version:
        chunks = search_across_versions(query, per_version=2)
        best = chunks[0]["distance"] if chunks else float("inf")
        log.append(f"Attempt 2 (broadened, all versions): best_distance={best:.3f}")

        if chunks and best <= REFUSE_THRESHOLD:
            note = (
                f"No strong match was found specifically for version '{version}', "
                f"so results from other versions are shown. Say this clearly and "
                f"state which version each fact applies to."
            )
            answer = _call_llm(build_prompt(query, chunks, note=note))
            return _finalize(answer, log, broadened=True)

    log.append("No sufficiently relevant content found. Refusing.")
    return {"answer": REFUSAL_MESSAGE, "log": log, "broadened": False}


if __name__ == "__main__":
    for q, v in [
        ("How do I authenticate?", "v1"),
        ("How do I authenticate?", "v99"),
        ("What is the rate limit in v3?", "v3"),
    ]:
        result = ask(q, version=v)
        print(f"=== {q} (version={v}) ===")
        print(result["answer"])
        for line in result["log"]:
            print("  ", line)
        print()