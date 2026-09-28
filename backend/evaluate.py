import os
import sys
import time
os.environ["HF_HUB_OFFLINE"] = "1"

from retriever import search
from rag_pipeline import ask, REFUSE_THRESHOLD, REFUSAL_MESSAGE

# (question, version filter, acceptable sources as "version/file")
IN_SCOPE = [
    ("How do I authenticate?", "v1", {"v1/auth.md"}),
    ("How do I authenticate?", "v2", {"v2/auth.md"}),
    ("How do I authenticate?", "v3", {"v3/auth.md"}),
    ("Do API keys expire in v1?", "v1", {"v1/auth.md"}),
    ("How long do tokens last in v2?", "v2", {"v2/auth.md"}),
    ("How do I refresh an access token?", "v3", {"v3/auth.md"}),
    ("What changed in this release?", "v3", {"v3/changelog.md"}),
    ("Which version uses OAuth 2.0?", None, {"v3/auth.md", "v3/changelog.md"}),
    ("Which version uses query parameter API keys?", None, {"v1/auth.md"}),
    ("Are Bearer tokens deprecated?", None, {"v3/auth.md", "v3/changelog.md"}),
]

# Questions the docs cannot answer. The system should refuse these.
OUT_OF_SCOPE = [
    ("What is the capital of France?", "v3"),
    ("How do I bake a cake?", None),
    ("Who won the football world cup?", None),
    ("What is the rate limit in v3?", "v3"),
    ("How do I set up webhooks?", None),
    ("How do I reset my password?", None),
    ("What programming languages does the SDK support?", None),
]


def retrieval_eval():
    print("=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)
    hit1 = hit3 = 0
    in_dists = []

    for question, version, expected in IN_SCOPE:
        results = search(question, version=version, top_k=3)
        sources = [f"{r['version']}/{r['source_file']}" for r in results]
        dist = results[0]["distance"] if results else float("inf")
        in_dists.append(dist)

        in_top1 = bool(sources) and sources[0] in expected
        in_top3 = any(s in expected for s in sources)
        hit1 += in_top1
        hit3 += in_top3

        mark = "PASS" if in_top1 else ("top-3 only" if in_top3 else "FAIL")
        print(f"[{mark}] ({version or 'any'}) {question}  (dist {dist:.3f})")
        if not in_top1:
            print(f"        expected {sorted(expected)}, got {sources}")

    n = len(IN_SCOPE)
    print(f"\nCorrect source at rank 1: {hit1}/{n} ({hit1 / n:.0%})")
    print(f"Correct source in top 3:  {hit3}/{n} ({hit3 / n:.0%})")

    print("\n" + "=" * 60)
    print(f"REFUSAL EVALUATION, distance layer (refuse above {REFUSE_THRESHOLD})")
    print("=" * 60)
    refused = 0
    out_dists = []
    for question, version in OUT_OF_SCOPE:
        results = search(question, version=None, top_k=1)
        best = results[0]["distance"] if results else float("inf")
        out_dists.append(best)
        would_refuse = best > REFUSE_THRESHOLD
        refused += would_refuse
        mark = "PASS" if would_refuse else "FAIL"
        print(f"[{mark}] {question}  (dist {best:.3f})")

    m = len(OUT_OF_SCOPE)
    print(f"\nCorrectly refused: {refused}/{m} ({refused / m:.0%})")

    false_refusals = sum(d > REFUSE_THRESHOLD for d in in_dists)
    print(f"In-scope questions the distance layer would wrongly refuse: {false_refusals}/{n}")

    hi_in, lo_out = max(in_dists), min(out_dists)
    print(f"\nHighest in-scope distance:  {hi_in:.3f}")
    print(f"Lowest out-of-scope distance: {lo_out:.3f}")
    print(f"Gap: {lo_out - hi_in:+.3f}  (positive means a threshold can separate them)")


def llm_eval():
    print("\n" + "=" * 60)
    print("END-TO-END EVALUATION (uses Groq, both guardrails active)")
    print("=" * 60)

    false_refusals = 0
    for question, version, _ in IN_SCOPE:
        result = ask(question, version=version)
        refused = result["answer"] == REFUSAL_MESSAGE
        false_refusals += refused
        print(f"[{'FALSE REFUSAL' if refused else 'answered'}] ({version or 'any'}) {question}")
        time.sleep(1)

    correct = 0
    for question, version in OUT_OF_SCOPE:
        result = ask(question, version=version)
        refused = result["answer"] == REFUSAL_MESSAGE
        correct += refused
        print(f"[{'refused' if refused else 'ANSWERED (hallucination risk)'}] {question}")
        time.sleep(1)

    print(f"\nIn-scope wrongly refused: {false_refusals}/{len(IN_SCOPE)}")
    print(f"Out-of-scope correctly refused: {correct}/{len(OUT_OF_SCOPE)}")


if __name__ == "__main__":
    retrieval_eval()
    if "--llm" in sys.argv:
        llm_eval()