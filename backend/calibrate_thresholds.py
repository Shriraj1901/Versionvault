from retriever import search

test_cases = [
    ("How do I authenticate?", "v1"),
    ("How do I authenticate?", "v2"),
    ("How do I authenticate?", "v3"),
    ("What changed in the changelog?", "v3"),
]

print("Measuring real distances for KNOWN-GOOD matches:\n")
for query, version in test_cases:
    results = search(query, version=version, top_k=1)
    if results:
        print(f"  [{version}] \"{query}\" -> distance={results[0]['distance']:.3f}")
    else:
        print(f"  [{version}] \"{query}\" -> NO RESULTS")